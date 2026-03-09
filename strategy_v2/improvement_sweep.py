"""
V2-低リスク(+ETF) 改良スイープ

ベース（bo_lb=189, trail=3.5, stop=8%, pos=3）に対して
各改良を1つずつ単独で導入し、差分を比較する。

改良案:
  A: BEラチェット   — 含み益≥1ATRでSL→entryに引上げ
  B: pos=5          — 最大保有数 3→5
  C: タイムアウト60d — 60日で強制決済
  D: タイムアウト90d — 90日で強制決済
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import requests

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data.cache import CacheDB
from data.price_fetcher import fetch_prices
from data.screener import fetch_iwv_holdings, screen_tier1

logging.basicConfig(level=logging.WARNING)

# ============================================================
# ベースパラメータ（V2-低リスク）
# ============================================================
BASE_BO_LB     = 189
BASE_TRAIL     = 3.5
BASE_STOP      = 0.08
BASE_POS       = 3

SIM_START  = "2021-01-05"
SIM_END    = "2026-02-27"
DATA_START = "2019-10-01"

# ============================================================
# 改良バリアント定義
# ============================================================
# label, use_be, max_pos, max_hold(日, Noneは無制限)
VARIANTS = [
    ("BASE",             False, 3, None),
    ("A: BEラチェット",   True,  3, None),
    ("B: pos=5",         False, 5, None),
    ("C: timeout60d",    False, 3, 60),
    ("D: timeout90d",    False, 3, 90),
]

# ============================================================
# 固定パラメータ
# ============================================================
INITIAL_CAPITAL  = 4_000.0
STOP_LOOKBACK    = 126
ATR_PERIOD       = 14
MIN_VOLUME_MULT  = 1.5
VOLUME_LOOKBACK  = 20
RISK_PER_TRADE   = 0.015
MAX_ALLOC_PCT    = 0.20
MARKET_FILTER_MA = 200
RS_LOOKBACK      = 20
SECTOR_MA        = 50

SECTOR_ETF_MAP: dict[str, str] = {
    "Technology":             "XLK",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":       "XLP",
    "Energy":                 "XLE",
    "Financials":             "XLF",
    "Health Care":            "XLV",
    "Industrials":            "XLI",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Utilities":              "XLU",
}

ETF_UNIVERSE = [
    "SPY", "QQQ", "XLC", "XLY", "XLP",
    "XLE", "XLF", "XLV", "XLI", "XLB",
    "XLK", "XLRE", "XLU",
]
ETF_TOP_N = 2
ETF_MA    = 50
SAFE_ETF  = "SGOV"

MOMENTUM_PERIODS = {
    "1M":  (21,  0.30),
    "3M":  (63,  0.30),
    "6M":  (126, 0.30),
    "12M": (252, 0.10),
}

IWV_URL = (
    "https://www.ishares.com/us/products/239714/"
    "ishares-russell-3000-etf/1467271812596.ajax"
    "?fileType=csv&fileName=IWV_holdings&dataType=fund"
)


def _build_sector_map() -> dict[str, str]:
    cache = _ROOT / "data" / "db" / "sector_map.json"
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    try:
        resp = requests.get(IWV_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        return {}
    lines = resp.text.split("\n")
    hi = next((i for i, l in enumerate(lines) if l.startswith("Ticker,")), None)
    if hi is None:
        return {}
    reader = csv.DictReader(StringIO("\n".join(lines[hi:])))
    sec_map: dict[str, str] = {}
    for row in reader:
        t = (row.get("Ticker") or "").strip('"').strip()
        s = (row.get("Sector") or "").strip('"').strip()
        if t and s and (etf := SECTOR_ETF_MAP.get(s)):
            sec_map[t] = etf
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(sec_map, f)
    return sec_map


def calc_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float | None:
    if df is None or len(df) < period + 1:
        return None
    h, l, c = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (pc - l).abs()], axis=1).max(axis=1)
    val = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().iloc[-1]
    return float(val) if not np.isnan(val) else None


@dataclass
class Pos:
    ticker:       str
    entry_date:   str
    entry_price:  float
    stop_price:   float
    highest_high: float
    allocated:    float
    holding_days: int   = 0
    atr:          float = 0.0
    be_locked:    bool  = False   # BEラチェット発動済みフラグ


@dataclass
class Trade:
    pl_pct:       float
    holding_days: int
    ticker:       str   = ""
    exit_reason:  str   = ""  # "trail" / "be" / "timeout"
    allocated:    float = 0.0


# ============================================================
# ETFオーバーレイ補助
# ============================================================
def _etf_price(etf_close, ticker, date):
    s = etf_close.get(ticker)
    if s is None: return None
    past = s[s.index <= date]
    return float(past.iloc[-1]) if len(past) > 0 else None

def _etf_ma(etf_close, ticker, date, period=ETF_MA):
    s = etf_close.get(ticker)
    if s is None: return None
    past = s[s.index <= date]
    return float(past.iloc[-period:].mean()) if len(past) >= period else None

def _spy_bull(etf_close, date):
    s = etf_close.get("SPY")
    if s is None: return True
    past = s[s.index <= date]
    if len(past) < MARKET_FILTER_MA: return True
    return float(past.iloc[-1]) > float(past.iloc[-MARKET_FILTER_MA:].mean())

def _calc_momentum_scores(etf_close, date):
    rets = {}
    for ticker in ETF_UNIVERSE:
        s = etf_close.get(ticker)
        if s is None: continue
        past = s[s.index <= date]
        if len(past) == 0: continue
        current = float(past.iloc[-1])
        tr = {}; ok = True
        for name, (days, _) in MOMENTUM_PERIODS.items():
            if len(past) < days + 1: ok = False; break
            base = float(past.iloc[-(days + 1)])
            if base <= 0: ok = False; break
            tr[name] = (current - base) / base
        if ok: rets[ticker] = tr
    if len(rets) < 2: return []
    scores = {t: 0.0 for t in rets}
    for name, (_, weight) in MOMENTUM_PERIODS.items():
        ranked = sorted(rets.keys(), key=lambda t: rets[t][name], reverse=True)
        for rank, ticker in enumerate(ranked, 1):
            scores[ticker] += rank * weight
    return sorted(scores.items(), key=lambda x: x[1])

def _select_targets(etf_close, date, scored):
    result = []
    for ticker, _ in scored:
        if len(result) >= ETF_TOP_N: break
        p = _etf_price(etf_close, ticker, date)
        ma = _etf_ma(etf_close, ticker, date)
        if p and ma and p >= ma: result.append(ticker)
    return result


# ============================================================
# シミュレーション本体
# ============================================================
def run_sim(
    price_data, spy_df, sector_data, sector_map,
    spy_ok_map, spy_ret_map, sec_ok_map, etf_close,
    # バリアントパラメータ
    use_be:   bool,
    max_pos:  int,
    max_hold: int | None,
) -> tuple[list[Trade], list[dict]]:

    warmup = max(260, BASE_BO_LB + RS_LOOKBACK + 10)
    all_dates    = spy_df.index.tolist()
    sim_start_dt = pd.Timestamp(SIM_START)
    sim_end_dt   = pd.Timestamp(SIM_END)

    sim_indices   = [i for i, d in enumerate(all_dates) if sim_start_dt <= d <= sim_end_dt]
    if not sim_indices: return [], []
    run_start_idx = max(0, sim_indices[0] - warmup)

    cash      = INITIAL_CAPITAL
    positions: dict[str, Pos]   = {}
    etf_pos:   dict[str, float] = {}
    trades:    list[Trade]      = []
    equity_curve: list[dict]    = []

    prev_week: tuple | None = None
    current_scores: list[tuple[str, float]] = []
    prev_bull: bool | None  = None

    for idx in range(run_start_idx, len(all_dates)):
        date = all_dates[idx]
        if date > sim_end_dt: break
        date_str = date.strftime("%Y-%m-%d")
        in_sim   = date >= sim_start_dt
        spy_ok   = spy_ok_map.get(date_str, True)
        bull     = _spy_bull(etf_close, date)
        reg_chg  = (prev_bull is not None and bull != prev_bull)

        # ── ETFリバランス（毎週月曜）──
        week_key = (date.year, date.isocalendar()[1])
        if week_key != prev_week:
            prev_week = week_key
            if bull:
                current_scores = _calc_momentum_scores(etf_close, date)
                targets = _select_targets(etf_close, date, current_scores)
                for tk in list(etf_pos.keys()):
                    if tk not in targets:
                        p = _etf_price(etf_close, tk, date)
                        if p: cash += etf_pos[tk] * p
                        del etf_pos[tk]
                if targets and cash > 1.0:
                    alloc_each = cash / len(targets)
                    for tk in targets:
                        p = _etf_price(etf_close, tk, date)
                        if p and p > 0:
                            etf_pos[tk] = etf_pos.get(tk, 0.0) + alloc_each / p
                            cash -= alloc_each
            else:
                for tk in list(etf_pos.keys()):
                    if tk != SAFE_ETF:
                        p = _etf_price(etf_close, tk, date)
                        if p: cash += etf_pos[tk] * p
                        del etf_pos[tk]
                if cash > 1.0:
                    p = _etf_price(etf_close, SAFE_ETF, date)
                    if p:
                        etf_pos[SAFE_ETF] = etf_pos.get(SAFE_ETF, 0.0) + cash / p
                        cash = 0.0

        # ── レジーム変化 ──
        if reg_chg:
            if not bull:
                for tk in list(etf_pos.keys()):
                    if tk != SAFE_ETF:
                        p = _etf_price(etf_close, tk, date)
                        if p: cash += etf_pos[tk] * p
                        del etf_pos[tk]
                if cash > 1.0:
                    p = _etf_price(etf_close, SAFE_ETF, date)
                    if p:
                        etf_pos[SAFE_ETF] = etf_pos.get(SAFE_ETF, 0.0) + cash / p
                        cash = 0.0
            else:
                for tk in [SAFE_ETF]:
                    if tk in etf_pos:
                        p = _etf_price(etf_close, tk, date)
                        if p: cash += etf_pos[tk] * p
                        del etf_pos[tk]
        elif bull:
            for tk in list(etf_pos.keys()):
                if tk == SAFE_ETF: continue
                p = _etf_price(etf_close, tk, date)
                ma = _etf_ma(etf_close, tk, date)
                if p and ma and p < ma:
                    cash += etf_pos[tk] * p
                    del etf_pos[tk]
        prev_bull = bull

        # ── エグジット判定 ──
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            df  = price_data.get(ticker)
            if df is None or date not in df.index: continue
            row        = df.loc[date]
            today_high = float(row["High"])
            today_low  = float(row["Low"])
            today_open = float(row["Open"])
            today_close= float(row["Close"])

            if today_high > pos.highest_high:
                pos.highest_high = today_high
            atr = calc_atr(df.loc[:date])
            if atr: pos.atr = atr
            pos.holding_days += 1

            # ── BEラチェット ──
            if use_be and pos.atr > 0 and not pos.be_locked:
                profit = today_close - pos.entry_price
                if profit >= pos.atr:
                    be_stop = pos.entry_price
                    if be_stop > pos.stop_price:
                        pos.stop_price  = be_stop
                        pos.be_locked   = True

            # ── トレーリングストップ更新 ──
            if pos.atr > 0:
                new_trail = pos.highest_high - pos.atr * BASE_TRAIL
                if new_trail > pos.stop_price:
                    pos.stop_price = new_trail

            # ── タイムアウト強制決済 ──
            if max_hold is not None and pos.holding_days >= max_hold:
                exit_price = today_close
                pl_pct     = (exit_price - pos.entry_price) / pos.entry_price
                cash      += pos.allocated + pos.allocated * pl_pct
                if in_sim:
                    trades.append(Trade(
                        pl_pct=round(pl_pct * 100, 2),
                        holding_days=pos.holding_days,
                        ticker=ticker, exit_reason="timeout",
                        allocated=pos.allocated,
                    ))
                del positions[ticker]
                continue

            # ── ストップ判定 ──
            if today_low <= pos.stop_price:
                exit_price = float(np.clip(pos.stop_price, today_low, today_high))
                if today_open < pos.stop_price:
                    exit_price = today_open
                pl_pct = (exit_price - pos.entry_price) / pos.entry_price
                cash  += pos.allocated + pos.allocated * pl_pct
                reason = "be" if (use_be and pos.be_locked and abs(exit_price - pos.entry_price) < 0.01 * pos.entry_price) else "trail"
                if in_sim:
                    trades.append(Trade(
                        pl_pct=round(pl_pct * 100, 2),
                        holding_days=pos.holding_days,
                        ticker=ticker, exit_reason=reason,
                        allocated=pos.allocated,
                    ))
                del positions[ticker]

        # ── エントリー候補スキャン ──
        if spy_ok and len(positions) < max_pos:
            held = set(positions)
            candidates = []
            for ticker, df in price_data.items():
                if ticker in held or ticker == "SPY": continue
                if date not in df.index: continue
                loc = df.index.get_loc(date)
                if loc < BASE_BO_LB: continue
                close  = df["Close"].astype(float)
                volume = df["Volume"].astype(float) if "Volume" in df.columns else None
                cur_close = float(close.iloc[loc])
                prev_high = float(close.iloc[loc - BASE_BO_LB: loc].max())
                if cur_close <= prev_high: continue
                vol_ratio = 1.0
                if volume is not None and loc >= VOLUME_LOOKBACK:
                    avg_vol = float(volume.iloc[loc - VOLUME_LOOKBACK: loc].mean())
                    cur_vol = float(volume.iloc[loc])
                    if avg_vol > 0:
                        vol_ratio = cur_vol / avg_vol
                        if vol_ratio < MIN_VOLUME_MULT: continue
                spy_r = spy_ret_map.get(date_str)
                if spy_r is None or loc < RS_LOOKBACK: continue
                stk_r = (cur_close / float(close.iloc[loc - RS_LOOKBACK])) - 1
                if stk_r <= spy_r: continue
                etf_filt = sector_map.get(ticker)
                if etf_filt and not sec_ok_map.get(date_str, {}).get(etf_filt, True): continue
                bo_str = (cur_close - prev_high) / prev_high
                candidates.append((bo_str * vol_ratio, ticker, df, loc))
            candidates.sort(reverse=True, key=lambda x: x[0])

            for _, ticker, df, loc in candidates:
                if len(positions) >= max_pos: break
                if ticker in positions: continue
                if loc + 1 >= len(df): continue
                close_a     = df["Close"].astype(float)
                lows_a      = df["Low"].astype(float)
                entry_price = float(df.iloc[loc + 1]["Open"])
                if entry_price <= 0: continue
                low_26w      = float(lows_a.iloc[max(0, loc - STOP_LOOKBACK): loc + 1].min())
                hard_stop_p  = entry_price * (1 - BASE_STOP)
                initial_stop = max(low_26w, hard_stop_p)
                if initial_stop >= entry_price: initial_stop = hard_stop_p
                sl_pct = (entry_price - initial_stop) / entry_price
                if sl_pct < 0.005: continue
                etf_val  = sum(etf_pos[tk] * (_etf_price(etf_close, tk, date) or 0) for tk in etf_pos)
                total_eq = cash + sum(p.allocated for p in positions.values()) + etf_val
                alloc    = min(total_eq * RISK_PER_TRADE / sl_pct, total_eq * MAX_ALLOC_PCT, cash + etf_val)
                if cash < alloc and etf_pos:
                    shortage  = alloc - cash
                    score_map = {tk: s for tk, s in current_scores}
                    ranked    = sorted(etf_pos.keys(),
                                       key=lambda tk: (0 if tk == SAFE_ETF else 1, score_map.get(tk, 999)),
                                       reverse=True)
                    for tk in ranked:
                        if shortage <= 0: break
                        p = _etf_price(etf_close, tk, date)
                        if not p: continue
                        sell = min(shortage, etf_pos[tk] * p)
                        etf_pos[tk] -= sell / p
                        if etf_pos[tk] < 1e-9: del etf_pos[tk]
                        cash += sell; shortage -= sell
                alloc = min(alloc, cash)
                if alloc < 50: continue
                atr_val   = calc_atr(df.iloc[: loc + 1]) or (float(close_a.iloc[loc]) * 0.02)
                next_date = df.index[loc + 1].strftime("%Y-%m-%d")
                positions[ticker] = Pos(
                    ticker=ticker, entry_date=next_date,
                    entry_price=entry_price, stop_price=initial_stop,
                    highest_high=entry_price, allocated=round(alloc, 2), atr=atr_val,
                )
                cash -= alloc

        # ── 日次スナップショット ──
        if in_sim:
            etf_val   = sum(etf_pos[tk] * (_etf_price(etf_close, tk, date) or 0) for tk in etf_pos)
            stock_val = sum(
                p.allocated * float(price_data[t].loc[date, "Close"]) / p.entry_price
                if (df := price_data.get(t)) is not None and date in df.index else p.allocated
                for t, p in positions.items()
            )
            equity_curve.append({"date": date_str, "equity": round(cash + stock_val + etf_val, 2)})

    return trades, equity_curve


# ============================================================
# 集計
# ============================================================
def summarize(trades: list[Trade], equity_curve: list[dict]) -> dict:
    if not trades or not equity_curve:
        return {k: 0.0 for k in ["n","wr","avg_pl","avg_win","avg_loss",
                                   "max_win","max_loss","pf","total_ret",
                                   "ann_ret","max_dd","sharpe","calmar","final"]}
    pl     = [t.pl_pct for t in trades]
    wins   = [p for p in pl if p > 0]
    losses = [p for p in pl if p <= 0]
    eq     = pd.DataFrame(equity_curve)
    final  = float(eq["equity"].iloc[-1])
    arr    = eq["equity"].values
    peak   = np.maximum.accumulate(arr)
    dd     = (arr - peak) / peak * 100
    max_dd = float(dd.min())
    dr     = pd.Series(arr).pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    total_ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    n_years   = len(eq) / 252
    ann_ret   = ((final / INITIAL_CAPITAL) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0
    calmar    = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    gp = sum(t.allocated * t.pl_pct / 100 for t in trades if t.pl_pct > 0)
    gl = abs(sum(t.allocated * t.pl_pct / 100 for t in trades if t.pl_pct <= 0))
    return {
        "n": len(trades), "wr": len(wins) / len(trades) * 100,
        "avg_pl": float(np.mean(pl)),
        "avg_win": float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
        "max_win": float(max(pl)), "max_loss": float(min(pl)),
        "pf": gp / gl if gl > 0 else 9.99,
        "total_ret": total_ret, "ann_ret": ann_ret,
        "max_dd": max_dd, "sharpe": sharpe, "calmar": calmar, "final": final,
    }


def year_rets(equity_curve: list[dict]) -> dict[int, float]:
    eq = pd.DataFrame(equity_curve)
    eq["date"] = pd.to_datetime(eq["date"])
    eq["year"] = eq["date"].dt.year
    result = {}
    prev = INITIAL_CAPITAL
    for yr in sorted(eq["year"].unique()):
        end = float(eq[eq["year"] == yr]["equity"].iloc[-1])
        result[yr] = (end - prev) / prev * 100
        prev = end
    return result


# ============================================================
# メイン
# ============================================================
def main() -> None:
    print("=" * 76)
    print("  V2-低リスク(+ETF) 改良スイープ")
    print(f"  ベース: bo_lb={BASE_BO_LB}, trail={BASE_TRAIL}, stop={BASE_STOP*100:.0f}%, pos={BASE_POS}")
    print(f"  期間: {SIM_START} 〜 {SIM_END}")
    print("=" * 76)

    db = CacheDB(); db.initialize()

    print("\n⏳ データ準備中...")
    iwv     = fetch_iwv_holdings(db)
    tickers = screen_tier1(iwv, db)
    sec_etfs    = list(SECTOR_ETF_MAP.values())
    all_symbols = sorted(set(tickers) | {"SPY"} | set(sec_etfs))
    price_all   = fetch_prices(all_symbols, db=db, period="6y")
    spy_df      = price_all.pop("SPY", None)
    sector_data = {etf: price_all.pop(etf) for etf in sec_etfs if etf in price_all}
    price_data  = price_all
    sector_map  = _build_sector_map()

    spy_c   = spy_df["Close"].astype(float)
    spy_ma  = spy_c.rolling(MARKET_FILTER_MA).mean()
    spy_ret = spy_c.pct_change(RS_LOOKBACK)
    spy_ok_map, spy_ret_map = {}, {}
    for d in spy_df.index:
        ds = d.strftime("%Y-%m-%d")
        spy_ok_map[ds]  = bool(spy_c[d] >= spy_ma[d]) if not np.isnan(spy_ma[d]) else True
        if not np.isnan(spy_ret[d]): spy_ret_map[ds] = float(spy_ret[d])

    sec_ok_map: dict[str, dict[str, bool]] = {}
    for etf, df in sector_data.items():
        ec = df["Close"].astype(float); em = ec.rolling(SECTOR_MA).mean()
        for d in df.index:
            ds = d.strftime("%Y-%m-%d")
            if ds not in sec_ok_map: sec_ok_map[ds] = {}
            sec_ok_map[ds][etf] = bool(ec[d] >= em[d]) if not np.isnan(em[d]) else True

    raw = yf.download(ETF_UNIVERSE + [SAFE_ETF], start=DATA_START, end=SIM_END,
                      auto_adjust=True, progress=False)
    etf_close: dict[str, pd.Series] = {}
    for tk in ETF_UNIVERSE + [SAFE_ETF]:
        try:
            etf_close[tk] = raw["Close"][tk].dropna()
        except Exception:
            pass
    print(f"  → 銘柄: {len(price_data)}, ETF: {len(etf_close)} 本")

    # ============================================================
    # 各バリアント実行
    # ============================================================
    print("\n⏳ スイープ実行中...\n")
    results = []
    yr_table: dict[str, dict[int, float]] = {}

    for label, use_be, max_pos, max_hold in VARIANTS:
        desc = f"use_be={use_be}, pos={max_pos}, hold={max_hold or '∞'}"
        print(f"  [{label}] {desc} ...", end="", flush=True)
        trades, eq = run_sim(
            price_data, spy_df, sector_data, sector_map,
            spy_ok_map, spy_ret_map, sec_ok_map, etf_close,
            use_be=use_be, max_pos=max_pos, max_hold=max_hold,
        )
        s = summarize(trades, eq)
        s["label"] = label
        results.append(s)
        yr_table[label] = year_rets(eq)
        print(f" 完了  total={s['total_ret']:+.1f}%  Sharpe={s['sharpe']:.3f}  DD={s['max_dd']:.1f}%")

    base = results[0]

    # ============================================================
    # 比較テーブル
    # ============================================================
    print("\n" + "=" * 76)
    print("  改良効果 比較テーブル")
    print("=" * 76)
    hdr = f"  {'バリアント':<20} {'件数':>5} {'勝率':>6} {'avg':>7} {'勝avg':>7} {'負avg':>7} {'PF':>5} {'総Ret':>8} {'MaxDD':>6} {'Sharpe':>7} {'Calmar':>7} {'最終資産':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for s in results:
        is_base = s["label"] == "BASE"
        vs_ret    = "" if is_base else f"({s['total_ret']-base['total_ret']:+.1f})"
        vs_sharpe = "" if is_base else f"({s['sharpe']-base['sharpe']:+.3f})"
        vs_dd     = "" if is_base else f"({s['max_dd']-base['max_dd']:+.1f})"
        marker = "  " if is_base else ("★ " if s["total_ret"] > base["total_ret"] else "  ")
        print(f"{marker}{s['label']:<20} {s['n']:>5d}  {s['wr']:>5.1f}%  "
              f"{s['avg_pl']:>+6.2f}%  {s['avg_win']:>+6.2f}%  {s['avg_loss']:>+6.2f}%  "
              f"{s['pf']:>4.2f}  {s['total_ret']:>+7.1f}%{vs_ret:<8}  "
              f"{s['max_dd']:>5.1f}%{vs_dd:<6}  {s['sharpe']:>6.3f}{vs_sharpe:<8}  "
              f"{s['calmar']:>6.2f}  ${s['final']:>8,.0f}")

    # ============================================================
    # 年別リターン比較
    # ============================================================
    years = sorted({yr for yrs in yr_table.values() for yr in yrs})
    print("\n" + "=" * 76)
    print("  年別リターン比較")
    print("=" * 76)
    labels_short = [s["label"] for s in results]
    print(f"  {'年':<6} " + "  ".join(f"{l[:14]:>14}" for l in labels_short))
    print("  " + "-" * (8 + 16 * len(labels_short)))
    for yr in years:
        row = f"  {yr:<6} "
        for s in results:
            v = yr_table[s["label"]].get(yr, 0.0)
            row += f"  {v:>+13.1f}%"
        print(row)

    # ============================================================
    # BEラチェット詳細（exit_reason内訳）
    # ============================================================
    be_label = "A: BEラチェット"
    be_variant = next((v for v in VARIANTS if v[0] == be_label), None)
    if be_variant:
        print("\n" + "=" * 76)
        print("  BEラチェット 退場理由内訳")
        print("=" * 76)
        _, use_be, max_pos, max_hold = be_variant
        trades_be, _ = run_sim(
            price_data, spy_df, sector_data, sector_map,
            spy_ok_map, spy_ret_map, sec_ok_map, etf_close,
            use_be=True, max_pos=max_pos, max_hold=max_hold,
        )
        # BE発動時のP/L分布
        be_exits  = [t for t in trades_be if t.exit_reason == "be"]
        trail_exits = [t for t in trades_be if t.exit_reason == "trail"]
        print(f"  トレールヒット: {len(trail_exits)} 件  avg={np.mean([t.pl_pct for t in trail_exits]):+.2f}%")
        print(f"  BEヒット:       {len(be_exits)} 件  avg={np.mean([t.pl_pct for t in be_exits]):+.2f}%")
        if be_exits:
            print(f"    BEヒット P/L分布: "
                  f"0%以上={sum(1 for t in be_exits if t.pl_pct >= 0)}件 / "
                  f"マイナス={sum(1 for t in be_exits if t.pl_pct < 0)}件")
            # BE発動したが損したケース（ギャップダウン等）
            be_loss = [t for t in be_exits if t.pl_pct < -1]
            if be_loss:
                print(f"    BE後損失 top5: {sorted(be_loss, key=lambda t: t.pl_pct)[:5]}")

    # CSV保存
    out = _ROOT / "output" / "backtest" / "improvement_sweep_results.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\n📁 結果CSV: {out}")
    print("=" * 76)


if __name__ == "__main__":
    main()
