"""
Strategy V2 + ETF オーバーレイ 比較

V2（ブレイクアウト）の待機資金にベースラインと同じETFモメンタム戦略を適用する。
TOP 3候補でETF込みリターンを算出し、ベースライン(+288.3%)と比較する。

ETFオーバーレイ仕様（ベースラインと同一）:
  ユニバース: SPY QQQ XLC XLY XLP XLE XLF XLV XLI XLB XLK XLRE XLU
  保有: 上位2本（50MA以上のみ）、毎週月曜リバランス
  SPY>200MA: ETFモメンタム / SPY<200MA: SGOV待避
  個別株エントリー時: 弱いETFから売却して資金確保
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, field
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

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

# ============================================================
# TOP 3 候補
# ============================================================
CANDIDATES = [
    {"label": "高リターン", "bo_lb": 126, "trail_mult": 3.5, "hard_stop": 0.08, "max_pos": 7},
    {"label": "バランス",   "bo_lb": 189, "trail_mult": 3.5, "hard_stop": 0.08, "max_pos": 5},
    {"label": "低リスク",   "bo_lb": 189, "trail_mult": 3.5, "hard_stop": 0.08, "max_pos": 3},
]

BASELINE = {
    "label": "ベースライン(ETF込)", "total_ret": 288.3, "sharpe": 2.394,
    "max_dd": -13.43, "calmar": 21.47, "final": 15532.0,
}

SIM_START  = "2021-01-05"
SIM_END    = "2026-02-27"
DATA_START = "2019-10-01"

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
ETF_TOP_N  = 2
ETF_MA     = 50
SAFE_ETF   = "SGOV"

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


# ============================================================
# セクターマップ
# ============================================================
def _build_sector_map() -> dict[str, str]:
    cache = _ROOT / "data" / "db" / "sector_map.json"
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    try:
        resp = requests.get(IWV_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"  WARNING: {e}")
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


# ============================================================
# ATR
# ============================================================
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


@dataclass
class Trade:
    pl_pct:       float
    holding_days: int
    ticker:       str   = ""
    entry_date:   str   = ""
    exit_date:    str   = ""
    entry_price:  float = 0.0
    exit_price:   float = 0.0
    allocated:    float = 0.0


# ============================================================
# ETFオーバーレイ補助
# ============================================================
def _etf_price(etf_close: dict[str, pd.Series], ticker: str, date: pd.Timestamp) -> float | None:
    s = etf_close.get(ticker)
    if s is None:
        return None
    past = s[s.index <= date]
    return float(past.iloc[-1]) if len(past) > 0 else None


def _etf_ma(etf_close: dict[str, pd.Series], ticker: str, date: pd.Timestamp, period: int = ETF_MA) -> float | None:
    s = etf_close.get(ticker)
    if s is None:
        return None
    past = s[s.index <= date]
    return float(past.iloc[-period:].mean()) if len(past) >= period else None


def _spy_bull(etf_close: dict[str, pd.Series], date: pd.Timestamp) -> bool:
    s = etf_close.get("SPY")
    if s is None:
        return True
    past = s[s.index <= date]
    if len(past) < MARKET_FILTER_MA:
        return True
    return float(past.iloc[-1]) > float(past.iloc[-MARKET_FILTER_MA:].mean())


def _calc_momentum_scores(etf_close: dict[str, pd.Series], date: pd.Timestamp) -> list[tuple[str, float]]:
    rets: dict[str, dict[str, float]] = {}
    for ticker in ETF_UNIVERSE:
        s = etf_close.get(ticker)
        if s is None:
            continue
        past = s[s.index <= date]
        if len(past) == 0:
            continue
        current = float(past.iloc[-1])
        tr: dict[str, float] = {}
        ok = True
        for name, (days, _) in MOMENTUM_PERIODS.items():
            if len(past) < days + 1:
                ok = False
                break
            base = float(past.iloc[-(days + 1)])
            if base <= 0:
                ok = False
                break
            tr[name] = (current - base) / base
        if ok:
            rets[ticker] = tr
    if len(rets) < 2:
        return []
    scores: dict[str, float] = {t: 0.0 for t in rets}
    for name, (_, weight) in MOMENTUM_PERIODS.items():
        ranked = sorted(rets.keys(), key=lambda t: rets[t][name], reverse=True)
        for rank, ticker in enumerate(ranked, 1):
            scores[ticker] += rank * weight
    return sorted(scores.items(), key=lambda x: x[1])


def _select_targets(etf_close: dict[str, pd.Series], date: pd.Timestamp, scored: list[tuple[str, float]]) -> list[str]:
    result = []
    for ticker, _ in scored:
        if len(result) >= ETF_TOP_N:
            break
        p  = _etf_price(etf_close, ticker, date)
        ma = _etf_ma(etf_close, ticker, date)
        if p and ma and p >= ma:
            result.append(ticker)
    return result


# ============================================================
# メインシミュレーション（V2 + ETFオーバーレイ統合）
# ============================================================
def run_sim_with_etf(
    price_data:  dict[str, pd.DataFrame],
    spy_df:      pd.DataFrame,
    sector_data: dict[str, pd.DataFrame],
    sector_map:  dict[str, str],
    spy_ok_map:  dict[str, bool],
    spy_ret_map: dict[str, float],
    sec_ok_map:  dict[str, dict[str, bool]],
    etf_close:   dict[str, pd.Series],
    bo_lb:      int,
    trail_mult: float,
    hard_stop:  float,
    max_pos:    int,
    sim_start:  str,
    sim_end:    str,
) -> tuple[list[Trade], list[dict]]:

    warmup = max(260, bo_lb + RS_LOOKBACK + 10)
    all_dates = spy_df.index.tolist()
    sim_start_dt = pd.Timestamp(sim_start)
    sim_end_dt   = pd.Timestamp(sim_end)

    sim_indices = [i for i, d in enumerate(all_dates) if sim_start_dt <= d <= sim_end_dt]
    if not sim_indices:
        return [], []

    first_sim_idx = sim_indices[0]
    run_start_idx = max(0, first_sim_idx - warmup)

    cash = INITIAL_CAPITAL
    positions: dict[str, Pos]   = {}
    etf_pos:   dict[str, float] = {}  # ticker -> shares
    trades:    list[Trade]      = []
    equity_curve: list[dict]    = []

    prev_week: tuple | None    = None
    current_scores: list[tuple[str, float]] = []
    prev_bull: bool | None     = None

    for idx in range(run_start_idx, len(all_dates)):
        date = all_dates[idx]
        if date > sim_end_dt:
            break
        date_str = date.strftime("%Y-%m-%d")
        in_sim   = date >= sim_start_dt

        spy_ok = spy_ok_map.get(date_str, True)
        bull   = _spy_bull(etf_close, date)
        reg_chg = (prev_bull is not None and bull != prev_bull)

        # ── ① ETFリバランス（毎週月曜） ──
        week_key = (date.year, date.isocalendar()[1])
        if week_key != prev_week:
            prev_week = week_key
            if bull:
                current_scores = _calc_momentum_scores(etf_close, date)
                targets = _select_targets(etf_close, date, current_scores)
                # 外れたETFを売却
                for tk in list(etf_pos.keys()):
                    if tk not in targets:
                        p = _etf_price(etf_close, tk, date)
                        if p:
                            cash += etf_pos[tk] * p
                        del etf_pos[tk]
                # 余剰キャッシュをターゲットETFへ
                if targets and cash > 1.0:
                    alloc_each = cash / len(targets)
                    for tk in targets:
                        p = _etf_price(etf_close, tk, date)
                        if p and p > 0:
                            etf_pos[tk] = etf_pos.get(tk, 0.0) + alloc_each / p
                            cash -= alloc_each
            else:
                # ベア: SGOV待避
                for tk in list(etf_pos.keys()):
                    if tk != SAFE_ETF:
                        p = _etf_price(etf_close, tk, date)
                        if p:
                            cash += etf_pos[tk] * p
                        del etf_pos[tk]
                if cash > 1.0:
                    p = _etf_price(etf_close, SAFE_ETF, date)
                    if p:
                        etf_pos[SAFE_ETF] = etf_pos.get(SAFE_ETF, 0.0) + cash / p
                        cash = 0.0

        # ── ② レジーム変化対応 ──
        if reg_chg:
            if not bull:
                # ブル→ベア
                for tk in list(etf_pos.keys()):
                    if tk != SAFE_ETF:
                        p = _etf_price(etf_close, tk, date)
                        if p:
                            cash += etf_pos[tk] * p
                        del etf_pos[tk]
                if cash > 1.0:
                    p = _etf_price(etf_close, SAFE_ETF, date)
                    if p:
                        etf_pos[SAFE_ETF] = etf_pos.get(SAFE_ETF, 0.0) + cash / p
                        cash = 0.0
            else:
                # ベア→ブル
                for tk in [SAFE_ETF]:
                    if tk in etf_pos:
                        p = _etf_price(etf_close, tk, date)
                        if p:
                            cash += etf_pos[tk] * p
                        del etf_pos[tk]
        elif bull:
            # 50MA割れチェック
            for tk in list(etf_pos.keys()):
                if tk == SAFE_ETF:
                    continue
                p  = _etf_price(etf_close, tk, date)
                ma = _etf_ma(etf_close, tk, date)
                if p and ma and p < ma:
                    cash += etf_pos[tk] * p
                    del etf_pos[tk]

        prev_bull = bull

        # ── ③ V2 エグジット判定 ──
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            df  = price_data.get(ticker)
            if df is None or date not in df.index:
                continue
            row        = df.loc[date]
            today_high = float(row["High"])
            today_low  = float(row["Low"])
            today_open = float(row["Open"])

            if today_high > pos.highest_high:
                pos.highest_high = today_high
            atr = calc_atr(df.loc[:date])
            if atr:
                pos.atr = atr
            if pos.atr > 0:
                new_trail = pos.highest_high - pos.atr * trail_mult
                if new_trail > pos.stop_price:
                    pos.stop_price = new_trail
            pos.holding_days += 1

            if today_low <= pos.stop_price:
                exit_price = float(np.clip(pos.stop_price, today_low, today_high))
                if today_open < pos.stop_price:
                    exit_price = today_open
                pl_pct = (exit_price - pos.entry_price) / pos.entry_price
                cash  += pos.allocated + pos.allocated * pl_pct
                if in_sim:
                    trades.append(Trade(
                        pl_pct       = round(pl_pct * 100, 2),
                        holding_days = pos.holding_days,
                        ticker       = ticker,
                        entry_date   = pos.entry_date,
                        exit_date    = date_str,
                        entry_price  = pos.entry_price,
                        exit_price   = round(exit_price, 4),
                        allocated    = pos.allocated,
                    ))
                del positions[ticker]

        # ── ④ V2 エントリー候補スキャン ──
        if spy_ok and len(positions) < max_pos:
            held = set(positions)
            candidates = []

            for ticker, df in price_data.items():
                if ticker in held or ticker == "SPY":
                    continue
                if date not in df.index:
                    continue
                loc = df.index.get_loc(date)
                if loc < bo_lb:
                    continue

                close  = df["Close"].astype(float)
                volume = df["Volume"].astype(float) if "Volume" in df.columns else None

                cur_close = float(close.iloc[loc])
                prev_high = float(close.iloc[loc - bo_lb: loc].max())
                if cur_close <= prev_high:
                    continue

                vol_ratio = 1.0
                if volume is not None and loc >= VOLUME_LOOKBACK:
                    avg_vol = float(volume.iloc[loc - VOLUME_LOOKBACK: loc].mean())
                    cur_vol = float(volume.iloc[loc])
                    if avg_vol > 0:
                        vol_ratio = cur_vol / avg_vol
                        if vol_ratio < MIN_VOLUME_MULT:
                            continue

                spy_r = spy_ret_map.get(date_str)
                if spy_r is None or loc < RS_LOOKBACK:
                    continue
                stk_r = (cur_close / float(close.iloc[loc - RS_LOOKBACK])) - 1
                if stk_r <= spy_r:
                    continue

                etf_filt = sector_map.get(ticker)
                if etf_filt and not sec_ok_map.get(date_str, {}).get(etf_filt, True):
                    continue

                bo_str = (cur_close - prev_high) / prev_high
                candidates.append((bo_str * vol_ratio, ticker, df, loc))

            candidates.sort(reverse=True, key=lambda x: x[0])

            for _, ticker, df, loc in candidates:
                if len(positions) >= max_pos:
                    break
                if ticker in positions:
                    continue
                if loc + 1 >= len(df):
                    continue

                close_a     = df["Close"].astype(float)
                lows_a      = df["Low"].astype(float)
                entry_price = float(df.iloc[loc + 1]["Open"])
                if entry_price <= 0:
                    continue
                low_26w      = float(lows_a.iloc[max(0, loc - STOP_LOOKBACK): loc + 1].min())
                hard_stop_p  = entry_price * (1 - hard_stop)
                initial_stop = max(low_26w, hard_stop_p)
                if initial_stop >= entry_price:
                    initial_stop = hard_stop_p
                sl_pct = (entry_price - initial_stop) / entry_price
                if sl_pct < 0.005:
                    continue

                etf_val  = sum(etf_pos[tk] * (_etf_price(etf_close, tk, date) or 0) for tk in etf_pos)
                total_eq = cash + sum(p.allocated for p in positions.values()) + etf_val
                alloc    = min(total_eq * RISK_PER_TRADE / sl_pct, total_eq * MAX_ALLOC_PCT, cash + etf_val)

                # キャッシュ不足時はETFから売却（スコアの低い=弱いETFから）
                if cash < alloc and etf_pos:
                    shortage  = alloc - cash
                    score_map = {tk: s for tk, s in current_scores}
                    ranked = sorted(
                        etf_pos.keys(),
                        key=lambda tk: (0 if tk == SAFE_ETF else 1, score_map.get(tk, 999)),
                        reverse=True,
                    )
                    for tk in ranked:
                        if shortage <= 0:
                            break
                        p = _etf_price(etf_close, tk, date)
                        if not p:
                            continue
                        sell = min(shortage, etf_pos[tk] * p)
                        etf_pos[tk] -= sell / p
                        if etf_pos[tk] < 1e-9:
                            del etf_pos[tk]
                        cash += sell
                        shortage -= sell

                alloc = min(alloc, cash)
                if alloc < 50:
                    continue

                atr_val   = calc_atr(df.iloc[: loc + 1]) or (float(close_a.iloc[loc]) * 0.02)
                next_date = df.index[loc + 1].strftime("%Y-%m-%d")
                positions[ticker] = Pos(
                    ticker=ticker, entry_date=next_date,
                    entry_price=entry_price, stop_price=initial_stop,
                    highest_high=entry_price, allocated=round(alloc, 2), atr=atr_val,
                )
                cash -= alloc

        # ── ⑤ 日次スナップショット ──
        if in_sim:
            etf_val   = sum(etf_pos[tk] * (_etf_price(etf_close, tk, date) or 0) for tk in etf_pos)
            stock_val = sum(
                p.allocated * float(price_data[t].loc[date, "Close"]) / p.entry_price
                if (df := price_data.get(t)) is not None and date in df.index
                else p.allocated
                for t, p in positions.items()
            )
            equity_curve.append({"date": date_str, "equity": round(cash + stock_val + etf_val, 2)})

    return trades, equity_curve


# ============================================================
# 集計
# ============================================================
def summarize(trades: list[Trade], equity_curve: list[dict]) -> dict:
    if not trades or not equity_curve:
        return {"n": 0, "wr": 0.0, "avg_pl": 0.0, "total_ret": 0.0,
                "ann_ret": 0.0, "max_dd": 0.0, "sharpe": 0.0, "calmar": 0.0, "final": INITIAL_CAPITAL}
    pl    = [t.pl_pct for t in trades]
    wins  = [p for p in pl if p > 0]
    losses = [p for p in pl if p <= 0]
    eq    = pd.DataFrame(equity_curve)
    final = float(eq["equity"].iloc[-1])
    arr   = eq["equity"].values
    peak  = np.maximum.accumulate(arr)
    dd    = (arr - peak) / peak * 100
    max_dd = float(dd.min())
    dr    = pd.Series(arr).pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    total_ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    n_years = len(eq) / 252
    ann_ret = ((final / INITIAL_CAPITAL) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0
    calmar  = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    gross_profit = sum(t.allocated * t.pl_pct / 100 for t in trades if t.pl_pct > 0)
    gross_loss   = abs(sum(t.allocated * t.pl_pct / 100 for t in trades if t.pl_pct <= 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 9.99
    avg_hold_win  = float(np.mean([t.holding_days for t in trades if t.pl_pct > 0])) if wins else 0.0
    avg_hold_loss = float(np.mean([t.holding_days for t in trades if t.pl_pct <= 0])) if losses else 0.0
    return {
        "n": len(trades), "wr": len(wins) / len(trades) * 100,
        "avg_pl": float(np.mean(pl)),
        "avg_win": float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
        "max_win": float(max(pl)), "max_loss": float(min(pl)),
        "profit_factor": profit_factor,
        "avg_hold": float(np.mean([t.holding_days for t in trades])),
        "avg_hold_win": avg_hold_win,
        "avg_hold_loss": avg_hold_loss,
        "total_ret": total_ret, "ann_ret": ann_ret,
        "max_dd": max_dd, "sharpe": sharpe, "calmar": calmar, "final": final,
        "gross_profit": gross_profit, "gross_loss": gross_loss,
    }


def print_detail(label: str, trades: list[Trade], equity_curve: list[dict], s: dict) -> None:
    """詳細レポートを出力する"""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  詳細レポート: {label}")
    print(sep)

    # ── 損益サマリー ──
    print("\n【損益サマリー】")
    print(f"  取引件数      : {s['n']} 件")
    print(f"  勝率          : {s['wr']:.1f}%")
    print(f"  平均 P/L      : {s['avg_pl']:+.2f}%")
    print(f"  平均勝ち      : {s['avg_win']:+.2f}%")
    print(f"  平均負け      : {s['avg_loss']:+.2f}%")
    print(f"  最大勝ち      : {s['max_win']:+.2f}%")
    print(f"  最大負け      : {s['max_loss']:+.2f}%")
    print(f"  プロフィットF : {s['profit_factor']:.3f}")
    print(f"  総利益        : ${s['gross_profit']:>8,.2f}")
    print(f"  総損失        : ${s['gross_loss']:>8,.2f}")
    print(f"  平均保有日数  : {s['avg_hold']:.1f}d  (勝ち:{s['avg_hold_win']:.1f}d / 負け:{s['avg_hold_loss']:.1f}d)")

    # ── パフォーマンス指標 ──
    print("\n【パフォーマンス】")
    print(f"  総リターン    : {s['total_ret']:+.1f}%")
    print(f"  年率リターン  : {s['ann_ret']:+.1f}%")
    print(f"  最大DD        : {s['max_dd']:.1f}%")
    print(f"  Sharpe比      : {s['sharpe']:.3f}")
    print(f"  Calmar比      : {s['calmar']:.2f}")
    print(f"  最終資産      : ${s['final']:,.2f}  (初期: ${INITIAL_CAPITAL:,.2f})")

    # ── 年別リターン ──
    if equity_curve:
        eq = pd.DataFrame(equity_curve)
        eq["date"] = pd.to_datetime(eq["date"])
        eq["year"] = eq["date"].dt.year
        print("\n【年別リターン】")
        print(f"  {'年':<6} {'年間Ret':>9} {'最終残高':>12}")
        print("  " + "-" * 30)
        prev_val = INITIAL_CAPITAL
        for yr in sorted(eq["year"].unique()):
            yr_df = eq[eq["year"] == yr]
            end_val = float(yr_df["equity"].iloc[-1])
            yr_ret  = (end_val - prev_val) / prev_val * 100
            print(f"  {yr:<6} {yr_ret:>+8.1f}%   ${end_val:>10,.2f}")
            prev_val = end_val

    # ── P/L分布 ──
    pl_arr = [t.pl_pct for t in trades]
    bins = [-100, -15, -10, -5, 0, 5, 10, 20, 50, 200]
    labels_bin = ["<-15%", "-15〜-10%", "-10〜-5%", "-5〜0%",
                  "0〜+5%", "+5〜+10%", "+10〜+20%", "+20〜+50%", ">+50%"]
    print("\n【P/L分布】")
    print(f"  {'区間':<12} {'件数':>5} {'割合':>7}")
    print("  " + "-" * 28)
    for i, lbl in enumerate(labels_bin):
        lo, hi = bins[i], bins[i + 1]
        cnt = sum(1 for p in pl_arr if lo <= p < hi)
        pct = cnt / len(pl_arr) * 100
        bar = "█" * int(pct / 2)
        print(f"  {lbl:<12} {cnt:>5d}  {pct:>5.1f}%  {bar}")

    # ── トップ10勝ち取引 ──
    winners = sorted([t for t in trades if t.pl_pct > 0], key=lambda t: t.pl_pct, reverse=True)[:10]
    print("\n【トップ10 勝ち取引】")
    print(f"  {'#':>2}  {'銘柄':<8} {'エントリー':>12} {'エグジット':>12} "
          f"{'保有':>5} {'P/L':>8} {'投資額':>9}")
    print("  " + "-" * 66)
    for i, t in enumerate(winners, 1):
        print(f"  {i:>2}  {t.ticker:<8} {t.entry_date:>12} {t.exit_date:>12} "
              f"{t.holding_days:>4}d  {t.pl_pct:>+7.2f}%  ${t.allocated:>7,.0f}")

    # ── トップ10負け取引 ──
    losers = sorted([t for t in trades if t.pl_pct <= 0], key=lambda t: t.pl_pct)[:10]
    print("\n【トップ10 負け取引】")
    print(f"  {'#':>2}  {'銘柄':<8} {'エントリー':>12} {'エグジット':>12} "
          f"{'保有':>5} {'P/L':>8} {'投資額':>9}")
    print("  " + "-" * 66)
    for i, t in enumerate(losers, 1):
        print(f"  {i:>2}  {t.ticker:<8} {t.entry_date:>12} {t.exit_date:>12} "
              f"{t.holding_days:>4}d  {t.pl_pct:>+7.2f}%  ${t.allocated:>7,.0f}")


# ============================================================
# メイン
# ============================================================
def main() -> None:
    print("=" * 72)
    print("  Strategy V2 + ETF オーバーレイ vs ベースライン")
    print(f"  期間: {SIM_START} 〜 {SIM_END}")
    print(f"  V2: RISK={RISK_PER_TRADE*100:.1f}%, MAX_ALLOC={MAX_ALLOC_PCT*100:.0f}%")
    print("=" * 72)

    db = CacheDB()
    db.initialize()

    print("\n⏳ 銘柄スクリーニング中...")
    iwv     = fetch_iwv_holdings(db)
    tickers = screen_tier1(iwv, db)
    print(f"  → {len(tickers)} 銘柄")

    print("\n⏳ 株価データ取得中（キャッシュ利用）...")
    sec_etfs    = list(SECTOR_ETF_MAP.values())
    all_symbols = sorted(set(tickers) | {"SPY"} | set(sec_etfs))
    price_all   = fetch_prices(all_symbols, db=db, period="6y")

    spy_df      = price_all.pop("SPY", None)
    sector_data = {etf: price_all.pop(etf) for etf in sec_etfs if etf in price_all}
    price_data  = price_all
    print(f"  → 個別銘柄: {len(price_data)}  セクターETF: {len(sector_data)}")

    sector_map = _build_sector_map()

    spy_c   = spy_df["Close"].astype(float)
    spy_ma  = spy_c.rolling(MARKET_FILTER_MA).mean()
    spy_ret = spy_c.pct_change(RS_LOOKBACK)

    spy_ok_map, spy_ret_map = {}, {}
    for d in spy_df.index:
        ds = d.strftime("%Y-%m-%d")
        spy_ok_map[ds]  = bool(spy_c[d] >= spy_ma[d]) if not np.isnan(spy_ma[d]) else True
        if not np.isnan(spy_ret[d]):
            spy_ret_map[ds] = float(spy_ret[d])

    sec_ok_map: dict[str, dict[str, bool]] = {}
    for etf, df in sector_data.items():
        ec = df["Close"].astype(float)
        em = ec.rolling(SECTOR_MA).mean()
        for d in df.index:
            ds = d.strftime("%Y-%m-%d")
            if ds not in sec_ok_map:
                sec_ok_map[ds] = {}
            sec_ok_map[ds][etf] = bool(ec[d] >= em[d]) if not np.isnan(em[d]) else True

    # ── ETFデータ取得（yfinance、SGOVを含む）──
    print(f"\n⏳ ETFデータ取得中（yfinance）...")
    etf_tickers = ETF_UNIVERSE + [SAFE_ETF]
    raw = yf.download(etf_tickers, start=DATA_START, end=SIM_END,
                      auto_adjust=True, progress=False)
    etf_close: dict[str, pd.Series] = {}
    for tk in etf_tickers:
        try:
            s = raw["Close"][tk].dropna()
            etf_close[tk] = s
        except Exception:
            pass
    print(f"  → {len(etf_close)} 本取得完了")

    # ── シミュレーション実行 ──
    print("\n⏳ シミュレーション実行中...\n")

    results = []
    all_trades: list[tuple[str, list[Trade], list[dict]]] = []
    for cand in CANDIDATES:
        label = cand["label"]
        print(f"  [{label}] ...", end="", flush=True)
        trades, eq = run_sim_with_etf(
            price_data, spy_df, sector_data, sector_map,
            spy_ok_map, spy_ret_map, sec_ok_map,
            etf_close,
            cand["bo_lb"], cand["trail_mult"], cand["hard_stop"], cand["max_pos"],
            SIM_START, SIM_END,
        )
        s = summarize(trades, eq)
        s["label"] = f"V2-{label}(+ETF)"
        results.append(s)
        all_trades.append((s["label"], trades, eq))
        print(f" 完了 (n={s['n']}, total={s['total_ret']:+.1f}%)")

    # ── 比較テーブル ──
    print("\n" + "=" * 72)
    print("  比較結果")
    print("=" * 72)
    print(f"  {'戦略':<20} {'件数':>5} {'勝率':>6} {'avg':>8} "
          f"{'総Ret':>9} {'年率':>7} {'MaxDD':>7} {'Sharpe':>8} {'Calmar':>8} {'最終資産':>10}")
    print("  " + "-" * 86)

    b = BASELINE
    n_years_b = 5.15
    ann_b = ((1 + b["total_ret"] / 100) ** (1 / n_years_b) - 1) * 100
    print(f"  {'ベースライン(ETF込)':<20} {'—':>5}  {'60.3%':>6}  {'+0.93%':>7}  "
          f"{b['total_ret']:>+8.1f}%  {ann_b:>+6.1f}%  {b['max_dd']:>+6.1f}%  "
          f"{b['sharpe']:>7.3f}  {b['calmar']:>7.2f}  ${b['final']:>8,.0f}")
    print("  " + "-" * 86)

    for s in results:
        vs = s["total_ret"] - b["total_ret"]
        marker = "★" if s["total_ret"] > b["total_ret"] else " "
        print(f"{marker} {s['label']:<20} {s['n']:>5d}  {s['wr']:>5.1f}%  "
              f"{s['avg_pl']:>+7.2f}%  "
              f"{s['total_ret']:>+8.1f}%  {s['ann_ret']:>+6.1f}%  "
              f"{s['max_dd']:>+6.1f}%  "
              f"{s['sharpe']:>7.3f}  {s['calmar']:>7.2f}  ${s['final']:>8,.0f}  "
              f"({'vs Base: {:+.1f}pt'.format(vs)})")

    print("\n  ※ V2: RISK=1.5%, MAX_ALLOC=20%, ブレイクアウト + ETFオーバーレイ（待機資金）")
    print("  ※ ベースライン: RISK=0.5%, MAX_ALLOC=13/14%, ディップバイ + ETFオーバーレイ")

    # ── 詳細レポート（各候補） ──
    for (lbl, trades, eq), s in zip(all_trades, results):
        print_detail(lbl, trades, eq, s)

    # エクイティカーブCSV保存（各候補）
    safe_names = {
        "V2-高リターン(+ETF)": "v2_highret_daily.csv",
        "V2-バランス(+ETF)":   "v2_balance_daily.csv",
        "V2-低リスク(+ETF)":   "v2_lowrisk_daily.csv",
    }
    for (lbl, trades, eq) in all_trades:
        fname = safe_names.get(lbl)
        if fname:
            eq_out = _ROOT / "output" / "backtest" / fname
            pd.DataFrame(eq).to_csv(eq_out, index=False)
            print(f"📁 エクイティCSV: {eq_out}")

    # CSV保存（全取引履歴）
    trades_out = _ROOT / "output" / "backtest" / "v2_etf_overlay_trades.csv"
    all_rows = []
    for (lbl, trades, _) in all_trades:
        for t in trades:
            all_rows.append({
                "strategy": lbl, "ticker": t.ticker,
                "entry_date": t.entry_date, "exit_date": t.exit_date,
                "entry_price": t.entry_price, "exit_price": t.exit_price,
                "pl_pct": t.pl_pct, "holding_days": t.holding_days,
                "allocated": t.allocated,
            })
    pd.DataFrame(all_rows).to_csv(trades_out, index=False)
    print(f"\n📁 取引履歴CSV: {trades_out}")

    out = _ROOT / "output" / "backtest" / "v2_etf_overlay_results.csv"
    rows = [{"label": b["label"], "total_ret": b["total_ret"], "ann_ret": ann_b,
             "max_dd": b["max_dd"], "sharpe": b["sharpe"], "calmar": b["calmar"], "final": b["final"]}]
    for s in results:
        rows.append({k: s[k] for k in ["label", "n", "wr", "avg_pl", "total_ret",
                                        "ann_ret", "max_dd", "sharpe", "calmar", "final"]})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n📁 結果CSV: {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
