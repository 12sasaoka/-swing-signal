"""
Strategy V2 vs ベースライン 比較

全期間 (2021-01-05 〜 2026-02-27) で TOP 3 候補を実行し、
ベースライン結果と並べて比較する。

V2 パラメータ: RISK=1.5%, MAX_ALLOC=20%（最適化時の設定のまま）
ベースライン:  +288.3%, Sharpe 2.394, MaxDD -13.43%, Calmar 21.47
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
# 比較対象
# ============================================================
CANDIDATES = [
    {"label": "高リターン", "bo_lb": 126, "trail_mult": 3.5, "hard_stop": 0.08, "max_pos": 7},
    {"label": "バランス",   "bo_lb": 189, "trail_mult": 3.5, "hard_stop": 0.08, "max_pos": 5},
    {"label": "低リスク",   "bo_lb": 189, "trail_mult": 3.5, "hard_stop": 0.08, "max_pos": 3},
]

BASELINE = {
    "label":     "ベースライン",
    "total_ret": 288.3,
    "sharpe":    2.394,
    "max_dd":    -13.43,
    "calmar":    21.47,
    "n":         1394,
    "wr":        60.3,
    "avg_pl":    0.93,
}

SIM_START = "2021-01-05"
SIM_END   = "2026-02-27"

# ============================================================
# 固定パラメータ
# ============================================================
INITIAL_CAPITAL  = 4_000.0
STOP_LOOKBACK    = 126
ATR_PERIOD       = 14
MIN_VOLUME_MULT  = 1.5
VOLUME_LOOKBACK  = 20
RISK_PER_TRADE   = 0.015   # V2: 1.5%
MAX_ALLOC_PCT    = 0.20    # V2: 20%
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
    print("  → IWV CSVからセクターマップ構築中...")
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


def run_sim(
    price_data, spy_df, sector_data, sector_map,
    spy_ok_map, spy_ret_map, sec_ok_map,
    bo_lb, trail_mult, hard_stop, max_pos,
    sim_start, sim_end,
):
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
    positions: dict[str, Pos] = {}
    trades: list[Trade] = []
    equity_curve: list[dict] = []

    for idx in range(run_start_idx, len(all_dates)):
        date = all_dates[idx]
        if date > sim_end_dt:
            break
        date_str = date.strftime("%Y-%m-%d")
        in_sim = date >= sim_start_dt

        spy_ok = spy_ok_map.get(date_str, True)

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
                    trades.append(Trade(round(pl_pct * 100, 2), pos.holding_days))
                del positions[ticker]

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

                etf = sector_map.get(ticker)
                if etf and not sec_ok_map.get(date_str, {}).get(etf, True):
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
                total_eq = cash + sum(p.allocated for p in positions.values())
                alloc = min(total_eq * RISK_PER_TRADE / sl_pct, total_eq * MAX_ALLOC_PCT, cash)
                if alloc < 50:
                    continue
                atr_val = calc_atr(df.iloc[: loc + 1]) or (float(close_a.iloc[loc]) * 0.02)
                next_date = df.index[loc + 1].strftime("%Y-%m-%d")
                positions[ticker] = Pos(
                    ticker=ticker, entry_date=next_date,
                    entry_price=entry_price, stop_price=initial_stop,
                    highest_high=entry_price, allocated=round(alloc, 2), atr=atr_val,
                )
                cash -= alloc

        if in_sim:
            stock_val = sum(
                p.allocated * float(price_data[t].loc[date, "Close"]) / p.entry_price
                if (df := price_data.get(t)) is not None and date in df.index
                else p.allocated
                for t, p in positions.items()
            )
            equity_curve.append({"date": date_str, "equity": round(cash + stock_val, 2)})

    return trades, equity_curve


def summarize(trades, equity_curve):
    if not trades or not equity_curve:
        return {"n": 0, "wr": 0.0, "avg_pl": 0.0, "total_ret": 0.0,
                "max_dd": 0.0, "sharpe": 0.0, "calmar": 0.0, "final": INITIAL_CAPITAL}
    pl    = [t.pl_pct for t in trades]
    wins  = [p for p in pl if p > 0]
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
    return {
        "n": len(trades), "wr": len(wins) / len(trades) * 100,
        "avg_pl": float(np.mean(pl)),
        "total_ret": total_ret, "ann_ret": ann_ret,
        "max_dd": max_dd, "sharpe": sharpe, "calmar": calmar,
        "final": final,
    }


def main():
    print("=" * 72)
    print("  Strategy V2 vs ベースライン 比較")
    print(f"  期間: {SIM_START} 〜 {SIM_END}")
    print(f"  V2パラメータ: RISK={RISK_PER_TRADE*100:.1f}%, MAX_ALLOC={MAX_ALLOC_PCT*100:.0f}%")
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

    print("\n⏳ シミュレーション実行中...\n")

    results = []
    for cand in CANDIDATES:
        label = cand["label"]
        print(f"  [{label}] ...", end="", flush=True)
        trades, eq = run_sim(
            price_data, spy_df, sector_data, sector_map,
            spy_ok_map, spy_ret_map, sec_ok_map,
            cand["bo_lb"], cand["trail_mult"], cand["hard_stop"], cand["max_pos"],
            SIM_START, SIM_END,
        )
        s = summarize(trades, eq)
        s["label"] = label
        s["bo_lb"] = cand["bo_lb"]
        s["trail"] = cand["trail_mult"]
        s["stop"]  = cand["hard_stop"]
        s["pos"]   = cand["max_pos"]
        results.append(s)
        print(f" 完了 (n={s['n']}, total={s['total_ret']:+.1f}%)")

    # ============================================================
    # 比較テーブル
    # ============================================================
    print("\n" + "=" * 72)
    print("  比較結果（全期間: 2021-01-05 〜 2026-02-27）")
    print("=" * 72)
    print(f"  {'戦略':<14} {'件数':>5} {'勝率':>6} {'avg P/L':>8} "
          f"{'総Ret':>9} {'年率':>7} {'MaxDD':>7} {'Sharpe':>8} {'Calmar':>8} {'最終資産':>10}")
    print("  " + "-" * 80)

    # ベースライン行
    b = BASELINE
    # ベースラインの年率・最終資産を計算（2021-01-05〜2026-02-27 ≈ 5.15年）
    n_years_base = 5.15
    ann_ret_base = ((1 + b["total_ret"] / 100) ** (1 / n_years_base) - 1) * 100
    final_base   = INITIAL_CAPITAL * (1 + b["total_ret"] / 100)
    print(f"  {'ベースライン':<14} {b['n']:>5d}  {b['wr']:>5.1f}%  {b['avg_pl']:>+7.2f}%  "
          f"{b['total_ret']:>+8.1f}%  {ann_ret_base:>+6.1f}%  {b['max_dd']:>+6.1f}%  "
          f"{b['sharpe']:>7.3f}  {b['calmar']:>7.2f}  ${final_base:>8,.0f}")
    print("  " + "-" * 80)

    for s in results:
        vs_ret = s["total_ret"] - b["total_ret"]
        marker = "★" if s["total_ret"] > b["total_ret"] else " "
        print(f"{marker} {'V2-'+s['label']:<13} {s['n']:>5d}  {s['wr']:>5.1f}%  "
              f"{s['avg_pl']:>+7.2f}%  "
              f"{s['total_ret']:>+8.1f}%  {s['ann_ret']:>+6.1f}%  "
              f"{s['max_dd']:>+6.1f}%  "
              f"{s['sharpe']:>7.3f}  {s['calmar']:>7.2f}  ${s['final']:>8,.0f}  "
              f"({'vs Base: {:+.1f}pt'.format(vs_ret)})")

    print("\n  ※ V2 パラメータ: RISK=1.5%, MAX_ALLOC=20%, ブレイクアウト戦略")
    print("  ※ ベースライン: RISK=0.5%, MAX_ALLOC=13/14%, ディップバイ戦略")
    print()

    # 各候補の詳細
    print("  パラメータ詳細:")
    for s in results:
        print(f"    {s['label']}: bo_lb={s['bo_lb']}, trail={s['trail']}, "
              f"stop={s['stop']*100:.0f}%, pos={s['pos']}")

    # CSV保存
    out = _ROOT / "output" / "backtest" / "v2_vs_baseline.csv"
    rows = [{"label": "ベースライン", "n": b["n"], "wr": b["wr"], "avg_pl": b["avg_pl"],
             "total_ret": b["total_ret"], "ann_ret": ann_ret_base,
             "max_dd": b["max_dd"], "sharpe": b["sharpe"], "calmar": b["calmar"],
             "final": final_base}]
    for s in results:
        rows.append({k: s[k] for k in ["label","n","wr","avg_pl","total_ret","ann_ret",
                                         "max_dd","sharpe","calmar","final"]})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n📁 比較CSV: {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
