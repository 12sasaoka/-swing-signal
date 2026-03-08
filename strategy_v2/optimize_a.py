"""
Strategy A — パラメータ最適化（グリッドサーチ）

RS+SEC+LB フィルター固定で主要パラメータを探索する。

探索パラメータ:
  bo_lb      : ブレイクアウト期間  [63, 126, 189]
  trail_mult : トレーリング幅      [2.5, 3.0, 3.5, 4.0]
  hard_stop  : ハードストップ %    [0.08, 0.10, 0.12]
  max_pos    : 最大同時保有数      [3, 5, 7]

固定パラメータ（sweep_a.py で確認済み）:
  use_rs=True, use_sec=True, short_lb=True
  RS_LOOKBACK=20, SECTOR_MA=50, MIN_VOLUME_MULT=1.5
  RISK_PER_TRADE=1.5%, MAX_ALLOC_PCT=20%
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from io import StringIO
from itertools import product
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
# 探索グリッド
# ============================================================
PARAM_GRID = {
    "bo_lb":      [63, 126, 189],
    "trail_mult": [2.5, 3.0, 3.5, 4.0],
    "hard_stop":  [0.08, 0.10, 0.12],
    "max_pos":    [3, 5, 7],
}

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
    print(f"  → {len(sec_map)} 銘柄にマッピング")
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


# ============================================================
# シミュレーション
# ============================================================
def run_sim(
    price_data:  dict[str, pd.DataFrame],
    spy_df:      pd.DataFrame,
    sector_data: dict[str, pd.DataFrame],
    sector_map:  dict[str, str],
    spy_ok_map:  dict[str, bool],
    spy_ret_map: dict[str, float],
    sec_ok_map:  dict[str, dict[str, bool]],
    # 探索パラメータ
    bo_lb:      int,
    trail_mult: float,
    hard_stop:  float,
    max_pos:    int,
) -> tuple[list[Trade], list[dict]]:

    warmup    = max(260, bo_lb + RS_LOOKBACK + 10)
    all_dates = spy_df.index.tolist()
    cash      = INITIAL_CAPITAL
    positions: dict[str, Pos] = {}
    trades:    list[Trade]    = []
    equity_curve: list[dict]  = []

    for idx, date in enumerate(all_dates):
        date_str = date.strftime("%Y-%m-%d")

        if idx < warmup:
            equity_curve.append({"date": date_str, "equity": INITIAL_CAPITAL})
            continue

        spy_ok = spy_ok_map.get(date_str, True)

        # ── エグジット判定 ──
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
                trades.append(Trade(
                    pl_pct       = round(pl_pct * 100, 2),
                    holding_days = pos.holding_days,
                ))
                del positions[ticker]

        # ── エントリー候補スキャン ──
        if spy_ok and len(positions) < max_pos:
            held       = set(positions)
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

                # RS フィルター
                spy_r = spy_ret_map.get(date_str)
                if spy_r is None or loc < RS_LOOKBACK:
                    continue
                stk_r = (cur_close / float(close.iloc[loc - RS_LOOKBACK])) - 1
                if stk_r <= spy_r:
                    continue

                # セクターETF フィルター
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

                close_a      = df["Close"].astype(float)
                lows_a       = df["Low"].astype(float)
                entry_price  = float(df.iloc[loc + 1]["Open"])
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
                    ticker       = ticker,
                    entry_date   = next_date,
                    entry_price  = entry_price,
                    stop_price   = initial_stop,
                    highest_high = entry_price,
                    allocated    = round(alloc, 2),
                    atr          = atr_val,
                )
                cash -= alloc

        stock_val = sum(
            p.allocated * float(price_data[t].loc[date, "Close"]) / p.entry_price
            if (df := price_data.get(t)) is not None and date in df.index
            else p.allocated
            for t, p in positions.items()
        )
        equity_curve.append({"date": date_str, "equity": round(cash + stock_val, 2)})

    return trades, equity_curve


# ============================================================
# 集計
# ============================================================
def summarize(trades: list[Trade], equity_curve: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "wr": 0.0, "avg_pl": 0.0, "avg_win": 0.0,
                "avg_loss": 0.0, "total_ret": 0.0, "max_dd": 0.0, "sharpe": 0.0}
    pl     = [t.pl_pct for t in trades]
    wins   = [p for p in pl if p > 0]
    losses = [p for p in pl if p <= 0]
    eq     = pd.DataFrame(equity_curve)
    final  = float(eq["equity"].iloc[-1])
    arr    = eq["equity"].values
    peak   = np.maximum.accumulate(arr)
    dd     = (arr - peak) / peak * 100
    dr     = pd.Series(arr).pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    return {
        "n":        len(trades),
        "wr":       len(wins) / len(trades) * 100,
        "avg_pl":   float(np.mean(pl)),
        "avg_win":  float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
        "total_ret":(final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        "max_dd":   float(dd.min()),
        "sharpe":   sharpe,
    }


# ============================================================
# メイン
# ============================================================
def main() -> None:
    total_runs = (len(PARAM_GRID["bo_lb"]) * len(PARAM_GRID["trail_mult"]) *
                  len(PARAM_GRID["hard_stop"]) * len(PARAM_GRID["max_pos"]))

    print("=" * 72)
    print("  Strategy A — パラメータ最適化 (RS+SEC+LB 固定)")
    print(f"  探索数: {total_runs} 通り")
    print("=" * 72)

    db = CacheDB()
    db.initialize()

    print("\n⏳ 銘柄スクリーニング中...")
    iwv     = fetch_iwv_holdings(db)
    tickers = screen_tier1(iwv, db)
    print(f"  → {len(tickers)} 銘柄")

    print("\n⏳ 株価データ取得中（6y）...")
    sec_etfs    = list(SECTOR_ETF_MAP.values())
    all_symbols = sorted(set(tickers) | {"SPY"} | set(sec_etfs))
    price_all   = fetch_prices(all_symbols, db=db, period="6y")

    spy_df      = price_all.pop("SPY", None)
    if spy_df is None or spy_df.empty:
        print("ERROR: SPY取得失敗")
        return
    sector_data = {etf: price_all.pop(etf) for etf in sec_etfs if etf in price_all}
    price_data  = price_all
    print(f"  → 個別銘柄: {len(price_data)}  セクターETF: {len(sector_data)}")

    print("\n⏳ セクターマップ取得中...")
    sector_map = _build_sector_map()

    # SPY マップを一度だけ構築（全runs共通）
    spy_c   = spy_df["Close"].astype(float)
    spy_ma  = spy_c.rolling(MARKET_FILTER_MA).mean()
    spy_ret = spy_c.pct_change(RS_LOOKBACK)

    spy_ok_map:  dict[str, bool]  = {}
    spy_ret_map: dict[str, float] = {}
    for d in spy_df.index:
        ds = d.strftime("%Y-%m-%d")
        spy_ok_map[ds]  = bool(spy_c[d] >= spy_ma[d]) if not np.isnan(spy_ma[d]) else True
        if not np.isnan(spy_ret[d]):
            spy_ret_map[ds] = float(spy_ret[d])

    # セクターETF MAマップを一度だけ構築
    sec_ok_map: dict[str, dict[str, bool]] = {}
    for etf, df in sector_data.items():
        ec = df["Close"].astype(float)
        em = ec.rolling(SECTOR_MA).mean()
        for d in df.index:
            ds = d.strftime("%Y-%m-%d")
            if ds not in sec_ok_map:
                sec_ok_map[ds] = {}
            sec_ok_map[ds][etf] = bool(ec[d] >= em[d]) if not np.isnan(em[d]) else True

    # ============================================================
    # グリッドサーチ
    # ============================================================
    all_results = []
    run_n = 0

    print(f"\n⏳ グリッドサーチ実行中... ({total_runs} runs)\n")
    print(f"  {'#':>4} {'bo_lb':>6} {'trail':>6} {'stop':>6} {'pos':>4} | "
          f"{'件数':>5} {'勝率':>6} {'avg':>7} {'total':>8} {'DD':>6} {'Sharpe':>7}")
    print("  " + "-" * 70)

    for bo_lb, trail_mult, hard_stop, max_pos in product(
        PARAM_GRID["bo_lb"],
        PARAM_GRID["trail_mult"],
        PARAM_GRID["hard_stop"],
        PARAM_GRID["max_pos"],
    ):
        run_n += 1
        trades, eq = run_sim(
            price_data, spy_df, sector_data, sector_map,
            spy_ok_map, spy_ret_map, sec_ok_map,
            bo_lb, trail_mult, hard_stop, max_pos,
        )
        s = summarize(trades, eq)
        s.update({"bo_lb": bo_lb, "trail": trail_mult, "stop": hard_stop, "pos": max_pos})
        all_results.append(s)

        print(f"  {run_n:>4d} {bo_lb:>6d} {trail_mult:>6.1f} {hard_stop*100:>5.0f}% {max_pos:>4d} | "
              f"{s['n']:>5d}  {s['wr']:>5.1f}%  {s['avg_pl']:>+6.2f}%  "
              f"{s['total_ret']:>+7.1f}%  {s['max_dd']:>5.1f}%  {s['sharpe']:>6.3f}")

    # ============================================================
    # 上位20件 (総リターン順)
    # ============================================================
    print("\n" + "=" * 72)
    print("  TOP 20 — 総リターン順")
    print(f"  {'bo_lb':>6} {'trail':>6} {'stop':>6} {'pos':>4} | "
          f"{'件数':>5} {'勝率':>6} {'avg':>7} {'avg勝':>7} {'avg負':>7} "
          f"{'total':>8} {'DD':>6} {'Sharpe':>7}")
    print("  " + "-" * 78)
    for s in sorted(all_results, key=lambda x: x["total_ret"], reverse=True)[:20]:
        print(f"  {s['bo_lb']:>6d} {s['trail']:>6.1f} {s['stop']*100:>5.0f}% {s['pos']:>4d} | "
              f"{s['n']:>5d}  {s['wr']:>5.1f}%  {s['avg_pl']:>+6.2f}%  "
              f"{s['avg_win']:>+6.2f}%  {s['avg_loss']:>+6.2f}%  "
              f"{s['total_ret']:>+7.1f}%  {s['max_dd']:>5.1f}%  {s['sharpe']:>6.3f}")

    print()
    print("  TOP 20 — Sharpe順")
    print(f"  {'bo_lb':>6} {'trail':>6} {'stop':>6} {'pos':>4} | "
          f"{'件数':>5} {'勝率':>6} {'avg':>7} {'total':>8} {'DD':>6} {'Sharpe':>7}")
    print("  " + "-" * 66)
    for s in sorted(all_results, key=lambda x: x["sharpe"], reverse=True)[:20]:
        print(f"  {s['bo_lb']:>6d} {s['trail']:>6.1f} {s['stop']*100:>5.0f}% {s['pos']:>4d} | "
              f"{s['n']:>5d}  {s['wr']:>5.1f}%  {s['avg_pl']:>+6.2f}%  "
              f"{s['total_ret']:>+7.1f}%  {s['max_dd']:>5.1f}%  {s['sharpe']:>6.3f}")
    print("=" * 72)

    # CSV保存
    out = _ROOT / "output" / "backtest" / "optimize_a_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).sort_values("total_ret", ascending=False).to_csv(out, index=False)
    print(f"\n📁 全結果CSV: {out}")


if __name__ == "__main__":
    main()
