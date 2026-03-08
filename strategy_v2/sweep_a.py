"""
Strategy A — フィルター組み合わせスイープ (2^4 = 16 runs)

4つのフィルター × 全組み合わせを一括検証し、結果をテーブルで比較する。

フィルター:
  F1 RS  : 相対強度 — 過去 RS_LOOKBACK 日リターンが SPY を上回る
  F2 PB  : 押し目待ち — ブレイクアウト後 PB_WAIT 日以内の最初の陰線翌日に入る
  F3 SEC : セクターETF — 同セクターETFが SECTOR_MA 日MAを上回る
  F4 LB  : 短期ルックバック — BREAKOUT_LOOKBACK=126（6ヶ月高値）
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
# 固定パラメータ
# ============================================================
INITIAL_CAPITAL   = 4_000.0
BREAKOUT_LOOKBACK = 252      # F4=False 時の基準
SHORT_LOOKBACK    = 126      # F4=True  時の短期版
STOP_LOOKBACK     = 126      # 26週安値をストップ基準に使う
HARD_STOP_PCT     = 0.12
TRAIL_ATR_MULT    = 3.0
ATR_PERIOD        = 14
MIN_VOLUME_MULT   = 1.5
VOLUME_LOOKBACK   = 20
MAX_POSITIONS     = 5
RISK_PER_TRADE    = 0.015
MAX_ALLOC_PCT     = 0.20
MARKET_FILTER_MA  = 200
WARMUP_DAYS       = 260

RS_LOOKBACK = 20   # F1: SPY比較期間
PB_WAIT     = 5    # F2: 押し目最大待ち日数
SECTOR_MA   = 50   # F3: セクターETF MAの期間

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
# セクターマップ (ticker → セクターETF)
# ============================================================
def _build_sector_map() -> dict[str, str]:
    """IWV CSV から ticker → セクターETF のマップを返す。data/db/sector_map.json にキャッシュ。"""
    cache = _ROOT / "data" / "db" / "sector_map.json"
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            return json.load(f)

    print("  → IWV CSVからセクターマップ構築中...")
    try:
        resp = requests.get(IWV_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"  WARNING: IWV CSV取得失敗({e}) → セクターフィルターは全銘柄通過")
        return {}

    lines = resp.text.split("\n")
    header_idx = next((i for i, l in enumerate(lines) if l.startswith("Ticker,")), None)
    if header_idx is None:
        return {}

    reader = csv.DictReader(StringIO("\n".join(lines[header_idx:])))
    sec_map: dict[str, str] = {}
    for row in reader:
        t = (row.get("Ticker") or "").strip('"').strip()
        s = (row.get("Sector") or "").strip('"').strip()
        if t and s and (etf := SECTOR_ETF_MAP.get(s)):
            sec_map[t] = etf

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(sec_map, f)
    print(f"  → セクターマップ: {len(sec_map)} 銘柄")
    return sec_map


# ============================================================
# ユーティリティ
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
    ticker:       str
    entry_date:   str
    exit_date:    str
    entry_price:  float
    exit_price:   float
    result:       str
    pl_pct:       float
    holding_days: int
    allocated:    float


# ============================================================
# シミュレーション（フィルターフラグ受け取り）
# ============================================================
def run_sim(
    price_data:  dict[str, pd.DataFrame],
    spy_df:      pd.DataFrame,
    sector_data: dict[str, pd.DataFrame],
    sector_map:  dict[str, str],
    use_rs:      bool,
    use_pb:      bool,
    use_sec:     bool,
    short_lb:    bool,
) -> tuple[list[Trade], list[dict]]:

    bo_lb  = SHORT_LOOKBACK if short_lb else BREAKOUT_LOOKBACK
    warmup = max(WARMUP_DAYS, bo_lb + RS_LOOKBACK + 10)

    # SPY 200MA & 相対強度マップ
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

    # セクターETF MAマップ
    sec_ok_map: dict[str, dict[str, bool]] = {}
    if use_sec:
        for etf, df in sector_data.items():
            ec = df["Close"].astype(float)
            em = ec.rolling(SECTOR_MA).mean()
            for d in df.index:
                ds = d.strftime("%Y-%m-%d")
                if ds not in sec_ok_map:
                    sec_ok_map[ds] = {}
                sec_ok_map[ds][etf] = bool(ec[d] >= em[d]) if not np.isnan(em[d]) else True

    all_dates = spy_df.index.tolist()
    cash      = INITIAL_CAPITAL
    positions: dict[str, Pos]   = {}
    # pending: ブレイクアウト確認済み、押し目待ち中 {ticker: {stop, atr, max_date}}
    pending:  dict[str, dict]   = {}
    # ready:   押し目確認済み、翌日オープンで入る {ticker: {stop, atr}}
    ready:    dict[str, dict]   = {}
    trades:   list[Trade]       = []
    equity_curve: list[dict]    = []

    for idx, date in enumerate(all_dates):
        date_str = date.strftime("%Y-%m-%d")

        if idx < warmup:
            equity_curve.append({"date": date_str, "equity": INITIAL_CAPITAL})
            continue

        spy_ok = spy_ok_map.get(date_str, True)

        # ── 既存ポジション: 最高値更新・ATR更新・トレーリング・エグジット ──
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
                new_trail = pos.highest_high - pos.atr * TRAIL_ATR_MULT
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
                    ticker       = ticker,
                    entry_date   = pos.entry_date,
                    exit_date    = date_str,
                    entry_price  = pos.entry_price,
                    exit_price   = round(exit_price, 4),
                    result       = "trail_stop" if pl_pct > 0 else "sl_hit",
                    pl_pct       = round(pl_pct * 100, 2),
                    holding_days = pos.holding_days,
                    allocated    = round(pos.allocated, 2),
                ))
                del positions[ticker]

        # ── PBモード: ready → 今日のオープンでエントリー ──
        if use_pb:
            for ticker in list(ready.keys()):
                if ticker in positions or len(positions) >= MAX_POSITIONS or not spy_ok:
                    del ready[ticker]
                    continue
                r  = ready[ticker]
                df = price_data.get(ticker)
                if df is None or date not in df.index:
                    del ready[ticker]
                    continue
                entry_price = float(df.loc[date, "Open"])
                if entry_price <= 0:
                    del ready[ticker]
                    continue
                sl_pct = (entry_price - r["stop"]) / entry_price
                if sl_pct < 0.005:
                    del ready[ticker]
                    continue
                total_eq = cash + sum(p.allocated for p in positions.values())
                alloc = min(total_eq * RISK_PER_TRADE / sl_pct, total_eq * MAX_ALLOC_PCT, cash)
                if alloc < 50:
                    del ready[ticker]
                    continue
                atr = r["atr"] or (entry_price * 0.02)
                positions[ticker] = Pos(
                    ticker       = ticker,
                    entry_date   = date_str,
                    entry_price  = entry_price,
                    stop_price   = r["stop"],
                    highest_high = entry_price,
                    allocated    = round(alloc, 2),
                    atr          = atr,
                )
                cash -= alloc
                del ready[ticker]

        # ── PBモード: pending → 押し目チェック ──
        if use_pb:
            for ticker in list(pending.keys()):
                if ticker in positions or ticker in ready:
                    del pending[ticker]
                    continue
                p = pending[ticker]
                if date > p["max_date"]:
                    del pending[ticker]
                    continue
                df = price_data.get(ticker)
                if df is None or date not in df.index:
                    continue
                loc = df.index.get_loc(date)
                if loc < 1:
                    continue
                if float(df["Close"].iloc[loc]) < float(df["Close"].iloc[loc - 1]):
                    # 押し目発見 → 翌日オープンでエントリー
                    ready[ticker] = {"stop": p["stop"], "atr": p["atr"]}
                    del pending[ticker]

        # ── 新規エントリー候補スキャン ──
        if spy_ok and len(positions) < MAX_POSITIONS:
            held       = set(positions) | set(pending) | set(ready)
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

                # 出来高フィルター
                vol_ratio = 1.0
                if volume is not None and loc >= VOLUME_LOOKBACK:
                    avg_vol = float(volume.iloc[loc - VOLUME_LOOKBACK: loc].mean())
                    cur_vol = float(volume.iloc[loc])
                    if avg_vol > 0:
                        vol_ratio = cur_vol / avg_vol
                        if vol_ratio < MIN_VOLUME_MULT:
                            continue

                # F1: 相対強度
                if use_rs:
                    spy_r = spy_ret_map.get(date_str)
                    if spy_r is None or loc < RS_LOOKBACK:
                        continue
                    stk_r = (cur_close / float(close.iloc[loc - RS_LOOKBACK])) - 1
                    if stk_r <= spy_r:
                        continue

                # F3: セクターETF
                if use_sec:
                    etf = sector_map.get(ticker)
                    if etf and not sec_ok_map.get(date_str, {}).get(etf, True):
                        continue

                bo_str = (cur_close - prev_high) / prev_high
                candidates.append((bo_str * vol_ratio, ticker, df, loc))

            candidates.sort(reverse=True, key=lambda x: x[0])

            for _, ticker, df, loc in candidates:
                if len(positions) >= MAX_POSITIONS:
                    break
                if ticker in positions or ticker in pending or ticker in ready:
                    continue

                close_a = df["Close"].astype(float)
                lows_a  = df["Low"].astype(float)
                low_26w = float(lows_a.iloc[max(0, loc - STOP_LOOKBACK): loc + 1].min())
                atr_val = calc_atr(df.iloc[: loc + 1]) or (float(close_a.iloc[loc]) * 0.02)

                if not use_pb:
                    # 翌日オープンでエントリー（ベースと同じ）
                    if loc + 1 >= len(df):
                        continue
                    entry_price  = float(df.iloc[loc + 1]["Open"])
                    if entry_price <= 0:
                        continue
                    hard_stop    = entry_price * (1 - HARD_STOP_PCT)
                    initial_stop = max(low_26w, hard_stop)
                    if initial_stop >= entry_price:
                        initial_stop = hard_stop
                    sl_pct = (entry_price - initial_stop) / entry_price
                    if sl_pct < 0.005:
                        continue
                    total_eq = cash + sum(p.allocated for p in positions.values())
                    alloc = min(total_eq * RISK_PER_TRADE / sl_pct, total_eq * MAX_ALLOC_PCT, cash)
                    if alloc < 50:
                        continue
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
                else:
                    # PBモード: pending に追加し押し目を待つ
                    cur_c    = float(close_a.iloc[loc])
                    hard_stp = cur_c * (1 - HARD_STOP_PCT)
                    init_stp = max(low_26w, hard_stp)
                    if init_stp >= cur_c:
                        init_stp = hard_stp
                    max_wait_date = all_dates[min(idx + PB_WAIT, len(all_dates) - 1)]
                    pending[ticker] = {"stop": init_stp, "atr": atr_val, "max_date": max_wait_date}

        # ── 評価額記録 ──
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
                "avg_loss": 0.0, "max_win": 0.0, "total_ret": 0.0,
                "max_dd": 0.0, "sharpe": 0.0}
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
        "max_win":  float(max(pl)),
        "total_ret":(final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        "max_dd":   float(dd.min()),
        "sharpe":   sharpe,
    }


# ============================================================
# メイン
# ============================================================
def main() -> None:
    print("=" * 72)
    print("  Strategy A — フィルター組み合わせスイープ (2^4 = 16 runs)")
    print("  F1=RS(相対強度)  F2=PB(押し目)  F3=SEC(セクター)  F4=LB(6ヶ月高値)")
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
        print("ERROR: SPY データ取得失敗")
        return
    sector_data = {etf: price_all.pop(etf) for etf in sec_etfs if etf in price_all}
    price_data  = price_all
    print(f"  → 個別銘柄: {len(price_data)}  セクターETF: {len(sector_data)}")

    print("\n⏳ セクターマップ取得中...")
    sector_map = _build_sector_map()

    # ============================================================
    # 16 通りを一括実行
    # ============================================================
    print("\n⏳ 16通りのシミュレーション実行中...\n")
    print(f"  {'[bits] 設定':<26} {'件数':>5} {'勝率':>6} {'avg':>7} "
          f"{'avg勝':>7} {'avg負':>7} {'総Ret':>8} {'DD':>6} {'Sharpe':>7}")
    print("  " + "-" * 68)

    results = []
    for bits in range(16):
        use_rs  = bool(bits & 8)
        use_pb  = bool(bits & 4)
        use_sec = bool(bits & 2)
        short_lb = bool(bits & 1)

        parts = (["RS"] if use_rs else []) + (["PB"] if use_pb else []) + \
                (["SEC"] if use_sec else []) + (["LB"] if short_lb else [])
        label = "+".join(parts) if parts else "Base"

        trades, eq = run_sim(
            price_data, spy_df, sector_data, sector_map,
            use_rs, use_pb, use_sec, short_lb,
        )
        s = summarize(trades, eq)
        s["label"] = label
        results.append(s)

        print(f"  [{bits:04b}] {label:<22} {s['n']:>5d}  {s['wr']:>5.1f}%  "
              f"{s['avg_pl']:>+6.2f}%  {s['avg_win']:>+6.2f}%  "
              f"{s['avg_loss']:>+6.2f}%  {s['total_ret']:>+7.1f}%  "
              f"{s['max_dd']:>5.1f}%  {s['sharpe']:>6.3f}")

    # ============================================================
    # 集計: avg_pl 降順
    # ============================================================
    print("\n" + "=" * 72)
    print("  結果サマリー（avg_pl 降順）")
    print(f"  {'設定':<22} {'件数':>5} {'勝率':>6} {'avg':>7} "
          f"{'avg勝':>7} {'avg負':>7} {'総Ret':>8} {'DD':>6} {'Sharpe':>7}")
    print("  " + "-" * 68)
    for s in sorted(results, key=lambda x: x["avg_pl"], reverse=True):
        print(f"  {s['label']:<22} {s['n']:>5d}  {s['wr']:>5.1f}%  "
              f"{s['avg_pl']:>+6.2f}%  {s['avg_win']:>+6.2f}%  "
              f"{s['avg_loss']:>+6.2f}%  {s['total_ret']:>+7.1f}%  "
              f"{s['max_dd']:>5.1f}%  {s['sharpe']:>6.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
