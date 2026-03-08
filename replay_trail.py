"""
Trail Multiplier リプレイスクリプト

既存の signal CSV からエントリー情報を読み取り、
新しい TRAILING_STOP_ATR_MULTIPLIER で出口だけ再計算して
新しい signal CSV を生成する。

使い方:
  python replay_trail.py                     # BUY trail を engine.py の値で再計算
  python replay_trail.py --trail-buy 4.0     # BUY trail を 4.0 で再計算
"""
import argparse
import csv
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backtest.engine import (
    TRAILING_STOP_ATR_MULTIPLIER,
    SB_TRAILING_ATR_MULT,
    PROGRESSIVE_TRAIL_LEVELS,
    BREAKEVEN_TRIGGER_ATR,
    BREAKEVEN_TRIGGER_DAYS,
    BREAKEVEN_LOCK_ATR,
    PEAK_TRAIL_ATR_MULT,
    PEAK_TRAIL_MIN_PCT,
    MAX_HOLD_DAYS,
    SB_MAX_HOLD_DAYS,
    STAGED_TIMEOUT,
)

BASE = Path(_ROOT)
SRC_CSV = BASE / "output/backtest/signal_backtest_20260305_223046_sb_trail4.0.csv"
OUT_DIR = BASE / "output/backtest"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trail-buy", type=float, default=TRAILING_STOP_ATR_MULTIPLIER,
                   help=f"BUY trail 倍率 (default: engine.py の値 {TRAILING_STOP_ATR_MULTIPLIER})")
    p.add_argument("--trail-sb", type=float, default=SB_TRAILING_ATR_MULT,
                   help=f"STRONG_BUY trail 倍率 (default: {SB_TRAILING_ATR_MULT})")
    p.add_argument("--src", type=str, default=str(SRC_CSV),
                   help="入力 signal CSV パス")
    return p.parse_args()


def _get_trail_level(highest_high, entry_price, atr, trail_buy, trail_sb, signal, holding_days):
    """engine.py の _get_trail_level と同一ロジック（trail 倍率を引数で指定）。"""
    is_sb = (signal == "STRONG_BUY")
    base_mult = trail_sb if is_sb else trail_buy

    if atr <= 0:
        return highest_high - atr * base_mult

    peak_profit_atr = (highest_high - entry_price) / atr

    trail_mult = base_mult
    if not is_sb:  # BUY: progressive tightening
        for min_atr, mult in PROGRESSIVE_TRAIL_LEVELS:
            if peak_profit_atr >= min_atr:
                trail_mult = mult
                break

    trail_atr = highest_high - atr * trail_mult

    if highest_high > 0:
        pct_room = max(PEAK_TRAIL_MIN_PCT, PEAK_TRAIL_ATR_MULT * atr / highest_high)
        trail_peak = highest_high * (1.0 - pct_room)
    else:
        trail_peak = trail_atr

    trail = max(trail_atr, trail_peak)

    if peak_profit_atr >= BREAKEVEN_TRIGGER_ATR and holding_days >= BREAKEVEN_TRIGGER_DAYS:
        trail = max(trail, entry_price + atr * BREAKEVEN_LOCK_ATR)

    return trail


def simulate_trade(ticker, signal, entry_date, entry_price, sl_price, atr,
                   future_df, trail_buy, trail_sb):
    """1トレードの出口を再シミュレート。engine._simulate_trade と同一ロジック。"""
    is_sb = (signal == "STRONG_BUY")
    max_hold = SB_MAX_HOLD_DAYS if is_sb else MAX_HOLD_DAYS
    highest_high = entry_price

    for idx in range(len(future_df)):
        row = future_df.iloc[idx]
        date_str = future_df.index[idx].strftime("%Y-%m-%d")
        high  = float(row["High"])
        low   = float(row["Low"])
        close = float(row["Close"])
        open_ = float(row["Open"]) if "Open" in row else close
        holding_days = idx + 1

        if high > highest_high:
            highest_high = high

        # 決算チェックは省略（rerplay用のため）

        trail = _get_trail_level(highest_high, entry_price, atr,
                                 trail_buy, trail_sb, signal, holding_days)

        # 1) SL
        if low <= sl_price:
            ep = min(sl_price, open_)
            pl = (ep - entry_price) / entry_price * 100
            return date_str, ep, "sl_hit", pl

        # 2) Trail（エントリー価格を超えた場合のみ）
        if trail > entry_price and close <= trail:
            pl = (trail - entry_price) / entry_price * 100
            return date_str, trail, "trailing_stop", pl

        # 3) 段階的タイムアウト (BUYのみ)
        if not is_sb:
            pl_now = (close - entry_price) / entry_price
            for threshold_days, min_profit in STAGED_TIMEOUT:
                if holding_days >= threshold_days and pl_now < min_profit:
                    pl = pl_now * 100
                    return date_str, close, f"timeout_{threshold_days}d", pl

        # 4) 絶対タイムアウト
        if holding_days >= max_hold:
            pl = (close - entry_price) / entry_price * 100
            return date_str, close, f"timeout_{max_hold}d", pl

    # 期間終了（最終行で強制決済）
    last_row = future_df.iloc[-1]
    last_date = future_df.index[-1].strftime("%Y-%m-%d")
    last_close = float(last_row["Close"])
    pl = (last_close - entry_price) / entry_price * 100
    return last_date, last_close, "timeout", pl


def main():
    args = parse_args()
    trail_buy = args.trail_buy
    trail_sb  = args.trail_sb
    src_path  = Path(args.src)

    print(f"BUY trail: {trail_buy}  /  STRONG_BUY trail: {trail_sb}")
    print(f"入力CSV: {src_path.name}")

    # ── エントリー読み込み ──────────────────────────────────────────
    entries = []
    with open(src_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            entries.append({
                "ticker":      row["ticker"],
                "signal":      row["signal"],
                "score":       row["score"],
                "signal_date": row["signal_date"],
                "entry_date":  row["entry_date"],
                "entry_price": float(row["entry_price"]),
                "sl_price":    float(row["sl_price"]),
                "tp_price":    row["tp_price"],
                # ATR逆算: SL = entry - 2×ATR → ATR = (entry - sl) / 2
                # ※ -6%フロアが効いている場合は近似値
                "atr": (float(row["entry_price"]) - float(row["sl_price"])) / 2.0,
            })

    print(f"エントリー数: {len(entries)}")

    # ── 価格データ取得（DBキャッシュ優先）──────────────────────────
    tickers = sorted(set(e["ticker"] for e in entries))
    print(f"価格データ取得中 ({len(tickers)} 銘柄)...")

    from data.cache import CacheDB
    from data.price_fetcher import fetch_prices
    db = CacheDB()
    db.initialize()
    price_data = fetch_prices(tickers, db=db, period="6y", allow_stale=True)
    print(f"  → {len(price_data)} 銘柄取得完了")

    # ── トレード再シミュレーション ──────────────────────────────────
    print("出口再計算中...")
    results = []
    skipped = 0

    for e in entries:
        df = price_data.get(e["ticker"])
        if df is None or df.empty:
            skipped += 1
            continue

        entry_dt = pd.Timestamp(e["entry_date"])
        future = df[df.index > entry_dt].copy()
        if future.empty:
            skipped += 1
            continue

        exit_date, exit_price, result, pl_pct = simulate_trade(
            ticker=e["ticker"],
            signal=e["signal"],
            entry_date=e["entry_date"],
            entry_price=e["entry_price"],
            sl_price=e["sl_price"],
            atr=e["atr"],
            future_df=future,
            trail_buy=trail_buy,
            trail_sb=trail_sb,
        )

        results.append({
            "ticker":      e["ticker"],
            "signal":      e["signal"],
            "score":       e["score"],
            "signal_date": e["signal_date"],
            "entry_date":  e["entry_date"],
            "entry_price": e["entry_price"],
            "exit_date":   exit_date,
            "exit_price":  round(exit_price, 4),
            "result":      result,
            "pl_pct":      round(pl_pct, 4),
            "sl_price":    e["sl_price"],
            "tp_price":    e["tp_price"],
        })

    print(f"  完了: {len(results)} 件  スキップ: {skipped} 件")

    # ── 簡易サマリー ──────────────────────────────────────────────
    wins   = sum(1 for r in results if r["pl_pct"] > 0)
    losses = sum(1 for r in results if r["pl_pct"] < 0)
    total_pl = sum(r["pl_pct"] for r in results)
    avg_pl   = total_pl / len(results) if results else 0
    avg_win  = sum(r["pl_pct"] for r in results if r["pl_pct"] > 0) / wins if wins else 0
    avg_loss = sum(r["pl_pct"] for r in results if r["pl_pct"] < 0) / losses if losses else 0
    result_counts = {}
    for r in results:
        result_counts[r["result"]] = result_counts.get(r["result"], 0) + 1

    print(f"\n=== 再シミュレーション結果 ===")
    print(f"  総件数  : {len(results)}")
    print(f"  勝ち    : {wins} ({wins/len(results)*100:.1f}%)")
    print(f"  負け    : {losses} ({losses/len(results)*100:.1f}%)")
    print(f"  平均P/L : {avg_pl:+.2f}%")
    print(f"  勝ち平均: {avg_win:+.2f}%")
    print(f"  負け平均: {avg_loss:+.2f}%")
    print(f"  Exit理由: {result_counts}")

    # ── CSV出力 ──────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"signal_backtest_{ts}_buy_trail{trail_buy}_sb_trail{trail_sb}.csv"
    out_path = OUT_DIR / out_name
    fieldnames = ["ticker","signal","score","signal_date","entry_date","entry_price",
                  "exit_date","exit_price","result","pl_pct","sl_price","tp_price"]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nCSV出力: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    main()
