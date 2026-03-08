"""
Strategy A — 52週高値ブレイクアウト ポジショントレード

コンセプト:
  ベースライン戦略の「ディップ買い × 小利益 × 多数件数」とは逆の発想。
  52週高値を更新した「強さが証明された銘柄」に集中投資し、
  トレンドが続く限り長期保有することで1件あたりのP/Lを最大化する。

根拠:
  - George & Hwang (2004): 52週高値に近い銘柄はその後12ヶ月アウトパフォーム
  - 高値更新 = 機関の上値抵抗が解消された状態 → 追随買いが入りやすい
  - 出来高急増でエントリー → 機関の本格参入を確認してから乗る

パラメータ（全てここで管理）:
  - BREAKOUT_LOOKBACK : 52週 = 252営業日
  - STOP_LOOKBACK     : 26週 = 126営業日安値をストップ基準
  - HARD_STOP_PCT     : -12% ハードストップ（構造的ストップより広い場合の上限）
  - TRAIL_ATR_MULT    : 3.0  最高値 - ATR×3.0 トレーリング
  - MAX_POSITIONS     : 5    集中投資
  - RISK_PER_TRADE    : 1.5% 現行の3倍リスク
  - MAX_ALLOC_PCT     : 20%

実験したい値:
  HARD_STOP_PCT: 0.10 / 0.12 / 0.15
  MAX_POSITIONS: 3 / 5
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

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
# パラメータ（ここを変えて実験）
# ============================================================

INITIAL_CAPITAL   = 4_000.0

BREAKOUT_LOOKBACK = 252      # 52週高値（営業日）
STOP_LOOKBACK     = 126      # 26週安値（営業日）
HARD_STOP_PCT     = 0.12     # ハードストップ -12%（構造的ストップが近すぎる場合の上限）
TRAIL_ATR_MULT    = 3.0      # トレーリング幅: 最高値 - ATR × 3.0
ATR_PERIOD        = 14

MIN_VOLUME_MULT   = 1.5      # エントリー出来高: 20日平均 × N倍以上
VOLUME_LOOKBACK   = 20

MAX_POSITIONS     = 5        # 最大同時保有（集中投資）
RISK_PER_TRADE    = 0.015    # 総資産の1.5% をリスクにさらす
MAX_ALLOC_PCT     = 0.20     # 1銘柄最大20%配分

MARKET_FILTER_MA  = 200      # SPY 200日MA フィルター
WARMUP_DAYS       = 260      # 52週分のウォームアップ


# ============================================================
# データクラス
# ============================================================

@dataclass
class PositionA:
    ticker:        str
    entry_date:    str
    entry_price:   float
    stop_price:    float    # 現在のストップ（上方向にのみ更新）
    highest_high:  float
    allocated:     float
    holding_days:  int   = 0
    atr:           float = 0.0


@dataclass
class TradeA:
    ticker:       str
    entry_date:   str
    exit_date:    str
    entry_price:  float
    exit_price:   float
    result:       str    # "trail_stop" | "sl_hit"
    pl_pct:       float
    holding_days: int
    allocated:    float


# ============================================================
# ATR計算
# ============================================================

def calc_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float | None:
    if df is None or len(df) < period + 1:
        return None
    h  = df["High"].astype(float)
    l  = df["Low"].astype(float)
    c  = df["Close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (pc - l).abs()], axis=1).max(axis=1)
    val = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().iloc[-1]
    return float(val) if not np.isnan(val) else None


# ============================================================
# バックテストシミュレーション
# ============================================================

def run_backtest_a(
    price_data: dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
) -> tuple[list[TradeA], list[dict]]:
    """全銘柄の価格データを受け取りポジショントレードをシミュレートする。"""

    # SPY 200MA フィルターを日付→bool で事前計算
    spy_close = spy_df["Close"].astype(float)
    spy_ma    = spy_close.rolling(MARKET_FILTER_MA).mean()
    spy_ok_map: dict[str, bool] = {}
    for d in spy_df.index:
        ma_val = spy_ma[d]
        spy_ok_map[d.strftime("%Y-%m-%d")] = bool(spy_close[d] >= ma_val) if not np.isnan(ma_val) else True

    all_dates = spy_df.index.tolist()

    cash      = INITIAL_CAPITAL
    positions: dict[str, PositionA] = {}
    trades:    list[TradeA]          = []
    equity_curve: list[dict]         = []

    for idx, date in enumerate(all_dates):
        date_str = date.strftime("%Y-%m-%d")

        if idx < WARMUP_DAYS:
            equity_curve.append({"date": date_str, "equity": INITIAL_CAPITAL})
            continue

        spy_ok = spy_ok_map.get(date_str, True)

        # ── 既存ポジションの更新・エグジット判定 ──
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            df  = price_data.get(ticker)
            if df is None or date not in df.index:
                continue

            row         = df.loc[date]
            today_high  = float(row["High"])
            today_low   = float(row["Low"])
            today_open  = float(row["Open"])

            # 最高値更新
            if today_high > pos.highest_high:
                pos.highest_high = today_high

            # ATR更新
            atr = calc_atr(df.loc[:date])
            if atr:
                pos.atr = atr

            # トレーリングストップ更新（上方向のみ）
            if pos.atr > 0:
                new_trail = pos.highest_high - pos.atr * TRAIL_ATR_MULT
                if new_trail > pos.stop_price:
                    pos.stop_price = new_trail

            pos.holding_days += 1

            # ストップ判定（当日安値がストップ以下）
            if today_low <= pos.stop_price:
                exit_price = float(np.clip(pos.stop_price, today_low, today_high))
                # ギャップダウン時は始値で約定
                if today_open < pos.stop_price:
                    exit_price = today_open
                pl_pct = (exit_price - pos.entry_price) / pos.entry_price
                pnl    = pos.allocated * pl_pct
                cash  += pos.allocated + pnl
                trades.append(TradeA(
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

        # ── 新規エントリー候補スキャン ──
        if spy_ok and len(positions) < MAX_POSITIONS:
            held       = set(positions.keys())
            candidates = []

            for ticker, df in price_data.items():
                if ticker in held or ticker == "SPY":
                    continue
                if date not in df.index:
                    continue
                loc = df.index.get_loc(date)
                if loc < BREAKOUT_LOOKBACK:
                    continue

                close  = df["Close"].astype(float)
                volume = df["Volume"].astype(float) if "Volume" in df.columns else None

                cur_close = float(close.iloc[loc])

                # 52週高値ブレイクアウト（当日を除く過去252日の高値を超えているか）
                prev_high = float(close.iloc[loc - BREAKOUT_LOOKBACK: loc].max())
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

                # ランキングスコア: ブレイクアウト幅 × 出来高倍率
                bo_strength = (cur_close - prev_high) / prev_high
                score       = bo_strength * vol_ratio
                candidates.append((score, ticker, df, loc))

            # スコア降順にソートして空きスロットを埋める
            candidates.sort(reverse=True, key=lambda x: x[0])

            for score, ticker, df, loc in candidates:
                if len(positions) >= MAX_POSITIONS:
                    break
                if loc + 1 >= len(df):
                    continue

                next_date_ts = df.index[loc + 1]
                next_row     = df.loc[next_date_ts]
                entry_price  = float(next_row["Open"])
                if entry_price <= 0:
                    continue

                # 初期ストップ: max(26週安値, エントリー×(1-12%))
                lows        = df["Low"].astype(float)
                low_start   = max(0, loc - STOP_LOOKBACK)
                low_26w     = float(lows.iloc[low_start: loc + 1].min())
                hard_stop   = entry_price * (1 - HARD_STOP_PCT)
                initial_stop = max(low_26w, hard_stop)

                if initial_stop >= entry_price:
                    initial_stop = hard_stop  # フォールバック

                sl_pct = (entry_price - initial_stop) / entry_price
                if sl_pct < 0.005:
                    continue

                # ポジションサイジング
                total_equity = cash + sum(p.allocated for p in positions.values())
                alloc = min(
                    total_equity * RISK_PER_TRADE / sl_pct,
                    total_equity * MAX_ALLOC_PCT,
                    cash,
                )
                if alloc < 50:
                    continue

                atr = calc_atr(df.iloc[: loc + 1]) or (entry_price * 0.02)

                positions[ticker] = PositionA(
                    ticker       = ticker,
                    entry_date   = next_date_ts.strftime("%Y-%m-%d"),
                    entry_price  = entry_price,
                    stop_price   = initial_stop,
                    highest_high = entry_price,
                    allocated    = round(alloc, 2),
                    atr          = atr,
                )
                cash -= alloc

        # 評価額を記録
        stock_val = 0.0
        for t, p in positions.items():
            df = price_data.get(t)
            if df is not None and date in df.index:
                stock_val += p.allocated * float(df.loc[date, "Close"]) / p.entry_price
            else:
                stock_val += p.allocated

        equity_curve.append({"date": date_str, "equity": round(cash + stock_val, 2)})

    return trades, equity_curve


# ============================================================
# 結果集計・表示
# ============================================================

def print_results(trades: list[TradeA], equity_curve: list[dict]) -> None:
    if not trades:
        print("トレードなし")
        return

    pl    = [t.pl_pct for t in trades]
    wins  = [p for p in pl if p > 0]
    losses= [p for p in pl if p <= 0]

    eq_df = pd.DataFrame(equity_curve)
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    final = float(eq_df["equity"].iloc[-1])

    arr  = eq_df["equity"].values
    peak = np.maximum.accumulate(arr)
    dd   = (arr - peak) / peak * 100
    max_dd = float(dd.min())

    daily_ret = pd.Series(arr).pct_change().dropna()
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0

    years = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days / 365.25
    cagr  = (final / INITIAL_CAPITAL) ** (1 / years) - 1 if years > 0 else 0.0
    total_ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    print()
    print("━" * 58)
    print(f"  トレード数     : {len(trades)}")
    print(f"  勝率           : {len(wins)/len(trades)*100:.1f}%")
    print(f"  平均P/L        : {np.mean(pl):+.2f}%")
    print(f"  平均勝ち       : {np.mean(wins):+.2f}%  ({len(wins)}件)")
    print(f"  平均負け       : {np.mean(losses):+.2f}%  ({len(losses)}件)")
    print(f"  最大勝ち       : {max(pl):+.2f}%")
    print(f"  最大負け       : {min(pl):+.2f}%")
    print(f"  平均保有日数   : {np.mean([t.holding_days for t in trades]):.1f}日")
    print(f"  総リターン     : {total_ret:+.1f}%  (${final:,.0f})")
    print(f"  CAGR           : {cagr*100:+.1f}%")
    print(f"  最大DD         : {max_dd:.2f}%")
    print(f"  Sharpe         : {sharpe:.3f}")
    print()

    tdf = pd.DataFrame([{"year": t.exit_date[:4], "pl_pct": t.pl_pct} for t in trades])
    print("  ── 年別 ──")
    for yr, grp in tdf.groupby("year"):
        w = (grp["pl_pct"] > 0).sum()
        print(f"  {yr}: {len(grp):3d}件  勝率{w/len(grp)*100:.0f}%  平均{grp['pl_pct'].mean():+.2f}%")

    print()
    print("  ── 決済内訳 ──")
    for res, cnt in pd.Series([t.result for t in trades]).value_counts().items():
        print(f"  {res:<18}: {cnt}件")
    print("━" * 58)


# ============================================================
# メイン
# ============================================================

def main() -> None:
    print("=" * 58)
    print("  Strategy A — 52週高値ブレイクアウト ポジショントレード")
    print(f"  HARD_STOP={HARD_STOP_PCT*100:.0f}%  TRAIL×{TRAIL_ATR_MULT}  "
          f"MAX_POS={MAX_POSITIONS}  RISK={RISK_PER_TRADE*100:.1f}%")
    print("=" * 58)

    db = CacheDB()
    db.initialize()

    print("\n⏳ 銘柄スクリーニング中...")
    iwv     = fetch_iwv_holdings(db)
    tickers = screen_tier1(iwv, db)
    print(f"  → {len(tickers)} 銘柄")

    print("\n⏳ 株価データ取得中（キャッシュ優先）...")
    all_tickers = sorted(set(tickers) | {"SPY"})
    price_data  = fetch_prices(all_tickers, db=db, period="6y")
    spy_df      = price_data.pop("SPY", None)
    if spy_df is None or spy_df.empty:
        print("ERROR: SPY データ取得失敗")
        return
    print(f"  → {len(price_data)} 銘柄取得")

    print("\n⏳ バックテスト実行中...")
    trades, equity_curve = run_backtest_a(price_data, spy_df)

    print_results(trades, equity_curve)

    # CSV保存
    out_dir  = _ROOT / "output" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"strategy_a_{ts}.csv"
    pd.DataFrame([{
        "ticker":       t.ticker,
        "entry_date":   t.entry_date,
        "exit_date":    t.exit_date,
        "entry_price":  t.entry_price,
        "exit_price":   t.exit_price,
        "pl_pct":       t.pl_pct,
        "holding_days": t.holding_days,
        "allocated":    t.allocated,
        "result":       t.result,
    } for t in trades]).to_csv(csv_path, index=False)
    print(f"\n📁 CSV: {csv_path}")


if __name__ == "__main__":
    main()
