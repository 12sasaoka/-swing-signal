"""
Strategy B — PEAD（Post-Earnings Announcement Drift）

コンセプト:
  決算翌日に大きく跳ねた銘柄に追随する。
  根拠: Ball & Brown (1968) 以来60年以上実証されたアノマリー。
  機関投資家が情報を数週間かけて消化しながらポジション構築する
  「ドリフト」の動きを狙う。

エントリー条件（3つ全て満たす）:
  1. 決算後の最初の営業日（reaction day）に +EARNINGS_MOVE_MIN 以上の上昇
  2. reaction day の出来高が20日平均の VOLUME_MULT 倍以上（機関参入確認）
  3. エントリーは reaction day 翌日の始値（反応を確認してから乗る）

リスク管理（現行システムの実証済み設計を踏襲）:
  - 初期ストップ : ATR×2.0 or -8% フロア
  - BEラチェット : 含み益≥1ATR で SL を entry+1ATR に引上げ
  - トレーリング : 最高値 - ATR×3.5
  - タイムアウト : 45営業日（PEADの効果は60日以内が中心）

パラメータ（全てここで管理）:
  - EARNINGS_MOVE_MIN : +5%（reaction day の最低リターン）
  - VOLUME_MULT       : 2.0 倍
  - MAX_HOLD_DAYS     : 45 営業日
  - RISK_PER_TRADE    : 1.0%（現行の2倍）

実験したい値:
  EARNINGS_MOVE_MIN: 0.03 / 0.05 / 0.08
  MAX_HOLD_DAYS    : 30 / 45 / 60
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
from data.earnings_fetcher import fetch_earnings_dates
from data.price_fetcher import fetch_prices
from data.screener import fetch_iwv_holdings, screen_tier1

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

# ============================================================
# パラメータ（ここを変えて実験）
# ============================================================

INITIAL_CAPITAL   = 4_000.0

EARNINGS_MOVE_MIN = 0.05    # reaction day の最低リターン (+5%)
VOLUME_MULT       = 2.0     # reaction day の出来高倍率
VOLUME_LOOKBACK   = 20

ATR_PERIOD        = 14
SL_ATR_MULT       = 2.0     # 初期ストップ: entry - ATR×2.0
SL_HARD_PCT       = 0.08    # ハードフロア -8%
BE_TRIGGER_ATR    = 1.0     # BEラチェット発動: 含み益≥1ATR
BE_LOCK_ATR       = 1.0     # BEロック水準: entry + 1ATR
TRAIL_ATR_MULT    = 3.5     # トレーリング幅

MAX_HOLD_DAYS     = 45      # タイムアウト（営業日）
MAX_POSITIONS     = 8
RISK_PER_TRADE    = 0.010   # 1.0% リスク/トレード
MAX_ALLOC_PCT     = 0.15    # 1銘柄最大15%

MARKET_FILTER_MA  = 200
WARMUP_DAYS       = 220


# ============================================================
# データクラス
# ============================================================

@dataclass
class PositionB:
    ticker:       str
    entry_date:   str
    entry_price:  float
    stop_price:   float
    highest_high: float
    allocated:    float
    holding_days: int   = 0
    atr:          float = 0.0
    be_locked:    bool  = False


@dataclass
class TradeB:
    ticker:           str
    entry_date:       str
    exit_date:        str
    entry_price:      float
    exit_price:       float
    result:           str    # "trailing_stop" | "sl_hit" | "timeout"
    pl_pct:           float
    holding_days:     int
    allocated:        float
    earnings_move_pct: float  # reaction day のリターン(%)


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
# PEADシグナルの事前スキャン
# ============================================================

def find_pead_signals(
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
    earnings_map: dict[str, list[str]],
) -> list[dict]:
    """全銘柄・全決算日をスキャンし、PEADエントリー条件を満たす候補を返す。

    Returns:
        entry_date 昇順にソートされた dict のリスト。
        各 dict は {ticker, earnings_date, reaction_day, entry_date,
                    entry_day_ts, entry_price, stop_price, atr, earnings_move_pct}
    """
    signals = []

    for ticker in tickers:
        df          = price_data.get(ticker)
        earn_dates  = earnings_map.get(ticker, [])
        if df is None or df.empty or not earn_dates:
            continue

        close  = df["Close"].astype(float)
        volume = df["Volume"].astype(float) if "Volume" in df.columns else None

        for earn_str in earn_dates:
            earn_dt = pd.Timestamp(earn_str)

            # reaction day = 決算後の最初の営業日
            after = df[df.index > earn_dt]
            if len(after) < 2:
                continue

            reaction_day_ts = after.index[0]
            entry_day_ts    = after.index[1]   # reaction day翌日にエントリー

            react_idx = df.index.get_loc(reaction_day_ts)
            if react_idx < WARMUP_DAYS:
                continue

            # reaction day のリターン（前日終値比）
            prev_close  = float(close.iloc[react_idx - 1])
            react_close = float(close.iloc[react_idx])
            if prev_close <= 0:
                continue
            react_ret = (react_close - prev_close) / prev_close

            if react_ret < EARNINGS_MOVE_MIN:
                continue

            # 出来高フィルター
            if volume is not None and react_idx >= VOLUME_LOOKBACK:
                avg_vol = float(volume.iloc[react_idx - VOLUME_LOOKBACK: react_idx].mean())
                cur_vol = float(volume.iloc[react_idx])
                if avg_vol > 0 and cur_vol < avg_vol * VOLUME_MULT:
                    continue

            # エントリー価格（reaction day 翌日の始値）
            entry_price = float(df.loc[entry_day_ts, "Open"])
            if entry_price <= 0:
                continue

            # ATR・ストップ計算（エントリー日時点のウィンドウで）
            window = df.iloc[: df.index.get_loc(entry_day_ts) + 1]
            atr    = calc_atr(window)
            if atr is None or atr <= 0:
                continue

            sl_atr   = entry_price - atr * SL_ATR_MULT
            sl_hard  = entry_price * (1 - SL_HARD_PCT)
            stop_price = max(sl_atr, sl_hard)

            signals.append({
                "ticker":            ticker,
                "earnings_date":     earn_str,
                "reaction_day":      reaction_day_ts.strftime("%Y-%m-%d"),
                "entry_date":        entry_day_ts.strftime("%Y-%m-%d"),
                "entry_day_ts":      entry_day_ts,
                "entry_price":       entry_price,
                "stop_price":        stop_price,
                "atr":               atr,
                "earnings_move_pct": round(react_ret * 100, 2),
            })

    signals.sort(key=lambda x: x["entry_day_ts"])
    return signals


# ============================================================
# バックテストシミュレーション
# ============================================================

def run_backtest_b(
    price_data: dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    signals: list[dict],
) -> tuple[list[TradeB], list[dict]]:

    # SPY フィルター事前計算
    spy_close = spy_df["Close"].astype(float)
    spy_ma    = spy_close.rolling(MARKET_FILTER_MA).mean()
    spy_ok_map: dict[str, bool] = {
        d.strftime("%Y-%m-%d"): (bool(spy_close[d] >= spy_ma[d]) if not np.isnan(spy_ma[d]) else True)
        for d in spy_df.index
    }

    # エントリー日 → シグナルリスト のマップ
    sig_by_date: dict[str, list[dict]] = {}
    for sig in signals:
        sig_by_date.setdefault(sig["entry_date"], []).append(sig)

    all_dates = spy_df.index.tolist()

    cash      = INITIAL_CAPITAL
    positions: dict[str, PositionB] = {}
    trades:    list[TradeB]          = []
    equity_curve: list[dict]         = []

    for date in all_dates:
        date_str = date.strftime("%Y-%m-%d")
        spy_ok   = spy_ok_map.get(date_str, True)

        # ── 既存ポジション更新・エグジット判定 ──
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            df  = price_data.get(ticker)
            if df is None or date not in df.index:
                continue

            row        = df.loc[date]
            today_high = float(row["High"])
            today_low  = float(row["Low"])
            today_close= float(row["Close"])
            today_open = float(row["Open"])

            # 最高値更新
            if today_high > pos.highest_high:
                pos.highest_high = today_high

            # ATR更新
            atr = calc_atr(df.loc[:date])
            if atr:
                pos.atr = atr

            pos.holding_days += 1

            # BEラチェット（含み益≥1ATRでSLをentry+1ATRに引上げ）
            if not pos.be_locked and pos.atr > 0:
                gain = pos.highest_high - pos.entry_price
                if gain >= pos.atr * BE_TRIGGER_ATR:
                    be_level = pos.entry_price + pos.atr * BE_LOCK_ATR
                    if be_level > pos.stop_price:
                        pos.stop_price = be_level
                        pos.be_locked  = True

            # トレーリングストップ更新（上方向のみ）
            if pos.atr > 0:
                new_trail = pos.highest_high - pos.atr * TRAIL_ATR_MULT
                if new_trail > pos.stop_price:
                    pos.stop_price = new_trail

            # エグジット判定
            exit_reason: str | None = None
            exit_price:  float      = today_close

            if today_low <= pos.stop_price:
                # ストップヒット（ギャップダウン考慮）
                exit_price  = today_open if today_open < pos.stop_price else float(np.clip(pos.stop_price, today_low, today_high))
                gain        = exit_price - pos.entry_price
                exit_reason = "trailing_stop" if gain > 0 else "sl_hit"
            elif pos.holding_days >= MAX_HOLD_DAYS:
                exit_price  = today_close
                exit_reason = "timeout"

            if exit_reason:
                pl_pct = (exit_price - pos.entry_price) / pos.entry_price
                pnl    = pos.allocated * pl_pct
                cash  += pos.allocated + pnl
                trades.append(TradeB(
                    ticker            = ticker,
                    entry_date        = pos.entry_date,
                    exit_date         = date_str,
                    entry_price       = pos.entry_price,
                    exit_price        = round(exit_price, 4),
                    result            = exit_reason,
                    pl_pct            = round(pl_pct * 100, 2),
                    holding_days      = pos.holding_days,
                    allocated         = round(pos.allocated, 2),
                    earnings_move_pct = 0.0,  # シグナルスキャン時に記録していないため0
                ))
                del positions[ticker]

        # ── 新規エントリー ──
        if spy_ok and date_str in sig_by_date and len(positions) < MAX_POSITIONS:
            total_equity = cash + sum(
                p.allocated * (float(price_data[t].loc[date, "Close"]) / p.entry_price)
                if (t in price_data and date in price_data[t].index) else p.allocated
                for t, p in positions.items()
            )

            # 決算サプライズ幅が大きい順に優先
            sorted_sigs = sorted(sig_by_date[date_str],
                                 key=lambda s: s["earnings_move_pct"], reverse=True)

            for sig in sorted_sigs:
                if len(positions) >= MAX_POSITIONS:
                    break
                ticker = sig["ticker"]
                if ticker in positions:
                    continue

                entry_price = sig["entry_price"]
                stop_price  = sig["stop_price"]
                atr         = sig["atr"]

                sl_pct = (entry_price - stop_price) / entry_price
                if sl_pct < 0.005:
                    continue

                alloc = min(
                    total_equity * RISK_PER_TRADE / sl_pct,
                    total_equity * MAX_ALLOC_PCT,
                    cash,
                )
                if alloc < 50:
                    continue

                positions[ticker] = PositionB(
                    ticker       = ticker,
                    entry_date   = date_str,
                    entry_price  = entry_price,
                    stop_price   = stop_price,
                    highest_high = entry_price,
                    allocated    = round(alloc, 2),
                    atr          = atr,
                )
                cash -= alloc

        # 評価額記録
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

def print_results(trades: list[TradeB], equity_curve: list[dict]) -> None:
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
    print("  Strategy B — PEAD（決算サプライズ後ドリフト）")
    print(f"  MOVE_MIN={EARNINGS_MOVE_MIN*100:.0f}%  VOL×{VOLUME_MULT}  "
          f"TIMEOUT={MAX_HOLD_DAYS}d  RISK={RISK_PER_TRADE*100:.1f}%")
    print("=" * 58)

    db = CacheDB()
    db.initialize()

    print("\n⏳ 銘柄スクリーニング中...")
    iwv     = fetch_iwv_holdings(db)
    tickers = screen_tier1(iwv, db)
    print(f"  → {len(tickers)} 銘柄")

    print("\n⏳ 株価データ取得中（キャッシュ優先）...")
    price_data = fetch_prices(sorted(set(tickers) | {"SPY"}), db=db, period="6y")
    spy_df     = price_data.pop("SPY", None)
    if spy_df is None or spy_df.empty:
        print("ERROR: SPY データ取得失敗")
        return
    print(f"  → {len(price_data)} 銘柄取得")

    print("\n⏳ 決算日データ取得中...")
    earnings_map = fetch_earnings_dates(tickers)
    n_earn = sum(1 for v in earnings_map.values() if v)
    print(f"  → {n_earn} 銘柄で決算日データあり")

    print("\n⏳ PEADシグナルスキャン中...")
    signals = find_pead_signals(tickers, price_data, earnings_map)
    print(f"  → {len(signals)} 件のPEADシグナルを検出")

    if not signals:
        print("\n⚠ シグナルが0件です。EARNINGS_MOVE_MIN を下げて再試行してください。")
        return

    print("\n⏳ バックテスト実行中...")
    trades, equity_curve = run_backtest_b(price_data, spy_df, signals)

    print_results(trades, equity_curve)

    # CSV保存
    out_dir  = _ROOT / "output" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"strategy_b_{ts}.csv"
    pd.DataFrame([{
        "ticker":            t.ticker,
        "entry_date":        t.entry_date,
        "exit_date":         t.exit_date,
        "entry_price":       t.entry_price,
        "exit_price":        t.exit_price,
        "pl_pct":            t.pl_pct,
        "holding_days":      t.holding_days,
        "allocated":         t.allocated,
        "result":            t.result,
    } for t in trades]).to_csv(csv_path, index=False)
    print(f"\n📁 CSV: {csv_path}")


if __name__ == "__main__":
    main()
