"""
Live Trading — 日次実行スクリプト

米国市場の引け後（JST 火〜土曜 07:00）に実行する。
Windows Task Scheduler から自動起動するか、手動で実行する。

使い方:
  python live/daily.py --init --capital 4000   初期化（初回のみ）
  python live/daily.py --dry-run               動作確認（LINE送信なし・保存なし）
  python live/daily.py                         本番実行
  python live/daily.py --no-notify             LINE送信なし
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime

import pandas as pd

# プロジェクトルートを sys.path に追加
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Windows コンソールの UTF-8 対応
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from backtest.engine import (
    MARKET_FILTER_MA_PERIOD,
    MAX_CONCURRENT_POSITIONS,
)
from config.universe import get_sector
from data.cache import CacheDB
from data.fundamental_fetcher import fetch_fundamentals
from data.news_fetcher import fetch_news
from data.price_fetcher import fetch_price_single, fetch_prices
from data.screener import fetch_iwv_holdings, screen_tier1
from live.notifier import send_daily_notification
from live.tracker import (
    DEFAULT_POSITIONS_PATH,
    LivePosition,
    check_exit,
    init_state,
    load_state,
    recalculate_holding_days,
    save_state,
    update_highest_high,
)
from strategy.risk import calc_risk_levels
from strategy.scorer import STRONG_BUY, BUY, score_universe

logger = logging.getLogger(__name__)

# ── ETFオーバーレイ設定 ─────────────────────────────────────────────
ETF_UNIVERSE = [
    "SPY", "QQQ", "XLC", "XLY", "XLP",
    "XLE", "XLF", "XLV", "XLI", "XLB",
    "XLK", "XLRE", "XLU",
]
ETF_TOP_N    = 2
ETF_MA       = 50
SAFE_TICKER  = "SGOV"
_ETF_MOM_PERIODS = {
    "1M":  (21,  0.30),
    "3M":  (63,  0.30),
    "6M":  (126, 0.30),
    "12M": (252, 0.10),
}


# ============================================================
# ETFオーバーレイ ヘルパー
# ============================================================

def _etf_calc_scores(price_data: dict[str, pd.DataFrame]) -> dict[str, float]:
    """ETFモメンタムスコアを計算する（バックテストと同一ロジック、小さいほど順位が良い）。"""
    rets: dict[str, dict[str, float]] = {}
    for ticker in ETF_UNIVERSE:
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue
        close = df["Close"].astype(float)
        current = float(close.iloc[-1])
        tr: dict[str, float] = {}
        ok = True
        for name, (days, _) in _ETF_MOM_PERIODS.items():
            if len(close) < days + 1:
                ok = False; break
            base = float(close.iloc[-(days + 1)])
            if base <= 0:
                ok = False; break
            tr[name] = (current - base) / base
        if ok:
            rets[ticker] = tr
    if len(rets) < 2:
        return {}
    scores: dict[str, float] = {t: 0.0 for t in rets}
    for name, (_, weight) in _ETF_MOM_PERIODS.items():
        ranked = sorted(rets.keys(), key=lambda t: rets[t][name], reverse=True)
        for rank, ticker in enumerate(ranked, 1):
            scores[ticker] += rank * weight
    return scores  # 小さいほど順位が良い


def _etf_select_targets(
    price_data: dict[str, pd.DataFrame],
    scores: dict[str, float],
) -> list[str]:
    """50MA以上のETFからスコア上位ETF_TOP_N本を返す。"""
    result = []
    for ticker in sorted(scores, key=lambda t: scores[t]):
        if len(result) >= ETF_TOP_N:
            break
        df = price_data.get(ticker)
        if df is None or len(df) < ETF_MA:
            continue
        close = float(df["Close"].iloc[-1])
        ma = float(df["Close"].iloc[-ETF_MA:].mean())
        if close >= ma:
            result.append(ticker)
    return result


def _process_etf(
    state,
    price_data: dict[str, pd.DataFrame],
    today: str,
    spy_bull: bool,
    dry_run: bool,
) -> tuple[list[dict], dict[str, float]]:
    """ETFオーバーレイ処理（リバランス + 50MA監視 + レジーム管理）。

    Returns:
        (orders, scores)
        orders: 今日実行すべきETF売買指示リスト
        scores: モメンタムスコア（ENTRY時のETF売却優先順位に使用、小さいほど良い）
    """
    orders: list[dict] = []
    today_dt = pd.Timestamp(today)
    is_monday = today_dt.dayofweek == 0
    scores = _etf_calc_scores(price_data) if spy_bull else {}

    def get_close(ticker: str) -> float | None:
        df = price_data.get(ticker)
        return float(df["Close"].iloc[-1]) if df is not None and not df.empty else None

    def do_sell(ticker: str, reason: str) -> float:
        shares = state.etf_positions.get(ticker, 0.0)
        if shares < 1e-9:
            return 0.0
        p = get_close(ticker) or 0.0
        value = shares * p
        if not dry_run:
            state.etf_positions.pop(ticker, None)
            state.cash += value
        orders.append({"action": "SELL", "ticker": ticker,
                        "shares": shares, "price": p, "value": value, "reason": reason})
        return value

    def do_buy(ticker: str, amount: float, reason: str) -> None:
        if amount < 1.0:
            return
        p = get_close(ticker)
        if not p:
            return
        shares = amount / p
        if not dry_run:
            state.etf_positions[ticker] = state.etf_positions.get(ticker, 0.0) + shares
            state.cash -= amount
        orders.append({"action": "BUY", "ticker": ticker,
                        "shares": shares, "price": p, "value": amount, "reason": reason})

    # ── ベア相場: 全セクターETF → SGOV ────────────────────────────
    if not spy_bull:
        sold_total = 0.0
        for tk in list(state.etf_positions.keys()):
            if tk != SAFE_TICKER:
                sold_total += do_sell(tk, "ベア転換")
        avail = state.cash if not dry_run else state.cash + sold_total
        if avail > 1.0 and SAFE_TICKER not in state.etf_positions:
            do_buy(SAFE_TICKER, avail, "ベア退避")

    # ── ブル相場 ─────────────────────────────────────────────────
    else:
        # SGOV → キャッシュに戻す（ブル転換）
        if SAFE_TICKER in state.etf_positions:
            do_sell(SAFE_TICKER, "ブル転換")

        # 日次 50MA 監視（50MA割れは即売却）
        for tk in list(state.etf_positions.keys()):
            df = price_data.get(tk)
            if df is None or len(df) < ETF_MA:
                continue
            if float(df["Close"].iloc[-1]) < float(df["Close"].iloc[-ETF_MA:].mean()):
                do_sell(tk, "50MA割れ")

        # 月曜リバランス
        if is_monday and scores:
            targets = _etf_select_targets(price_data, scores)
            sold = 0.0
            for tk in list(state.etf_positions.keys()):
                if tk not in targets:
                    sold += do_sell(tk, "リバランス")
            avail = state.cash if not dry_run else state.cash + sold
            if targets and avail > 1.0:
                alloc_each = avail / len(targets)
                for tk in targets:
                    do_buy(tk, alloc_each, "リバランス")

    return orders, scores


# ============================================================
# CLI
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live Trading — 日次シグナル & ポジション管理",
    )
    p.add_argument("--init", action="store_true", help="positions.json を新規初期化")
    p.add_argument("--capital", type=float, default=4000.0, help="初期資金（--init と一緒に使用）")
    p.add_argument("--dry-run", action="store_true", help="保存・LINE送信なし（テスト用）")
    p.add_argument("--no-notify", action="store_true", help="LINE 通知を送信しない")
    p.add_argument("-v", "--verbose", action="store_true", help="デバッグログを表示")
    return p


# ============================================================
# メイン処理
# ============================================================

def main() -> None:
    args = _build_parser().parse_args()
    _setup_logging(args.verbose)

    if args.init:
        _do_init(args)
        return

    _do_daily(args)


def _do_init(args: argparse.Namespace) -> None:
    """初期化: positions.json を新規作成する。"""
    path = DEFAULT_POSITIONS_PATH
    if os.path.exists(path) and not _confirm(
        f"既存の {path} を上書きします。よろしいですか？ [y/N]: "
    ):
        print("キャンセルしました。")
        return

    state = init_state(args.capital, path)
    print(f"\n✅ 初期化完了: {path}")
    print(f"   初期資金: ${state.initial_capital:,.2f}")
    print(f"\n次のステップ:")
    print(f"   python live/daily.py --dry-run   動作確認")
    print(f"   python live/daily.py             本番実行")
    print(f"   python live/scheduler_setup.py  Task Scheduler 登録")


def _do_daily(args: argparse.Namespace) -> None:
    """日次メイン処理。"""
    today = date.today().strftime("%Y-%m-%d")
    dry_run = args.dry_run
    no_notify = args.no_notify or dry_run

    print()
    print("=" * 60)
    print(f"  Live Trading Daily Runner  [{today}]")
    if dry_run:
        print("  [DRY RUN — 保存・通知なし]")
    print("=" * 60)
    print()

    # ---- Step 1: 状態読み込み ----
    state = load_state()
    print(f"📂 ポジション読み込み: {len(state.positions)} 件 | キャッシュ: ${state.cash:,.2f}")

    # ---- Step 2: DB 初期化 ----
    db = CacheDB()
    db.initialize()

    # ---- Step 3a: Russell 3000 Tier1 スクリーニング ----
    print(f"\n⏳ [0/4] 対象銘柄スクリーニング中 (Russell 3000 Tier1)...")
    iwv_tickers = fetch_iwv_holdings(db)
    core_tickers = screen_tier1(iwv_tickers, db)
    print(f"  → {len(core_tickers)} 銘柄が対象")

    # ---- Step 3b: 全必要銘柄の株価取得 ----
    position_tickers = [p.ticker for p in state.positions]
    all_tickers = sorted(set(core_tickers) | set(position_tickers) | {"SPY"} | set(ETF_UNIVERSE) | {SAFE_TICKER})

    print(f"\n⏳ [1/4] 株価データ取得中 ({len(all_tickers)} 銘柄)...")
    price_data = fetch_prices(all_tickers, db=db)
    print(f"  → {len(price_data)} 銘柄を取得")

    # ---- Step 4: SPY 200MA フィルター ----
    spy_filter, spy_close, spy_ma200 = _check_spy_filter(price_data.get("SPY"))
    spy_status = "✅ 強気相場" if spy_filter else "🚫 弱気相場 — 新規エントリーなし"
    print(f"\n📈 SPY: ${spy_close:.2f} | MA200: ${spy_ma200:.2f} | {spy_status}")

    # ---- Step 5: 保有ポジション holding_days 再計算 ----
    recalculate_holding_days(state.positions, today)

    # ---- Step 5b: 決算日取得（pre_earnings 強制決済用）----
    _earnings_map: dict = {}
    _pos_tickers = [p.ticker for p in state.positions]
    if _pos_tickers:
        try:
            from data.earnings_fetcher import fetch_earnings_dates
            _earnings_map = fetch_earnings_dates(_pos_tickers)
        except Exception:
            logger.warning("決算日取得失敗 — pre_earnings 判定をスキップ")

    # ---- Step 5c: ETFオーバーレイ処理（リバランス + 50MA + レジーム管理）----
    print(f"\n⏳ [ETF] ETFオーバーレイ処理中...")
    etf_orders, etf_scores = _process_etf(state, price_data, today, spy_filter, dry_run)
    if etf_orders:
        for o in etf_orders:
            icon = "🟢" if o["action"] == "BUY" else "🟠"
            print(
                f"  {icon} ETF {o['action']:<4} {o['ticker']:<5} "
                f"@${o['price']:.2f}  {o['shares']:.4f}株  ${o['value']:.0f}"
                f"  [{o['reason']}]"
            )
    else:
        print("  → ETFオーダーなし")

    # ---- Step 6: 実行開始時の総資産を計算（ポジションサイジング用）----
    total_assets_start = _calc_total_assets(state, price_data, today)
    print(f"\n💼 総資産（評価額）: ${total_assets_start:,.2f}")

    # ---- Step 7: エグジット判定 ----
    print(f"\n⏳ [2/4] エグジット判定中 ({len(state.positions)} ポジション)...")
    exits = _process_exits(state, price_data, today, dry_run, earnings_map=_earnings_map)
    if exits:
        print(f"  → {len(exits)} 件のエグジット")
    else:
        print("  → エグジットなし")

    # ---- Step 8: シグナルスキャン（SPY フィルター通過時のみ）----
    print(f"\n⏳ [3/4] 新規シグナルスキャン中...")
    entries = []
    if spy_filter and len(state.positions) < MAX_CONCURRENT_POSITIONS:
        entries = _process_entries(
            state, price_data, db, today,
            total_assets_start, dry_run,
            tickers=core_tickers,
            etf_scores=etf_scores,
        )
        if entries:
            print(f"  → {len(entries)} 件の新規エントリー")
        else:
            print("  → 新規エントリーなし")
    elif not spy_filter:
        print("  → SPY フィルター: 弱気相場のためスキップ")
    else:
        print(f"  → ポジション満杯 ({MAX_CONCURRENT_POSITIONS}/{MAX_CONCURRENT_POSITIONS})")

    # ---- Step 9: 状態保存 ----
    if not dry_run:
        save_state(state)
        print(f"\n✅ 状態保存完了")

    # ---- Step 10: LINE 通知 ----
    print(f"\n⏳ [4/4] LINE 通知送信中...")
    final_balance = _calc_total_assets(state, price_data, today)
    total_return_pct = (final_balance - state.initial_capital) / state.initial_capital * 100

    if not no_notify:
        send_daily_notification(
            exits=exits,
            entries=entries,
            state=state,
            spy_filter=spy_filter,
            spy_close=spy_close,
            spy_ma200=spy_ma200,
            final_balance=final_balance,
            total_return_pct=total_return_pct,
            today=today,
            etf_orders=etf_orders,
        )
        print("  → LINE 通知送信完了")
    else:
        print("  → 通知スキップ（--no-notify / --dry-run）")

    # ---- コンソールサマリー ----
    _print_summary(state, exits, entries, final_balance, total_return_pct, today, price_data)


# ============================================================
# エグジット処理
# ============================================================

def _process_exits(
    state,
    price_data: dict[str, pd.DataFrame],
    today: str,
    dry_run: bool,
    earnings_map: dict | None = None,
) -> list[dict]:
    """保有中ポジションのエグジット判定を行い、決済処理する。"""
    from bisect import bisect_left as _bisect_left

    exits = []
    remaining = []

    for pos in state.positions:
        df = price_data.get(pos.ticker)
        if df is None or df.empty:
            remaining.append(pos)
            continue

        # 今日のデータ取得（最新行）
        today_row = df.iloc[-1]
        today_open = float(today_row["Open"])
        today_high = float(today_row["High"])
        today_low = float(today_row["Low"])
        today_close = float(today_row["Close"])

        # エントリー日の分割修正済み価格を取得
        adj_entry = _get_adj_entry(df, pos.entry_date, pos.entry_price)

        # highest_high を今日の高値で更新
        update_highest_high(pos, adj_entry, today_high)

        # 決算前日数を計算（pre_earnings 決済用）
        days_to_earnings = None
        if earnings_map:
            ticker_earnings = earnings_map.get(pos.ticker, [])
            if ticker_earnings:
                ei = _bisect_left(ticker_earnings, today)
                if ei < len(ticker_earnings):
                    days_to_earnings = (
                        pd.Timestamp(ticker_earnings[ei]) - pd.Timestamp(today)
                    ).days

        # エグジット判定
        reason, exit_price = check_exit(
            pos, adj_entry, today_high, today_low, today_close,
            today_open=today_open, days_to_earnings=days_to_earnings,
        )

        if reason:
            pl_pct = (exit_price - adj_entry) / adj_entry * 100
            pnl_dollar = pos.allocated * (pl_pct / 100)
            if not dry_run:
                state.cash += pos.allocated + pnl_dollar

            exit_record = {
                "ticker": pos.ticker,
                "signal": pos.signal,
                "entry_date": pos.entry_date,
                "exit_date": today,
                "entry_price": adj_entry,
                "exit_price": exit_price,
                "result": reason,
                "pl_pct": round(pl_pct, 2),
                "allocated": pos.allocated,
                "pnl_dollar": round(pnl_dollar, 2),
                "holding_days": pos.holding_days,
            }
            exits.append(exit_record)
            if not dry_run:
                state.history.append(exit_record)

            label = _exit_label(reason)
            sign = "+" if pnl_dollar >= 0 else ""
            print(
                f"  🔴 EXIT  {pos.ticker:<6} {label:<12} "
                f"@${exit_price:.2f}  {sign}{pl_pct:.1f}%  {sign}${pnl_dollar:.2f}"
            )
        else:
            remaining.append(pos)

    if not dry_run:
        state.positions = remaining
    return exits


# ============================================================
# エントリー処理
# ============================================================

def _process_entries(
    state,
    price_data: dict[str, pd.DataFrame],
    db: CacheDB,
    today: str,
    total_assets: float,
    dry_run: bool,
    tickers: list[str] | None = None,
    etf_scores: dict[str, float] | None = None,
) -> list[dict]:
    """新規シグナルをスキャンして空きスロットにエントリーする。"""
    from data.news_fetcher import fetch_news

    held_tickers = {p.ticker for p in state.positions}

    if not tickers:
        logger.warning("対象銘柄リストが空 — エントリースキャンをスキップ")
        return []

    try:
        fund_data = fetch_fundamentals(tickers, db=db)
    except Exception:
        fund_data = {}
    try:
        news_data = fetch_news(tickers)
    except Exception:
        news_data = {}

    universe = {}
    for ticker in tickers:
        universe[ticker] = {
            "sector": get_sector(ticker),
            "price_df": price_data.get(ticker),
            "fundamentals": fund_data.get(ticker),
            "headlines": news_data.get(ticker, []),
        }
    results = score_universe(universe)

    buy_signals = [
        r for r in results
        if r.signal in (STRONG_BUY, BUY)
        and r.ticker not in held_tickers
        and r.risk is not None
        and price_data.get(r.ticker) is not None
    ]
    buy_signals.sort(key=lambda r: r.final_score, reverse=True)

    entries = []
    available_slots = MAX_CONCURRENT_POSITIONS - len(state.positions)

    for r in buy_signals:
        if available_slots <= 0:
            break

        df = price_data[r.ticker]
        entry_price = float(df.iloc[-1]["Close"])

        _RISK_PER_TRADE = 0.005
        _MAX_ALLOC_PCT = 0.140 if r.signal == STRONG_BUY else 0.130
        sl_pct = (entry_price - r.risk.stop_loss) / entry_price
        if sl_pct <= 0:
            continue

        need = min(
            total_assets * _RISK_PER_TRADE / sl_pct,
            total_assets * _MAX_ALLOC_PCT,
        )

        # キャッシュ不足時: ETFから調達（SGOV優先 → 弱いETF順）
        if state.cash < need and state.etf_positions:
            shortage = need - state.cash
            ranked = sorted(
                state.etf_positions.keys(),
                key=lambda tk: (
                    0 if tk == SAFE_TICKER else 1,
                    (etf_scores or {}).get(tk, 999),
                ),
                reverse=True,
            )
            for tk in ranked:
                if shortage <= 0:
                    break
                etf_df = price_data.get(tk)
                if etf_df is None:
                    continue
                p = float(etf_df["Close"].iloc[-1])
                avail_val = state.etf_positions.get(tk, 0.0) * p
                sell_val = min(shortage, avail_val)
                sell_shares = sell_val / p
                if not dry_run:
                    state.etf_positions[tk] = state.etf_positions.get(tk, 0.0) - sell_shares
                    if state.etf_positions.get(tk, 0.0) < 1e-9:
                        state.etf_positions.pop(tk, None)
                    state.cash += sell_val
                shortage -= sell_val
                print(f"    (ETF売却 {tk}: ${sell_val:.0f})")

        alloc = min(need, state.cash)
        if alloc < 10:
            break

        atr_pct = r.risk.atr / entry_price

        pos = LivePosition(
            ticker=r.ticker,
            signal=r.signal,
            entry_date=today,
            entry_price=entry_price,
            sl_pct=round(sl_pct, 6),
            atr_pct=round(atr_pct, 6),
            highest_high_ratio=1.0,
            allocated=round(alloc, 4),
            holding_days=0,
        )

        entry_record = {
            "ticker": r.ticker,
            "signal": r.signal,
            "entry_date": today,
            "entry_price": entry_price,
            "sl_pct": sl_pct,
            "sl_price": r.risk.stop_loss,
            "atr": r.risk.atr,
            "allocated": round(alloc, 2),
        }
        entries.append(entry_record)

        if not dry_run:
            state.cash -= alloc
            state.positions.append(pos)

        held_tickers.add(r.ticker)
        available_slots -= 1

        sign_label = "🟢 STRONG" if r.signal == STRONG_BUY else "🟢 BUY   "
        print(
            f"  {sign_label} {r.ticker:<6} @${entry_price:.2f}  "
            f"SL:-{sl_pct*100:.1f}%  Alloc:${alloc:.0f}"
        )

    return entries


# ============================================================
# SPY フィルター
# ============================================================

def _check_spy_filter(
    spy_df: pd.DataFrame | None,
) -> tuple[bool, float, float]:
    """SPY が 200日MA を上回っているか確認する。

    Returns:
        (フィルター通過=True, SPY終値, SPY MA200)
    """
    if spy_df is None or len(spy_df) < MARKET_FILTER_MA_PERIOD:
        logger.warning("SPY データ不足 — フィルターなし（BUY許可）")
        return True, 0.0, 0.0

    close = spy_df["Close"].astype(float)
    ma200 = close.rolling(window=MARKET_FILTER_MA_PERIOD).mean()
    spy_close = float(close.iloc[-1])
    spy_ma200_val = float(ma200.iloc[-1])

    if pd.isna(spy_ma200_val):
        return True, spy_close, 0.0

    return spy_close >= spy_ma200_val, spy_close, spy_ma200_val


# ============================================================
# 総資産計算
# ============================================================

def _calc_total_assets(state, price_data: dict[str, pd.DataFrame], today: str) -> float:
    """現在の総資産を計算する（キャッシュ + 株式ポジション評価額 + ETF評価額）。"""
    total = state.cash
    for pos in state.positions:
        df = price_data.get(pos.ticker)
        if df is not None and not df.empty:
            current_price = float(df.iloc[-1]["Close"])
            adj_entry = _get_adj_entry(df, pos.entry_date, pos.entry_price)
            current_value = pos.allocated * (current_price / adj_entry)
            total += current_value
        else:
            total += pos.allocated  # 価格取得失敗時はコストで評価
    for tk, shares in state.etf_positions.items():
        df = price_data.get(tk)
        if df is not None and not df.empty:
            total += shares * float(df["Close"].iloc[-1])
    return total


def _get_adj_entry(df: pd.DataFrame, entry_date: str, original_entry: float) -> float:
    """yfinance の修正済み株価からエントリー日の終値を取得する（株式分割対応）。

    エントリー日のデータがない場合（上場前・祝日等）は元の価格を返す。
    """
    try:
        entry_dt = pd.Timestamp(entry_date)
        # 完全一致を試みる
        if entry_dt in df.index:
            return float(df.loc[entry_dt, "Close"])
        # 直近の前営業日で代替
        before = df[df.index <= entry_dt]
        if not before.empty:
            return float(before.iloc[-1]["Close"])
    except Exception:
        pass
    return original_entry


# ============================================================
# ユーティリティ
# ============================================================

def _print_summary(
    state,
    exits: list[dict],
    entries: list[dict],
    final_balance: float,
    total_return_pct: float,
    today: str,
    price_data: dict | None = None,
) -> None:
    """コンソールにサマリーを表示する。"""
    print()
    print("=" * 60)
    print("  PORTFOLIO SUMMARY")
    print("=" * 60)
    print(f"  Date:        {today}")
    print(f"  Balance:     ${final_balance:>10,.2f}  ({total_return_pct:+.1f}%)")
    print(f"  Cash:        ${state.cash:>10,.2f}")
    print(f"  Open pos:    {len(state.positions)}/{MAX_CONCURRENT_POSITIONS}")
    if state.etf_positions and price_data:
        etf_parts = []
        for tk, shares in state.etf_positions.items():
            df = price_data.get(tk)
            if df is not None and not df.empty:
                etf_parts.append(f"{tk}=${shares * float(df['Close'].iloc[-1]):.0f}")
        if etf_parts:
            print(f"  ETF保有:     {' | '.join(etf_parts)}")
    print(f"  Total exits: {len(state.history)}")
    print()

    if exits:
        wins = sum(1 for e in exits if e["pnl_dollar"] > 0)
        total_pnl = sum(e["pnl_dollar"] for e in exits)
        print(f"  Today exits: {len(exits)} ({wins}W/{len(exits)-wins}L)  P/L: ${total_pnl:+.2f}")
    if entries:
        print(f"  Today entry: {len(entries)} 銘柄  " + ", ".join(e["ticker"] for e in entries))
    print("=" * 60)


def _exit_label(reason: str) -> str:
    labels = {
        "sl_hit": "SL Hit",
        "trailing_stop": "Trail Stop",
        "pre_earnings": "PreEarn",
        "timeout_30d": "TO-30d",
        "timeout_60d": "TO-60d",
        "timeout_90d": "TO-90d",
    }
    return labels.get(reason, reason)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("yfinance").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)


def _confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        return False


if __name__ == "__main__":
    main()
