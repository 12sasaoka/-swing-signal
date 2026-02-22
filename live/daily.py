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
from config.universe import get_all_tickers, get_sector
from data.cache import CacheDB
from data.fundamental_fetcher import fetch_fundamentals
from data.news_fetcher import fetch_news
from data.price_fetcher import fetch_price_single, fetch_prices
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

    # ---- Step 3: 全必要銘柄の株価取得 ----
    core_tickers = get_all_tickers()
    position_tickers = [p.ticker for p in state.positions]
    all_tickers = sorted(set(core_tickers) | set(position_tickers) | {"SPY"})

    print(f"\n⏳ [1/4] 株価データ取得中 ({len(all_tickers)} 銘柄)...")
    price_data = fetch_prices(all_tickers, db=db)
    print(f"  → {len(price_data)} 銘柄を取得")

    # ---- Step 4: SPY 200MA フィルター ----
    spy_filter, spy_close, spy_ma200 = _check_spy_filter(price_data.get("SPY"))
    spy_status = "✅ 強気相場" if spy_filter else "🚫 弱気相場 — 新規エントリーなし"
    print(f"\n📈 SPY: ${spy_close:.2f} | MA200: ${spy_ma200:.2f} | {spy_status}")

    # ---- Step 5: 保有ポジション holding_days 再計算 ----
    recalculate_holding_days(state.positions, today)

    # ---- Step 6: 実行開始時の総資産を計算（ポジションサイジング用）----
    total_assets_start = _calc_total_assets(state, price_data, today)
    print(f"\n💼 総資産（評価額）: ${total_assets_start:,.2f}")

    # ---- Step 7: エグジット判定 ----
    print(f"\n⏳ [2/4] エグジット判定中 ({len(state.positions)} ポジション)...")
    exits = _process_exits(state, price_data, today, dry_run)
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
        )
        print("  → LINE 通知送信完了")
    else:
        print("  → 通知スキップ（--no-notify / --dry-run）")

    # ---- コンソールサマリー ----
    _print_summary(state, exits, entries, final_balance, total_return_pct, today)


# ============================================================
# エグジット処理
# ============================================================

def _process_exits(
    state,
    price_data: dict[str, pd.DataFrame],
    today: str,
    dry_run: bool,
) -> list[dict]:
    """保有中ポジションのエグジット判定を行い、決済処理する。"""
    exits = []
    remaining = []

    for pos in state.positions:
        df = price_data.get(pos.ticker)
        if df is None or df.empty:
            remaining.append(pos)
            continue

        # 今日のデータ取得（最新行）
        today_row = df.iloc[-1]
        today_high = float(today_row["High"])
        today_low = float(today_row["Low"])
        today_close = float(today_row["Close"])

        # エントリー日の分割修正済み価格を取得
        adj_entry = _get_adj_entry(df, pos.entry_date, pos.entry_price)

        # highest_high を今日の高値で更新
        update_highest_high(pos, adj_entry, today_high)

        # エグジット判定
        reason, exit_price = check_exit(
            pos, adj_entry, today_high, today_low, today_close
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
) -> list[dict]:
    """新規シグナルをスキャンして空きスロットにエントリーする。"""
    from data.news_fetcher import fetch_news

    # 既存ポジションの銘柄セット（重複エントリー防止）
    held_tickers = {p.ticker for p in state.positions}

    # ファンダメンタル & ニュース取得
    tickers = get_all_tickers()
    try:
        fund_data = fetch_fundamentals(tickers, db=db)
    except Exception:
        fund_data = {}
    try:
        news_data = fetch_news(tickers)
    except Exception:
        news_data = {}

    # スコアリング
    universe = {}
    for ticker in tickers:
        universe[ticker] = {
            "sector": get_sector(ticker),
            "price_df": price_data.get(ticker),
            "fundamentals": fund_data.get(ticker),
            "headlines": news_data.get(ticker, []),
        }
    results = score_universe(universe)

    # BUY系シグナルを選別（スコア順）
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

        # ポジションサイジング: 総資産の10%、キャッシュ上限あり
        alloc = min(total_assets * 0.10, state.cash)
        if alloc < 10:
            break  # キャッシュ不足

        df = price_data[r.ticker]
        entry_price = float(df.iloc[-1]["Close"])
        sl_pct = (entry_price - r.risk.stop_loss) / entry_price
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
    """現在の総資産を計算する（キャッシュ + 全ポジション評価額）。"""
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
