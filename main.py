#!/usr/bin/env python3
"""
Swing Trade Signal System — CLI エントリーポイント

日足ベースのスイングトレード（手動発注前提）に特化したシグナル配信バッチシステム。
1日1回の実行で、全銘柄のシグナル（Buy/Sell/Hold）と
損切り価格（Stop Loss）・利確価格（Take Profit）を算出する。

使い方:
  python main.py              フル分析（ファクター + Claude API）
  python main.py --quick      クイック分析（ファクターのみ、Claude 不使用）
  python main.py --ticker NVDA  1銘柄の詳細分析
  python main.py --screen     Russell 3000 スクリーニング → クイック分析
  python main.py --backtest   直近5年バックテスト（シグナル損益 + スクリーニング精度）
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any

import pandas as pd


# ============================================================
# ロギング設定
# ============================================================

def _setup_logging(verbose: bool = False) -> None:
    """ロギングを設定する。"""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

    # yfinance の過剰なログを抑制
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)


logger = logging.getLogger("main")


# ============================================================
# CLI 引数定義
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    """argparse パーサーを構築する。"""
    parser = argparse.ArgumentParser(
        description="Swing Trade Signal System — 日足スイングトレードシグナル配信",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
実行例:
  python main.py                  全銘柄フル分析（Claude API 使用）
  python main.py --quick          全銘柄クイック分析（Claude API 不使用）
  python main.py --ticker NVDA    NVDA の詳細分析
  python main.py --quick --no-notify   通知なしクイック分析
  python main.py --screen              Russell 3000 スクリーニング
  python main.py --backtest            直近5年バックテスト
        """,
    )
    # 実行モード
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick",
        action="store_true",
        help="クイックモード: Claude API を使わず高速分析",
    )
    mode_group.add_argument(
        "--ticker",
        type=str,
        metavar="SYMBOL",
        help="単一銘柄の詳細分析 (例: --ticker NVDA)",
    )
    mode_group.add_argument(
        "--screen",
        action="store_true",
        help="Russell 3000 スクリーニング → 候補銘柄をクイック分析",
    )
    mode_group.add_argument(
        "--backtest",
        action="store_true",
        help="直近5年バックテスト（シグナル損益 + スクリーニング精度）",
    )

    # オプション
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="LINE 通知を送信しない",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="CSV ファイルを保存しない",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="キャッシュを無視して全データを再取得",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="デバッグログを表示",
    )

    return parser


# ============================================================
# メインパイプライン
# ============================================================

def main() -> int:
    """メインエントリーポイント。終了コードを返す。"""
    parser = _build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    start_time = time.time()

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║     Swing Trade Signal System                   ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    try:
        if args.ticker:
            _run_single_ticker(args)
        elif args.screen:
            _run_screening(args)
        elif args.backtest:
            _run_backtest(args)
        elif args.quick:
            _run_quick(args)
        else:
            _run_full(args)

        elapsed = time.time() - start_time
        print(f"\n✅ 完了 (所要時間: {elapsed:.1f}秒)")
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠ ユーザーにより中断されました")
        return 130
    except Exception:
        logger.error("予期せぬエラーが発生しました", exc_info=True)
        print("\n❌ エラーが発生しました。--verbose オプションで詳細を確認してください。")
        return 1


# ============================================================
# モード1: フル分析（デフォルト）
# ============================================================

def _run_full(args: argparse.Namespace) -> None:
    """全銘柄フル分析（Claude API 使用）。Russell 3000 Tier1ベース。"""
    from analysis.claude_analyst import analyze_batch
    from config.settings import CLAUDE_API_CONFIG
    from config.universe import get_sector
    from data.cache import CacheDB
    from data.fundamental_fetcher import fetch_fundamentals
    from data.news_fetcher import fetch_news
    from data.price_fetcher import fetch_prices
    from data.screener import fetch_iwv_holdings, screen_tier1
    from output.notifier import notify_signals
    from output.report import generate_report
    from strategy.scorer import score_universe

    print("📊 モード: フル分析（Claude API 使用）")
    print("  ユニバース: Russell 3000 (IWV) Tier1フィルター")

    # Step 1: DB 初期化
    db = CacheDB()
    db.initialize()
    logger.info("SQLite DB 初期化完了")

    # Step 2: ティッカー取得（Russell 3000 → Tier1フィルタ）
    print("📡 対象銘柄を選定中（Russell 3000 Tier1）...")
    iwv_tickers = fetch_iwv_holdings(db)
    tickers = screen_tier1(iwv_tickers, db)
    print(f"  対象: {len(tickers)} 銘柄")
    print()

    # Step 3: データ取得
    print("⏳ [1/4] 株価データ取得中...")
    price_data = fetch_prices(tickers, db=db, force_refresh=args.force_refresh)
    print(f"  → {len(price_data)} 銘柄の株価を取得")

    print("⏳ [2/4] ファンダメンタルデータ取得中...")
    fund_data = fetch_fundamentals(tickers, db=db, force_refresh=args.force_refresh)
    print(f"  → {len(fund_data)} 銘柄のファンダメンタルを取得")

    print("⏳ [3/4] ニュースヘッドライン取得中...")
    news_data = fetch_news(tickers)
    print(f"  → {sum(len(v) for v in news_data.values())} 件のニュースを取得")

    # Step 4: Claude API 分析
    has_api_key = bool(CLAUDE_API_CONFIG.api_key)
    claude_scores: dict[str, float] = {}

    if has_api_key:
        print("⏳ [3.5/4] Claude API 分析中...")
        batch_input = {
            t: {
                "fundamentals": fund_data.get(t, {}),
                "headlines": news_data.get(t, []),
            }
            for t in tickers
        }
        claude_results = analyze_batch(batch_input)
        claude_scores = {
            t: r["combined_score"] for t, r in claude_results.items()
        }
        print(f"  → {len(claude_scores)} 銘柄の Claude 分析完了")
    else:
        print("  ⚠ ANTHROPIC_API_KEY 未設定 — Claude 分析はスキップ")

    # Step 5: スコアリング
    print("⏳ [4/4] スコアリング中...")
    universe = _build_universe(
        tickers, price_data, fund_data, news_data, claude_scores,
    )
    results = score_universe(universe)
    print(f"  → {len(results)} 銘柄のスコアリング完了")

    # Step 6: シグナルログ保存
    _save_signal_logs(results, db)

    # Step 7: 出力
    print()
    csv_path = generate_report(
        results,
        show_console=True,
        save_csv=not args.no_csv,
    )
    if csv_path:
        print(f"📁 CSV 保存先: {csv_path}")

    # Step 8: LINE 通知
    if not args.no_notify:
        notify_signals(results)


# ============================================================
# モード2: クイック分析
# ============================================================

def _run_quick(args: argparse.Namespace) -> None:
    """全銘柄クイック分析（Claude API 不使用）。Russell 3000 Tier1ベース。"""
    from data.cache import CacheDB
    from data.fundamental_fetcher import fetch_fundamentals
    from data.news_fetcher import fetch_news
    from data.price_fetcher import fetch_prices
    from data.screener import fetch_iwv_holdings, screen_tier1
    from output.notifier import notify_signals
    from output.report import generate_report
    from strategy.scorer import score_universe

    print("⚡ モード: クイック分析（Claude API 不使用）")
    print("  ユニバース: Russell 3000 (IWV) Tier1フィルター")

    # Step 1: DB 初期化
    db = CacheDB()
    db.initialize()

    # Step 2: ティッカー取得（Russell 3000 → Tier1フィルタ）
    print("📡 対象銘柄を選定中（Russell 3000 Tier1）...")
    iwv_tickers = fetch_iwv_holdings(db)
    tickers = screen_tier1(iwv_tickers, db)
    print(f"  対象: {len(tickers)} 銘柄")
    print()

    # Step 3: データ取得
    print("⏳ [1/3] 株価データ取得中...")
    price_data = fetch_prices(tickers, db=db, force_refresh=args.force_refresh)
    print(f"  → {len(price_data)} 銘柄の株価を取得")

    print("⏳ [2/3] ファンダメンタルデータ取得中...")
    fund_data = fetch_fundamentals(tickers, db=db, force_refresh=args.force_refresh)
    print(f"  → {len(fund_data)} 銘柄のファンダメンタルを取得")

    print("⏳ [3/3] ニュース取得 & スコアリング中...")
    news_data = fetch_news(tickers)
    print(f"  → {sum(len(v) for v in news_data.values())} 件のニュースを取得")

    # Step 4: スコアリング（claude_score=None → キーワードセンチメント使用）
    universe = _build_universe(
        tickers, price_data, fund_data, news_data, claude_scores={},
    )
    results = score_universe(universe)
    print(f"  → {len(results)} 銘柄のスコアリング完了")

    # Step 5: シグナルログ保存
    _save_signal_logs(results, db)

    # Step 6: 出力
    print()
    csv_path = generate_report(
        results,
        show_console=True,
        save_csv=not args.no_csv,
    )
    if csv_path:
        print(f"📁 CSV 保存先: {csv_path}")

    # Step 7: LINE 通知
    if not args.no_notify:
        notify_signals(results)


# ============================================================
# モード3: 単一銘柄分析
# ============================================================

def _run_single_ticker(args: argparse.Namespace) -> None:
    """1銘柄の詳細分析。"""
    from analysis.claude_analyst import analyze_ticker as claude_analyze
    from config.settings import CLAUDE_API_CONFIG
    from config.universe import get_sector
    from data.cache import CacheDB
    from data.fundamental_fetcher import fetch_fundamental_single
    from data.news_fetcher import fetch_news_single
    from data.price_fetcher import fetch_price_single
    from output.report import print_single_ticker
    from strategy.scorer import score_ticker

    ticker = args.ticker.upper()
    sector = get_sector(ticker)
    print(f"🔍 モード: 単一銘柄分析 — {ticker} ({sector})")
    print()

    # Step 1: DB 初期化
    db = CacheDB()
    db.initialize()

    # Step 2: データ取得
    print("⏳ 株価データ取得中...")
    price_df = fetch_price_single(ticker, db=db, period="1y")
    if price_df is not None:
        print(f"  → {len(price_df)} 日分の株価データ")
    else:
        print("  ⚠ 株価データ取得失敗")

    print("⏳ ファンダメンタルデータ取得中...")
    fundamentals = fetch_fundamental_single(ticker, db=db)
    if fundamentals:
        print(f"  → {len(fundamentals)} 指標を取得")
    else:
        print("  ⚠ ファンダメンタルデータ取得失敗")

    print("⏳ ニュースヘッドライン取得中...")
    headlines = fetch_news_single(ticker)
    print(f"  → {len(headlines)} 件のニュースを取得")

    # Step 3: Claude API 分析（キーがあれば）
    claude_score: float | None = None
    has_api_key = bool(CLAUDE_API_CONFIG.api_key)

    if has_api_key:
        print("⏳ Claude API 分析中...")
        claude_result = claude_analyze(fundamentals or {}, headlines, ticker)
        claude_score = claude_result["combined_score"]
        print(f"  → Claude スコア: {claude_score:+.3f}")
    else:
        print("  ⚠ ANTHROPIC_API_KEY 未設定 — キーワードセンチメント使用")

    # Step 4: スコアリング
    result = score_ticker(
        ticker=ticker,
        sector=sector,
        price_df=price_df,
        fundamentals=fundamentals,
        headlines=headlines,
        claude_score=claude_score,
    )

    # Step 5: シグナルログ保存
    _save_signal_logs([result], db)

    # Step 6: 詳細出力
    print_single_ticker(result)


# ============================================================
# モード4: Russell 3000 スクリーニング
# ============================================================

def _run_screening(args: argparse.Namespace) -> None:
    """Russell 3000 スクリーニング → クイック分析。"""
    from config.universe import get_all_tickers, get_sector
    from data.cache import CacheDB
    from data.fundamental_fetcher import fetch_fundamentals
    from data.news_fetcher import fetch_news
    from data.price_fetcher import fetch_prices
    from data.screener import run_screening
    from output.notifier import notify_signals
    from output.report import generate_report
    from strategy.scorer import score_universe

    print("🔎 モード: Russell 3000 スクリーニング")
    print()

    # Step 1: DB 初期化
    db = CacheDB()
    db.initialize()

    # Step 2: スクリーニング（Tier1 + Tier2）
    print("📡 スクリーニング実行中...")
    screened_tickers = run_screening(db)

    if not screened_tickers:
        print("\n⚠ スクリーニングで候補銘柄が見つかりませんでした")
        return

    # Step 3: コアユニバースとマージ（重複除去）
    core_tickers = set(get_all_tickers())
    merged = sorted(set(screened_tickers) | core_tickers)
    new_count = len(merged) - len(core_tickers)
    print(f"\n  スクリーニング候補: {len(screened_tickers)} 銘柄")
    print(f"  コアユニバース: {len(core_tickers)} 銘柄")
    print(f"  マージ後合計: {len(merged)} 銘柄 (新規: {new_count})")
    print()

    # Step 4: クイック分析パイプライン
    print("⏳ [1/3] 株価データ取得中...")
    price_data = fetch_prices(merged, db=db, force_refresh=args.force_refresh)
    print(f"  → {len(price_data)} 銘柄の株価を取得")

    print("⏳ [2/3] ファンダメンタルデータ取得中...")
    fund_data = fetch_fundamentals(merged, db=db, force_refresh=args.force_refresh)
    print(f"  → {len(fund_data)} 銘柄のファンダメンタルを取得")

    print("⏳ [3/3] ニュース取得 & スコアリング中...")
    news_data = fetch_news(merged)
    print(f"  → {sum(len(v) for v in news_data.values())} 件のニュースを取得")

    # Step 5: スコアリング
    universe = _build_universe(
        merged, price_data, fund_data, news_data, claude_scores={},
    )
    results = score_universe(universe)
    print(f"  → {len(results)} 銘柄のスコアリング完了")

    # Step 6: シグナルログ保存
    _save_signal_logs(results, db)

    # Step 7: 出力
    print()
    csv_path = generate_report(
        results,
        show_console=True,
        save_csv=not args.no_csv,
    )
    if csv_path:
        print(f"📁 CSV 保存先: {csv_path}")

    # Step 8: LINE 通知
    if not args.no_notify:
        notify_signals(results)


# ============================================================
# モード5: バックテスト
# ============================================================

def _run_backtest(args: argparse.Namespace) -> None:
    """直近5年バックテスト（STRONG_BUY trail×4.0 vs trail×4.5 比較）。

    Phase1（銘柄スコアリング）を1回だけ実行し、
    Phase2（ポートフォリオシミュレーション）をトレーリング幅ごとに実行することで
    フルバックテスト2回分の時間を節約する。
    """
    from backtest.engine import run_signal_backtest_multi_trail
    from backtest.report import print_signal_backtest, save_signal_backtest_csv
    from data.cache import CacheDB
    from data.fundamental_fetcher import fetch_fundamentals
    from data.price_fetcher import fetch_prices
    from data.screener import fetch_iwv_holdings, screen_tier1

    print("📊 モード: バックテスト (直近4年 / SB trail×4.0 vs trail×4.5 比較)")
    print()

    # Step 1: DB 初期化
    db = CacheDB()
    db.initialize()

    # Step 2: 対象銘柄リスト（Tier1 フィルタのみ）
    # ※ Tier2（モメンタム上位100選）はバックテストエンジン内でローリング月次で適用する。
    #   ここで Tier2 を適用すると「現在のスコア」でフィルタするルックアヘッドバイアスが発生するため削除。
    print("📡 対象銘柄を選定中...")
    print("  [1/2] IWV 構成銘柄を取得中...")
    iwv_tickers = fetch_iwv_holdings(db)
    print(f"    → {len(iwv_tickers)} 銘柄を取得")

    print("  [2/2] Tier 1 フィルタ中（時価総額・出来高・株価）...")
    tier1_tickers = screen_tier1(iwv_tickers, db)
    print(f"    → {len(tier1_tickers)} 銘柄が通過")

    tickers = sorted(set(tier1_tickers))
    print(f"  最終対象: {len(tickers)} 銘柄")
    print(f"  ※ Tier2（モメンタム上位100）はエンジン内でローリング月次適用（バイアスなし）")
    print()

    # Step 3: 6年分株価取得（4年 + 200MA ウォームアップ用 2年分バッファ）
    # allow_stale=True: バックテストは今日の最新データ不要、DBの履歴データをそのまま使用
    print("⏳ [1/5] 6年分の株価データ取得中（DBキャッシュ優先）...")
    price_data = fetch_prices(
        tickers, db=db, period="6y",
        force_refresh=args.force_refresh,
        allow_stale=not args.force_refresh,
    )
    print(f"  → {len(price_data)} 銘柄の株価を取得")

    # Step 3.5: SPY データ取得（市場環境フィルター用）
    print("⏳ [2/5] SPY データ取得中（市場環境フィルター用）...")
    from data.price_fetcher import fetch_price_single
    spy_results = fetch_prices(["SPY"], db=db, period="6y", allow_stale=not args.force_refresh)
    spy_df = spy_results.get("SPY")
    if spy_df is not None:
        print(f"  → SPY: {len(spy_df)} 日分")
    else:
        print("  ⚠ SPY データ取得失敗 — 市場環境フィルター無効")

    # Step 4: ファンダメンタル取得（現時点の値を全期間に適用）
    print("⏳ [3/5] ファンダメンタルデータ取得中...")
    fund_data = fetch_fundamentals(tickers, db=db)
    print(f"  → {len(fund_data)} 銘柄のファンダメンタルを取得")

    # Step 4.5: 決算日取得（決算前エントリー禁止 / 保有ポジション強制決済用）
    print("⏳ [4/5] 決算日データ取得中...")
    from data.earnings_fetcher import fetch_earnings_dates
    earnings_dates = fetch_earnings_dates(tickers, force_refresh=False)  # 7日キャッシュ有効
    ed_count = sum(1 for v in earnings_dates.values() if v)
    print(f"  → {ed_count} / {len(tickers)} 銘柄で決算日データあり")

    # Step 4.8: 四半期財務データ取得（時系列 Quality/Sentiment 対応）
    print("⏳ [4.8/5] 四半期財務データ取得中（時系列 Quality/Sentiment 用）...")
    from data.quarterly_fetcher import fetch_quarterly_data
    quarterly_data = fetch_quarterly_data(tickers, db=db)
    q_count = sum(1 for v in quarterly_data.values() if v.get("quarters"))
    print(f"  → {q_count} 銘柄で四半期財務データあり")
    print(f"  → ウェイト: Momentum=65%, Quality=35%, Sentiment=0%（ライブは Claude AI）")

    # Step 5: シグナルバックテスト（trail×4.0 と trail×4.5 を比較）
    # Phase1（銘柄スコアリング）は1回だけ実行し、Phase2をtrail値ごとに実行
    print("⏳ [5/5] バックテスト実行中（SB trail×4.0 と trail×4.5 を1回のPhase1で比較）...")
    trail_results = run_signal_backtest_multi_trail(
        tickers, price_data, fund_data,
        spy_df=spy_df, earnings_dates=earnings_dates,
        quarterly_data=quarterly_data,
        sb_trail_mults=[4.0, 4.5],
    )

    # Step 6: 結果表示（各trail値の詳細 + 比較サマリー）
    for mult in sorted(trail_results.keys()):
        result = trail_results[mult]
        print(f"\n【STRONG_BUY trail×{mult:.1f}】")
        print_signal_backtest(result)

    # 比較サマリー表
    print()
    print("━" * 68)
    print("  📊 STRONG_BUY トレーリング幅 比較サマリー")
    print("━" * 68)
    print(f"  {'trail倍率':<10} {'トレード数':>8} {'勝率':>8} {'平均損益':>10} {'累計損益':>12} {'Sharpe':>8}")
    print("  " + "-" * 60)
    for mult in sorted(trail_results.keys()):
        r = trail_results[mult]
        # 総資産換算（$4,000 スタート想定）
        capital = 4000.0
        slots = 12  # MAX_CONCURRENT_POSITIONS
        total_val = capital * (1 + r.avg_pl_pct) ** r.total_trades if r.total_trades > 0 else capital
        print(
            f"  trail×{mult:<4.1f}   "
            f"{r.total_trades:>8} "
            f"{r.win_rate * 100:>7.1f}% "
            f"{r.avg_pl_pct * 100:>+9.2f}% "
            f"{r.total_pl_pct * 100:>+11.2f}% "
            f"{r.sharpe_ratio:>8.2f}"
        )
    print("━" * 68)

    # Step 7: CSV 保存
    if not args.no_csv:
        for mult in sorted(trail_results.keys()):
            result = trail_results[mult]
            label = f"sb_trail{mult:.1f}"
            sig_path = save_signal_backtest_csv(result, label=label)
            if sig_path:
                print(f"\n📁 CSV (trail×{mult:.1f}): {sig_path}")


# ============================================================
# 共通ヘルパー
# ============================================================

def _build_universe(
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
    fund_data: dict[str, dict[str, Any]],
    news_data: dict[str, list[str]],
    claude_scores: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """score_universe に渡すユニバース辞書を構築する。"""
    from config.universe import get_sector

    universe: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        entry: dict[str, Any] = {
            "sector": get_sector(ticker),
            "price_df": price_data.get(ticker),
            "fundamentals": fund_data.get(ticker),
            "headlines": news_data.get(ticker, []),
        }
        # Claude スコアがあればセット
        if ticker in claude_scores:
            entry["claude_score"] = claude_scores[ticker]

        universe[ticker] = entry

    return universe


def _save_signal_logs(
    results: list,
    db: Any,
) -> None:
    """スコアリング結果をシグナルログに保存する。"""
    from strategy.scorer import HOLD

    count = 0
    for r in results:
        # HOLD 以外のシグナル、またはスコアが有意な場合のみ記録
        if r.signal != HOLD or abs(r.final_score) >= 0.3:
            db.log_signal(
                ticker=r.ticker,
                signal=r.signal,
                score=r.final_score,
                stop_loss=r.risk.stop_loss if r.risk else None,
                take_profit=r.risk.take_profit if r.risk else None,
                reason=r.reason,
            )
            count += 1

    if count > 0:
        logger.info("シグナルログに %d 件を記録", count)


def _print_universe_summary(summary: dict[str, int]) -> None:
    """ユニバースのサマリーを表示する。"""
    parts = []
    for key, count in summary.items():
        if key == "total_unique":
            continue
        parts.append(f"{key}({count})")
    print(f"  ユニバース: {', '.join(parts)}")


# ============================================================
# エントリーポイント
# ============================================================

if __name__ == "__main__":
    sys.exit(main())
