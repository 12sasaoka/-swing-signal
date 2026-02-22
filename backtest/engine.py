"""
Swing Trade Signal System — バックテストエンジン

2種類のバックテストを提供する:

1. シグナルバックテスト (run_signal_backtest)
   - 株価データでウォークフォワード分析（デフォルト5年）
   - BUY/STRONG_BUY シグナル発生後、SLヒットまたは段階的タイムアウトでポジションを決済
   - 固定TPなし — 利益を伸ばすため段階的タイムアウトで保有期間を制御
   - 段階的タイムアウト:
     - 30営業日: 含み損（< 0%）→ 強制決済、含み益なら保有継続
     - 90営業日: 無条件で強制決済（絶対タイムアウト）
   - 勝率・平均損益・Sharpe比・最大ドローダウンを集計
   - 市場環境フィルター: SPY終値 < SPY 200日MA のときはBUYシグナルを停止
   - 同時ポジション上限: 最大10銘柄まで（集中リスク抑制）

2. スクリーニングバックテスト (run_screening_backtest)
   - 月次でモメンタムスコア上位銘柄を選択
   - 翌月の実績リターンで「選ばれた銘柄 vs 選ばれなかった銘柄」を比較

注意事項:
- ルックアヘッドバイアス防止: スコアリング時は「その日まで」のデータのみ使用
- ファンダメンタル: 時系列データが取得困難なため現時点の値を全期間に適用
- Sentiment: バックテスト中はウェイトを0にしてMomentum/Qualityに再分配
- SL: ATRベースの動的SL（ATR×2.0、-8%フロア）
- TP: 固定TPなし — トレーリングストップ（最高値 - ATR×3.5）+ 段階的タイムアウト
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
from bisect import bisect_left
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# Phase1 並列処理ワーカー
# ============================================================

_worker_shared: dict = {}


def _init_phase1_worker(spy_df, market_ok, warmup_days, earnings_dates, quarterly_data=None):
    """Phase1ワーカーの初期化（プロセスごとに1回実行）。

    spy_df と market_ok はワーカー間で共有（spawn時に各プロセスへコピー）。
    quarterly_data が渡された場合は時系列対応のウェイト（M=60%, Q=30%, S=10%）を設定する。
    quarterly_data が None の場合は Momentum=100% フォールバック。
    earnings_dates: ticker → sorted list of earnings date strings (YYYY-MM-DD)
    """
    import config.settings as cfg
    if quarterly_data:
        # 時系列 Quality 有効: M=65%, Q=35%
        # Sentiment はバックテストでは 0%（ライブでは Claude AI が担当）
        cfg.FACTOR_WEIGHTS = cfg.FactorWeights(
            momentum=0.65, value=0.00, quality=0.35, sentiment=0.00,
        )
    else:
        # フォールバック: Momentum のみ（ルックアヘッドバイアス回避）
        cfg.FACTOR_WEIGHTS = cfg.FactorWeights(
            momentum=1.00, value=0.00, quality=0.00, sentiment=0.00,
        )
    _worker_shared["spy_df"] = spy_df
    _worker_shared["market_ok"] = market_ok
    _worker_shared["warmup_days"] = warmup_days
    _worker_shared["earnings_dates"] = earnings_dates or {}
    _worker_shared["quarterly_data"] = quarterly_data or {}


def _score_ticker_phase1(args):
    """1銘柄の全日付をスコアリングし、シグナル候補リストを返す。

    メインプロセスの score_ticker と同じロジックだが、
    multiprocessing 用にモジュールレベル関数として定義。
    """
    ticker, df, sector, fundamentals = args
    spy_df = _worker_shared["spy_df"]
    market_ok = _worker_shared["market_ok"]
    warmup_days = _worker_shared["warmup_days"]

    from strategy.scorer import score_ticker as _score_ticker

    BUY_SIG = "BUY"
    STRONG_BUY_SIG = "STRONG_BUY"

    candidates = []

    try:
        if df is None or df.empty or len(df) < warmup_days + 5:
            return candidates

        trading_dates = df.index.tolist()

        for i in range(warmup_days, len(trading_dates)):
            current_date = trading_dates[i]
            current_date_str = current_date.strftime("%Y-%m-%d")

            # 市場環境フィルター
            if market_ok and not market_ok.get(current_date_str, True):
                continue

            # 翌営業日がなければスキップ
            if i + 1 >= len(trading_dates):
                continue

            # スコアリング
            window_df = df.iloc[: i + 1]
            try:
                # 時系列対応: その日時点で公表済みの最新四半期データを使用
                quarterly_data = _worker_shared.get("quarterly_data", {})
                if quarterly_data:
                    from data.quarterly_fetcher import get_fundamentals_as_of
                    hist_fund = get_fundamentals_as_of(ticker, current_date_str, quarterly_data)
                else:
                    hist_fund = fundamentals

                result = _score_ticker(
                    ticker=ticker,
                    sector=sector,
                    price_df=window_df,
                    fundamentals=hist_fund,
                    headlines=None,
                    claude_score=None,  # バックテストは Sentiment=0%（ライブは Claude AI）
                    spy_df=spy_df,
                )

                # ---- 動的重み正規化 ----
                # quality データがない期間（2019年など）は
                # quality の重みをモメンタムに再配分して閾値スケールを維持する
                # quality あり: M=65%, Q=35%  → composite スケール維持
                # quality なし: M=65/65=100%  → Momentum のみ（スケール維持）
                if quarterly_data and hist_fund is None:
                    import numpy as _np
                    from strategy.scorer import determine_signal as _det_sig
                    # quality データなし → Momentum=100% で再計算
                    result.final_score = float(_np.clip(result.momentum_score, -1.0, 1.0))
                    result.signal = _det_sig(result.final_score)
            except Exception:
                continue

            if result.signal not in (BUY_SIG, STRONG_BUY_SIG):
                continue
            if result.risk is None:
                continue

            # ---- 出来高確認フィルター ----
            # シグナル日の出来高が過去N日平均の VOLUME_FILTER_MULTIPLIER 倍以上か確認
            if (
                VOLUME_FILTER_MULTIPLIER > 0
                and "Volume" in window_df.columns
                and len(window_df) > VOLUME_FILTER_PERIOD
            ):
                current_vol = float(window_df["Volume"].iloc[-1])
                avg_vol = float(window_df["Volume"].iloc[-(VOLUME_FILTER_PERIOD + 1):-1].mean())
                if avg_vol > 0 and current_vol < avg_vol * VOLUME_FILTER_MULTIPLIER:
                    continue  # 出来高不足 → スキップ

            entry_row = df.iloc[i + 1]
            entry_price = float(entry_row["Open"])
            if entry_price <= 0:
                continue

            # 決算前エントリー禁止チェック
            # Phase2の強制決済閾値（5カレンダー日）と一致させ、同日Entry+Exit問題を防ぐ
            earnings_map = _worker_shared.get("earnings_dates", {})
            ticker_earnings = earnings_map.get(ticker, [])
            if ticker_earnings:
                entry_date_chk = trading_dates[i + 1].strftime("%Y-%m-%d")
                ei = bisect_left(ticker_earnings, entry_date_chk)
                if ei < len(ticker_earnings):
                    days_to_e = (
                        pd.Timestamp(ticker_earnings[ei]) - pd.Timestamp(entry_date_chk)
                    ).days
                    if days_to_e <= 5:  # Phase2強制決済と同じ5カレンダー日以内は禁止
                        continue

            candidates.append({
                "date": current_date_str,
                "entry_date_str": trading_dates[i + 1].strftime("%Y-%m-%d"),
                "ticker": ticker,
                "signal": result.signal,
                "score": result.final_score,
                "atr": result.risk.atr,
                "entry_price": entry_price,
                "df_idx": i,
            })
    except Exception:
        pass  # ワーカーエラーは握り潰し（他銘柄は継続）

    return candidates


@contextmanager
def _backtest_weights():
    """バックテスト中はMomentum 100%に設定する。

    Quality/Sentimentは現時点のファンダメンタルデータに依存するため、
    過去バックテストではルックアヘッドバイアスになる。
    Momentumは純粋な価格データのみで算出されるため唯一クリーンに使える。

    通常: Momentum=0.50, Value=0.00, Quality=0.30, Sentiment=0.20
    BT時: Momentum=1.00, Value=0.00, Quality=0.00, Sentiment=0.00
    """
    import config.settings as cfg

    original = cfg.FACTOR_WEIGHTS
    cfg.FACTOR_WEIGHTS = cfg.FactorWeights(
        momentum=1.00,
        value=0.00,
        quality=0.00,
        sentiment=0.00,
    )
    try:
        yield
    finally:
        cfg.FACTOR_WEIGHTS = original

# 段階的タイムアウト設定
# (経過営業日, 最低含み益%) — 含み益が基準未満なら強制決済
STAGED_TIMEOUT = [
    (30, 0.00),   # 30営業日: 含み損（< 0%）なら決済、含み益なら保有継続
]
# 絶対タイムアウト（無条件で強制決済）
MAX_HOLD_DAYS = 90

# トレーリングストップ（シャンデリア・エグジット）デフォルト倍率
# 含み益が少ない初期段階ではこの倍率を使用
TRAILING_STOP_ATR_MULTIPLIER = 3.5

# 段階的トレーリングストップ: (最小含み益 ATR倍率, trail 倍率)
# 含み益が閾値以上になったら trail 倍率を縮小して利益をロック
# 降順で定義（大きい含み益から優先評価）
PROGRESSIVE_TRAIL_LEVELS: list[tuple[float, float]] = [
    (6.0, 2.0),  # 含み益 ≥ 6.0 ATR → 最高値 − ATR×2.0（タイト）
    (4.0, 2.5),  # 含み益 ≥ 4.0 ATR → 最高値 − ATR×2.5
    (2.0, 3.0),  # 含み益 ≥ 2.0 ATR → 最高値 − ATR×3.0
]

# ブレイクイーブン移動トリガー: 含み益がこの ATR 倍率以上で SL ≥ エントリー価格を保証
BREAKEVEN_TRIGGER_ATR: float = 1.0

# 出来高確認フィルター（エントリーシグナル時点）
VOLUME_FILTER_MULTIPLIER: float = 1.35   # 当日出来高 ≥ N日平均 × 1.35
VOLUME_FILTER_PERIOD: int = 20           # 平均算出期間（営業日）

# ウォームアップ日数（200日MAのため最低200日必要）
WARMUP_DAYS = 200

# 同時保有ポジション上限（ドローダウン抑制）
MAX_CONCURRENT_POSITIONS = 10

# 市場環境フィルター: SPY 200日MA を下回ったら BUY シグナルを停止
MARKET_FILTER_MA_PERIOD = 200


# ============================================================
# データクラス
# ============================================================

@dataclass
class TradeResult:
    """1トレードの結果。"""
    ticker: str
    signal: str             # "STRONG_BUY" | "BUY"
    signal_date: str        # シグナル発生日
    entry_date: str         # エントリー日（翌営業日）
    entry_price: float
    exit_date: str
    exit_price: float
    result: str             # "sl_hit" | "trailing_stop" | "timeout_30d" | "timeout_90d" | "pre_earnings" | "timeout"
    pl_pct: float           # 損益率 (-1.0 〜 +∞)
    sl_price: float
    tp_price: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "signal": self.signal,
            "signal_date": self.signal_date,
            "entry_date": self.entry_date,
            "entry_price": round(self.entry_price, 4),
            "exit_date": self.exit_date,
            "exit_price": round(self.exit_price, 4),
            "result": self.result,
            "pl_pct": round(self.pl_pct * 100, 2),
            "sl_price": round(self.sl_price, 4),
            "tp_price": round(self.tp_price, 4),
        }


@dataclass
class SignalBacktestResult:
    """シグナルバックテストの集計結果。"""
    trades: list[TradeResult] = field(default_factory=list)
    total_trades: int = 0
    win_rate: float = 0.0          # 勝率 (pl_pct > 0 の割合)
    avg_pl_pct: float = 0.0        # 平均損益率 (%)
    total_pl_pct: float = 0.0      # 累計損益率 (%)
    max_drawdown: float = 0.0      # 最大ドローダウン (%)
    sharpe_ratio: float = 0.0      # Sharpe比 (年率換算)
    by_signal: dict[str, dict] = field(default_factory=dict)   # シグナル別集計
    by_year: dict[str, dict] = field(default_factory=dict)     # 年別集計
    by_result: dict[str, int] = field(default_factory=dict)    # sl_hit/timeout_30d/60d/90d 別件数


@dataclass
class MonthlyRecord:
    """スクリーニングバックテストの1ヶ月分記録。"""
    month: str              # "YYYY-MM"
    selected_tickers: list[str]
    selected_return: float  # 選択銘柄の翌月平均リターン
    all_return: float       # 全銘柄の翌月平均リターン
    outperformance: float   # selected - all


@dataclass
class ScreeningBacktestResult:
    """スクリーニングバックテストの集計結果。"""
    monthly_records: list[MonthlyRecord] = field(default_factory=list)
    avg_selected_return: float = 0.0
    avg_all_return: float = 0.0
    avg_outperformance: float = 0.0
    win_months: int = 0     # アウトパフォームした月数
    total_months: int = 0


# ============================================================
# シグナルバックテスト
# ============================================================

def _build_market_filter(spy_df: pd.DataFrame | None) -> dict[str, bool]:
    """SPY の 200日MA フィルターを事前計算する。

    Returns:
        日付文字列 → True(BUY許可) / False(BUY禁止) の辞書。
        SPY データがない場合は空辞書（= フィルターなし）。
    """
    if spy_df is None or spy_df.empty:
        return {}

    close = spy_df["Close"].astype(float)
    ma200 = close.rolling(window=MARKET_FILTER_MA_PERIOD).mean()
    result: dict[str, bool] = {}
    for i in range(len(spy_df)):
        date_str = spy_df.index[i].strftime("%Y-%m-%d")
        if pd.isna(ma200.iloc[i]):
            result[date_str] = True  # MA算出前は許可
        else:
            result[date_str] = float(close.iloc[i]) >= float(ma200.iloc[i])
    return result


def run_signal_backtest(
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
    fund_data: dict[str, dict[str, Any]],
    spy_df: pd.DataFrame | None = None,
    earnings_dates: dict[str, list[str]] | None = None,
    quarterly_data: dict[str, dict] | None = None,
) -> SignalBacktestResult:
    """シグナルバックテストを実行する。

    Args:
        tickers:        分析対象のティッカーリスト。
        price_data:     ティッカー → OHLCV DataFrame。
        fund_data:      ティッカー → ファンダメンタル辞書（現時点の値を全期間に適用）。
        spy_df:         SPY の OHLCV DataFrame（市場環境フィルター用、Noneならフィルター無効）。
        earnings_dates:  ティッカー → 決算日リスト（YYYY-MM-DD、昇順）。
                         Noneの場合は決算フィルター無効。
        quarterly_data:  ティッカー → 四半期財務データ辞書。
                         Noneの場合は Momentum=100% フォールバック（ルックアヘッドバイアス回避）。
                         data.quarterly_fetcher.fetch_quarterly_data() の戻り値を渡すこと。

    Returns:
        SignalBacktestResult。
    """
    from config.universe import get_sector
    from strategy.scorer import score_ticker, BUY, STRONG_BUY
    from config.settings import ATR_PARAMS, TRADE_RULES

    # ---- 市場環境フィルター事前計算 ----
    market_ok = _build_market_filter(spy_df)
    if market_ok:
        bearish_days = sum(1 for v in market_ok.values() if not v)
        logger.info("市場環境フィルター: %d日中 %d日がベア判定（BUY禁止）",
                     len(market_ok), bearish_days)
    else:
        logger.info("市場環境フィルター: SPYデータなし → フィルター無効")

    # ---- SL パラメータ ----
    # ATRベースの動的SL（-8%ハードストップをフロアとして適用）
    sl_multiplier = ATR_PARAMS.stop_loss_multiplier  # 2.0
    hard_stop_pct = TRADE_RULES.hard_stop_loss_pct   # -0.08
    # 固定TPなし — 段階的タイムアウトで利益を伸ばす

    all_trades: list[TradeResult] = []

    # ---- 同時ポジション管理（全銘柄横断） ----
    # 銘柄ごとに直列処理するため、日付→ポジション数のグローバルカウンタを使う
    # まず日付ごとの処理に変換する必要がある
    # → 銘柄ごとにシグナル候補を収集し、日付順にポートフォリオ全体で管理する

    # Phase 1: 銘柄ごとにシグナル候補日とスコアを事前計算（並列処理）
    signal_candidates: list[dict] = []

    # ワーカーに渡すタスクを準備
    tasks = []
    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue
        if len(df) < WARMUP_DAYS + 5:
            continue
        sector = get_sector(ticker)
        fundamentals = fund_data.get(ticker)
        tasks.append((ticker, df, sector, fundamentals))

    n_workers = max(1, min((mp.cpu_count() or 4) - 1, 8))
    total_tasks = len(tasks)
    logger.info("Phase1: %d 銘柄を %d ワーカーで並列処理開始", total_tasks, n_workers)

    chunksize = max(1, total_tasks // (n_workers * 4))
    completed = 0

    if quarterly_data:
        logger.info("四半期財務データ: %d 銘柄分 (M=60%%, Q=30%%, S=10%%)", len(quarterly_data))
    else:
        logger.info("四半期財務データなし → Momentum=100%% フォールバック")

    with mp.Pool(
        processes=n_workers,
        initializer=_init_phase1_worker,
        initargs=(spy_df, market_ok, WARMUP_DAYS, earnings_dates or {}, quarterly_data or {}),
    ) as pool:
        for ticker_candidates in pool.imap_unordered(
            _score_ticker_phase1, tasks, chunksize=chunksize,
        ):
            completed += 1
            if completed % 50 == 0 or completed == total_tasks:
                logger.info(
                    "Phase1 進捗: %d / %d 銘柄 (%d%%)",
                    completed, total_tasks, int(completed / total_tasks * 100),
                )
            for c in ticker_candidates:
                c["df"] = price_data[c["ticker"]]
                signal_candidates.append(c)

    logger.info("シグナル候補: %d 件を収集", len(signal_candidates))

    # Phase 2: 日付順にソートし、同時ポジション制約付きでトレード実行
    signal_candidates.sort(key=lambda c: (c["date"], -c["score"]))

    # アクティブポジション管理: ticker → TradeResult (仮)
    active_positions: dict[str, dict] = {}  # ticker → {entry, sl, tp, trade_result, df, df_idx}

    # 日付ごとにグループ化して処理
    from itertools import groupby

    # 全日付を網羅するために、全銘柄の全日付を統合
    all_dates: set[str] = set()
    for ticker in tickers:
        df = price_data.get(ticker)
        if df is not None and not df.empty:
            for dt in df.index:
                all_dates.add(dt.strftime("%Y-%m-%d"))
    sorted_dates = sorted(all_dates)

    # シグナル候補を日付でインデックス化
    candidates_by_date: dict[str, list[dict]] = {}
    for c in signal_candidates:
        candidates_by_date.setdefault(c["date"], []).append(c)

    # 銘柄ごとに「最後に決済した日付」を記録（同一銘柄の重複エントリー防止）
    last_exit_date: dict[str, str] = {}

    for date_str in sorted_dates:
        # Step 1: 既存ポジションの決済判定
        closed_tickers = []
        for ticker, pos in list(active_positions.items()):
            df = pos["df"]
            # この日の行を取得
            if date_str not in [d.strftime("%Y-%m-%d") for d in df.index]:
                continue
            date_idx = None
            for idx_i, d in enumerate(df.index):
                if d.strftime("%Y-%m-%d") == date_str:
                    date_idx = idx_i
                    break
            if date_idx is None:
                continue

            row = df.iloc[date_idx]

            # 決算前 3営業日（≈ 5カレンダー日）強制決済チェック
            trade = None
            if earnings_dates:
                ticker_earnings = earnings_dates.get(ticker, [])
                if ticker_earnings:
                    ei = bisect_left(ticker_earnings, date_str)
                    if ei < len(ticker_earnings):
                        days_to_e = (
                            pd.Timestamp(ticker_earnings[ei]) - pd.Timestamp(date_str)
                        ).days
                        if 0 < days_to_e <= 5:  # 5カレンダー日以内（≈ 3営業日）
                            close_price = float(row["Close"])
                            tr = pos["trade_result"]
                            pl_pct = (close_price - tr.entry_price) / tr.entry_price
                            trade = TradeResult(
                                ticker=tr.ticker, signal=tr.signal,
                                signal_date=tr.signal_date, entry_date=tr.entry_date,
                                entry_price=tr.entry_price, exit_date=date_str,
                                exit_price=close_price, result="pre_earnings",
                                pl_pct=pl_pct,
                                sl_price=tr.sl_price, tp_price=0.0,
                            )

            # 通常の決済判定（SL / トレーリングストップ / タイムアウト）
            if trade is None:
                trade, new_hh = _check_exit(
                    pos["trade_result"], date_str, row,
                    atr=pos["atr"], highest_high=pos["highest_high"],
                )
                pos["highest_high"] = new_hh  # 最高値を更新

            if trade is not None:
                all_trades.append(trade)
                last_exit_date[ticker] = date_str
                closed_tickers.append(ticker)

        for t in closed_tickers:
            del active_positions[t]

        # Step 2: 新規エントリー（スコア降順、同時上限まで）
        day_candidates = candidates_by_date.get(date_str, [])
        for c in day_candidates:
            # 同時ポジション上限チェック
            if len(active_positions) >= MAX_CONCURRENT_POSITIONS:
                break

            ticker = c["ticker"]

            # 既にポジション保有中ならスキップ
            if ticker in active_positions:
                continue

            # 直前に決済した銘柄はこの日は再エントリーしない
            if last_exit_date.get(ticker) == date_str:
                continue

            entry_price = c["entry_price"]
            atr = c["atr"]

            # ATRベースの動的SL（-8%ハードストップをフロアとして適用）
            # risk.py と同様: max(ATRベースSL, -8%ハードSL) = 損失が小さい方を採用
            atr_sl   = entry_price - atr * sl_multiplier
            hard_sl  = entry_price * (1.0 + hard_stop_pct)   # -8%
            sl_price = max(atr_sl, hard_sl)

            # SL が 0以下にならないように最低保証
            if sl_price <= 0:
                sl_price = entry_price * 0.90  # フォールバック: -10%

            df = c["df"]
            df_idx = c["df_idx"]

            # 将来データ（エントリー日の翌日から最大 MAX_HOLD_DAYS）
            future_start = df_idx + 2
            future_end = min(df_idx + 2 + MAX_HOLD_DAYS, len(df))
            future_df = df.iloc[future_start:future_end]

            trade = _simulate_trade(
                ticker=ticker,
                signal=c["signal"],
                signal_date=date_str,
                entry_date=c["entry_date_str"],
                entry_price=entry_price,
                sl_price=sl_price,
                atr=atr,
                future_df=future_df,
            )

            active_positions[ticker] = {
                "trade_result": trade,
                "df": df,
                "atr": atr,
                "highest_high": entry_price,
            }

    # 期間終了時に残っているポジションを強制決済
    for ticker, pos in active_positions.items():
        tr = pos["trade_result"]
        df = pos["df"]
        last_row = df.iloc[-1]
        last_date = df.index[-1].strftime("%Y-%m-%d")
        pl_pct = (float(last_row["Close"]) - tr.entry_price) / tr.entry_price
        all_trades.append(TradeResult(
            ticker=tr.ticker,
            signal=tr.signal,
            signal_date=tr.signal_date,
            entry_date=tr.entry_date,
            entry_price=tr.entry_price,
            exit_date=last_date,
            exit_price=float(last_row["Close"]),
            result="timeout",
            pl_pct=pl_pct,
            sl_price=tr.sl_price,
            tp_price=tr.tp_price,
        ))

    logger.info("シグナルバックテスト完了: 合計 %d トレード", len(all_trades))
    return _aggregate_signal_results(all_trades)


def _get_trail_level(
    highest_high: float,
    entry_price: float,
    atr: float,
) -> float:
    """段階的トレーリングストップの水準を返す。

    含み益（ATR倍率）に応じてトレール倍率を縮小し、利益をロックする。
    BREAKEVEN_TRIGGER_ATR 以上の含み益では SL ≥ エントリー価格を保証（床）。

    Args:
        highest_high: 保有中の最高値。
        entry_price:  エントリー価格。
        atr:          エントリー時点のATR値。

    Returns:
        トレーリングストップ価格。
    """
    if atr <= 0:
        return highest_high - atr * TRAILING_STOP_ATR_MULTIPLIER

    peak_profit_atr = (highest_high - entry_price) / atr

    # 段階的倍率: 含み益大きいほどタイトに
    trail_mult = TRAILING_STOP_ATR_MULTIPLIER
    for min_atr, mult in PROGRESSIVE_TRAIL_LEVELS:
        if peak_profit_atr >= min_atr:
            trail_mult = mult
            break

    trail = highest_high - atr * trail_mult

    # ブレイクイーブン床: 含み益 ≥ BREAKEVEN_TRIGGER_ATR → SL はエントリー価格以上
    if peak_profit_atr >= BREAKEVEN_TRIGGER_ATR:
        trail = max(trail, entry_price)

    return trail


def _simulate_trade(
    ticker: str,
    signal: str,
    signal_date: str,
    entry_date: str,
    entry_price: float,
    sl_price: float,
    atr: float,
    future_df: pd.DataFrame,
) -> TradeResult:
    """SL + トレーリングストップ + 段階的タイムアウトでトレードを模擬する。

    - SL: エントリー時の固定SL（ATR×2.0、-8%フロア）
    - トレーリングストップ: 最高値 - ATR×3.5 を下回ったら利確（エントリー価格以上のみ発動）
    - 段階的タイムアウト:
      - 30営業日: 含み損（< 0%）→ timeout_30d
      - 90営業日: 無条件 → timeout_90d

    Args:
        atr:        エントリー時点のATR値（トレーリングストップ計算用）。
        future_df:  エントリー日翌日以降の価格DataFrame（最大MAX_HOLD_DAYS行）。

    Returns:
        TradeResult（ポジションとして仮保存した状態）。
    """
    highest_high = entry_price  # エントリー価格を初期値

    for idx in range(len(future_df)):
        row = future_df.iloc[idx]
        date_str = future_df.index[idx].strftime("%Y-%m-%d")
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        holding_days = idx + 1  # 1-indexed（エントリー翌日が1日目）

        # 最高値を更新
        if high > highest_high:
            highest_high = high

        # SL判定（ギャップダウン対応: 始値がSL以下なら始値で約定）
        open_price_row = float(row["Open"]) if "Open" in row.index else float(row["Low"])
        if low <= sl_price or open_price_row <= sl_price:
            exit_price_sl = min(sl_price, open_price_row)
            return TradeResult(
                ticker=ticker, signal=signal, signal_date=signal_date,
                entry_date=entry_date, entry_price=entry_price,
                exit_date=date_str, exit_price=exit_price_sl,
                result="sl_hit",
                pl_pct=(exit_price_sl - entry_price) / entry_price,
                sl_price=sl_price, tp_price=0.0,
            )

        # トレーリングストップ判定（段階的トレーリング + ブレイクイーブン床）
        trail_level = _get_trail_level(highest_high, entry_price, atr)
        if trail_level >= entry_price and close <= trail_level:
            pl_pct = (trail_level - entry_price) / entry_price
            return TradeResult(
                ticker=ticker, signal=signal, signal_date=signal_date,
                entry_date=entry_date, entry_price=entry_price,
                exit_date=date_str, exit_price=trail_level,
                result="trailing_stop",
                pl_pct=pl_pct,
                sl_price=sl_price, tp_price=0.0,
            )

        # 段階的タイムアウト判定
        pl_pct = (close - entry_price) / entry_price
        for threshold_days, min_profit in STAGED_TIMEOUT:
            if holding_days == threshold_days and pl_pct < min_profit:
                return TradeResult(
                    ticker=ticker, signal=signal, signal_date=signal_date,
                    entry_date=entry_date, entry_price=entry_price,
                    exit_date=date_str, exit_price=close,
                    result=f"timeout_{threshold_days}d",
                    pl_pct=pl_pct,
                    sl_price=sl_price, tp_price=0.0,
                )

    # 絶対タイムアウト（90営業日）: 最終日の終値で強制決済
    if len(future_df) > 0:
        last_row = future_df.iloc[-1]
        last_date = future_df.index[-1].strftime("%Y-%m-%d")
        last_close = float(last_row["Close"])
    else:
        last_date = entry_date
        last_close = entry_price

    return TradeResult(
        ticker=ticker, signal=signal, signal_date=signal_date,
        entry_date=entry_date, entry_price=entry_price,
        exit_date=last_date, exit_price=last_close,
        result="timeout_90d",
        pl_pct=(last_close - entry_price) / entry_price,
        sl_price=sl_price, tp_price=0.0,
    )


def _check_exit(
    position: TradeResult,
    current_date_str: str,
    row: pd.Series,
    atr: float,
    highest_high: float,
) -> tuple[TradeResult | None, float]:
    """保有中ポジションの決済判定（SL + トレーリングストップ + 段階的タイムアウト）。

    Args:
        atr:          エントリー時点のATR値。
        highest_high: これまでの最高値（呼び出し側で管理・更新する）。

    Returns:
        (決済TradeResult or None, 更新後のhighest_high) のタプル。
    """
    open_price = float(row["Open"]) if "Open" in row.index else float(row["Low"])
    high = float(row["High"])
    low = float(row["Low"])
    close = float(row["Close"])

    # 最高値を更新
    if high > highest_high:
        highest_high = high

    # SL判定（ギャップダウン対応: 始値がSL以下なら始値で約定）
    if low <= position.sl_price or open_price <= position.sl_price:
        exit_price = min(position.sl_price, open_price)
        pl_pct = (exit_price - position.entry_price) / position.entry_price
        return TradeResult(
            ticker=position.ticker, signal=position.signal,
            signal_date=position.signal_date, entry_date=position.entry_date,
            entry_price=position.entry_price, exit_date=current_date_str,
            exit_price=exit_price, result="sl_hit", pl_pct=pl_pct,
            sl_price=position.sl_price, tp_price=0.0,
        ), highest_high

    # トレーリングストップ判定（段階的トレーリング + ブレイクイーブン床）
    trail_level = _get_trail_level(highest_high, position.entry_price, atr)
    if trail_level >= position.entry_price and close <= trail_level:
        pl_pct = (trail_level - position.entry_price) / position.entry_price
        return TradeResult(
            ticker=position.ticker, signal=position.signal,
            signal_date=position.signal_date, entry_date=position.entry_date,
            entry_price=position.entry_price, exit_date=current_date_str,
            exit_price=trail_level, result="trailing_stop", pl_pct=pl_pct,
            sl_price=position.sl_price, tp_price=0.0,
        ), highest_high

    # 段階的タイムアウト判定
    try:
        entry_dt = pd.Timestamp(position.entry_date)
        current_dt = pd.Timestamp(current_date_str)
        calendar_days = (current_dt - entry_dt).days
        # カレンダー日数 → 営業日近似 (× 5/7)
        approx_biz_days = int(calendar_days * 5 / 7)

        pl_pct = (close - position.entry_price) / position.entry_price

        # 段階的タイムアウトチェック
        for threshold_days, min_profit in STAGED_TIMEOUT:
            if approx_biz_days >= threshold_days:
                if pl_pct < min_profit:
                    return TradeResult(
                        ticker=position.ticker, signal=position.signal,
                        signal_date=position.signal_date, entry_date=position.entry_date,
                        entry_price=position.entry_price, exit_date=current_date_str,
                        exit_price=close, result=f"timeout_{threshold_days}d",
                        pl_pct=pl_pct,
                        sl_price=position.sl_price, tp_price=0.0,
                    ), highest_high

        # 絶対タイムアウト（90営業日 ≈ 126カレンダー日）
        if approx_biz_days >= MAX_HOLD_DAYS:
            return TradeResult(
                ticker=position.ticker, signal=position.signal,
                signal_date=position.signal_date, entry_date=position.entry_date,
                entry_price=position.entry_price, exit_date=current_date_str,
                exit_price=close, result="timeout_90d", pl_pct=pl_pct,
                sl_price=position.sl_price, tp_price=0.0,
            ), highest_high
    except Exception:
        pass

    return None, highest_high


def _aggregate_signal_results(trades: list[TradeResult]) -> SignalBacktestResult:
    """トレード結果リストを集計してSignalBacktestResultを返す。"""
    if not trades:
        return SignalBacktestResult()

    result = SignalBacktestResult(trades=trades)
    result.total_trades = len(trades)

    pl_list = [t.pl_pct for t in trades]
    wins = [t for t in trades if t.pl_pct > 0]

    result.win_rate = len(wins) / len(trades) if trades else 0.0
    result.avg_pl_pct = float(np.mean(pl_list)) if pl_list else 0.0
    result.total_pl_pct = float(np.sum(pl_list))

    # 最大ドローダウン（累積損益ベース）
    cumulative = np.cumsum(pl_list)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    result.max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Sharpe比（日次リターンから年率換算）
    if len(pl_list) > 1:
        std = float(np.std(pl_list, ddof=1))
        if std > 0:
            # 取引頻度を考慮した年率換算
            # 平均保有期間を実績から推定
            avg_hold = 45  # 段階的タイムアウトの中央値を近似
            result.sharpe_ratio = (result.avg_pl_pct / std) * math.sqrt(252 / avg_hold)
        else:
            result.sharpe_ratio = 0.0

    # result別件数
    result_types = set(t.result for t in trades)
    result.by_result = {r: sum(1 for t in trades if t.result == r) for r in sorted(result_types)}

    # シグナル別集計
    for sig in ("STRONG_BUY", "BUY"):
        sig_trades = [t for t in trades if t.signal == sig]
        if not sig_trades:
            continue
        sig_pl = [t.pl_pct for t in sig_trades]
        result.by_signal[sig] = {
            "count": len(sig_trades),
            "win_rate": sum(1 for t in sig_trades if t.pl_pct > 0) / len(sig_trades),
            "avg_pl_pct": float(np.mean(sig_pl)),
        }

    # 年別集計
    for trade in trades:
        year = trade.entry_date[:4]
        if year not in result.by_year:
            result.by_year[year] = {"count": 0, "wins": 0, "total_pl": 0.0}
        result.by_year[year]["count"] += 1
        if trade.pl_pct > 0:
            result.by_year[year]["wins"] += 1
        result.by_year[year]["total_pl"] += trade.pl_pct

    for year_data in result.by_year.values():
        cnt = year_data["count"]
        year_data["win_rate"] = year_data["wins"] / cnt if cnt > 0 else 0.0
        year_data["avg_pl_pct"] = year_data["total_pl"] / cnt if cnt > 0 else 0.0

    return result


# ============================================================
# スクリーニングバックテスト
# ============================================================

def run_screening_backtest(
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
) -> ScreeningBacktestResult:
    """スクリーニングバックテストを実行する。

    各月末に momentum スコア上位銘柄を選択し、翌月の実績リターンと比較する。

    Args:
        tickers:    対象ティッカーリスト。
        price_data: ティッカー → OHLCV DataFrame。

    Returns:
        ScreeningBacktestResult。
    """
    from analysis.momentum import calc_momentum_score
    from config.screening import SCREENING_PARAMS

    # 全銘柄の月次リターンを算出（各月の最終営業日の終値ベース）
    monthly_close: dict[str, pd.Series] = {}
    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue
        # 月末終値（各月の最終営業日）
        monthly = df["Close"].resample("ME").last()
        monthly_close[ticker] = monthly

    if not monthly_close:
        return ScreeningBacktestResult()

    # 月次リターン計算
    monthly_returns: dict[str, pd.Series] = {}
    for ticker, series in monthly_close.items():
        monthly_returns[ticker] = series.pct_change()

    # 共通の月インデックスを取得
    all_months = sorted(set().union(*[set(s.index) for s in monthly_close.values()]))

    records: list[MonthlyRecord] = []

    # ウォームアップ後の月から開始（252日 ≈ 12ヶ月）
    start_idx = 12
    max_candidates = SCREENING_PARAMS.max_screen_candidates

    for m_idx in range(start_idx, len(all_months) - 1):
        month_end = all_months[m_idx]
        next_month = all_months[m_idx + 1]
        month_str = month_end.strftime("%Y-%m")

        # 各銘柄のmomentumスコアを月末時点のDataFrameで計算
        scored: list[tuple[str, float]] = []
        for ticker in tickers:
            df = price_data.get(ticker)
            if df is None or df.empty:
                continue
            # 月末日以前のデータのみ使用（ルックアヘッド防止）
            window = df.loc[:month_end]
            if len(window) < 30:
                continue
            try:
                score = calc_momentum_score(window)
                scored.append((ticker, score))
            except Exception:
                continue

        if not scored:
            continue

        # スコア上位を選択
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in scored[:max_candidates]]

        # 翌月のリターンを集計
        def get_next_month_return(ticker: str) -> float | None:
            ret_series = monthly_returns.get(ticker)
            if ret_series is None:
                return None
            if next_month not in ret_series.index:
                return None
            val = ret_series.loc[next_month]
            if pd.isna(val):
                return None
            return float(val)

        selected_rets = [r for t in selected if (r := get_next_month_return(t)) is not None]
        all_rets = [r for t in tickers if (r := get_next_month_return(t)) is not None]

        if not selected_rets or not all_rets:
            continue

        sel_avg = float(np.mean(selected_rets))
        all_avg = float(np.mean(all_rets))
        outperf = sel_avg - all_avg

        records.append(MonthlyRecord(
            month=month_str,
            selected_tickers=selected,
            selected_return=sel_avg,
            all_return=all_avg,
            outperformance=outperf,
        ))

    logger.info("スクリーニングバックテスト完了: %d ヶ月", len(records))
    return _aggregate_screening_results(records)


def _aggregate_screening_results(records: list[MonthlyRecord]) -> ScreeningBacktestResult:
    """月次記録を集計してScreeningBacktestResultを返す。"""
    if not records:
        return ScreeningBacktestResult()

    result = ScreeningBacktestResult(monthly_records=records)
    result.total_months = len(records)
    result.avg_selected_return = float(np.mean([r.selected_return for r in records]))
    result.avg_all_return = float(np.mean([r.all_return for r in records]))
    result.avg_outperformance = result.avg_selected_return - result.avg_all_return
    result.win_months = sum(1 for r in records if r.outperformance > 0)

    return result
