"""
Swing Trade Signal System — 株価データ取得

yfinance を使用して日足 OHLCV データをバッチ取得し、SQLite にキャッシュする。

主な機能:
  - バッチ取得: 複数銘柄を一括ダウンロード（100銘柄/バッチ）
  - 並列処理: ThreadPoolExecutor でバッチを並列実行
  - フォールバック: バッチ失敗時は個別取得にリトライ
  - キャッシュ: 取得済みデータは SQLite に保存し、重複ダウンロードを回避
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from config.settings import FETCHER_CONFIG, TECHNICAL_PARAMS
from data.cache import CacheDB

logger = logging.getLogger(__name__)


# ============================================================
# 公開 API
# ============================================================

def fetch_prices(
    tickers: list[str],
    db: CacheDB | None = None,
    period: str | None = None,
    force_refresh: bool = False,
    allow_stale: bool = False,
) -> dict[str, pd.DataFrame]:
    """指定銘柄の日足 OHLCV データを取得する。

    キャッシュが新鮮であればDBから読み込み、古ければ yfinance から再取得する。

    Args:
        tickers:       取得対象のティッカーリスト。
        db:            CacheDB インスタンス。None の場合は新規作成。
        period:        yfinance の period パラメータ（例: "1y"）。
                       None の場合は settings の price_history_days から算出。
        force_refresh: True の場合、キャッシュを無視して全銘柄を再取得。

    Returns:
        ティッカー → DataFrame (columns: Open, High, Low, Close, Volume, date index) の辞書。
        取得失敗した銘柄は辞書に含まれない。
    """
    if not tickers:
        return {}

    if db is None:
        db = CacheDB()
        db.initialize()

    # period から cache_days を算出（長期 period 対応）
    _period_to_days = {
        "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180,
        "1y": 365, "2y": 730, "3y": 1100, "4y": 1460, "5y": 1830,
        "6y": 2190, "7y": 2556, "10y": 3650,
        "ytd": 365, "max": 5000,
    }
    cache_days = _period_to_days.get(period) if period else None

    # キャッシュが新鮮な銘柄と、取得が必要な銘柄を仕分け
    if force_refresh:
        stale_tickers = list(tickers)
        cached_results: dict[str, pd.DataFrame] = {}
    elif allow_stale:
        # バックテスト用: 鮮度チェックをスキップし、DBにデータがあればそのまま使用
        cached_results, stale_tickers = _separate_cached(
            tickers, db, cache_days=cache_days, allow_stale=True
        )
    else:
        cached_results, stale_tickers = _separate_cached(tickers, db, cache_days=cache_days)

    if cached_results:
        logger.info(
            "キャッシュから %d 銘柄を読み込み済み（取得スキップ）",
            len(cached_results),
        )

    # yfinance から取得が必要な銘柄をバッチ処理
    if stale_tickers:
        logger.info("yfinance から %d 銘柄を取得開始", len(stale_tickers))
        fetched = _fetch_and_store(stale_tickers, db, period)
        cached_results.update(fetched)

    logger.info(
        "株価取得完了: 成功 %d / 全体 %d",
        len(cached_results),
        len(tickers),
    )
    return cached_results


def fetch_price_single(
    ticker: str,
    db: CacheDB | None = None,
    period: str | None = None,
) -> pd.DataFrame | None:
    """1銘柄の日足 OHLCV データを取得する（便利ラッパー）。

    Args:
        ticker: ティッカーシンボル。
        db:     CacheDB インスタンス。
        period: yfinance の period パラメータ。

    Returns:
        DataFrame。取得失敗時は None。
    """
    results = fetch_prices([ticker], db=db, period=period)
    return results.get(ticker)


# ============================================================
# 内部: キャッシュ判定
# ============================================================

def _separate_cached(
    tickers: list[str],
    db: CacheDB,
    cache_days: int | None = None,
    allow_stale: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """キャッシュが新鮮な銘柄をDBから読み込み、古い銘柄リストを返す。

    営業日ベースで鮮度を判定する:
    - 最新データ日が直近の営業日（=最新の取引日）以上であれば新鮮とみなす
    - 週末・祝日をまたいでも不要な再取得を回避する

    Args:
        cache_days:   DBから読み込む日数。None の場合は price_history_days。
        allow_stale:  True の場合、鮮度チェックをスキップしてDBにデータがあれば使用。
                      バックテスト用途向け（最新データでなくても歴史データとして有効）。

    Returns:
        (キャッシュ済み結果の辞書, 再取得が必要なティッカーリスト)
    """
    cached: dict[str, pd.DataFrame] = {}
    stale: list[str] = []
    latest_trading_day = _get_latest_trading_day()

    for ticker in tickers:
        latest_date_str = db.get_latest_price_date(ticker)
        if latest_date_str is None:
            stale.append(ticker)
            continue

        try:
            latest_date = datetime.strptime(latest_date_str, "%Y-%m-%d")
        except ValueError:
            stale.append(ticker)
            continue

        # 鮮度チェック: allow_stale=True の場合はスキップ（DBにデータがあれば使用）
        if not allow_stale and latest_date.date() < latest_trading_day:
            stale.append(ticker)
            continue

        # キャッシュから読み込み
        df = _load_from_cache(ticker, db, days=cache_days)
        if df is not None and not df.empty:
            cached[ticker] = df
        else:
            stale.append(ticker)

    return cached, stale


def _get_latest_trading_day() -> 'date':
    """直近の米国株式市場の営業日（取引日）を返す。

    簡易的に土日のみをスキップする（祝日は考慮しない）。
    市場がまだ開いていない時間帯（EST 16:00 以前）は前営業日を返す。
    """
    from datetime import date

    now = datetime.now()
    today = now.date()

    # 週末を巻き戻し: 日曜(6)→金曜, 土曜(5)→金曜
    weekday = today.weekday()
    if weekday == 6:  # Sunday
        today -= timedelta(days=2)
    elif weekday == 5:  # Saturday
        today -= timedelta(days=1)

    # 平日でも市場クローズ前（UTC 21:00 ≈ EST 16:00）なら前営業日
    # 日本時間だと翌朝6時頃なので、UTC基準で判定
    if weekday < 5 and now.hour < 6:
        today -= timedelta(days=1)
        # さらに週末チェック
        if today.weekday() == 6:
            today -= timedelta(days=2)
        elif today.weekday() == 5:
            today -= timedelta(days=1)

    return today


def _load_from_cache(ticker: str, db: CacheDB, days: int | None = None) -> pd.DataFrame | None:
    """SQLite キャッシュから株価データを DataFrame として読み込む。

    Args:
        days: 取得日数。None の場合は price_history_days を使用。
    """
    n_days = days if days is not None else TECHNICAL_PARAMS.price_history_days
    rows = db.get_prices(ticker, days=n_days)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    # ticker 列はDataFrame本体から除外（インデックスの一部にしない）
    df = df.drop(columns=["ticker"], errors="ignore")
    return df[["Open", "High", "Low", "Close", "Volume"]]


# ============================================================
# 内部: yfinance からの取得とDB保存
# ============================================================

def _fetch_and_store(
    tickers: list[str],
    db: CacheDB,
    period: str | None,
) -> dict[str, pd.DataFrame]:
    """yfinance からバッチ取得し、SQLite に保存して結果を返す。

    長期 period（1y 超）の場合はレート制限を避けるため逐次処理にする。
    各バッチ失敗時はリトライ（最大3回、指数バックオフ）を行う。
    """
    if period is None:
        period = f"{TECHNICAL_PARAMS.price_history_days}d"

    batch_size = FETCHER_CONFIG.price_batch_size
    max_workers = FETCHER_CONFIG.max_workers

    # 長期期間フェッチ（バックテスト用）はレート制限リスクが高いため逐次処理
    _long_periods = {"2y", "3y", "4y", "5y", "6y", "7y", "10y", "ytd", "max"}
    sequential = period in _long_periods

    if sequential:
        # バックテスト用: 小バッチ + バッチ間スリープ
        batch_size = min(batch_size, 20)
        logger.info(
            "長期期間 (%s) のため逐次バッチ取得モード（バッチサイズ=%d）",
            period, batch_size,
        )

    # ティッカーリストをバッチに分割
    batches = [
        tickers[i : i + batch_size]
        for i in range(0, len(tickers), batch_size)
    ]

    results: dict[str, pd.DataFrame] = {}

    if sequential:
        # 逐次処理: バッチ間にスリープを挟む
        for idx, batch in enumerate(batches):
            if idx > 0:
                time.sleep(2.0)  # バッチ間 2 秒待機
            batch_result = _download_batch_with_retry(batch, period)
            results.update(batch_result)
            logger.info(
                "バッチ %d/%d 完了: %d 銘柄取得",
                idx + 1, len(batches), len(batch_result),
            )
    else:
        # 通常処理: ThreadPoolExecutor で並列取得
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(_download_batch_with_retry, batch, period): batch
                for batch in batches
            }
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_result = future.result()
                    results.update(batch_result)
                except Exception:
                    logger.warning(
                        "バッチ取得失敗（%d 銘柄）— スキップ",
                        len(batch),
                        exc_info=True,
                    )

    # 取得成功分を一括で DB に保存
    _store_to_cache(results, db)

    return results


def _download_batch_with_retry(
    tickers: list[str],
    period: str,
    max_retries: int = 3,
) -> dict[str, pd.DataFrame]:
    """レート制限エラー時にリトライする _download_batch ラッパー。

    YFRateLimitError が発生した場合は指数バックオフ（10s, 30s, 90s）で再試行する。
    それ以外の例外はフォールバックとして個別ダウンロードを試みる。
    """
    delay = 10.0
    for attempt in range(max_retries):
        try:
            return _download_batch(tickers, period)
        except Exception as exc:
            exc_type = type(exc).__name__
            is_rate_limit = "RateLimit" in exc_type or "rate" in str(exc).lower()

            if is_rate_limit and attempt < max_retries - 1:
                logger.warning(
                    "レート制限検出 (attempt %d/%d): %ds 待機後リトライ",
                    attempt + 1, max_retries, int(delay),
                )
                time.sleep(delay)
                delay *= 3  # 10s → 30s → 90s
            else:
                logger.warning(
                    "バッチ取得失敗（%d 銘柄, attempt %d）— 個別フォールバック: %s",
                    len(tickers), attempt + 1, exc_type,
                )
                # フォールバック: 個別ダウンロード
                results: dict[str, pd.DataFrame] = {}
                for ticker in tickers:
                    df = _download_single_with_retry(ticker, period)
                    if df is not None:
                        results[ticker] = df
                    time.sleep(0.5)  # 個別取得間も少し待機
                return results

    return {}


def _download_single_with_retry(
    ticker: str,
    period: str,
    max_retries: int = 3,
) -> pd.DataFrame | None:
    """レート制限時にリトライする個別ダウンロード。"""
    delay = 5.0
    for attempt in range(max_retries):
        try:
            raw = yf.download(
                ticker,
                period=period,
                auto_adjust=True,
                progress=False,
            )
            return _clean_dataframe(raw)
        except Exception as exc:
            exc_type = type(exc).__name__
            is_rate_limit = "RateLimit" in exc_type or "rate" in str(exc).lower()
            if is_rate_limit and attempt < max_retries - 1:
                logger.warning("個別レート制限 %s (attempt %d): %ds 待機", ticker, attempt + 1, int(delay))
                time.sleep(delay)
                delay *= 3
            else:
                logger.debug("個別ダウンロード失敗: %s — %s", ticker, exc_type)
                return None
    return None


def _download_batch(
    tickers: list[str],
    period: str,
) -> dict[str, pd.DataFrame]:
    """yfinance.download で複数銘柄を一括取得する。

    Returns:
        ティッカー → DataFrame の辞書（取得失敗銘柄は含まない）。
    """
    ticker_str = " ".join(tickers)
    logger.debug("バッチダウンロード開始: %d 銘柄", len(tickers))

    raw = yf.download(
        ticker_str,
        period=period,
        group_by="ticker",
        auto_adjust=True,
        threads=False,  # 内部スレッドを無効化してレート制限を回避
        progress=False,
    )

    if raw.empty:
        logger.warning("バッチダウンロード結果が空: %s", ticker_str[:80])
        return {}

    results: dict[str, pd.DataFrame] = {}

    if len(tickers) == 1:
        # 1銘柄の場合、MultiIndex にならないので直接処理
        df = _clean_dataframe(raw)
        if df is not None:
            results[tickers[0]] = df
    else:
        # 複数銘柄: MultiIndex (ticker, OHLCV) 構造
        for ticker in tickers:
            try:
                if ticker not in raw.columns.get_level_values(0):
                    logger.debug("銘柄 %s のデータがバッチ結果に存在しない", ticker)
                    continue
                ticker_df = raw[ticker].copy()
                df = _clean_dataframe(ticker_df)
                if df is not None:
                    results[ticker] = df
            except (KeyError, TypeError):
                logger.debug("銘柄 %s のデータ抽出に失敗", ticker, exc_info=True)

    logger.debug("バッチ完了: %d / %d 銘柄取得成功", len(results), len(tickers))
    return results


def _download_single(ticker: str, period: str) -> pd.DataFrame | None:
    """1銘柄を個別にダウンロードする（フォールバック用）。"""
    logger.debug("個別ダウンロード: %s", ticker)
    return _download_single_with_retry(ticker, period)


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame | None:
    """取得した DataFrame をクリーンアップする。

    - NaN 行を除去
    - 必要カラムの存在確認
    - カラム名を統一 (Open, High, Low, Close, Volume)

    Returns:
        クリーン済み DataFrame。データが空の場合は None。
    """
    if df is None or df.empty:
        return None

    # MultiIndex columns を平坦化（yfinance v0.2.x+ 対応）
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # カラム名の統一（大文字始まりに正規化）
    rename_map: dict[str, str] = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower == "open":
            rename_map[col] = "Open"
        elif col_lower == "high":
            rename_map[col] = "High"
        elif col_lower == "low":
            rename_map[col] = "Low"
        elif col_lower == "close":
            rename_map[col] = "Close"
        elif col_lower == "volume":
            rename_map[col] = "Volume"
    if rename_map:
        df = df.rename(columns=rename_map)

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        logger.debug("必要カラム不足: %s", missing)
        return None

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if df.empty:
        return None

    return df


# ============================================================
# 内部: DB 保存
# ============================================================

def _store_to_cache(
    results: dict[str, pd.DataFrame],
    db: CacheDB,
) -> None:
    """取得した全銘柄の株価データを SQLite に一括保存する。"""
    bulk_rows: list[tuple[str, str, float, float, float, float, int]] = []

    for ticker, df in results.items():
        for dt, row in df.iterrows():
            date_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
            try:
                bulk_rows.append((
                    ticker,
                    date_str,
                    float(row["Open"]),
                    float(row["High"]),
                    float(row["Low"]),
                    float(row["Close"]),
                    int(row["Volume"]),
                ))
            except (ValueError, TypeError):
                logger.debug("行データ変換失敗: %s %s", ticker, date_str)

    if bulk_rows:
        db.upsert_prices_bulk(bulk_rows)
        logger.info("SQLite に %d 行を保存", len(bulk_rows))
