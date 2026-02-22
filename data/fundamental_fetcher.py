"""
Swing Trade Signal System — ファンダメンタルデータ取得

yfinance の .info を使用して各銘柄のファンダメンタル指標を取得し、
JSON 形式で SQLite にキャッシュする。

取得指標:
  - trailing_pe, forward_pe      (PER)
  - price_to_book                (PBR)
  - return_on_equity             (ROE)
  - ev_to_ebitda                 (EV/EBITDA)
  - revenue_growth               (売上成長率)
  - debt_to_equity               (負債比率)
  - current_ratio                (流動比率)
  - market_cap                   (時価総額)
  - free_cashflow                (フリーキャッシュフロー)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import yfinance as yf

from config.settings import FETCHER_CONFIG
from data.cache import CacheDB

logger = logging.getLogger(__name__)


# ============================================================
# yfinance キー → 内部キー のマッピング
# ============================================================

_YF_KEY_MAP: dict[str, str] = {
    "trailingPE":          "trailing_pe",
    "forwardPE":           "forward_pe",
    "priceToBook":         "price_to_book",
    "returnOnEquity":      "return_on_equity",
    "enterpriseToEbitda":  "ev_to_ebitda",
    "revenueGrowth":       "revenue_growth",
    "earningsGrowth":      "earnings_growth",  # ← ★追加
    "debtToEquity":        "debt_to_equity",
    "currentRatio":        "current_ratio",
    "marketCap":           "market_cap",
    "freeCashflow":        "free_cashflow",
}

# yfinance に存在しない場合のデフォルト値
_DEFAULTS: dict[str, float | None] = {
    "trailing_pe":      None,
    "forward_pe":       None,
    "price_to_book":    None,
    "return_on_equity": 0.0,
    "ev_to_ebitda":     None,
    "revenue_growth":   0.0,
    "earnings_growth":  0.0,  # ← ★追加（これを忘れるとエラーになります！）
    "debt_to_equity":   0.0,
    "current_ratio":    0.0,
    "market_cap":       None,
    "free_cashflow":        None,
}

# ============================================================
# 公開 API
# ============================================================

def fetch_fundamentals(
    tickers: list[str],
    db: CacheDB | None = None,
    force_refresh: bool = False,
) -> dict[str, dict[str, Any]]:
    """指定銘柄のファンダメンタルデータを取得する。

    キャッシュが有効（デフォルト7日）であればDBから読み込み、
    古ければ yfinance から再取得する。

    Args:
        tickers:       取得対象のティッカーリスト。
        db:            CacheDB インスタンス。None の場合は新規作成。
        force_refresh: True の場合、キャッシュを無視して全銘柄を再取得。

    Returns:
        ティッカー → ファンダメンタル指標辞書 の辞書。
        取得失敗した銘柄は辞書に含まれない。
    """
    if not tickers:
        return {}

    if db is None:
        db = CacheDB()
        db.initialize()

    results: dict[str, dict[str, Any]] = {}
    fetch_needed: list[str] = []

    # キャッシュチェック
    if not force_refresh:
        for ticker in tickers:
            cached = db.get_fundamentals(ticker)
            if cached is not None:
                results[ticker] = cached
            else:
                fetch_needed.append(ticker)
    else:
        fetch_needed = list(tickers)

    if results:
        logger.info(
            "キャッシュから %d 銘柄のファンダメンタルを読み込み",
            len(results),
        )

    # yfinance から取得
    if fetch_needed:
        logger.info("yfinance から %d 銘柄のファンダメンタルを取得開始", len(fetch_needed))
        fetched = _fetch_parallel(fetch_needed)

        # DB に保存
        for ticker, data in fetched.items():
            db.upsert_fundamentals(ticker, data)

        results.update(fetched)
        logger.info(
            "ファンダメンタル取得完了: 成功 %d / 要取得 %d",
            len(fetched),
            len(fetch_needed),
        )

    logger.info(
        "ファンダメンタル総計: 成功 %d / 全体 %d",
        len(results),
        len(tickers),
    )
    return results


def fetch_fundamental_single(
    ticker: str,
    db: CacheDB | None = None,
    force_refresh: bool = False,
) -> dict[str, Any] | None:
    """1銘柄のファンダメンタルデータを取得する（便利ラッパー）。

    Args:
        ticker:        ティッカーシンボル。
        db:            CacheDB インスタンス。
        force_refresh: True の場合、キャッシュを無視。

    Returns:
        ファンダメンタル指標辞書。取得失敗時は None。
    """
    results = fetch_fundamentals([ticker], db=db, force_refresh=force_refresh)
    return results.get(ticker)


# ============================================================
# 内部: 並列取得
# ============================================================

def _fetch_parallel(
    tickers: list[str],
) -> dict[str, dict[str, Any]]:
    """ThreadPoolExecutor で複数銘柄を並列取得する。"""
    results: dict[str, dict[str, Any]] = {}
    max_workers = min(FETCHER_CONFIG.max_workers, len(tickers))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(_fetch_single_info, ticker): ticker
            for ticker in tickers
        }
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                data = future.result()
                if data is not None:
                    results[ticker] = data
            except Exception:
                logger.error(
                    "ファンダメンタル取得で予期せぬエラー: %s",
                    ticker,
                    exc_info=True,
                )

    return results


def _fetch_single_info(ticker: str) -> dict[str, Any] | None:
    """1銘柄のファンダメンタルデータを yfinance から取得する。

    Args:
        ticker: ティッカーシンボル。

    Returns:
        正規化されたファンダメンタル指標辞書。取得失敗時は None。
    """
    logger.debug("ファンダメンタル取得: %s", ticker)
    try:
        yf_ticker = yf.Ticker(ticker)
        info: dict[str, Any] = yf_ticker.info or {}
    except Exception:
        logger.warning("yfinance .info 取得失敗: %s", ticker, exc_info=True)
        return None

    if not info:
        logger.warning("yfinance .info が空: %s", ticker)
        return None

    # yfinance キー → 内部キーに変換し、欠損値にデフォルトを適用
    data = _extract_and_normalize(info)

    # 最低限のバリデーション: market_cap が取れていれば有効とみなす
    if data.get("market_cap") is None:
        logger.warning(
            "market_cap が取得できず、データ無効とみなす: %s",
            ticker,
        )
        return None

    logger.debug(
        "ファンダメンタル取得成功: %s (PE=%.1f, ROE=%.2f, MCap=%s)",
        ticker,
        data.get("trailing_pe") or 0.0,
        data.get("return_on_equity") or 0.0,
        _format_market_cap(data.get("market_cap")),
    )
    return data


# ============================================================
# 内部: データ正規化
# ============================================================

def _extract_and_normalize(info: dict[str, Any]) -> dict[str, Any]:
    """yfinance の info 辞書から必要な指標を抽出・正規化する。

    - 存在しないキーにはデフォルト値を適用
    - 数値でない値（文字列の "Infinity" 等）を除去
    - debt_to_equity は yfinance がパーセント表記（例: 150.0 = 150%）で返すので
      小数表記（1.5）に変換する
    """
    data: dict[str, Any] = {}

    for yf_key, internal_key in _YF_KEY_MAP.items():
        raw_value = info.get(yf_key)
        default = _DEFAULTS[internal_key]

        if raw_value is None:
            data[internal_key] = default
            continue

        # 数値変換を試みる
        cleaned = _to_float(raw_value)
        if cleaned is None:
            data[internal_key] = default
        else:
            data[internal_key] = cleaned

    # debt_to_equity の単位変換（% → 小数）
    if data.get("debt_to_equity") is not None:
        data["debt_to_equity"] = data["debt_to_equity"] / 100.0

    return data


def _to_float(value: Any) -> float | None:
    """値を float に変換する。変換不可の場合は None。

    yfinance が返す特殊値（Infinity, NaN, 文字列）にも対処する。
    """
    if value is None:
        return None

    try:
        f = float(value)
    except (ValueError, TypeError):
        return None

    # NaN / Infinity を除去
    if f != f or f == float("inf") or f == float("-inf"):
        return None

    return f


# ============================================================
# ユーティリティ
# ============================================================

def _format_market_cap(value: float | None) -> str:
    """時価総額を読みやすい文字列にフォーマットする（ログ用）。"""
    if value is None:
        return "N/A"
    if value >= 1e12:
        return f"${value / 1e12:.1f}T"
    if value >= 1e9:
        return f"${value / 1e9:.1f}B"
    if value >= 1e6:
        return f"${value / 1e6:.0f}M"
    return f"${value:,.0f}"


def get_fundamental_keys() -> list[str]:
    """取得されるファンダメンタル指標のキー一覧を返す。

    Returns:
        内部キー名のリスト。
    """
    return list(_YF_KEY_MAP.values())
