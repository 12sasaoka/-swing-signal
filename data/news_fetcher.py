"""
Swing Trade Signal System — ニュースヘッドライン取得

yfinance の .news を使用して各銘柄の直近ニュースヘッドラインを取得する。
Claude API 分析およびキーワードセンチメント分析の入力データとなる。

取得上限: 銘柄あたり最大10件（FETCHER_CONFIG.max_news_articles）
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import yfinance as yf

from config.settings import FETCHER_CONFIG

logger = logging.getLogger(__name__)


# ============================================================
# 公開 API
# ============================================================

def fetch_news(
    tickers: list[str],
) -> dict[str, list[str]]:
    """複数銘柄のニュースヘッドラインを並列取得する。

    Args:
        tickers: ティッカーリスト。

    Returns:
        ティッカー → ヘッドラインリスト の辞書。
        取得失敗またはニュースなしの銘柄は空リスト。
    """
    if not tickers:
        return {}

    results: dict[str, list[str]] = {}
    max_workers = min(FETCHER_CONFIG.max_workers, len(tickers))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(fetch_news_single, t): t
            for t in tickers
        }
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                results[ticker] = future.result()
            except Exception:
                logger.debug("ニュース取得例外: %s", ticker, exc_info=True)
                results[ticker] = []

    total_headlines = sum(len(v) for v in results.values())
    logger.info(
        "ニュース取得完了: %d 銘柄, 合計 %d 件",
        len(results), total_headlines,
    )
    return results


def fetch_news_single(ticker: str) -> list[str]:
    """1銘柄のニュースヘッドラインを取得する。

    Args:
        ticker: ティッカーシンボル。

    Returns:
        ヘッドラインのリスト（最大 max_news_articles 件）。
        取得失敗時は空リスト。
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        news_list: list[dict[str, Any]] = yf_ticker.news or []
    except Exception:
        logger.debug("yfinance .news 取得失敗: %s", ticker, exc_info=True)
        return []

    if not news_list:
        return []

    headlines: list[str] = []
    max_articles = FETCHER_CONFIG.max_news_articles

    for article in news_list[:max_articles]:
        title = article.get("title") or article.get("headline", "")
        if title and isinstance(title, str):
            headlines.append(title.strip())

    logger.debug("ニュース: %s → %d 件", ticker, len(headlines))
    return headlines
