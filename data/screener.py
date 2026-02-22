"""
Swing Trade Signal System — Russell 3000 スクリーニング

IWV ETF (iShares Russell 3000 ETF) の構成銘柄を取得し、
3段階のフィルタリングで有望な銘柄候補を絞り込む。

パイプライン:
  1. IWV 構成銘柄をiShares公式CSVから取得 (~2500銘柄)
  2. Tier 1: ファンダメンタル事前フィルタ (時価総額・流動性・株価)
  3. Tier 2: テクニカルフィルタ (モメンタムスコア上位)
"""

from __future__ import annotations

import csv
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from typing import Any

import requests
import yfinance as yf

from config.screening import SCREENING_PARAMS
from data.cache import CacheDB

logger = logging.getLogger(__name__)

# iShares IWV holdings CSV URL
_IWV_CSV_URL = (
    "https://www.ishares.com/us/products/239714/"
    "ishares-russell-3000-etf/1467271812596.ajax"
    "?fileType=csv&fileName=IWV_holdings&dataType=fund"
)

# iSharesのCSVではティッカーのハイフンが削除される場合がある
# 既知の変換マッピング
_TICKER_FIXUP: dict[str, str] = {
    "BRKB": "BRK-B",
}

# 有効なティッカーのパターン (1-5文字のアルファベット、ハイフン付きも許容)
_VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}(-[A-Z]{1,2})?$")


# ============================================================
# 公開 API
# ============================================================

def run_screening(db: CacheDB) -> list[str]:
    """Russell 3000 スクリーニングを実行し、候補銘柄リストを返す。

    Args:
        db: CacheDB インスタンス。

    Returns:
        Tier2 を通過した銘柄のティッカーリスト（モメンタムスコア降順）。
    """
    # Step 1: IWV 構成銘柄を取得
    print("  [Screen 1/3] IWV 構成銘柄を取得中...")
    all_tickers = fetch_iwv_holdings(db)
    print(f"    → {len(all_tickers)} 銘柄を取得")

    if not all_tickers:
        logger.error("IWV 構成銘柄の取得に失敗しました")
        return []

    # Step 2: Tier 1 — ファンダメンタルフィルタ
    print("  [Screen 2/3] Tier 1: ファンダメンタルフィルタ中...")
    tier1_pass = screen_tier1(all_tickers, db)
    print(f"    → {len(tier1_pass)} 銘柄が通過 (時価総額≥${SCREENING_PARAMS.min_market_cap/1e9:.0f}B, 出来高≥{SCREENING_PARAMS.min_avg_volume:,})")

    if not tier1_pass:
        logger.warning("Tier 1 を通過した銘柄がありません")
        return []

    # Step 3: Tier 2 — テクニカルフィルタ
    print("  [Screen 3/3] Tier 2: モメンタムフィルタ中...")
    tier2_pass = screen_tier2(tier1_pass, db)
    print(f"    → {len(tier2_pass)} 銘柄が通過 (モメンタム≥{SCREENING_PARAMS.min_momentum_score}, 上限{SCREENING_PARAMS.max_screen_candidates})")

    return tier2_pass


# ============================================================
# IWV 構成銘柄取得
# ============================================================

def fetch_iwv_holdings(db: CacheDB) -> list[str]:
    """IWV ETF の構成銘柄リストを取得する。

    キャッシュが有効であればDBから読み込み、
    期限切れの場合はiShares公式CSVから再取得する。

    Args:
        db: CacheDB インスタンス。

    Returns:
        ティッカーシンボルのリスト。
    """
    # キャッシュチェック
    cached = db.get_screening_cache(
        "iwv_holdings",
        max_age_days=SCREENING_PARAMS.holdings_cache_days,
    )
    if cached:
        logger.info("IWV 構成銘柄をキャッシュから読み込み: %d 銘柄", len(cached))
        return cached

    # iShares CSV からダウンロード
    tickers = _download_iwv_csv()

    if tickers:
        db.upsert_screening_cache("iwv_holdings", tickers)
        logger.info("IWV 構成銘柄を取得・キャッシュ保存: %d 銘柄", len(tickers))

    return tickers


def _download_iwv_csv() -> list[str]:
    """iShares 公式サイトから IWV 構成銘柄の CSV をダウンロード・パースする。

    Returns:
        有効なティッカーシンボルのリスト。
    """
    logger.info("iShares CSV ダウンロード開始: IWV")
    try:
        resp = requests.get(
            _IWV_CSV_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=60,
        )
        resp.raise_for_status()
    except requests.RequestException:
        logger.error("iShares CSV ダウンロード失敗", exc_info=True)
        return []

    return _parse_ishares_csv(resp.text)


def _parse_ishares_csv(text: str) -> list[str]:
    """iShares CSV テキストをパースしてティッカーリストを抽出する。

    CSV は先頭にメタデータ行があり、'Ticker,...' 行がヘッダーになる。

    Args:
        text: CSV テキスト全文。

    Returns:
        有効なティッカーシンボルのリスト。
    """
    lines = text.split("\n")

    # ヘッダー行を探す
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Ticker,"):
            header_idx = i
            break

    if header_idx is None:
        logger.error("iShares CSV のヘッダー行が見つかりません")
        return []

    data_text = "\n".join(lines[header_idx:])
    reader = csv.DictReader(StringIO(data_text))

    tickers: list[str] = []
    for row in reader:
        ticker = (row.get("Ticker") or "").strip('"').strip()
        asset_class = (row.get("Asset Class") or "").strip('"').strip()

        if not ticker or asset_class != "Equity":
            continue

        # ティッカー修正
        ticker = _TICKER_FIXUP.get(ticker, ticker)

        # 有効性チェック（英字1-5文字）
        if _VALID_TICKER_RE.match(ticker):
            tickers.append(ticker)

    logger.info("iShares CSV パース完了: %d 銘柄", len(tickers))
    return tickers


# ============================================================
# Tier 1: ファンダメンタル事前フィルタ
# ============================================================

def screen_tier1(tickers: list[str], db: CacheDB) -> list[str]:
    """Tier 1: 時価総額・流動性・株価でフィルタリングする。

    yfinance の fast_info を使って高速にスクリーニングする。

    Args:
        tickers: スクリーニング対象のティッカーリスト。
        db:      CacheDB インスタンス。

    Returns:
        フィルタを通過したティッカーリスト。
    """
    # キャッシュチェック
    cached = db.get_screening_cache(
        "tier1_pass",
        max_age_days=SCREENING_PARAMS.tier1_cache_days,
    )
    if cached:
        logger.info("Tier1 結果をキャッシュから読み込み: %d 銘柄", len(cached))
        return cached

    passed: list[str] = []
    failed = 0
    max_workers = min(SCREENING_PARAMS.tier1_max_workers, len(tickers))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(_check_tier1, ticker): ticker
            for ticker in tickers
        }
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                if future.result():
                    passed.append(ticker)
                else:
                    failed += 1
            except Exception:
                failed += 1
                logger.debug("Tier1 チェック失敗: %s", ticker, exc_info=True)

    logger.info(
        "Tier1 完了: 通過 %d / 不合格 %d / 全体 %d",
        len(passed), failed, len(tickers),
    )

    # キャッシュ保存
    if passed:
        db.upsert_screening_cache("tier1_pass", sorted(passed))

    return sorted(passed)


def _check_tier1(ticker: str) -> bool:
    """1銘柄の Tier1 フィルタチェックを行う。

    yfinance の fast_info を使って時価総額・出来高・株価を取得する。
    fast_info は .info より高速（HTTP リクエストが少ない）。

    Args:
        ticker: ティッカーシンボル。

    Returns:
        フィルタを通過したら True。
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        fi = yf_ticker.fast_info
    except Exception:
        return False

    try:
        market_cap = getattr(fi, "market_cap", None)
        if market_cap is None or market_cap < SCREENING_PARAMS.min_market_cap:
            return False

        last_price = getattr(fi, "last_price", None)
        if last_price is None or last_price < SCREENING_PARAMS.min_price:
            return False

        # fast_info には three_month_average_volume がある
        avg_volume = getattr(fi, "three_month_average_volume", None)
        if avg_volume is None or avg_volume < SCREENING_PARAMS.min_avg_volume:
            return False

    except Exception:
        return False

    return True


# ============================================================
# Tier 2: テクニカルフィルタ
# ============================================================

def screen_tier2(tickers: list[str], db: CacheDB) -> list[str]:
    """Tier 2: モメンタムスコアでフィルタリングする。

    既存の price_fetcher + momentum モジュールを再利用する。

    Args:
        tickers: Tier1通過済みのティッカーリスト。
        db:      CacheDB インスタンス。

    Returns:
        モメンタムスコア上位のティッカーリスト（スコア降順）。
    """
    from analysis.momentum import calc_momentum_score
    from data.price_fetcher import fetch_prices

    # バッチ株価取得
    price_data = fetch_prices(tickers, db=db)

    # モメンタムスコア計算
    scored: list[tuple[str, float]] = []
    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue
        score = calc_momentum_score(df)
        if score >= SCREENING_PARAMS.min_momentum_score:
            scored.append((ticker, score))

    # スコア降順でソート
    scored.sort(key=lambda x: x[1], reverse=True)

    # 上限数で切り詰め
    top = scored[: SCREENING_PARAMS.max_screen_candidates]

    logger.info(
        "Tier2 完了: 通過 %d / モメンタム算出 %d / 全体 %d",
        len(top), len(scored), len(tickers),
    )

    return [ticker for ticker, _ in top]
