"""
Swing Trade Signal System — 四半期ファンダメンタル取得（時系列対応）

yfinance の quarterly_financials / quarterly_balance_sheet / earnings_dates を使用して
決算期ごとの財務指標と収益サプライズを取得する。

バックテスト時は「その日時点で公表済みの最新四半期データ」を使用することで
ルックアヘッドバイアスを排除する。

保存形式:
  {
    "quarters": {
      "2023-09-30": {
        "return_on_equity": 0.35,
        "debt_to_equity":   1.20,
        "revenue_growth":   0.08,
        "current_ratio":    1.45,
        "earnings_growth":  0.12,
      },
      ...
    },
    "surprise": {
      "2023-11-02": 4.5,   # +4.5% beat
      "2024-02-01": -2.1,  # -2.1% miss
      ...
    }
  }

注意:
  - 決算発表ラグ: 四半期末から45日後を「公表日」とみなす（保守的な仮定）
  - surprise は earnings_dates の Surprise(%) または (Actual - Estimate) / |Estimate| * 100
  - yfinance が提供する quarterly_* データは直近4〜5年分のみ
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import FETCHER_CONFIG
from data.cache import CacheDB

logger = logging.getLogger(__name__)

# キャッシュ有効期間（日数）— 過去データは変わらないため長めに設定
_CACHE_DAYS = 30


# ============================================================
# 公開 API
# ============================================================

def fetch_quarterly_data(
    tickers: list[str],
    db: CacheDB | None = None,
    force_refresh: bool = False,
) -> dict[str, dict]:
    """複数銘柄の四半期財務データを取得する。

    Args:
        tickers:       取得対象ティッカーリスト。
        db:            CacheDB インスタンス。None の場合は新規作成。
        force_refresh: True の場合、キャッシュを無視して再取得。

    Returns:
        {ticker: {"quarters": {date_str: fund_dict}, "surprise": {date_str: float}}}
    """
    if not tickers:
        return {}

    if db is None:
        db = CacheDB()
        db.initialize()

    results: dict[str, dict] = {}
    fetch_needed: list[str] = []

    if not force_refresh:
        for ticker in tickers:
            cached = db.get_quarterly_data(ticker)
            if cached is not None:
                results[ticker] = cached
            else:
                fetch_needed.append(ticker)
    else:
        fetch_needed = list(tickers)

    if results:
        logger.info("キャッシュから %d 銘柄の四半期データを読み込み", len(results))

    if fetch_needed:
        logger.info("yfinance から %d 銘柄の四半期データを取得開始", len(fetch_needed))
        max_workers = min(FETCHER_CONFIG.max_workers, len(fetch_needed), 20)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(_fetch_single_quarterly, t): t
                for t in fetch_needed
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data is not None:
                        db.upsert_quarterly_data(ticker, data)
                        results[ticker] = data
                except Exception:
                    logger.debug("四半期データ取得失敗: %s", ticker, exc_info=True)

        fetched_count = sum(1 for t in fetch_needed if t in results)
        logger.info(
            "四半期データ取得完了: 成功 %d / 要取得 %d",
            fetched_count, len(fetch_needed),
        )

    return results


def get_fundamentals_as_of(
    ticker: str,
    as_of_date: str,
    quarterly_data: dict[str, dict],
) -> dict[str, Any] | None:
    """指定日時点で公表済みの最新四半期ファンダメンタルを返す。

    決算発表の遅延を考慮して、四半期末日から45日後を公表日と仮定する。
    （例: 9月末決算 → 11月14日以降に公表）

    Args:
        ticker:         ティッカーシンボル。
        as_of_date:     評価基準日 (YYYY-MM-DD)。
        quarterly_data: fetch_quarterly_data の戻り値。

    Returns:
        その日時点で使用可能な最新の財務指標辞書。データなしは None。
    """
    ticker_data = quarterly_data.get(ticker)
    if not ticker_data:
        return None

    quarters = ticker_data.get("quarters", {})
    if not quarters:
        return None

    # 四半期末 + 45日 <= as_of_date を満たす最新の四半期を探す
    available: dict[str, dict] = {}
    for quarter_end, fund_dict in quarters.items():
        try:
            publish_date = (
                pd.Timestamp(quarter_end) + pd.Timedelta(days=45)
            ).strftime("%Y-%m-%d")
            if publish_date <= as_of_date:
                available[quarter_end] = fund_dict
        except Exception:
            continue

    if not available:
        return None

    latest = max(available.keys())
    return available[latest]


def get_earnings_surprise_as_of(
    ticker: str,
    as_of_date: str,
    quarterly_data: dict[str, dict],
) -> float | None:
    """指定日時点での最新収益サプライズスコアを返す。

    Args:
        ticker:         ティッカーシンボル。
        as_of_date:     評価基準日 (YYYY-MM-DD)。
        quarterly_data: fetch_quarterly_data の戻り値。

    Returns:
        サプライズ % (例: +5.0 = 5% beat)。データなしは None。
        (+10% = スコア +1.0、-10% = スコア -1.0 に正規化済み)
    """
    ticker_data = quarterly_data.get(ticker)
    if not ticker_data:
        return None

    surprise_map = ticker_data.get("surprise", {})
    if not surprise_map:
        return None

    # 基準日以前の最新サプライズを取得
    available = {d: v for d, v in surprise_map.items() if d <= as_of_date}
    if not available:
        return None

    latest_date = max(available.keys())
    raw_pct = available[latest_date]  # 例: +5.0 = 5% beat

    # 10% beat/miss = max/min スコアに正規化してクリップ
    score = float(np.clip(raw_pct / 10.0, -1.0, 1.0))
    return score


# ============================================================
# 内部: 1銘柄の四半期データ取得
# ============================================================

def _fetch_single_quarterly(ticker: str) -> dict | None:
    """1銘柄の四半期財務データをyfinanceから取得して変換する。"""
    try:
        t = yf.Ticker(ticker)

        qf = _safe_df(t, "quarterly_financials")
        qb = _safe_df(t, "quarterly_balance_sheet")

        quarters = _build_quarterly_metrics(qf, qb)
        surprise = _build_earnings_surprise(t)

        if not quarters and not surprise:
            return None

        return {"quarters": quarters, "surprise": surprise}

    except Exception:
        logger.debug("四半期データ取得エラー: %s", ticker, exc_info=True)
        return None


def _safe_df(ticker_obj: yf.Ticker, attr: str) -> pd.DataFrame | None:
    """yfinance から DataFrame を安全に取得する。"""
    try:
        df = getattr(ticker_obj, attr)
        if df is None or (hasattr(df, "empty") and df.empty):
            return None
        return df
    except Exception:
        return None


def _build_quarterly_metrics(
    qf: pd.DataFrame | None,
    qb: pd.DataFrame | None,
) -> dict[str, dict[str, Any]]:
    """四半期財務指標を決算期ごとにまとめる。

    Returns:
        {quarter_end_date: fund_dict}
        fund_dict のキーは calc_quality_score() が期待するフォーマットと同じ
    """
    if qf is None and qb is None:
        return {}

    # 対象期の集合（income + balance sheet の和集合）
    dates: set[pd.Timestamp] = set()
    if qf is not None:
        dates.update(qf.columns)
    if qb is not None:
        dates.update(qb.columns)

    if not dates:
        return {}

    # 収入・EPS の時系列（YoY 成長率計算用）
    revenue_series: dict[pd.Timestamp, float] = {}
    eps_series: dict[pd.Timestamp, float] = {}

    if qf is not None:
        for col in qf.columns:
            rev = _pick(qf, col, ["Total Revenue", "Operating Revenue"])
            if rev is not None:
                revenue_series[col] = rev
            eps = _pick(qf, col, ["Diluted EPS", "Basic EPS"])
            if eps is not None:
                eps_series[col] = eps

    sorted_dates = sorted(dates)
    quarters: dict[str, dict] = {}

    for col in sorted_dates:
        date_str = col.strftime("%Y-%m-%d")
        fund: dict[str, Any] = {}

        # ---- ROE = Net Income (annualized) / Equity ----
        net_income = None
        equity = None
        if qf is not None:
            net_income = _pick(qf, col, [
                "Net Income", "Net Income Common Stockholders",
                "Net Income From Continuing Operation Net Minority Interest",
            ])
        if qb is not None:
            equity = _pick(qb, col, [
                "Common Stock Equity", "Stockholders Equity",
                "Total Equity Gross Minority Interest",
            ])
        if net_income is not None and equity is not None and equity != 0:
            fund["return_on_equity"] = float(net_income * 4 / abs(equity))
        else:
            fund["return_on_equity"] = None

        # ---- Gross Profitability = Gross Profit / Total Assets ----
        # Novy-Marx (2013): 収益性ファクター
        gross_profit = None
        total_assets = None
        if qf is not None:
            gross_profit = _pick(qf, col, ["Gross Profit"])
        if qb is not None:
            total_assets = _pick(qb, col, ["Total Assets"])
        if gross_profit is not None and total_assets is not None and total_assets != 0:
            fund["gross_profitability"] = float(gross_profit / abs(total_assets))
        else:
            fund["gross_profitability"] = None

        # ---- Revenue Growth (YoY: 4四半期前と比較) ----
        rev_now = revenue_series.get(col)
        if rev_now is not None:
            sorted_rev = sorted(revenue_series.keys())
            idx = sorted_rev.index(col) if col in sorted_rev else -1
            if idx >= 4:
                rev_yago = revenue_series[sorted_rev[idx - 4]]
                fund["revenue_growth"] = float((rev_now - rev_yago) / abs(rev_yago)) if rev_yago else None
            else:
                fund["revenue_growth"] = None
        else:
            fund["revenue_growth"] = None

        # ---- Current Ratio = Current Assets / Current Liabilities ----
        cur_assets = cur_liab = None
        if qb is not None:
            cur_assets = _pick(qb, col, ["Current Assets"])
            cur_liab = _pick(qb, col, ["Current Liabilities"])
        if cur_assets is not None and cur_liab is not None and cur_liab != 0:
            fund["current_ratio"] = float(cur_assets / cur_liab)
        else:
            fund["current_ratio"] = None

        # ---- Earnings Growth (EPS YoY) ----
        eps_now = eps_series.get(col)
        if eps_now is not None:
            sorted_eps = sorted(eps_series.keys())
            idx = sorted_eps.index(col) if col in sorted_eps else -1
            if idx >= 4:
                eps_yago = eps_series[sorted_eps[idx - 4]]
                if eps_yago is not None and eps_yago > 0:
                    fund["earnings_growth"] = float((eps_now - eps_yago) / abs(eps_yago))
                else:
                    fund["earnings_growth"] = None
            else:
                fund["earnings_growth"] = None
        else:
            fund["earnings_growth"] = None

        # データが1つでもあれば保存
        if any(v is not None for v in fund.values()):
            quarters[date_str] = fund

    return quarters


def _build_earnings_surprise(ticker_obj: yf.Ticker) -> dict[str, float]:
    """収益サプライズを取得する。

    yfinance earnings_dates から Surprise(%) または (Actual - Estimate)/|Estimate|*100 を取得。

    Returns:
        {announcement_date_str: surprise_pct}
    """
    surprise: dict[str, float] = {}

    try:
        ed = ticker_obj.earnings_dates
        if ed is None or (hasattr(ed, "empty") and ed.empty):
            return surprise

        # カラム名を小文字で検索
        col_lower = {str(c).lower(): c for c in ed.columns}
        est_col = next((col_lower[k] for k in col_lower if "estimate" in k), None)
        act_col = next(
            (col_lower[k] for k in col_lower if "reported" in k or "actual" in k),
            None,
        )
        surp_col = next(
            (col_lower[k] for k in col_lower if "surprise" in k and "%" in k),
            None,
        )

        for idx, row in ed.iterrows():
            try:
                date_str = pd.Timestamp(idx).strftime("%Y-%m-%d")

                # Surprise(%) カラムが直接あればそれを使う
                if surp_col is not None:
                    val = row[surp_col]
                    if pd.notna(val):
                        surprise[date_str] = float(val)
                        continue

                # Estimate / Actual から計算
                if est_col is not None and act_col is not None:
                    est = row[est_col]
                    act = row[act_col]
                    if pd.notna(est) and pd.notna(act) and float(est) != 0:
                        pct = (float(act) - float(est)) / abs(float(est)) * 100
                        surprise[date_str] = pct
            except Exception:
                continue

    except Exception:
        logger.debug("収益サプライズ取得失敗", exc_info=True)

    return surprise


def _pick(
    df: pd.DataFrame,
    col: pd.Timestamp,
    row_names: list[str],
) -> float | None:
    """DataFrame から指定カラム・候補行名の値を安全に取得する。"""
    if col not in df.columns:
        return None
    for name in row_names:
        if name in df.index:
            val = df.loc[name, col]
            if pd.notna(val):
                try:
                    f = float(val)
                    if f == f and f not in (float("inf"), float("-inf")):
                        return f
                except (ValueError, TypeError):
                    pass
    return None
