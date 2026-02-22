"""
Swing Trade Signal System — 決算日フェッチャー

各銘柄の決算（Earnings）日を yfinance から取得し、JSONファイルでキャッシュする。
バックテスト・ライブトレードで以下のルールに使用する:
  - 決算前 48時間（2カレンダー日）以内: 新規エントリー禁止
  - 決算前 3営業日（≈ 5カレンダー日）以内: 保有ポジションを強制決済

キャッシュ仕様:
  - 保存先: data/db/earnings_cache.json
  - 有効期間: 7日間（バックテスト用途のため長め）
  - 構造: { "TICKER": { "fetched_at": "ISO datetime", "dates": ["YYYY-MM-DD", ...] } }
"""

from __future__ import annotations

import json
import logging
from bisect import bisect_left
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# デフォルトキャッシュパス
_DEFAULT_CACHE: Path = Path(__file__).resolve().parent.parent / "data" / "db" / "earnings_cache.json"
_CACHE_DAYS: int = 7   # キャッシュ有効期間（日）
_LIMIT: int = 30       # 取得する決算日の最大件数（≈ 7.5年分 = 30四半期）


# ============================================================
# 公開 API
# ============================================================

def fetch_earnings_dates(
    tickers: list[str],
    cache_path: Path | str | None = None,
    force_refresh: bool = False,
) -> dict[str, list[str]]:
    """各銘柄の決算日リストを取得する。

    yfinance の earnings_dates を使用し、JSONファイルでキャッシュする。
    取得失敗した銘柄は空リストとして扱い、エラーは抑制する。

    Args:
        tickers:       ティッカーリスト。
        cache_path:    キャッシュJSONパス（None = デフォルト）。
        force_refresh: True の場合キャッシュを無視して強制再取得。

    Returns:
        ticker → 決算日文字列リスト（YYYY-MM-DD、昇順）の辞書。
        決済日が取得できなかった銘柄は空リスト。
    """
    path = Path(cache_path) if cache_path else _DEFAULT_CACHE

    # キャッシュ読み込み
    cached: dict[str, Any] = _load_cache(path) if not force_refresh else {}

    result: dict[str, list[str]] = {}
    to_fetch: list[str] = []
    now = datetime.now()

    for ticker in tickers:
        if ticker in cached:
            entry = cached[ticker]
            try:
                fetched_at = datetime.fromisoformat(entry["fetched_at"])
                if now - fetched_at < timedelta(days=_CACHE_DAYS):
                    result[ticker] = entry["dates"]
                    continue
            except Exception:
                pass  # 破損キャッシュは再取得
        to_fetch.append(ticker)

    if to_fetch:
        logger.info("決算日を yfinance から取得中: %d 銘柄", len(to_fetch))
        fetched = _fetch_from_yfinance(to_fetch)

        now_str = now.isoformat()
        for ticker, dates in fetched.items():
            result[ticker] = dates
            cached[ticker] = {"fetched_at": now_str, "dates": dates}

        _save_cache(path, cached)
        success = sum(1 for d in fetched.values() if d)
        logger.info("決算日取得完了: %d / %d 銘柄でデータあり", success, len(to_fetch))

    # 取得失敗銘柄は空リストを保証
    for ticker in tickers:
        result.setdefault(ticker, [])

    return result


def days_to_next_earnings(
    date_str: str,
    earnings_list: list[str],
) -> int:
    """指定日から次の決算日までのカレンダー日数を返す。

    Args:
        date_str:      基準日（YYYY-MM-DD）。
        earnings_list: 決算日リスト（昇順ソート済み、YYYY-MM-DD）。

    Returns:
        次の決算日（>= date_str）までのカレンダー日数。
        決算日がなければ 9999。
    """
    if not earnings_list:
        return 9999

    # 次の決算日のインデックスを bisect で高速検索
    idx = bisect_left(earnings_list, date_str)
    if idx >= len(earnings_list):
        return 9999

    try:
        d1 = datetime.strptime(date_str, "%Y-%m-%d")
        d2 = datetime.strptime(earnings_list[idx], "%Y-%m-%d")
        return max(0, (d2 - d1).days)
    except Exception:
        return 9999


# ============================================================
# 内部処理
# ============================================================

def _fetch_from_yfinance(tickers: list[str]) -> dict[str, list[str]]:
    """yfinance から決算日を取得する。

    Returns:
        ticker → 決算日リスト（YYYY-MM-DD、昇順）の辞書。
        失敗銘柄は空リスト。
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance が未インストールのため決算日を取得できません")
        return {t: [] for t in tickers}

    result: dict[str, list[str]] = {}

    for ticker in tickers:
        try:
            t_obj = yf.Ticker(ticker)
            ed = t_obj.earnings_dates

            if ed is None or ed.empty:
                result[ticker] = []
                continue

            # インデックスは Timestamp 型（決算日）
            dates: list[str] = []
            for ts in ed.index:
                try:
                    dates.append(ts.strftime("%Y-%m-%d"))
                except Exception:
                    pass

            dates.sort()  # 昇順ソート
            # 最大 _LIMIT 件に絞る（古い順から _LIMIT 件）
            if len(dates) > _LIMIT:
                dates = dates[-_LIMIT:]

            result[ticker] = dates

        except Exception as e:
            logger.debug("決算日取得失敗 [%s]: %s", ticker, e)
            result[ticker] = []

    return result


def _load_cache(path: Path) -> dict[str, Any]:
    """キャッシュJSONを読み込む。ファイル未存在・パース失敗時は空辞書。"""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_cache(path: Path, data: dict[str, Any]) -> None:
    """キャッシュJSONを保存する。失敗しても例外を投げない。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("決算日キャッシュ保存失敗: %s", e)
