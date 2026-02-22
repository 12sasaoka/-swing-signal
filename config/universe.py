"""
Swing Trade Signal System — 銘柄ユニバース定義

コア銘柄は廃止。全モードで Russell 3000 (IWV) ベースの
スクリーニングを使用する。
セクター判定は get_sector() を通じて利用可能（未知銘柄は "unknown"）。
"""

from __future__ import annotations


# ============================================================
# コア銘柄・ETF（廃止 — Russell 3000 に統一）
# ============================================================

CORE_UNIVERSE: dict[str, list[str]] = {}
CORE_ETFS: dict[str, str] = {}


# ============================================================
# ヘルパー関数（後方互換性のため維持）
# ============================================================

def get_core_tickers() -> list[str]:
    """コア銘柄のティッカーリストを返す（現在は空）。"""
    tickers: set[str] = set()
    for sector_tickers in CORE_UNIVERSE.values():
        tickers.update(sector_tickers)
    return sorted(tickers)


def get_etf_tickers() -> list[str]:
    """主要ETFのティッカーリストを返す（現在は空）。"""
    return sorted(CORE_ETFS.keys())


def get_all_tickers() -> list[str]:
    """コア銘柄 + ETFの全ティッカーリストを返す（現在は空）。"""
    all_tickers: set[str] = set(get_core_tickers())
    all_tickers.update(get_etf_tickers())
    return sorted(all_tickers)


def get_sector(ticker: str) -> str:
    """指定ティッカーのセクターを返す。

    Args:
        ticker: ティッカーシンボル（例: "NVDA"）。

    Returns:
        セクター名。コア登録外は "unknown"。
    """
    for sector, tickers in CORE_UNIVERSE.items():
        if ticker in tickers:
            return sector
    if ticker in CORE_ETFS:
        return "etf"
    return "unknown"


def get_tickers_by_sector(sector: str) -> list[str]:
    """指定セクターに属するティッカーリストを返す。"""
    if sector == "etf":
        return get_etf_tickers()
    return list(CORE_UNIVERSE.get(sector, []))


def get_all_sectors() -> list[str]:
    """定義されている全セクター名をリストで返す。"""
    sectors = sorted(CORE_UNIVERSE.keys())
    sectors.append("etf")
    return sectors


def get_universe_summary() -> dict[str, int]:
    """ユニバースの概要（セクター別銘柄数）を返す。"""
    summary: dict[str, int] = {}
    for sector, tickers in CORE_UNIVERSE.items():
        summary[sector] = len(tickers)
    summary["etf"] = len(CORE_ETFS)
    summary["total_unique"] = len(get_all_tickers())
    return summary
