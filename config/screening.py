"""
Swing Trade Signal System — スクリーニング設定

Russell 3000 からの銘柄スクリーニングに使用する閾値・パラメータを定義する。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScreeningParams:
    """スクリーニング用の閾値パラメータ。"""

    # ---- Tier 1: ファンダメンタル事前フィルタ ----
    min_market_cap: float = 1.5e8        # 最低時価総額 ($150M)
    min_avg_volume: int = 300_000        # 最低平均出来高 (株/日)
    min_price: float = 5.0               # 最低株価 (ペニーストック除外)

    # ---- Tier 2: テクニカルフィルタ ----
    min_momentum_score: float = 0.0      # 最低モメンタムスコア
    max_screen_candidates: int = 100     # Tier2通過後の最大銘柄数

    # ---- キャッシュ ----
    holdings_cache_days: int = 7         # IWV構成銘柄キャッシュ有効日数
    tier1_cache_days: int = 1            # Tier1結果キャッシュ有効日数

    # ---- 並列処理 ----
    tier1_max_workers: int = 16          # Tier1並列ワーカー数


SCREENING_PARAMS = ScreeningParams()
