"""
Swing Trade Signal System — バリューファクター計算

セクター別ベンチマークと比較して、割安度をスコア化する。

3指標の均等加重:
  - EV/EBITDA: セクター基準比較 → clip((bench - actual) / bench, -1, 1)
  - FCF Yield: FCF / 時価総額 → clip(yield / 0.05, -1, 1)
  - PBR:       セクター基準比較 → clip((bench - actual) / bench, -1, 1)

入力: ファンダメンタル指標の辞書 + セクター名
出力: スコア (-1.0 〜 +1.0)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from analysis.utils import safe_get as _safe_get
from config.settings import SECTOR_BENCHMARKS

logger = logging.getLogger(__name__)

# 3指標の均等ウェイト
_W_EV_EBITDA: float = 1.0 / 3.0
_W_FCF_YIELD: float = 1.0 / 3.0
_W_PBR: float = 1.0 / 3.0


# ============================================================
# 公開 API
# ============================================================

def calc_value_score(
    fundamentals: dict[str, Any],
    sector: str = "default",
) -> float:
    """バリューファクターの統合スコアを算出する。

    Args:
        fundamentals: ファンダメンタル指標の辞書。必要なキー:
            - ev_to_ebitda (float | None)
            - free_cashflow (float | None)
            - market_cap (float | None)
            - price_to_book (float | None)
        sector: セクター名（ベンチマーク選択用）。

    Returns:
        バリュースコア (-1.0 〜 +1.0)。データ不足時は 0.0。
    """
    if not fundamentals:
        logger.warning("バリュー計算: ファンダメンタルデータが空")
        return 0.0

    try:
        bench = _get_benchmark(sector)

        ev_score = _calc_ev_ebitda_score(fundamentals, bench)
        fcf_score = _calc_fcf_yield_score(fundamentals)
        pbr_score = _calc_pbr_score(fundamentals, bench)

        # 計算可能な指標のみで加重平均（欠損指標は除外）
        scores = []
        weights = []
        for s, w in [(ev_score, _W_EV_EBITDA), (fcf_score, _W_FCF_YIELD), (pbr_score, _W_PBR)]:
            if s is not None:
                scores.append(s)
                weights.append(w)

        if not scores:
            logger.debug("バリュー計算: 算出可能な指標がゼロ")
            return 0.0

        # ウェイト再正規化
        total_w = sum(weights)
        composite = sum(s * w / total_w for s, w in zip(scores, weights))
        score = float(np.clip(composite, -1.0, 1.0))

        logger.debug(
            "バリュー [%s]: EV=%.3f FCF=%.3f PBR=%.3f → %.3f",
            sector,
            ev_score if ev_score is not None else float("nan"),
            fcf_score if fcf_score is not None else float("nan"),
            pbr_score if pbr_score is not None else float("nan"),
            score,
        )
        return score

    except Exception:
        logger.error("バリュー計算で例外発生", exc_info=True)
        return 0.0


def calc_value_detail(
    fundamentals: dict[str, Any],
    sector: str = "default",
) -> dict[str, float | None]:
    """バリューの各サブ指標を個別に返す（デバッグ・詳細表示用）。

    Args:
        fundamentals: ファンダメンタル指標の辞書。
        sector: セクター名。

    Returns:
        サブ指標名 → スコアの辞書。
    """
    if not fundamentals:
        return {"score": 0.0, "error": "empty_data"}

    try:
        bench = _get_benchmark(sector)
        detail: dict[str, float | None] = {
            "sector": hash(sector),  # セクター識別用（文字列は避ける）
            "ev_ebitda_raw": _safe_get(fundamentals, "ev_to_ebitda"),
            "ev_ebitda_bench": bench["ev_ebitda"],
            "ev_ebitda_score": _calc_ev_ebitda_score(fundamentals, bench),
            "fcf_yield_raw": _calc_fcf_yield_raw(fundamentals),
            "fcf_yield_score": _calc_fcf_yield_score(fundamentals),
            "pbr_raw": _safe_get(fundamentals, "price_to_book"),
            "pbr_bench": bench["pb"],
            "pbr_score": _calc_pbr_score(fundamentals, bench),
        }
        detail["score"] = calc_value_score(fundamentals, sector)
        return detail

    except Exception:
        logger.error("バリュー詳細計算で例外発生", exc_info=True)
        return {"score": 0.0, "error": "exception"}


# ============================================================
# サブ指標計算
# ============================================================

def _calc_ev_ebitda_score(
    fund: dict[str, Any],
    bench: dict[str, float],
) -> float | None:
    """EV/EBITDA スコアを算出する。

    ベンチマークより低い（割安）ほど正のスコア。

    Returns:
        スコア (-1.0 〜 +1.0)。データ不足時は None。
    """
    ev_ebitda = _safe_get(fund, "ev_to_ebitda")
    if ev_ebitda is None:
        return None

    bench_val = bench["ev_ebitda"]
    if bench_val == 0:
        return 0.0

    # (ベンチマーク - 実績値) / ベンチマーク → 割安なら正、割高なら負
    score = (bench_val - ev_ebitda) / bench_val
    return float(np.clip(score, -1.0, 1.0))


def _calc_fcf_yield_score(fund: dict[str, Any]) -> float | None:
    """FCF Yield スコアを算出する。

    FCF Yield = Free Cash Flow / Market Cap
    高い FCF Yield ほど割安。

    Returns:
        スコア (-1.0 〜 +1.0)。データ不足時は None。
    """
    fcf = _safe_get(fund, "free_cashflow")
    mcap = _safe_get(fund, "market_cap")

    if fcf is None or mcap is None or mcap == 0:
        return None

    fcf_yield = fcf / mcap
    # 基準値: 5% (0.05)
    return float(np.clip(fcf_yield / 0.05, -1.0, 1.0))


def _calc_fcf_yield_raw(fund: dict[str, Any]) -> float | None:
    """FCF Yield の生の値を返す（デバッグ用）。"""
    fcf = _safe_get(fund, "free_cashflow")
    mcap = _safe_get(fund, "market_cap")
    if fcf is None or mcap is None or mcap == 0:
        return None
    return float(fcf / mcap)


def _calc_pbr_score(
    fund: dict[str, Any],
    bench: dict[str, float],
) -> float | None:
    """PBR (Price to Book) スコアを算出する。

    ベンチマークより低い（割安）ほど正のスコア。

    Returns:
        スコア (-1.0 〜 +1.0)。データ不足時は None。
    """
    pb = _safe_get(fund, "price_to_book")
    if pb is None:
        return None

    bench_val = bench["pb"]
    if bench_val == 0:
        return 0.0

    score = (bench_val - pb) / bench_val
    return float(np.clip(score, -1.0, 1.0))


# ============================================================
# ユーティリティ
# ============================================================

def _get_benchmark(sector: str) -> dict[str, float]:
    """セクター別ベンチマークを取得する。未知のセクターはデフォルト。"""
    return SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS["default"])
