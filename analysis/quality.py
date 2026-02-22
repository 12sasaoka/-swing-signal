"""
Swing Trade Signal System — クオリティファクター計算

企業の財務健全性をスコア化する。

5指標の均等加重 (各20%):
  - ROE:                 clip(roe / 0.20, -1, 1)           — 高いほど良い（基準20%）
  - Revenue Growth:      clip(rev_growth / 0.15, -1, 1)    — 高いほど良い（基準15%）
  - Current Ratio:       clip((cr - 1.0) / 1.0, -1, 1)    — 1.5以上が健全
  - Earnings Growth:     clip(eg / 0.20, -1, 1)            — 高いほど良い
  - Gross Profitability: clip(gp_assets / 0.40, -1, 1)     — GP/Assets 高いほど良い（基準40%）

入力: ファンダメンタル指標の辞書
出力: スコア (-1.0 〜 +1.0)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from analysis.utils import safe_get as _safe_get

logger = logging.getLogger(__name__)

# 5指標の均等ウェイト
_W_ROE: float = 0.20
_W_REVENUE: float = 0.20
_W_CURRENT: float = 0.20
_W_EARNINGS: float = 0.20
_W_GROSS_PROFIT: float = 0.20


# ============================================================
# 公開 API
# ============================================================

def calc_quality_score(fundamentals: dict[str, Any]) -> float:
    """クオリティファクターの統合スコアを算出する。

    Args:
        fundamentals: ファンダメンタル指標の辞書。使用するキー:
            - return_on_equity   (float | None)
            - revenue_growth     (float | None)
            - current_ratio      (float | None)
            - earnings_growth    (float | None)
            - gross_profitability (float | None)  ← GP / Total Assets

    Returns:
        クオリティスコア (-1.0 〜 +1.0)。データ不足時は 0.0。
    """
    if not fundamentals:
        logger.warning("クオリティ計算: ファンダメンタルデータが空")
        return 0.0

    try:
        roe_score    = _calc_roe_score(fundamentals)
        rev_score    = _calc_revenue_growth_score(fundamentals)
        cr_score     = _calc_current_ratio_score(fundamentals)
        earn_score   = _calc_earnings_growth_score(fundamentals)
        gp_score     = _calc_gross_profitability_score(fundamentals)

        # 計算可能な指標のみで加重平均（欠損指標の重みを他に再配分）
        scores = []
        weights = []
        for s, w in [
            (roe_score,  _W_ROE),
            (rev_score,  _W_REVENUE),
            (cr_score,   _W_CURRENT),
            (earn_score, _W_EARNINGS),
            (gp_score,   _W_GROSS_PROFIT),
        ]:
            if s is not None:
                scores.append(s)
                weights.append(w)

        if not scores:
            logger.warning("クオリティ計算: 算出可能な指標がゼロ")
            return 0.0

        total_w = sum(weights)
        composite = sum(s * w / total_w for s, w in zip(scores, weights))
        score = float(np.clip(composite, -1.0, 1.0))

        logger.debug(
            "クオリティ: ROE=%.3f Rev=%.3f CR=%.3f Earn=%.3f GP=%.3f → %.3f",
            roe_score    if roe_score  is not None else float("nan"),
            rev_score    if rev_score  is not None else float("nan"),
            cr_score     if cr_score   is not None else float("nan"),
            earn_score   if earn_score is not None else float("nan"),
            gp_score     if gp_score   is not None else float("nan"),
            score,
        )
        return score

    except Exception:
        logger.error("クオリティ計算で例外発生", exc_info=True)
        return 0.0


def calc_quality_detail(fundamentals: dict[str, Any]) -> dict[str, float | None]:
    """クオリティの各サブ指標を個別に返す（デバッグ・詳細表示用）。"""
    if not fundamentals:
        return {"score": 0.0, "error": "empty_data"}

    try:
        detail: dict[str, float | None] = {
            "roe_raw":               _safe_get(fundamentals, "return_on_equity"),
            "roe_score":             _calc_roe_score(fundamentals),
            "revenue_growth_raw":    _safe_get(fundamentals, "revenue_growth"),
            "revenue_score":         _calc_revenue_growth_score(fundamentals),
            "current_ratio_raw":     _safe_get(fundamentals, "current_ratio"),
            "current_ratio_score":   _calc_current_ratio_score(fundamentals),
            "earnings_growth_raw":   _safe_get(fundamentals, "earnings_growth"),
            "earnings_score":        _calc_earnings_growth_score(fundamentals),
            "gross_profitability_raw": _safe_get(fundamentals, "gross_profitability"),
            "gross_profitability_score": _calc_gross_profitability_score(fundamentals),
        }
        detail["score"] = calc_quality_score(fundamentals)
        return detail

    except Exception:
        logger.error("クオリティ詳細計算で例外発生", exc_info=True)
        return {"score": 0.0, "error": "exception"}


# ============================================================
# サブ指標計算
# ============================================================

def _calc_roe_score(fund: dict[str, Any]) -> float | None:
    """ROE スコア: clip(roe / 0.20, -1, 1)。基準: 20%以上で満点。"""
    roe = _safe_get(fund, "return_on_equity")
    if roe is None:
        return None
    return float(np.clip(roe / 0.20, -1.0, 1.0))


def _calc_revenue_growth_score(fund: dict[str, Any]) -> float | None:
    """Revenue Growth スコア: clip(rev_growth / 0.15, -1, 1)。基準: 15%以上で満点。"""
    rg = _safe_get(fund, "revenue_growth")
    if rg is None:
        return None
    return float(np.clip(rg / 0.15, -1.0, 1.0))


def _calc_current_ratio_score(fund: dict[str, Any]) -> float | None:
    """Current Ratio スコア: clip((cr - 1.0) / 1.0, -1, 1)。1.5以上が健全。

    CR = 1.0 → score =  0.0 (ギリギリ)
    CR = 2.0 → score = +1.0 (健全)
    CR = 0.5 → score = -0.5 (流動性不足)
    """
    cr = _safe_get(fund, "current_ratio")
    if cr is None:
        return None
    score = (cr - 1.0) / 1.0
    return float(np.clip(score, -1.0, 1.0))


def _calc_earnings_growth_score(fund: dict[str, Any]) -> float | None:
    """Earnings Growth スコア: clip(eps_growth / 0.20, -1, 1)。高いほど良い。"""
    eg = _safe_get(fund, "earnings_growth")
    if eg is None:
        return None
    return float(np.clip(eg / 0.20, -1.0, 1.0))


def _calc_gross_profitability_score(fund: dict[str, Any]) -> float | None:
    """Gross Profitability スコア: clip(gp_assets / 0.40, -1, 1)。

    Novy-Marx (2013) の定義: Gross Profit / Total Assets
    基準: 40%以上（GP/Assets ≥ 0.40）で満点。
    高いほど収益性・競争優位が高い。
    """
    gp = _safe_get(fund, "gross_profitability")
    if gp is None:
        return None
    return float(np.clip(gp / 0.40, -1.0, 1.0))
