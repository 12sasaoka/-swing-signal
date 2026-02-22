"""
Swing Trade Signal System — キーワードベース簡易センチメント分析

ニュースヘッドラインのリストを受け取り、強気/弱気キーワードの
出現頻度からセンチメントスコアを算出する。

Claude API を使わない軽量版で、--quick モード時のセンチメントファクターとして使用。

入力: ニュースヘッドラインのリスト (list[str])
出力: スコア (-1.0 〜 +1.0)
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# キーワード辞書
# ============================================================

# (キーワード / フレーズ, ウェイト) のリスト
# ウェイトが大きいほど強いシグナル

_BULLISH_KEYWORDS: list[tuple[str, float]] = [
    # 決算関連（強い）
    ("beats estimates", 1.5),
    ("beats expectations", 1.5),
    ("record revenue", 1.5),
    ("record earnings", 1.5),
    ("record profit", 1.5),
    ("blowout quarter", 1.5),
    ("strong results", 1.2),
    ("tops estimates", 1.2),
    # アップグレード・目標株価
    ("upgrade", 1.0),
    ("price target raised", 1.2),
    ("target raised", 1.0),
    ("outperform", 0.8),
    ("overweight", 0.8),
    ("buy rating", 1.0),
    ("strong buy", 1.2),
    # ポジティブ動向
    ("surge", 1.0),
    ("soar", 1.0),
    ("rally", 0.8),
    ("breakout", 0.8),
    ("all-time high", 1.0),
    ("new high", 0.8),
    ("breakthrough", 1.0),
    ("innovation", 0.6),
    # 事業関連
    ("partnership", 0.6),
    ("acquisition", 0.5),
    ("contract win", 0.8),
    ("new contract", 0.7),
    ("fda approval", 1.2),
    ("approval", 0.6),
    ("dividend increase", 0.7),
    ("buyback", 0.6),
    ("share repurchase", 0.6),
    # 成長
    ("revenue growth", 0.5),
    ("earnings growth", 0.5),
    ("strong growth", 0.6),
    ("expansion", 0.4),
    ("strong demand", 0.5),
    ("bullish", 0.6),
    ("optimistic", 0.5),
    ("positive outlook", 0.4),
    ("positive guidance", 0.5),
    ("upbeat", 0.5),
]

_BEARISH_KEYWORDS: list[tuple[str, float]] = [
    # 決算関連（強い）
    ("misses estimates", 1.5),
    ("misses expectations", 1.5),
    ("revenue miss", 1.5),
    ("earnings miss", 1.5),
    ("profit warning", 1.5),
    ("guidance cut", 1.5),
    ("lowers guidance", 1.5),
    ("weak results", 1.2),
    ("disappointing", 1.0),
    # ダウングレード
    ("downgrade", 1.0),
    ("price target cut", 1.2),
    ("target lowered", 1.0),
    ("underperform", 0.8),
    ("underweight", 0.8),
    ("sell rating", 1.0),
    # ネガティブ動向
    ("revenue decline", 0.8),
    ("earnings decline", 0.8),
    ("shares decline", 0.6),
    ("plunge", 1.0),
    ("crash", 1.2),
    ("tumble", 0.8),
    ("selloff", 0.8),
    ("sell-off", 0.8),
    ("slump", 0.8),
    ("stock drops", 0.6),
    ("shares drop", 0.6),
    ("price drop", 0.6),
    ("shares fall", 0.5),
    ("stock falls", 0.5),
    # リスク・問題
    ("lawsuit", 0.8),
    ("investigation", 0.7),
    ("fraud", 1.2),
    ("scandal", 1.0),
    ("recall", 0.7),
    ("layoff", 0.6),
    ("layoffs", 0.6),
    ("restructuring", 0.5),
    ("bankruptcy", 1.5),
    ("loan default", 1.0),
    ("debt default", 1.0),
    ("debt concern", 0.8),
    # 規制・マクロ
    ("tariff", 0.6),
    ("sanction", 0.7),
    ("recession", 0.8),
    ("bearish", 0.6),
    ("pessimistic", 0.5),
    ("negative outlook", 0.4),
    ("negative guidance", 0.5),
    ("profit warning", 1.5),
    ("downside risk", 0.5),
]


# ============================================================
# 公開 API
# ============================================================

def calc_sentiment_score(headlines: list[str]) -> float:
    """ニュースヘッドラインからセンチメントスコアを算出する。

    Args:
        headlines: ニュースヘッドラインのリスト。各要素は1件のタイトル文字列。

    Returns:
        センチメントスコア (-1.0 〜 +1.0)。
        ヘッドラインが空の場合は 0.0（中立）。
    """
    if not headlines:
        logger.debug("センチメント: ヘッドラインが空 → 0.0")
        return 0.0

    try:
        bullish_total = 0.0
        bearish_total = 0.0
        bull_hits = 0
        bear_hits = 0

        for headline in headlines:
            lower = headline.lower()

            for keyword, weight in _BULLISH_KEYWORDS:
                if keyword in lower:
                    bullish_total += weight
                    bull_hits += 1

            for keyword, weight in _BEARISH_KEYWORDS:
                if keyword in lower:
                    bearish_total += weight
                    bear_hits += 1

        total_weight = bullish_total + bearish_total
        if total_weight == 0:
            logger.debug("センチメント: キーワードヒットなし → 0.0")
            return 0.0

        # net score: 正なら強気、負なら弱気
        net = bullish_total - bearish_total

        # ヘッドライン数で正規化（多くのニュースがあるほど信頼度が高い）
        # 基準: 5件のヘッドラインで最大インパクト
        normalizer = min(len(headlines), 5)
        # 1ヘッドラインあたり平均 weight 1.0 と仮定 → 5件で最大5.0
        normalized = net / (normalizer * 1.5)

        score = float(np.clip(normalized, -1.0, 1.0))

        logger.debug(
            "センチメント: %d件, bull=%.1f(%d hits), bear=%.1f(%d hits) → %.3f",
            len(headlines), bullish_total, bull_hits, bearish_total, bear_hits, score,
        )
        return score

    except Exception:
        logger.error("センチメント計算で例外発生", exc_info=True)
        return 0.0


def calc_sentiment_detail(headlines: list[str]) -> dict[str, Any]:
    """センチメントの詳細分析結果を返す（デバッグ・詳細表示用）。

    Args:
        headlines: ニュースヘッドラインのリスト。

    Returns:
        分析詳細の辞書。
    """
    if not headlines:
        return {
            "score": 0.0,
            "headline_count": 0,
            "bullish_keywords": [],
            "bearish_keywords": [],
        }

    try:
        bull_matches: list[dict[str, Any]] = []
        bear_matches: list[dict[str, Any]] = []
        bullish_total = 0.0
        bearish_total = 0.0

        for headline in headlines:
            lower = headline.lower()

            for keyword, weight in _BULLISH_KEYWORDS:
                if keyword in lower:
                    bullish_total += weight
                    bull_matches.append({
                        "keyword": keyword,
                        "weight": weight,
                        "headline": headline[:80],
                    })

            for keyword, weight in _BEARISH_KEYWORDS:
                if keyword in lower:
                    bearish_total += weight
                    bear_matches.append({
                        "keyword": keyword,
                        "weight": weight,
                        "headline": headline[:80],
                    })

        return {
            "score": calc_sentiment_score(headlines),
            "headline_count": len(headlines),
            "bullish_total_weight": bullish_total,
            "bearish_total_weight": bearish_total,
            "bullish_hit_count": len(bull_matches),
            "bearish_hit_count": len(bear_matches),
            "bullish_keywords": bull_matches,
            "bearish_keywords": bear_matches,
        }

    except Exception:
        logger.error("センチメント詳細計算で例外発生", exc_info=True)
        return {"score": 0.0, "error": "exception"}
