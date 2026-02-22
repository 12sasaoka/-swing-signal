"""センチメント分析のユニットテスト。"""

import pytest

from analysis.sentiment import calc_sentiment_score, calc_sentiment_detail


# ============================================================
# calc_sentiment_score テスト
# ============================================================

class TestCalcSentimentScore:
    """センチメントスコア計算のテスト。"""

    def test_empty_headlines(self):
        assert calc_sentiment_score([]) == 0.0

    def test_bullish_headlines(self):
        headlines = [
            "NVDA beats estimates with record revenue",
            "Analysts upgrade NVDA to strong buy",
            "NVDA stock surges after earnings",
        ]
        score = calc_sentiment_score(headlines)
        assert score > 0.0, f"強気ニュースのスコアは正であるべき: {score}"

    def test_bearish_headlines(self):
        headlines = [
            "Company misses estimates badly",
            "Analyst downgrades stock after profit warning",
            "Stock plunges on weak results",
        ]
        score = calc_sentiment_score(headlines)
        assert score < 0.0, f"弱気ニュースのスコアは負であるべき: {score}"

    def test_neutral_headlines(self):
        headlines = [
            "CEO attends conference in New York",
            "Company announces new office location",
        ]
        score = calc_sentiment_score(headlines)
        assert abs(score) < 0.3, f"中立ニュースのスコアはゼロ付近: {score}"

    def test_score_range(self):
        headlines = ["beats estimates"] * 20
        score = calc_sentiment_score(headlines)
        assert -1.0 <= score <= 1.0

    def test_mixed_headlines(self):
        headlines = [
            "Record revenue reported",
            "Major lawsuit filed against company",
        ]
        score = calc_sentiment_score(headlines)
        assert isinstance(score, float)


# ============================================================
# calc_sentiment_detail テスト
# ============================================================

class TestCalcSentimentDetail:
    """センチメント詳細分析のテスト。"""

    def test_returns_dict(self):
        detail = calc_sentiment_detail(["beats estimates"])
        assert isinstance(detail, dict)
        assert "score" in detail
        assert "headline_count" in detail

    def test_empty(self):
        detail = calc_sentiment_detail([])
        assert detail["score"] == 0.0
        assert detail["headline_count"] == 0

    def test_bullish_matches(self):
        detail = calc_sentiment_detail(["NVDA beats estimates with record revenue"])
        assert detail["bullish_hit_count"] > 0
