"""クオリティファクターのユニットテスト。"""

import pytest

from analysis.quality import calc_quality_score, calc_quality_detail


# ============================================================
# テストデータ
# ============================================================

def _high_quality() -> dict:
    return {
        "return_on_equity": 0.30,
        "debt_to_equity": 0.3,
        "revenue_growth": 0.25,
        "current_ratio": 2.5,
        "earnings_growth": 0.20,
    }


def _low_quality() -> dict:
    return {
        "return_on_equity": -0.05,
        "debt_to_equity": 3.0,
        "revenue_growth": -0.15,
        "current_ratio": 0.3,
        "earnings_growth": -0.20,
    }


def _partial_data() -> dict:
    return {
        "return_on_equity": 0.15,
        "revenue_growth": 0.10,
    }


# ============================================================
# calc_quality_score テスト
# ============================================================

class TestCalcQualityScore:
    """クオリティスコア計算のテスト。"""

    def test_high_quality_positive(self):
        score = calc_quality_score(_high_quality())
        assert score > 0.0, f"高クオリティは正スコア: {score}"

    def test_low_quality_negative(self):
        score = calc_quality_score(_low_quality())
        assert score < 0.0, f"低クオリティは負スコア: {score}"

    def test_empty_data(self):
        assert calc_quality_score({}) == 0.0

    def test_none_data(self):
        assert calc_quality_score(None) == 0.0

    def test_score_range(self):
        score = calc_quality_score(_high_quality())
        assert -1.0 <= score <= 1.0

    def test_partial_data(self):
        """一部の指標のみでも計算できる。"""
        score = calc_quality_score(_partial_data())
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0


# ============================================================
# calc_quality_detail テスト
# ============================================================

class TestCalcQualityDetail:
    """クオリティ詳細のテスト。"""

    def test_contains_all_scores(self):
        detail = calc_quality_detail(_high_quality())
        assert "roe_score" in detail
        assert "debt_score" in detail
        assert "revenue_score" in detail
        assert "current_ratio_score" in detail
        assert "earnings_score" in detail
        assert "score" in detail

    def test_empty(self):
        detail = calc_quality_detail({})
        assert detail["score"] == 0.0
