"""バリューファクターのユニットテスト。"""

import pytest

from analysis.value import calc_value_score, calc_value_detail


# ============================================================
# テストデータ
# ============================================================

def _cheap_stock() -> dict:
    """割安な銘柄のデータ。"""
    return {
        "ev_to_ebitda": 8.0,   # ベンチマーク25.0より大幅に低い
        "free_cashflow": 10e9,
        "market_cap": 100e9,    # FCF yield = 10%
        "price_to_book": 2.0,   # ベンチマーク5.0より低い
    }


def _expensive_stock() -> dict:
    """割高な銘柄のデータ。"""
    return {
        "ev_to_ebitda": 50.0,   # ベンチマークの2倍
        "free_cashflow": 1e9,
        "market_cap": 500e9,    # FCF yield = 0.2%
        "price_to_book": 10.0,  # ベンチマークの2倍
    }


# ============================================================
# calc_value_score テスト
# ============================================================

class TestCalcValueScore:
    """バリュースコア計算のテスト。"""

    def test_cheap_positive(self):
        score = calc_value_score(_cheap_stock(), "semiconductor")
        assert score > 0.0, f"割安銘柄は正スコア: {score}"

    def test_expensive_negative(self):
        score = calc_value_score(_expensive_stock(), "semiconductor")
        assert score < 0.0, f"割高銘柄は負スコア: {score}"

    def test_empty_data(self):
        assert calc_value_score({}) == 0.0

    def test_score_range(self):
        score = calc_value_score(_cheap_stock(), "semiconductor")
        assert -1.0 <= score <= 1.0

    def test_unknown_sector_uses_default(self):
        score = calc_value_score(_cheap_stock(), "nonexistent_sector")
        assert isinstance(score, float)

    def test_partial_data(self):
        """一部の指標のみでも計算できる。"""
        data = {"ev_to_ebitda": 10.0}
        score = calc_value_score(data, "semiconductor")
        assert isinstance(score, float)


# ============================================================
# calc_value_detail テスト
# ============================================================

class TestCalcValueDetail:
    """バリュー詳細のテスト。"""

    def test_contains_subscores(self):
        detail = calc_value_detail(_cheap_stock(), "semiconductor")
        assert "ev_ebitda_score" in detail
        assert "fcf_yield_score" in detail
        assert "pbr_score" in detail
        assert "score" in detail

    def test_empty(self):
        detail = calc_value_detail({})
        assert detail["score"] == 0.0
