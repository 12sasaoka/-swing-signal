"""モメンタムファクターのユニットテスト。"""

import numpy as np
import pandas as pd
import pytest

from analysis.momentum import (
    calc_momentum_score,
    calc_momentum_detail,
    _calc_rsi,
    _calc_return,
    _calc_roc,
    _calc_macd_score,
    _calc_ma_deviation,
    _rsi_to_score,
    _norm,
)


# ============================================================
# テストデータ生成ヘルパー
# ============================================================

def _make_ohlcv(closes: list[float], days: int | None = None) -> pd.DataFrame:
    """テスト用の OHLCV DataFrame を生成する。"""
    n = len(closes)
    dates = pd.date_range(end="2026-02-20", periods=n, freq="B")
    df = pd.DataFrame({
        "Open": [c * 0.99 for c in closes],
        "High": [c * 1.02 for c in closes],
        "Low": [c * 0.98 for c in closes],
        "Close": closes,
        "Volume": [1_000_000] * n,
    }, index=dates)
    return df


def _make_uptrend(n: int = 300, start: float = 100.0) -> pd.DataFrame:
    """上昇トレンドのテストデータを生成する。"""
    closes = [start * (1 + 0.001 * i) for i in range(n)]
    return _make_ohlcv(closes)


def _make_downtrend(n: int = 300, start: float = 200.0) -> pd.DataFrame:
    """下落トレンドのテストデータを生成する。"""
    closes = [start * (1 - 0.001 * i) for i in range(n)]
    return _make_ohlcv(closes)


def _make_flat(n: int = 300, price: float = 100.0) -> pd.DataFrame:
    """横ばいのテストデータを生成する。"""
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, n)
    closes = [price + noise[i] for i in range(n)]
    return _make_ohlcv(closes)


# ============================================================
# calc_momentum_score テスト
# ============================================================

class TestCalcMomentumScore:
    """calc_momentum_score の基本テスト。"""

    def test_returns_float(self):
        df = _make_uptrend()
        score = calc_momentum_score(df)
        assert isinstance(score, float)

    def test_score_range(self):
        df = _make_uptrend()
        score = calc_momentum_score(df)
        assert -1.0 <= score <= 1.0

    def test_uptrend_positive(self):
        df = _make_uptrend()
        score = calc_momentum_score(df)
        assert score > 0.0, f"上昇トレンドのスコアは正であるべき: {score}"

    def test_downtrend_negative(self):
        df = _make_downtrend()
        score = calc_momentum_score(df)
        assert score < 0.0, f"下落トレンドのスコアは負であるべき: {score}"

    def test_flat_near_zero(self):
        df = _make_flat()
        score = calc_momentum_score(df)
        assert abs(score) < 0.5, f"横ばいのスコアはゼロ付近であるべき: {score}"

    def test_none_input(self):
        assert calc_momentum_score(None) == 0.0

    def test_empty_dataframe(self):
        assert calc_momentum_score(pd.DataFrame()) == 0.0

    def test_insufficient_data(self):
        df = _make_ohlcv([100.0] * 10)
        assert calc_momentum_score(df) == 0.0

    def test_missing_close_column(self):
        df = _make_uptrend()
        df = df.drop(columns=["Close"])
        assert calc_momentum_score(df) == 0.0


# ============================================================
# calc_momentum_detail テスト
# ============================================================

class TestCalcMomentumDetail:
    """calc_momentum_detail の基本テスト。"""

    def test_returns_dict(self):
        df = _make_uptrend()
        detail = calc_momentum_detail(df)
        assert isinstance(detail, dict)
        assert "score" in detail

    def test_contains_subindicators(self):
        df = _make_uptrend()
        detail = calc_momentum_detail(df)
        expected_keys = {"score", "return_1w", "rsi_14", "short_term", "mid_term", "long_term"}
        assert expected_keys.issubset(detail.keys())

    def test_score_matches_main_function(self):
        """calc_momentum_detail と calc_momentum_score が同じスコアを返す。"""
        df = _make_uptrend()
        score_main = calc_momentum_score(df)
        score_detail = calc_momentum_detail(df)["score"]
        assert abs(score_main - score_detail) < 0.01, (
            f"score不整合: calc_momentum_score={score_main}, "
            f"calc_momentum_detail={score_detail}"
        )

    def test_none_input(self):
        detail = calc_momentum_detail(None)
        assert detail["score"] == 0.0


# ============================================================
# ヘルパー関数テスト
# ============================================================

class TestRSI:
    """RSI 計算のテスト。"""

    def test_rsi_strong_uptrend(self):
        closes = pd.Series([100 + i * 2 for i in range(30)])
        rsi = _calc_rsi(closes, 14)
        assert rsi > 70, f"強い上昇トレンドのRSIは70超: {rsi}"

    def test_rsi_strong_downtrend(self):
        closes = pd.Series([200 - i * 2 for i in range(30)])
        rsi = _calc_rsi(closes, 14)
        assert rsi < 30, f"強い下落トレンドのRSIは30未満: {rsi}"

    def test_rsi_insufficient_data(self):
        closes = pd.Series([100.0] * 5)
        rsi = _calc_rsi(closes, 14)
        assert rsi == 50.0  # 中立

    def test_rsi_range(self):
        closes = pd.Series([100 + np.sin(i * 0.3) * 10 for i in range(50)])
        rsi = _calc_rsi(closes, 14)
        assert 0.0 <= rsi <= 100.0


class TestRSIToScore:
    """RSI → スコア変換のテスト。"""

    def test_overbought(self):
        assert _rsi_to_score(85) == -0.30
        assert _rsi_to_score(75) == -0.15

    def test_oversold(self):
        assert _rsi_to_score(15) == 0.30
        assert _rsi_to_score(25) == 0.15

    def test_neutral(self):
        assert _rsi_to_score(50) == 0.0
        assert _rsi_to_score(45) == 0.0


class TestCalcReturn:
    """リターン計算のテスト。"""

    def test_positive_return(self):
        closes = pd.Series([100.0, 110.0])
        ret = _calc_return(closes, 1)
        assert abs(ret - 0.10) < 1e-6

    def test_negative_return(self):
        closes = pd.Series([100.0, 90.0])
        ret = _calc_return(closes, 1)
        assert abs(ret - (-0.10)) < 1e-6

    def test_insufficient_data(self):
        closes = pd.Series([100.0])
        assert _calc_return(closes, 5) == 0.0


class TestNorm:
    """_norm 正規化のテスト。"""

    def test_within_range(self):
        assert _norm(0.05, 0.10) == 0.5

    def test_clip_upper(self):
        assert _norm(0.20, 0.10) == 1.0

    def test_clip_lower(self):
        assert _norm(-0.20, 0.10) == -1.0

    def test_zero_reference(self):
        assert _norm(0.05, 0.0) == 0.0
