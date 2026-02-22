"""リスク管理のユニットテスト。"""

import numpy as np
import pandas as pd
import pytest

from strategy.risk import (
    RiskLevels,
    calc_risk_levels,
    calc_atr,
    calc_risk_levels_from_price,
)


# ============================================================
# テストデータ生成ヘルパー
# ============================================================

def _make_ohlcv(n: int = 50, base_price: float = 100.0) -> pd.DataFrame:
    """テスト用の OHLCV DataFrame を生成する。"""
    np.random.seed(42)
    dates = pd.date_range(end="2026-02-20", periods=n, freq="B")
    closes = [base_price]
    for i in range(1, n):
        closes.append(closes[-1] * (1 + np.random.normal(0, 0.02)))
    return pd.DataFrame({
        "Open": [c * 0.995 for c in closes],
        "High": [c * 1.015 for c in closes],
        "Low": [c * 0.985 for c in closes],
        "Close": closes,
        "Volume": [1_000_000] * n,
    }, index=dates)


# ============================================================
# calc_atr テスト
# ============================================================

class TestCalcATR:
    """ATR 計算のテスト。"""

    def test_returns_float(self):
        df = _make_ohlcv()
        atr = calc_atr(df, 14)
        assert isinstance(atr, float)
        assert atr > 0

    def test_none_on_insufficient_data(self):
        df = _make_ohlcv(n=5)
        assert calc_atr(df, 14) is None

    def test_none_on_empty(self):
        assert calc_atr(None) is None
        assert calc_atr(pd.DataFrame()) is None


# ============================================================
# calc_risk_levels テスト
# ============================================================

class TestCalcRiskLevels:
    """リスクレベル計算のテスト。"""

    def test_returns_risk_levels(self):
        df = _make_ohlcv()
        risk = calc_risk_levels(df)
        assert isinstance(risk, RiskLevels)

    def test_stop_loss_below_current(self):
        df = _make_ohlcv()
        risk = calc_risk_levels(df)
        assert risk.stop_loss < risk.current_price

    def test_take_profit_above_current(self):
        df = _make_ohlcv()
        risk = calc_risk_levels(df)
        assert risk.take_profit > risk.current_price

    def test_stop_loss_atr_based(self):
        """ストップロスはATRベース（-8%をフロア）。"""
        df = _make_ohlcv()
        risk = calc_risk_levels(df)
        # SLは現在値から負の方向なのでpctは負
        assert risk.stop_loss_pct < 0
        # SL は最低でもハード-8%（= 0.92 × current_price）以上
        hard_sl = risk.current_price * 0.92
        assert risk.stop_loss >= hard_sl

    def test_risk_reward_positive(self):
        df = _make_ohlcv()
        risk = calc_risk_levels(df)
        assert risk.risk_reward_ratio > 0

    def test_none_on_insufficient_data(self):
        df = _make_ohlcv(n=5)
        assert calc_risk_levels(df) is None

    def test_none_on_empty(self):
        assert calc_risk_levels(None) is None

    def test_none_on_missing_columns(self):
        df = pd.DataFrame({"Close": [100.0] * 50})
        assert calc_risk_levels(df) is None

    def test_to_dict(self):
        df = _make_ohlcv()
        risk = calc_risk_levels(df)
        d = risk.to_dict()
        assert "current_price" in d
        assert "stop_loss" in d
        assert "take_profit" in d
        assert "risk_reward_ratio" in d


# ============================================================
# calc_risk_levels_from_price テスト
# ============================================================

class TestCalcRiskLevelsFromPrice:
    """事前計算済みATRからのリスクレベル計算テスト。"""

    def test_basic(self):
        risk = calc_risk_levels_from_price(100.0, 5.0)
        assert risk is not None
        assert risk.current_price == 100.0
        # SL = max(100 - 5*2, 100*0.92) = max(90, 92) = 92
        assert risk.stop_loss == 92.0
        # TP = 100 + 5 * 4.0 = 120.0
        assert risk.take_profit == 120.0

    def test_invalid_price(self):
        assert calc_risk_levels_from_price(0.0, 5.0) is None
        assert calc_risk_levels_from_price(-10.0, 5.0) is None

    def test_invalid_atr(self):
        assert calc_risk_levels_from_price(100.0, 0.0) is None
        assert calc_risk_levels_from_price(100.0, -1.0) is None
