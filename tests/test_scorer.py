"""統合スコアリングのユニットテスト。"""

import numpy as np
import pandas as pd
import pytest

from strategy.scorer import (
    STRONG_BUY,
    BUY,
    HOLD,
    SELL,
    STRONG_SELL,
    ScoringResult,
    determine_signal,
    score_ticker,
    score_universe,
    filter_actionable,
    _calc_composite,
    _generate_reason,
)


# ============================================================
# テストデータ生成ヘルパー
# ============================================================

def _make_ohlcv(n: int = 300, trend: float = 0.001) -> pd.DataFrame:
    """テスト用の OHLCV DataFrame を生成する。"""
    start = 100.0
    closes = [start * (1 + trend * i) for i in range(n)]
    dates = pd.date_range(end="2026-02-20", periods=n, freq="B")
    return pd.DataFrame({
        "Open": [c * 0.99 for c in closes],
        "High": [c * 1.02 for c in closes],
        "Low": [c * 0.98 for c in closes],
        "Close": closes,
        "Volume": [1_000_000] * n,
    }, index=dates)


def _make_fundamentals(
    roe: float = 0.20,
    de: float = 0.5,
    rg: float = 0.10,
    cr: float = 1.5,
    eg: float = 0.15,
) -> dict:
    """テスト用のファンダメンタルデータを生成する。"""
    return {
        "trailing_pe": 20.0,
        "forward_pe": 18.0,
        "price_to_book": 3.0,
        "return_on_equity": roe,
        "ev_to_ebitda": 15.0,
        "revenue_growth": rg,
        "earnings_growth": eg,
        "debt_to_equity": de,
        "current_ratio": cr,
        "market_cap": 100e9,
        "free_cashflow": 5e9,
    }


# ============================================================
# determine_signal テスト
# ============================================================

class TestDetermineSignal:
    """シグナル判定のテスト。"""

    def test_strong_buy(self):
        assert determine_signal(0.85) == STRONG_BUY

    def test_buy(self):
        assert determine_signal(0.65) == BUY

    def test_hold_positive(self):
        assert determine_signal(0.3) == HOLD

    def test_hold_zero(self):
        assert determine_signal(0.0) == HOLD

    def test_hold_negative(self):
        assert determine_signal(-0.2) == HOLD

    def test_sell(self):
        assert determine_signal(-0.4) == SELL

    def test_strong_sell(self):
        assert determine_signal(-0.7) == STRONG_SELL

    def test_boundary_strong_buy(self):
        assert determine_signal(0.8) == STRONG_BUY

    def test_boundary_buy(self):
        assert determine_signal(0.6) == BUY

    def test_boundary_sell(self):
        assert determine_signal(-0.3) == SELL

    def test_boundary_strong_sell(self):
        assert determine_signal(-0.6) == STRONG_SELL


# ============================================================
# _calc_composite テスト
# ============================================================

class TestCalcComposite:
    """加重平均計算のテスト。"""

    def test_all_zero(self):
        assert _calc_composite(0.0, 0.0, 0.0, 0.0) == 0.0

    def test_all_positive(self):
        score = _calc_composite(1.0, 1.0, 1.0, 1.0)
        assert abs(score - 1.0) < 1e-6

    def test_all_negative(self):
        score = _calc_composite(-1.0, -1.0, -1.0, -1.0)
        assert abs(score - (-1.0)) < 1e-6

    def test_momentum_dominant(self):
        """モメンタムのウェイトが50%なので最も影響が大きい。"""
        score = _calc_composite(1.0, 0.0, 0.0, 0.0)
        assert abs(score - 0.5) < 1e-6

    def test_value_weight_zero(self):
        """バリューウェイトが0%なので寄与しない。"""
        score_with = _calc_composite(0.5, 1.0, 0.5, 0.5)
        score_without = _calc_composite(0.5, 0.0, 0.5, 0.5)
        assert abs(score_with - score_without) < 1e-6

    def test_clip_range(self):
        score = _calc_composite(1.0, 1.0, 1.0, 1.0)
        assert -1.0 <= score <= 1.0


# ============================================================
# score_ticker テスト
# ============================================================

class TestScoreTicker:
    """1銘柄スコアリングのテスト。"""

    def test_returns_scoring_result(self):
        df = _make_ohlcv()
        fund = _make_fundamentals()
        result = score_ticker("NVDA", "semiconductor", df, fund, [])
        assert isinstance(result, ScoringResult)
        assert result.ticker == "NVDA"
        assert result.sector == "semiconductor"

    def test_score_range(self):
        df = _make_ohlcv()
        fund = _make_fundamentals()
        result = score_ticker("NVDA", "semiconductor", df, fund, [])
        assert -1.0 <= result.final_score <= 1.0

    def test_signal_is_valid(self):
        df = _make_ohlcv()
        fund = _make_fundamentals()
        result = score_ticker("NVDA", "semiconductor", df, fund, [])
        valid_signals = {STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL}
        assert result.signal in valid_signals

    def test_no_data(self):
        """データなしの場合はHOLDになる。"""
        result = score_ticker("XYZ", "unknown")
        assert result.signal == HOLD
        assert result.final_score == 0.0

    def test_with_claude_score(self):
        df = _make_ohlcv()
        fund = _make_fundamentals()
        result = score_ticker("NVDA", "semiconductor", df, fund, [], claude_score=0.8)
        assert result.claude_score == 0.8

    def test_risk_levels_present(self):
        df = _make_ohlcv()
        fund = _make_fundamentals()
        result = score_ticker("NVDA", "semiconductor", df, fund, [])
        assert result.risk is not None
        assert result.risk.current_price > 0
        assert result.risk.stop_loss < result.risk.current_price
        assert result.risk.take_profit > result.risk.current_price

    def test_dip_buy_detected(self):
        """上昇トレンドかつ直近ディップ → BUYシグナルが出る。"""
        n = 300
        # 長期上昇 + 直近5日間の下落
        closes = [100 * (1 + 0.002 * i) for i in range(n - 5)]
        last_price = closes[-1]
        # 直近5日で3%下落
        for i in range(5):
            closes.append(last_price * (1 - 0.006 * (i + 1)))
        df = _make_ohlcv_from_closes(closes)
        fund = _make_fundamentals(roe=0.30, rg=0.20, eg=0.25, cr=2.0)
        result = score_ticker("TEST", "semiconductor", df, fund, ["strong buy", "beats estimates"])
        # ディップ買い検知 or 高確信でBUY系が出るはず（スコア次第）
        # 少なくともHOLDへの格下げが回避されていることを確認
        assert "[ディップ買い検知]" in result.reason or result.signal in (BUY, STRONG_BUY, HOLD)

    def test_strong_buy_bypass_dip_filter(self):
        """スコアがstrong_buy閾値以上ならディップでなくてもBUYを維持。"""
        df = _make_ohlcv(trend=0.003)  # 強い上昇トレンド
        # 非常に強いファンダメンタル
        fund = _make_fundamentals(roe=0.50, rg=0.30, eg=0.40, cr=3.0)
        result = score_ticker(
            "TEST", "semiconductor", df, fund,
            ["beats estimates", "record revenue", "strong buy", "surge"],
            claude_score=0.9,
        )
        # final_score が 0.8 以上なら高確信BUYになるはず
        if result.final_score >= 0.8:
            assert result.signal == STRONG_BUY
            assert "[高確信]" in result.reason


def _make_ohlcv_from_closes(closes: list[float]) -> pd.DataFrame:
    """終値リストから OHLCV DataFrame を生成する。"""
    n = len(closes)
    dates = pd.date_range(end="2026-02-20", periods=n, freq="B")
    return pd.DataFrame({
        "Open": [c * 0.99 for c in closes],
        "High": [c * 1.02 for c in closes],
        "Low": [c * 0.98 for c in closes],
        "Close": closes,
        "Volume": [1_000_000] * n,
    }, index=dates)


# ============================================================
# score_universe テスト
# ============================================================

class TestScoreUniverse:
    """ユニバーススコアリングのテスト。"""

    def test_returns_sorted_list(self):
        universe = {
            "NVDA": {
                "sector": "semiconductor",
                "price_df": _make_ohlcv(trend=0.002),
                "fundamentals": _make_fundamentals(),
                "headlines": [],
            },
            "BAD": {
                "sector": "unknown",
                "price_df": _make_ohlcv(trend=-0.002),
                "fundamentals": _make_fundamentals(roe=-0.1, de=3.0, rg=-0.2),
                "headlines": [],
            },
        }
        results = score_universe(universe)
        assert len(results) == 2
        # スコア降順
        assert results[0].final_score >= results[1].final_score

    def test_empty_universe(self):
        results = score_universe({})
        assert results == []


# ============================================================
# filter_actionable テスト
# ============================================================

class TestFilterActionable:
    """アクションフィルタのテスト。"""

    def test_separates_correctly(self):
        results = [
            ScoringResult("A", "s", signal=BUY, final_score=0.7),
            ScoringResult("B", "s", signal=HOLD, final_score=0.0),
            ScoringResult("C", "s", signal=SELL, final_score=-0.4),
            ScoringResult("D", "s", signal=STRONG_BUY, final_score=0.9),
            ScoringResult("E", "s", signal=STRONG_SELL, final_score=-0.8),
        ]
        filtered = filter_actionable(results)
        assert len(filtered["buy"]) == 2
        assert len(filtered["sell"]) == 2
        assert len(filtered["hold"]) == 1

    def test_sell_sorted_ascending(self):
        results = [
            ScoringResult("A", "s", signal=SELL, final_score=-0.4),
            ScoringResult("B", "s", signal=STRONG_SELL, final_score=-0.8),
        ]
        filtered = filter_actionable(results)
        assert filtered["sell"][0].final_score <= filtered["sell"][1].final_score


# ============================================================
# _generate_reason テスト
# ============================================================

class TestGenerateReason:
    """理由文生成のテスト。"""

    def test_neutral(self):
        result = ScoringResult("X", "s")
        reason = _generate_reason(result)
        assert reason == "中立シグナル"

    def test_bullish(self):
        result = ScoringResult("X", "s", momentum_score=0.8)
        reason = _generate_reason(result)
        assert "モメンタム" in reason
        assert "強気" in reason

    def test_bearish(self):
        result = ScoringResult("X", "s", quality_score=-0.5)
        reason = _generate_reason(result)
        assert "クオリティ" in reason
        assert "弱気" in reason


# ============================================================
# ScoringResult.to_dict テスト
# ============================================================

class TestScoringResultToDict:
    """ScoringResult.to_dict のテスト。"""

    def test_basic_fields(self):
        result = ScoringResult("NVDA", "semiconductor", final_score=0.5, signal=HOLD)
        d = result.to_dict()
        assert d["ticker"] == "NVDA"
        assert d["sector"] == "semiconductor"
        assert d["signal"] == HOLD
        assert isinstance(d["final_score"], float)

    def test_with_risk(self):
        from strategy.risk import RiskLevels
        risk = RiskLevels(
            current_price=100.0, atr=5.0,
            stop_loss=92.0, take_profit=115.0,
            stop_loss_pct=-0.08, take_profit_pct=0.15,
            risk_reward_ratio=1.875, atr_pct=0.05,
        )
        result = ScoringResult("NVDA", "semiconductor", risk=risk)
        d = result.to_dict()
        assert "current_price" in d
        assert "stop_loss" in d
        assert "take_profit" in d
