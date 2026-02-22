"""バックテストエンジンのユニットテスト。"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backtest.engine import (
    TradeResult,
    SignalBacktestResult,
    ScreeningBacktestResult,
    _simulate_trade,
    _aggregate_signal_results,
    _aggregate_screening_results,
    MonthlyRecord,
    WARMUP_DAYS,
    MAX_HOLD_DAYS,
    STAGED_TIMEOUT,
    TRAILING_STOP_ATR_MULTIPLIER,
)


# ============================================================
# テスト用ヘルパー
# ============================================================

def _make_price_df(n: int = 300, start_price: float = 100.0) -> pd.DataFrame:
    """テスト用のダミー OHLCV DataFrame を生成する（上昇トレンド）。"""
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    prices = start_price + np.linspace(0, 30, n)  # 緩やかな上昇
    df = pd.DataFrame({
        "Open":   prices * 0.995,
        "High":   prices * 1.02,
        "Low":    prices * 0.98,
        "Close":  prices,
        "Volume": [1_000_000] * n,
    }, index=dates)
    return df


def _make_future_df(
    n: int = 90,
    start_price: float = 100.0,
    pattern: str = "flat",  # "flat" | "down_to_sl" | "gradual_up" | "strong_up" | "spike_then_drop"
    sl_price: float = 92.0,
) -> pd.DataFrame:
    """テスト用の将来価格DataFrame。"""
    dates = pd.date_range("2023-01-05", periods=n, freq="B")

    if pattern == "down_to_sl":
        # 3日目に安値がSLに到達
        highs = [start_price * 1.01] * n
        lows = [start_price * 0.99] * n
        closes = [start_price] * n
        lows[2] = sl_price - 0.5  # SLヒット

    elif pattern == "gradual_up":
        # 緩やかに上昇（30日目で+3%程度 → timeout_30d で決済される）
        closes = [start_price * (1 + 0.001 * i) for i in range(n)]
        highs = [c * 1.005 for c in closes]
        lows = [c * 0.995 for c in closes]

    elif pattern == "strong_up":
        # 強い上昇（30日目で+8%以上 → 保有継続）
        # ATR=5の場合、trail_level = high - 15。closeが小幅変動なのでトレーリング発動しにくい
        closes = [start_price * (1 + 0.003 * i) for i in range(n)]
        highs = [c * 1.005 for c in closes]
        lows = [c * 0.995 for c in closes]

    elif pattern == "spike_then_drop":
        # 急騰後に下落 → トレーリングストップ発動
        # 10日目まで急騰（+20%）→ その後急落
        closes = []
        for i in range(n):
            if i <= 10:
                closes.append(start_price * (1 + 0.02 * i))  # +2%/日
            else:
                # 10日目のピーク(120)から下落
                peak = start_price * 1.20
                closes.append(peak * (1 - 0.015 * (i - 10)))  # -1.5%/日
        highs = [c * 1.005 for c in closes]
        lows = [c * 0.995 for c in closes]

    else:  # flat
        highs = [start_price * 1.005] * n
        lows = [start_price * 0.995] * n
        closes = [start_price] * n

    return pd.DataFrame({
        "Open": [start_price] * n,
        "High": highs,
        "Low":  lows,
        "Close": closes,
        "Volume": [500_000] * n,
    }, index=dates)


# ============================================================
# _simulate_trade のテスト
# ============================================================

class TestSimulateTrade:
    """SL + トレーリングストップ + 段階的タイムアウト判定ロジックのテスト。"""

    BASE = dict(
        ticker="NVDA",
        signal="BUY",
        signal_date="2023-01-04",
        entry_date="2023-01-05",
        entry_price=100.0,
        sl_price=92.0,
        atr=5.0,
    )

    def test_sl_hit(self):
        """安値がSLに到達した場合は損切り。"""
        future = _make_future_df(pattern="down_to_sl", sl_price=92.0)
        trade = _simulate_trade(**self.BASE, future_df=future)
        assert trade.result == "sl_hit"
        assert trade.exit_price == 92.0
        assert trade.pl_pct < 0

    def test_sl_hit_pl_approx_minus_8_pct(self):
        """SLヒット時の損益率は-8%に近い。"""
        future = _make_future_df(pattern="down_to_sl", sl_price=92.0)
        trade = _simulate_trade(**self.BASE, future_df=future)
        assert abs(trade.pl_pct - (-0.08)) < 0.001

    def test_timeout_30d_low_profit(self):
        """30日経過時に含み益+5%未満ならtimeout_30dで決済。"""
        future = _make_future_df(n=90, pattern="flat", sl_price=92.0)
        trade = _simulate_trade(**self.BASE, future_df=future)
        assert trade.result == "timeout_30d"

    def test_staged_timeout_gradual_up(self):
        """緩やかな上昇(+3%@30日) → timeout_30dで決済。"""
        future = _make_future_df(n=90, pattern="gradual_up", sl_price=92.0)
        trade = _simulate_trade(**self.BASE, future_df=future)
        assert trade.result == "timeout_30d"

    def test_trailing_stop_after_spike(self):
        """急騰後に下落 → トレーリングストップで利確。"""
        future = _make_future_df(n=90, pattern="spike_then_drop", sl_price=92.0)
        trade = _simulate_trade(**self.BASE, future_df=future)
        assert trade.result == "trailing_stop"
        assert trade.pl_pct > 0  # 利益で決済

    def test_trailing_stop_exit_price(self):
        """トレーリングストップの決済価格は highest_high - ATR*3.0。"""
        future = _make_future_df(n=90, pattern="spike_then_drop", sl_price=92.0)
        trade = _simulate_trade(**self.BASE, future_df=future)
        # trail_level = highest_high - 5.0 * 3.0 = highest_high - 15.0
        assert trade.result == "trailing_stop"
        assert trade.exit_price > trade.entry_price

    def test_trailing_stop_not_below_entry(self):
        """トレーリングストップはエントリー価格以上でのみ発動。"""
        # ATR=5, trail_multiplier=3.0 → trail_offset=15
        # 高値が114以下なら trail_level = 114-15=99 < entry(100) → 発動しない
        future = _make_future_df(n=90, pattern="flat", sl_price=92.0)
        trade = _simulate_trade(**self.BASE, future_df=future)
        # flatパターンでは高値=100.5、trail_level=100.5-15=85.5 < entry(100) → 発動しない
        assert trade.result != "trailing_stop"

    def test_strong_up_trailing_or_timeout(self):
        """強い上昇 → トレーリングストップまたはtimeout_90dで決済。"""
        future = _make_future_df(n=90, pattern="strong_up", sl_price=92.0)
        trade = _simulate_trade(**self.BASE, future_df=future)
        assert trade.result in ("trailing_stop", "timeout_90d")
        assert trade.pl_pct > 0.15

    def test_empty_future_df(self):
        """将来データが空の場合はtimeout_90d（エントリー価格で決済）。"""
        empty_future = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        trade = _simulate_trade(**self.BASE, future_df=empty_future)
        assert trade.result == "timeout_90d"
        assert trade.exit_price == self.BASE["entry_price"]
        assert trade.pl_pct == 0.0

    def test_trade_result_fields(self):
        """TradeResultの全フィールドが正しく設定される。"""
        future = _make_future_df(pattern="down_to_sl", sl_price=92.0)
        trade = _simulate_trade(**self.BASE, future_df=future)
        assert trade.ticker == "NVDA"
        assert trade.signal == "BUY"
        assert trade.signal_date == "2023-01-04"
        assert trade.entry_date == "2023-01-05"
        assert trade.entry_price == 100.0
        assert trade.sl_price == 92.0

    def test_no_fixed_tp(self):
        """固定TPは撤廃 — tp_priceは0.0。"""
        future = _make_future_df(n=90, pattern="strong_up", sl_price=92.0)
        trade = _simulate_trade(**self.BASE, future_df=future)
        assert trade.tp_price == 0.0


# ============================================================
# TradeResult.to_dict のテスト
# ============================================================

class TestTradeResultToDict:
    def test_to_dict_keys(self):
        trade = TradeResult(
            ticker="AAPL", signal="STRONG_BUY", signal_date="2023-01-04",
            entry_date="2023-01-05", entry_price=150.0,
            exit_date="2023-01-10", exit_price=165.0,
            result="trailing_stop", pl_pct=0.10,
            sl_price=138.0, tp_price=0.0,
        )
        d = trade.to_dict()
        assert set(d.keys()) == {
            "ticker", "signal", "signal_date", "entry_date", "entry_price",
            "exit_date", "exit_price", "result", "pl_pct", "sl_price", "tp_price",
        }

    def test_to_dict_pl_pct_in_percent(self):
        trade = TradeResult(
            ticker="AAPL", signal="BUY", signal_date="2023-01-04",
            entry_date="2023-01-05", entry_price=100.0,
            exit_date="2023-01-10", exit_price=108.0,
            result="timeout_30d", pl_pct=0.08,
            sl_price=92.0, tp_price=0.0,
        )
        assert trade.to_dict()["pl_pct"] == 8.0  # パーセント表示


# ============================================================
# _aggregate_signal_results のテスト
# ============================================================

class TestAggregateSignalResults:
    def _make_trade(self, result: str, pl_pct: float, signal: str = "BUY", year: str = "2023") -> TradeResult:
        return TradeResult(
            ticker="TEST", signal=signal, signal_date=f"{year}-01-04",
            entry_date=f"{year}-01-05", entry_price=100.0,
            exit_date=f"{year}-01-15", exit_price=100.0 * (1 + pl_pct),
            result=result, pl_pct=pl_pct,
            sl_price=92.0, tp_price=0.0,
        )

    def test_empty_trades(self):
        result = _aggregate_signal_results([])
        assert result.total_trades == 0
        assert result.win_rate == 0.0

    def test_win_rate_calculation(self):
        """勝率はpl_pct > 0 の割合で算出。"""
        trades = [
            self._make_trade("trailing_stop", 0.15),
            self._make_trade("timeout_90d", 0.12),
            self._make_trade("sl_hit", -0.08),
            self._make_trade("timeout_30d", 0.02),
        ]
        result = _aggregate_signal_results(trades)
        assert result.total_trades == 4
        # pl_pct > 0 は 3件（0.15, 0.12, 0.02）
        assert abs(result.win_rate - 0.75) < 0.001

    def test_avg_pl_pct(self):
        trades = [
            self._make_trade("trailing_stop", 0.10),
            self._make_trade("sl_hit", -0.08),
        ]
        result = _aggregate_signal_results(trades)
        assert abs(result.avg_pl_pct - 0.01) < 0.0001

    def test_by_result_counts(self):
        trades = [
            self._make_trade("trailing_stop", 0.10),
            self._make_trade("timeout_30d", 0.02),
            self._make_trade("sl_hit", -0.08),
            self._make_trade("timeout_60d", 0.03),
        ]
        result = _aggregate_signal_results(trades)
        assert result.by_result["trailing_stop"] == 1
        assert result.by_result["timeout_30d"] == 1
        assert result.by_result["timeout_60d"] == 1
        assert result.by_result["sl_hit"] == 1

    def test_by_signal_breakdown(self):
        trades = [
            self._make_trade("trailing_stop", 0.15, "STRONG_BUY"),
            self._make_trade("sl_hit", -0.08, "BUY"),
        ]
        result = _aggregate_signal_results(trades)
        assert "STRONG_BUY" in result.by_signal
        assert "BUY" in result.by_signal
        assert result.by_signal["STRONG_BUY"]["count"] == 1
        assert result.by_signal["BUY"]["count"] == 1

    def test_by_year_breakdown(self):
        trades = [
            self._make_trade("trailing_stop", 0.10, year="2022"),
            self._make_trade("sl_hit", -0.08, year="2023"),
            self._make_trade("timeout_30d", 0.12, year="2023"),
        ]
        result = _aggregate_signal_results(trades)
        assert "2022" in result.by_year
        assert "2023" in result.by_year
        assert result.by_year["2022"]["count"] == 1
        assert result.by_year["2023"]["count"] == 2


# ============================================================
# _aggregate_screening_results のテスト
# ============================================================

class TestAggregateScreeningResults:
    def _make_record(self, sel_ret: float, all_ret: float, month: str = "2023-01") -> MonthlyRecord:
        return MonthlyRecord(
            month=month,
            selected_tickers=["NVDA", "AAPL"],
            selected_return=sel_ret,
            all_return=all_ret,
            outperformance=sel_ret - all_ret,
        )

    def test_empty_records(self):
        result = _aggregate_screening_results([])
        assert result.total_months == 0

    def test_avg_outperformance(self):
        records = [
            self._make_record(0.05, 0.02, "2023-01"),
            self._make_record(0.01, 0.03, "2023-02"),
        ]
        result = _aggregate_screening_results(records)
        assert result.total_months == 2
        assert abs(result.avg_selected_return - 0.03) < 0.0001
        assert abs(result.avg_all_return - 0.025) < 0.0001

    def test_win_months_count(self):
        records = [
            self._make_record(0.05, 0.02, "2023-01"),  # outperf +
            self._make_record(0.01, 0.03, "2023-02"),  # outperf -
            self._make_record(0.04, 0.01, "2023-03"),  # outperf +
        ]
        result = _aggregate_screening_results(records)
        assert result.win_months == 2
        assert result.total_months == 3
