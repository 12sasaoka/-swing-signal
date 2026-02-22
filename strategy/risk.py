"""
Swing Trade Signal System — ATR ベースリスク管理

Average True Range (ATR) を用いて銘柄ごとに動的な
Stop Loss / Take Profit 価格を算出する。

計算ロジック:
  1. ATR(14): 過去14日の True Range の平均
  2. Stop Loss  = max(現在値 − ATR × 2.0, 現在値 × 0.92)
     → ATRベースSLとハードストップ-8%の大きい方（損失が小さい方）を採用
  3. Take Profit = 現在値 + ATR × 4.0

入力: 日足 OHLCV の DataFrame
出力: RiskLevels(stop_loss, take_profit, atr, current_price, ...)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config.settings import ATR_PARAMS, TRADE_RULES

logger = logging.getLogger(__name__)


# ============================================================
# データクラス
# ============================================================

@dataclass(frozen=True)
class RiskLevels:
    """1銘柄のリスク管理パラメータ。"""

    current_price: float
    atr: float
    stop_loss: float
    take_profit: float
    stop_loss_pct: float        # 現在値からの下落率 (負の値)
    take_profit_pct: float      # 現在値からの上昇率 (正の値)
    risk_reward_ratio: float    # リスクリワード比 (TP距離 / SL距離)
    atr_pct: float              # ATR / 現在値 (ボラティリティ%)

    def to_dict(self) -> dict[str, float]:
        """辞書形式で返す。"""
        return {
            "current_price": self.current_price,
            "atr": round(self.atr, 4),
            "stop_loss": round(self.stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
            "stop_loss_pct": round(self.stop_loss_pct, 4),
            "take_profit_pct": round(self.take_profit_pct, 4),
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "atr_pct": round(self.atr_pct, 4),
        }


# ============================================================
# 公開 API
# ============================================================

def calc_risk_levels(
    df: pd.DataFrame,
    atr_period: int | None = None,
    sl_multiplier: float | None = None,
    tp_multiplier: float | None = None,
) -> RiskLevels | None:
    """日足 OHLCV データから ATR ベースの SL/TP を算出する。

    Args:
        df:            日足 OHLCV DataFrame (columns: High, Low, Close)。
                       日付昇順・最低 atr_period+1 行必要。
        atr_period:    ATR 計算期間。None の場合は settings のデフォルト。
        sl_multiplier: SL 乗数。None の場合は settings のデフォルト。
        tp_multiplier: TP 乗数。None の場合は settings のデフォルト。

    Returns:
        RiskLevels インスタンス。データ不足時は None。
    """
    if df is None or df.empty:
        logger.warning("リスク計算: 入力データが空")
        return None

    required = {"High", "Low", "Close"}
    if not required.issubset(df.columns):
        logger.warning("リスク計算: 必要カラム不足 (要: %s)", required)
        return None

    period = atr_period or ATR_PARAMS.period
    sl_mult = sl_multiplier or ATR_PARAMS.stop_loss_multiplier
    tp_mult = tp_multiplier or ATR_PARAMS.take_profit_multiplier

    if len(df) < period + 1:
        logger.warning(
            "リスク計算: データ不足 (%d行 < %d行)",
            len(df), period + 1,
        )
        return None

    try:
        atr = calc_atr(df, period)
        if atr is None or atr <= 0:
            logger.warning("リスク計算: ATR が無効 (atr=%s)", atr)
            return None

        current_price = float(df["Close"].iloc[-1])
        if current_price <= 0 or np.isnan(current_price):
            logger.warning("リスク計算: 現在値が無効 (%s)", current_price)
            return None

        return _build_risk_levels(current_price, atr, sl_mult, tp_mult)

    except Exception:
        logger.error("リスク計算で例外発生", exc_info=True)
        return None


def calc_atr(
    df: pd.DataFrame,
    period: int | None = None,
) -> float | None:
    """ATR (Average True Range) を算出する。

    True Range = max(
        当日高値 - 当日安値,
        |当日高値 - 前日終値|,
        |前日終値 - 当日安値|
    )
    ATR = True Range の period 日間の移動平均

    Args:
        df:     日足 OHLCV DataFrame (columns: High, Low, Close)。
        period: ATR の平均期間。None の場合は settings のデフォルト。

    Returns:
        ATR 値。計算不能時は None。
    """
    if df is None or df.empty:
        return None

    period = period or ATR_PARAMS.period

    if len(df) < period + 1:
        return None

    try:
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        close = df["Close"].astype(float)
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (prev_close - low).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_series = true_range.rolling(window=period).mean()

        atr_value = atr_series.iloc[-1]
        if np.isnan(atr_value):
            return None

        return float(atr_value)

    except Exception:
        logger.error("ATR 計算で例外発生", exc_info=True)
        return None


def calc_risk_levels_from_price(
    current_price: float,
    atr: float,
    sl_multiplier: float | None = None,
    tp_multiplier: float | None = None,
) -> RiskLevels | None:
    """現在値と ATR から直接 SL/TP を算出する（ATR を事前計算済みの場合）。

    Args:
        current_price: 現在の株価。
        atr:           ATR 値。
        sl_multiplier: SL 乗数。
        tp_multiplier: TP 乗数。

    Returns:
        RiskLevels インスタンス。入力不正時は None。
    """
    if current_price <= 0 or atr <= 0:
        return None

    sl_mult = sl_multiplier or ATR_PARAMS.stop_loss_multiplier
    tp_mult = tp_multiplier or ATR_PARAMS.take_profit_multiplier

    return _build_risk_levels(current_price, atr, sl_mult, tp_mult)


# ============================================================
# 内部ロジック
# ============================================================

def _build_risk_levels(
    current_price: float,
    atr: float,
    sl_multiplier: float,
    tp_multiplier: float,
) -> RiskLevels:
    """ATRベースの動的リスク管理。

    SL = 現在値 − ATR × sl_multiplier  (ボラティリティ適応)
    TP = 現在値 + ATR × tp_multiplier  (利益を伸ばす)

    ハードストップ(-8%)をフロアとして維持し、
    ATRベースのSLがそれより広い場合はATRベースを採用する。
    """
    # --- 損切り（ATRベース、-8%をフロア） ---
    atr_sl = current_price - atr * sl_multiplier
    hard_sl = current_price * (1.0 + TRADE_RULES.hard_stop_loss_pct)  # -8%
    # ATR-SL と ハード-SL の大きい方（= 損失が小さい方）を採用
    stop_loss = max(atr_sl, hard_sl)

    sl_pct = (stop_loss - current_price) / current_price  # 負の値

    # --- 利確目安（ATRベース） ---
    take_profit = current_price + atr * tp_multiplier
    tp_pct = (take_profit - current_price) / current_price

    # リスクリワード比
    sl_distance = abs(current_price - stop_loss)
    tp_distance = abs(take_profit - current_price)
    rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0.0

    # ATR% (参考値として保持)
    atr_pct = atr / current_price if current_price > 0 else 0.0

    return RiskLevels(
        current_price=round(current_price, 2),
        atr=atr,
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
        stop_loss_pct=sl_pct,
        take_profit_pct=tp_pct,
        risk_reward_ratio=rr_ratio,
        atr_pct=atr_pct,
    )