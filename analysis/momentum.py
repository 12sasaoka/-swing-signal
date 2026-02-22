"""
Swing Trade Signal System — モメンタムファクター計算

3層構造でモメンタムスコアを算出する:
  - 短期モメンタム (30%): 1週間リターン、1ヶ月リターン、RSI(14)、相対出来高
  - 中期モメンタム (50%): 3ヶ月超過リターン(vs SPY)、MACD、50日MA乖離率、ADX
  - 長期モメンタム (20%): 12-1ヶ月モメンタム、200日MA乖離率、6ヶ月超過リターン(vs SPY)

全リターン指標は銘柄固有のボラティリティで正規化（P0: ボラ調整）。
ADX（トレンド強度）を中期レイヤーに追加（P1）。

入力: 日足 OHLCV の DataFrame (columns: Open, High, Low, Close, Volume)
出力: スコア (-1.0 〜 +1.0)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config.settings import TECHNICAL_PARAMS

logger = logging.getLogger(__name__)

# 層別ウェイト
_W_SHORT: float = 0.30
_W_MID: float = 0.50
_W_LONG: float = 0.20

# ボラティリティ計算のルックバック期間
_VOL_LOOKBACK: int = 60


# ============================================================
# 公開 API
# ============================================================

def calc_momentum_score(
    df: pd.DataFrame,
    spy_df: pd.DataFrame | None = None,
) -> float:
    """モメンタムファクターの統合スコアを算出する。

    calc_momentum_detail() のラッパー。スコアのみを返す。

    Args:
        df:     日足 OHLCV DataFrame。日付昇順・最低 252 行推奨。
        spy_df: SPY の OHLCV DataFrame（超過リターン計算用、Noneなら絶対リターン使用）。

    Returns:
        モメンタムスコア (-1.0 〜 +1.0)。データ不足時は 0.0。
    """
    return calc_momentum_detail(df, spy_df=spy_df).get("score", 0.0)


def calc_momentum_detail(
    df: pd.DataFrame,
    spy_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """モメンタムの各サブ指標を個別に返す。

    Args:
        df:     日足 OHLCV DataFrame。
        spy_df: SPY の OHLCV DataFrame（超過リターン計算用）。

    Returns:
        score を含むサブ指標辞書。
    """
    if df is None or df.empty or "Close" not in df.columns:
        return {"score": 0.0, "error": "invalid_input"}

    close = df["Close"].astype(float)

    if len(close) < 30:
        return {"score": 0.0, "error": "insufficient_data"}

    try:
        # Volume
        if "Volume" in df.columns:
            volumes = df["Volume"].astype(float)
        else:
            volumes = None

        # SPY close（同じ終了日まで窓掛け）
        spy_close = None
        if spy_df is not None and not spy_df.empty and "Close" in spy_df.columns:
            try:
                current_date = close.index[-1]
                spy_close = spy_df["Close"].astype(float).loc[:current_date]
                if len(spy_close) < 64:  # 3ヶ月分未満はfallback
                    spy_close = None
            except Exception:
                spy_close = None

        # P0: 銘柄固有のボラティリティ（日次標準偏差）
        daily_vol = _calc_volatility(close, _VOL_LOOKBACK)

        # P1: ADXスコア（中期レイヤー用）
        adx_val = _calc_adx(df)
        adx_score = _adx_to_score(adx_val)

        # --- 各層のスコア算出 ---
        short_term = _calc_short_term(close, volumes, daily_vol)
        mid_term = _calc_mid_term(close, spy_close, daily_vol, adx_score)
        long_term = _calc_long_term(close, spy_close, daily_vol)

        total_score = float(np.clip(
            short_term * _W_SHORT + mid_term * _W_MID + long_term * _W_LONG,
            -1.0, 1.0,
        ))

        # --- 追加のサブ指標（ディップ買い判定に使用）---
        ret_1w = _calc_return(close, 5)
        rsi_14 = _calc_rsi(close, TECHNICAL_PARAMS.rsi_period)
        ma200_dev = _calc_ma_deviation(close, TECHNICAL_PARAMS.sma_200)

        # 出来高急増
        if volumes is not None:
            vol_5d_avg = volumes.iloc[-5:].mean() if len(volumes) >= 5 else 0.0
            vol_20d_avg = volumes.iloc[-20:].mean() if len(volumes) >= 20 else 1.0
            vol_ratio = vol_5d_avg / vol_20d_avg if vol_20d_avg > 0 else 1.0
        else:
            vol_ratio = 1.0

        logger.debug(
            "モメンタム: short=%.3f mid=%.3f long=%.3f vol=%.4f adx=%.1f → %.3f",
            short_term, mid_term, long_term, daily_vol, adx_val, total_score,
        )

        return {
            "score": total_score,
            "return_1w": ret_1w,
            "rsi_14": rsi_14,
            "vol_ratio": vol_ratio,
            "ma200_dev": ma200_dev,
            "daily_vol": daily_vol,
            "adx": adx_val,
            "short_term": short_term,
            "mid_term": mid_term,
            "long_term": long_term,
        }

    except Exception:
        logger.error("モメンタム計算で例外発生", exc_info=True)
        return {"score": 0.0, "error": "exception"}


# ============================================================
# P0: ボラティリティ算出
# ============================================================

def _calc_volatility(close: pd.Series, lookback: int = 60) -> float:
    """直近N日の日次リターンの標準偏差（日次ボラティリティ）を算出する。

    各リターン指標の正規化に使用。銘柄固有のボラティリティで割ることで
    「その銘柄にとって異常な動きかどうか」を評価できる。

    Returns:
        日次ボラティリティ。データ不足時は 0.0（固定値フォールバックを使用）。
    """
    if len(close) < lookback + 1:
        return 0.0
    daily_returns = close.pct_change().iloc[-lookback:]
    std = daily_returns.std()
    if np.isnan(std) or std <= 0:
        return 0.0
    return float(std)


def _vol_adjusted_ref(daily_vol: float, period_days: int, fallback: float) -> float:
    """ボラティリティ調整された正規化基準値を返す。

    基準値 = daily_vol × √period_days（ランダムウォーク近似）。
    daily_vol が無効な場合は固定の fallback を使用。

    Args:
        daily_vol:    日次ボラティリティ。
        period_days:  リターン計算の日数。
        fallback:     ボラデータなし時の固定値。

    Returns:
        正規化基準値（最低 0.01）。
    """
    if daily_vol > 0:
        return max(daily_vol * np.sqrt(period_days), 0.01)
    return fallback


# ============================================================
# P1: ADX（Average Directional Index）
# ============================================================

def _calc_adx(df: pd.DataFrame, period: int = 14) -> float:
    """ADX (Average Directional Index) を算出する。

    Wilder's smoothing を使用。トレンドの「強さ」（方向ではない）を測定する。
    ADX > 25 = 強いトレンド、ADX < 20 = レンジ相場。

    Returns:
        ADX値 (0〜100)。データ不足時は 20.0（中立）。
    """
    if df is None or len(df) < period * 3:
        return 20.0

    required = ["High", "Low", "Close"]
    if not all(col in df.columns for col in required):
        return 20.0

    try:
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        close = df["Close"].astype(float)
        prev_close = close.shift(1)

        # True Range
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        # Directional Movement
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = up_move.copy()
        plus_dm[(up_move <= down_move) | (up_move <= 0)] = 0.0

        minus_dm = down_move.copy()
        minus_dm[(down_move <= up_move) | (down_move <= 0)] = 0.0

        # Wilder's smoothing (EMA with alpha=1/period)
        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        smooth_plus = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        smooth_minus = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        # Directional Indicators
        plus_di = 100.0 * smooth_plus / atr
        minus_di = 100.0 * smooth_minus / atr

        # DX → ADX
        di_sum = plus_di + minus_di
        di_diff = (plus_di - minus_di).abs()
        dx = 100.0 * di_diff / di_sum
        dx = dx.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        last_adx = float(adx.iloc[-1])
        return last_adx if not np.isnan(last_adx) else 20.0

    except Exception:
        logger.debug("ADX計算失敗", exc_info=True)
        return 20.0


def _adx_to_score(adx: float) -> float:
    """ADX値をスコアに変換する。

    連続的スコアリング:
      ADX > 35 → +1.0 (非常に強いトレンド)
      ADX = 20 →  0.0 (中立)
      ADX < 5  → -1.0 (トレンドなし/レンジ)
    """
    return float(np.clip((adx - 20.0) / 15.0, -1.0, 1.0))


# ============================================================
# 短期モメンタム (30%)
# ============================================================

def _calc_short_term(
    close: pd.Series,
    volumes: pd.Series | None = None,
    daily_vol: float = 0.0,
) -> float:
    """短期モメンタムスコアを算出する。

    サブ指標:
      - 1週間リターン (30%): ボラ調整正規化
      - 1ヶ月リターン (25%): ボラ調整正規化
      - RSI(14) (30%): 順張りスコア（トレンドフォロー型）
      - 相対出来高 (15%): clip((vol_ratio - 1.0) / 0.5, -1, 1)
    """
    ref_1w = _vol_adjusted_ref(daily_vol, 5, fallback=0.05)
    ref_1m = _vol_adjusted_ref(daily_vol, 21, fallback=0.10)

    ret_1w = _norm(_calc_return(close, 5), ref_1w)
    ret_1m = _norm(_calc_return(close, 21), ref_1m)
    rsi_score = _rsi_to_score(_calc_rsi(close, TECHNICAL_PARAMS.rsi_period))
    vol_score = _calc_volume_score(volumes)

    score = ret_1w * 0.30 + ret_1m * 0.25 + rsi_score * 0.30 + vol_score * 0.15
    return float(np.clip(score, -1.0, 1.0))


# ============================================================
# 中期モメンタム (50%)
# ============================================================

def _calc_mid_term(
    close: pd.Series,
    spy_close: pd.Series | None = None,
    daily_vol: float = 0.0,
    adx_score: float = 0.0,
) -> float:
    """中期モメンタムスコアを算出する。

    サブ指標:
      - 3ヶ月超過リターン vs SPY (30%): ボラ調整正規化
      - MACDシグナル比較 (30%): 正規化MACDヒストグラム
      - 50日MA乖離率 (25%): ボラ調整正規化
      - ADXトレンド強度 (15%): トレンドの明確さを評価
    """
    ref_3m = _vol_adjusted_ref(daily_vol, 63, fallback=0.20)

    # 3ヶ月超過リターン（SPYデータがあれば超過、なければ絶対）
    stock_ret_3m = _calc_return(close, 63)
    if spy_close is not None and len(spy_close) >= 64:
        spy_ret_3m = _calc_return(spy_close, 63)
        ret_3m = stock_ret_3m - spy_ret_3m  # 超過リターン
    else:
        ret_3m = stock_ret_3m  # fallback: 絶対リターン

    ret_3m_norm = _norm(ret_3m, ref_3m)
    macd_score = _calc_macd_score(close)
    # 50日MA乖離率もボラ調整（期間は50日だが乖離率の基準はボラに連動）
    ref_ma50 = _vol_adjusted_ref(daily_vol, 50, fallback=0.15)
    ma50_dev = _norm(_calc_ma_deviation(close, TECHNICAL_PARAMS.sma_long), ref_ma50)

    score = ret_3m_norm * 0.30 + macd_score * 0.30 + ma50_dev * 0.25 + adx_score * 0.15
    return float(np.clip(score, -1.0, 1.0))


# ============================================================
# 長期モメンタム (20%)
# ============================================================

def _calc_long_term(
    close: pd.Series,
    spy_close: pd.Series | None = None,
    daily_vol: float = 0.0,
) -> float:
    """長期モメンタムスコアを算出する。

    サブ指標:
      - 12-1ヶ月モメンタム (50%): ボラ調整正規化 ※絶対リターン維持
      - 200日MA乖離率 (30%): ボラ調整正規化
      - 6ヶ月超過リターン vs SPY (20%): ボラ調整正規化
    """
    ref_12_1m = _vol_adjusted_ref(daily_vol, 231, fallback=0.30)  # 252-21=231日
    ref_6m = _vol_adjusted_ref(daily_vol, 126, fallback=0.25)
    ref_ma200 = _vol_adjusted_ref(daily_vol, 200, fallback=0.20)

    ret_12_1m = _norm(_calc_return_skip_recent(close, 252, 21), ref_12_1m)
    ma200_dev = _norm(_calc_ma_deviation(close, TECHNICAL_PARAMS.sma_200), ref_ma200)

    # 6ヶ月超過リターン（SPYデータがあれば超過、なければ絶対）
    stock_ret_6m = _calc_return(close, 126)
    if spy_close is not None and len(spy_close) >= 127:
        spy_ret_6m = _calc_return(spy_close, 126)
        ret_6m = stock_ret_6m - spy_ret_6m  # 超過リターン
    else:
        ret_6m = stock_ret_6m  # fallback: 絶対リターン

    ret_6m_norm = _norm(ret_6m, ref_6m)

    score = ret_12_1m * 0.50 + ma200_dev * 0.30 + ret_6m_norm * 0.20
    return float(np.clip(score, -1.0, 1.0))


# ============================================================
# テクニカル指標ヘルパー
# ============================================================

def _calc_return(close: pd.Series, days: int) -> float:
    """N日間リターンを算出する。データ不足時は 0.0。"""
    if len(close) < days + 1:
        return 0.0
    current = close.iloc[-1]
    past = close.iloc[-days - 1]
    if past == 0 or np.isnan(past) or np.isnan(current):
        return 0.0
    return float((current - past) / past)


def _calc_return_skip_recent(
    close: pd.Series,
    lookback: int,
    skip_recent: int,
) -> float:
    """直近 skip_recent 日を除いたリターン（12-1ヶ月モメンタム等）。

    Args:
        close:       終値系列。
        lookback:    起点（例: 252日前）。
        skip_recent: 除外する直近日数（例: 21日）。

    Returns:
        リターン。データ不足時は 0.0。
    """
    if len(close) < lookback + 1:
        return 0.0
    end_price = close.iloc[-skip_recent - 1] if skip_recent < len(close) else close.iloc[-1]
    start_price = close.iloc[-lookback - 1] if lookback < len(close) else close.iloc[0]
    if start_price == 0 or np.isnan(start_price) or np.isnan(end_price):
        return 0.0
    return float((end_price - start_price) / start_price)


def _calc_rsi(close: pd.Series, period: int) -> float:
    """RSI (Relative Strength Index) を算出する。

    Wilder's smoothing (EMA) 方式。

    Returns:
        RSI値 (0〜100)。計算不能時は 50.0（中立）。
    """
    if len(close) < period + 1:
        return 50.0

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    last_avg_gain = avg_gain.iloc[-1]
    last_avg_loss = avg_loss.iloc[-1]

    if last_avg_loss == 0:
        return 100.0 if last_avg_gain > 0 else 50.0

    rs = last_avg_gain / last_avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi)


def _rsi_to_score(rsi: float) -> float:
    """RSI値をモメンタムスコアに変換する（順張り / トレンドフォロー型）。

    強いトレンド（高RSI）をポジティブに評価する:
      RSI > 75  → +0.30 (非常に強いトレンド)
      RSI 60-75 → +0.15 (強いトレンド)
      RSI 40-60 →  0.00 (中立)
      RSI 30-40 → -0.15 (弱い)
      RSI < 30  → -0.30 (非常に弱い)
    """
    if rsi > 75:
        return 0.30
    if rsi > 60:
        return 0.15
    if rsi < 30:
        return -0.30
    if rsi < 40:
        return -0.15
    return 0.0


def _calc_volume_score(volumes: pd.Series | None) -> float:
    """相対出来高スコアを算出する。

    5日平均出来高 / 20日平均出来高 を正規化。
    1.0 = 平均的、>1.5 = 出来高急増、<0.5 = 出来高減少。

    Returns:
        スコア (-1.0 〜 +1.0)。データ不足時は 0.0。
    """
    if volumes is None or len(volumes) < 20:
        return 0.0

    vol_5d = float(volumes.iloc[-5:].mean()) if len(volumes) >= 5 else 0.0
    vol_20d = float(volumes.iloc[-20:].mean())

    if vol_20d <= 0:
        return 0.0

    ratio = vol_5d / vol_20d
    # ratio=1.0 → 0.0, ratio=1.5 → +1.0, ratio=0.5 → -1.0
    return float(np.clip((ratio - 1.0) / 0.5, -1.0, 1.0))


def _calc_roc(close: pd.Series, period: int) -> float:
    """Rate of Change を算出する（後方互換性のため残存）。"""
    if len(close) < period + 1:
        return 0.0
    current = close.iloc[-1]
    past = close.iloc[-period - 1]
    if past == 0 or np.isnan(past) or np.isnan(current):
        return 0.0
    return float((current - past) / past)


def _calc_macd_score(close: pd.Series) -> float:
    """MACD ヒストグラムをスコア化する。

    MACD Line = EMA(12) - EMA(26)
    Signal Line = EMA(9) of MACD Line
    Histogram = MACD - Signal

    ヒストグラムを現在の株価で正規化し、-1〜+1 にクリップする。
    """
    fast = TECHNICAL_PARAMS.macd_fast
    slow = TECHNICAL_PARAMS.macd_slow
    signal_period = TECHNICAL_PARAMS.macd_signal

    if len(close) < slow + signal_period:
        return 0.0

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    current_price = close.iloc[-1]
    if current_price == 0 or np.isnan(current_price):
        return 0.0

    # ヒストグラムを株価で正規化（例: $5 のヒストグラム / $130 の株価 ≈ 0.038）
    normalized = histogram.iloc[-1] / current_price
    # 0.02 を基準として正規化（十分な感度を持たせる）
    return float(np.clip(normalized / 0.02, -1.0, 1.0))


def _calc_ma_deviation(close: pd.Series, period: int) -> float:
    """移動平均からの乖離率を算出する。

    Returns:
        (現在値 - MA) / MA。データ不足時は 0.0。
    """
    if len(close) < period:
        return 0.0

    ma = close.rolling(window=period).mean().iloc[-1]
    current = close.iloc[-1]

    if ma == 0 or np.isnan(ma) or np.isnan(current):
        return 0.0

    return float((current - ma) / ma)


# ============================================================
# ユーティリティ
# ============================================================

def _norm(value: float, reference: float) -> float:
    """値を基準値で割って -1〜+1 にクリップする。

    clip(value / reference, -1, 1)
    """
    if reference == 0:
        return 0.0
    return float(np.clip(value / reference, -1.0, 1.0))
