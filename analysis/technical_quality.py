"""
Swing Trade Signal System — テクニカル品質ファクター

Qualityファクター（ROE等のファンダメンタル）の代替。
OHLCVデータのみで算出するため先読みバイアスなし。

従来の Quality 問題点:
  1. ファンダメンタル（ROE等）は現時点の値をyfinanceで取得 → 先読みバイアス
  2. 30日スイングと無関係な長期財務指標 → スコア-P/L相関 0.042

新ファクター: 2指標の加重平均
  - ディップ品質 (60%): ディップ中の出来高が低いほど良質な押し目
  - OBVトレンド   (40%): 上昇日出来高 > 下落日出来高 = 機関の買い蓄積

入力: 日足 OHLCV の DataFrame
出力: スコア (-1.0 〜 +1.0)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ディップ品質: 直近N日の出来高 vs 参照期間の平均
_DIP_LOOKBACK: int = 10   # ディップ中とみなす期間（営業日）
_REF_LOOKBACK: int = 40   # 参照期間（ディップ前の通常状態）

# OBVトレンド: 直近N日の上昇日/下落日出来高比率
_OBV_PERIOD: int = 20

# 2指標のウェイト
_W_DIP: float = 0.60
_W_OBV: float = 0.40


def calc_technical_quality_score(df: pd.DataFrame) -> float:
    """テクニカル品質スコアを算出する。

    Args:
        df: 日足 OHLCV DataFrame（Volume 列必須、50行以上推奨）。

    Returns:
        テクニカル品質スコア (-1.0 〜 +1.0)。データ不足時は 0.0。
    """
    if df is None or df.empty or "Volume" not in df.columns:
        return 0.0
    if len(df) < _REF_LOOKBACK + _DIP_LOOKBACK + 1:
        return 0.0

    try:
        dip_score = _calc_dip_quality(df)
        obv_score = _calc_obv_trend(df)
        score = dip_score * _W_DIP + obv_score * _W_OBV
        return float(np.clip(score, -1.0, 1.0))
    except Exception:
        logger.debug("テクニカル品質計算失敗", exc_info=True)
        return 0.0


def _calc_dip_quality(df: pd.DataFrame) -> float:
    """ディップ品質スコア。

    ディップ中（直近 _DIP_LOOKBACK 日）の出来高が
    参照期間（その前の _REF_LOOKBACK 日）より低いほど高スコア。

    低出来高のディップ = 機関が手放していない = 良質な押し目
    高出来高のディップ = 売り圧力が強い = 悪質な下落

    スコアリング:
      出来高比 < 0.5 (50%減) → +1.0  (ドライアップ = 最良)
      出来高比 = 1.0 (平均的) →  0.0  (中立)
      出来高比 > 1.5 (50%増)  → -1.0  (大商いで売られた = 最悪)
    """
    volumes = df["Volume"].astype(float)

    # 参照出来高: ディップ前 _REF_LOOKBACK 日の平均
    ref_vol = float(volumes.iloc[-(_REF_LOOKBACK + _DIP_LOOKBACK):-_DIP_LOOKBACK].mean())
    # ディップ中の出来高: 直近 _DIP_LOOKBACK 日の平均
    dip_vol = float(volumes.iloc[-_DIP_LOOKBACK:].mean())

    if ref_vol <= 0 or np.isnan(ref_vol) or np.isnan(dip_vol):
        return 0.0

    ratio = dip_vol / ref_vol
    # ratio=0.5 → +1.0, ratio=1.0 → 0.0, ratio=1.5 → -1.0
    return float(np.clip((1.0 - ratio) / 0.50, -1.0, 1.0))


def _calc_obv_trend(df: pd.DataFrame) -> float:
    """OBVトレンドスコア。

    直近 _OBV_PERIOD 日における上昇日出来高の比率で買い蓄積を評価する。

    上昇日に出来高が集中 = 機関が積極的に買っている = 高スコア
    下落日に出来高が集中 = 機関が売っている = 低スコア

    スコアリング:
      上昇日出来高比 > 0.75 → +1.0  (強い買い蓄積)
      上昇日出来高比 = 0.50 →  0.0  (中立)
      上昇日出来高比 < 0.25 → -1.0  (強い売り圧力)
    """
    if len(df) < _OBV_PERIOD + 1:
        return 0.0

    recent = df.iloc[-_OBV_PERIOD:].copy()
    close_diff = recent["Close"].astype(float).diff()
    volumes = recent["Volume"].astype(float)

    up_vol = float(volumes[close_diff > 0].sum())
    down_vol = float(volumes[close_diff < 0].sum())
    total = up_vol + down_vol

    if total <= 0 or np.isnan(total):
        return 0.0

    ratio = up_vol / total  # 0.5 = neutral
    # ratio=0.75 → +1.0, ratio=0.50 → 0.0, ratio=0.25 → -1.0
    return float(np.clip((ratio - 0.50) / 0.25, -1.0, 1.0))
