"""
Swing Trade Signal System — 統合スコアリング

4つのファクタースコアを加重平均し、最終シグナルを決定する。

パイプライン:
  1. 各ファクタースコア (Momentum, Value, Quality, Sentiment) を受け取る
  2. FACTOR_WEIGHTS に従って加重平均 → 最終スコア
  3. SIGNAL_THRESHOLDS に従ってシグナル判定
  4. risk.py を呼び出して Stop Loss / Take Profit を付与

シグナル種別:
  STRONG_BUY  (≥ 0.8)
  BUY         (0.6 〜 0.8)
  HOLD        (-0.3 〜 0.6)
  SELL        (-0.6 〜 -0.3)
  STRONG_SELL (< -0.6)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from analysis.momentum import calc_momentum_score,calc_momentum_detail
from analysis.quality import calc_quality_score
from analysis.sentiment import calc_sentiment_score
from analysis.technical_quality import calc_technical_quality_score
from analysis.value import calc_value_score
import config.settings as _cfg
from config.settings import SIGNAL_THRESHOLDS
from strategy.risk import RiskLevels, calc_risk_levels

logger = logging.getLogger(__name__)


# ============================================================
# シグナル定数
# ============================================================

STRONG_BUY = "STRONG_BUY"
BUY = "BUY"
HOLD = "HOLD"
SELL = "SELL"
STRONG_SELL = "STRONG_SELL"


# ============================================================
# データクラス
# ============================================================

@dataclass
class ScoringResult:
    """1銘柄の統合スコアリング結果。"""

    ticker: str
    sector: str

    # 個別ファクタースコア (-1.0 〜 +1.0)
    momentum_score: float = 0.0
    value_score: float = 0.0
    quality_score: float = 0.0
    sentiment_score: float = 0.0

    # Claude API スコア (使用時のみ)
    claude_score: float | None = None

    # 統合
    final_score: float = 0.0
    signal: str = HOLD

    # リスク管理
    risk: RiskLevels | None = None

    # メタ
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """辞書形式で返す（CSV出力・通知用）。"""
        d: dict[str, Any] = {
            "ticker": self.ticker,
            "sector": self.sector,
            "momentum": round(self.momentum_score, 4),
            "value": round(self.value_score, 4),
            "quality": round(self.quality_score, 4),
            "sentiment": round(self.sentiment_score, 4),
            "final_score": round(self.final_score, 4),
            "signal": self.signal,
            "reason": self.reason,
        }
        if self.claude_score is not None:
            d["claude_score"] = round(self.claude_score, 4)
        if self.risk:
            d["current_price"] = self.risk.current_price
            d["stop_loss"] = self.risk.stop_loss
            d["take_profit"] = self.risk.take_profit
            d["stop_loss_pct"] = round(self.risk.stop_loss_pct * 100, 2)
            d["take_profit_pct"] = round(self.risk.take_profit_pct * 100, 2)
            d["risk_reward"] = self.risk.risk_reward_ratio
            d["atr"] = round(self.risk.atr, 4)
        return d


# ============================================================
# 公開 API
# ============================================================

def score_ticker(
    ticker: str,
    sector: str,
    price_df: pd.DataFrame | None = None,
    fundamentals: dict[str, Any] | None = None,
    headlines: list[str] | None = None,
    claude_score: float | None = None,
    spy_df: pd.DataFrame | None = None,
) -> ScoringResult:
    """1銘柄の統合スコアリングを行う。

    Args:
        spy_df: SPY の OHLCV DataFrame（モメンタムの超過リターン計算用）。
    """
    result = ScoringResult(ticker=ticker, sector=sector)

    try:
        # ---- 1. 個別ファクタースコア詳細算出 ----
        mom_detail = calc_momentum_detail(price_df, spy_df=spy_df) if price_df is not None else {}
        result.momentum_score = mom_detail.get("score", 0.0)

        result.value_score = (
            calc_value_score(fundamentals, sector)
            if fundamentals is not None
            else 0.0
        )
        result.quality_score = (
            calc_quality_score(fundamentals)
            if fundamentals is not None
            else 0.0
        )
        result.sentiment_score = (
            calc_sentiment_score(headlines)
            if headlines is not None
            else 0.0
        )

        # ---- 2. Claude API スコアで上書き（フルモード時） ----
        # sentiment_available: ニュース情報が実際に存在するかを示すフラグ
        # True  → Mom×0.50 + Q×0.30 + S×0.20 で計算
        # False → S枠を M/Q に比例配分: Mom×0.625 + Q×0.375
        if claude_score is not None:
            result.claude_score = claude_score
            sentiment_for_composite = claude_score
            sentiment_available = True   # Claude がスコアを返した = ニュースあり
        else:
            sentiment_for_composite = result.sentiment_score
            # キーワードベース: headlines が空でなければニュースあり扱い
            sentiment_available = bool(headlines)

        # ---- 3. 統合スコア ----
        result.final_score = _calc_composite(
            result.momentum_score,
            result.value_score,
            result.quality_score,
            sentiment_for_composite,
            sentiment_available=sentiment_available,
        )

        # ---- 4. シグナル判定 ----
        result.signal = determine_signal(result.final_score)
        result.reason = _generate_reason(result)

        # ---- 5. SL/TP 算出 ----
        if price_df is not None:
            result.risk = calc_risk_levels(price_df)

        logger.debug(
            "スコアリング [%s]: M=%.2f V=%.2f Q=%.2f S=%.2f → %.2f (%s)",
            ticker,
            result.momentum_score,
            result.value_score,
            result.quality_score,
            sentiment_for_composite,
            result.final_score,
            result.signal,
        )

    except Exception:
        logger.error("スコアリングエラー: %s", ticker, exc_info=True)
        result.signal = HOLD
        result.final_score = 0.0
        result.reason = "スコアリングエラー"

    return result


def score_universe(
    universe: dict[str, dict[str, Any]],
    spy_df: pd.DataFrame | None = None,
) -> list[ScoringResult]:
    """ユニバース全体をスコアリングし、スコア降順で返す。

    Args:
        universe: ティッカー → {
            "sector": str,
            "price_df": DataFrame | None,
            "fundamentals": dict | None,
            "headlines": list[str] | None,
            "claude_score": float | None,  (オプション)
        } の辞書。
        spy_df: SPY の OHLCV DataFrame（超過リターン計算用）。

    Returns:
        ScoringResult のリスト（final_score 降順）。
    """
    results: list[ScoringResult] = []

    for ticker, data in universe.items():
        result = score_ticker(
            ticker=ticker,
            sector=data.get("sector", "unknown"),
            price_df=data.get("price_df"),
            fundamentals=data.get("fundamentals"),
            headlines=data.get("headlines"),
            claude_score=data.get("claude_score"),
            spy_df=spy_df,
        )
        results.append(result)

    # スコア降順でソート
    results.sort(key=lambda r: r.final_score, reverse=True)

    logger.info(
        "ユニバーススコアリング完了: %d 銘柄 (BUY系: %d, SELL系: %d)",
        len(results),
        sum(1 for r in results if r.signal in (STRONG_BUY, BUY)),
        sum(1 for r in results if r.signal in (SELL, STRONG_SELL)),
    )
    return results


# ============================================================
# シグナル判定
# ============================================================

def determine_signal(score: float) -> str:
    """統合スコアからシグナル文字列を判定する。

    Args:
        score: 統合スコア (-1.0 〜 +1.0)。

    Returns:
        シグナル文字列。
    """
    if score >= SIGNAL_THRESHOLDS.strong_buy:
        return STRONG_BUY
    if score >= SIGNAL_THRESHOLDS.buy:
        return BUY
    if score <= SIGNAL_THRESHOLDS.strong_sell:
        return STRONG_SELL
    if score <= SIGNAL_THRESHOLDS.sell:
        return SELL
    return HOLD


# ============================================================
# 内部ロジック
# ============================================================

def _calc_composite(
    momentum: float,
    value: float,
    quality: float,
    sentiment: float,
    sentiment_available: bool = True,
) -> float:
    """4ファクターの加重平均を算出する。

    sentiment_available=False のとき、sentimentウェイト(20%)を
    MomentumとQualityに比例配分して再正規化する。
      通常:       Mom×0.50 + Q×0.30 + S×0.20
      ニュースなし: Mom×0.625 + Q×0.375 (比率を維持したまま100%に引き伸ばし)
    """
    w = _cfg.FACTOR_WEIGHTS
    if not sentiment_available:
        mq_total = w.momentum + w.quality  # 0.80
        m_w = w.momentum / mq_total        # 0.625
        q_w = w.quality  / mq_total        # 0.375
        composite = momentum * m_w + value * w.value + quality * q_w
    else:
        composite = (
            momentum * w.momentum
            + value * w.value
            + quality * w.quality
            + sentiment * w.sentiment
        )
    return float(np.clip(composite, -1.0, 1.0))


def _generate_reason(result: ScoringResult) -> str:
    """スコアリング結果から人間に読みやすい理由文を生成する。"""
    parts: list[str] = []

    # 各ファクターの貢献を説明
    factors = [
        ("モメンタム", result.momentum_score),
        ("バリュー", result.value_score),
        ("クオリティ", result.quality_score),
        ("センチメント", result.claude_score if result.claude_score is not None else result.sentiment_score),
    ]

    # 絶対値が大きい順にソート
    factors.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, score in factors:
        if abs(score) < 0.1:
            continue
        direction = "強気" if score > 0 else "弱気"
        parts.append(f"{name}{direction}({score:+.2f})")

    if not parts:
        return "中立シグナル"

    return " / ".join(parts)


# ============================================================
# フィルタリングユーティリティ
# ============================================================

def filter_actionable(
    results: list[ScoringResult],
) -> dict[str, list[ScoringResult]]:
    """アクション可能なシグナルをフィルタリングする。

    Returns:
        "buy":  STRONG_BUY + BUY のリスト (スコア降順)
        "sell": SELL + STRONG_SELL のリスト (スコア昇順)
        "hold": HOLD のリスト
    """
    buy_signals = [r for r in results if r.signal in (STRONG_BUY, BUY)]
    sell_signals = [r for r in results if r.signal in (SELL, STRONG_SELL)]
    hold_signals = [r for r in results if r.signal == HOLD]

    # sell はスコア昇順（最も弱気が先頭）
    sell_signals.sort(key=lambda r: r.final_score)

    return {
        "buy": buy_signals,
        "sell": sell_signals,
        "hold": hold_signals,
    }
