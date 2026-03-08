"""
Live Trading — ポジション状態管理

positions.json にオープンポジションと決済履歴を保存する。
SL / トレーリングストップ / 段階的タイムアウトのエグジット判定はバックテストと同一ロジック。

株式分割対応:
  SL・ATR・最高値を「エントリー価格比の比率」で保存する。
  毎日 yfinance がエントリー日の株価を分割修正するため、比率から逆算した絶対価格も自動修正される。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd

from backtest.engine import (
    BREAKEVEN_LOCK_ATR,
    BREAKEVEN_TRIGGER_ATR,
    BREAKEVEN_TRIGGER_DAYS,
    MAX_HOLD_DAYS,
    PEAK_TRAIL_ATR_MULT,
    PEAK_TRAIL_MIN_PCT,
    PROGRESSIVE_TRAIL_LEVELS,
    SB_MAX_HOLD_DAYS,
    SB_TRAILING_ATR_MULT,
    STAGED_TIMEOUT,
    TRAILING_STOP_ATR_MULTIPLIER,
)

logger = logging.getLogger(__name__)

# デフォルト保存先
DEFAULT_POSITIONS_PATH = os.path.join(
    os.path.dirname(__file__), "positions.json"
)


# ============================================================
# データクラス
# ============================================================

@dataclass
class LivePosition:
    """保有中のポジション。株式分割に対応するため比率で保存。"""
    ticker: str
    signal: str               # "BUY" | "STRONG_BUY"
    entry_date: str           # "YYYY-MM-DD"
    entry_price: float        # yfinance auto_adjust 済みのエントリー日終値
    sl_pct: float             # SL比率 (例: 0.082 = エントリー比 -8.2%)
    atr_pct: float            # ATR / entry_price (例: 0.041)
    highest_high_ratio: float # highest_high / entry_price (初期値 = 1.0)
    allocated: float          # 配分金額 ($)
    holding_days: int = 0     # 保有営業日数（毎日実行時に再計算）


@dataclass
class PortfolioState:
    """ポートフォリオ全体の状態。"""
    initial_capital: float
    cash: float
    positions: list[LivePosition] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    etf_positions: dict[str, float] = field(default_factory=dict)  # ticker → 保有株数


# ============================================================
# JSON 永続化
# ============================================================

def load_state(path: str = DEFAULT_POSITIONS_PATH) -> PortfolioState:
    """positions.json からポートフォリオ状態を読み込む。ファイルがなければ空状態を返す。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"positions.json が見つかりません: {path}\n"
            "初期化するには: python live/daily.py --init --capital <金額>"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    positions = [LivePosition(**p) for p in data.get("positions", [])]
    return PortfolioState(
        initial_capital=data["initial_capital"],
        cash=data["cash"],
        positions=positions,
        history=data.get("history", []),
        etf_positions=data.get("etf_positions", {}),
    )


def save_state(state: PortfolioState, path: str = DEFAULT_POSITIONS_PATH) -> None:
    """ポートフォリオ状態を positions.json に保存する。"""
    data = {
        "initial_capital": state.initial_capital,
        "cash": round(state.cash, 4),
        "positions": [asdict(p) for p in state.positions],
        "history": state.history,
        "etf_positions": state.etf_positions,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.debug("状態を保存: %s (%d ポジション)", path, len(state.positions))


def init_state(
    initial_capital: float,
    path: str = DEFAULT_POSITIONS_PATH,
) -> PortfolioState:
    """positions.json を新規作成する（初回セットアップ用）。"""
    state = PortfolioState(
        initial_capital=initial_capital,
        cash=initial_capital,
        positions=[],
        history=[],
    )
    save_state(state, path)
    logger.info("ポートフォリオ初期化完了: 初期資金 $%.2f", initial_capital)
    return state


# ============================================================
# holding_days の再計算
# ============================================================

def recalculate_holding_days(positions: list[LivePosition], today: str) -> None:
    """保有中ポジションの holding_days を今日の日付から営業日数で再計算する（インプレース）。"""
    today_dt = pd.Timestamp(today)
    for pos in positions:
        entry_dt = pd.Timestamp(pos.entry_date)
        # bdate_range: 開始日を含む営業日数 - 1 = 保有日数
        days = len(pd.bdate_range(entry_dt, today_dt)) - 1
        pos.holding_days = max(0, days)


# ============================================================
# トレーリングストップ水準計算（バックテストエンジンと同一ロジック）
# ============================================================

def _get_trail_level(
    highest_high: float,
    entry_price: float,
    atr: float,
    trail_mult_override: float | None = None,
    holding_days: int = 0,
) -> float:
    """段階的トレーリングストップの水準を返す。

    BUY : 含み益に応じて 3.5 -> 3.0 -> 2.5 -> 2.0 ATR と縮小
    STRONG_BUY : trail_mult_override=4.0 固定（段階変化なし）
    いずれも PEAK_TRAIL_MIN_PCT=5% ルームおよび BEロックを適用。
    """
    base_mult = trail_mult_override if trail_mult_override is not None else TRAILING_STOP_ATR_MULTIPLIER

    if atr <= 0:
        return highest_high - atr * base_mult

    peak_profit_atr = (highest_high - entry_price) / atr

    # BUY: 段階的縮小  / STRONG_BUY (override指定): 固定倍率
    trail_mult = base_mult
    if trail_mult_override is None:
        for min_atr, mult in PROGRESSIVE_TRAIL_LEVELS:
            if peak_profit_atr >= min_atr:
                trail_mult = mult
                break

    trail_atr = highest_high - atr * trail_mult

    # 直近高値ベーストレーリング（可変ルーム: max(5%, ATR×1.5÷高値)）
    if highest_high > 0:
        pct_room = max(PEAK_TRAIL_MIN_PCT, PEAK_TRAIL_ATR_MULT * atr / highest_high)
        trail_peak = highest_high * (1.0 - pct_room)
    else:
        trail_peak = trail_atr

    trail = max(trail_atr, trail_peak)

    # BEロック: 含み益 >= 1ATR かつ保有日数条件を満たしたら entry+1ATR を床とする
    if peak_profit_atr >= BREAKEVEN_TRIGGER_ATR and holding_days >= BREAKEVEN_TRIGGER_DAYS:
        trail = max(trail, entry_price + atr * BREAKEVEN_LOCK_ATR)

    return trail


# ============================================================
# エグジット判定（バックテストと同一ロジック）
# ============================================================

def check_exit(
    pos: LivePosition,
    adj_entry: float,
    today_high: float,
    today_low: float,
    today_close: float,
    today_open: float | None = None,
    days_to_earnings: int | None = None,
) -> tuple[Optional[str], float]:
    """今日の価格データからエグジット条件を判定する。

    Args:
        pos:              チェック対象ポジション。
        adj_entry:        yfinance で取得したエントリー日の修正済み終値（分割対応）。
        today_high:       今日の高値。
        today_low:        今日の安値。
        today_close:      今日の終値。
        today_open:       今日の始値（ギャップダウン SL 対応）。
        days_to_earnings: 次回決算までのカレンダー日数（None = 不明）。

    Returns:
        (exit_reason, exit_price)
        exit_reason: "pre_earnings" | "sl_hit" | "trailing_stop" |
                     "timeout_30d" | "timeout_60d" | "timeout_90d"
                     エグジット条件なしなら None。
        exit_price:  決済価格（0.0 ならエグジットなし）。
    """
    # 比率から絶対価格を逆算（株式分割対応）
    sl_abs = adj_entry * (1.0 - pos.sl_pct)
    atr_abs = adj_entry * pos.atr_pct
    highest_high = adj_entry * pos.highest_high_ratio

    # STRONG_BUY / BUY でトレール倍率・最大保有日数を分岐
    is_strong_buy = (pos.signal == "STRONG_BUY")
    trail_mult_override = SB_TRAILING_ATR_MULT if is_strong_buy else None
    max_hold = SB_MAX_HOLD_DAYS if is_strong_buy else MAX_HOLD_DAYS

    trail_level = _get_trail_level(
        highest_high, adj_entry, atr_abs,
        trail_mult_override=trail_mult_override,
        holding_days=pos.holding_days,
    )

    # 0) 決算前強制決済（5カレンダー日以内）
    if days_to_earnings is not None and 0 < days_to_earnings <= 5:
        return "pre_earnings", today_close

    # 1) ストップロス（ギャップダウン対応: 始値 < SL なら始値で決済）
    if today_low <= sl_abs:
        exit_price = min(sl_abs, today_open) if today_open is not None else sl_abs
        return "sl_hit", exit_price

    # 2) トレーリングストップ（trail_level がエントリー価格より上の場合のみ発動）
    if trail_level > adj_entry and today_close <= trail_level:
        return "trailing_stop", trail_level

    # 3) 段階的タイムアウト
    pl_pct = (today_close - adj_entry) / adj_entry
    for threshold_days, min_profit in STAGED_TIMEOUT:
        if pos.holding_days >= threshold_days and pl_pct < min_profit:
            return f"timeout_{threshold_days}d", today_close

    # 4) 絶対タイムアウト（BUY: 60日 / STRONG_BUY: 90日）
    if pos.holding_days >= max_hold:
        return f"timeout_{max_hold}d", today_close

    return None, 0.0


def update_highest_high(pos: LivePosition, adj_entry: float, today_high: float) -> None:
    """今日の高値で highest_high_ratio を更新する（インプレース）。"""
    current_hh = adj_entry * pos.highest_high_ratio
    if today_high > current_hh:
        pos.highest_high_ratio = today_high / adj_entry
