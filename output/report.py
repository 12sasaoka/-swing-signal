"""
Swing Trade Signal System — レポート出力

分析結果を2つの形式で出力する:
  1. CSV ファイル: output/results_YYYYMMDD_HHMMSS.csv
  2. コンソール: BUY/SELL 候補を見やすいテーブル形式で表示

ScoringResult のリストを受け取り、人間が手動発注するための
判断材料として整形する。
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import OUTPUT_DIR
from strategy.scorer import (
    BUY,
    HOLD,
    SELL,
    STRONG_BUY,
    STRONG_SELL,
    ScoringResult,
    filter_actionable,
)

logger = logging.getLogger(__name__)


# ============================================================
# CSV カラム定義
# ============================================================

_CSV_COLUMNS: list[str] = [
    "ticker",
    "sector",
    "signal",
    "final_score",
    "momentum",
    "value",
    "quality",
    "sentiment",
    "current_price",
    "stop_loss",
    "stop_loss_pct",
    "take_profit",
    "take_profit_pct",
    "risk_reward",
    "atr",
    "reason",
]


# ============================================================
# 公開 API
# ============================================================

def generate_report(
    results: list[ScoringResult],
    output_dir: Path | str | None = None,
    show_console: bool = True,
    save_csv: bool = True,
) -> Path | None:
    """分析結果のレポートを生成する。

    Args:
        results:     ScoringResult のリスト（スコア降順推奨）。
        output_dir:  CSV 保存先ディレクトリ。None の場合はデフォルト。
        show_console: True の場合はコンソールにサマリーを表示。
        save_csv:     True の場合は CSV ファイルを保存。

    Returns:
        保存した CSV ファイルのパス。save_csv=False または失敗時は None。
    """
    if not results:
        logger.warning("レポート生成: 結果が空")
        print("\n⚠ 分析結果が空です。レポートを生成できません。")
        return None

    csv_path: Path | None = None

    if save_csv:
        csv_path = _save_csv(results, output_dir)

    if show_console:
        _print_console(results)

    return csv_path


def save_csv_only(
    results: list[ScoringResult],
    output_dir: Path | str | None = None,
) -> Path | None:
    """CSV のみ保存する（コンソール出力なし）。"""
    return _save_csv(results, output_dir)


def print_console_only(results: list[ScoringResult]) -> None:
    """コンソール出力のみ行う（CSV 保存なし）。"""
    _print_console(results)


def print_single_ticker(result: ScoringResult) -> None:
    """1銘柄の詳細分析結果をコンソールに表示する。"""
    d = result.to_dict()

    print()
    print(f"{'=' * 56}")
    print(f"  銘柄詳細分析: {result.ticker} ({result.sector})")
    print(f"{'=' * 56}")
    print()

    # シグナル
    signal_icon = _signal_icon(result.signal)
    print(f"  シグナル:     {signal_icon} {result.signal}")
    print(f"  統合スコア:   {result.final_score:+.4f}")
    print()

    # ファクター内訳
    print(f"  {'─' * 40}")
    print(f"  ファクター内訳:")
    print(f"    モメンタム:   {result.momentum_score:+.4f}")
    print(f"    バリュー:     {result.value_score:+.4f}")
    print(f"    クオリティ:   {result.quality_score:+.4f}")
    print(f"    センチメント: {result.sentiment_score:+.4f}")
    if result.claude_score is not None:
        print(f"    Claude API:   {result.claude_score:+.4f}")
    print()

    # リスク管理
    if result.risk:
        r = result.risk
        print(f"  {'─' * 40}")
        print(f"  リスク管理:")
        print(f"    現在値:       ${r.current_price:>10,.2f}")
        print(f"    損切り (SL):  ${r.stop_loss:>10,.2f}  ({r.stop_loss_pct * 100:+.2f}%)")
        print(f"    利確   (TP):  ${r.take_profit:>10,.2f}  ({r.take_profit_pct * 100:+.2f}%)")
        print(f"    ATR(14):      ${r.atr:>10,.4f}  ({r.atr_pct * 100:.2f}%)")
        print(f"    RR比:         {r.risk_reward_ratio:>10.2f}")
    print()

    # 理由
    if result.reason:
        print(f"  {'─' * 40}")
        print(f"  判定理由: {result.reason}")

    print(f"{'=' * 56}")
    print()


# ============================================================
# CSV 保存
# ============================================================

def _save_csv(
    results: list[ScoringResult],
    output_dir: Path | str | None,
) -> Path | None:
    """分析結果を CSV に保存する。"""
    try:
        out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.csv"
        csv_path = out_dir / filename

        rows = [_result_to_csv_row(r) for r in results]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("CSV 保存完了: %s (%d 銘柄)", csv_path, len(rows))
        return csv_path

    except Exception:
        logger.error("CSV 保存失敗", exc_info=True)
        return None


def _result_to_csv_row(result: ScoringResult) -> dict[str, Any]:
    """ScoringResult を CSV 行の辞書に変換する。"""
    d = result.to_dict()
    row: dict[str, Any] = {}
    for col in _CSV_COLUMNS:
        row[col] = d.get(col, "")
    return row


# ============================================================
# コンソール表示
# ============================================================

def _print_console(results: list[ScoringResult]) -> None:
    """分析結果をコンソールに見やすく表示する。"""
    filtered = filter_actionable(results)
    buy_list = filtered["buy"]
    sell_list = filtered["sell"]
    hold_list = filtered["hold"]

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    print()
    print(f"{'═' * 76}")
    print(f"  Swing Trade Signal Report — {now}")
    print(f"{'═' * 76}")
    print(f"  全{len(results)}銘柄分析完了: "
          f"BUY系 {len(buy_list)} / HOLD {len(hold_list)} / SELL系 {len(sell_list)}")
    print()

    # BUY 候補
    if buy_list:
        _print_signal_table("🟢 BUY 候補（スコア降順）", buy_list)
    else:
        print("  🟢 BUY 候補: なし")
        print()

    # SELL 候補
    if sell_list:
        _print_signal_table("🔴 SELL 候補（スコア昇順）", sell_list)
    else:
        print("  🔴 SELL 候補: なし")
        print()

    # HOLD 上位5
    if hold_list:
        top_hold = hold_list[:5]
        _print_signal_table("⚪ HOLD 上位5（参考）", top_hold)

    print(f"{'═' * 76}")
    print()


def _print_signal_table(title: str, results: list[ScoringResult]) -> None:
    """シグナルテーブルを表示する。"""
    print(f"  {title}")
    print(f"  {'─' * 72}")
    print(f"  {'Ticker':<7} {'Signal':<12} {'Score':>7} "
          f"{'Price':>9} {'SL':>9} {'TP':>9} {'SL%':>7} {'TP%':>7} {'RR':>5}")
    print(f"  {'─' * 72}")

    for r in results:
        icon = _signal_icon(r.signal)
        price = f"${r.risk.current_price:,.2f}" if r.risk else "N/A"
        sl = f"${r.risk.stop_loss:,.2f}" if r.risk else "N/A"
        tp = f"${r.risk.take_profit:,.2f}" if r.risk else "N/A"
        sl_pct = f"{r.risk.stop_loss_pct * 100:+.1f}%" if r.risk else "N/A"
        tp_pct = f"{r.risk.take_profit_pct * 100:+.1f}%" if r.risk else "N/A"
        rr = f"{r.risk.risk_reward_ratio:.1f}" if r.risk else "N/A"

        print(f"  {r.ticker:<7} {icon}{r.signal:<11} {r.final_score:+.3f} "
              f"{price:>9} {sl:>9} {tp:>9} {sl_pct:>7} {tp_pct:>7} {rr:>5}")

    print()


# ============================================================
# ユーティリティ
# ============================================================

def _signal_icon(signal: str) -> str:
    """シグナルに対応するアイコンを返す。"""
    icons = {
        STRONG_BUY: "🟢",
        BUY: "🟢",
        HOLD: "⚪",
        SELL: "🔴",
        STRONG_SELL: "🔴",
    }
    return icons.get(signal, "⚪")
