"""
Swing Trade Signal System — バックテストレポート出力

バックテスト結果をコンソールとCSVに出力する。
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from backtest.engine import SignalBacktestResult, ScreeningBacktestResult

_BACKTEST_OUTPUT_DIR = Path(__file__).parent.parent / "output" / "backtest"


# ============================================================
# シグナルバックテスト出力
# ============================================================

def print_signal_backtest(result: SignalBacktestResult) -> None:
    """シグナルバックテスト結果をコンソールに表示する。"""
    print()
    print("━" * 58)
    print("  📈 シグナルバックテスト結果")
    print("━" * 58)

    if result.total_trades == 0:
        print("  トレードが発生しませんでした")
        return

    print(f"  総トレード数  : {result.total_trades}")
    print(f"  勝率          : {result.win_rate * 100:.1f}%")
    print(f"  平均損益率    : {result.avg_pl_pct * 100:+.2f}%")
    print(f"  累計損益率    : {result.total_pl_pct * 100:+.2f}%")
    print(f"  最大ドローダウン: {result.max_drawdown * 100:.2f}%")
    print(f"  Sharpe比      : {result.sharpe_ratio:.2f}")
    print()

    # 結果内訳
    print("  ── 決済内訳 ──")
    total = result.total_trades
    sl = result.by_result.get("sl_hit", 0)
    ts = result.by_result.get("trailing_stop", 0)
    t30 = result.by_result.get("timeout_30d", 0)
    t60 = result.by_result.get("timeout_60d", 0)
    t90 = result.by_result.get("timeout_90d", 0)
    print(f"  損切り (SL Hit)    : {sl} 件 ({sl / total * 100:.1f}%)")
    print(f"  トレーリング利確   : {ts} 件 ({ts / total * 100:.1f}%)")
    print(f"  30日タイムアウト   : {t30} 件 ({t30 / total * 100:.1f}%)")
    print(f"  60日タイムアウト   : {t60} 件 ({t60 / total * 100:.1f}%)")
    print(f"  90日タイムアウト   : {t90} 件 ({t90 / total * 100:.1f}%)")
    print()

    # シグナル別
    if result.by_signal:
        print("  ── シグナル別 ──")
        for sig, data in sorted(result.by_signal.items()):
            print(
                f"  {sig:<12}: {data['count']} 件  "
                f"勝率 {data['win_rate'] * 100:.1f}%  "
                f"平均 {data['avg_pl_pct'] * 100:+.2f}%"
            )
        print()

    # 年別
    if result.by_year:
        print("  ── 年別 ──")
        for year, data in sorted(result.by_year.items()):
            print(
                f"  {year}: {data['count']} 件  "
                f"勝率 {data['win_rate'] * 100:.1f}%  "
                f"平均 {data['avg_pl_pct'] * 100:+.2f}%"
            )
    print("━" * 58)


def save_signal_backtest_csv(
    result: SignalBacktestResult,
    output_dir: Path | str | None = None,
    label: str | None = None,
) -> Path | None:
    """シグナルバックテスト結果をCSVに保存する。

    Args:
        label: ファイル名に付加するラベル（例: "trail4.0"）。
               複数パラメータを比較する際にファイルを区別するために使用。

    Returns:
        保存したCSVファイルのパス。トレードがない場合は None。
    """
    if not result.trades:
        return None

    out_dir = Path(output_dir) if output_dir else _BACKTEST_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    csv_path = out_dir / f"signal_backtest_{timestamp}{suffix}.csv"

    fieldnames = [
        "ticker", "signal", "score", "signal_date", "entry_date", "entry_price",
        "exit_date", "exit_price", "result", "pl_pct", "sl_price", "tp_price",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trade in result.trades:
            writer.writerow(trade.to_dict())

    return csv_path


# ============================================================
# スクリーニングバックテスト出力
# ============================================================

def print_screening_backtest(result: ScreeningBacktestResult) -> None:
    """スクリーニングバックテスト結果をコンソールに表示する。"""
    print()
    print("━" * 58)
    print("  🔍 スクリーニングバックテスト結果")
    print("━" * 58)

    if result.total_months == 0:
        print("  データが不足しているため集計できませんでした")
        return

    print(f"  集計期間      : {result.total_months} ヶ月")
    print(
        f"  選択銘柄平均リターン: {result.avg_selected_return * 100:+.2f}% / 月"
    )
    print(
        f"  全銘柄平均リターン  : {result.avg_all_return * 100:+.2f}% / 月"
    )
    print(
        f"  アウトパフォーマンス: {result.avg_outperformance * 100:+.2f}% / 月"
    )
    win_rate = result.win_months / result.total_months if result.total_months > 0 else 0.0
    print(
        f"  上回った月数  : {result.win_months} / {result.total_months} ({win_rate * 100:.1f}%)"
    )

    # 月次詳細（最新12ヶ月）
    if result.monthly_records:
        print()
        print("  ── 月次詳細（最新12ヶ月）──")
        print(f"  {'月':<8} {'選択':>8} {'全体':>8} {'差':>8}")
        print("  " + "-" * 36)
        for rec in result.monthly_records[-12:]:
            outperf_str = f"{rec.outperformance * 100:+.2f}%"
            print(
                f"  {rec.month:<8} "
                f"{rec.selected_return * 100:>+7.2f}% "
                f"{rec.all_return * 100:>+7.2f}% "
                f"{outperf_str:>8}"
            )
    print("━" * 58)


def save_screening_backtest_csv(
    result: ScreeningBacktestResult,
    output_dir: Path | str | None = None,
) -> Path | None:
    """スクリーニングバックテスト結果をCSVに保存する。"""
    if not result.monthly_records:
        return None

    out_dir = Path(output_dir) if output_dir else _BACKTEST_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"screening_backtest_{timestamp}.csv"

    fieldnames = ["month", "selected_return_pct", "all_return_pct", "outperformance_pct", "selected_tickers"]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in result.monthly_records:
            writer.writerow({
                "month": rec.month,
                "selected_return_pct": round(rec.selected_return * 100, 4),
                "all_return_pct": round(rec.all_return * 100, 4),
                "outperformance_pct": round(rec.outperformance * 100, 4),
                "selected_tickers": "|".join(rec.selected_tickers),
            })

    return csv_path
