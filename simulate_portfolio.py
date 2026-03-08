"""
ポートフォリオシミュレーション（リスクベース・サイジング）

バックテストCSVを読み込み、同時ポジション上限10の制約のもと
リスクベースサイジングで資金配分・損益計算を行い、全取引履歴を出力する。

サイジングロジック:
  1. 1トレードの許容リスク = Total Equity × 1.0%
  2. risk_ratio = (entry_price - sl_price) / entry_price（最低0.5%保証）
  3. investment_amount = risk_amount / risk_ratio
  4. final_investment = min(investment_amount, Total Equity × 15%, 残現金)
  5. 株数 = floor(final_investment / entry_price)
"""
import sys
import os
import csv
import math
from dataclasses import dataclass
from datetime import datetime

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

INITIAL_CAPITAL = 4000.0
MAX_POSITIONS = 11
RISK_PER_TRADE = 0.0050    # 1トレードあたり許容リスク = 総資産の0.50%（WFT最適化）
MAX_ALLOC_PCT = 0.130       # 1トレード最大配分 = 総資産の13.0% (BUY)
SB_MAX_ALLOC_PCT = 0.140    # STRONG_BUY 最大配分 = 総資産の14.0%
MIN_RISK_RATIO = 0.005      # 最低リスク距離 = 0.5%（極端低ボラ対策）
SIGNAL_FILTER = None            # None or list: e.g. ["STRONG_BUY"] or ["BUY"] or None(全件)

CSV_PATH = r"C:\Users\mh121\OneDrive\Desktop\swing_signal\output\backtest\signal_backtest_20260303_022802_sb_trail4.0.csv"

RESULT_LABELS = {
    "sl_hit":        "SL Hit",
    "trailing_stop": "Trail Stop",
    "timeout_30d":   "TO-30d",
    "timeout_60d":   "TO-60d",
    "timeout_90d":   "TO-90d",
    "timeout":       "Timeout",
    "pre_earnings":  "Pre-Earn",
    "riopon_20d":    "Riopon-1M",
    "riopon_40d":    "Riopon-2M",
    "riopon_60d":    "Riopon-3M",
    # 部分利確ラベル
    "partial_sl":    "Part+SL",
    "partial_trail": "Part+Trail",
    "partial_to30d": "Part+TO30",
    "partial_to60d": "Part+TO60",
    "partial_to90d": "Part+TO90",
}


@dataclass
class Trade:
    ticker: str
    signal: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    result: str
    pl_pct: float
    sl_price: float
    tp_price: float


def load_trades(path: str) -> list[Trade]:
    trades = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(Trade(
                ticker=row["ticker"],
                signal=row["signal"],
                entry_date=row["entry_date"],
                exit_date=row["exit_date"],
                entry_price=float(row["entry_price"]),
                exit_price=float(row["exit_price"]),
                result=row["result"],
                pl_pct=float(row["pl_pct"]),
                sl_price=float(row["sl_price"]),
                tp_price=float(row["tp_price"]),
            ))
    return trades


def simulate(trades: list[Trade]) -> None:
    # ---- 日付イベント管理 ----
    # エントリー日でソート（同日は元CSV順）
    trades.sort(key=lambda t: t.entry_date)

    # 各取引をイベントに分解: (date, type, trade_index)
    # ソートキー: (日付, 優先度)
    #   同日の場合: EXIT=0(先) / ENTRY=1(後)
    #   ただし entry_date == exit_date の同一トレードは ENTRY→EXIT の順で処理したいため
    #   entry_date == exit_date の場合はEXITをentryイベント扱い(優先度=2)に下げる
    events: list[tuple[str, str, int]] = []
    same_day_set: set[int] = {
        i for i, t in enumerate(trades) if t.entry_date == t.exit_date
    }
    for i, t in enumerate(trades):
        events.append((t.entry_date, "entry", i))
        # 同日トレードのexitイベントは entry より後に処理する（優先度=2）
        events.append((t.exit_date, "exit", i))
    def _event_key(e: tuple[str, str, int]) -> tuple[str, int]:
        date, etype, idx = e
        if etype == "exit":
            # 同日トレードのexitは後回し（priority=2）、それ以外は先（priority=0）
            priority = 2 if idx in same_day_set else 0
        else:
            priority = 1
        return (date, priority)
    events.sort(key=_event_key)

    cash = INITIAL_CAPITAL
    active: dict[int, float] = {}  # trade_index -> allocated_amount
    peak_balance = INITIAL_CAPITAL
    max_dd = 0.0
    max_dd_pct = 0.0

    # 取引履歴
    history: list[dict] = []
    trade_num = 0
    skipped = 0

    # イベント駆動シミュレーション
    processed_entries: set[int] = set()
    processed_exits: set[int] = set()

    for date, etype, idx in events:
        t = trades[idx]

        if etype == "exit" and idx in processed_entries and idx not in processed_exits:
            processed_exits.add(idx)
            alloc = active.pop(idx)
            pnl_dollar = alloc * (t.pl_pct / 100.0)
            cash += alloc + pnl_dollar
            trade_num += 1

            total_balance = cash + sum(active.values())

            if total_balance > peak_balance:
                peak_balance = total_balance
            dd = peak_balance - total_balance
            dd_pct = (dd / peak_balance) * 100 if peak_balance > 0 else 0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
                max_dd = dd

            history.append({
                "no": trade_num,
                "ticker": t.ticker,
                "signal": t.signal,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "result": RESULT_LABELS.get(t.result, t.result),
                "pl_pct": t.pl_pct,
                "alloc": alloc,
                "pnl_dollar": pnl_dollar,
                "balance": total_balance,
                "cash": cash,
                "active_count": len(active),
            })

        elif etype == "entry" and idx not in processed_entries:
            if len(active) >= MAX_POSITIONS:
                skipped += 1
                continue

            # リスクベース・サイジング
            total_equity = cash + sum(active.values())
            risk_amount = total_equity * RISK_PER_TRADE  # 総資産の1%

            # SL距離（リスク比率）を計算
            risk_ratio = (t.entry_price - t.sl_price) / t.entry_price if t.entry_price > 0 else 0.08
            risk_ratio = max(risk_ratio, MIN_RISK_RATIO)  # 最低0.5%保証

            # 投資金額 = 許容リスク額 / リスク比率
            investment = risk_amount / risk_ratio

            # キャップ: min(投資額, 総資産×配分上限, 残現金)
            # STRONG_BUY は 10%、BUY は 9.5%
            _alloc_pct = SB_MAX_ALLOC_PCT if t.signal == "STRONG_BUY" else MAX_ALLOC_PCT
            max_alloc = total_equity * _alloc_pct
            alloc = min(investment, max_alloc, cash)

            # 株数 = floor(alloc / entry_price) → 端数切り捨てで再計算
            if t.entry_price > 0:
                shares = math.floor(alloc / t.entry_price)
                alloc = shares * t.entry_price
            else:
                shares = 0

            if alloc <= 0 or cash <= 0 or shares == 0:
                skipped += 1
                continue
            cash -= alloc
            active[idx] = alloc
            processed_entries.add(idx)

    # ---- 結果出力 ----
    output_path = r"C:\Users\mh121\OneDrive\Desktop\swing_signal\output\backtest\portfolio_simulation_4000.csv"

    # CSV出力
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "No", "Ticker", "Signal", "Entry Date", "Exit Date",
            "Entry Price", "Exit Price", "Exit Reason",
            "P/L %", "Allocated $", "P/L $", "Balance $",
        ])
        for h in history:
            writer.writerow([
                h["no"],
                h["ticker"],
                h["signal"],
                h["entry_date"],
                h["exit_date"],
                f"{h['entry_price']:.4f}",
                f"{h['exit_price']:.4f}",
                h["result"],
                f"{h['pl_pct']:+.2f}%",
                f"${h['alloc']:.2f}",
                f"${h['pnl_dollar']:+.2f}",
                f"${h['balance']:.2f}",
            ])

    print(f"CSV saved: {output_path}")
    print()

    # コンソール出力 (全取引)
    print("=" * 130)
    print(f"{'Portfolio Simulation - Initial Capital: $4,000.00':^130}")
    print("=" * 130)
    print(f"{'No':>4} | {'Ticker':<6} | {'Signal':<11} | {'Entry Date':<11} | {'Exit Date':<11} | "
          f"{'Entry $':>10} | {'Exit $':>10} | {'Reason':<8} | {'P/L %':>8} | {'Alloc $':>10} | "
          f"{'P/L $':>10} | {'Balance $':>11}")
    print("-" * 130)

    for h in history:
        print(f"{h['no']:>4} | {h['ticker']:<6} | {h['signal']:<11} | {h['entry_date']:<11} | "
              f"{h['exit_date']:<11} | {h['entry_price']:>10.4f} | {h['exit_price']:>10.4f} | "
              f"{h['result']:<8} | {h['pl_pct']:>+7.2f}% | ${h['alloc']:>9.2f} | "
              f"${h['pnl_dollar']:>+9.2f} | ${h['balance']:>10.2f}")

    print("=" * 130)

    # サマリー
    final_balance = cash + sum(active.values())
    total_return = ((final_balance - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    total_trades = len(history)
    wins = sum(1 for h in history if h["pnl_dollar"] > 0)
    losses = sum(1 for h in history if h["pnl_dollar"] < 0)
    breakeven = total_trades - wins - losses

    total_profit = sum(h["pnl_dollar"] for h in history if h["pnl_dollar"] > 0)
    total_loss = sum(h["pnl_dollar"] for h in history if h["pnl_dollar"] < 0)
    avg_win = total_profit / wins if wins > 0 else 0
    avg_loss = total_loss / losses if losses > 0 else 0
    pf = total_profit / abs(total_loss) if total_loss != 0 else float("inf")
    plr = avg_win / abs(avg_loss) if avg_loss != 0 else float("inf")

    print()
    print("PORTFOLIO SUMMARY")
    print("-" * 50)
    print(f"  Initial Capital:    ${INITIAL_CAPITAL:>12,.2f}")
    print(f"  Final Balance:      ${final_balance:>12,.2f}")
    print(f"  Total Return:       {total_return:>+12.2f}%")
    print(f"  Total P/L:          ${final_balance - INITIAL_CAPITAL:>+12,.2f}")
    print()
    print(f"  Total Trades:       {total_trades:>12}")
    print(f"  Wins:               {wins:>12} ({wins/total_trades*100:.1f}%)")
    print(f"  Losses:             {losses:>12} ({losses/total_trades*100:.1f}%)")
    print(f"  Breakeven:          {breakeven:>12}")
    print()
    print(f"  Total Profit:       ${total_profit:>+12,.2f}")
    print(f"  Total Loss:         ${total_loss:>+12,.2f}")
    print(f"  Avg Win:            ${avg_win:>+12,.2f}")
    print(f"  Avg Loss:           ${avg_loss:>+12,.2f}")
    print()
    print(f"  Profit Factor:      {pf:>12.3f}")
    print(f"  P/L Ratio:          {plr:>12.3f}")
    print(f"  Max Drawdown:       ${max_dd:>12,.2f} ({max_dd_pct:.2f}%)")
    print(f"  Peak Balance:       ${peak_balance:>12,.2f}")
    print()
    print(f"  Skipped (full):     {skipped:>12}")

    # 年別サマリー
    print()
    print("YEARLY BREAKDOWN")
    print("-" * 70)
    print(f"  {'Year':<6} | {'Trades':>7} | {'Wins':>5} | {'Win%':>6} | {'P/L $':>12} | {'Year-End $':>12}")
    print("  " + "-" * 65)

    years = sorted(set(h["exit_date"][:4] for h in history))
    for yr in years:
        yr_trades = [h for h in history if h["exit_date"].startswith(yr)]
        yr_wins = sum(1 for h in yr_trades if h["pnl_dollar"] > 0)
        yr_pnl = sum(h["pnl_dollar"] for h in yr_trades)
        yr_end_bal = yr_trades[-1]["balance"] if yr_trades else 0
        yr_winrate = yr_wins / len(yr_trades) * 100 if yr_trades else 0
        print(f"  {yr:<6} | {len(yr_trades):>7} | {yr_wins:>5} | {yr_winrate:>5.1f}% | ${yr_pnl:>+11,.2f} | ${yr_end_bal:>11,.2f}")


if __name__ == "__main__":
    trades = load_trades(CSV_PATH)
    if SIGNAL_FILTER:
        trades = [t for t in trades if t.signal in SIGNAL_FILTER]
        print(f"[SIGNAL_FILTER={SIGNAL_FILTER}] {len(trades)}件に絞り込み")
    simulate(trades)
