"""
銘柄別トレード履歴レポート生成スクリプト

signal_backtest CSV（全シグナルトレード）と
portfolio_simulation CSV（実際の資金配分付き）を読み込み、
銘柄別・年別にまとめた総合レポートを出力する。
"""
import csv
import os
from collections import defaultdict

# ---- 入力ファイル ----
SIGNAL_CSV   = r"C:\Users\mh121\OneDrive\Desktop\swing_signal\output\backtest\signal_backtest_20260222_092959.csv"
PORTFOLIO_CSV = r"C:\Users\mh121\OneDrive\Desktop\swing_signal\output\backtest\portfolio_simulation_4000.csv"
OUTPUT_DIR   = r"C:\Users\mh121\OneDrive\Desktop\swing_signal\output\backtest"

RESULT_LABELS = {
    "sl_hit":        "SL Hit",
    "trailing_stop": "Trail Stop",
    "timeout_30d":   "TO-30d",
    "timeout_60d":   "TO-60d",
    "timeout_90d":   "TO-90d",
    "timeout":       "Timeout",
    "pre_earnings":  "Pre-Earn",
}


# ============================================================
# データ読み込み
# ============================================================

def load_signal_trades(path):
    trades = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            trades.append({
                "ticker":      row["ticker"],
                "signal":      row["signal"],
                "signal_date": row["signal_date"],
                "entry_date":  row["entry_date"],
                "exit_date":   row["exit_date"],
                "entry_price": float(row["entry_price"]),
                "exit_price":  float(row["exit_price"]),
                "result":      row["result"],
                "pl_pct":      float(row["pl_pct"]),
                "sl_price":    float(row["sl_price"]),
                "year":        row["exit_date"][:4],
            })
    return trades


def load_portfolio_trades(path):
    trades = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            alloc_str = row["Allocated $"].replace("$", "").replace(",", "").strip()
            pnl_str   = row["P/L $"].replace("$", "").replace(",", "").strip()
            bal_str   = row["Balance $"].replace("$", "").replace(",", "").strip()
            trades.append({
                "no":          int(row["No"]),
                "ticker":      row["Ticker"],
                "signal":      row["Signal"],
                "entry_date":  row["Entry Date"],
                "exit_date":   row["Exit Date"],
                "entry_price": float(row["Entry Price"]),
                "exit_price":  float(row["Exit Price"]),
                "result":      row["Exit Reason"],
                "pl_pct":      float(row["P/L %"].replace("%", "").strip()),
                "alloc":       float(alloc_str),
                "pnl_dollar":  float(pnl_str),
                "balance":     float(bal_str),
                "year":        row["Exit Date"][:4],
            })
    return trades


# ============================================================
# 集計ヘルパー
# ============================================================

def stats(pl_list):
    if not pl_list:
        return {"trades": 0, "wins": 0, "losses": 0,
                "win_rate": 0, "avg_pl": 0, "total_pl": 0,
                "max_win": 0, "max_loss": 0}
    wins   = [p for p in pl_list if p > 0]
    losses = [p for p in pl_list if p < 0]
    return {
        "trades":   len(pl_list),
        "wins":     len(wins),
        "losses":   len(losses),
        "win_rate": len(wins) / len(pl_list) * 100,
        "avg_pl":   sum(pl_list) / len(pl_list),
        "total_pl": sum(pl_list),
        "max_win":  max(pl_list),
        "max_loss": min(pl_list),
    }


# ============================================================
# レポート生成
# ============================================================

def generate_report():
    sig_trades   = load_signal_trades(SIGNAL_CSV)
    port_trades  = load_portfolio_trades(PORTFOLIO_CSV)

    # ---- Section 1: 概要 ----
    lines = []
    lines.append("=" * 110)
    lines.append(f"{'SWING TRADE BACKTEST - 総合トレードレポート':^110}")
    lines.append("=" * 110)
    lines.append(f"  シグナルバックテスト CSV : {os.path.basename(SIGNAL_CSV)}")
    lines.append(f"  ポートフォリオ CSV       : {os.path.basename(PORTFOLIO_CSV)}")
    lines.append("")

    # シグナルバックテスト全体
    all_pl = [t["pl_pct"] for t in sig_trades]
    s = stats(all_pl)
    lines.append("━" * 110)
    lines.append("  【1】シグナルバックテスト概要（全384トレード）")
    lines.append("━" * 110)
    lines.append(f"  総トレード数  : {s['trades']:>6}")
    lines.append(f"  勝率          : {s['win_rate']:>6.1f}%")
    lines.append(f"  平均損益率    : {s['avg_pl']:>+7.2f}%")
    lines.append(f"  累計損益率    : {s['total_pl']:>+8.2f}%")
    lines.append(f"  最大利益      : {s['max_win']:>+7.2f}%")
    lines.append(f"  最大損失      : {s['max_loss']:>+7.2f}%")
    lines.append("")

    # 決済内訳
    result_count = defaultdict(int)
    for t in sig_trades:
        result_count[t["result"]] += 1
    lines.append("  決済内訳:")
    for r, cnt in sorted(result_count.items(), key=lambda x: -x[1]):
        label = RESULT_LABELS.get(r, r)
        lines.append(f"    {label:<18}: {cnt:>4} 件 ({cnt/len(sig_trades)*100:.1f}%)")
    lines.append("")

    # シグナル別
    for sig in ["STRONG_BUY", "BUY"]:
        pl = [t["pl_pct"] for t in sig_trades if t["signal"] == sig]
        s2 = stats(pl)
        lines.append(f"  {sig:<12}: {s2['trades']:>4}件  勝率{s2['win_rate']:>5.1f}%  "
                     f"平均{s2['avg_pl']:>+6.2f}%  最大利益{s2['max_win']:>+7.2f}%  最大損失{s2['max_loss']:>+7.2f}%")
    lines.append("")

    # 年別
    lines.append("  年別:")
    by_year = defaultdict(list)
    for t in sig_trades:
        by_year[t["year"]].append(t["pl_pct"])
    for yr in sorted(by_year):
        s3 = stats(by_year[yr])
        lines.append(f"    {yr}: {s3['trades']:>4}件  勝率{s3['win_rate']:>5.1f}%  "
                     f"平均{s3['avg_pl']:>+6.2f}%  累計{s3['total_pl']:>+8.2f}%")
    lines.append("")

    # ポートフォリオ概要
    if port_trades:
        p_pl = [t["pnl_dollar"] for t in port_trades]
        wins_p = [p for p in p_pl if p > 0]
        losses_p = [p for p in p_pl if p < 0]
        final_bal = port_trades[-1]["balance"]
        initial   = 4000.0
        lines.append("━" * 110)
        lines.append("  【2】ポートフォリオシミュレーション概要（初期資金 $4,000 / リスクベースサイジング）")
        lines.append("━" * 110)
        lines.append(f"  実行トレード  : {len(port_trades):>6}")
        lines.append(f"  勝率          : {len(wins_p)/len(port_trades)*100:>6.1f}%")
        lines.append(f"  最終残高      : ${final_bal:>10,.2f}")
        lines.append(f"  総リターン    : {(final_bal-initial)/initial*100:>+7.2f}%")
        pf = sum(wins_p) / abs(sum(losses_p)) if losses_p else float("inf")
        plr = (sum(wins_p)/len(wins_p)) / abs(sum(losses_p)/len(losses_p)) if wins_p and losses_p else 0
        lines.append(f"  Profit Factor : {pf:>6.3f}")
        lines.append(f"  P/L Ratio     : {plr:>6.3f}")
        lines.append("")

        # ポートフォリオ年別
        lines.append("  年別 (実資金):")
        py_year = defaultdict(list)
        for t in port_trades:
            py_year[t["year"]].append(t)
        for yr in sorted(py_year):
            yr_t = py_year[yr]
            yr_wins = sum(1 for t in yr_t if t["pnl_dollar"] > 0)
            yr_pnl  = sum(t["pnl_dollar"] for t in yr_t)
            yr_end  = yr_t[-1]["balance"]
            lines.append(f"    {yr}: {len(yr_t):>4}件  勝率{yr_wins/len(yr_t)*100:>5.1f}%  "
                         f"P/L ${yr_pnl:>+9,.2f}  年末残高 ${yr_end:>9,.2f}")
        lines.append("")

    # ---- Section 2: 銘柄別サマリー ----
    lines.append("━" * 110)
    lines.append("  【3】銘柄別パフォーマンスサマリー（シグナルバックテスト全件 / 累計損益降順）")
    lines.append("━" * 110)
    lines.append(f"  {'Ticker':<7} {'取引':>5} {'勝率':>7} {'平均PL':>8} {'累計PL':>9} "
                 f"{'最大利益':>9} {'最大損失':>9} {'SL':>4} {'Trail':>6} {'TO':>4} {'シグナル(SB/B)':>14}")
    lines.append("  " + "-" * 105)

    by_ticker = defaultdict(list)
    for t in sig_trades:
        by_ticker[t["ticker"]].append(t)

    ticker_stats = []
    for ticker, trades in by_ticker.items():
        pl = [t["pl_pct"] for t in trades]
        s4 = stats(pl)
        sl_cnt    = sum(1 for t in trades if t["result"] == "sl_hit")
        trail_cnt = sum(1 for t in trades if t["result"] == "trailing_stop")
        to_cnt    = sum(1 for t in trades if "timeout" in t["result"])
        sb_cnt    = sum(1 for t in trades if t["signal"] == "STRONG_BUY")
        b_cnt     = sum(1 for t in trades if t["signal"] == "BUY")
        ticker_stats.append({
            "ticker":    ticker,
            "total_pl":  s4["total_pl"],
            "trades":    s4["trades"],
            "win_rate":  s4["win_rate"],
            "avg_pl":    s4["avg_pl"],
            "max_win":   s4["max_win"],
            "max_loss":  s4["max_loss"],
            "sl_cnt":    sl_cnt,
            "trail_cnt": trail_cnt,
            "to_cnt":    to_cnt,
            "sb_cnt":    sb_cnt,
            "b_cnt":     b_cnt,
        })

    ticker_stats.sort(key=lambda x: -x["total_pl"])
    for ts in ticker_stats:
        lines.append(
            f"  {ts['ticker']:<7} {ts['trades']:>5} {ts['win_rate']:>6.1f}% "
            f"{ts['avg_pl']:>+7.2f}% {ts['total_pl']:>+8.2f}% "
            f"{ts['max_win']:>+8.2f}% {ts['max_loss']:>+8.2f}% "
            f"{ts['sl_cnt']:>4} {ts['trail_cnt']:>6} {ts['to_cnt']:>4} "
            f"  {ts['sb_cnt']:>4}SB / {ts['b_cnt']:<4}B"
        )
    lines.append("")

    # ---- Section 3: 全トレード詳細（銘柄別ソート） ----
    lines.append("━" * 110)
    lines.append("  【4】全トレード詳細（銘柄別・日付順）")
    lines.append("━" * 110)
    lines.append(f"  {'Ticker':<7} {'Signal':<11} {'Entry Date':<12} {'Exit Date':<12} "
                 f"{'Entry $':>9} {'Exit $':>9} {'SL $':>9} {'P/L %':>8} {'Result':<12}")
    lines.append("  " + "-" * 105)

    sorted_trades = sorted(sig_trades, key=lambda t: (t["ticker"], t["entry_date"]))
    current_ticker = None
    for t in sorted_trades:
        if t["ticker"] != current_ticker:
            if current_ticker is not None:
                lines.append("")
            current_ticker = t["ticker"]
        result_label = RESULT_LABELS.get(t["result"], t["result"])
        win_mark = "+" if t["pl_pct"] > 0 else ("-" if t["pl_pct"] < 0 else " ")
        lines.append(
            f"  {t['ticker']:<7} {t['signal']:<11} {t['entry_date']:<12} {t['exit_date']:<12} "
            f"{t['entry_price']:>9.4f} {t['exit_price']:>9.4f} {t['sl_price']:>9.4f} "
            f"{t['pl_pct']:>+7.2f}% {win_mark} {result_label}"
        )
    lines.append("")

    # ---- Section 4: ポートフォリオ全取引（実資金付き） ----
    if port_trades:
        lines.append("━" * 110)
        lines.append("  【5】ポートフォリオ全取引（実際の資金配分・時系列順）")
        lines.append("━" * 110)
        lines.append(f"  {'No':>4} {'Ticker':<7} {'Signal':<11} {'Entry Date':<12} {'Exit Date':<12} "
                     f"{'Entry $':>9} {'Exit $':>9} {'P/L %':>8} {'Alloc $':>9} {'P/L $':>9} {'Balance $':>11}")
        lines.append("  " + "-" * 107)
        current_year = None
        for t in port_trades:
            if t["year"] != current_year:
                if current_year is not None:
                    lines.append("")
                current_year = t["year"]
                lines.append(f"  --- {current_year} ---")
            lines.append(
                f"  {t['no']:>4} {t['ticker']:<7} {t['signal']:<11} "
                f"{t['entry_date']:<12} {t['exit_date']:<12} "
                f"{t['entry_price']:>9.4f} {t['exit_price']:>9.4f} "
                f"{t['pl_pct']:>+7.2f}% ${t['alloc']:>8,.2f} "
                f"${t['pnl_dollar']:>+8,.2f} ${t['balance']:>10,.2f}"
            )
        lines.append("")

    # ---- 出力 ----
    output_path = os.path.join(OUTPUT_DIR, "trade_report_full.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"レポート出力完了: {output_path}")
    print(f"総行数: {len(lines):,}")

    # ---- 銘柄別CSVも別途出力 ----
    csv_path = os.path.join(OUTPUT_DIR, "ticker_summary.csv")
    ticker_stats.sort(key=lambda x: -x["total_pl"])
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Ticker", "Trades", "Wins", "Losses", "WinRate%",
            "AvgPL%", "TotalPL%", "MaxWin%", "MaxLoss%",
            "SL_Hit", "Trail_Stop", "Timeout", "STRONG_BUY", "BUY"
        ])
        for ts in ticker_stats:
            writer.writerow([
                ts["ticker"],
                ts["trades"],
                int(ts["win_rate"] / 100 * ts["trades"]),
                ts["trades"] - int(ts["win_rate"] / 100 * ts["trades"]),
                round(ts["win_rate"], 1),
                round(ts["avg_pl"], 2),
                round(ts["total_pl"], 2),
                round(ts["max_win"], 2),
                round(ts["max_loss"], 2),
                ts["sl_cnt"],
                ts["trail_cnt"],
                ts["to_cnt"],
                ts["sb_cnt"],
                ts["b_cnt"],
            ])

    print(f"銘柄別サマリー CSV: {csv_path}")
    print(f"銘柄数: {len(ticker_stats)}")


if __name__ == "__main__":
    generate_report()
