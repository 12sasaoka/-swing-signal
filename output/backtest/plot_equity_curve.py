"""
資産曲線（Equity Curve）チャート生成スクリプト
- 上段: Portfolio vs SPY Buy&Hold 比較ライン
- 中段: ドローダウン（Portfolio vs SPY）
- 下段: 年別P/Lバーチャート
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import yfinance as yf
from datetime import datetime

PORTFOLIO_CSV   = r"C:\Users\mh121\OneDrive\Desktop\swing_signal\output\backtest\portfolio_simulation_4000.csv"
OUTPUT_DIR      = r"C:\Users\mh121\OneDrive\Desktop\swing_signal\output\backtest"
INITIAL_CAPITAL = 4000.0

# ============================================================
# 1. ポートフォリオデータ読み込み
# ============================================================
trades = []
with open(PORTFOLIO_CSV, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        bal_str = row["Balance $"].replace("$", "").replace(",", "").strip()
        pnl_str = row["P/L $"].replace("$", "").replace(",", "").strip()
        trades.append({
            "exit_date": datetime.strptime(row["Exit Date"], "%Y-%m-%d"),
            "balance":   float(bal_str),
            "pnl":       float(pnl_str),
            "year":      int(row["Exit Date"][:4]),
            "win":       float(pnl_str) > 0,
        })

port_dates    = [datetime(2019, 1, 2)] + [t["exit_date"] for t in trades]
port_balances = [INITIAL_CAPITAL]      + [t["balance"]   for t in trades]

start_date = port_dates[0]
end_date   = port_dates[-1]

# ============================================================
# 2. SPY Buy&Hold データ取得
# ============================================================
print("SPYデータ取得中...")
spy_raw = yf.download("SPY", start=start_date.strftime("%Y-%m-%d"),
                      end=end_date.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
spy_raw.index = spy_raw.index.tz_localize(None)
spy_close = spy_raw["Close"].dropna()

# 同期: port_dates に合わせてSPY残高を補間
spy_price_start = float(spy_close.iloc[0])
spy_dates    = [start_date]
spy_balances = [INITIAL_CAPITAL]
for d in port_dates[1:]:
    ts = spy_close.asof(d)
    if ts is None or (hasattr(ts, '__len__') and len(ts) == 0):
        spy_balances.append(spy_balances[-1])
    else:
        spy_balances.append(INITIAL_CAPITAL * float(ts) / spy_price_start)
    spy_dates.append(d)

# ============================================================
# 3. ドローダウン計算
# ============================================================
def calc_drawdown(balances):
    peak = balances[0]
    dd = [0.0]
    for b in balances[1:]:
        peak = max(peak, b)
        dd.append((b - peak) / peak * 100)
    return dd

port_dd = calc_drawdown(port_balances)
spy_dd  = calc_drawdown(spy_balances)

max_dd_val = min(port_dd)
max_dd_idx = port_dd.index(max_dd_val)

# ============================================================
# 4. 年別集計
# ============================================================
from collections import defaultdict
year_pnl  = defaultdict(float)
year_wins = defaultdict(int)
year_cnt  = defaultdict(int)
for t in trades:
    year_pnl[t["year"]]  += t["pnl"]
    year_cnt[t["year"]]  += 1
    if t["win"]:
        year_wins[t["year"]] += 1

years     = sorted(year_pnl.keys())
bar_vals  = [year_pnl[y] for y in years]
bar_wins  = [year_wins[y] / year_cnt[y] * 100 if year_cnt[y] else 0 for y in years]
bar_colors = ["#00d46a" if v >= 0 else "#ff4444" for v in bar_vals]

# ============================================================
# 5. フィギュア構築 (3段)
# ============================================================
fig = plt.figure(figsize=(18, 13), facecolor="#1a1a2e")
gs  = fig.add_gridspec(3, 2,
    height_ratios=[3, 1.2, 1.8],
    width_ratios=[3, 1],
    hspace=0.08, wspace=0.25,
    left=0.06, right=0.97, top=0.93, bottom=0.06
)

ax_eq   = fig.add_subplot(gs[0, 0])   # 上段左: equity curve
ax_stat = fig.add_subplot(gs[0, 1])   # 上段右: サマリー統計
ax_dd   = fig.add_subplot(gs[1, 0], sharex=ax_eq)  # 中段左: DD
ax_dd2  = fig.add_subplot(gs[1, 1])   # 中段右: 空白
ax_bar  = fig.add_subplot(gs[2, 0])   # 下段左: 年別バー
ax_wr   = fig.add_subplot(gs[2, 1])   # 下段右: 年別勝率

DARK_BG  = "#16213e"
GRID_COL = "#2a2a4a"

for ax in [ax_eq, ax_stat, ax_dd, ax_dd2, ax_bar, ax_wr]:
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors="#cccccc", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444")

ax_dd2.set_visible(False)

# ---- 年帯 ----
for yr in years:
    yr_pts = [d for d in port_dates if d.year == yr]
    if not yr_pts:
        continue
    x0, x1 = min(yr_pts), max(yr_pts)
    col = "#2a2a3e" if yr % 2 == 0 else "#1e1e32"
    for ax in [ax_eq, ax_dd]:
        ax.axvspan(x0, x1, color=col, alpha=0.6, zorder=0)

# ============================================================
# 6. 上段: Equity Curve + SPY
# ============================================================
final     = port_balances[-1]
ret_port  = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
spy_final = spy_balances[-1]
ret_spy   = (spy_final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

ax_eq.plot(port_dates, port_balances, color="#00d4ff", linewidth=2.0,
           zorder=4, label=f"Swing Signal  +{ret_port:.1f}%")
ax_eq.plot(spy_dates, spy_balances, color="#ff9900", linewidth=1.4,
           zorder=3, linestyle="--", label=f"SPY Buy&Hold  +{ret_spy:.1f}%")
ax_eq.fill_between(port_dates, spy_balances, port_balances,
                   where=[p >= s for p, s in zip(port_balances, spy_balances)],
                   color="#00d4ff", alpha=0.10, zorder=2, label="Alpha(+)")
ax_eq.fill_between(port_dates, spy_balances, port_balances,
                   where=[p < s for p, s in zip(port_balances, spy_balances)],
                   color="#ff4444", alpha=0.14, zorder=2, label="Alpha(-)")
ax_eq.axhline(INITIAL_CAPITAL, color="#666", linewidth=0.7, linestyle=":", zorder=1)

# Peak マーカー
peak_idx = port_balances.index(max(port_balances))
ax_eq.scatter(port_dates[peak_idx], port_balances[peak_idx],
              color="#ffd700", s=90, zorder=6, marker="^", label="Peak")
ax_eq.annotate(f"${port_balances[peak_idx]:,.0f}",
               xy=(port_dates[peak_idx], port_balances[peak_idx]),
               xytext=(8, 6), textcoords="offset points",
               color="#ffd700", fontsize=7.5)

# Final ラベル
ax_eq.annotate(f"${final:,.0f}\n+{ret_port:.1f}%",
               xy=(port_dates[-1], final),
               xytext=(-80, 12), textcoords="offset points",
               color="#00ff88", fontsize=9, fontweight="bold",
               arrowprops=dict(arrowstyle="->", color="#00ff88", lw=0.9))
ax_eq.annotate(f"SPY ${spy_final:,.0f}\n+{ret_spy:.1f}%",
               xy=(spy_dates[-1], spy_final),
               xytext=(-80, -28), textcoords="offset points",
               color="#ff9900", fontsize=8,
               arrowprops=dict(arrowstyle="->", color="#ff9900", lw=0.8))

ax_eq.set_ylabel("Balance ($)", color="#cccccc", fontsize=10)
ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax_eq.legend(loc="upper left", facecolor=DARK_BG, edgecolor="#555",
             labelcolor="#cccccc", fontsize=8, ncol=2)
ax_eq.grid(axis="y", color=GRID_COL, linewidth=0.5)
ax_eq.set_title("Swing Signal — Portfolio vs SPY Buy & Hold  (2019–2026)",
                color="#ffffff", fontsize=13, fontweight="bold", pad=10)

# ============================================================
# 7. 上段右: サマリー統計テキスト
# ============================================================
ax_stat.axis("off")
wins  = sum(1 for t in trades if t["win"])
total = len(trades)
pf_wins  = [t["pnl"] for t in trades if t["pnl"] > 0]
pf_loss  = [t["pnl"] for t in trades if t["pnl"] < 0]
pf_val = sum(pf_wins) / abs(sum(pf_loss)) if pf_loss else float("inf")
plr    = (sum(pf_wins)/len(pf_wins)) / abs(sum(pf_loss)/len(pf_loss)) if pf_wins and pf_loss else 0

stats_lines = [
    ("Initial",        f"${INITIAL_CAPITAL:,.0f}"),
    ("Final",          f"${final:,.2f}"),
    ("Return",         f"+{ret_port:.2f}%"),
    ("vs SPY",         f"{ret_port - ret_spy:+.2f}% alpha"),
    ("",               ""),
    ("Trades",         f"{total}"),
    ("Win Rate",       f"{wins/total*100:.1f}%"),
    ("Profit Factor",  f"{pf_val:.3f}"),
    ("P/L Ratio",      f"{plr:.3f}"),
    ("",               ""),
    ("Max Drawdown",   f"{max_dd_val:.1f}%"),
    ("Peak",           f"${max(port_balances):,.0f}"),
]

y_pos = 0.97
for label, val in stats_lines:
    if not label:
        y_pos -= 0.04
        continue
    color = "#00ff88" if "Return" in label or "Final" in label else \
            "#ff9900" if "alpha" in val else \
            "#ff6666" if "Drawdown" in label else "#cccccc"
    ax_stat.text(0.05, y_pos, label, transform=ax_stat.transAxes,
                 color="#888888", fontsize=9, va="top")
    ax_stat.text(0.98, y_pos, val, transform=ax_stat.transAxes,
                 color=color, fontsize=9, va="top", ha="right", fontweight="bold")
    y_pos -= 0.072

ax_stat.set_facecolor("#0d1020")
for spine in ax_stat.spines.values():
    spine.set_color("#333")

# ============================================================
# 8. 中段: ドローダウン (Portfolio + SPY)
# ============================================================
ax_dd.fill_between(port_dates, port_dd, 0, color="#ff4444", alpha=0.50, zorder=2)
ax_dd.plot(port_dates, port_dd, color="#ff6666", linewidth=1.0, zorder=3,
           label=f"Portfolio DD (max {max_dd_val:.1f}%)")
ax_dd.plot(spy_dates, spy_dd, color="#ff9900", linewidth=1.0, linestyle="--",
           zorder=3, label=f"SPY DD (max {min(spy_dd):.1f}%)")
ax_dd.axhline(0, color="#666", linewidth=0.5, linestyle=":")
ax_dd.axhline(max_dd_val, color="#ff2222", linewidth=0.7, linestyle=":")
ax_dd.set_ylabel("Drawdown (%)", color="#cccccc", fontsize=9)
ax_dd.set_ylim(min(min(port_dd), min(spy_dd)) * 1.25, 5)
ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax_dd.legend(loc="lower right", facecolor=DARK_BG, edgecolor="#444",
             labelcolor="#cccccc", fontsize=7.5)
ax_dd.grid(axis="y", color=GRID_COL, linewidth=0.4)
ax_dd.xaxis.set_major_locator(mdates.YearLocator())
ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_dd.tick_params(axis="x", colors="#cccccc", labelsize=8)

# ============================================================
# 9. 下段左: 年別P/Lバーチャート
# ============================================================
x = np.arange(len(years))
bars = ax_bar.bar(x, bar_vals, color=bar_colors, width=0.6, zorder=3)
ax_bar.axhline(0, color="#888", linewidth=0.8)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([str(y) for y in years], color="#cccccc", fontsize=8)
ax_bar.set_ylabel("P/L ($)", color="#cccccc", fontsize=9)
ax_bar.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:+,.0f}"))
ax_bar.set_title("年別 P/L ($)", color="#cccccc", fontsize=9, pad=4)
ax_bar.grid(axis="y", color=GRID_COL, linewidth=0.4)
for bar, val in zip(bars, bar_vals):
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (30 if val >= 0 else -80),
                f"${val:+,.0f}", ha="center", va="bottom",
                fontsize=7, color="#cccccc")

# ============================================================
# 10. 下段右: 年別勝率バーチャート
# ============================================================
wr_colors = ["#00d46a" if w >= 40 else "#ffaa00" if w >= 33 else "#ff4444"
             for w in bar_wins]
ax_wr.bar(x, bar_wins, color=wr_colors, width=0.6, zorder=3)
ax_wr.axhline(50, color="#666", linewidth=0.6, linestyle="--")
ax_wr.set_xticks(x)
ax_wr.set_xticklabels([str(y) for y in years], color="#cccccc", fontsize=8)
ax_wr.set_ylabel("Win Rate (%)", color="#cccccc", fontsize=9)
ax_wr.set_ylim(0, 75)
ax_wr.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax_wr.set_title("年別 勝率", color="#cccccc", fontsize=9, pad=4)
ax_wr.grid(axis="y", color=GRID_COL, linewidth=0.4)
for i, (wr, cnt) in enumerate(zip(bar_wins, [year_cnt[y] for y in years])):
    ax_wr.text(i, wr + 1.5, f"{wr:.0f}%\n({cnt}件)",
               ha="center", va="bottom", fontsize=6.5, color="#cccccc")

# ============================================================
# 保存
# ============================================================
out_path = os.path.join(OUTPUT_DIR, "equity_curve.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"チャート保存: {out_path}")
