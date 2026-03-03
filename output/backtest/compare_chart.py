"""
最新バックテスト結果 vs SPY vs QQQ 比較チャート
portfolio_simulation_4000.csv (trail4.5, $4000初期) を使用
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import yfinance as yf
import numpy as np
from datetime import datetime

# ── 1. ポートフォリオ資産曲線を構築 ──────────────────────────
pf = pd.read_csv("output/backtest/portfolio_simulation_4000.csv")
pf["Exit Date"] = pd.to_datetime(pf["Exit Date"])
pf["Balance $"] = pf["Balance $"].str.replace("$", "", regex=False).str.replace(",", "").astype(float)

START_DATE = pd.Timestamp("2021-01-04")
END_DATE   = pd.Timestamp("2026-02-28")
INITIAL    = 4000.0

# 日次インデックスを生成（最初の取引日から最後まで）
date_range = pd.bdate_range(START_DATE, END_DATE)
equity = pd.Series(index=date_range, dtype=float)

# 各取引のexit日に残高をセット
for _, row in pf.iterrows():
    d = row["Exit Date"]
    if d in equity.index:
        equity[d] = row["Balance $"]

# 最初に初期値をセット、前埋め
equity.iloc[0] = INITIAL
equity = equity.fillna(method="ffill")
equity = equity.fillna(INITIAL)  # 前方に初期値なければ初期値で埋め

# ── 2. SPY / QQQ ダウンロード ──────────────────────────────
raw = yf.download(["SPY", "QQQ"], start="2021-01-04", end="2026-03-01", progress=False)
spy_close = raw["Close"]["SPY"].dropna()
qqq_close = raw["Close"]["QQQ"].dropna()

# 初期価格に合わせて $4000 スタートに正規化
spy_norm = spy_close / spy_close.iloc[0] * INITIAL
qqq_norm = qqq_close / qqq_close.iloc[0] * INITIAL

# 共通インデックスに揃える
all_dates = equity.index.union(spy_norm.index).union(qqq_norm.index)
all_dates = all_dates[(all_dates >= START_DATE) & (all_dates <= END_DATE)]

equity_a   = equity.reindex(all_dates).fillna(method="ffill").fillna(INITIAL)
spy_a      = spy_norm.reindex(all_dates).fillna(method="ffill")
qqq_a      = qqq_norm.reindex(all_dates).fillna(method="ffill")

# ── 3. 最終メトリクス計算 ──────────────────────────────────
final_pf  = equity_a.dropna().iloc[-1]
final_spy = spy_a.dropna().iloc[-1]
final_qqq = qqq_a.dropna().iloc[-1]

ret_pf  = (final_pf  - INITIAL) / INITIAL * 100
ret_spy = (final_spy - INITIAL) / INITIAL * 100
ret_qqq = (final_qqq - INITIAL) / INITIAL * 100

# 年別リターン (ポートフォリオ)
def annual_returns(s):
    years = sorted(set(s.index.year))
    res = {}
    for y in years:
        yr = s[s.index.year == y]
        if len(yr) < 2:
            continue
        prev_yr = s[s.index.year == y - 1]
        start_v = prev_yr.iloc[-1] if len(prev_yr) > 0 else yr.iloc[0]
        end_v   = yr.iloc[-1]
        res[y] = (end_v - start_v) / start_v * 100
    return res

pf_annual  = annual_returns(equity_a)
spy_annual = annual_returns(spy_a)
qqq_annual = annual_returns(qqq_a)

# Max Drawdown
def max_drawdown(s):
    peak = s.cummax()
    dd = (s - peak) / peak * 100
    return dd.min()

dd_pf  = max_drawdown(equity_a.dropna())
dd_spy = max_drawdown(spy_a.dropna())
dd_qqq = max_drawdown(qqq_a.dropna())

# Sharpe (daily returns, rf=0)
def sharpe(s):
    r = s.pct_change().dropna()
    return (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0

sh_pf  = sharpe(equity_a.dropna())
sh_spy = sharpe(spy_a.dropna())
sh_qqq = sharpe(qqq_a.dropna())

# ── 4. 描画 ────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 22), facecolor="#0d1117")
fig.suptitle("Swing Signal Backtest  vs  SPY  vs  QQQ\n$4,000 Initial Capital  |  Jan 2021 - Feb 2026  (trail stop 4.5%)",
             color="white", fontsize=16, fontweight="bold", y=0.98)

gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.35,
                      top=0.94, bottom=0.04, left=0.07, right=0.97)

COLORS = {
    "Strategy": "#00d4ff",
    "SPY":      "#f0a500",
    "QQQ":      "#ff6b6b",
}

ax_style = dict(facecolor="#161b22", tick_params=dict(colors="white"),
                spine_color="#30363d")

def style_ax(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#adbac7")
    ax.xaxis.label.set_color("#adbac7")
    ax.yaxis.label.set_color("#adbac7")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", linewidth=0.7, linestyle="--")

# ── Panel 1: 資産推移 (絶対値) ─────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
style_ax(ax1)
ax1.plot(equity_a.index, equity_a,   color=COLORS["Strategy"], lw=2,   label=f"Strategy  ${final_pf:,.0f} (+{ret_pf:.1f}%)", zorder=3)
ax1.plot(spy_a.index,    spy_a,      color=COLORS["SPY"],      lw=1.5, label=f"SPY       ${final_spy:,.0f} (+{ret_spy:.1f}%)")
ax1.plot(qqq_a.index,    qqq_a,      color=COLORS["QQQ"],      lw=1.5, label=f"QQQ       ${final_qqq:,.0f} (+{ret_qqq:.1f}%)")
ax1.axhline(INITIAL, color="#6e7681", lw=0.8, linestyle=":")
ax1.set_title("Portfolio Value  ($4,000 start)", color="white", fontsize=12, fontweight="bold")
ax1.set_ylabel("Portfolio Value ($)", color="#adbac7")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())
legend = ax1.legend(loc="upper left", facecolor="#21262d", edgecolor="#30363d",
                    labelcolor="white", fontsize=10)
ax1.set_xlim(START_DATE, END_DATE)

# ── Panel 2: 正規化リターン (%) ────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
style_ax(ax2)
pf_pct  = (equity_a / INITIAL - 1) * 100
spy_pct = (spy_a    / INITIAL - 1) * 100
qqq_pct = (qqq_a    / INITIAL - 1) * 100

ax2.plot(equity_a.index, pf_pct,  color=COLORS["Strategy"], lw=2,   label="Strategy", zorder=3)
ax2.plot(spy_a.index,    spy_pct, color=COLORS["SPY"],      lw=1.5, label="SPY")
ax2.plot(qqq_a.index,    qqq_pct, color=COLORS["QQQ"],      lw=1.5, label="QQQ")
ax2.axhline(0, color="#6e7681", lw=0.8, linestyle=":")
ax2.fill_between(equity_a.index, pf_pct, 0, where=(pf_pct > 0), alpha=0.1, color=COLORS["Strategy"])
ax2.fill_between(equity_a.index, pf_pct, 0, where=(pf_pct < 0), alpha=0.15, color="red")
ax2.set_title("Cumulative Return (%)", color="white", fontsize=12, fontweight="bold")
ax2.set_ylabel("Return (%)", color="#adbac7")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.legend(loc="upper left", facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=10)
ax2.set_xlim(START_DATE, END_DATE)

# ── Panel 3: ドローダウン ────────────────────────────────
ax3 = fig.add_subplot(gs[2, :])
style_ax(ax3)

def drawdown_series(s):
    peak = s.cummax()
    return (s - peak) / peak * 100

ax3.fill_between(equity_a.index, drawdown_series(equity_a), 0, color=COLORS["Strategy"], alpha=0.3, label="Strategy")
ax3.fill_between(spy_a.index,    drawdown_series(spy_a),    0, color=COLORS["SPY"],      alpha=0.25, label="SPY")
ax3.fill_between(qqq_a.index,    drawdown_series(qqq_a),    0, color=COLORS["QQQ"],      alpha=0.2,  label="QQQ")
ax3.plot(equity_a.index, drawdown_series(equity_a), color=COLORS["Strategy"], lw=1.2, zorder=3)
ax3.set_title("Drawdown (%)", color="white", fontsize=12, fontweight="bold")
ax3.set_ylabel("Drawdown (%)", color="#adbac7")
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax3.xaxis.set_major_locator(mdates.YearLocator())
ax3.legend(loc="lower left", facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=10)
ax3.set_xlim(START_DATE, END_DATE)

# ── Panel 4: 年別リターン (棒グラフ) ────────────────────────
ax4 = fig.add_subplot(gs[3, 0])
style_ax(ax4)
years = sorted(set(list(pf_annual.keys()) + list(spy_annual.keys())))
x = np.arange(len(years))
w = 0.28
b1 = ax4.bar(x - w, [pf_annual.get(y, 0)  for y in years], w, label="Strategy", color=COLORS["Strategy"], alpha=0.85)
b2 = ax4.bar(x,     [spy_annual.get(y, 0) for y in years], w, label="SPY",      color=COLORS["SPY"],      alpha=0.85)
b3 = ax4.bar(x + w, [qqq_annual.get(y, 0) for y in years], w, label="QQQ",      color=COLORS["QQQ"],      alpha=0.85)
ax4.axhline(0, color="#6e7681", lw=0.8)
ax4.set_xticks(x)
ax4.set_xticklabels([str(y) for y in years], rotation=45, ha="right", color="#adbac7", fontsize=8)
ax4.set_title("Annual Return (%)", color="white", fontsize=12, fontweight="bold")
ax4.set_ylabel("Return (%)", color="#adbac7")
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax4.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=9)

# ── Panel 5: 統計サマリーテーブル ────────────────────────────
ax5 = fig.add_subplot(gs[3, 1])
style_ax(ax5)
ax5.axis("off")

def fmt_ret(v):
    return f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%"

col_labels = ["Metric", "Strategy", "SPY", "QQQ"]
rows = [
    ["Total Return",   fmt_ret(ret_pf),  fmt_ret(ret_spy),  fmt_ret(ret_qqq)],
    ["Final Value",    f"${final_pf:,.0f}", f"${final_spy:,.0f}", f"${final_qqq:,.0f}"],
    ["Max Drawdown",   f"{dd_pf:.1f}%",  f"{dd_spy:.1f}%",  f"{dd_qqq:.1f}%"],
    ["Sharpe Ratio",   f"{sh_pf:.2f}",   f"{sh_spy:.2f}",   f"{sh_qqq:.2f}"],
    ["2021",           fmt_ret(pf_annual.get(2021,0)), fmt_ret(spy_annual.get(2021,0)), fmt_ret(qqq_annual.get(2021,0))],
    ["2022",           fmt_ret(pf_annual.get(2022,0)), fmt_ret(spy_annual.get(2022,0)), fmt_ret(qqq_annual.get(2022,0))],
    ["2023",           fmt_ret(pf_annual.get(2023,0)), fmt_ret(spy_annual.get(2023,0)), fmt_ret(qqq_annual.get(2023,0))],
    ["2024",           fmt_ret(pf_annual.get(2024,0)), fmt_ret(spy_annual.get(2024,0)), fmt_ret(qqq_annual.get(2024,0))],
    ["2025",           fmt_ret(pf_annual.get(2025,0)), fmt_ret(spy_annual.get(2025,0)), fmt_ret(qqq_annual.get(2025,0))],
    ["2026 (YTD)",     fmt_ret(pf_annual.get(2026,0)), fmt_ret(spy_annual.get(2026,0)), fmt_ret(qqq_annual.get(2026,0))],
]

tbl = ax5.table(
    cellText=rows,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.0, 1.55)

# スタイル
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#21262d")
    cell.set_edgecolor("#30363d")
    cell.set_text_props(color="white")
    if r == 0:
        cell.set_facecolor("#1f6feb")
        cell.set_text_props(color="white", fontweight="bold")
    elif c == 1:
        cell.set_text_props(color=COLORS["Strategy"], fontweight="bold")
    elif c == 2:
        cell.set_text_props(color=COLORS["SPY"])
    elif c == 3:
        cell.set_text_props(color=COLORS["QQQ"])

ax5.set_title("Performance Summary", color="white", fontsize=12, fontweight="bold", pad=10)

# 保存
out = "output/backtest/comparison_chart.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved: {out}")
print(f"\n=== Summary ===")
print(f"Strategy : ${final_pf:,.2f}  (+{ret_pf:.1f}%)  MaxDD={dd_pf:.1f}%  Sharpe={sh_pf:.2f}")
print(f"SPY      : ${final_spy:,.2f}  (+{ret_spy:.1f}%)  MaxDD={dd_spy:.1f}%  Sharpe={sh_spy:.2f}")
print(f"QQQ      : ${final_qqq:,.2f}  (+{ret_qqq:.1f}%)  MaxDD={dd_qqq:.1f}%  Sharpe={sh_qqq:.2f}")
