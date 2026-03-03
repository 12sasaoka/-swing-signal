"""
最新バックテスト結果：資産推移 + 待機資金チャート
risk=0.5%, trail4.0, $4,000スタート
"""
import csv, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
import yfinance as yf

BASE    = Path(__file__).resolve().parent.parent.parent
CSV_IN  = BASE / "output/backtest/signal_backtest_20260303_022802_sb_trail4.0.csv"
INITIAL = 4000.0
RISK    = 0.005
ALLOC   = 0.13
MAX_POS = 11

# ── トレード読み込み ────────────────────────────────────────────
trades_raw = []
with open(CSV_IN, "r", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        trades_raw.append({
            "signal":      row["signal"],
            "entry_date":  row["entry_date"],
            "exit_date":   row["exit_date"],
            "entry_price": float(row["entry_price"]),
            "pl_pct":      float(row["pl_pct"]),
            "sl_price":    float(row["sl_price"]),
        })

# ── シミュレーション（日次スナップショット付き） ──────────────
trades_raw = sorted(trades_raw, key=lambda t: t["entry_date"])
same_day = {i for i,t in enumerate(trades_raw) if t["entry_date"]==t["exit_date"]}
events = []
for i,t in enumerate(trades_raw):
    events.append((t["entry_date"], "entry", i))
    events.append((t["exit_date"],  "exit",  i))
def _key(e):
    d,et,i = e
    return (d, 2 if (et=="exit" and i in same_day) else (1 if et=="entry" else 0))
events.sort(key=_key)

cash = INITIAL; active = {}; peak = INITIAL; max_dd = 0.0
history = []; snapshots = []
pe, px = set(), set()

for date, etype, idx in events:
    t = trades_raw[idx]
    if etype == "exit" and idx in pe and idx not in px:
        px.add(idx)
        amt = active.pop(idx)
        pnl = amt * (t["pl_pct"] / 100)
        cash += amt + pnl
        bal = cash + sum(active.values())
        if bal > peak: peak = bal
        dd = (peak - bal) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd
        history.append({"date": date, "balance": bal, "pnl": pnl, "cash": cash,
                        "invested": sum(active.values()), "n_pos": len(active)})
    elif etype == "entry" and idx not in pe:
        if len(active) >= MAX_POS: continue
        eq  = cash + sum(active.values())
        rr  = max((t["entry_price"] - t["sl_price"]) / t["entry_price"]
                  if t["entry_price"] > 0 else 0.08, 0.005)
        inv = (eq * RISK) / rr
        ap  = ALLOC + (0.01 if t["signal"] == "STRONG_BUY" else 0)
        a   = min(inv, eq * ap, cash)
        if t["entry_price"] > 0:
            sh = math.floor(a / t["entry_price"]); a = sh * t["entry_price"]
        else: sh = 0
        if a <= 0 or sh == 0 or cash <= 0: continue
        cash -= a; active[idx] = a; pe.add(idx)
        snapshots.append({"date": date, "cash": cash,
                          "invested": sum(active.values()),
                          "n_pos": len(active)})

# 日次系列に変換
df_hist = pd.DataFrame(history)
df_hist["date"] = pd.to_datetime(df_hist["date"])
df_hist = df_hist.sort_values("date")

# 同日に複数トレード終了 → 日付ごとに最終値を取る
df_daily = df_hist.groupby("date").last().reset_index()
df_daily = df_daily.sort_values("date")

# 営業日インデックスで補完
bdays = pd.bdate_range(df_daily["date"].min(), df_daily["date"].max())
eq_series   = df_daily.set_index("date")["balance"].reindex(bdays).ffill().fillna(INITIAL)
cash_series = df_daily.set_index("date")["cash"].reindex(bdays).ffill().fillna(INITIAL)
inv_series  = df_daily.set_index("date")["invested"].reindex(bdays).ffill().fillna(0)
npos_series = df_daily.set_index("date")["n_pos"].reindex(bdays).ffill().fillna(0)

cash_pct = cash_series / eq_series * 100
inv_pct  = inv_series  / eq_series * 100

# ── SPY ───────────────────────────────────────────────────────
raw = yf.download("SPY", start="2021-01-04", end="2026-03-01", progress=False)
spy = (raw["Close"]["SPY"] if isinstance(raw.columns, pd.MultiIndex)
       else raw["Close"]).dropna()
spy_norm = spy / spy.iloc[0] * INITIAL

# ── 年別サマリー ──────────────────────────────────────────────
yearly = {}
for _, row in df_hist.iterrows():
    yr = str(row["date"].year)
    yearly.setdefault(yr, []).append(row)

yr_stats = {}
for yr, rows in yearly.items():
    all_rows = [r for r in df_hist.itertuples()
                if str(r.date.year) <= yr]
    prev_rows = [r for r in df_hist.itertuples()
                 if str(r.date.year) < yr]
    start_bal = prev_rows[-1].balance if prev_rows else INITIAL
    end_bal   = rows[-1]["balance"]
    yr_ret    = (end_bal - start_bal) / start_bal * 100
    yr_pnl    = sum(r["pnl"] for r in rows)
    yr_stats[yr] = {"ret": yr_ret, "pnl": yr_pnl,
                    "end": end_bal, "trades": len(rows)}

# ── 統計 ──────────────────────────────────────────────────────
final_bal  = eq_series.iloc[-1]
total_ret  = (final_bal - INITIAL) / INITIAL * 100
avg_cash_pct = cash_pct.mean()
avg_inv_pct  = inv_pct.mean()
wins   = sum(1 for h in history if h["pnl"] > 0)
n_tr   = len(history)
gp     = sum(h["pnl"] for h in history if h["pnl"] > 0)
gl     = abs(sum(h["pnl"] for h in history if h["pnl"] < 0))
pf     = gp / gl if gl > 0 else 9.99
spy_ret = (spy.iloc[-1] - spy.iloc[0]) / spy.iloc[0] * 100

# ── 描画 ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 22), facecolor="#0d1117")
fig.suptitle(
    f"Swing Signal Backtest  |  risk=0.5%  trail4.0  |  Jan 2021 - Feb 2026\n"
    f"Initial: ${INITIAL:,.0f}   Final: ${final_bal:,.2f}   Return: {total_ret:+.1f}%",
    color="white", fontsize=15, fontweight="bold", y=0.99
)

gs = fig.add_gridspec(4, 2, hspace=0.48, wspace=0.30,
                      top=0.95, bottom=0.04, left=0.08, right=0.97)

def style_ax(ax):
    ax.set_facecolor("#161b22"); ax.tick_params(colors="#adbac7", labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", lw=0.7, linestyle="--")

C_EQ   = "#00d4ff"
C_CASH = "#f0a500"
C_INV  = "#7ee787"
C_SPY  = "#888888"
C_POS  = "#c084fc"

# ─── Panel 1: 資産推移（絶対額） ────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
style_ax(ax1)

ax1.fill_between(eq_series.index, INITIAL, eq_series,
                 where=(eq_series >= INITIAL), alpha=0.08, color=C_EQ)
ax1.fill_between(eq_series.index, INITIAL, eq_series,
                 where=(eq_series < INITIAL),  alpha=0.15, color="red")
ax1.plot(eq_series.index, eq_series, color=C_EQ, lw=2.2,
         label=f"Strategy  ${final_bal:,.0f}  ({total_ret:+.1f}%)", zorder=3)
ax1.plot(spy_norm.index, spy_norm, color=C_SPY, lw=1.3, alpha=0.7,
         label=f"SPY  ${spy_norm.iloc[-1]:,.0f}  ({spy_ret:+.1f}%)")
ax1.axhline(INITIAL, color="#6e7681", lw=0.8, linestyle=":")

# 年末マーカー + 残高注記
for yr, st in yr_stats.items():
    if yr == "2026": continue
    yr_end_date = pd.Timestamp(f"{yr}-12-31")
    nearest = eq_series.index[eq_series.index <= yr_end_date]
    if len(nearest) == 0: continue
    d = nearest[-1]; v = eq_series[d]
    ax1.annotate(f"${v:,.0f}", xy=(d, v),
                 xytext=(0, 12), textcoords="offset points",
                 color=C_EQ, fontsize=8, ha="center",
                 arrowprops=dict(arrowstyle="-", color=C_EQ, lw=0.7))

# 最終値
ax1.annotate(f"${final_bal:,.0f}", xy=(eq_series.index[-1], final_bal),
             xytext=(-55, 10), textcoords="offset points",
             color=C_EQ, fontsize=10, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C_EQ))

ax1.set_title("Portfolio Value", color="white", fontsize=13, fontweight="bold")
ax1.set_ylabel("Value ($)", color="#adbac7")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white",
           fontsize=10, loc="upper left")
ax1.set_xlim(eq_series.index[0], eq_series.index[-1])

# ─── Panel 2: 投資額 vs 待機資金（絶対額スタック） ───────────────
ax2 = fig.add_subplot(gs[1, :])
style_ax(ax2)

ax2.stackplot(eq_series.index,
              inv_series.values, cash_series.values,
              labels=[f"Deployed (avg ${inv_series.mean():,.0f})",
                      f"Idle Cash (avg ${cash_series.mean():,.0f})"],
              colors=[C_INV, C_CASH], alpha=0.75)
ax2.plot(eq_series.index, eq_series, color=C_EQ, lw=1.5, linestyle="--",
         alpha=0.6, label="Total Equity")

# 年ラベル
for yr in ["2022","2023","2024","2025","2026"]:
    ax2.axvline(pd.Timestamp(f"{yr}-01-01"), color="#6e7681",
                lw=0.7, linestyle="--", alpha=0.5)

# 年別平均待機資金の注記
for yr in ["2021","2022","2023","2024","2025"]:
    mask = (eq_series.index.year == int(yr))
    avg_idle = cash_series[mask].mean()
    avg_eq   = eq_series[mask].mean()
    mid_date = pd.Timestamp(f"{yr}-07-01")
    ax2.text(mid_date, avg_eq * 0.5, f"{yr}\nidle\n${avg_idle:,.0f}",
             ha="center", va="center", color="#0d1117",
             fontsize=7.5, fontweight="bold")

ax2.set_title("Deployed Capital  vs  Idle Cash  (absolute $)", color="white",
              fontsize=13, fontweight="bold")
ax2.set_ylabel("Amount ($)", color="#adbac7")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white",
           fontsize=9, loc="upper left")
ax2.set_xlim(eq_series.index[0], eq_series.index[-1])

# ─── Panel 3: 待機資金 % + 保有銘柄数 ───────────────────────────
ax3 = fig.add_subplot(gs[2, :])
style_ax(ax3)

ax3.fill_between(eq_series.index, cash_pct, alpha=0.35, color=C_CASH, label="Idle Cash %")
ax3.plot(eq_series.index, cash_pct, color=C_CASH, lw=1.5)
ax3.axhline(cash_pct.mean(), color=C_CASH, lw=1.2, linestyle="--",
            label=f"Average idle: {cash_pct.mean():.1f}%")
ax3.axhline(40, color="#6e7681", lw=0.7, linestyle=":", alpha=0.6)

ax3r = ax3.twinx()
ax3r.plot(eq_series.index, npos_series, color=C_POS, lw=1.2, alpha=0.7,
          label=f"Active positions (avg {npos_series.mean():.1f})")
ax3r.set_ylabel("# Positions", color=C_POS)
ax3r.tick_params(colors=C_POS)
ax3r.set_ylim(0, MAX_POS + 3)

# 年別帯
for yr, col in [("2021","#1a2a1a"),("2022","#2a1a1a"),("2023","#1a2a1a"),
                ("2024","#1a2a2a"),("2025","#1a1a2a")]:
    s = pd.Timestamp(f"{yr}-01-01"); e = pd.Timestamp(f"{yr}-12-31")
    s = max(s, eq_series.index[0]); e = min(e, eq_series.index[-1])
    ax3.axvspan(s, e, alpha=0.12, color=col)
    mask = (eq_series.index.year == int(yr))
    avg_c = cash_pct[mask].mean()
    mid   = pd.Timestamp(f"{yr}-07-01")
    ax3.text(mid, 92, f"{yr}\n{avg_c:.0f}% idle",
             ha="center", va="top", color="#adbac7", fontsize=8)

ax3.set_title("Idle Cash (%)  &  Active Positions", color="white",
              fontsize=13, fontweight="bold")
ax3.set_ylabel("Idle Cash (%)", color=C_CASH)
ax3.set_ylim(0, 100)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax3.xaxis.set_major_locator(mdates.YearLocator())
lines1, lab1 = ax3.get_legend_handles_labels()
lines2, lab2 = ax3r.get_legend_handles_labels()
ax3.legend(lines1+lines2, lab1+lab2, facecolor="#21262d",
           edgecolor="#30363d", labelcolor="white", fontsize=9, loc="upper right")
ax3.set_xlim(eq_series.index[0], eq_series.index[-1])

# ─── Panel 4: 年別リターン棒グラフ ──────────────────────────────
ax4 = fig.add_subplot(gs[3, 0])
style_ax(ax4)

yrs  = sorted(yr_stats.keys())
rets = [yr_stats[y]["ret"] for y in yrs]
cols = [C_EQ if r >= 0 else "#ff6b6b" for r in rets]
bars = ax4.bar(yrs, rets, color=cols, alpha=0.85, width=0.6)
ax4.axhline(0, color="#6e7681", lw=0.8)
for bar, yr, val in zip(bars, yrs, rets):
    pnl = yr_stats[yr]["pnl"]
    end = yr_stats[yr]["end"]
    ax4.text(bar.get_x() + bar.get_width()/2,
             val + (1.5 if val >= 0 else -2.5),
             f"{val:+.1f}%\n${pnl:+,.0f}",
             ha="center", va="bottom" if val >= 0 else "top",
             color="white", fontsize=8.5, fontweight="bold")

ax4.set_title("Annual Return  &  P/L ($)", color="white",
              fontsize=12, fontweight="bold")
ax4.set_ylabel("Return (%)", color="#adbac7")
ax4.tick_params(axis="x", colors="#adbac7")

# ─── Panel 5: 統計サマリー ───────────────────────────────────────
ax5 = fig.add_subplot(gs[3, 1])
style_ax(ax5); ax5.axis("off")

def fmt(v): return f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%"

stats_data = [
    ("Initial Capital",   f"${INITIAL:,.0f}"),
    ("Final Balance",     f"${final_bal:,.2f}"),
    ("Total Return",      fmt(total_ret)),
    ("Total P/L",         f"${final_bal-INITIAL:+,.2f}"),
    ("",                  ""),
    ("Total Trades",      str(n_tr)),
    ("Win Rate",          f"{wins/n_tr*100:.1f}%"),
    ("Profit Factor",     f"{pf:.3f}"),
    ("Max Drawdown",      f"{max_dd:.1f}%"),
    ("",                  ""),
    ("Avg Idle Cash",     f"{cash_pct.mean():.1f}%  (${cash_series.mean():,.0f})"),
    ("Avg Deployed",      f"{inv_pct.mean():.1f}%  (${inv_series.mean():,.0f})"),
    ("Avg Positions",     f"{npos_series.mean():.1f} / {MAX_POS}"),
    ("",                  ""),
    ("vs SPY",            f"{total_ret - spy_ret:+.1f}% outperformance"),
]

y_pos = 0.97
for label, val in stats_data:
    if label == "":
        y_pos -= 0.045; continue
    weight = "bold" if label in ("Final Balance","Total Return","Avg Idle Cash") else "normal"
    color  = C_EQ if label == "Final Balance" else \
             C_CASH if "Idle" in label else \
             C_INV  if "Deployed" in label else "white"
    ax5.text(0.02, y_pos, label, transform=ax5.transAxes,
             color="#adbac7", fontsize=10, va="top")
    ax5.text(0.58, y_pos, val, transform=ax5.transAxes,
             color=color, fontsize=10, va="top", fontweight=weight)
    y_pos -= 0.065

ax5.set_title("Performance Summary", color="white",
              fontsize=12, fontweight="bold", pad=10)

out = BASE / "output/backtest/equity_idle_chart.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved: {out}")
print(f"Final: ${final_bal:,.2f} ({total_ret:+.1f}%)")
print(f"Avg idle cash: {cash_pct.mean():.1f}% (${cash_series.mean():,.0f})")
