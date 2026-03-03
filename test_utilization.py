"""
資金利用率の分析
risk=1.0% vs 0.5% で実際にどれだけキャッシュが遊んでいるか確認
→ max_positions や max_alloc を調整して最適解を探る
"""
import csv, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import yfinance as yf

BASE   = Path(__file__).resolve().parent
CSV_IN = BASE / "output/backtest/signal_backtest_20260303_022802_sb_trail4.0.csv"
INITIAL = 4000.0

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

def simulate(trades, initial, risk, alloc, max_pos):
    trades = sorted(trades, key=lambda t: t["entry_date"])
    same_day = {i for i,t in enumerate(trades) if t["entry_date"]==t["exit_date"]}
    events = []
    for i,t in enumerate(trades):
        events.append((t["entry_date"],"entry",i))
        events.append((t["exit_date"],"exit",i))
    def _key(e):
        d,et,i = e
        return (d, 2 if (et=="exit" and i in same_day) else (1 if et=="entry" else 0))
    events.sort(key=_key)

    cash=initial; active={}; peak=initial; max_dd=0.0
    history=[]; snapshots=[]
    pe,px=set(),set(); skipped=0

    for date,etype,idx in events:
        t=trades[idx]
        if etype=="exit" and idx in pe and idx not in px:
            px.add(idx)
            amt=active.pop(idx)
            pnl=amt*(t["pl_pct"]/100)
            cash+=amt+pnl
            bal=cash+sum(active.values())
            if bal>peak: peak=bal
            dd=(peak-bal)/peak*100 if peak>0 else 0
            if dd>max_dd: max_dd=dd
            history.append({"exit_date":date,"balance":bal,"pnl":pnl})

        elif etype=="entry" and idx not in pe:
            eq=cash+sum(active.values())
            # スキップ理由を記録
            if len(active)>=max_pos:
                skipped+=1; continue
            rr=(t["entry_price"]-t["sl_price"])/t["entry_price"] if t["entry_price"]>0 else 0.08
            rr=max(rr,0.005)
            inv=(eq*risk)/rr
            ap=alloc+(0.01 if t["signal"]=="STRONG_BUY" else 0)
            a=min(inv,eq*ap,cash)
            if t["entry_price"]>0:
                sh=math.floor(a/t["entry_price"]); a=sh*t["entry_price"]
            else: sh=0
            if a<=0 or sh==0 or cash<=0:
                skipped+=1; continue
            # エントリー時スナップショット
            eq_now=cash+sum(active.values())
            snapshots.append({
                "date":         date,
                "alloc_amt":    a,
                "alloc_pct":    a/eq_now*100 if eq_now>0 else 0,
                "cash_before":  cash,
                "cash_pct":     cash/eq_now*100 if eq_now>0 else 0,
                "n_active":     len(active),
                "risk_ratio":   rr,
                "inv_pct":      inv/eq_now*100,
                "capped":       inv > eq_now*ap,  # allocキャップが効いたか
            })
            cash-=a; active[idx]=a; pe.add(idx)

    final=cash+sum(active.values())
    wins=sum(1 for h in history if h["pnl"]>0)
    gp=sum(h["pnl"] for h in history if h["pnl"]>0)
    gl=abs(sum(h["pnl"] for h in history if h["pnl"]<0))
    pf=gp/gl if gl>0 else 9.99

    # 日次キャッシュ比率（全営業日）
    snap_df = pd.DataFrame(snapshots)

    return {
        "final":final, "ret":(final-initial)/initial*100,
        "trades":len(history), "win_rate":wins/len(history)*100 if history else 0,
        "pf":pf, "max_dd":max_dd, "history":history,
        "snapshots":snap_df, "skipped":skipped,
    }

def sharpe(history, initial):
    if not history: return 0.0
    s=pd.Series([h["balance"] for h in history],
                index=pd.to_datetime([h["exit_date"] for h in history]))
    s=s[~s.index.duplicated(keep="last")].sort_index()
    dr=pd.bdate_range(s.index[0],s.index[-1])
    s=s.reindex(dr).ffill().fillna(initial)
    r=s.pct_change().dropna()
    return (r.mean()/r.std())*np.sqrt(252) if r.std()>0 else 0.0

# ── 現状 vs 0.5%のみ まず比較 ────────────────────────────────────
r_cur  = simulate(trades_raw, INITIAL, 0.010, 0.13, 11)
r_half = simulate(trades_raw, INITIAL, 0.005, 0.13, 11)

print("=== 資金利用率 比較 ===")
print(f"{'指標':<30} {'risk=1.0%':>12} {'risk=0.5%':>12}")
print("-"*56)
for label, v1, v2 in [
    ("平均ポジションサイズ(%of equity)",
     r_cur["snapshots"]["alloc_pct"].mean(),
     r_half["snapshots"]["alloc_pct"].mean()),
    ("最大ポジションサイズ(%)",
     r_cur["snapshots"]["alloc_pct"].max(),
     r_half["snapshots"]["alloc_pct"].max()),
    ("平均同時保有数",
     r_cur["snapshots"]["n_active"].mean(),
     r_half["snapshots"]["n_active"].mean()),
    ("alloc上限キャップ率(%of entries)",
     r_cur["snapshots"]["capped"].mean()*100,
     r_half["snapshots"]["capped"].mean()*100),
    ("スキップ件数(pos上限超)",
     r_cur["skipped"], r_half["skipped"]),
    ("実行トレード数",
     r_cur["trades"], r_half["trades"]),
]:
    print(f"  {label:<30} {v1:>12.1f}  {v2:>12.1f}")

# キャッシュが遊んでいる割合を推定
print()
print("=== ポジションサイズ分布 (risk=0.5%) ===")
snap = r_half["snapshots"]
for lo,hi in [(0,5),(5,8),(8,10),(10,13),(13,100)]:
    cnt = ((snap["alloc_pct"]>=lo) & (snap["alloc_pct"]<hi)).sum()
    print(f"  {lo}〜{hi}%: {cnt}件 ({cnt/len(snap)*100:.1f}%)")

print()
print("=== 平均投資額 vs alloc上限 ===")
print(f"  risk=1.0%: 平均投資額={r_cur['snapshots']['inv_pct'].mean():.1f}% → "
      f"キャップ後={r_cur['snapshots']['alloc_pct'].mean():.1f}%  "
      f"(キャップ率={r_cur['snapshots']['capped'].mean()*100:.1f}%)")
print(f"  risk=0.5%: 平均投資額={r_half['snapshots']['inv_pct'].mean():.1f}% → "
      f"キャップ後={r_half['snapshots']['alloc_pct'].mean():.1f}%  "
      f"(キャップ率={r_half['snapshots']['capped'].mean()*100:.1f}%)")

# ── max_positions を増やして比較 ──────────────────────────────────
print()
print("=== risk=0.5% + max_positions 調整 ===")
print(f"{'Config':<25} {'Return':>9} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'Skip':>6} {'AvgPos%':>8} {'AvgN':>6}")
print("-"*80)

configs = [
    ("risk=1.0%  pos=11 [現状]",  0.010, 0.13, 11),
    ("risk=0.5%  pos=11",          0.005, 0.13, 11),
    ("risk=0.5%  pos=15",          0.005, 0.13, 15),
    ("risk=0.5%  pos=18",          0.005, 0.13, 18),
    ("risk=0.5%  pos=22",          0.005, 0.13, 22),
    ("risk=0.5%  pos=11 alloc=20%",0.005, 0.20, 11),
    ("risk=0.5%  pos=15 alloc=18%",0.005, 0.18, 15),
]

best_results = {}
for label, risk, alloc, max_pos in configs:
    res = simulate(trades_raw, INITIAL, risk, alloc, max_pos)
    sh  = sharpe(res["history"], INITIAL)
    snap= res["snapshots"]
    avg_pos = snap["alloc_pct"].mean() if len(snap)>0 else 0
    avg_n   = snap["n_active"].mean()  if len(snap)>0 else 0
    best_results[label] = {**res, "sharpe": sh, "avg_pos": avg_pos, "avg_n": avg_n}
    marker = " <--" if "現状" in label else ""
    print(f"  {label:<25} {res['ret']:>+8.1f}% {sh:>8.2f} {res['max_dd']:>7.1f}% "
          f"{res['trades']:>8} {res['skipped']:>6} {avg_pos:>7.1f}% {avg_n:>6.1f}{marker}")

# ── チャート ──────────────────────────────────────────────────────
print("\nチャート生成中...")

spy_raw = yf.download("SPY", start="2021-01-04", end="2026-03-01", progress=False)
spy = (spy_raw["Close"]["SPY"] if isinstance(spy_raw.columns, pd.MultiIndex)
       else spy_raw["Close"]).dropna()
spy_norm = spy/spy.iloc[0]*INITIAL
spy_ret  = (spy.iloc[-1]-spy.iloc[0])/spy.iloc[0]*100

def make_eq(history, initial):
    s=pd.Series([h["balance"] for h in history],
                index=pd.to_datetime([h["exit_date"] for h in history]))
    s=s[~s.index.duplicated(keep="last")].sort_index()
    dr=pd.bdate_range(s.index[0],s.index[-1])
    return s.reindex(dr).ffill().fillna(initial)

fig = plt.figure(figsize=(16, 18), facecolor="#0d1117")
fig.suptitle("risk=0.5%  |  Capital Utilization & Position Adjustment\n"
             "Comparing max_positions settings to address idle cash",
             color="white", fontsize=14, fontweight="bold", y=0.98)

gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.32,
                      top=0.93, bottom=0.05, left=0.08, right=0.97)

def style_ax(ax):
    ax.set_facecolor("#161b22"); ax.tick_params(colors="#adbac7")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", lw=0.7, linestyle="--")

highlight = {
    "risk=1.0%  pos=11 [現状]":      ("#ff6b6b", 2.0, "--"),
    "risk=0.5%  pos=11":             ("#888888", 1.2, "-"),
    "risk=0.5%  pos=15":             ("#7ee787", 1.8, "-"),
    "risk=0.5%  pos=18":             ("#00d4ff", 2.0, "-"),
    "risk=0.5%  pos=22":             ("#f0a500", 1.5, "-"),
    "risk=0.5%  pos=11 alloc=20%":   ("#c084fc", 1.5, "-"),
    "risk=0.5%  pos=15 alloc=18%":   ("#fb923c", 2.0, "-"),
}

# Panel 1: 資産曲線
ax1 = fig.add_subplot(gs[0,:])
style_ax(ax1)
for label, res in best_results.items():
    col, lw, ls = highlight.get(label, ("#888888", 1.2, "-"))
    eq  = make_eq(res["history"], INITIAL)
    lbl = f"{label}  ${res['final']:,.0f} ({res['ret']:+.1f}%)  Sh={res['sharpe']:.2f}"
    ax1.plot(eq.index, eq, color=col, lw=lw, linestyle=ls, label=lbl)
ax1.plot(spy_norm.index, spy_norm, color="#444", lw=1, alpha=0.5,
         label=f"SPY ({spy_ret:+.1f}%)")
ax1.axhline(INITIAL, color="#6e7681", lw=0.8, linestyle=":")
ax1.set_title("Portfolio Value  ($4,000 start)", color="white", fontsize=12, fontweight="bold")
ax1.set_ylabel("Value ($)", color="#adbac7")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white",
           fontsize=8.5, loc="upper left")

# Panel 2: Sharpe
ax2 = fig.add_subplot(gs[1,0])
style_ax(ax2)
lbls   = list(best_results.keys())
sharpes= [best_results[l]["sharpe"] for l in lbls]
cols2  = [highlight[l][0] for l in lbls]
bars2  = ax2.barh(range(len(lbls)), sharpes, color=cols2, alpha=0.85)
ax2.set_yticks(range(len(lbls)))
ax2.set_yticklabels([l.replace("risk=","r=").replace("alloc=","a=").replace("pos=","p=")
                      for l in lbls], fontsize=8.5, color="#adbac7")
ax2.set_title("Sharpe Ratio", color="white", fontsize=12, fontweight="bold")
ax2.axvline(1.58, color="#ff6b6b", lw=1, linestyle="--", label="current=1.58")
ax2.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=8)
for bar, val in zip(bars2, sharpes):
    ax2.text(val+0.01, bar.get_y()+bar.get_height()/2,
             f"{val:.2f}", va="center", color="white", fontsize=8.5)

# Panel 3: Return vs MaxDD バブル
ax3 = fig.add_subplot(gs[1,1])
style_ax(ax3)
for label, res in best_results.items():
    col, lw, _ = highlight[label]
    ax3.scatter(res["max_dd"], res["ret"], color=col, s=120, zorder=3)
    ax3.annotate(label.replace("risk=","r=").replace(" pos=","\np=").replace(" alloc=","\na="),
                 (res["max_dd"], res["ret"]), fontsize=7, color="#adbac7",
                 xytext=(4,3), textcoords="offset points")
ax3.set_xlabel("Max Drawdown (%)", color="#adbac7")
ax3.set_ylabel("Total Return (%)", color="#adbac7")
ax3.set_title("Return vs MaxDD  (upper-left = better)", color="white",
              fontsize=12, fontweight="bold")

# Panel 4: サマリーテーブル
ax4 = fig.add_subplot(gs[2,:])
style_ax(ax4); ax4.axis("off")

col_labels = ["Config","Return","Final $","Sharpe","MaxDD","Trades","Skip","AvgPos%","AvgActive"]
rows_tbl = []
for label, res in best_results.items():
    rows_tbl.append([
        label,
        f"{res['ret']:+.1f}%", f"${res['final']:,.0f}",
        f"{res['sharpe']:.2f}", f"{res['max_dd']:.1f}%",
        str(res['trades']), str(res['skipped']),
        f"{res['avg_pos']:.1f}%", f"{res['avg_n']:.1f}",
    ])

tbl = ax4.table(cellText=rows_tbl, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 2.1)
for (r,c), cell in tbl.get_celld().items():
    cell.set_facecolor("#21262d"); cell.set_edgecolor("#30363d")
    cell.set_text_props(color="white")
    if r==0:
        cell.set_facecolor("#1f6feb")
        cell.set_text_props(color="white", fontweight="bold")
    elif r==1:  # 現状
        cell.set_facecolor("#1a0f0f")
        cell.set_text_props(color="#ff6b6b")
    elif r==4:  # pos=18 (推奨候補)
        cell.set_text_props(color="#00d4ff", fontweight="bold")

ax4.set_title("Config Summary", color="white", fontsize=12, fontweight="bold", pad=8)

out = BASE/"output/backtest/utilization_chart.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved: {out}")
