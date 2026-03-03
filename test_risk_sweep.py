"""
リスク/トレード スイープ比較
0.5% / 0.7% / 0.75% / 0.8% / 1.0% / 1.3% を比較
固定: alloc=13%, pos=11, trail4.0 CSV
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

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE    = Path(__file__).resolve().parent
CSV_IN  = BASE / "output/backtest/signal_backtest_20260303_022802_sb_trail4.0.csv"
INITIAL = 4000.0

RISK_LEVELS = [0.005, 0.007, 0.0075, 0.008, 0.010, 0.013]
MAX_ALLOC   = 0.13
MAX_POS     = 11

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

# ── シミュレーション関数 ────────────────────────────────────────
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

    cash=initial; active={}; peak=initial; max_dd=0.0; history=[]
    pe,px=set(),set()
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
            if len(active)>=max_pos: continue
            eq=cash+sum(active.values())
            rr=(t["entry_price"]-t["sl_price"])/t["entry_price"] if t["entry_price"]>0 else 0.08
            rr=max(rr,0.005)
            inv=(eq*risk)/rr
            ap=alloc+(0.01 if t["signal"]=="STRONG_BUY" else 0)
            a=min(inv,eq*ap,cash)
            if t["entry_price"]>0:
                sh=math.floor(a/t["entry_price"]); a=sh*t["entry_price"]
            else: sh=0
            if a<=0 or sh==0 or cash<=0: continue
            cash-=a; active[idx]=a; pe.add(idx)

    final=cash+sum(active.values())
    wins  =sum(1 for h in history if h["pnl"]>0)
    gp    =sum(h["pnl"] for h in history if h["pnl"]>0)
    gl    =abs(sum(h["pnl"] for h in history if h["pnl"]<0))
    pf    =gp/gl if gl>0 else 9.99

    # 年別
    yearly={}
    for h in history:
        yr=h["exit_date"][:4]
        yearly.setdefault(yr,[]).append(h)

    return {
        "final":final, "ret":(final-initial)/initial*100,
        "trades":len(history), "win_rate":wins/len(history)*100 if history else 0,
        "pf":pf, "max_dd":max_dd, "history":history, "yearly":yearly,
    }

def make_equity(history, initial):
    if not history: return pd.Series(dtype=float)
    s=pd.Series([h["balance"] for h in history],
                index=pd.to_datetime([h["exit_date"] for h in history]))
    s=s[~s.index.duplicated(keep="last")].sort_index()
    dr=pd.bdate_range(s.index[0],s.index[-1])
    return s.reindex(dr).ffill().fillna(initial)

def sharpe(eq):
    r=eq.pct_change().dropna()
    return (r.mean()/r.std())*np.sqrt(252) if r.std()>0 else 0.0

# ── 全リスクレベル実行 ──────────────────────────────────────────
print(f"{'Risk':>7} {'Final $':>12} {'Return':>9} {'Sharpe':>8} {'MaxDD':>8} "
      f"{'WinRate':>8} {'PF':>7} {'Trades':>7}")
print("-"*66)

results = {}
for risk in RISK_LEVELS:
    res = simulate(trades_raw, INITIAL, risk, MAX_ALLOC, MAX_POS)
    eq  = make_equity(res["history"], INITIAL)
    sh  = sharpe(eq.dropna())
    res["sharpe"] = sh
    res["equity"] = eq
    results[risk] = res
    marker = " <-- current" if risk==0.010 else ""
    print(f"{risk*100:>6.2f}%  ${res['final']:>11,.2f}  {res['ret']:>+8.1f}%  "
          f"{sh:>8.2f}  {res['max_dd']:>7.1f}%  {res['win_rate']:>7.1f}%  "
          f"{res['pf']:>7.3f}  {res['trades']:>7}{marker}")

# ── 年別リターン ─────────────────────────────────────────────────
print()
years = ["2021","2022","2023","2024","2025","2026"]
header = f"{'Year':<6}" + "".join(f"  risk{r*100:.2f}%" for r in RISK_LEVELS)
print(header)
print("-"*len(header))
for yr in years:
    row = f"{yr:<6}"
    for risk in RISK_LEVELS:
        res  = results[risk]
        hist = res["yearly"].get(yr,[])
        if hist:
            start = results[risk]["history"][0]["balance"] if yr=="2021" else None
            # 年末残高と年始残高から年次リターン計算
            yr_end  = hist[-1]["balance"]
            all_h   = res["history"]
            prev_yr = [h for h in all_h if h["exit_date"][:4] < yr]
            yr_start = prev_yr[-1]["balance"] if prev_yr else INITIAL
            yr_ret  = (yr_end - yr_start)/yr_start*100
            row += f"  {yr_ret:>+8.1f}%"
        else:
            row += f"  {'---':>9}"
    print(row)

# ── SPY / QQQ ───────────────────────────────────────────────────
raw = yf.download(["SPY","QQQ"], start="2021-01-04", end="2026-03-01", progress=False)
spy = raw["Close"]["SPY"].dropna()
qqq = raw["Close"]["QQQ"].dropna()
spy_norm = spy/spy.iloc[0]*INITIAL
qqq_norm = qqq/qqq.iloc[0]*INITIAL
spy_ret  = (spy.iloc[-1]-spy.iloc[0])/spy.iloc[0]*100
qqq_ret  = (qqq.iloc[-1]-qqq.iloc[0])/qqq.iloc[0]*100
spy_sh   = sharpe(spy_norm.dropna())
qqq_sh   = sharpe(qqq_norm.dropna())

# ── 描画 ─────────────────────────────────────────────────────────
CMAP   = plt.cm.plasma(np.linspace(0.15, 0.85, len(RISK_LEVELS)))
COLORS = {r: CMAP[i] for i,r in enumerate(RISK_LEVELS)}
C_SPY  = "#7ee787"
C_CUR  = "#ff6b6b"   # 現在値 (1.0%) のハイライト

fig = plt.figure(figsize=(16, 20), facecolor="#0d1117")
fig.suptitle("Risk-per-Trade Sweep  |  0.5% ~ 1.3%\n"
             "Fixed: alloc=13%, pos=11, trail4.0  |  Jan 2021 - Feb 2026",
             color="white", fontsize=14, fontweight="bold", y=0.98)

gs = fig.add_gridspec(4, 2, hspace=0.42, wspace=0.32,
                      top=0.93, bottom=0.04, left=0.08, right=0.97)

def style_ax(ax):
    ax.set_facecolor("#161b22"); ax.tick_params(colors="#adbac7")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", lw=0.7, linestyle="--")

# Panel 1: 資産曲線
ax1 = fig.add_subplot(gs[0,:])
style_ax(ax1)
for risk,res in results.items():
    eq  = res["equity"]
    lw  = 2.5 if risk==0.010 else 1.5
    ls  = "--" if risk==0.010 else "-"
    lbl = f"risk={risk*100:.2f}%  ${res['final']:,.0f} ({res['ret']:+.1f}%)  Sh={res['sharpe']:.2f}"
    ax1.plot(eq.index, eq, color=COLORS[risk], lw=lw, linestyle=ls, label=lbl)
ax1.plot(spy_norm.index, spy_norm, color=C_SPY, lw=1.2, alpha=0.6,
         label=f"SPY  ${spy_norm.iloc[-1]:,.0f} ({spy_ret:+.1f}%)")
ax1.axhline(INITIAL, color="#6e7681", lw=0.8, linestyle=":")
ax1.set_title("Portfolio Value  ($4,000 start)", color="white", fontsize=12, fontweight="bold")
ax1.set_ylabel("Value ($)", color="#adbac7")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=9,
           loc="upper left")

# Panel 2: Sharpe vs Risk
ax2 = fig.add_subplot(gs[1,0])
style_ax(ax2)
risks_pct  = [r*100 for r in RISK_LEVELS]
sharpes    = [results[r]["sharpe"] for r in RISK_LEVELS]
bar_colors = [C_CUR if r==0.010 else COLORS[r] for r in RISK_LEVELS]
bars = ax2.bar([f"{r:.2f}%" for r in risks_pct], sharpes, color=bar_colors, alpha=0.85)
ax2.axhline(spy_sh, color=C_SPY, lw=1.2, linestyle="--", label=f"SPY Sharpe={spy_sh:.2f}")
for bar,val in zip(bars,sharpes):
    ax2.text(bar.get_x()+bar.get_width()/2, val+0.02, f"{val:.2f}",
             ha="center", va="bottom", color="white", fontsize=9)
ax2.set_title("Sharpe Ratio vs Risk Level", color="white", fontsize=12, fontweight="bold")
ax2.set_ylabel("Sharpe Ratio", color="#adbac7")
ax2.tick_params(axis="x", colors="#adbac7", labelsize=9)
ax2.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=9)

# Panel 3: MaxDD vs Risk
ax3 = fig.add_subplot(gs[1,1])
style_ax(ax3)
max_dds = [results[r]["max_dd"] for r in RISK_LEVELS]
bars3 = ax3.bar([f"{r:.2f}%" for r in risks_pct], max_dds, color=bar_colors, alpha=0.85)
spy_dd = abs(((spy_norm-spy_norm.cummax())/spy_norm.cummax()*100).min())
ax3.axhline(spy_dd, color=C_SPY, lw=1.2, linestyle="--", label=f"SPY MaxDD={spy_dd:.1f}%")
for bar,val in zip(bars3,max_dds):
    ax3.text(bar.get_x()+bar.get_width()/2, val+0.2, f"{val:.1f}%",
             ha="center", va="bottom", color="white", fontsize=9)
ax3.set_title("Max Drawdown vs Risk Level", color="white", fontsize=12, fontweight="bold")
ax3.set_ylabel("Max Drawdown (%)", color="#adbac7")
ax3.tick_params(axis="x", colors="#adbac7", labelsize=9)
ax3.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=9)

# Panel 4: 年別リターン（ヒートマップ風）
ax4 = fig.add_subplot(gs[2,:])
style_ax(ax4)
risk_labels = [f"{r*100:.2f}%" for r in RISK_LEVELS]
x = np.arange(len(years))
w = 0.13
offsets = np.linspace(-(len(RISK_LEVELS)-1)/2*w, (len(RISK_LEVELS)-1)/2*w, len(RISK_LEVELS))
for i,(risk,offset) in enumerate(zip(RISK_LEVELS,offsets)):
    res  = results[risk]
    rets = []
    for yr in years:
        hist = res["yearly"].get(yr,[])
        if hist:
            all_h   = res["history"]
            prev_yr = [h for h in all_h if h["exit_date"][:4]<yr]
            yr_start= prev_yr[-1]["balance"] if prev_yr else INITIAL
            yr_end  = hist[-1]["balance"]
            rets.append((yr_end-yr_start)/yr_start*100)
        else:
            rets.append(0)
    lw  = 1.5 if risk==0.010 else 0
    ec  = C_CUR if risk==0.010 else "none"
    ax4.bar(x+offset, rets, w, color=COLORS[risk], alpha=0.85,
            linewidth=lw, edgecolor=ec,
            label=f"risk={risk*100:.2f}%")
ax4.axhline(0, color="#6e7681", lw=0.8)
ax4.set_xticks(x); ax4.set_xticklabels(years, color="#adbac7")
ax4.set_title("Annual Return (%) by Risk Level  |  red border = current (1.0%)",
              color="white", fontsize=12, fontweight="bold")
ax4.set_ylabel("Return (%)", color="#adbac7")
ax4.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white",
           fontsize=8.5, ncol=3, loc="upper left")

# Panel 5: サマリーテーブル
ax5 = fig.add_subplot(gs[3,:])
style_ax(ax5); ax5.axis("off")

def fmt(v): return f"+{v:.1f}%" if v>=0 else f"{v:.1f}%"

col_labels = ["Risk/Trade","Total Return","Final $","Max DD","Sharpe","PF","WinRate","Trades"]
rows_tbl   = []
for risk in RISK_LEVELS:
    r   = results[risk]
    tag = " ← current" if risk==0.010 else ""
    rows_tbl.append([
        f"{risk*100:.2f}%{tag}",
        fmt(r["ret"]), f"${r['final']:,.0f}",
        f"{r['max_dd']:.1f}%", f"{r['sharpe']:.2f}",
        f"{r['pf']:.3f}", f"{r['win_rate']:.1f}%", str(r["trades"]),
    ])
rows_tbl.append(["SPY", fmt(spy_ret), f"${spy_norm.iloc[-1]:,.0f}",
                 f"{spy_dd:.1f}%", f"{spy_sh:.2f}", "-", "-", "-"])

tbl = ax5.table(cellText=rows_tbl, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.0, 2.2)
for (r,c), cell in tbl.get_celld().items():
    cell.set_facecolor("#21262d"); cell.set_edgecolor("#30363d")
    cell.set_text_props(color="white")
    if r==0:
        cell.set_facecolor("#1f6feb")
        cell.set_text_props(color="white", fontweight="bold")
    # 現在値行をハイライト
    elif r==5:   # risk=1.0%は5番目(0-indexed: 4, tblは1始まり)
        cell.set_facecolor("#1a1a2e")
        cell.set_text_props(color=C_CUR)
    # SPY行
    elif r==len(rows_tbl):
        cell.set_facecolor("#0d2818")
        cell.set_text_props(color=C_SPY)
    # Sharpe列: 最大値を緑に
    if r>0 and c==4 and r<=len(RISK_LEVELS):
        sh_val = results[RISK_LEVELS[r-1]]["sharpe"]
        if sh_val == max(results[rr]["sharpe"] for rr in RISK_LEVELS):
            cell.set_text_props(color="#00d4ff", fontweight="bold")
ax5.set_title("Performance Summary", color="white", fontsize=12, fontweight="bold", pad=10)

out = BASE/"output/backtest/risk_sweep_chart.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"\nChart saved: {out}")

# ── 最適リスク水準の推奨 ──────────────────────────────────────────
best_sharpe_risk = max(RISK_LEVELS, key=lambda r: results[r]["sharpe"])
best_return_risk = max(RISK_LEVELS, key=lambda r: results[r]["ret"])
print(f"\n推奨:")
print(f"  Sharpe最大 → risk={best_sharpe_risk*100:.2f}%  (Sharpe={results[best_sharpe_risk]['sharpe']:.2f})")
print(f"  Return最大 → risk={best_return_risk*100:.2f}%  (Return={results[best_return_risk]['ret']:+.1f}%)")
print(f"  現在       → risk=1.00%  (Sharpe={results[0.010]['sharpe']:.2f}  Return={results[0.010]['ret']:+.1f}%)")
