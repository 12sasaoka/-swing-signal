"""
WFT 比較チャート: オリジナル vs 条件付き決算前決済
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import yfinance as yf
import csv, math
from pathlib import Path

BASE    = Path(__file__).resolve().parent.parent.parent
INITIAL = 4000.0

# ── 両WFTの結果を直接埋め込み ──────────────────────────────────
# (walkforward.py を2回実行した結果をここで比較)

ORIG_FOLDS = [
    {"fold":"Fold1","test":"2022","train_pf":1.681,"test_pf":0.918,"test_ret":-0.6, "chosen":"trail4.0"},
    {"fold":"Fold2","test":"2023","train_pf":1.579,"test_pf":1.068,"test_ret":+2.6, "chosen":"trail4.0"},
    {"fold":"Fold3","test":"2024","train_pf":1.365,"test_pf":1.573,"test_ret":+32.0,"chosen":"trail4.0"},
    {"fold":"Fold4","test":"2025","train_pf":1.519,"test_pf":1.605,"test_ret":+15.8,"chosen":"trail4.0"},
    {"fold":"Fold5","test":"2026","train_pf":1.547,"test_pf":2.400,"test_ret":+9.8, "chosen":"trail4.0"},
]

COND_FOLDS = [
    {"fold":"Fold1","test":"2022","train_pf":1.690,"test_pf":0.918,"test_ret":-0.6, "chosen":"conditional"},
    {"fold":"Fold2","test":"2023","train_pf":1.587,"test_pf":1.062,"test_ret":+2.4, "chosen":"conditional"},
    {"fold":"Fold3","test":"2024","train_pf":1.365,"test_pf":1.573,"test_ret":+32.0,"chosen":"trail4.0"},
    {"fold":"Fold4","test":"2025","train_pf":1.519,"test_pf":1.605,"test_ret":+15.8,"chosen":"trail4.0"},
    {"fold":"Fold5","test":"2026","train_pf":1.547,"test_pf":2.400,"test_ret":+9.8, "chosen":"trail4.0"},
]

ORIG_OOS = 71.2
COND_OOS = 70.8

# ── ポートフォリオシミュレーション（OOS資産曲線用） ─────────────
def portfolio_sim(csv_path, initial=4000.0, risk=0.005, alloc=0.10, max_pos=11):
    trades = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            trades.append({
                "signal":      row["signal"],
                "entry_date":  row["entry_date"],
                "exit_date":   row["exit_date"],
                "entry_price": float(row["entry_price"]),
                "pl_pct":      float(row["pl_pct"]),
                "sl_price":    float(row["sl_price"]),
            })
    trades.sort(key=lambda t: t["entry_date"])
    same_day = {i for i,t in enumerate(trades) if t["entry_date"] == t["exit_date"]}
    events = []
    for i,t in enumerate(trades):
        events.append((t["entry_date"],"entry",i))
        events.append((t["exit_date"],"exit",i))
    def _key(e):
        d,et,i = e
        return (d, 2 if (et=="exit" and i in same_day) else (1 if et=="entry" else 0))
    events.sort(key=_key)
    cash = initial; active = {}; peak = initial; max_dd = 0.0; history = []
    pe, px = set(), set()
    for date,etype,idx in events:
        t = trades[idx]
        if etype=="exit" and idx in pe and idx not in px:
            px.add(idx)
            amt = active.pop(idx)
            pnl = amt*(t["pl_pct"]/100)
            cash += amt+pnl
            bal = cash+sum(active.values())
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
    return {"final":final,"ret":(final-initial)/initial*100,"history":history,"max_dd":max_dd}

# WFTと同じパラメータ（risk=0.005, alloc=0.1）でOOS期間（2022〜）のみシム
CSV_45   = BASE/"output/backtest/signal_backtest_20260303_022802_sb_trail4.5.csv"
CSV_COND = BASE/"output/backtest/signal_backtest_conditional_earn.csv"

# OOS期間のトレードだけ抽出して渡すため一時ファイルなしで直接フィルタ
def portfolio_sim_filtered(csv_path, year_from, initial=4000.0,
                            risk=0.005, alloc=0.10, max_pos=11):
    trades = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if int(row["entry_date"][:4]) >= year_from:
                trades.append({
                    "signal":      row["signal"],
                    "entry_date":  row["entry_date"],
                    "exit_date":   row["exit_date"],
                    "entry_price": float(row["entry_price"]),
                    "pl_pct":      float(row["pl_pct"]),
                    "sl_price":    float(row["sl_price"]),
                })
    trades.sort(key=lambda t: t["entry_date"])
    same_day = {i for i,t in enumerate(trades) if t["entry_date"] == t["exit_date"]}
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
            history.append({"exit_date":date,"balance":bal})
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
    return {"final":final,"ret":(final-initial)/initial*100,"history":history,"max_dd":max_dd}

print("シミュレーション中...")
res_orig = portfolio_sim_filtered(CSV_45,   2022)
res_cond = portfolio_sim_filtered(CSV_COND, 2022)

def make_eq(history, initial):
    if not history: return pd.Series(dtype=float)
    s = pd.Series([h["balance"] for h in history],
                  index=pd.to_datetime([h["exit_date"] for h in history]))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    dr = pd.bdate_range(s.index[0], s.index[-1])
    return s.reindex(dr).ffill().fillna(initial)

eq_orig = make_eq(res_orig["history"], INITIAL)
eq_cond = make_eq(res_cond["history"], INITIAL)

# SPY
raw = yf.download("SPY", start="2022-01-03", end="2026-03-01", progress=False)
spy = (raw["Close"]["SPY"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]).dropna()
spy_norm = spy / spy.iloc[0] * INITIAL
spy_ret  = (spy.iloc[-1]-spy.iloc[0])/spy.iloc[0]*100

# ── 描画 ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 18), facecolor="#0d1117")
fig.suptitle("WFT Comparison: Original  vs  Conditional Pre-Earnings Exit\n"
             "OOS Period: 2022-2026  |  Hold through earnings if in-profit >= +5%",
             color="white", fontsize=14, fontweight="bold", y=0.98)

gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.32,
                      top=0.93, bottom=0.05, left=0.08, right=0.96)

COLORS = {"orig":"#f0a500","cond":"#00d4ff","spy":"#7ee787","pos":"#00d4ff","neg":"#ff6b6b"}

def style_ax(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#adbac7")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", lw=0.7, linestyle="--")

fold_names = [f["fold"] for f in ORIG_FOLDS]
yr_labels  = [f["test"] for f in ORIG_FOLDS]
x = np.arange(len(fold_names))

# ── Panel 1: OOS 資産曲線 ────────────────────────────────────────
ax1 = fig.add_subplot(gs[0,:])
style_ax(ax1)
final_o = eq_orig.dropna().iloc[-1]; final_c = eq_cond.dropna().iloc[-1]
ax1.plot(eq_orig.index, eq_orig, color=COLORS["orig"], lw=2,
         label=f"Original      ${final_o:,.0f} ({res_orig['ret']:+.1f}%)")
ax1.plot(eq_cond.index, eq_cond, color=COLORS["cond"], lw=2, linestyle="--",
         label=f"Conditional   ${final_c:,.0f} ({res_cond['ret']:+.1f}%)")
ax1.plot(spy_norm.index, spy_norm, color=COLORS["spy"], lw=1.2, alpha=0.6,
         label=f"SPY           ${spy_norm.iloc[-1]:,.0f} ({spy_ret:+.1f}%)")
ax1.axhline(INITIAL, color="#6e7681", lw=0.8, linestyle=":")
for yr in ["2022","2023","2024","2025","2026"]:
    ax1.axvline(pd.Timestamp(f"{yr}-01-01"), color="#6e7681", lw=0.6, linestyle="--", alpha=0.5)
ax1.set_title("OOS Portfolio Value  ($4,000 start, fixed params: risk=0.5%)", color="white",
              fontsize=12, fontweight="bold")
ax1.set_ylabel("Value ($)", color="#adbac7")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=10)

# ── Panel 2: Test PF 比較 ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[1,0])
style_ax(ax2)
orig_pfs = [f["test_pf"] for f in ORIG_FOLDS]
cond_pfs = [f["test_pf"] for f in COND_FOLDS]
w = 0.35
ax2.bar(x-w/2, orig_pfs, w, label="Original",    color=COLORS["orig"], alpha=0.85)
ax2.bar(x+w/2, cond_pfs, w, label="Conditional", color=COLORS["cond"], alpha=0.85)
ax2.axhline(1.0, color="#f0a500", lw=1.2, linestyle="--", alpha=0.7, label="PF=1.0")
ax2.set_xticks(x); ax2.set_xticklabels(yr_labels, color="#adbac7")
ax2.set_title("Test Profit Factor per Fold", color="white", fontsize=12, fontweight="bold")
ax2.set_ylabel("Profit Factor", color="#adbac7")
ax2.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=9)

# ── Panel 3: Test Return 比較 ─────────────────────────────────────
ax3 = fig.add_subplot(gs[1,1])
style_ax(ax3)
orig_rets = [f["test_ret"] for f in ORIG_FOLDS]
cond_rets = [f["test_ret"] for f in COND_FOLDS]
ax3.bar(x-w/2, orig_rets, w, label="Original",    color=COLORS["orig"], alpha=0.85)
ax3.bar(x+w/2, cond_rets, w, label="Conditional", color=COLORS["cond"], alpha=0.85)
ax3.axhline(0, color="#6e7681", lw=0.8)
ax3.set_xticks(x); ax3.set_xticklabels(yr_labels, color="#adbac7")
ax3.set_title("Test Return (%) per Fold", color="white", fontsize=12, fontweight="bold")
ax3.set_ylabel("Return (%)", color="#adbac7")
ax3.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=9)

# ── Panel 4: どのCSVが選ばれたか ──────────────────────────────────
ax4 = fig.add_subplot(gs[2,0])
style_ax(ax4)
ax4.axis("off")

chosen_map = {"trail4.0":"trail4.0","trail4.5":"trail4.5","conditional":"conditional"}
col_labels = ["Fold","Test Year","Original chosen","Conditional chosen","Test Ret Diff"]
rows_tbl = []
for o,c in zip(ORIG_FOLDS, COND_FOLDS):
    diff = c["test_ret"] - o["test_ret"]
    diff_str = f"{diff:+.1f}%" if diff != 0 else "same"
    rows_tbl.append([o["fold"], o["test"], o["chosen"], c["chosen"], diff_str])

tbl = ax4.table(cellText=rows_tbl, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.0, 2.0)
for (r,c), cell in tbl.get_celld().items():
    cell.set_facecolor("#21262d"); cell.set_edgecolor("#30363d")
    cell.set_text_props(color="white")
    if r==0:
        cell.set_facecolor("#1f6feb")
        cell.set_text_props(color="white", fontweight="bold")
    # conditionalが選ばれたセルをハイライト
    if r>0 and c==3 and rows_tbl[r-1][3]=="conditional":
        cell.set_facecolor("#0d4429")
        cell.set_text_props(color=COLORS["cond"], fontweight="bold")
ax4.set_title("Parameter Selection per Fold", color="white", fontsize=12, fontweight="bold", pad=10)

# ── Panel 5: サマリーテーブル ──────────────────────────────────────
ax5 = fig.add_subplot(gs[2,1])
style_ax(ax5)
ax5.axis("off")

def fmt(v): return f"+{v:.1f}%" if v>=0 else f"{v:.1f}%"

# Max DD計算
def calc_dd(eq):
    pk = eq.cummax(); return ((eq-pk)/pk*100).min()

dd_o = calc_dd(eq_orig.dropna())
dd_c = calc_dd(eq_cond.dropna())
dd_s = calc_dd(spy_norm.dropna())

summary_rows = [
    ["OOS Return",     fmt(res_orig["ret"]),   fmt(res_cond["ret"]),   fmt(spy_ret)],
    ["Final Value",    f"${final_o:,.0f}",      f"${final_c:,.0f}",      f"${spy_norm.iloc[-1]:,.0f}"],
    ["Max Drawdown",   f"{dd_o:.1f}%",          f"{dd_c:.1f}%",          f"{dd_s:.1f}%"],
    ["Avg Test PF",    f"{np.mean(orig_pfs):.3f}", f"{np.mean(cond_pfs):.3f}", "-"],
    ["Folds Profit",   f"{sum(1 for r in orig_rets if r>0)}/5",
                       f"{sum(1 for r in cond_rets if r>0)}/5", "-"],
    ["Conditional",    "chosen 0/5",            "chosen 2/5",            "-"],
]
col_labels2 = ["Metric","Original","Conditional","SPY"]
tbl2 = ax5.table(cellText=summary_rows, colLabels=col_labels2, loc="center", cellLoc="center")
tbl2.auto_set_font_size(False); tbl2.set_fontsize(9.5); tbl2.scale(1.0, 2.0)
for (r,c), cell in tbl2.get_celld().items():
    cell.set_facecolor("#21262d"); cell.set_edgecolor("#30363d")
    cell.set_text_props(color="white")
    if r==0:
        cell.set_facecolor("#1f6feb")
        cell.set_text_props(color="white", fontweight="bold")
    elif c==1: cell.set_text_props(color=COLORS["orig"])
    elif c==2: cell.set_text_props(color=COLORS["cond"])
    elif c==3: cell.set_text_props(color=COLORS["spy"])
ax5.set_title("OOS Summary", color="white", fontsize=12, fontweight="bold", pad=10)

out = BASE/"output/backtest/wft_comparison_chart.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved: {out}")
print(f"\nOriginal OOS:    {res_orig['ret']:+.1f}%  MaxDD={dd_o:.1f}%")
print(f"Conditional OOS: {res_cond['ret']:+.1f}%  MaxDD={dd_c:.1f}%")
print(f"差:              {res_cond['ret']-res_orig['ret']:+.1f}%")
