"""
条件付き決算前決済 テスト

ルール:
  含み益 < HOLD_THRESHOLD  → 従来通り決算前に強制決済
  含み益 >= HOLD_THRESHOLD → 決算を通過してトレーリングストップのみで管理

対象: signal_backtest_20260303_022802_sb_trail4.5.csv
      pre_earnings かつ pl_pct >= 5.0% の 23件
"""

import csv
import math
import os
import sys
from pathlib import Path
from copy import deepcopy

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE      = Path(__file__).resolve().parent
CSV_IN    = BASE / "output/backtest/signal_backtest_20260303_022802_sb_trail4.5.csv"
CSV_NEW   = BASE / "output/backtest/signal_backtest_conditional_earn.csv"

HOLD_THRESHOLD = 5.0   # 含み益 >= 5% なら決算通過

# トレーリング設定（engine.pyと同じ）
ATR_MULT_DEFAULT = 3.5
PROGRESSIVE = [(6.0, 2.0), (4.0, 2.5), (2.0, 3.0)]
MAX_HOLD_DAYS = 60
SB_MAX_HOLD_DAYS = 90


# ── 元CSVを読み込み ────────────────────────────────────────────
rows = []
with open(CSV_IN, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        rows.append(row)

df_orig = pd.DataFrame(rows)
df_orig["pl_pct"]       = df_orig["pl_pct"].astype(float)
df_orig["entry_price"]  = df_orig["entry_price"].astype(float)
df_orig["sl_price"]     = df_orig["sl_price"].astype(float)
df_orig["exit_price"]   = df_orig["exit_price"].astype(float)
df_orig["entry_date"]   = pd.to_datetime(df_orig["entry_date"])
df_orig["exit_date"]    = pd.to_datetime(df_orig["exit_date"])

# 対象: pre_earnings かつ pl_pct >= HOLD_THRESHOLD
mask = (df_orig["result"] == "pre_earnings") & (df_orig["pl_pct"] >= HOLD_THRESHOLD)
targets = df_orig[mask].copy()
print(f"対象トレード: {len(targets)}件 (pre_earnings & PL >= {HOLD_THRESHOLD}%)")
print()


# ── 価格データ一括ダウンロード ─────────────────────────────────
tickers = targets["ticker"].unique().tolist()
print(f"価格データ取得: {len(tickers)}銘柄...")

all_prices = {}
for tkr in tickers:
    # entry_dateから最大120日後まで取得
    min_entry = targets[targets["ticker"] == tkr]["entry_date"].min()
    end_dt    = (min_entry + pd.Timedelta(days=150)).strftime("%Y-%m-%d")
    start_dt  = (min_entry - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    raw = yf.download(tkr, start=start_dt, end=end_dt, progress=False)
    if not raw.empty:
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        all_prices[tkr] = raw[["Open","High","Low","Close"]].dropna()

print(f"取得完了: {len(all_prices)}銘柄")
print()


# ── 各トレードを「保有継続」シミュレーション ──────────────────
def simulate_hold_through(row, price_df):
    """
    決算前決済日(exit_date)以降も保有し続けた場合の最終PL%を返す。
    ATR推定: (entry_price - sl_price) / 2.0  (SL = entry - ATR×2.0 より)
    """
    ticker       = row["ticker"]
    entry_price  = row["entry_price"]
    sl_price     = row["sl_price"]
    signal       = row["signal"]
    entry_date   = row["entry_date"]
    forced_exit  = row["exit_date"]   # 従来の決算前決済日

    # ATR推定
    atr = (entry_price - sl_price) / 2.0
    if atr <= 0:
        atr = entry_price * 0.03   # フォールバック: 3%

    # 強制決済日まで保有した時点での最高値を取得
    mask_to_exit = (price_df.index >= entry_date) & (price_df.index <= forced_exit)
    pre_df = price_df[mask_to_exit]
    if pre_df.empty:
        return None, None, None  # データなし
    max_price = pre_df["High"].max()

    # ブレイクイーブン後のSL（entry + ATR×1.0）を最低保証
    be_sl = entry_price + atr
    current_sl = max(be_sl, max_price - atr * ATR_MULT_DEFAULT)

    # 強制決済日以降の日次データで継続シミュレーション
    max_hold = SB_MAX_HOLD_DAYS if signal == "STRONG_BUY" else MAX_HOLD_DAYS
    deadline  = entry_date + pd.tseries.offsets.BDay(max_hold)

    post_mask = (price_df.index > forced_exit) & (price_df.index <= deadline)
    post_df   = price_df[post_mask]

    final_exit_price  = None
    final_exit_reason = None
    final_exit_date   = None

    for date, bar in post_df.iterrows():
        # 最高値更新
        if bar["High"] > max_price:
            max_price = bar["High"]

        # 含み益(ATR倍数)を計算 → Progressive trail
        gain_atr = (max_price - entry_price) / atr if atr > 0 else 0
        trail_mult = ATR_MULT_DEFAULT
        for gain_thresh, mult in PROGRESSIVE:
            if gain_atr >= gain_thresh:
                trail_mult = mult
                break

        # トレーリングSL更新
        new_trail = max_price - atr * trail_mult
        new_sl = max(current_sl, max(be_sl, new_trail))
        current_sl = new_sl

        # SLヒット判定（日中安値で確認）
        if bar["Low"] <= current_sl:
            final_exit_price  = current_sl
            final_exit_reason = "trailing_stop"
            final_exit_date   = date
            break

    # タイムアウト
    if final_exit_price is None:
        if not post_df.empty:
            final_exit_price  = post_df["Close"].iloc[-1]
            final_exit_reason = "timeout"
            final_exit_date   = post_df.index[-1]
        else:
            # 強制決済日に終値で決済（変化なし）
            final_exit_price  = row["exit_price"]
            final_exit_reason = "pre_earnings_hold_unchanged"
            final_exit_date   = forced_exit

    new_pl_pct = (final_exit_price - entry_price) / entry_price * 100
    return new_pl_pct, final_exit_reason, final_exit_date


# ── 各対象トレードをシミュレーション ──────────────────────────
print(f"{'Ticker':<6} {'Entry':>11} {'OldExit':>11} {'OldPL':>8} {'NewPL':>8} {'Diff':>7} {'NewReason'}")
print("-" * 75)

results = []
for idx, row in targets.iterrows():
    tkr = row["ticker"]
    if tkr not in all_prices:
        print(f"{tkr:<6} データなし")
        continue

    new_pl, new_reason, new_exit_date = simulate_hold_through(row, all_prices[tkr])
    if new_pl is None:
        print(f"{tkr:<6} シミュレーション不可")
        continue

    diff = new_pl - row["pl_pct"]
    results.append({
        "ticker":      tkr,
        "entry_date":  row["entry_date"],
        "old_exit":    row["exit_date"],
        "new_exit":    new_exit_date,
        "old_pl":      row["pl_pct"],
        "new_pl":      new_pl,
        "diff":        diff,
        "new_reason":  new_reason,
        "orig_idx":    idx,
    })
    print(f"{tkr:<6} {str(row['entry_date'].date()):>11} {str(row['exit_date'].date()):>11} "
          f"{row['pl_pct']:>+7.2f}% {new_pl:>+7.2f}% {diff:>+6.2f}% {new_reason}")

print("-" * 75)
old_total = sum(r["old_pl"] for r in results)
new_total = sum(r["new_pl"] for r in results)
print(f"{'合計':>40} {old_total:>+7.1f}% {new_total:>+7.1f}% {new_total-old_total:>+6.1f}%")
print()

improved = sum(1 for r in results if r["diff"] > 0)
worsened = sum(1 for r in results if r["diff"] < 0)
print(f"改善: {improved}件  悪化: {worsened}件  変化なし: {len(results)-improved-worsened}件")


# ── 修正版CSVを生成 ────────────────────────────────────────────
new_rows = deepcopy(rows)
for r in results:
    idx = r["orig_idx"]
    new_rows[idx]["pl_pct"]    = str(round(r["new_pl"], 4))
    new_rows[idx]["result"]    = r["new_reason"]
    new_rows[idx]["exit_date"] = r["new_exit"].strftime("%Y-%m-%d") if hasattr(r["new_exit"], "strftime") else str(r["new_exit"])

with open(CSV_NEW, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(new_rows)
print(f"\n修正版CSV保存: {CSV_NEW}")


# ── ポートフォリオシミュレーション（元 vs 修正） ──────────────
print("\n\nポートフォリオシミュレーション比較...")

import math as _math

def portfolio_sim(csv_path, initial=4000.0, risk=0.01, alloc=0.13, max_pos=11):
    trades = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            trades.append({
                "ticker":       row["ticker"],
                "signal":       row["signal"],
                "entry_date":   row["entry_date"],
                "exit_date":    row["exit_date"],
                "entry_price":  float(row["entry_price"]),
                "pl_pct":       float(row["pl_pct"]),
                "sl_price":     float(row["sl_price"]),
            })

    trades.sort(key=lambda t: t["entry_date"])
    same_day = {i for i, t in enumerate(trades) if t["entry_date"] == t["exit_date"]}
    events = []
    for i, t in enumerate(trades):
        events.append((t["entry_date"], "entry", i))
        events.append((t["exit_date"],  "exit",  i))

    def _key(e):
        d, et, i = e
        return (d, 2 if (et=="exit" and i in same_day) else (1 if et=="entry" else 0))
    events.sort(key=_key)

    cash = initial
    active = {}
    peak = initial
    max_dd = 0.0
    history = []
    proc_e, proc_x = set(), set()

    for date, etype, idx in events:
        t = trades[idx]
        if etype == "exit" and idx in proc_e and idx not in proc_x:
            proc_x.add(idx)
            alloc_amt = active.pop(idx)
            pnl = alloc_amt * (t["pl_pct"] / 100.0)
            cash += alloc_amt + pnl
            bal = cash + sum(active.values())
            if bal > peak: peak = bal
            dd = (peak - bal) / peak * 100 if peak > 0 else 0
            if dd > max_dd: max_dd = dd
            history.append({"exit_date": date, "balance": bal, "pnl": pnl})

        elif etype == "entry" and idx not in proc_e:
            if len(active) >= max_pos: continue
            eq = cash + sum(active.values())
            rr = (t["entry_price"] - t["sl_price"]) / t["entry_price"] if t["entry_price"] > 0 else 0.08
            rr = max(rr, 0.005)
            inv = (eq * risk) / rr
            ap = alloc + (0.01 if t["signal"] == "STRONG_BUY" else 0)
            amt = min(inv, eq * ap, cash)
            if t["entry_price"] > 0:
                shares = _math.floor(amt / t["entry_price"])
                amt = shares * t["entry_price"]
            else:
                shares = 0
            if amt <= 0 or shares == 0 or cash <= 0: continue
            cash -= amt
            active[idx] = amt
            proc_e.add(idx)

    final = cash + sum(active.values())
    total_ret = (final - initial) / initial * 100
    wins  = sum(1 for h in history if h["pnl"] > 0)
    total = len(history)
    gp    = sum(h["pnl"] for h in history if h["pnl"] > 0)
    gl    = abs(sum(h["pnl"] for h in history if h["pnl"] < 0))
    pf    = gp / gl if gl > 0 else 9.99

    return {
        "final": final, "ret": total_ret, "trades": total,
        "win_rate": wins/total*100 if total else 0,
        "pf": pf, "max_dd": max_dd, "history": history,
    }

orig_res = portfolio_sim(CSV_IN)
new_res  = portfolio_sim(CSV_NEW)

print()
print(f"{'指標':<16} {'元のルール':>14} {'条件付き決済':>14} {'差':>10}")
print("-" * 58)
for label, k, fmt in [
    ("最終残高($)",    "final",    lambda v: f"${v:,.2f}"),
    ("総リターン",     "ret",      lambda v: f"{v:+.2f}%"),
    ("Profit Factor",  "pf",       lambda v: f"{v:.3f}"),
    ("勝率",           "win_rate", lambda v: f"{v:.1f}%"),
    ("最大DD",         "max_dd",   lambda v: f"{v:.2f}%"),
    ("トレード数",     "trades",   lambda v: f"{v}"),
]:
    ov = orig_res[k]
    nv = new_res[k]
    diff = nv - ov
    print(f"{label:<16} {fmt(ov):>14} {fmt(nv):>14} {diff:>+9.2f}")


# ── 資産曲線チャート ───────────────────────────────────────────
def make_equity(history, initial, label):
    if not history:
        return pd.Series(dtype=float)
    s = pd.Series(
        [h["balance"] for h in history],
        index=pd.to_datetime([h["exit_date"] for h in history])
    )
    s = s[~s.index.duplicated(keep="last")].sort_index()
    date_range = pd.bdate_range(s.index[0], s.index[-1])
    s = s.reindex(date_range).ffill().fillna(initial)
    return s

eq_orig = make_equity(orig_res["history"], 4000, "Original")
eq_new  = make_equity(new_res["history"],  4000, "Conditional")

# SPY取得
spy_raw = yf.download("SPY", start="2021-01-04", end="2026-03-01", progress=False)
spy = spy_raw["Close"]["SPY"].dropna() if isinstance(spy_raw.columns, pd.MultiIndex) else spy_raw["Close"].dropna()
spy_norm = spy / spy.iloc[0] * 4000.0

fig, axes = plt.subplots(3, 1, figsize=(14, 15), facecolor="#0d1117")
fig.suptitle("Conditional Pre-Earnings Exit  vs  Original\n"
             "Hold through earnings if P/L >= +5%  (else exit as before)",
             color="white", fontsize=14, fontweight="bold")

def style_ax(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#adbac7")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", lw=0.7, linestyle="--")

COLORS = {"orig": "#f0a500", "new": "#00d4ff", "spy": "#7ee787"}

# Panel 1: 資産曲線
ax1 = axes[0]
style_ax(ax1)
ax1.plot(eq_orig.index, eq_orig, color=COLORS["orig"], lw=1.8,
         label=f"Original       ${orig_res['final']:,.0f} ({orig_res['ret']:+.1f}%)")
ax1.plot(eq_new.index,  eq_new,  color=COLORS["new"],  lw=2,
         label=f"Conditional    ${new_res['final']:,.0f} ({new_res['ret']:+.1f}%)")
ax1.plot(spy_norm.index, spy_norm, color=COLORS["spy"], lw=1.2, alpha=0.7,
         label="SPY (normalized)")
ax1.axhline(4000, color="#6e7681", lw=0.8, linestyle=":")
ax1.set_title("Portfolio Value ($4,000 start)", color="white", fontsize=12, fontweight="bold")
ax1.set_ylabel("Value ($)", color="#adbac7")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=10)

# Panel 2: 個別トレードのPL比較（散布図）
ax2 = axes[1]
style_ax(ax2)
old_pls = [r["old_pl"] for r in results]
new_pls = [r["new_pl"] for r in results]
colors2 = [COLORS["new"] if r["diff"] > 0 else "#ff6b6b" for r in results]
ax2.scatter(old_pls, new_pls, c=colors2, s=80, zorder=3, alpha=0.9)
max_val = max(max(old_pls), max(new_pls)) * 1.1
ax2.plot([0, max_val], [0, max_val], color="#6e7681", lw=1, linestyle="--", label="y=x (no change)")
for r in results:
    ax2.annotate(r["ticker"], (r["old_pl"], r["new_pl"]),
                 fontsize=7.5, color="#adbac7",
                 xytext=(3, 3), textcoords="offset points")
ax2.set_title("Individual Trade PL: Original vs Conditional", color="white", fontsize=12, fontweight="bold")
ax2.set_xlabel("Original PL (%)", color="#adbac7")
ax2.set_ylabel("Conditional PL (%)", color="#adbac7")
ax2.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=9)

# Panel 3: 改善・悪化の差分バー
ax3 = axes[2]
style_ax(ax3)
tickers_r = [r["ticker"] + "\n" + str(r["entry_date"].date().year) for r in results]
diffs     = [r["diff"] for r in results]
bar_cols  = [COLORS["new"] if d > 0 else "#ff6b6b" for d in diffs]
bars = ax3.bar(tickers_r, diffs, color=bar_cols, alpha=0.85)
ax3.axhline(0, color="#6e7681", lw=0.8)
for bar, val in zip(bars, diffs):
    ax3.text(bar.get_x() + bar.get_width()/2,
             val + (0.3 if val >= 0 else -0.8),
             f"{val:+.1f}%", ha="center", va="bottom" if val >= 0 else "top",
             color="white", fontsize=7.5)
ax3.set_title("PL Change per Trade (Conditional - Original)", color="white", fontsize=12, fontweight="bold")
ax3.set_ylabel("PL Diff (%)", color="#adbac7")
ax3.tick_params(axis="x", labelsize=7, colors="#adbac7")

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = BASE / "output/backtest/conditional_earnings_chart.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"\nChart saved: {out_path}")
