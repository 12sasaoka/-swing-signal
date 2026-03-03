"""
Walk-Forward Test
アンカー型展開ウィンドウ（5フォールド）

パラメータグリッドサーチ:
  - trail_stop    : 4.0% / 4.5%  (既存CSVを流用)
  - RISK_PER_TRADE: 0.005 / 0.008 / 0.010 / 0.013 / 0.015
  - MAX_ALLOC_PCT : 0.10 / 0.12 / 0.13 / 0.16 / 0.20
  - MAX_POSITIONS : 10 / 11 / 12

評価指標: Profit Factor (訓練期間で最大化)
"""

import csv
import math
import os
import sys
import itertools
from dataclasses import dataclass
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

# ── パス ────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
CSV_40   = BASE / "output/backtest/signal_backtest_20260303_022802_sb_trail4.0.csv"
CSV_45   = BASE / "output/backtest/signal_backtest_20260303_022802_sb_trail4.5.csv"
CSV_COND = BASE / "output/backtest/signal_backtest_conditional_earn.csv"
INITIAL = 4000.0

# ── フォールド定義 ───────────────────────────────────────────────
FOLDS = [
    {"name": "Fold1", "train": (2021, 2021), "test": (2022, 2022)},
    {"name": "Fold2", "train": (2021, 2022), "test": (2023, 2023)},
    {"name": "Fold3", "train": (2021, 2023), "test": (2024, 2024)},
    {"name": "Fold4", "train": (2021, 2024), "test": (2025, 2025)},
    {"name": "Fold5", "train": (2021, 2025), "test": (2026, 2026)},
]

# ── パラメータグリッド ───────────────────────────────────────────
PARAM_GRID = list(itertools.product(
    [("trail4.0", CSV_40), ("trail4.5", CSV_45), ("conditional", CSV_COND)],  # trail stop
    [0.005, 0.008, 0.010, 0.013, 0.015],             # RISK_PER_TRADE
    [0.10, 0.12, 0.13, 0.16, 0.20],                  # MAX_ALLOC_PCT
    [10, 11, 12],                                     # MAX_POSITIONS
))
print(f"グリッドサイズ: {len(PARAM_GRID)} 通り × {len(FOLDS)} フォールド")


# ── トレードデータ読み込み ────────────────────────────────────────
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

_trade_cache: dict = {}

def load_trades(csv_path: Path) -> list[Trade]:
    key = str(csv_path)
    if key in _trade_cache:
        return _trade_cache[key]
    trades = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
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
            ))
    _trade_cache[key] = trades
    return trades


def filter_trades(trades: list[Trade], year_from: int, year_to: int) -> list[Trade]:
    return [t for t in trades if year_from <= int(t.entry_date[:4]) <= year_to]


# ── ポートフォリオシミュレーション ────────────────────────────────
def simulate(
    trades: list[Trade],
    initial: float,
    risk_per_trade: float,
    max_alloc_pct: float,
    max_positions: int,
    sb_alloc_extra: float = 0.01,
) -> dict:
    """シミュレーション実行。結果dictを返す。"""
    if not trades:
        return {"pf": 0.0, "total_return": 0.0, "sharpe": 0.0, "max_dd": 0.0,
                "trades": 0, "win_rate": 0.0, "history": [], "final": initial}

    trades = sorted(trades, key=lambda t: t.entry_date)

    same_day = {i for i, t in enumerate(trades) if t.entry_date == t.exit_date}
    events: list[tuple] = []
    for i, t in enumerate(trades):
        events.append((t.entry_date, "entry", i))
        events.append((t.exit_date, "exit", i))

    def _key(e):
        date, etype, idx = e
        if etype == "exit":
            priority = 2 if idx in same_day else 0
        else:
            priority = 1
        return (date, priority)

    events.sort(key=_key)

    cash = initial
    active: dict[int, float] = {}
    peak = initial
    max_dd_pct = 0.0
    history = []
    processed_entries: set[int] = set()
    processed_exits: set[int] = set()

    for date, etype, idx in events:
        t = trades[idx]

        if etype == "exit" and idx in processed_entries and idx not in processed_exits:
            processed_exits.add(idx)
            alloc = active.pop(idx)
            pnl = alloc * (t.pl_pct / 100.0)
            cash += alloc + pnl
            balance = cash + sum(active.values())
            if balance > peak:
                peak = balance
            dd_pct = (peak - balance) / peak * 100 if peak > 0 else 0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
            history.append({
                "exit_date": t.exit_date,
                "pnl": pnl,
                "balance": balance,
            })

        elif etype == "entry" and idx not in processed_entries:
            if len(active) >= max_positions:
                continue
            total_eq = cash + sum(active.values())
            risk_amount = total_eq * risk_per_trade
            risk_ratio = (t.entry_price - t.sl_price) / t.entry_price if t.entry_price > 0 else 0.08
            risk_ratio = max(risk_ratio, 0.005)
            investment = risk_amount / risk_ratio
            _ap = max_alloc_pct + (sb_alloc_extra if t.signal == "STRONG_BUY" else 0.0)
            alloc = min(investment, total_eq * _ap, cash)
            if t.entry_price > 0:
                shares = math.floor(alloc / t.entry_price)
                alloc = shares * t.entry_price
            else:
                shares = 0
            if alloc <= 0 or cash <= 0 or shares == 0:
                continue
            cash -= alloc
            active[idx] = alloc
            processed_entries.add(idx)

    final = cash + sum(active.values())
    total_return = (final - initial) / initial * 100

    wins   = sum(1 for h in history if h["pnl"] > 0)
    losses = sum(1 for h in history if h["pnl"] < 0)
    gross_profit = sum(h["pnl"] for h in history if h["pnl"] > 0)
    gross_loss   = abs(sum(h["pnl"] for h in history if h["pnl"] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else (9.99 if gross_profit > 0 else 0.0)

    # Sharpe（日次リターンベース）
    sharpe = 0.0
    if len(history) >= 5:
        df_h = pd.Series(
            [h["balance"] for h in history],
            index=pd.to_datetime([h["exit_date"] for h in history])
        ).resample("B").last().ffill()
        r = df_h.pct_change().dropna()
        if r.std() > 0:
            sharpe = (r.mean() / r.std()) * np.sqrt(252)

    return {
        "pf": pf,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_dd": max_dd_pct,
        "trades": len(history),
        "win_rate": wins / len(history) * 100 if history else 0.0,
        "history": history,
        "final": final,
    }


# ── メインWFTループ ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Walk-Forward Test  (anchored expanding window)")
print("=" * 70)

fold_results = []

for fold in FOLDS:
    name  = fold["name"]
    t_from, t_to = fold["train"]
    v_from, v_to = fold["test"]

    print(f"\n[{name}] Train: {t_from}-{t_to}  /  Test: {v_from}-{v_to}")
    print(f"  グリッドサーチ中... ", end="", flush=True)

    best_pf     = -1.0
    best_params = None
    best_train  = None

    for (trail_label, csv_path), risk, alloc, npos in PARAM_GRID:
        all_trades = load_trades(csv_path)
        train_trades = filter_trades(all_trades, t_from, t_to)
        if len(train_trades) < 10:
            continue
        res = simulate(train_trades, INITIAL, risk, alloc, npos)
        if res["pf"] > best_pf and res["trades"] >= 5:
            best_pf = res["pf"]
            best_params = {
                "trail": trail_label, "csv": csv_path,
                "risk": risk, "alloc": alloc, "npos": npos,
            }
            best_train = res

    if best_params is None:
        print("SKIP (データ不足)")
        continue

    # テスト期間に最良パラメータを適用
    all_trades = load_trades(best_params["csv"])
    test_trades = filter_trades(all_trades, v_from, v_to)
    test_res = simulate(test_trades, INITIAL, best_params["risk"],
                        best_params["alloc"], best_params["npos"])

    fold_results.append({
        "fold": name,
        "train": f"{t_from}-{t_to}",
        "test":  f"{v_from}-{v_to}",
        "best_params": best_params,
        "train_pf":    best_train["pf"],
        "train_ret":   best_train["total_return"],
        "test_pf":     test_res["pf"],
        "test_ret":    test_res["total_return"],
        "test_sharpe": test_res["sharpe"],
        "test_maxdd":  test_res["max_dd"],
        "test_wr":     test_res["win_rate"],
        "test_trades": test_res["trades"],
        "test_history": test_res["history"],
    })

    print(f"完了")
    print(f"  最良パラメータ : {best_params['trail']}  risk={best_params['risk']}  "
          f"alloc={best_params['alloc']}  pos={best_params['npos']}")
    print(f"  Train PF={best_train['pf']:.3f}  Return={best_train['total_return']:+.1f}%")
    print(f"  Test  PF={test_res['pf']:.3f}  Return={test_res['total_return']:+.1f}%  "
          f"Sharpe={test_res['sharpe']:.2f}  MaxDD={test_res['max_dd']:.1f}%  "
          f"WinRate={test_res['win_rate']:.1f}%  Trades={test_res['trades']}")


# ── OOS 連結資産曲線 ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  OOS 連結結果")
print("=" * 70)

# 各フォールドのhistoryを1本につなぐ（残高を引き継ぐ）
oos_history = []
running_balance = INITIAL

for fr in fold_results:
    hist = fr["test_history"]
    if not hist:
        continue
    # このフォールドの初期値からのスケール係数
    fold_init = INITIAL
    fold_final_raw = hist[-1]["balance"]
    scale = running_balance / fold_init
    for h in hist:
        oos_history.append({
            "exit_date": h["exit_date"],
            "balance": h["balance"] * scale,
        })
    running_balance = fold_final_raw * scale

oos_df = pd.DataFrame(oos_history)
oos_df["exit_date"] = pd.to_datetime(oos_df["exit_date"])
oos_df = oos_df.sort_values("exit_date")

print(f"\n  OOS 全期間:")
print(f"    初期資産   : ${INITIAL:,.2f}")
print(f"    最終資産   : ${running_balance:,.2f}")
oos_ret = (running_balance - INITIAL) / INITIAL * 100
print(f"    OOSリターン: {oos_ret:+.1f}%")

# フォールド別サマリー
print(f"\n  {'Fold':<7} {'Train':>10} {'Test':>10} {'TrainPF':>9} {'TestPF':>9} "
      f"{'TestRet':>9} {'Sharpe':>8} {'MaxDD':>8} {'WR':>7} {'Trades':>7}")
print("  " + "-" * 80)
for fr in fold_results:
    print(f"  {fr['fold']:<7} {fr['train']:>10} {fr['test']:>10} "
          f"{fr['train_pf']:>9.3f} {fr['test_pf']:>9.3f} "
          f"{fr['test_ret']:>+8.1f}% {fr['test_sharpe']:>8.2f} "
          f"{fr['test_maxdd']:>7.1f}% {fr['test_wr']:>6.1f}% {fr['test_trades']:>7}")
print("  " + "-" + "-" * 79)
avg_test_pf = np.mean([fr["test_pf"] for fr in fold_results])
print(f"  {'Avg':<7} {'':>10} {'':>10} {'':>9} {avg_test_pf:>9.3f}")


# ── SPY / QQQ ダウンロード ────────────────────────────────────────
START_DATE = "2022-01-03"
END_DATE   = "2026-03-01"
raw = yf.download(["SPY", "QQQ"], start=START_DATE, end=END_DATE, progress=False)
spy = raw["Close"]["SPY"].dropna()
qqq = raw["Close"]["QQQ"].dropna()

spy_norm = spy / spy.iloc[0] * INITIAL
qqq_norm = qqq / qqq.iloc[0] * INITIAL

spy_ret = (spy.iloc[-1] - spy.iloc[0]) / spy.iloc[0] * 100
qqq_ret = (qqq.iloc[-1] - qqq.iloc[0]) / qqq.iloc[0] * 100
print(f"\n  比較 (OOS期間 2022-2026):")
print(f"    OOS Strategy: {oos_ret:+.1f}%  /  SPY: {spy_ret:+.1f}%  /  QQQ: {qqq_ret:+.1f}%")


# ── チャート描画 ──────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 20), facecolor="#0d1117")
fig.suptitle("Walk-Forward Test  (OOS: 2022-2026)\n"
             "Anchored Expanding Window  |  Grid Search: trail / risk / alloc / positions",
             color="white", fontsize=14, fontweight="bold", y=0.98)

gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.35,
                      top=0.93, bottom=0.04, left=0.08, right=0.97)

COLORS = {"OOS": "#00d4ff", "IS": "#7ee787", "SPY": "#f0a500", "QQQ": "#ff6b6b"}

def style_ax(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#adbac7")
    ax.xaxis.label.set_color("#adbac7")
    ax.yaxis.label.set_color("#adbac7")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", linewidth=0.7, linestyle="--")


# ── Panel 1: OOS資産曲線 vs SPY vs QQQ ─────────────────────────
ax1 = fig.add_subplot(gs[0, :])
style_ax(ax1)

date_range_oos = pd.bdate_range(oos_df["exit_date"].min(), oos_df["exit_date"].max())
oos_series = oos_df.set_index("exit_date")["balance"]
oos_series = oos_series[~oos_series.index.duplicated(keep="last")]
oos_series = oos_series.reindex(date_range_oos).ffill().fillna(INITIAL)

spy_a = spy_norm.reindex(date_range_oos).ffill()
qqq_a = qqq_norm.reindex(date_range_oos).ffill()

final_oos = oos_series.dropna().iloc[-1]
final_spy = spy_a.dropna().iloc[-1]
final_qqq = qqq_a.dropna().iloc[-1]

ax1.plot(oos_series.index, oos_series,
         color=COLORS["OOS"], lw=2, label=f"OOS Strategy  ${final_oos:,.0f} ({oos_ret:+.1f}%)", zorder=3)
ax1.plot(spy_a.index, spy_a,
         color=COLORS["SPY"], lw=1.5, label=f"SPY  ${final_spy:,.0f} ({spy_ret:+.1f}%)")
ax1.plot(qqq_a.index, qqq_a,
         color=COLORS["QQQ"], lw=1.5, label=f"QQQ  ${final_qqq:,.0f} ({qqq_ret:+.1f}%)")
ax1.axhline(INITIAL, color="#6e7681", lw=0.8, linestyle=":")

# フォールド境界線
fold_test_starts = ["2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01", "2026-01-01"]
for fs in fold_test_starts:
    ax1.axvline(pd.Timestamp(fs), color="#6e7681", lw=0.7, linestyle="--", alpha=0.6)

ax1.set_title("OOS Portfolio Value  ($4,000 start)", color="white", fontsize=12, fontweight="bold")
ax1.set_ylabel("Value ($)", color="#adbac7")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.legend(loc="upper left", facecolor="#21262d", edgecolor="#30363d",
           labelcolor="white", fontsize=10)


# ── Panel 2: 累積リターン ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
style_ax(ax2)

oos_pct = (oos_series / INITIAL - 1) * 100
spy_pct = (spy_a / INITIAL - 1) * 100
qqq_pct = (qqq_a / INITIAL - 1) * 100

ax2.plot(oos_series.index, oos_pct,
         color=COLORS["OOS"], lw=2, label="OOS Strategy", zorder=3)
ax2.plot(spy_a.index, spy_pct, color=COLORS["SPY"], lw=1.5, label="SPY")
ax2.plot(qqq_a.index, qqq_pct, color=COLORS["QQQ"], lw=1.5, label="QQQ")
ax2.axhline(0, color="#6e7681", lw=0.8, linestyle=":")
ax2.fill_between(oos_series.index, oos_pct, 0,
                 where=(oos_pct > 0), alpha=0.1, color=COLORS["OOS"])
ax2.fill_between(oos_series.index, oos_pct, 0,
                 where=(oos_pct < 0), alpha=0.2, color="red")
for fs in fold_test_starts:
    ax2.axvline(pd.Timestamp(fs), color="#6e7681", lw=0.7, linestyle="--", alpha=0.6)
ax2.set_title("Cumulative Return OOS (%)", color="white", fontsize=12, fontweight="bold")
ax2.set_ylabel("Return (%)", color="#adbac7")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.legend(loc="upper left", facecolor="#21262d", edgecolor="#30363d",
           labelcolor="white", fontsize=10)


# ── Panel 3: フォールド別 Train vs Test PF ─────────────────────
ax3 = fig.add_subplot(gs[2, 0])
style_ax(ax3)

fold_names  = [fr["fold"] for fr in fold_results]
train_pfs   = [fr["train_pf"] for fr in fold_results]
test_pfs    = [fr["test_pf"]  for fr in fold_results]
x = np.arange(len(fold_names))
w = 0.35

ax3.bar(x - w/2, train_pfs, w, label="Train PF", color=COLORS["IS"],   alpha=0.85)
ax3.bar(x + w/2, test_pfs,  w, label="Test PF",  color=COLORS["OOS"],  alpha=0.85)
ax3.axhline(1.0, color="#f0a500", lw=1.2, linestyle="--", label="PF=1.0 (BEP)")
ax3.set_xticks(x)
ax3.set_xticklabels(fold_names, color="#adbac7")
ax3.set_title("Train vs Test  Profit Factor", color="white", fontsize=12, fontweight="bold")
ax3.set_ylabel("Profit Factor", color="#adbac7")
ax3.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=9)


# ── Panel 4: フォールド別テストリターン ─────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
style_ax(ax4)

test_rets = [fr["test_ret"] for fr in fold_results]
cols = [COLORS["OOS"] if r >= 0 else "#ff6b6b" for r in test_rets]
bars = ax4.bar(fold_names, test_rets, color=cols, alpha=0.85)
ax4.axhline(0, color="#6e7681", lw=0.8)
for bar, val in zip(bars, test_rets):
    ax4.text(bar.get_x() + bar.get_width()/2, val + (1 if val >= 0 else -2.5),
             f"{val:+.1f}%", ha="center", va="bottom", color="white", fontsize=9)
ax4.set_title("Test Period Return (%) per Fold", color="white", fontsize=12, fontweight="bold")
ax4.set_ylabel("Return (%)", color="#adbac7")
ax4.tick_params(colors="#adbac7")


# ── Panel 5: 最良パラメータの推移 ──────────────────────────────
ax5 = fig.add_subplot(gs[3, 0])
style_ax(ax5)

risks  = [fr["best_params"]["risk"]  for fr in fold_results]
allocs = [fr["best_params"]["alloc"] for fr in fold_results]
npos   = [fr["best_params"]["npos"]  for fr in fold_results]
trails = [fr["best_params"]["trail"] for fr in fold_results]
x = np.arange(len(fold_names))

ax5_r = ax5.twinx()
ax5.bar(x - 0.2, [r * 100 for r in risks],  0.4, label="Risk/Trade (%)", color="#7ee787", alpha=0.7)
ax5_r.plot(x, [a * 100 for a in allocs], "o--", color="#f0a500", lw=1.5, label="Max Alloc (%)")
ax5.set_xticks(x)
ax5.set_xticklabels(fold_names, color="#adbac7")
ax5.set_title("Optimal Parameters per Fold", color="white", fontsize=12, fontweight="bold")
ax5.set_ylabel("Risk/Trade (%)", color="#7ee787")
ax5_r.set_ylabel("Max Alloc (%)", color="#f0a500")
ax5_r.tick_params(colors="#f0a500")
ax5.tick_params(colors="#adbac7")
for i, (t, p) in enumerate(zip(trails, npos)):
    ax5.text(i, -0.15, f"{t}\npos={p}", ha="center", va="top",
             color="#adbac7", fontsize=7, transform=ax5.get_xaxis_transform())
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5_r.get_legend_handles_labels()
ax5.legend(lines1+lines2, labels1+labels2, facecolor="#21262d",
           edgecolor="#30363d", labelcolor="white", fontsize=8, loc="upper left")


# ── Panel 6: 統計サマリーテーブル ─────────────────────────────
ax6 = fig.add_subplot(gs[3, 1])
style_ax(ax6)
ax6.axis("off")

def fmt(v):
    return f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%"

# OOS Sharpe・MaxDD計算
if len(oos_series.dropna()) > 5:
    r_oos = oos_series.dropna().pct_change().dropna()
    sh_oos = (r_oos.mean() / r_oos.std()) * np.sqrt(252) if r_oos.std() > 0 else 0
    peak_oos = oos_series.cummax()
    dd_oos = ((oos_series - peak_oos) / peak_oos * 100).min()
else:
    sh_oos = 0.0; dd_oos = 0.0

r_spy = spy_a.dropna().pct_change().dropna()
sh_spy = (r_spy.mean() / r_spy.std()) * np.sqrt(252) if r_spy.std() > 0 else 0
peak_spy = spy_a.cummax()
dd_spy = ((spy_a - peak_spy) / peak_spy * 100).min()

r_qqq = qqq_a.dropna().pct_change().dropna()
sh_qqq = (r_qqq.mean() / r_qqq.std()) * np.sqrt(252) if r_qqq.std() > 0 else 0
peak_qqq = qqq_a.cummax()
dd_qqq = ((qqq_a - peak_qqq) / peak_qqq * 100).min()

col_labels = ["Metric", "OOS Strat", "SPY", "QQQ"]
rows = [
    ["Period",       "2022-2026", "2022-2026", "2022-2026"],
    ["Total Return", fmt(oos_ret),  fmt(spy_ret),  fmt(qqq_ret)],
    ["Final Value",  f"${final_oos:,.0f}", f"${final_spy:,.0f}", f"${final_qqq:,.0f}"],
    ["Max Drawdown", f"{dd_oos:.1f}%", f"{dd_spy:.1f}%", f"{dd_qqq:.1f}%"],
    ["Sharpe Ratio", f"{sh_oos:.2f}", f"{sh_spy:.2f}", f"{sh_qqq:.2f}"],
    ["Avg Fold PF",  f"{avg_test_pf:.3f}", "-", "-"],
    ["Folds Profit", f"{sum(1 for fr in fold_results if fr['test_ret']>0)}/{len(fold_results)}", "-", "-"],
]

tbl = ax6.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.0, 1.6)

for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#21262d")
    cell.set_edgecolor("#30363d")
    cell.set_text_props(color="white")
    if r == 0:
        cell.set_facecolor("#1f6feb")
        cell.set_text_props(color="white", fontweight="bold")
    elif c == 1:
        cell.set_text_props(color=COLORS["OOS"], fontweight="bold")
    elif c == 2:
        cell.set_text_props(color=COLORS["SPY"])
    elif c == 3:
        cell.set_text_props(color=COLORS["QQQ"])

ax6.set_title("OOS Performance Summary", color="white", fontsize=12, fontweight="bold", pad=10)

# 保存
out_path = BASE / "output/backtest/walkforward_conditional_chart.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"\n  Chart saved: {out_path}")
