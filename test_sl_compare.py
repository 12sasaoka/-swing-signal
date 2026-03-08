"""
SL パラメータ比較テスト
既存バックテストCSVからimplied ATRを逆算して3シナリオを近似シミュレーション

シナリオ:
  ベースライン : SL = ATR×2.0  フロア -6%
  案①         : SL = ATR×1.5  フロア -6%
  案②         : SL = ATR×2.0  フロア -5%
  案③         : SL = ATR×1.5  フロア -5%

前提（近似）:
  - sl_hit トレード  : 新SL価格でP&L再計算（直接効果あり）
  - trailing_stop 他 : 変化なし（保守的仮定。実際は一部が早期sl_hitに変わる）
  ⇒ 改善効果の「楽観的上限」として解釈すること
"""

import csv
import math
import sys
import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE   = Path(__file__).resolve().parent
CSV_IN = BASE / "output/backtest/signal_backtest_20260303_022802_sb_trail4.0.csv"

INITIAL  = 4000.0
RISK     = 0.005
ALLOC    = 0.13
SB_ALLOC = 0.14
MAX_POS  = 11
MIN_RR   = 0.005

ETF_UNIVERSE = [
    "SPY", "QQQ", "XLC", "XLY", "XLP",
    "XLE", "XLF", "XLV", "XLI", "XLB",
    "XLK", "XLRE", "XLU",
]
SPY_TREND_MA = 200
SAFE_TICKER  = "SGOV"
ETF_TOP_N    = 2
ETF_MA       = 50

MOMENTUM_PERIODS = {
    "1M":  (21,  0.30),
    "3M":  (63,  0.30),
    "6M":  (126, 0.30),
    "12M": (252, 0.10),
}
DATA_START = "2019-10-01"
SIM_START  = "2021-01-04"
SIM_END    = "2026-02-28"


# ── シナリオ定義 ─────────────────────────────────────────────────────
SCENARIOS = {
    "Base (ATR×2.0 / -6%)": {"sl_mult": 2.0, "floor_pct": -0.06},
    "案① (ATR×1.5 / -6%)": {"sl_mult": 1.5, "floor_pct": -0.06},
    "案② (ATR×2.0 / -5%)": {"sl_mult": 2.0, "floor_pct": -0.05},
    "案③ (ATR×1.5 / -5%)": {"sl_mult": 1.5, "floor_pct": -0.05},
}


# ── 元トレード読み込み ────────────────────────────────────────────────
print("個別株シグナル読み込み中...")
raw_trades = []
with open(CSV_IN, "r", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        raw_trades.append({
            "ticker":      row["ticker"],
            "signal":      row["signal"],
            "entry_date":  row["entry_date"],
            "exit_date":   row["exit_date"],
            "entry_price": float(row["entry_price"]),
            "exit_price":  float(row["exit_price"]),
            "result":      row["result"],
            "pl_pct":      float(row["pl_pct"]),
            "sl_price":    float(row["sl_price"]),
        })


def apply_sl_scenario(trades_orig: list[dict], sl_mult: float, floor_pct: float) -> list[dict]:
    """
    各トレードに新SLパラメータを適用して修正後のトレードリストを返す。
    sl_hit のみ再計算。それ以外は変化なし（保守的仮定）。
    """
    result = []
    for t in trades_orig:
        t2 = dict(t)
        entry = t["entry_price"]
        current_sl = t["sl_price"]

        # implied ATR を逆算
        # 現SL = max(entry - ATR×2.0,  entry×(1-0.06))
        # floor が binding か ATR-based が binding かを判定
        floor_sl_current = entry * (1 + (-0.06))
        if abs(current_sl - floor_sl_current) < entry * 0.005:
            # フロアが binding → ATRは current_sl からは推定不能。ATRは大きかった。
            # -6%フロアが binding ⇒ ATR × 2.0 ≤ entry × (-0.06)
            # つまり ATR ≥ entry × 0.03（3%以上）
            # 新しい SL: フロアのみ変更、ATR-based SLはさらに低い可能性があるため
            # new_sl = entry × (1 + floor_pct)  （新フロアのみ）
            new_sl = entry * (1 + floor_pct)
            # ただし ATR-based (×1.5) が新フロアより高い可能性も考慮
            # ATRが不明なので保守的に floor のみで計算
        else:
            # ATR-based が binding
            # implied ATR = (entry - current_sl) / 2.0
            implied_atr = (entry - current_sl) / 2.0
            atr_based_new = entry - implied_atr * sl_mult
            floor_new     = entry * (1 + floor_pct)
            new_sl        = max(atr_based_new, floor_new)

        t2["sl_price"] = new_sl

        if t["result"] == "sl_hit":
            # 旧SL基準の理論損失（gap なし想定）
            old_sl_pct = (current_sl - entry) / entry * 100
            actual_pct = t["pl_pct"]
            gap_extra  = actual_pct - old_sl_pct  # 負なら gap-down

            if gap_extra < -1.5:
                # gap-down で旧SLより1.5%超悪化 → 新SLでも同じ損失（突き抜け）
                new_pl_pct = actual_pct
            else:
                # 通常のSLヒット → 新SL価格で損失を再計算
                new_pl_pct = (new_sl - entry) / entry * 100

            t2["pl_pct"]     = new_pl_pct
            t2["exit_price"] = entry * (1 + new_pl_pct / 100)

        result.append(t2)
    return result


# ── ETFデータ取得 ─────────────────────────────────────────────────────
tickers_dl = ETF_UNIVERSE + [SAFE_TICKER]
print(f"ETFデータ取得中（{len(tickers_dl)}本）...")
raw_all = yf.download(tickers_dl, start=DATA_START, end=SIM_END,
                      auto_adjust=True, progress=False)
etf_close: dict[str, pd.Series] = {}
for tk in tickers_dl:
    try:
        etf_close[tk] = raw_all["Close"][tk].dropna()
    except Exception:
        pass
print(f"  取得完了: {len(etf_close)}本")
spy_series = etf_close.get("SPY")


# ── ETF補助関数 ──────────────────────────────────────────────────────
def etf_price(ticker, date):
    s = etf_close.get(ticker)
    if s is None: return None
    past = s[s.index <= date]
    return float(past.iloc[-1]) if len(past) > 0 else None

def etf_ma_val(ticker, date, period):
    s = etf_close.get(ticker)
    if s is None: return None
    past = s[s.index <= date]
    return float(past.iloc[-period:].mean()) if len(past) >= period else None

def spy_regime(date):
    s = etf_close.get("SPY")
    if s is None: return True
    past = s[s.index <= date]
    if len(past) < SPY_TREND_MA: return True
    return float(past.iloc[-1]) > float(past.iloc[-SPY_TREND_MA:].mean())

def calc_momentum_scores(date):
    rets = {}
    for ticker in ETF_UNIVERSE:
        s = etf_close.get(ticker)
        if s is None: continue
        past = s[s.index <= date]
        if len(past) == 0: continue
        current = float(past.iloc[-1])
        tr = {}; ok = True
        for name, (days, _) in MOMENTUM_PERIODS.items():
            if len(past) < days + 1: ok = False; break
            base = float(past.iloc[-(days+1)])
            if base <= 0: ok = False; break
            tr[name] = (current - base) / base
        if ok: rets[ticker] = tr
    if len(rets) < 2: return []
    scores = {t: 0.0 for t in rets}
    for name, (_, weight) in MOMENTUM_PERIODS.items():
        ranked = sorted(rets.keys(), key=lambda t: rets[t][name], reverse=True)
        for rank, ticker in enumerate(ranked, 1):
            scores[ticker] += rank * weight
    return sorted(scores.items(), key=lambda x: x[1])

def select_etfs(date, scored):
    result = []
    for ticker, _ in scored:
        if len(result) >= ETF_TOP_N: break
        p = etf_price(ticker, date)
        ma = etf_ma_val(ticker, date, ETF_MA)
        if p and ma and p >= ma:
            result.append(ticker)
    return result

def period_key(date):
    return (date.year, date.isocalendar()[1])


# ── ETFオーバーレイ付きシミュレーション ────────────────────────────────
def run_sim(trades_in: list[dict]) -> dict:
    trades_in = sorted(trades_in, key=lambda t: t["entry_date"])
    same_day = {i for i, t in enumerate(trades_in) if t["entry_date"] == t["exit_date"]}

    entries_by_date: dict[str, list[int]] = {}
    exits_by_date:   dict[str, list[int]] = {}
    for i, t in enumerate(trades_in):
        entries_by_date.setdefault(t["entry_date"], []).append(i)
        exits_by_date.setdefault(t["exit_date"],   []).append(i)

    bdays = pd.bdate_range(SIM_START, SIM_END)
    cash = INITIAL; stock_pos = {}; etf_pos = {}
    peak = INITIAL; max_dd = 0.0
    daily = []
    pe: set[int] = set(); px: set[int] = set()
    prev_period = None; current_scores = []; prev_bull = None

    for date in bdays:
        date_str = date.strftime("%Y-%m-%d")
        bull     = spy_regime(date)
        reg_chg  = (prev_bull is not None and bull != prev_bull)

        pk = period_key(date)
        if pk != prev_period:
            prev_period = pk
            if bull:
                current_scores = calc_momentum_scores(date)
                targets = select_etfs(date, current_scores)
                for tk in list(etf_pos.keys()):
                    if tk not in targets:
                        p = etf_price(tk, date)
                        if p: cash += etf_pos[tk] * p
                        del etf_pos[tk]
                if targets and cash > 1.0:
                    each = cash / len(targets)
                    for tk in targets:
                        p = etf_price(tk, date)
                        if p and p > 0:
                            etf_pos[tk] = etf_pos.get(tk, 0.0) + each / p
                            cash -= each
            else:
                for tk in list(etf_pos.keys()):
                    if tk != SAFE_TICKER:
                        p = etf_price(tk, date)
                        if p: cash += etf_pos[tk] * p
                        del etf_pos[tk]
                sp = etf_price(SAFE_TICKER, date)
                if sp and cash > 1.0:
                    etf_pos[SAFE_TICKER] = etf_pos.get(SAFE_TICKER, 0.0) + cash / sp
                    cash = 0.0

        if reg_chg and not bull:
            sp = etf_price(SAFE_TICKER, date)
            for tk in list(etf_pos.keys()):
                if tk != SAFE_TICKER:
                    p = etf_price(tk, date)
                    if p: cash += etf_pos[tk] * p
                    del etf_pos[tk]
            if sp and cash > 1.0:
                etf_pos[SAFE_TICKER] = etf_pos.get(SAFE_TICKER, 0.0) + cash / sp
                cash = 0.0
        elif reg_chg and bull:
            if SAFE_TICKER in etf_pos:
                p = etf_price(SAFE_TICKER, date)
                if p: cash += etf_pos[SAFE_TICKER] * p
                del etf_pos[SAFE_TICKER]
            for tk in list(etf_pos.keys()):
                p = etf_price(tk, date); ma = etf_ma_val(tk, date, ETF_MA)
                if p and ma and p < ma:
                    cash += etf_pos[tk] * p; del etf_pos[tk]
        elif bull:
            for tk in list(etf_pos.keys()):
                if tk == SAFE_TICKER: continue
                p = etf_price(tk, date); ma = etf_ma_val(tk, date, ETF_MA)
                if p and ma and p < ma:
                    cash += etf_pos[tk] * p; del etf_pos[tk]

        prev_bull = bull

        for idx in exits_by_date.get(date_str, []):
            if idx in same_day: continue
            if idx in pe and idx not in px:
                px.add(idx)
                t   = trades_in[idx]
                amt = stock_pos.pop(idx, 0)
                pnl = amt * (t["pl_pct"] / 100)
                cash += amt + pnl

        etf_val  = sum(etf_pos[tk] * (etf_price(tk, date) or 0) for tk in etf_pos)
        total_eq = cash + sum(stock_pos.values()) + etf_val

        for idx in entries_by_date.get(date_str, []):
            if idx in pe: continue
            if len(stock_pos) >= MAX_POS: continue
            t   = trades_in[idx]
            rr  = max((t["entry_price"] - t["sl_price"]) / t["entry_price"]
                      if t["entry_price"] > 0 else 0.08, MIN_RR)
            inv = (total_eq * RISK) / rr
            ap  = SB_ALLOC if t["signal"] == "STRONG_BUY" else ALLOC
            need = min(inv, total_eq * ap)

            if cash < need and etf_pos:
                shortage  = need - cash
                score_map = {tk: s for tk, s in current_scores}
                ranked = sorted(etf_pos.keys(),
                                key=lambda tk: (0 if tk == SAFE_TICKER else 1,
                                                score_map.get(tk, 999)), reverse=True)
                for tk in ranked:
                    if shortage <= 0: break
                    p = etf_price(tk, date)
                    if not p: continue
                    sell = min(shortage, etf_pos[tk] * p)
                    etf_pos[tk] -= sell / p
                    if etf_pos[tk] < 1e-9: del etf_pos[tk]
                    cash += sell; shortage -= sell

            alloc = min(need, cash)
            if t["entry_price"] > 0:
                sh = math.floor(alloc / t["entry_price"])
                alloc = sh * t["entry_price"]
            else:
                sh = 0
            if alloc <= 0 or sh == 0 or cash <= 0: continue
            cash -= alloc; stock_pos[idx] = alloc; pe.add(idx)
            if idx in same_day:
                px.add(idx)
                cash += alloc + alloc * (t["pl_pct"] / 100)
                stock_pos.pop(idx, None)

        etf_val   = sum(etf_pos[tk] * (etf_price(tk, date) or 0) for tk in etf_pos)
        stock_val = sum(stock_pos.values())
        total_eq  = cash + stock_val + etf_val
        if total_eq > peak: peak = total_eq
        dd = (peak - total_eq) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd
        daily.append({"date": date, "balance": total_eq})

    df = pd.DataFrame(daily)
    final       = float(df.iloc[-1]["balance"])
    total_ret   = (final - INITIAL) / INITIAL * 100
    r           = df.set_index("date")["balance"].pct_change().dropna()
    sharpe      = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0
    calmar      = total_ret / max_dd if max_dd > 0 else 0
    yrs         = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
    cagr        = (final / INITIAL) ** (1 / yrs) - 1 if yrs > 0 else 0

    # 個別株トレード統計（修正後）
    all_exits = [t for i, t in enumerate(trades_in) if i in px]
    wins  = [t for t in all_exits if t["pl_pct"] > 0]
    loss  = [t for t in all_exits if t["pl_pct"] <= 0]
    sl_hits = [t for t in all_exits if t["result"] == "sl_hit"]
    avg_win  = np.mean([t["pl_pct"] for t in wins])  if wins else 0
    avg_loss = np.mean([t["pl_pct"] for t in loss])  if loss else 0
    avg_sl   = np.mean([t["pl_pct"] for t in sl_hits]) if sl_hits else 0
    win_rate = len(wins) / len(all_exits) * 100 if all_exits else 0
    gp = sum(t["pl_pct"] for t in wins)
    gl = abs(sum(t["pl_pct"] for t in loss))
    pf = gp / gl if gl > 0 else 9.99
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 9.99

    return {
        "final": final,
        "total_ret": total_ret,
        "cagr": cagr * 100,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "calmar": calmar,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_sl_loss": avg_sl,
        "sl_count": len(sl_hits),
        "pf": pf,
        "rr": rr,
        "trades": len(all_exits),
        "daily": df,
    }


# ── 各シナリオ実行 ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SL パラメータ比較  (ETF v2 weekly 固定)")
print("  ※ trailing_stop は変化なし仮定（楽観的上限）")
print("=" * 65)

results = {}
for label, params in SCENARIOS.items():
    print(f"\n[{label}] 実行中...", end="", flush=True)
    trades_mod = apply_sl_scenario(raw_trades, params["sl_mult"], params["floor_pct"])
    res = run_sim(trades_mod)
    results[label] = res
    print(f" 完了  Return={res['total_ret']:+.1f}%  Sharpe={res['sharpe']:.3f}  "
          f"RR={res['rr']:.3f}  WR={res['win_rate']:.1f}%")


# ── 比較テーブル ──────────────────────────────────────────────────────
print("\n" + "=" * 95)
print(f"  {'指標':<18} " + "  ".join(f"{k:<22}" for k in results))
print("  " + "-" * 91)

metrics = [
    ("最終残高",     lambda r: f"${r['final']:>10,.0f}"),
    ("総リターン",   lambda r: f"{r['total_ret']:>+10.1f}%"),
    ("CAGR",        lambda r: f"{r['cagr']:>+10.2f}%"),
    ("最大DD",      lambda r: f"{r['max_dd']:>10.2f}%"),
    ("Sharpe",      lambda r: f"{r['sharpe']:>10.3f}"),
    ("Calmar",      lambda r: f"{r['calmar']:>10.2f}"),
    ("PF",          lambda r: f"{r['pf']:>10.3f}"),
    ("勝率",        lambda r: f"{r['win_rate']:>10.1f}%"),
    ("RR比",        lambda r: f"{r['rr']:>10.3f}"),
    ("平均利益",    lambda r: f"{r['avg_win']:>+10.2f}%"),
    ("平均損失",    lambda r: f"{r['avg_loss']:>+10.2f}%"),
    ("SL平均損失",  lambda r: f"{r['avg_sl_loss']:>+10.2f}%"),
    ("SL件数",      lambda r: f"{r['sl_count']:>10}"),
    ("取引数",      lambda r: f"{r['trades']:>10}"),
]

for name, fmt in metrics:
    row = f"  {name:<18} "
    for label, res in results.items():
        row += f"  {fmt(res):<22}"
    print(row)

print("  " + "-" * 91)


# ── 年別リターン比較 ──────────────────────────────────────────────────
print(f"\n  年別リターン比較")
print(f"  {'年':<6} " + "  ".join(f"{k:<22}" for k in results))
print("  " + "-" * 91)

for yr in range(2021, 2027):
    row = f"  {yr:<6} "
    for label, res in results.items():
        df = res["daily"]
        yd = df[df["date"].dt.year == yr]
        if yd.empty:
            row += f"  {'---':<22}"
        else:
            prev_yr_end = df[df["date"].dt.year == yr - 1]
            prev_bal = float(prev_yr_end.iloc[-1]["balance"]) if not prev_yr_end.empty else INITIAL
            yr_ret = (float(yd.iloc[-1]["balance"]) - prev_bal) / prev_bal * 100
            row += f"  {yr_ret:>+6.1f}%{'':<14}"
    print(row)


# ── チャート ──────────────────────────────────────────────────────────
print("\nチャート生成中...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#0d1117")
fig.suptitle(
    "SL Parameter Comparison  (ETF v2 Weekly fixed: TOP_N=2  MA=50  weekly)\n"
    "※ trailing_stop unchanged (optimistic upper bound)",
    color="white", fontsize=13, fontweight="bold", y=0.99,
)

COLORS_SCN = {
    "Base (ATR×2.0 / -6%)": "#6e7681",
    "案① (ATR×1.5 / -6%)": "#00d4ff",
    "案② (ATR×2.0 / -5%)": "#f0a500",
    "案③ (ATR×1.5 / -5%)": "#7ee787",
}

def style(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#adbac7", labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", lw=0.7, linestyle="--")
    for yr in range(2022, 2027):
        ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="#6e7681", lw=0.5, linestyle="--", alpha=0.4)

# Panel 1: 資産曲線
ax1 = axes[0][0]; style(ax1)
for label, res in results.items():
    df = res["daily"]
    lw = 2.5 if "③" in label else (2.0 if "②" in label or "①" in label else 1.5)
    ls = "--" if label.startswith("Base") else "-"
    ax1.plot(df["date"], df["balance"], color=COLORS_SCN[label], lw=lw, ls=ls,
             label=f"{label}  ${res['final']:,.0f} ({res['total_ret']:+.1f}%)")
ax1.axhline(INITIAL, color="#6e7681", lw=0.8, linestyle=":")
ax1.set_title("Portfolio Value", color="white", fontsize=11, fontweight="bold")
ax1.set_ylabel("Value ($)", color="#adbac7")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=8, loc="upper left")

# Panel 2: RR比比較
ax2 = axes[0][1]; style(ax2)
labels_short = ["Base", "案①\n(SL×1.5)", "案②\n(Floor-5%)", "案③\n(両方)"]
rr_vals  = [res["rr"]      for res in results.values()]
wr_vals  = [res["win_rate"] for res in results.values()]
pf_vals  = [res["pf"]      for res in results.values()]
cols     = list(COLORS_SCN.values())
x = np.arange(4); w = 0.25
bars = ax2.bar(x, rr_vals, width=0.5, color=cols, alpha=0.85)
ax2.axhline(1.0, color="#f0a500", lw=1.2, linestyle="--", label="RR=1.0")
for bar, val in zip(bars, rr_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01,
             f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=9)
ax2.set_xticks(x); ax2.set_xticklabels(labels_short, color="#adbac7")
ax2.set_title("RR比（勝ち平均 / 負け平均）", color="white", fontsize=11, fontweight="bold")
ax2.set_ylabel("RR Ratio", color="#adbac7")
ax2.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=9)

# Panel 3: 勝率 vs RR散布図
ax3 = axes[1][0]; style(ax3)
ax3.grid(True, color="#21262d", lw=0.7, linestyle="--")
for (label, res), col in zip(results.items(), cols):
    ax3.scatter(res["win_rate"], res["rr"], s=120, color=col, zorder=5,
                label=label.split(" ")[0])
    ax3.annotate(label.split(" ")[0], (res["win_rate"], res["rr"]),
                 textcoords="offset points", xytext=(6, 4),
                 color=col, fontsize=9)
ax3.axhline(1.0, color="#6e7681", lw=0.8, linestyle="--", alpha=0.5)
ax3.axvline(50,  color="#6e7681", lw=0.8, linestyle="--", alpha=0.5)
ax3.set_xlabel("勝率 (%)", color="#adbac7")
ax3.set_ylabel("RR 比", color="#adbac7")
ax3.set_title("勝率 vs RR比", color="white", fontsize=11, fontweight="bold")

# Panel 4: 主要指標比較バー
ax4 = axes[1][1]; style(ax4)
metrics_bar = ["Sharpe", "PF", "RR比"]
vals_by_metric = {
    "Sharpe": [res["sharpe"] for res in results.values()],
    "PF":     [res["pf"]     for res in results.values()],
    "RR比":   [res["rr"]     for res in results.values()],
}
x = np.arange(4); w = 0.22
for i, (mname, mvals) in enumerate(vals_by_metric.items()):
    offset = (i - 1) * w
    bars_m = ax4.bar(x + offset, mvals, width=w, color=cols, alpha=0.75,
                     label=mname)
    # 最初の指標だけラベル
ax4.set_xticks(x); ax4.set_xticklabels(labels_short, color="#adbac7")
ax4.set_title("Sharpe / PF / RR比 比較", color="white", fontsize=11, fontweight="bold")
ax4.set_ylabel("値", color="#adbac7")
# 凡例
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=c, label=l.split(" ")[0])
           for c, l in zip(cols, results.keys())]
ax4.legend(handles=patches, facecolor="#21262d", edgecolor="#30363d",
           labelcolor="white", fontsize=8)

plt.tight_layout()
out_path = BASE / "output/backtest/sl_compare_chart.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"チャート保存: {out_path}")
print("\n完了")
