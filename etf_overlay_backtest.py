"""
ETFオーバーレイ バックテスト  v2（決定版）

個別株スイング戦略 + ETFモメンタムオーバーレイ（待機資金活用）

ETFルール（v2 週次）:
  ユニバース  : SPY QQQ XLC XLY XLP XLE XLF XLV XLI XLB XLK XLRE XLU
  スコアリング: 1M×0.30 + 3M×0.30 + 6M×0.30 + 12M×0.10（順位ベース）
  リバランス  : 毎週月曜（週次）
  保有        : 上位2本（各ETFが50MA以上のみ）
  50MA割れ    : 即日売却
  SPYレジーム : SPY > 200MA → ETFモメンタム
                SPY < 200MA → 全セクターETF売却・SGOVで待避
  個別株資金  : SGOVまたは弱いETFから必要額を売却
"""
import csv
import math
import sys
import os
from pathlib import Path

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

# ── パラメータ ──────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parent
CSV_IN = BASE / "output/backtest/signal_backtest_20260303_022802_sb_trail4.0.csv"

INITIAL      = 4000.0
RISK         = 0.005
ALLOC        = 0.13
SB_ALLOC     = 0.14
MAX_POS      = 11
MIN_RR       = 0.005

ETF_UNIVERSE = [
    "SPY", "QQQ", "XLC", "XLY", "XLP",
    "XLE", "XLF", "XLV", "XLI", "XLB",
    "XLK", "XLRE", "XLU",
]
ETF_TOP_N    = 2
ETF_MA       = 50
SPY_TREND_MA = 200
SAFE_TICKER  = "SGOV"
REBAL_FREQ   = "weekly"   # "monthly" / "weekly" / "daily"

MOMENTUM_PERIODS = {
    "1M":  (21,  0.30),
    "3M":  (63,  0.30),
    "6M":  (126, 0.30),
    "12M": (252, 0.10),
}

SIM_START  = "2021-01-04"
SIM_END    = "2026-02-28"
DATA_START = "2019-10-01"


# ── 個別株シグナル読み込み ──────────────────────────────────────────
print("個別株シグナル読み込み中...")
trades_raw = []
with open(CSV_IN, "r", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        trades_raw.append({
            "ticker":      row["ticker"],
            "signal":      row["signal"],
            "entry_date":  row["entry_date"],
            "exit_date":   row["exit_date"],
            "entry_price": float(row["entry_price"]),
            "exit_price":  float(row["exit_price"]),
            "pl_pct":      float(row["pl_pct"]),
            "sl_price":    float(row["sl_price"]),
        })

trades_raw.sort(key=lambda t: t["entry_date"])
same_day = {i for i, t in enumerate(trades_raw) if t["entry_date"] == t["exit_date"]}

entries_by_date: dict[str, list[int]] = {}
exits_by_date:   dict[str, list[int]] = {}
for i, t in enumerate(trades_raw):
    entries_by_date.setdefault(t["entry_date"], []).append(i)
    exits_by_date.setdefault(t["exit_date"],   []).append(i)


# ── ETFデータ取得 ──────────────────────────────────────────────────
tickers_dl = ETF_UNIVERSE + [SAFE_TICKER]
print(f"ETFデータ取得中（{len(tickers_dl)}本）...")
raw_all = yf.download(tickers_dl, start=DATA_START, end=SIM_END,
                      auto_adjust=True, progress=False)

etf_close: dict[str, pd.Series] = {}
for tk in tickers_dl:
    try:
        s = raw_all["Close"][tk].dropna()
        etf_close[tk] = s
    except Exception as e:
        print(f"  Warning: {tk} 取得失敗 - {e}")

print(f"  取得完了: {list(etf_close.keys())}")
spy_series = etf_close.get("SPY")


# ── ETF補助関数 ────────────────────────────────────────────────────
def etf_price(ticker: str, date: pd.Timestamp) -> float | None:
    s = etf_close.get(ticker)
    if s is None: return None
    past = s[s.index <= date]
    return float(past.iloc[-1]) if len(past) > 0 else None


def etf_ma(ticker: str, date: pd.Timestamp, period: int = ETF_MA) -> float | None:
    s = etf_close.get(ticker)
    if s is None: return None
    past = s[s.index <= date]
    return float(past.iloc[-period:].mean()) if len(past) >= period else None


def spy_regime(date: pd.Timestamp) -> bool:
    """True = ブル（SPY > 200MA）"""
    s = etf_close.get("SPY")
    if s is None: return True
    past = s[s.index <= date]
    if len(past) < SPY_TREND_MA: return True
    return float(past.iloc[-1]) > float(past.iloc[-SPY_TREND_MA:].mean())


def calc_momentum_scores(date: pd.Timestamp) -> list[tuple[str, float]]:
    rets: dict[str, dict[str, float]] = {}
    for ticker in ETF_UNIVERSE:
        s = etf_close.get(ticker)
        if s is None: continue
        past = s[s.index <= date]
        if len(past) == 0: continue
        current = float(past.iloc[-1])
        tr = {}; ok = True
        for name, (days, _) in MOMENTUM_PERIODS.items():
            if len(past) < days + 1: ok = False; break
            base = float(past.iloc[-(days + 1)])
            if base <= 0: ok = False; break
            tr[name] = (current - base) / base
        if ok: rets[ticker] = tr
    if len(rets) < 2: return []
    scores: dict[str, float] = {t: 0.0 for t in rets}
    for name, (_, weight) in MOMENTUM_PERIODS.items():
        ranked = sorted(rets.keys(), key=lambda t: rets[t][name], reverse=True)
        for rank, ticker in enumerate(ranked, 1):
            scores[ticker] += rank * weight
    return sorted(scores.items(), key=lambda x: x[1])


def select_target_etfs(date: pd.Timestamp, scored: list[tuple[str, float]]) -> list[str]:
    result = []
    for ticker, _ in scored:
        if len(result) >= ETF_TOP_N: break
        p = etf_price(ticker, date); ma = etf_ma(ticker, date)
        if p and ma and p >= ma: result.append(ticker)
    return result


def get_period_key(date: pd.Timestamp) -> tuple:
    if REBAL_FREQ == "weekly":
        return (date.year, date.isocalendar()[1])
    elif REBAL_FREQ == "daily":
        return (date.year, date.month, date.day)
    else:
        return (date.year, date.month)


# ── シミュレーション本体 ───────────────────────────────────────────
def run_simulation(with_etf: bool) -> tuple[pd.DataFrame, list[dict], float]:
    bdays = pd.bdate_range(SIM_START, SIM_END)

    cash       = INITIAL
    stock_pos  = {}
    etf_pos    = {}
    peak       = INITIAL
    max_dd     = 0.0

    daily_records: list[dict] = []
    trade_history: list[dict] = []

    pe: set[int] = set()
    px: set[int] = set()

    prev_period    = None
    current_scores: list[tuple[str, float]] = []
    prev_bull      = None

    for date in bdays:
        date_str  = date.strftime("%Y-%m-%d")
        bull      = spy_regime(date) if with_etf else True
        reg_chg   = (prev_bull is not None and bull != prev_bull)

        # ─── ① リバランス ─────────────────────────────────────────
        period_key = get_period_key(date)
        if with_etf and period_key != prev_period:
            prev_period = period_key
            if bull:
                current_scores = calc_momentum_scores(date)
                targets = select_target_etfs(date, current_scores)
                for tk in list(etf_pos.keys()):
                    if tk not in targets:
                        p = etf_price(tk, date)
                        if p: cash += etf_pos[tk] * p
                        del etf_pos[tk]
                if targets and cash > 1.0:
                    alloc_each = cash / len(targets)
                    for tk in targets:
                        p = etf_price(tk, date)
                        if p and p > 0:
                            etf_pos[tk] = etf_pos.get(tk, 0.0) + alloc_each / p
                            cash -= alloc_each
            else:
                for tk in list(etf_pos.keys()):
                    if tk != SAFE_TICKER:
                        p = etf_price(tk, date)
                        if p: cash += etf_pos[tk] * p
                        del etf_pos[tk]
                sgov_p = etf_price(SAFE_TICKER, date)
                if sgov_p and cash > 1.0:
                    etf_pos[SAFE_TICKER] = etf_pos.get(SAFE_TICKER, 0.0) + cash / sgov_p
                    cash = 0.0

        # ─── ② 日次: レジーム変化 + 50MA割れ ──────────────────────
        if with_etf:
            if reg_chg and not bull:
                sgov_p = etf_price(SAFE_TICKER, date)
                for tk in list(etf_pos.keys()):
                    if tk != SAFE_TICKER:
                        p = etf_price(tk, date)
                        if p: cash += etf_pos[tk] * p
                        del etf_pos[tk]
                if sgov_p and cash > 1.0:
                    etf_pos[SAFE_TICKER] = etf_pos.get(SAFE_TICKER, 0.0) + cash / sgov_p
                    cash = 0.0
            elif reg_chg and bull:
                if SAFE_TICKER in etf_pos:
                    p = etf_price(SAFE_TICKER, date)
                    if p: cash += etf_pos[SAFE_TICKER] * p
                    del etf_pos[SAFE_TICKER]
                for tk in list(etf_pos.keys()):
                    p = etf_price(tk, date); ma = etf_ma(tk, date)
                    if p and ma and p < ma:
                        cash += etf_pos[tk] * p; del etf_pos[tk]
            elif bull:
                for tk in list(etf_pos.keys()):
                    if tk == SAFE_TICKER: continue
                    p = etf_price(tk, date); ma = etf_ma(tk, date)
                    if p and ma and p < ma:
                        cash += etf_pos[tk] * p; del etf_pos[tk]

        prev_bull = bull

        # ─── ③ 個別株EXIT ─────────────────────────────────────────
        for idx in exits_by_date.get(date_str, []):
            if idx in same_day: continue
            if idx in pe and idx not in px:
                px.add(idx)
                t   = trades_raw[idx]
                amt = stock_pos.pop(idx)
                pnl = amt * (t["pl_pct"] / 100)
                cash += amt + pnl
                etf_val = sum(etf_pos[tk] * (etf_price(tk, date) or 0) for tk in etf_pos)
                bal = cash + sum(stock_pos.values()) + etf_val
                if bal > peak: peak = bal
                dd = (peak - bal) / peak * 100 if peak > 0 else 0
                if dd > max_dd: max_dd = dd
                trade_history.append({
                    "date": date, "ticker": t["ticker"],
                    "pl_pct": t["pl_pct"], "pnl": pnl, "balance": bal,
                })

        # ─── ④ 個別株ENTRY ────────────────────────────────────────
        etf_val  = sum(etf_pos[tk] * (etf_price(tk, date) or 0) for tk in etf_pos)
        total_eq = cash + sum(stock_pos.values()) + etf_val

        for idx in entries_by_date.get(date_str, []):
            if idx in pe: continue
            if len(stock_pos) >= MAX_POS: continue
            t   = trades_raw[idx]
            rr  = max((t["entry_price"] - t["sl_price"]) / t["entry_price"]
                      if t["entry_price"] > 0 else 0.08, MIN_RR)
            inv = (total_eq * RISK) / rr
            ap  = SB_ALLOC if t["signal"] == "STRONG_BUY" else ALLOC
            need = min(inv, total_eq * ap)

            if with_etf and cash < need and etf_pos:
                shortage  = need - cash
                score_map = {tk: s for tk, s in current_scores}
                ranked = sorted(
                    etf_pos.keys(),
                    key=lambda tk: (0 if tk == SAFE_TICKER else 1,
                                    score_map.get(tk, 999)),
                    reverse=True,
                )
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
                sh    = math.floor(alloc / t["entry_price"])
                alloc = sh * t["entry_price"]
            else:
                sh = 0
            if alloc <= 0 or sh == 0 or cash <= 0: continue
            cash -= alloc; stock_pos[idx] = alloc; pe.add(idx)

            if idx in same_day:
                px.add(idx)
                cash += alloc + alloc * (t["pl_pct"] / 100)
                stock_pos.pop(idx, None)

        # ─── ⑤ 日次スナップショット ──────────────────────────────
        etf_val   = sum(etf_pos[tk] * (etf_price(tk, date) or 0) for tk in etf_pos)
        stock_val = sum(stock_pos.values())
        total_eq  = cash + stock_val + etf_val
        if total_eq > peak: peak = total_eq
        dd = (peak - total_eq) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd

        sgov_val = etf_pos.get(SAFE_TICKER, 0.0) * (etf_price(SAFE_TICKER, date) or 0)
        sect_val = etf_val - sgov_val

        daily_records.append({
            "date":      date,
            "balance":   total_eq,
            "cash":      cash,
            "stock_val": stock_val,
            "etf_val":   sect_val,
            "sgov_val":  sgov_val,
            "n_stocks":  len(stock_pos),
            "etf_names": "+".join(sorted(k for k in etf_pos if k != SAFE_TICKER)) or "-",
        })

    df = pd.DataFrame(daily_records)
    return df, trade_history, max_dd


# ── 実行 ──────────────────────────────────────────────────────────
print("\nベースライン シミュレーション実行中...")
df_base, hist_base, dd_base = run_simulation(with_etf=False)

print(f"ETFオーバーレイ v2（{REBAL_FREQ}）シミュレーション実行中...")
df_etf, hist_etf, dd_etf = run_simulation(with_etf=True)


# ── サマリー計算・出力 ────────────────────────────────────────────
def summary(df, hist, max_dd, label):
    final  = float(df.iloc[-1]["balance"])
    ret    = (final - INITIAL) / INITIAL * 100
    wins   = sum(1 for h in hist if h["pnl"] > 0)
    gp     = sum(h["pnl"] for h in hist if h["pnl"] > 0)
    gl     = abs(sum(h["pnl"] for h in hist if h["pnl"] < 0))
    pf     = gp / gl if gl > 0 else 9.99
    wr     = wins / len(hist) * 100 if hist else 0
    r      = df.set_index("date")["balance"].pct_change().dropna()
    sharpe = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0
    calmar = ret / max_dd if max_dd > 0 else 0
    yrs    = (pd.Timestamp(SIM_END) - pd.Timestamp(SIM_START)).days / 365.25
    cagr   = (final / INITIAL) ** (1 / yrs) - 1
    print(f"\n=== {label} ===")
    print(f"  最終残高  : ${final:>10,.2f}")
    print(f"  リターン  : {ret:>+10.2f}%")
    print(f"  CAGR      : {cagr*100:>+10.2f}%")
    print(f"  最大DD    : {max_dd:>10.2f}%")
    print(f"  Sharpe    : {sharpe:>10.3f}")
    print(f"  Calmar    : {calmar:>10.2f}")
    print(f"  PF        : {pf:>10.3f}")
    print(f"  勝率      : {wr:>10.1f}%")
    print(f"  取引数    : {len(hist):>10,}")
    return final, ret, cagr, max_dd, sharpe, calmar

final_base, ret_base, cagr_base, dd_b, sh_b, cal_b = summary(df_base, hist_base, dd_base, "ベースライン（個別株のみ）")
final_etf,  ret_etf,  cagr_etf,  dd_e, sh_e, cal_e = summary(df_etf,  hist_etf,  dd_etf,  f"ETF v2 {REBAL_FREQ}")

print(f"\n  差分リターン: {ret_etf - ret_base:+.2f}%  "
      f"差分P/L: ${final_etf - final_base:+,.2f}  "
      f"Sharpe差: {sh_e - sh_b:+.3f}")

# 年別
print("\n=== 年別リターン比較 ===")
print(f"{'年':<6} {'Base':>8} {'ETF v2':>8} {'差分':>8}")
print("-" * 34)
prev_b = prev_e = INITIAL
for yr in range(2021, 2027):
    b_yd = df_base[df_base["date"].dt.year == yr]
    e_yd = df_etf[df_etf["date"].dt.year == yr]
    if b_yd.empty: continue
    b_end = float(b_yd.iloc[-1]["balance"]); e_end = float(e_yd.iloc[-1]["balance"])
    b_r = (b_end - prev_b) / prev_b * 100; e_r = (e_end - prev_e) / prev_e * 100
    print(f"{yr:<6} {b_r:>+7.1f}%  {e_r:>+7.1f}%  {e_r-b_r:>+6.1f}%")
    prev_b = b_end; prev_e = e_end

# 配分内訳
avg_s  = df_etf["stock_val"].mean()
avg_e  = df_etf["etf_val"].mean()
avg_sg = df_etf["sgov_val"].mean()
avg_c  = df_etf["cash"].mean()
tot    = avg_s + avg_e + avg_sg + avg_c
print(f"\n=== ETF v2 平均配分 ===")
print(f"  個別株:     ${avg_s:>7,.0f}  ({avg_s/tot*100:.1f}%)")
print(f"  セクターETF:  ${avg_e:>5,.0f}  ({avg_e/tot*100:.1f}%)")
print(f"  SGOV:       ${avg_sg:>7,.0f}  ({avg_sg/tot*100:.1f}%)")
print(f"  キャッシュ:   ${avg_c:>5,.0f}  ({avg_c/tot*100:.1f}%)")


# ── SPYデータ取得 ──────────────────────────────────────────────────
spy_norm = None
if spy_series is not None:
    sim_spy  = spy_series[(spy_series.index >= pd.Timestamp(SIM_START)) &
                           (spy_series.index <= pd.Timestamp(SIM_END))]
    spy_norm = sim_spy / sim_spy.iloc[0] * INITIAL
    spy_ret  = (spy_norm.iloc[-1] / INITIAL - 1) * 100


# ── チャート ──────────────────────────────────────────────────────
print("\nチャート生成中...")
fig, axes = plt.subplots(3, 1, figsize=(16, 18), facecolor="#0d1117",
                         gridspec_kw={"height_ratios": [3, 2, 1.5], "hspace": 0.4})
fig.suptitle(
    f"ETF Overlay v2  ({REBAL_FREQ} rebalance  |  SPY 200MA regime  |  SGOV bear shelter)\n"
    f"risk=0.5%  trail4.0  |  Jan 2021 – Feb 2026",
    color="white", fontsize=13, fontweight="bold", y=0.99,
)

def style(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#adbac7", labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", lw=0.7, linestyle="--")
    for yr in range(2022, 2027):
        ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="#6e7681", lw=0.6, linestyle="--", alpha=0.4)

COLORS = {"base": "#00d4ff", "etf": "#7ee787", "spy": "#888888"}

# Panel 1: 資産曲線
ax1 = axes[0]; style(ax1)
ax1.plot(df_etf["date"],  df_etf["balance"],
         color=COLORS["etf"], lw=2.5,
         label=f"ETF v2 ({REBAL_FREQ})  ${final_etf:,.0f}  ({ret_etf:+.1f}%)")
ax1.plot(df_base["date"], df_base["balance"],
         color=COLORS["base"], lw=1.8, linestyle="--",
         label=f"Baseline  ${final_base:,.0f}  ({ret_base:+.1f}%)")
if spy_norm is not None:
    ax1.plot(spy_norm.index, spy_norm.values,
             color=COLORS["spy"], lw=1.3, alpha=0.7,
             label=f"SPY  ${spy_norm.iloc[-1]:,.0f}  ({spy_ret:+.1f}%)")
ax1.axhline(INITIAL, color="#6e7681", lw=0.8, linestyle=":")
ax1.set_title("Portfolio Value", color="white", fontsize=12, fontweight="bold")
ax1.set_ylabel("Value ($)", color="#adbac7")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white",
           fontsize=10, loc="upper left")

# Panel 2: 配分スタック
ax2 = axes[1]; style(ax2)
ax2.stackplot(
    df_etf["date"],
    df_etf["stock_val"].values,
    df_etf["etf_val"].values,
    df_etf["sgov_val"].values,
    df_etf["cash"].values,
    labels=[f"Stocks (avg ${avg_s:,.0f})",
            f"Sector ETF (avg ${avg_e:,.0f})",
            f"SGOV (avg ${avg_sg:,.0f})",
            f"Cash (avg ${avg_c:,.0f})"],
    colors=["#00d4ff", "#f0a500", "#7ee787", "#555555"],
    alpha=0.75,
)
ax2.set_title("Capital Allocation  (Stocks / Sector ETF / SGOV / Cash)",
              color="white", fontsize=12, fontweight="bold")
ax2.set_ylabel("Amount ($)", color="#adbac7")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white",
           fontsize=9, loc="upper left")

# Panel 3: 年別リターン比較
ax3 = axes[2]; style(ax3)
yr_list = list(range(2021, 2027))
yrets_b, yrets_e = [], []
prev_b2 = prev_e2 = INITIAL
for yr in yr_list:
    for lst, df in [(yrets_b, df_base), (yrets_e, df_etf)]:
        yd = df[df["date"].dt.year == yr]
        lst.append(0 if yd.empty else (float(yd.iloc[-1]["balance"]) - (prev_b2 if lst is yrets_b else prev_e2))
                   / (prev_b2 if lst is yrets_b else prev_e2) * 100)
    if not df_base[df_base["date"].dt.year == yr].empty:
        prev_b2 = float(df_base[df_base["date"].dt.year == yr].iloc[-1]["balance"])
    if not df_etf[df_etf["date"].dt.year == yr].empty:
        prev_e2 = float(df_etf[df_etf["date"].dt.year == yr].iloc[-1]["balance"])

x = np.arange(len(yr_list)); w = 0.35
ax3.bar(x - w/2, yrets_b, width=w, color=COLORS["base"], alpha=0.8, label="Baseline")
ax3.bar(x + w/2, yrets_e, width=w, color=COLORS["etf"],  alpha=0.8, label=f"ETF v2 ({REBAL_FREQ})")
ax3.axhline(0, color="#6e7681", lw=0.8)
ax3.set_xticks(x); ax3.set_xticklabels([str(y) for y in yr_list], color="#adbac7")
ax3.set_ylabel("Return (%)", color="#adbac7")
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
ax3.set_title("Annual Return Comparison", color="white", fontsize=12, fontweight="bold")
ax3.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=9)
for i, v in enumerate(yrets_e):
    ax3.text(i + w/2, v + (1 if v >= 0 else -2.5), f"{v:+.0f}%",
             ha="center", va="bottom" if v >= 0 else "top",
             color=COLORS["etf"], fontsize=8)

out_path = BASE / "output/backtest/etf_overlay_result.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"\nチャート保存: {out_path}")
