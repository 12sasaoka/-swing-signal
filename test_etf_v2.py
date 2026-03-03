"""
ETFオーバーレイ v2 バックテスト

v1（現行）との違い:
  - SPY > 200MA: 現行ETFモメンタム戦略（上位2本）
  - SPY < 200MA: 全セクターETFを売却 → SGOVで待機

比較: Baseline / ETF v1（現行）/ ETF v2（SPY200MAフィルター付き）
"""
import csv, math, sys, os
from pathlib import Path
from collections import defaultdict

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

INITIAL  = 4000.0
RISK     = 0.005
ALLOC    = 0.13
SB_ALLOC = 0.14
MAX_POS  = 11
MIN_RR   = 0.005

ETF_UNIVERSE = [
    "SPY","QQQ","XLC","XLY","XLP",
    "XLE","XLF","XLV","XLI","XLB",
    "XLK","XLRE","XLU",
]
ETF_TOP_N = 2
ETF_MA    = 50

MOMENTUM_PERIODS = {
    "1M":  (21,  0.30),
    "3M":  (63,  0.30),
    "6M":  (126, 0.30),
    "12M": (252, 0.10),
}

SPY_TREND_MA  = 200   # SPY レジームフィルター
SAFE_TICKER   = "SGOV"  # ベア相場時の待避先

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


# ── データ取得 ─────────────────────────────────────────────────────
tickers_dl = ETF_UNIVERSE + [SAFE_TICKER]
print(f"データ取得中（{len(tickers_dl)}本）...")
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


# ── 補助関数 ───────────────────────────────────────────────────────
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
    """True = SPY > 200MA（ブル）、False = ベア"""
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
        ticker_rets = {}
        ok = True
        for name, (days, _) in MOMENTUM_PERIODS.items():
            if len(past) < days + 1: ok = False; break
            base = float(past.iloc[-(days + 1)])
            if base <= 0: ok = False; break
            ticker_rets[name] = (current - base) / base
        if ok: rets[ticker] = ticker_rets

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
        p  = etf_price(ticker, date)
        ma = etf_ma(ticker, date)
        if p is not None and ma is not None and p >= ma:
            result.append(ticker)
    return result


# ── シミュレーション本体 ───────────────────────────────────────────
def run_simulation(mode: str, rebalance_freq: str = "monthly") -> tuple[pd.DataFrame, list[dict], float]:
    """
    mode: "baseline" / "etf_v1" / "etf_v2"
    rebalance_freq: "monthly" / "weekly" / "daily"
    """
    bdays = pd.bdate_range(SIM_START, SIM_END)

    cash       = INITIAL
    stock_pos  = {}   # trade_idx -> alloc ($)
    etf_pos    = {}   # ticker -> shares
    peak       = INITIAL
    max_dd     = 0.0

    daily_records: list[dict]  = []
    trade_history: list[dict]  = []

    pe: set[int] = set()
    px: set[int] = set()

    prev_period    = None
    current_scores: list[tuple[str, float]] = []

    with_etf  = mode in ("etf_v1", "etf_v2")
    use_spy_filter = (mode == "etf_v2")

    def get_period_key(d):
        if rebalance_freq == "weekly":
            return (d.year, d.isocalendar()[1])
        elif rebalance_freq == "daily":
            return (d.year, d.month, d.day)
        else:  # monthly
            return (d.year, d.month)

    for date in bdays:
        date_str = date.strftime("%Y-%m-%d")
        bull = spy_regime(date) if use_spy_filter else True

        # ─── ① リバランス ─────────────────────────────────────────
        period_key = get_period_key(date)
        if with_etf and period_key != prev_period:
            prev_period = period_key

            if bull:
                # ブル: モメンタム上位ETFに配分
                current_scores = calc_momentum_scores(date)
                targets = select_target_etfs(date, current_scores)

                # 非ターゲット（SGOVを含む）を売却
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
                            shares = alloc_each / p
                            etf_pos[tk] = etf_pos.get(tk, 0.0) + shares
                            cash -= alloc_each
            else:
                # ベア: 全セクターETFを売却 → SGOV待避
                for tk in list(etf_pos.keys()):
                    if tk != SAFE_TICKER:
                        p = etf_price(tk, date)
                        if p: cash += etf_pos[tk] * p
                        del etf_pos[tk]

                sgov_p = etf_price(SAFE_TICKER, date)
                if sgov_p and cash > 1.0:
                    shares = cash / sgov_p
                    etf_pos[SAFE_TICKER] = etf_pos.get(SAFE_TICKER, 0.0) + shares
                    cash = 0.0

        # ─── ② 日次: レジーム変化 + 50MA割れチェック ─────────────
        if with_etf:
            if use_spy_filter and not bull:
                # ベアに転換: セクターETF → SGOV
                sgov_p = etf_price(SAFE_TICKER, date)
                for tk in list(etf_pos.keys()):
                    if tk != SAFE_TICKER:
                        p = etf_price(tk, date)
                        if p: cash += etf_pos[tk] * p
                        del etf_pos[tk]
                if sgov_p and cash > 1.0:
                    shares = cash / sgov_p
                    etf_pos[SAFE_TICKER] = etf_pos.get(SAFE_TICKER, 0.0) + shares
                    cash = 0.0
            elif use_spy_filter and bull:
                # ブルに転換: SGOVを解放（翌月リバランスで再配分）
                if SAFE_TICKER in etf_pos:
                    p = etf_price(SAFE_TICKER, date)
                    if p: cash += etf_pos[SAFE_TICKER] * p
                    del etf_pos[SAFE_TICKER]
                # 通常の50MAチェック
                for tk in list(etf_pos.keys()):
                    p  = etf_price(tk, date)
                    ma = etf_ma(tk, date)
                    if p and ma and p < ma:
                        cash += etf_pos[tk] * p
                        del etf_pos[tk]
            else:
                # v1: 50MAチェックのみ
                for tk in list(etf_pos.keys()):
                    p  = etf_price(tk, date)
                    ma = etf_ma(tk, date)
                    if p and ma and p < ma:
                        cash += etf_pos[tk] * p
                        del etf_pos[tk]

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

                trade_history.append({"date": date, "ticker": t["ticker"],
                                       "pl_pct": t["pl_pct"], "pnl": pnl, "balance": bal})

        # ─── ④ 個別株ENTRY ────────────────────────────────────────
        etf_val  = sum(etf_pos[tk] * (etf_price(tk, date) or 0) for tk in etf_pos)
        total_eq = cash + sum(stock_pos.values()) + etf_val

        for idx in entries_by_date.get(date_str, []):
            if idx in pe: continue
            if len(stock_pos) >= MAX_POS: continue

            t  = trades_raw[idx]
            rr = max((t["entry_price"] - t["sl_price"]) / t["entry_price"]
                     if t["entry_price"] > 0 else 0.08, MIN_RR)
            inv  = (total_eq * RISK) / rr
            ap   = SB_ALLOC if t["signal"] == "STRONG_BUY" else ALLOC
            need = min(inv, total_eq * ap)

            # 現金不足 → ETF/SGOV売却
            if with_etf and cash < need and etf_pos:
                shortage  = need - cash
                score_map = {tk: s for tk, s in current_scores}
                # v2: SGOVを最優先で売る（ベア相場待避中）
                ranked = sorted(etf_pos.keys(),
                                key=lambda tk: (0 if tk == SAFE_TICKER else 1,
                                                score_map.get(tk, 999)),
                                reverse=True)
                for tk in ranked:
                    if shortage <= 0: break
                    p = etf_price(tk, date)
                    if not p: continue
                    avail = etf_pos[tk] * p
                    sell  = min(shortage, avail)
                    etf_pos[tk] -= sell / p
                    if etf_pos[tk] < 1e-9: del etf_pos[tk]
                    cash     += sell
                    shortage -= sell

            alloc = min(need, cash)
            if t["entry_price"] > 0:
                sh    = math.floor(alloc / t["entry_price"])
                alloc = sh * t["entry_price"]
            else:
                sh = 0
            if alloc <= 0 or sh == 0 or cash <= 0: continue

            cash -= alloc
            stock_pos[idx] = alloc
            pe.add(idx)

            if idx in same_day:
                px.add(idx)
                pnl  = alloc * (t["pl_pct"] / 100)
                cash += alloc + pnl
                stock_pos.pop(idx, None)

        # ─── ⑤ 日次スナップショット ──────────────────────────────
        etf_val   = sum(etf_pos[tk] * (etf_price(tk, date) or 0) for tk in etf_pos)
        stock_val = sum(stock_pos.values())
        total_eq  = cash + stock_val + etf_val

        if total_eq > peak: peak = total_eq
        dd = (peak - total_eq) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd

        sgov_val  = etf_pos.get(SAFE_TICKER, 0.0) * (etf_price(SAFE_TICKER, date) or 0)
        sect_val  = etf_val - sgov_val

        daily_records.append({
            "date":      date,
            "balance":   total_eq,
            "cash":      cash,
            "stock_val": stock_val,
            "etf_val":   sect_val,
            "sgov_val":  sgov_val,
            "n_stocks":  len(stock_pos),
        })

    df = pd.DataFrame(daily_records)
    return df, trade_history, max_dd


# ── シミュレーション実行 ──────────────────────────────────────────
print("\nBaseline 実行中...")
df_base, hist_base, dd_base = run_simulation("baseline")

print("ETF v2 月次 実行中...")
df_monthly, hist_monthly, dd_monthly = run_simulation("etf_v2", "monthly")

print("ETF v2 週次 実行中...")
df_weekly,  hist_weekly,  dd_weekly  = run_simulation("etf_v2", "weekly")

print("ETF v2 日次 実行中...")
df_daily,   hist_daily,   dd_daily   = run_simulation("etf_v2", "daily")

# 日次データCSV保存
df_monthly.to_csv(BASE / "output/backtest/etf_v2_daily.csv",   index=False, encoding="utf-8-sig")
df_base.to_csv(BASE / "output/backtest/etf_base_daily.csv", index=False, encoding="utf-8-sig")

# v1用（チャート互換）
df_v1, hist_v1, dd_v1 = df_monthly, hist_monthly, dd_monthly
df_v2 = df_weekly


# ── サマリー出力 ──────────────────────────────────────────────────
def get_summary(df, hist, max_dd, label):
    final  = float(df.iloc[-1]["balance"])
    ret    = (final - INITIAL) / INITIAL * 100
    wins   = sum(1 for h in hist if h["pnl"] > 0)
    gp     = sum(h["pnl"] for h in hist if h["pnl"] > 0)
    gl     = abs(sum(h["pnl"] for h in hist if h["pnl"] < 0))
    pf     = gp / gl if gl > 0 else 9.99
    wr     = wins / len(hist) * 100 if hist else 0
    r      = df.set_index("date")["balance"].pct_change().dropna()
    sharpe = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0
    peak_s = df["balance"].cummax()
    calmar = ret / max_dd if max_dd > 0 else 0
    start  = pd.Timestamp("2021-01-04")
    end    = pd.Timestamp("2026-02-28")
    yrs    = (end - start).days / 365.25
    cagr   = (final / INITIAL) ** (1 / yrs) - 1
    return {"label": label, "final": final, "ret": ret, "cagr": cagr*100,
            "max_dd": max_dd, "sharpe": sharpe, "calmar": calmar,
            "pf": pf, "wr": wr, "trades": len(hist)}

s_base    = get_summary(df_base,    hist_base,    dd_base,    "Baseline")
s_monthly = get_summary(df_monthly, hist_monthly, dd_monthly, "v2 月次")
s_weekly  = get_summary(df_weekly,  hist_weekly,  dd_weekly,  "v2 週次")
s_daily   = get_summary(df_daily,   hist_daily,   dd_daily,   "v2 日次")

print("\n" + "=" * 82)
print(f"{'指標':<18} {'Baseline':>13} {'v2 月次':>13} {'v2 週次':>13} {'v2 日次':>13}")
print("-" * 82)
rows_def = [
    ("final",  "最終残高",   "${:,.0f}"),
    ("ret",    "総リターン", "{:+.2f}%"),
    ("cagr",   "CAGR",       "{:+.2f}%"),
    ("max_dd", "最大DD",     "{:.2f}%"),
    ("sharpe", "Sharpe",     "{:.3f}"),
    ("calmar", "Calmar",     "{:.2f}"),
    ("pf",     "PF",         "{:.3f}"),
    ("wr",     "勝率",       "{:.1f}%"),
]
for key, lbl, fmt in rows_def:
    print(f"{lbl:<18} {fmt.format(s_base[key]):>13} "
          f"{fmt.format(s_monthly[key]):>13} "
          f"{fmt.format(s_weekly[key]):>13} "
          f"{fmt.format(s_daily[key]):>13}")

# 年別比較
print("\n=== 年別リターン比較 ===")
print(f"{'年':<6} {'Base':>8} {'月次':>8} {'週次':>8} {'日次':>8}  {'週次-Base':>10}")
print("-" * 58)
prevs = {k: INITIAL for k in ["b","mo","wk","dy"]}
for yr in range(2021, 2027):
    rets = {}
    for k, df in [("b",df_base),("mo",df_monthly),("wk",df_weekly),("dy",df_daily)]:
        yd = df[df["date"].dt.year == yr]
        if yd.empty: rets[k] = 0.0; continue
        end = float(yd.iloc[-1]["balance"])
        rets[k] = (end - prevs[k]) / prevs[k] * 100
        prevs[k] = end
    diff = rets["wk"] - rets["b"]
    mark = "★" if diff > 1 else "▼" if diff < -1 else " "
    print(f"{yr:<6} {rets['b']:>+7.1f}% {rets['mo']:>+7.1f}% "
          f"{rets['wk']:>+7.1f}% {rets['dy']:>+7.1f}%  {diff:>+8.1f}% {mark}")

# 配分比較
print(f"\n=== 平均キャッシュ比率の比較 ===")
for lbl, df in [("Baseline", df_base), ("月次", df_monthly), ("週次", df_weekly), ("日次", df_daily)]:
    cp = (df["cash"] / df["balance"] * 100).mean()
    ep = (df["etf_val"] / df["balance"] * 100).mean()
    gp = (df["sgov_val"] / df["balance"] * 100).mean()
    sp = (df["stock_val"] / df["balance"] * 100).mean()
    print(f"  {lbl:<8}  株:{sp:>5.1f}%  ETF:{ep:>5.1f}%  SGOV:{gp:>5.1f}%  現金:{cp:>5.1f}%")


# ── チャート ──────────────────────────────────────────────────────
print("\nチャート生成中...")
spy_sim = spy_series[spy_series.index >= pd.Timestamp(SIM_START)]
spy_sim = spy_sim[spy_sim.index <= pd.Timestamp(SIM_END)]
spy_norm = spy_sim / spy_sim.iloc[0] * INITIAL

fig, axes = plt.subplots(3, 1, figsize=(16, 18), facecolor="#0d1117",
                         gridspec_kw={"height_ratios": [3, 2, 1.5], "hspace": 0.4})
fig.suptitle("ETF Overlay v2 (SPY 200MA Regime Filter + SGOV)  vs  v1  vs  Baseline\n"
             "risk=0.5%  trail4.0  |  Jan 2021 – Feb 2026",
             color="white", fontsize=13, fontweight="bold", y=0.99)

COLORS = {"base": "#00d4ff", "v1": "#f0a500", "v2": "#7ee787", "spy": "#888888"}

def style(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#adbac7", labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", lw=0.7, linestyle="--")
    for yr in range(2022, 2027):
        ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="#6e7681", lw=0.6, linestyle="--", alpha=0.4)

# Panel 1: 資産曲線
ax = axes[0]; style(ax)
ax.plot(df_v2["date"],   df_v2["balance"],   color=COLORS["v2"],   lw=2.5,
        label=f"ETF v2 (SPY200MA+SGOV)  ${s_weekly['final']:,.0f}  ({s_weekly['ret']:+.1f}%)")
ax.plot(df_v1["date"],   df_v1["balance"],   color=COLORS["v1"],   lw=1.8, linestyle="--",
        label=f"ETF v1 (現行)            ${s_monthly['final']:,.0f}  ({s_monthly['ret']:+.1f}%)")
ax.plot(df_base["date"], df_base["balance"], color=COLORS["base"], lw=1.5, linestyle=":",
        label=f"Baseline                 ${s_base['final']:,.0f}  ({s_base['ret']:+.1f}%)")
ax.plot(spy_norm.index,  spy_norm.values,    color=COLORS["spy"],  lw=1.2, alpha=0.6,
        label=f"SPY  ${spy_norm.iloc[-1]:,.0f}  ({(spy_norm.iloc[-1]/INITIAL-1)*100:+.1f}%)")
ax.axhline(INITIAL, color="#6e7681", lw=0.8, linestyle=":")
ax.set_title("Portfolio Value", color="white", fontsize=12, fontweight="bold")
ax.set_ylabel("Value ($)", color="#adbac7")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=10, loc="upper left")

# Panel 2: v2 配分スタック
ax2 = axes[1]; style(ax2)
ax2.stackplot(
    df_v2["date"],
    df_v2["stock_val"].values,
    df_v2["etf_val"].values,
    df_v2["sgov_val"].values,
    df_v2["cash"].values,
    labels=["Stocks", "Sector ETF", "SGOV", "Cash"],
    colors=["#00d4ff", "#f0a500", "#7ee787", "#555555"],
    alpha=0.75,
)
ax2.set_title("ETF v2  Capital Allocation  (Stocks / Sector ETF / SGOV / Cash)",
              color="white", fontsize=12, fontweight="bold")
ax2.set_ylabel("Amount ($)", color="#adbac7")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=9, loc="upper left")

# Panel 3: 年別リターン比較棒グラフ
ax3 = axes[2]; style(ax3)
yr_list = list(range(2021, 2027))
prev_b = prev_v1 = prev_v2 = INITIAL
yrets_b, yrets_v1, yrets_v2 = [], [], []
for yr in yr_list:
    for k, df, lst, prev in [
        ("b",  df_base, yrets_b,  prev_b),
        ("v1", df_v1,   yrets_v1, prev_v1),
        ("v2", df_v2,   yrets_v2, prev_v2),
    ]:
        yd = df[df["date"].dt.year == yr]
        lst.append(0 if yd.empty else (float(yd.iloc[-1]["balance"]) - prev) / prev * 100)
    prev_b  = float(df_base[df_base["date"].dt.year == yr].iloc[-1]["balance"]) if not df_base[df_base["date"].dt.year == yr].empty else prev_b
    prev_v1 = float(df_v1[df_v1["date"].dt.year == yr].iloc[-1]["balance"])   if not df_v1[df_v1["date"].dt.year == yr].empty else prev_v1
    prev_v2 = float(df_v2[df_v2["date"].dt.year == yr].iloc[-1]["balance"])   if not df_v2[df_v2["date"].dt.year == yr].empty else prev_v2

x = np.arange(len(yr_list)); w = 0.25
ax3.bar(x - w, yrets_b,  width=w, color=COLORS["base"], alpha=0.8, label="Baseline")
ax3.bar(x,     yrets_v1, width=w, color=COLORS["v1"],  alpha=0.8, label="v2月次")
ax3.bar(x + w, yrets_v2, width=w, color=COLORS["v2"],  alpha=0.8, label="v2週次")
ax3.axhline(0, color="#6e7681", lw=0.8)
ax3.set_xticks(x); ax3.set_xticklabels([str(y) for y in yr_list], color="#adbac7")
ax3.set_ylabel("Return (%)", color="#adbac7")
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
ax3.set_title("Annual Return Comparison", color="white", fontsize=12, fontweight="bold")
ax3.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=9)
for i, v in enumerate(yrets_v2):
    ax3.text(i + w, v + (1 if v >= 0 else -2.5), f"{v:+.0f}%",
             ha="center", va="bottom" if v >= 0 else "top", color=COLORS["v2"], fontsize=8)

out = BASE / "output/backtest/etf_v2_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"チャート保存: {out}")
