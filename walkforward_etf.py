"""
ETF v2 Weekly  Walk-Forward Test
アンカー型展開ウィンドウ（5フォールド）

固定パラメータ:
  - ETF_TOP_N  = 2
  - ETF_MA     = 50
  - REBAL_FREQ = "weekly"
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

# ── パス・定数 ─────────────────────────────────────────────────────
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
SPY_TREND_MA = 200
SAFE_TICKER  = "SGOV"

MOMENTUM_PERIODS = {
    "1M":  (21,  0.30),
    "3M":  (63,  0.30),
    "6M":  (126, 0.30),
    "12M": (252, 0.10),
}

DATA_START = "2019-10-01"
SIM_END    = "2026-02-28"

# ── フォールド定義 ──────────────────────────────────────────────────
FOLDS = [
    {"name": "Fold1", "train_start": "2021-01-04", "train_end": "2021-12-31", "test_start": "2022-01-03", "test_end": "2022-12-30"},
    {"name": "Fold2", "train_start": "2021-01-04", "train_end": "2022-12-30", "test_start": "2023-01-03", "test_end": "2023-12-29"},
    {"name": "Fold3", "train_start": "2021-01-04", "train_end": "2023-12-29", "test_start": "2024-01-02", "test_end": "2024-12-31"},
    {"name": "Fold4", "train_start": "2021-01-04", "train_end": "2024-12-31", "test_start": "2025-01-02", "test_end": "2025-12-31"},
    {"name": "Fold5", "train_start": "2021-01-04", "train_end": "2025-12-31", "test_start": "2026-01-02", "test_end": "2026-02-28"},
]

# ── 固定パラメータ ─────────────────────────────────────────────────
FIXED_TOP_N = 2
FIXED_MA    = 50
FIXED_FREQ  = "weekly"
print(f"固定パラメータ: TOP_N={FIXED_TOP_N}  MA={FIXED_MA}  REBAL_FREQ={FIXED_FREQ}")
print(f"フォールド数: {len(FOLDS)}")


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
same_day_global = {i for i, t in enumerate(trades_raw) if t["entry_date"] == t["exit_date"]}


# ── ETFデータ取得（1回だけ） ──────────────────────────────────────
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


# ── ETF補助関数 ─────────────────────────────────────────────────────
def etf_price(ticker: str, date: pd.Timestamp) -> float | None:
    s = etf_close.get(ticker)
    if s is None: return None
    past = s[s.index <= date]
    return float(past.iloc[-1]) if len(past) > 0 else None


def etf_ma(ticker: str, date: pd.Timestamp, period: int) -> float | None:
    s = etf_close.get(ticker)
    if s is None: return None
    past = s[s.index <= date]
    return float(past.iloc[-period:].mean()) if len(past) >= period else None


def spy_regime(date: pd.Timestamp) -> bool:
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


def select_target_etfs(date: pd.Timestamp, scored: list[tuple[str, float]],
                       etf_top_n: int, etf_ma_period: int) -> list[str]:
    result = []
    for ticker, _ in scored:
        if len(result) >= etf_top_n: break
        p = etf_price(ticker, date)
        ma = etf_ma(ticker, date, etf_ma_period)
        if p and ma and p >= ma:
            result.append(ticker)
    return result


def get_period_key(date: pd.Timestamp, rebal_freq: str) -> tuple:
    if rebal_freq == "weekly":
        return (date.year, date.isocalendar()[1])
    elif rebal_freq == "daily":
        return (date.year, date.month, date.day)
    else:
        return (date.year, date.month)


# ── シミュレーション本体（日付範囲・パラメータを受け取る） ──────────
def run_simulation(
    sim_start: str,
    sim_end: str,
    etf_top_n: int = 2,
    etf_ma_period: int = 50,
    rebal_freq: str = "weekly",
) -> dict:
    bdays = pd.bdate_range(sim_start, sim_end)
    if len(bdays) == 0:
        return {"sharpe": 0.0, "calmar": 0.0, "total_return": 0.0,
                "max_dd": 0.0, "final": INITIAL, "daily": []}

    # 対象期間のトレードインデックスをフィルタ
    sim_s = pd.Timestamp(sim_start)
    sim_e = pd.Timestamp(sim_end)
    period_indices = [
        i for i, t in enumerate(trades_raw)
        if pd.Timestamp(t["entry_date"]) >= sim_s
        and pd.Timestamp(t["entry_date"]) <= sim_e
    ]
    same_day_local = {i for i in period_indices if i in same_day_global}

    entries_by_date: dict[str, list[int]] = {}
    exits_by_date:   dict[str, list[int]] = {}
    for i in period_indices:
        t = trades_raw[i]
        entries_by_date.setdefault(t["entry_date"], []).append(i)
        exits_by_date.setdefault(t["exit_date"],   []).append(i)
    # exitが期間外にある場合も追加（エントリが期間内のトレード）
    for i in period_indices:
        t = trades_raw[i]
        if pd.Timestamp(t["exit_date"]) > sim_e:
            exits_by_date.setdefault(t["exit_date"], []).append(i)

    cash      = INITIAL
    stock_pos = {}
    etf_pos   = {}
    peak      = INITIAL
    max_dd    = 0.0

    daily_records = []
    pe: set[int] = set()
    px: set[int] = set()

    prev_period = None
    current_scores: list[tuple[str, float]] = []
    prev_bull = None

    for date in bdays:
        date_str = date.strftime("%Y-%m-%d")
        bull     = spy_regime(date)
        reg_chg  = (prev_bull is not None and bull != prev_bull)

        # ① リバランス
        period_key = get_period_key(date, rebal_freq)
        if period_key != prev_period:
            prev_period = period_key
            if bull:
                current_scores = calc_momentum_scores(date)
                targets = select_target_etfs(date, current_scores, etf_top_n, etf_ma_period)
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

        # ② 日次: レジーム変化・50MA割れ
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
                p = etf_price(tk, date)
                ma = etf_ma(tk, date, etf_ma_period)
                if p and ma and p < ma:
                    cash += etf_pos[tk] * p; del etf_pos[tk]
        elif bull:
            for tk in list(etf_pos.keys()):
                if tk == SAFE_TICKER: continue
                p = etf_price(tk, date)
                ma = etf_ma(tk, date, etf_ma_period)
                if p and ma and p < ma:
                    cash += etf_pos[tk] * p; del etf_pos[tk]

        prev_bull = bull

        # ③ 個別株EXIT
        for idx in exits_by_date.get(date_str, []):
            if idx in same_day_local: continue
            if idx in pe and idx not in px:
                px.add(idx)
                t   = trades_raw[idx]
                amt = stock_pos.pop(idx, 0)
                pnl = amt * (t["pl_pct"] / 100)
                cash += amt + pnl

        # ④ 個別株ENTRY
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

            if cash < need and etf_pos:
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

            if idx in same_day_local:
                px.add(idx)
                cash += alloc + alloc * (t["pl_pct"] / 100)
                stock_pos.pop(idx, None)

        # ⑤ 日次スナップショット
        etf_val   = sum(etf_pos[tk] * (etf_price(tk, date) or 0) for tk in etf_pos)
        stock_val = sum(stock_pos.values())
        total_eq  = cash + stock_val + etf_val
        if total_eq > peak: peak = total_eq
        dd = (peak - total_eq) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd

        daily_records.append({"date": date, "balance": total_eq})

    df = pd.DataFrame(daily_records)
    final = float(df.iloc[-1]["balance"])
    total_return = (final - INITIAL) / INITIAL * 100

    r = df.set_index("date")["balance"].pct_change().dropna()
    sharpe = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0.0
    calmar = total_return / max_dd if max_dd > 0 else 0.0

    return {
        "sharpe": sharpe,
        "calmar": calmar,
        "total_return": total_return,
        "max_dd": max_dd,
        "final": final,
        "daily": daily_records,
    }


# ── WFTメインループ ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  ETF v2 Walk-Forward Test  (anchored expanding window)")
print(f"  固定パラメータ: TOP_N={FIXED_TOP_N}  MA={FIXED_MA}  REBAL_FREQ={FIXED_FREQ}")
print("=" * 70)

fold_results = []

for fold in FOLDS:
    name = fold["name"]
    print(f"\n[{name}] Train: {fold['train_start'][:4]}–{fold['train_end'][:4]}  "
          f"/  Test: {fold['test_start'][:4]}–{fold['test_end'][:4]}")

    train_res = run_simulation(
        sim_start=fold["train_start"],
        sim_end=fold["train_end"],
        etf_top_n=FIXED_TOP_N,
        etf_ma_period=FIXED_MA,
        rebal_freq=FIXED_FREQ,
    )
    print(f"  Train  Sharpe={train_res['sharpe']:.3f}  "
          f"Return={train_res['total_return']:+.1f}%  MaxDD={train_res['max_dd']:.1f}%")

    test_res = run_simulation(
        sim_start=fold["test_start"],
        sim_end=fold["test_end"],
        etf_top_n=FIXED_TOP_N,
        etf_ma_period=FIXED_MA,
        rebal_freq=FIXED_FREQ,
    )
    print(f"  Test   Sharpe={test_res['sharpe']:.3f}  "
          f"Return={test_res['total_return']:+.1f}%  "
          f"MaxDD={test_res['max_dd']:.1f}%  "
          f"Calmar={test_res['calmar']:.2f}")

    fold_results.append({
        "fold": name,
        "train": f"{fold['train_start'][:4]}–{fold['train_end'][:4]}",
        "test":  f"{fold['test_start'][:4]}–{fold['test_end'][:4]}",
        "best_params":  {"top_n": FIXED_TOP_N, "ma": FIXED_MA, "freq": FIXED_FREQ},
        "train_sharpe": train_res["sharpe"],
        "train_ret":    train_res["total_return"],
        "test_sharpe":  test_res["sharpe"],
        "test_ret":     test_res["total_return"],
        "test_calmar":  test_res["calmar"],
        "test_maxdd":   test_res["max_dd"],
        "test_daily":   test_res["daily"],
        "test_final":   test_res["final"],
    })


# ── OOS 連結資産曲線 ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("  OOS 連結結果")
print("=" * 70)

oos_records = []
running_balance = INITIAL

for fr in fold_results:
    daily = fr["test_daily"]
    if not daily: continue
    fold_final_raw = daily[-1]["balance"]
    scale = running_balance / INITIAL
    for rec in daily:
        oos_records.append({
            "date":    rec["date"],
            "balance": rec["balance"] * scale,
        })
    running_balance = fold_final_raw * scale

oos_df = pd.DataFrame(oos_records).sort_values("date")

oos_final  = running_balance
oos_ret    = (oos_final - INITIAL) / INITIAL * 100

print(f"\n  OOS 全期間（2022-2026）:")
print(f"    初期資産   : ${INITIAL:,.2f}")
print(f"    最終資産   : ${oos_final:,.2f}")
print(f"    OOSリターン: {oos_ret:+.1f}%")

# OOS Sharpe / MaxDD
r_oos = oos_df.set_index("date")["balance"].pct_change().dropna()
sh_oos = (r_oos.mean() / r_oos.std()) * np.sqrt(252) if r_oos.std() > 0 else 0
peak_oos = oos_df["balance"].cummax()
dd_oos = ((oos_df["balance"] - peak_oos) / peak_oos * 100).min()
yrs_oos = (oos_df["date"].iloc[-1] - oos_df["date"].iloc[0]).days / 365.25
cagr_oos = (oos_final / INITIAL) ** (1 / yrs_oos) - 1 if yrs_oos > 0 else 0
calmar_oos = oos_ret / abs(dd_oos) if dd_oos != 0 else 0

print(f"    CAGR       : {cagr_oos*100:+.2f}%")
print(f"    Sharpe     : {sh_oos:.3f}")
print(f"    最大DD     : {dd_oos:.2f}%")
print(f"    Calmar     : {calmar_oos:.2f}")

# フォールド別サマリー
print(f"\n  {'Fold':<7} {'Train':>11} {'Test':>11} {'TrainSh':>9} {'TestSh':>9} "
      f"{'TestRet':>9} {'MaxDD':>8} {'Calmar':>8} {'TOP_N':>6} {'MA':>5} {'freq':>8}")
print("  " + "-" * 95)
for fr in fold_results:
    bp = fr["best_params"]
    print(f"  {fr['fold']:<7} {fr['train']:>11} {fr['test']:>11} "
          f"{fr['train_sharpe']:>9.3f} {fr['test_sharpe']:>9.3f} "
          f"{fr['test_ret']:>+8.1f}% {fr['test_maxdd']:>7.1f}% "
          f"{fr['test_calmar']:>8.2f} "
          f"{bp['top_n']:>6} {bp['ma']:>5} {bp['freq']:>8}")
print("  " + "-" * 95)
avg_sh = np.mean([fr["test_sharpe"] for fr in fold_results])
avg_ret = np.mean([fr["test_ret"] for fr in fold_results])
print(f"  {'Avg':<7} {'':>11} {'':>11} {'':>9} {avg_sh:>9.3f} {avg_ret:>+8.1f}%")

# ── SPY / QQQ ベンチマーク ────────────────────────────────────────
oos_start = "2022-01-03"
oos_end   = "2026-02-28"
bm_raw = yf.download(["SPY", "QQQ"], start=oos_start, end=oos_end,
                      auto_adjust=True, progress=False)
spy_bm = bm_raw["Close"]["SPY"].dropna()
qqq_bm = bm_raw["Close"]["QQQ"].dropna()
spy_norm = spy_bm / spy_bm.iloc[0] * INITIAL
qqq_norm = qqq_bm / qqq_bm.iloc[0] * INITIAL
spy_ret  = (spy_bm.iloc[-1] / spy_bm.iloc[0] - 1) * 100
qqq_ret  = (qqq_bm.iloc[-1] / qqq_bm.iloc[0] - 1) * 100

print(f"\n  比較（OOS期間 2022-2026）:")
print(f"    OOS ETF v2 WFT : {oos_ret:+.1f}%  Sharpe={sh_oos:.3f}")
print(f"    SPY            : {spy_ret:+.1f}%")
print(f"    QQQ            : {qqq_ret:+.1f}%")


# ── チャート ──────────────────────────────────────────────────────
print("\nチャート生成中...")
fig = plt.figure(figsize=(16, 22), facecolor="#0d1117")
fig.suptitle(
    "ETF v2 Weekly  Walk-Forward Test  (OOS: 2022–2026)\n"
    f"Anchored Expanding Window  |  Fixed: TOP_N={FIXED_TOP_N}  MA={FIXED_MA}  REBAL_FREQ={FIXED_FREQ}",
    color="white", fontsize=14, fontweight="bold", y=0.99,
)
gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.35,
                      top=0.94, bottom=0.04, left=0.08, right=0.97)

COLORS = {"oos": "#00d4ff", "train": "#7ee787", "spy": "#f0a500", "qqq": "#ff6b6b"}

def style_ax(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#adbac7")
    ax.xaxis.label.set_color("#adbac7")
    ax.yaxis.label.set_color("#adbac7")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", linewidth=0.7, linestyle="--")


# Panel 1: OOS資産曲線
ax1 = fig.add_subplot(gs[0, :])
style_ax(ax1)

date_range = pd.bdate_range(oos_df["date"].min(), oos_df["date"].max())
oos_ser = oos_df.set_index("date")["balance"]
oos_ser = oos_ser[~oos_ser.index.duplicated(keep="last")]
oos_ser = oos_ser.reindex(date_range).ffill().fillna(INITIAL)

spy_a = spy_norm.reindex(date_range).ffill()
qqq_a = qqq_norm.reindex(date_range).ffill()

ax1.plot(oos_ser.index, oos_ser,
         color=COLORS["oos"], lw=2.5,
         label=f"OOS ETF v2 WFT  ${oos_final:,.0f} ({oos_ret:+.1f}%)", zorder=3)
ax1.plot(spy_a.index, spy_a,
         color=COLORS["spy"], lw=1.5, alpha=0.8,
         label=f"SPY  ${spy_norm.iloc[-1]:,.0f} ({spy_ret:+.1f}%)")
ax1.plot(qqq_a.index, qqq_a,
         color=COLORS["qqq"], lw=1.5, alpha=0.8,
         label=f"QQQ  ${qqq_norm.iloc[-1]:,.0f} ({qqq_ret:+.1f}%)")
ax1.axhline(INITIAL, color="#6e7681", lw=0.8, linestyle=":")

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


# Panel 2: 累積リターン
ax2 = fig.add_subplot(gs[1, :])
style_ax(ax2)

oos_pct = (oos_ser / INITIAL - 1) * 100
spy_pct = (spy_a / INITIAL - 1) * 100
qqq_pct = (qqq_a / INITIAL - 1) * 100

ax2.plot(oos_ser.index, oos_pct, color=COLORS["oos"], lw=2, label="OOS ETF v2 WFT", zorder=3)
ax2.plot(spy_a.index, spy_pct, color=COLORS["spy"], lw=1.5, label="SPY")
ax2.plot(qqq_a.index, qqq_pct, color=COLORS["qqq"], lw=1.5, label="QQQ")
ax2.axhline(0, color="#6e7681", lw=0.8, linestyle=":")
ax2.fill_between(oos_ser.index, oos_pct, 0,
                 where=(oos_pct > 0), alpha=0.1, color=COLORS["oos"])
ax2.fill_between(oos_ser.index, oos_pct, 0,
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


# Panel 3: Train vs Test Sharpe
ax3 = fig.add_subplot(gs[2, 0])
style_ax(ax3)
fold_names  = [fr["fold"] for fr in fold_results]
train_shs   = [fr["train_sharpe"] for fr in fold_results]
test_shs    = [fr["test_sharpe"]  for fr in fold_results]
x = np.arange(len(fold_names)); w = 0.35
ax3.bar(x - w/2, train_shs, w, label="Train Sharpe", color=COLORS["train"], alpha=0.85)
ax3.bar(x + w/2, test_shs,  w, label="Test Sharpe",  color=COLORS["oos"],   alpha=0.85)
ax3.axhline(0, color="#f0a500", lw=1.2, linestyle="--")
ax3.set_xticks(x); ax3.set_xticklabels(fold_names, color="#adbac7")
ax3.set_title("Train vs Test  Sharpe Ratio", color="white", fontsize=12, fontweight="bold")
ax3.set_ylabel("Sharpe", color="#adbac7")
ax3.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=9)


# Panel 4: フォールド別リターン
ax4 = fig.add_subplot(gs[2, 1])
style_ax(ax4)
test_rets = [fr["test_ret"] for fr in fold_results]
cols = [COLORS["oos"] if r >= 0 else "#ff6b6b" for r in test_rets]
bars = ax4.bar(fold_names, test_rets, color=cols, alpha=0.85)
ax4.axhline(0, color="#6e7681", lw=0.8)
for bar, val in zip(bars, test_rets):
    ax4.text(bar.get_x() + bar.get_width()/2, val + (1 if val >= 0 else -2.5),
             f"{val:+.1f}%", ha="center", va="bottom", color="white", fontsize=9)
ax4.set_title("Test Period Return (%) per Fold", color="white", fontsize=12, fontweight="bold")
ax4.set_ylabel("Return (%)", color="#adbac7")
ax4.tick_params(colors="#adbac7")


# Panel 5: 最良パラメータ推移
ax5 = fig.add_subplot(gs[3, 0])
style_ax(ax5)
top_ns = [fr["best_params"]["top_n"] for fr in fold_results]
mas    = [fr["best_params"]["ma"]    for fr in fold_results]
freqs  = [fr["best_params"]["freq"]  for fr in fold_results]
x = np.arange(len(fold_names))
ax5_r = ax5.twinx()
ax5.bar(x - 0.2, top_ns, 0.4, label="ETF_TOP_N", color=COLORS["train"], alpha=0.7)
ax5_r.plot(x, mas, "o--", color="#f0a500", lw=1.5, label="ETF_MA")
ax5.set_xticks(x); ax5.set_xticklabels(fold_names, color="#adbac7")
ax5.set_title("Optimal Parameters per Fold", color="white", fontsize=12, fontweight="bold")
ax5.set_ylabel("ETF_TOP_N", color=COLORS["train"])
ax5_r.set_ylabel("ETF_MA", color="#f0a500")
ax5_r.tick_params(colors="#f0a500")
ax5.tick_params(colors="#adbac7")
for i, f in enumerate(freqs):
    ax5.text(i, -0.3, f, ha="center", va="top",
             color="#adbac7", fontsize=8, transform=ax5.get_xaxis_transform())
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5_r.get_legend_handles_labels()
ax5.legend(lines1+lines2, labels1+labels2, facecolor="#21262d",
           edgecolor="#30363d", labelcolor="white", fontsize=8, loc="upper left")


# Panel 6: サマリーテーブル
ax6 = fig.add_subplot(gs[3, 1])
style_ax(ax6); ax6.axis("off")

r_spy_bm = spy_norm.pct_change().dropna()
sh_spy = (r_spy_bm.mean() / r_spy_bm.std()) * np.sqrt(252) if r_spy_bm.std() > 0 else 0
dd_spy = ((spy_norm - spy_norm.cummax()) / spy_norm.cummax() * 100).min()

r_qqq_bm = qqq_norm.pct_change().dropna()
sh_qqq = (r_qqq_bm.mean() / r_qqq_bm.std()) * np.sqrt(252) if r_qqq_bm.std() > 0 else 0
dd_qqq = ((qqq_norm - qqq_norm.cummax()) / qqq_norm.cummax() * 100).min()

col_labels = ["Metric", "OOS WFT", "SPY", "QQQ"]
rows = [
    ["Period",       "2022–2026",        "2022–2026",       "2022–2026"],
    ["Total Return", f"{oos_ret:+.1f}%", f"{spy_ret:+.1f}%", f"{qqq_ret:+.1f}%"],
    ["Final Value",  f"${oos_final:,.0f}", f"${spy_norm.iloc[-1]:,.0f}", f"${qqq_norm.iloc[-1]:,.0f}"],
    ["CAGR",         f"{cagr_oos*100:+.1f}%", "-", "-"],
    ["Max Drawdown", f"{dd_oos:.1f}%",   f"{dd_spy:.1f}%",  f"{dd_qqq:.1f}%"],
    ["Sharpe Ratio", f"{sh_oos:.3f}",    f"{sh_spy:.3f}",   f"{sh_qqq:.3f}"],
    ["Calmar",       f"{calmar_oos:.2f}", "-", "-"],
    ["Folds ≥0",    f"{sum(1 for fr in fold_results if fr['test_ret']>=0)}/{len(fold_results)}", "-", "-"],
]

tbl = ax6.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.0, 1.55)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#21262d")
    cell.set_edgecolor("#30363d")
    cell.set_text_props(color="white")
    if r == 0:
        cell.set_facecolor("#1f6feb")
        cell.set_text_props(color="white", fontweight="bold")
    elif c == 1:
        cell.set_text_props(color=COLORS["oos"], fontweight="bold")
    elif c == 2:
        cell.set_text_props(color=COLORS["spy"])
    elif c == 3:
        cell.set_text_props(color=COLORS["qqq"])
ax6.set_title("OOS Performance Summary", color="white", fontsize=12, fontweight="bold", pad=10)


out_path = BASE / "output/backtest/walkforward_etf_chart.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"\nチャート保存: {out_path}")
print("完了")
