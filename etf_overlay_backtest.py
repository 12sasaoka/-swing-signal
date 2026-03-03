"""
ETFオーバーレイ バックテスト

個別株スイング戦略 + ETFモメンタムオーバーレイ（待機資金活用）

ETFルール:
  ユニバース : SPY QQQ XLC XLY XLP XLE XLF XLV XLI XLB XLK XLRE XLU
  スコアリング: 1M×0.30 + 3M×0.30 + 6M×0.30 + 12M×0.10（順位ベース）
  保有       : 上位2本（各ETFが50MA以上のみ）
  リバランス : 毎月第1営業日（全待機資金を均等配分）
  50MA割れ   : 即日売却 → 翌月リバランスまでキャッシュ
  全売り     : なし
  個別株資金 : スコア弱い方のETFから必要額だけ売却
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

INITIAL    = 4000.0
RISK       = 0.005
ALLOC      = 0.13
SB_ALLOC   = 0.14
MAX_POS    = 11
MIN_RR     = 0.005

ETF_UNIVERSE = [
    "SPY","QQQ","XLC","XLY","XLP",
    "XLE","XLF","XLV","XLI","XLB",
    "XLK","XLRE","XLU",
]
ETF_TOP_N  = 2
ETF_MA     = 50   # 50日MA フィルター

# モメンタム期間: {名前: (取引日数, 重み)}
MOMENTUM_PERIODS = {
    "1M":  (21,  0.30),
    "3M":  (63,  0.30),
    "6M":  (126, 0.30),
    "12M": (252, 0.10),
}

SIM_START  = "2021-01-04"
SIM_END    = "2026-02-28"
DATA_START = "2019-10-01"  # 12Mモメンタム計算のため早めに取得


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
print(f"ETFデータ取得中（{len(ETF_UNIVERSE)}本）...")
raw_all = yf.download(ETF_UNIVERSE, start=DATA_START, end=SIM_END,
                      auto_adjust=True, progress=False)

etf_close: dict[str, pd.Series] = {}
for ticker in ETF_UNIVERSE:
    try:
        if isinstance(raw_all.columns, pd.MultiIndex):
            s = raw_all["Close"][ticker].dropna()
        else:
            s = raw_all["Close"].dropna()
        etf_close[ticker] = s
    except Exception as e:
        print(f"  Warning: {ticker} 取得失敗 - {e}")

print(f"  取得完了: {list(etf_close.keys())}")

# SPYベースライン用
spy_series = etf_close.get("SPY")


# ── ETF補助関数 ────────────────────────────────────────────────────
def etf_price(ticker: str, date: pd.Timestamp) -> float | None:
    s = etf_close.get(ticker)
    if s is None:
        return None
    past = s[s.index <= date]
    return float(past.iloc[-1]) if len(past) > 0 else None


def etf_ma(ticker: str, date: pd.Timestamp, period: int = ETF_MA) -> float | None:
    s = etf_close.get(ticker)
    if s is None:
        return None
    past = s[s.index <= date]
    return float(past.iloc[-period:].mean()) if len(past) >= period else None


def calc_momentum_scores(date: pd.Timestamp) -> list[tuple[str, float]]:
    """全ETFのモメンタムスコアを計算。(ticker, score) リスト（スコア低い=強い）"""
    rets: dict[str, dict[str, float]] = {}
    for ticker in ETF_UNIVERSE:
        s = etf_close.get(ticker)
        if s is None:
            continue
        past = s[s.index <= date]
        if len(past) == 0:
            continue
        current = float(past.iloc[-1])
        ticker_rets = {}
        ok = True
        for name, (days, _) in MOMENTUM_PERIODS.items():
            if len(past) < days + 1:
                ok = False
                break
            base = float(past.iloc[-(days + 1)])
            if base <= 0:
                ok = False
                break
            ticker_rets[name] = (current - base) / base
        if ok:
            rets[ticker] = ticker_rets

    if len(rets) < 2:
        return []

    # 各期間で順位付け → 加重スコア合算
    scores: dict[str, float] = {t: 0.0 for t in rets}
    for name, (_, weight) in MOMENTUM_PERIODS.items():
        ranked = sorted(rets.keys(), key=lambda t: rets[t][name], reverse=True)
        for rank, ticker in enumerate(ranked, 1):
            scores[ticker] += rank * weight

    return sorted(scores.items(), key=lambda x: x[1])  # 低スコア = 強い


def select_target_etfs(date: pd.Timestamp,
                       scored: list[tuple[str, float]]) -> list[str]:
    """50MAフィルター適用後、上位N本を返す"""
    result = []
    for ticker, _ in scored:
        if len(result) >= ETF_TOP_N:
            break
        p  = etf_price(ticker, date)
        ma = etf_ma(ticker, date)
        if p is not None and ma is not None and p >= ma:
            result.append(ticker)
    return result


# ── シミュレーション本体 ───────────────────────────────────────────
def run_simulation(with_etf: bool) -> tuple[pd.DataFrame, list[dict], float]:
    """
    with_etf=True  → ETFオーバーレイあり
    with_etf=False → ベースライン（個別株のみ）
    Returns: (daily_df, trade_history, max_dd)
    """
    bdays = pd.bdate_range(SIM_START, SIM_END)

    cash       = INITIAL
    stock_pos  = {}   # trade_idx -> alloc ($)
    etf_pos    = {}   # ticker -> shares (小数可)
    peak       = INITIAL
    max_dd     = 0.0

    daily_records: list[dict]  = []
    trade_history: list[dict]  = []

    pe: set[int] = set()  # processed entries
    px: set[int] = set()  # processed exits

    prev_month     = None
    current_scores: list[tuple[str, float]] = []

    for date in bdays:
        date_str = date.strftime("%Y-%m-%d")

        # ─── ① 月初リバランス ───────────────────────────────────
        month_key = (date.year, date.month)
        if with_etf and month_key != prev_month:
            prev_month     = month_key
            current_scores = calc_momentum_scores(date)
            targets        = select_target_etfs(date, current_scores)

            # 非ターゲットETFを売却
            for tk in list(etf_pos.keys()):
                if tk not in targets:
                    p = etf_price(tk, date)
                    if p:
                        cash += etf_pos[tk] * p
                    del etf_pos[tk]

            # 全現金をターゲットETFへ均等配分（追加購入）
            if targets and cash > 1.0:
                alloc_each = cash / len(targets)
                for tk in targets:
                    p = etf_price(tk, date)
                    if p and p > 0:
                        shares = alloc_each / p
                        etf_pos[tk] = etf_pos.get(tk, 0.0) + shares
                        cash -= alloc_each

        # ─── ② 日次: 50MA割れチェック（即売り） ─────────────────
        if with_etf:
            for tk in list(etf_pos.keys()):
                p  = etf_price(tk, date)
                ma = etf_ma(tk, date)
                if p and ma and p < ma:
                    cash += etf_pos[tk] * p
                    del etf_pos[tk]

        # ─── ③ 個別株EXIT（同日を除く） ─────────────────────────
        for idx in exits_by_date.get(date_str, []):
            if idx in same_day:
                continue
            if idx in pe and idx not in px:
                px.add(idx)
                t   = trades_raw[idx]
                amt = stock_pos.pop(idx)
                pnl = amt * (t["pl_pct"] / 100)
                cash += amt + pnl

                etf_val = sum(etf_pos[tk] * (etf_price(tk, date) or 0)
                              for tk in etf_pos)
                bal = cash + sum(stock_pos.values()) + etf_val

                if bal > peak:
                    peak = bal
                dd = (peak - bal) / peak * 100 if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

                trade_history.append({
                    "date":    date,
                    "ticker":  t["ticker"],
                    "pl_pct":  t["pl_pct"],
                    "pnl":     pnl,
                    "balance": bal,
                })

        # ─── ④ 個別株ENTRY ───────────────────────────────────────
        etf_val   = sum(etf_pos[tk] * (etf_price(tk, date) or 0) for tk in etf_pos)
        total_eq  = cash + sum(stock_pos.values()) + etf_val

        for idx in entries_by_date.get(date_str, []):
            if idx in pe:
                continue
            if len(stock_pos) >= MAX_POS:
                continue

            t  = trades_raw[idx]
            rr = max((t["entry_price"] - t["sl_price"]) / t["entry_price"]
                     if t["entry_price"] > 0 else 0.08, MIN_RR)
            inv     = (total_eq * RISK) / rr
            ap      = SB_ALLOC if t["signal"] == "STRONG_BUY" else ALLOC
            need    = min(inv, total_eq * ap)

            # 現金不足 → 弱い方のETFから売る
            if with_etf and cash < need and etf_pos:
                shortage = need - cash
                # スコアが弱い（数値大きい）順にソート
                score_map = {tk: s for tk, s in current_scores}
                ranked = sorted(etf_pos.keys(),
                                key=lambda tk: score_map.get(tk, 999),
                                reverse=True)
                for tk in ranked:
                    if shortage <= 0:
                        break
                    p = etf_price(tk, date)
                    if not p:
                        continue
                    avail       = etf_pos[tk] * p
                    sell        = min(shortage, avail)
                    sell_shares = sell / p
                    etf_pos[tk] -= sell_shares
                    if etf_pos[tk] < 1e-9:
                        del etf_pos[tk]
                    cash     += sell
                    shortage -= sell

            alloc = min(need, cash)
            if t["entry_price"] > 0:
                sh    = math.floor(alloc / t["entry_price"])
                alloc = sh * t["entry_price"]
            else:
                sh = 0

            if alloc <= 0 or sh == 0 or cash <= 0:
                continue

            cash -= alloc
            stock_pos[idx] = alloc
            pe.add(idx)

            # 同日EXIT
            if idx in same_day:
                px.add(idx)
                pnl  = alloc * (t["pl_pct"] / 100)
                cash += alloc + pnl
                stock_pos.pop(idx, None)

        # ─── ⑤ 日次スナップショット ──────────────────────────────
        etf_val  = sum(etf_pos[tk] * (etf_price(tk, date) or 0) for tk in etf_pos)
        stock_val = sum(stock_pos.values())
        total_eq  = cash + stock_val + etf_val

        if total_eq > peak:
            peak = total_eq
        dd = (peak - total_eq) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

        daily_records.append({
            "date":      date,
            "balance":   total_eq,
            "cash":      cash,
            "stock_val": stock_val,
            "etf_val":   etf_val,
            "n_stocks":  len(stock_pos),
            "n_etfs":    len(etf_pos),
            "etf_names": "+".join(sorted(etf_pos.keys())) if etf_pos else "-",
        })

    df = pd.DataFrame(daily_records)
    return df, trade_history, max_dd


# ── 両方実行 ──────────────────────────────────────────────────────
print("\nベースライン シミュレーション実行中...")
df_base, hist_base, dd_base = run_simulation(with_etf=False)

print("ETFオーバーレイ シミュレーション実行中...")
df_etf, hist_etf, dd_etf = run_simulation(with_etf=True)


# ── サマリー計算 ──────────────────────────────────────────────────
def summary(df: pd.DataFrame, hist: list[dict], max_dd: float, label: str):
    final   = df.iloc[-1]["balance"]
    ret     = (final - INITIAL) / INITIAL * 100
    n_tr    = len(hist)
    wins    = sum(1 for h in hist if h["pnl"] > 0)
    gp      = sum(h["pnl"] for h in hist if h["pnl"] > 0)
    gl      = abs(sum(h["pnl"] for h in hist if h["pnl"] < 0))
    pf      = gp / gl if gl > 0 else 9.99
    wrate   = wins / n_tr * 100 if n_tr > 0 else 0
    print(f"\n=== {label} ===")
    print(f"  最終残高  : ${final:>10,.2f}")
    print(f"  リターン  : {ret:>+10.2f}%")
    print(f"  最大DD    : {max_dd:>10.2f}%")
    print(f"  取引数    : {n_tr:>10}")
    print(f"  勝率      : {wrate:>10.1f}%")
    print(f"  PF        : {pf:>10.3f}")
    return final, ret


final_base, ret_base = summary(df_base, hist_base, dd_base, "ベースライン（個別株のみ）")
final_etf,  ret_etf  = summary(df_etf,  hist_etf,  dd_etf,  "ETFオーバーレイ")

print(f"\n  差分リターン: {ret_etf - ret_base:+.2f}%")
print(f"  差分P/L    : ${final_etf - final_base:+,.2f}")

# ── SPYデータ ──────────────────────────────────────────────────────
spy_norm = None
if spy_series is not None:
    sim_spy = spy_series[spy_series.index >= pd.Timestamp(SIM_START)]
    sim_spy = sim_spy[sim_spy.index <= pd.Timestamp(SIM_END)]
    spy_norm = sim_spy / sim_spy.iloc[0] * INITIAL


# ── ETF保有推移の集計（月別） ─────────────────────────────────────
etf_monthly: dict[str, dict[str, float]] = {}
for _, row in df_etf.iterrows():
    ym = row["date"].strftime("%Y-%m")
    if ym not in etf_monthly:
        etf_monthly[ym] = {tk: 0 for tk in ETF_UNIVERSE}
        etf_monthly[ym]["CASH"] = 0
    names = row["etf_names"].split("+") if row["etf_names"] != "-" else []
    for tk in names:
        if tk in etf_monthly[ym]:
            etf_monthly[ym][tk] += 1
    if not names:
        etf_monthly[ym]["CASH"] += 1

# 月別支配ETF（最も多く保有された日数）
monthly_dominant: dict[str, str] = {}
for ym, counts in etf_monthly.items():
    dominant = max(counts, key=counts.get)
    monthly_dominant[ym] = dominant


# ── チャート ──────────────────────────────────────────────────────
print("\nチャート生成中...")

ETF_COLORS = {
    "SPY":  "#aaaaaa", "QQQ":  "#00d4ff", "XLC":  "#8ecae6",
    "XLY":  "#ff9f1c", "XLP":  "#2ec4b6", "XLE":  "#e9c46a",
    "XLF":  "#f4a261", "XLV":  "#a8dadc", "XLI":  "#457b9d",
    "XLB":  "#6a994e", "XLK":  "#7b2d8b", "XLRE": "#e63946",
    "XLU":  "#9b5de5", "CASH": "#444444", "-":    "#444444",
}

fig = plt.figure(figsize=(16, 20), facecolor="#0d1117")
fig.suptitle(
    "ETF Overlay vs Baseline  |  risk=0.5%  trail4.0  |  Jan 2021 - Feb 2026",
    color="white", fontsize=14, fontweight="bold", y=0.99,
)
gs = fig.add_gridspec(4, 1, hspace=0.42, top=0.96, bottom=0.04,
                      left=0.08, right=0.97,
                      height_ratios=[3, 2, 1.5, 1.5])

def style_ax(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#adbac7", labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", lw=0.7, linestyle="--")


# ─── Panel 1: 資産推移比較 ───────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
style_ax(ax1)

ax1.plot(df_etf["date"],  df_etf["balance"],
         color="#7ee787", lw=2.5,
         label=f"ETF Overlay  ${final_etf:,.0f}  ({ret_etf:+.1f}%)")
ax1.plot(df_base["date"], df_base["balance"],
         color="#00d4ff", lw=2.0, linestyle="--",
         label=f"Baseline     ${final_base:,.0f}  ({ret_base:+.1f}%)")
if spy_norm is not None:
    spy_ret_pct = (spy_norm.iloc[-1] - INITIAL) / INITIAL * 100
    ax1.plot(spy_norm.index, spy_norm.values,
             color="#888888", lw=1.3, alpha=0.7,
             label=f"SPY              ${spy_norm.iloc[-1]:,.0f}  ({spy_ret_pct:+.1f}%)")

ax1.axhline(INITIAL, color="#6e7681", lw=0.8, linestyle=":")
for yr in range(2022, 2027):
    ax1.axvline(pd.Timestamp(f"{yr}-01-01"), color="#6e7681",
                lw=0.6, linestyle="--", alpha=0.4)

# 年末バランス注記（ETFオーバーレイ）
eq_idx = df_etf.set_index("date")["balance"]
for yr in range(2021, 2026):
    yr_end = pd.Timestamp(f"{yr}-12-31")
    near   = eq_idx.index[eq_idx.index <= yr_end]
    if len(near) == 0:
        continue
    d = near[-1]; v = float(eq_idx[d])
    ax1.annotate(f"${v:,.0f}", xy=(d, v), xytext=(0, 14),
                 textcoords="offset points", color="#7ee787",
                 fontsize=8, ha="center",
                 arrowprops=dict(arrowstyle="-", color="#7ee787", lw=0.7))

ax1.set_title("Portfolio Value Comparison", color="white", fontsize=12, fontweight="bold")
ax1.set_ylabel("Value ($)", color="#adbac7")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white",
           fontsize=10, loc="upper left")


# ─── Panel 2: 資本配分（個別株・ETF・キャッシュ） ────────────────
ax2 = fig.add_subplot(gs[1])
style_ax(ax2)

ax2.stackplot(
    df_etf["date"],
    df_etf["stock_val"].values,
    df_etf["etf_val"].values,
    df_etf["cash"].values,
    labels=[
        f"Stocks  (avg ${df_etf['stock_val'].mean():,.0f})",
        f"ETF     (avg ${df_etf['etf_val'].mean():,.0f})",
        f"Cash    (avg ${df_etf['cash'].mean():,.0f})",
    ],
    colors=["#00d4ff", "#7ee787", "#f0a500"],
    alpha=0.75,
)
for yr in range(2022, 2027):
    ax2.axvline(pd.Timestamp(f"{yr}-01-01"), color="#6e7681",
                lw=0.6, linestyle="--", alpha=0.5)

ax2.set_title("Capital Allocation  (Stocks / ETF / Cash)", color="white",
              fontsize=12, fontweight="bold")
ax2.set_ylabel("Amount ($)", color="#adbac7")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white",
           fontsize=9, loc="upper left")


# ─── Panel 3: 保有ETF（月別ヒートマップ風） ─────────────────────
ax3 = fig.add_subplot(gs[2])
style_ax(ax3)

months   = sorted(monthly_dominant.keys())
dom_etfs = [monthly_dominant[m] for m in months]
month_ts = [pd.Timestamp(m + "-01") for m in months]
colors3  = [ETF_COLORS.get(e, "#555555") for e in dom_etfs]

for i, (ts, etf, col) in enumerate(zip(month_ts, dom_etfs, colors3)):
    ax3.bar(ts, 1, width=25, color=col, alpha=0.85)
    if i == 0 or dom_etfs[i] != dom_etfs[i - 1]:
        ax3.text(ts, 0.5, etf, color="white", fontsize=7,
                 ha="left", va="center", fontweight="bold")

ax3.set_yticks([])
ax3.set_title("Monthly ETF Holdings  (dominant position per month)", color="white",
              fontsize=12, fontweight="bold")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax3.xaxis.set_major_locator(mdates.YearLocator())
ax3.set_xlim(pd.Timestamp(SIM_START), pd.Timestamp(SIM_END))

# 凡例（保有されたETFのみ）
held_etfs = sorted(set(dom_etfs) - {"-"})
patches   = [plt.Rectangle((0, 0), 1, 1,
             color=ETF_COLORS.get(e, "#555555"), alpha=0.85)
             for e in held_etfs]
ax3.legend(patches, held_etfs, facecolor="#21262d", edgecolor="#30363d",
           labelcolor="white", fontsize=8, ncol=len(held_etfs),
           loc="lower right")


# ─── Panel 4: 年別リターン比較 ───────────────────────────────────
ax4 = fig.add_subplot(gs[3])
style_ax(ax4)

years = sorted(set(
    list(range(2021, 2027)) +
    [d.year for d in df_base["date"]]
))

def yearly_returns(df: pd.DataFrame) -> dict[int, float]:
    out = {}
    prev = INITIAL
    for yr in years:
        yr_df = df[df["date"].dt.year == yr]
        if yr_df.empty:
            continue
        end  = float(yr_df.iloc[-1]["balance"])
        ret  = (end - prev) / prev * 100
        out[yr] = ret
        prev = end
    return out

yr_base = yearly_returns(df_base)
yr_etf  = yearly_returns(df_etf)

yr_list = sorted(set(yr_base) | set(yr_etf))
x       = np.arange(len(yr_list))
w       = 0.35

bars1 = ax4.bar(x - w/2, [yr_base.get(y, 0) for y in yr_list],
                width=w, color="#00d4ff", alpha=0.8, label="Baseline")
bars2 = ax4.bar(x + w/2, [yr_etf.get(y, 0)  for y in yr_list],
                width=w, color="#7ee787", alpha=0.8, label="ETF Overlay")

ax4.axhline(0, color="#6e7681", lw=0.8)
ax4.set_xticks(x)
ax4.set_xticklabels([str(y) for y in yr_list], color="#adbac7")
ax4.set_ylabel("Return (%)", color="#adbac7")
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
ax4.set_title("Annual Return Comparison", color="white", fontsize=12, fontweight="bold")
ax4.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=9)

for bar, yr in zip(bars2, yr_list):
    v = yr_etf.get(yr, 0)
    ax4.text(bar.get_x() + bar.get_width() / 2,
             v + (1 if v >= 0 else -2),
             f"{v:+.0f}%",
             ha="center", va="bottom" if v >= 0 else "top",
             color="#7ee787", fontsize=8)


# ── 保存 ──────────────────────────────────────────────────────────
out_path = BASE / "output/backtest/etf_overlay_result.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"\nチャート保存: {out_path}")
