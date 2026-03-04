"""
Swing Signal Dashboard
streamlit run dashboard.py
"""
import csv, math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ── 設定 ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Swing Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent
CSV_DIR = BASE / "output/backtest"

INITIAL    = 4000.0
RISK       = 0.005
ALLOC      = 0.13
SB_ALLOC   = 0.14
MAX_POS    = 11
MIN_RR     = 0.005

RESULT_LABELS = {
    "sl_hit":        "SL Hit",
    "trailing_stop": "Trail Stop",
    "timeout_30d":   "TO-30d",
    "timeout_60d":   "TO-60d",
    "timeout_90d":   "TO-90d",
    "timeout":       "Timeout",
    "pre_earnings":  "Pre-Earn",
}

# ── サイドバー：CSVファイル選択 ────────────────────────────────────
st.sidebar.title("設定")
csv_files = sorted(CSV_DIR.glob("signal_backtest_*.csv"), reverse=True)
csv_names = [f.name for f in csv_files]

selected_csv = st.sidebar.selectbox("バックテストCSV", csv_names)
csv_path = CSV_DIR / selected_csv

signal_options = ["全件", "STRONG_BUY", "BUY"]
signal_filter  = st.sidebar.selectbox("シグナルフィルター", signal_options)

# ── シミュレーション（キャッシュ） ──────────────────────────────────
@st.cache_data(ttl=300)
def run_simulation(csv_file: str, sig_filter: str):
    path = CSV_DIR / csv_file
    trades_raw = []
    with open(path, "r", encoding="utf-8-sig") as f:
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
                "tp_price":    float(row["tp_price"]),
                "result":      row["result"],
            })

    if sig_filter != "全件":
        trades_raw = [t for t in trades_raw if t["signal"] == sig_filter]

    trades_raw = sorted(trades_raw, key=lambda t: t["entry_date"])
    same_day = {i for i, t in enumerate(trades_raw) if t["entry_date"] == t["exit_date"]}

    events = []
    for i, t in enumerate(trades_raw):
        events.append((t["entry_date"], "entry", i))
        events.append((t["exit_date"],  "exit",  i))

    def _key(e):
        d, et, i = e
        return (d, 2 if (et == "exit" and i in same_day) else (1 if et == "entry" else 0))

    events.sort(key=_key)

    cash = INITIAL
    active = {}
    peak = INITIAL
    max_dd = 0.0
    history = []
    pe, px = set(), set()

    for date, etype, idx in events:
        t = trades_raw[idx]
        if etype == "exit" and idx in pe and idx not in px:
            px.add(idx)
            amt = active.pop(idx)
            pnl = amt * (t["pl_pct"] / 100)
            cash += amt + pnl
            bal  = cash + sum(active.values())
            if bal > peak: peak = bal
            dd = (peak - bal) / peak * 100 if peak > 0 else 0
            if dd > max_dd: max_dd = dd
            history.append({
                "date":        date,
                "ticker":      t["ticker"],
                "signal":      t["signal"],
                "entry_date":  t["entry_date"],
                "entry_price": t["entry_price"],
                "exit_price":  t["exit_price"],
                "result":      RESULT_LABELS.get(t["result"], t["result"]),
                "pl_pct":      t["pl_pct"],
                "balance":     bal,
                "pnl":         pnl,
                "alloc":       amt,
                "cash":        cash,
                "invested":    sum(active.values()),
                "n_pos":       len(active),
            })
        elif etype == "entry" and idx not in pe:
            if len(active) >= MAX_POS:
                continue
            eq  = cash + sum(active.values())
            rr  = max((t["entry_price"] - t["sl_price"]) / t["entry_price"]
                      if t["entry_price"] > 0 else 0.08, MIN_RR)
            inv = (eq * RISK) / rr
            ap  = SB_ALLOC if t["signal"] == "STRONG_BUY" else ALLOC
            a   = min(inv, eq * ap, cash)
            if t["entry_price"] > 0:
                sh = math.floor(a / t["entry_price"])
                a  = sh * t["entry_price"]
            else:
                sh = 0
            if a <= 0 or sh == 0 or cash <= 0:
                continue
            cash -= a
            active[idx] = a
            pe.add(idx)

    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["date"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    return df, max_dd, peak


@st.cache_data(ttl=3600)
def get_benchmark(ticker: str, start: str, end: str):
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    s = (raw["Close"][ticker] if isinstance(raw.columns, pd.MultiIndex)
         else raw["Close"]).dropna()
    return s


def build_daily(df: pd.DataFrame):
    df_d = df.groupby("date").last().reset_index().sort_values("date")
    bdays = pd.bdate_range(df_d["date"].min(), df_d["date"].max())
    eq   = df_d.set_index("date")["balance"].reindex(bdays).ffill().fillna(INITIAL)
    cash = df_d.set_index("date")["cash"].reindex(bdays).ffill().fillna(INITIAL)
    inv  = df_d.set_index("date")["invested"].reindex(bdays).ffill().fillna(0)
    npos = df_d.set_index("date")["n_pos"].reindex(bdays).ffill().fillna(0)
    return eq, cash, inv, npos


@st.cache_data(ttl=3600)
def load_etf_daily(path_str: str):
    return pd.read_csv(path_str, parse_dates=["date"])


@st.cache_data(ttl=3600)
def load_etf_txn(path_str: str):
    return pd.read_csv(path_str, encoding="utf-8-sig")


# ── データ準備 ────────────────────────────────────────────────────
df, max_dd_base, peak_base = run_simulation(selected_csv, signal_filter)

if df.empty:
    st.error("取引データが見つかりません。")
    st.stop()

eq_s, cash_s, inv_s, npos_s = build_daily(df)

start_str = eq_s.index[0].strftime("%Y-%m-%d")
end_str   = eq_s.index[-1].strftime("%Y-%m-%d")

spy = get_benchmark("SPY", start_str, end_str)
qqq = get_benchmark("QQQ", start_str, end_str)
spy_norm = spy / spy.iloc[0] * INITIAL
qqq_norm = qqq / qqq.iloc[0] * INITIAL
spy_ret  = (float(spy.iloc[-1]) - float(spy.iloc[0])) / float(spy.iloc[0]) * 100
qqq_ret  = (float(qqq.iloc[-1]) - float(qqq.iloc[0])) / float(qqq.iloc[0]) * 100

# 個別株メトリクス
final_bal    = eq_s.iloc[-1]
total_ret    = (final_bal - INITIAL) / INITIAL * 100
n_tr         = len(df)
wins         = (df["pnl"] > 0).sum()
losses       = (df["pnl"] < 0).sum()
gp           = df.loc[df["pnl"] > 0, "pnl"].sum()
gl           = abs(df.loc[df["pnl"] < 0, "pnl"].sum())
pf           = gp / gl if gl > 0 else 9.99
winrate      = wins / n_tr * 100
avg_idle_pct = (cash_s / eq_s * 100).mean()
avg_dep_pct  = (inv_s  / eq_s * 100).mean()
r_base       = eq_s.pct_change().dropna()
sharpe_base  = (r_base.mean() / r_base.std()) * np.sqrt(252) if r_base.std() > 0 else 0
yrs_base     = (eq_s.index[-1] - eq_s.index[0]).days / 365.25
cagr_base    = ((final_bal / INITIAL) ** (1 / yrs_base) - 1) * 100
calmar_base  = total_ret / max_dd_base if max_dd_base > 0 else 0


# 年別統計
def yearly_stats(df_: pd.DataFrame):
    out = {}
    prev_bal = INITIAL
    for yr in sorted(df_["date"].dt.year.unique()):
        yr_df = df_[df_["date"].dt.year == yr]
        if yr_df.empty:
            continue
        end_bal = yr_df.iloc[-1]["balance"]
        yr_ret  = (end_bal - prev_bal) / prev_bal * 100
        yr_pnl  = yr_df["pnl"].sum()
        yr_wins = (yr_df["pnl"] > 0).sum()
        out[yr] = {"ret": yr_ret, "pnl": yr_pnl, "end": end_bal,
                   "trades": len(yr_df), "wins": yr_wins}
        prev_bal = end_bal
    return out


yr_stats_base = yearly_stats(df)

# ETF v2 データ
etf_daily_path = CSV_DIR / "etf_v2_weekly_daily.csv"
etf_txn_path   = CSV_DIR / "etf_v2_txn_history.csv"
etf_available  = etf_daily_path.exists()

if etf_available:
    df_etf_d   = load_etf_daily(str(etf_daily_path))
    final_etf  = float(df_etf_d.iloc[-1]["balance"])
    ret_etf    = (final_etf - INITIAL) / INITIAL * 100
    r_etf      = df_etf_d.set_index("date")["balance"].pct_change().dropna()
    sharpe_etf = (r_etf.mean() / r_etf.std()) * np.sqrt(252) if r_etf.std() > 0 else 0
    bal_arr    = df_etf_d["balance"].values
    peak_arr   = np.maximum.accumulate(bal_arr)
    max_dd_etf = float(((peak_arr - bal_arr) / peak_arr * 100).max())
    yrs_etf    = (df_etf_d["date"].iloc[-1] - df_etf_d["date"].iloc[0]).days / 365.25
    cagr_etf   = ((final_etf / INITIAL) ** (1 / yrs_etf) - 1) * 100
    calmar_etf = ret_etf / max_dd_etf if max_dd_etf > 0 else 0

    # ETF v2 年別リターン
    yr_rets_etf = {}
    prev_e = INITIAL
    for yr_ in range(2021, 2027):
        yd = df_etf_d[df_etf_d["date"].dt.year == yr_]
        if yd.empty:
            continue
        end_e = float(yd.iloc[-1]["balance"])
        yr_rets_etf[yr_] = (end_e - prev_e) / prev_e * 100
        prev_e = end_e
else:
    # フォールバック（ETF CSV 未生成時）
    df_etf_d   = None
    final_etf  = final_bal
    ret_etf    = total_ret
    cagr_etf   = cagr_base
    max_dd_etf = max_dd_base
    sharpe_etf = sharpe_base
    calmar_etf = calmar_base
    yr_rets_etf = {}

# ── ヘッダー ─────────────────────────────────────────────────────
st.title("Swing Signal ダッシュボード")
st.caption(
    f"ETF v2 weekly（個別株 + モメンタムETF複合）  |  CSV: `{selected_csv}`  |  "
    f"risk=0.5%  trail4.0  初期資金=${INITIAL:,.0f}"
)

# ── KPIカード (ETF v2 メイン) ─────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("最終残高",  f"${final_etf:,.0f}",   f"Base: ${final_bal:,.0f}")
c2.metric("リターン",  f"{ret_etf:+.1f}%",     f"SPY:{spy_ret:+.1f}%  QQQ:{qqq_ret:+.1f}%")
c3.metric("CAGR",      f"{cagr_etf:+.1f}%",    f"Base: {cagr_base:+.1f}%")
c4.metric("最大DD",    f"{max_dd_etf:.1f}%",   f"Base: {max_dd_base:.1f}%", delta_color="inverse")
c5.metric("Sharpe",    f"{sharpe_etf:.3f}",    f"Base: {sharpe_base:.3f}")
c6.metric("Calmar",    f"{calmar_etf:.2f}",    f"Base: {calmar_base:.2f}")

st.divider()

# ── タブ ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 資産推移", "🗂 資本配分", "📅 年別", "📋 取引一覧", "📊 サマリー"]
)

# ── Tab 1: 資産推移 ───────────────────────────────────────────────
with tab1:
    fig = go.Figure()

    if etf_available:
        fig.add_trace(go.Scatter(
            x=df_etf_d["date"], y=df_etf_d["balance"],
            name=f"ETF v2 weekly  ${final_etf:,.0f}  ({ret_etf:+.1f}%)",
            line=dict(color="#7ee787", width=2.5),
            fill="tozeroy", fillcolor="rgba(126,231,135,0.05)",
        ))

    fig.add_trace(go.Scatter(
        x=eq_s.index, y=eq_s.values,
        name=f"個別株のみ  ${final_bal:,.0f}  ({total_ret:+.1f}%)",
        line=dict(color="#00d4ff", width=1.8, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=spy_norm.index, y=spy_norm.values,
        name=f"SPY  ${spy_norm.iloc[-1]:,.0f}  ({spy_ret:+.1f}%)",
        line=dict(color="#888888", width=1.3, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=qqq_norm.index, y=qqq_norm.values,
        name=f"QQQ  ${qqq_norm.iloc[-1]:,.0f}  ({qqq_ret:+.1f}%)",
        line=dict(color="#f0a500", width=1.3, dash="dot"),
    ))

    # 年末アノテーション（ETF v2）
    if etf_available:
        for yr_ann in yr_rets_etf:
            yr_end = pd.Timestamp(f"{yr_ann}-12-31")
            mask = df_etf_d["date"] <= yr_end
            if mask.sum() == 0:
                continue
            last = df_etf_d[mask].iloc[-1]
            fig.add_annotation(
                x=last["date"], y=last["balance"],
                text=f"${last['balance']:,.0f}",
                showarrow=True, arrowhead=2, arrowcolor="#7ee787",
                font=dict(color="#7ee787", size=10), ax=0, ay=-35,
            )

    fig.add_hline(y=INITIAL, line_dash="dot", line_color="#6e7681",
                  annotation_text=f"${INITIAL:,.0f}", annotation_position="left")

    fig.update_layout(
        title="ETF v2 weekly vs 個別株 / SPY / QQQ（初期資金 $4,000）",
        xaxis_title="日付", yaxis_title="残高 ($)",
        yaxis_tickformat="$,.0f",
        hovermode="x unified",
        height=520,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4戦略比較テーブル
    cmp_df = pd.DataFrame({
        "戦略":    ["ETF v2 weekly", "個別株のみ", "SPY", "QQQ"],
        "最終残高": [f"${final_etf:,.0f}", f"${final_bal:,.0f}",
                    f"${spy_norm.iloc[-1]:,.0f}", f"${qqq_norm.iloc[-1]:,.0f}"],
        "リターン": [f"{ret_etf:+.1f}%", f"{total_ret:+.1f}%",
                    f"{spy_ret:+.1f}%", f"{qqq_ret:+.1f}%"],
        "CAGR":    [f"{cagr_etf:+.1f}%", f"{cagr_base:+.1f}%", "—", "—"],
        "最大DD":  [f"{max_dd_etf:.1f}%" if etf_available else "—",
                    f"{max_dd_base:.1f}%", "—", "—"],
        "Sharpe":  [f"{sharpe_etf:.3f}" if etf_available else "—",
                    f"{sharpe_base:.3f}", "—", "—"],
        "Calmar":  [f"{calmar_etf:.2f}" if etf_available else "—",
                    f"{calmar_base:.2f}", "—", "—"],
    })
    st.dataframe(cmp_df, use_container_width=True, hide_index=True)

# ── Tab 2: 資本配分 ─────────────────────────────────────────────
with tab2:
    if etf_available:
        avg_s  = df_etf_d["stock_val"].mean()
        avg_e  = df_etf_d["etf_val"].mean()
        avg_sg = df_etf_d["sgov_val"].mean()
        avg_c  = df_etf_d["cash"].mean()
        tot_   = avg_s + avg_e + avg_sg + avg_c

        col_a, col_b = st.columns([3, 1])

        with col_a:
            fig2 = go.Figure()
            for label, col_name, color in [
                (f"個別株 (avg ${avg_s:,.0f} / {avg_s/tot_*100:.1f}%)",      "stock_val", "rgba(0,212,255,0.65)"),
                (f"セクターETF (avg ${avg_e:,.0f} / {avg_e/tot_*100:.1f}%)", "etf_val",   "rgba(240,165,0,0.65)"),
                (f"SGOV (avg ${avg_sg:,.0f} / {avg_sg/tot_*100:.1f}%)",      "sgov_val",  "rgba(126,231,135,0.65)"),
                (f"現金 (avg ${avg_c:,.0f} / {avg_c/tot_*100:.1f}%)",        "cash",      "rgba(85,85,85,0.65)"),
            ]:
                fig2.add_trace(go.Scatter(
                    x=df_etf_d["date"], y=df_etf_d[col_name],
                    name=label, stackgroup="one",
                    fillcolor=color, line=dict(width=0),
                ))
            for yr in range(2022, 2027):
                fig2.add_vline(x=f"{yr}-01-01", line_dash="dot",
                               line_color="#444", line_width=1)
            fig2.update_layout(
                title="資本配分の推移（個別株 / セクターETF / SGOV / 現金）",
                yaxis_tickformat="$,.0f", hovermode="x unified", height=420,
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
            )
            st.plotly_chart(fig2, use_container_width=True)

            # 個別株 待機資金 % + ポジション数
            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
            cash_pct = cash_s / eq_s * 100
            fig3.add_trace(go.Scatter(
                x=eq_s.index, y=cash_pct.values,
                name=f"個別株待機 % (平均 {cash_pct.mean():.1f}%)",
                fill="tozeroy", fillcolor="rgba(240,165,0,0.25)",
                line=dict(color="#f0a500", width=1.5),
            ), secondary_y=False)
            fig3.add_trace(go.Scatter(
                x=eq_s.index, y=npos_s.values,
                name=f"保有銘柄数 (平均 {npos_s.mean():.1f})",
                line=dict(color="#c084fc", width=1.2),
            ), secondary_y=True)
            fig3.update_yaxes(title_text="待機資金 (%)", secondary_y=False,
                              tickformat=".0f", ticksuffix="%", range=[0, 100])
            fig3.update_yaxes(title_text="保有銘柄数", secondary_y=True,
                              range=[0, MAX_POS + 3])
            fig3.update_layout(
                title="個別株 待機資金 (%) & 保有銘柄数",
                hovermode="x unified", height=350,
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col_b:
            fig_pi = px.pie(
                values=[avg_s, avg_e, avg_sg, avg_c],
                names=["個別株", "セクターETF", "SGOV", "現金"],
                color_discrete_sequence=["#00d4ff", "#f0a500", "#7ee787", "#555555"],
                hole=0.4, title="全期間平均配分",
            )
            fig_pi.update_layout(height=280, margin=dict(t=40, b=10, l=5, r=5))
            st.plotly_chart(fig_pi, use_container_width=True)

            st.subheader("年別配分率")
            yr_cash_rows = {}
            for yr_ in range(2021, 2027):
                m = df_etf_d["date"].dt.year == yr_
                if m.sum() == 0:
                    continue
                yr_cash_rows[yr_] = {
                    "株式率":  f"{(df_etf_d.loc[m,'stock_val'] / df_etf_d.loc[m,'balance'] * 100).mean():.1f}%",
                    "ETF率":   f"{(df_etf_d.loc[m,'etf_val']  / df_etf_d.loc[m,'balance'] * 100).mean():.1f}%",
                    "SGOV率":  f"{(df_etf_d.loc[m,'sgov_val'] / df_etf_d.loc[m,'balance'] * 100).mean():.1f}%",
                    "現金率":  f"{(df_etf_d.loc[m,'cash']     / df_etf_d.loc[m,'balance'] * 100).mean():.1f}%",
                }
            st.dataframe(pd.DataFrame(yr_cash_rows).T, use_container_width=True)

    else:
        # フォールバック（ETF CSV 未生成時）
        st.warning("ETF v2データが未生成です。`python etf_overlay_backtest.py` を実行してください。")
        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=eq_s.index, y=inv_s.values,
                name=f"投資中 (平均 ${inv_s.mean():,.0f})",
                stackgroup="one",
                line=dict(color="#7ee787", width=0),
                fillcolor="rgba(126,231,135,0.6)",
            ))
            fig2.add_trace(go.Scatter(
                x=eq_s.index, y=cash_s.values,
                name=f"待機資金 (平均 ${cash_s.mean():,.0f})",
                stackgroup="one",
                line=dict(color="#f0a500", width=0),
                fillcolor="rgba(240,165,0,0.5)",
            ))
            fig2.update_layout(
                title="投資中資金 vs 待機資金（絶対額）",
                yaxis_tickformat="$,.0f", hovermode="x unified", height=400,
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
            )
            st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3: 年別 ───────────────────────────────────────────────────
with tab3:
    col_c, col_d = st.columns([3, 2])

    base_yr_rets = {yr_: s["ret"] for yr_, s in yr_stats_base.items()}
    spy_yr_rets, qqq_yr_rets = {}, {}
    for yr_ in range(2021, 2027):
        spy_yr = spy[spy.index.year == yr_]
        if not spy_yr.empty:
            spy_yr_rets[yr_] = (float(spy_yr.iloc[-1]) - float(spy_yr.iloc[0])) / float(spy_yr.iloc[0]) * 100
        qqq_yr = qqq[qqq.index.year == yr_]
        if not qqq_yr.empty:
            qqq_yr_rets[yr_] = (float(qqq_yr.iloc[-1]) - float(qqq_yr.iloc[0])) / float(qqq_yr.iloc[0]) * 100

    common_yrs = sorted(base_yr_rets.keys())

    with col_c:
        fig4 = go.Figure()
        traces = [
            ("ETF v2", yr_rets_etf, "#7ee787"),
            ("個別株", base_yr_rets, "#00d4ff"),
            ("SPY",    spy_yr_rets,  "#888888"),
            ("QQQ",    qqq_yr_rets,  "#f0a500"),
        ]
        if not etf_available:
            traces = traces[1:]
        for name_, rets_, color_ in traces:
            fig4.add_trace(go.Bar(
                x=[str(y) for y in common_yrs],
                y=[rets_.get(y, 0) for y in common_yrs],
                name=name_, marker_color=color_,
                text=[f"{rets_.get(y, 0):+.1f}%" for y in common_yrs],
                textposition="outside",
            ))
        fig4.add_hline(y=0, line_color="#888")
        fig4.update_layout(
            barmode="group",
            title="年別リターン比較（ETF v2 / 個別株 / SPY / QQQ）",
            yaxis_ticksuffix="%", height=420,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col_d:
        yr_tbl = []
        for yr_ in common_yrs:
            yr_tbl.append({
                "年":           yr_,
                "ETF v2":       f"{yr_rets_etf.get(yr_, 0):+.1f}%",
                "個別株":       f"{base_yr_rets.get(yr_, 0):+.1f}%",
                "SPY":          f"{spy_yr_rets.get(yr_, 0):+.1f}%",
                "QQQ":          f"{qqq_yr_rets.get(yr_, 0):+.1f}%",
                "超過(vs SPY)": f"{yr_rets_etf.get(yr_, 0) - spy_yr_rets.get(yr_, 0):+.1f}%",
            })
        st.subheader("年別リターン")
        st.dataframe(pd.DataFrame(yr_tbl), use_container_width=True, hide_index=True)

        yr_detail = []
        for yr_, s in yr_stats_base.items():
            yr_detail.append({
                "年":      yr_,
                "取引数":  s["trades"],
                "勝ち":    s["wins"],
                "勝率":    f"{s['wins']/s['trades']*100:.1f}%",
                "損益$":   f"${s['pnl']:+,.0f}",
                "年末残高": f"${s['end']:,.0f}",
            })
        st.subheader("個別株 年別詳細")
        st.dataframe(pd.DataFrame(yr_detail), use_container_width=True, hide_index=True)

# ── Tab 4: 取引一覧 ───────────────────────────────────────────────
with tab4:
    txn_sub1, txn_sub2 = st.tabs(["📋 個別株取引", "📊 ETF取引"])

    with txn_sub1:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            result_options = ["全件"] + sorted(df["result"].unique().tolist())
            result_filter_t4 = st.selectbox("終了理由フィルター", result_options, key="t4_res")
        with col_f2:
            year_options = ["全件"] + sorted(df["date"].dt.year.unique().astype(str).tolist())
            year_filter_t4 = st.selectbox("年", year_options, key="t4_yr")
        with col_f3:
            sort_col = st.selectbox("並び替え", ["date", "pl_pct", "pnl", "ticker"], key="t4_srt")

        disp = df.copy()
        if result_filter_t4 != "全件":
            disp = disp[disp["result"] == result_filter_t4]
        if year_filter_t4 != "全件":
            disp = disp[disp["date"].dt.year == int(year_filter_t4)]
        disp = disp.sort_values(sort_col, ascending=(sort_col not in ["pl_pct", "pnl"]))

        disp_show = disp[[
            "date", "ticker", "signal", "entry_date", "entry_price",
            "exit_price", "result", "pl_pct", "alloc", "pnl", "balance",
        ]].copy()
        disp_show.columns = [
            "決済日", "銘柄", "シグナル", "エントリー日", "買値$",
            "売値$", "終了理由", "損益%", "投資額$", "損益$", "残高$",
        ]
        disp_show["買値$"]   = disp_show["買値$"].map(lambda x: f"${x:.2f}")
        disp_show["売値$"]   = disp_show["売値$"].map(lambda x: f"${x:.2f}")
        disp_show["損益%"]   = disp_show["損益%"].map(lambda x: f"{x:+.2f}%")
        disp_show["投資額$"] = disp_show["投資額$"].map(lambda x: f"${x:,.2f}")
        disp_show["損益$"]   = disp_show["損益$"].map(lambda x: f"${x:+,.2f}")
        disp_show["残高$"]   = disp_show["残高$"].map(lambda x: f"${x:,.2f}")
        disp_show["決済日"]      = disp_show["決済日"].dt.strftime("%Y-%m-%d")
        disp_show["エントリー日"] = disp_show["エントリー日"].dt.strftime("%Y-%m-%d")

        st.caption(f"{len(disp)} 件")
        st.dataframe(disp_show, use_container_width=True, hide_index=True)

        st.subheader("終了理由別内訳")
        res_cnt = df["result"].value_counts().reset_index()
        res_cnt.columns = ["終了理由", "件数"]
        res_cnt["勝ち"] = res_cnt["終了理由"].map(
            lambda r: (df[df["result"] == r]["pnl"] > 0).sum()
        )
        res_cnt["平均損益%"] = res_cnt["終了理由"].map(
            lambda r: df[df["result"] == r]["pl_pct"].mean()
        ).map(lambda x: f"{x:+.2f}%")
        res_cnt["合計損益$"] = res_cnt["終了理由"].map(
            lambda r: df[df["result"] == r]["pnl"].sum()
        ).map(lambda x: f"${x:+,.0f}")
        st.dataframe(res_cnt, use_container_width=True, hide_index=True)

    with txn_sub2:
        if etf_txn_path.exists():
            df_txn = load_etf_txn(str(etf_txn_path))

            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                act_f = st.selectbox("売買", ["全件", "BUY", "SELL"], key="etf_act")
            with col_f2:
                tk_f = st.selectbox("Ticker",
                                    ["全件"] + sorted(df_txn["ticker"].unique().tolist()),
                                    key="etf_tk")
            with col_f3:
                yr_f = st.selectbox("年",
                                    ["全件"] + [str(y) for y in range(2021, 2027)],
                                    key="etf_yr")

            disp_txn = df_txn.copy()
            if act_f != "全件": disp_txn = disp_txn[disp_txn["action"] == act_f]
            if tk_f  != "全件": disp_txn = disp_txn[disp_txn["ticker"] == tk_f]
            if yr_f  != "全件": disp_txn = disp_txn[disp_txn["date"].str.startswith(yr_f)]

            st.caption(f"{len(disp_txn)} 件")
            st.dataframe(disp_txn, use_container_width=True, hide_index=True)

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.subheader("理由別集計")
                rsn = df_txn.groupby("reason")["value"].agg(
                    件数="count", 合計金額="sum", 平均金額="mean"
                ).round(0).sort_values("合計金額", ascending=False)
                st.dataframe(rsn, use_container_width=True)
            with col_r2:
                st.subheader("Ticker別 BUY集計")
                buy_df_ = df_txn[df_txn["action"] == "BUY"]
                buy_sum = buy_df_.groupby("ticker")["value"].agg(
                    BUY回数="count", 合計投資額="sum", 平均投資額="mean"
                ).round(0).sort_values("合計投資額", ascending=False)
                st.dataframe(buy_sum, use_container_width=True)
        else:
            st.warning("ETF取引履歴が見つかりません。`python etf_overlay_backtest.py` を実行してください。")

# ── Tab 5: サマリー ───────────────────────────────────────────────
with tab5:
    c_l, c_r = st.columns(2)

    with c_l:
        st.subheader("パフォーマンスサマリー（ETF v2 weekly）")
        summary = {
            "初期資金":         f"${INITIAL:,.0f}",
            "最終残高":         f"${final_etf:,.0f}",
            "トータルリターン": f"{ret_etf:+.1f}%",
            "合計損益":         f"${final_etf - INITIAL:+,.0f}",
            "CAGR":             f"{cagr_etf:+.1f}%",
            "Sharpe":           f"{sharpe_etf:.3f}",
            "Calmar":           f"{calmar_etf:.2f}",
            "最大ドローダウン": f"{max_dd_etf:.1f}%",
            "":                  "",
            "個別株 取引数":    str(n_tr),
            "勝率":             f"{winrate:.1f}%  ({wins}勝 / {losses}敗)",
            "プロフィットファクター": f"{pf:.3f}",
            " ":                 "",
            "バックテスト期間": f"{start_str} → {end_str}",
            "CSV":               selected_csv,
        }
        for k, v in summary.items():
            if k.strip() == "":
                st.write("")
            else:
                col_k, col_v = st.columns([1.2, 1.8])
                col_k.write(f"**{k}**")
                col_v.write(v)

        st.divider()
        st.subheader("4戦略比較")
        summary_df = pd.DataFrame({
            "指標":          ["最終残高", "リターン", "CAGR", "最大DD", "Sharpe", "Calmar"],
            "ETF v2 weekly": [f"${final_etf:,.0f}", f"{ret_etf:+.1f}%", f"{cagr_etf:+.1f}%",
                              f"{max_dd_etf:.1f}%", f"{sharpe_etf:.3f}", f"{calmar_etf:.2f}"],
            "個別株のみ":    [f"${final_bal:,.0f}", f"{total_ret:+.1f}%", f"{cagr_base:+.1f}%",
                              f"{max_dd_base:.1f}%", f"{sharpe_base:.3f}", f"{calmar_base:.2f}"],
            "SPY":           [f"${spy_norm.iloc[-1]:,.0f}", f"{spy_ret:+.1f}%", "—", "—", "—", "—"],
            "QQQ":           [f"${qqq_norm.iloc[-1]:,.0f}", f"{qqq_ret:+.1f}%", "—", "—", "—", "—"],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with c_r:
        st.subheader("終了理由の分布（個別株）")
        res_counts = df["result"].value_counts()
        fig5 = px.pie(
            values=res_counts.values,
            names=res_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        fig5.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig5, use_container_width=True)

        st.subheader("損益分布（個別株）")
        fig6 = go.Figure()
        fig6.add_trace(go.Histogram(
            x=df["pl_pct"],
            nbinsx=40,
            marker_color=["#7ee787" if v >= 0 else "#ff6b6b" for v in df["pl_pct"]],
            name="損益%",
        ))
        fig6.add_vline(x=0, line_color="#888", line_width=1)
        fig6.add_vline(x=df["pl_pct"].mean(), line_color="#00d4ff",
                       line_dash="dash",
                       annotation_text=f"平均 {df['pl_pct'].mean():+.2f}%")
        fig6.update_layout(
            xaxis_title="損益 (%)", yaxis_title="件数",
            xaxis_ticksuffix="%",
            height=300, margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig6, use_container_width=True)

        st.subheader("ETF v2 戦略ルール")
        st.markdown("""
**SPY レジームフィルター（200MA）**

| 相場 | 判定 | アクション |
|------|------|-----------|
| ブル | SPY > 200MA | ETFモメンタム戦略 |
| ベア | SPY < 200MA | セクターETF全売却 → SGOV待避 |

**ETFユニバース**: SPY / QQQ / XLC / XLY / XLP / XLE / XLF / XLV / XLI / XLB / XLK / XLRE / XLU

**スコアリング（週次）**: 1M×30% + 3M×30% + 6M×30% + 12M×10%（順位ベース）

**保有**: 上位2本（50MA以上）/ **リバランス**: 毎週月曜
        """)
