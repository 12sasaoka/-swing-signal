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
                "date":     date,
                "ticker":   t["ticker"],
                "signal":   t["signal"],
                "entry_date":  t["entry_date"],
                "entry_price": t["entry_price"],
                "exit_price":  t["exit_price"],
                "result":   RESULT_LABELS.get(t["result"], t["result"]),
                "pl_pct":   t["pl_pct"],
                "balance":  bal,
                "pnl":      pnl,
                "alloc":    amt,
                "cash":     cash,
                "invested": sum(active.values()),
                "n_pos":    len(active),
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
def get_spy(start: str, end: str):
    raw = yf.download("SPY", start=start, end=end, progress=False)
    spy = (raw["Close"]["SPY"] if isinstance(raw.columns, pd.MultiIndex)
           else raw["Close"]).dropna()
    return spy


def build_daily(df: pd.DataFrame):
    df_d = df.groupby("date").last().reset_index().sort_values("date")
    bdays = pd.bdate_range(df_d["date"].min(), df_d["date"].max())
    eq   = df_d.set_index("date")["balance"].reindex(bdays).ffill().fillna(INITIAL)
    cash = df_d.set_index("date")["cash"].reindex(bdays).ffill().fillna(INITIAL)
    inv  = df_d.set_index("date")["invested"].reindex(bdays).ffill().fillna(0)
    npos = df_d.set_index("date")["n_pos"].reindex(bdays).ffill().fillna(0)
    return eq, cash, inv, npos


# ── データ準備 ────────────────────────────────────────────────────
df, max_dd, peak = run_simulation(selected_csv, signal_filter)

if df.empty:
    st.error("取引データが見つかりません。")
    st.stop()

eq_s, cash_s, inv_s, npos_s = build_daily(df)

start_str = eq_s.index[0].strftime("%Y-%m-%d")
end_str   = eq_s.index[-1].strftime("%Y-%m-%d")
spy = get_spy(start_str, end_str)
spy_norm = spy / spy.iloc[0] * INITIAL

final_bal  = eq_s.iloc[-1]
total_ret  = (final_bal - INITIAL) / INITIAL * 100
spy_ret    = (spy.iloc[-1] - spy.iloc[0]) / spy.iloc[0] * 100
n_tr       = len(df)
wins       = (df["pnl"] > 0).sum()
losses     = (df["pnl"] < 0).sum()
gp         = df.loc[df["pnl"] > 0, "pnl"].sum()
gl         = abs(df.loc[df["pnl"] < 0, "pnl"].sum())
pf         = gp / gl if gl > 0 else 9.99
winrate    = wins / n_tr * 100
avg_idle_pct = (cash_s / eq_s * 100).mean()
avg_dep_pct  = (inv_s  / eq_s * 100).mean()

# 年別
def yearly_stats(df: pd.DataFrame):
    out = {}
    prev_bal = INITIAL
    for yr in sorted(df["date"].dt.year.unique()):
        yr_df = df[df["date"].dt.year == yr]
        if yr_df.empty:
            continue
        end_bal  = yr_df.iloc[-1]["balance"]
        yr_ret   = (end_bal - prev_bal) / prev_bal * 100
        yr_pnl   = yr_df["pnl"].sum()
        yr_wins  = (yr_df["pnl"] > 0).sum()
        out[yr]  = {"ret": yr_ret, "pnl": yr_pnl, "end": end_bal,
                    "trades": len(yr_df), "wins": yr_wins}
        prev_bal = end_bal
    return out

yr_stats = yearly_stats(df)

# ── ヘッダー ─────────────────────────────────────────────────────
st.title("Swing Signal ダッシュボード")
st.caption(f"CSV: `{selected_csv}`  |  シグナル: `{signal_filter}`  |  "
           f"risk=0.5%  trail4.0  初期資金=${INITIAL:,.0f}")

# ── KPIカード ─────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("最終残高",       f"${final_bal:,.2f}")
c2.metric("トータルリターン", f"{total_ret:+.1f}%",  f"SPY: {spy_ret:+.1f}%")
c3.metric("勝率",           f"{winrate:.1f}%",     f"{wins}勝 / {losses}敗")
c4.metric("プロフィットファクター", f"{pf:.3f}")
c5.metric("最大ドローダウン", f"{max_dd:.1f}%")
c6.metric("平均待機資金",    f"{avg_idle_pct:.1f}%", f"${cash_s.mean():,.0f}")

st.divider()

# ── タブ ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📈 資産推移", "💰 資金使用率", "📅 年別", "📋 取引一覧", "📊 サマリー", "🔀 ETFオーバーレイ"]
)

# ── Tab 1: 資産推移 ───────────────────────────────────────────────
with tab1:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=eq_s.index, y=eq_s.values,
        name=f"Strategy  ${final_bal:,.0f}  ({total_ret:+.1f}%)",
        line=dict(color="#00d4ff", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
    ))
    fig.add_trace(go.Scatter(
        x=spy_norm.index, y=spy_norm.values,
        name=f"SPY  ${spy_norm.iloc[-1]:,.0f}  ({spy_ret:+.1f}%)",
        line=dict(color="#888888", width=1.5, dash="dot"),
    ))

    # 年末マーカー
    for yr, st_ in yr_stats.items():
        yr_end = pd.Timestamp(f"{yr}-12-31")
        nearest = eq_s.index[eq_s.index <= yr_end]
        if len(nearest) == 0:
            continue
        d = nearest[-1]; v = eq_s[d]
        fig.add_annotation(
            x=d, y=v,
            text=f"${v:,.0f}",
            showarrow=True, arrowhead=2, arrowcolor="#00d4ff",
            font=dict(color="#00d4ff", size=11),
            ax=0, ay=-35,
        )

    fig.add_hline(y=INITIAL, line_dash="dot", line_color="#6e7681",
                  annotation_text=f"初期資金 ${INITIAL:,.0f}", annotation_position="left")

    fig.update_layout(
        title="資産推移 vs SPY",
        xaxis_title="日付", yaxis_title="残高 ($)",
        yaxis_tickformat="$,.0f",
        hovermode="x unified",
        height=500,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: 資金使用率 ─────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns([2, 1])

    with col_a:
        # スタックエリア（絶対額）
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
        fig2.add_trace(go.Scatter(
            x=eq_s.index, y=eq_s.values,
            name="総資産",
            line=dict(color="#00d4ff", width=1.5, dash="dash"),
        ))
        for yr in range(2022, 2027):
            fig2.add_vline(x=f"{yr}-01-01", line_dash="dot",
                          line_color="#444", line_width=1)

        fig2.update_layout(
            title="投資中資金 vs 待機資金（絶対額）",
            yaxis_tickformat="$,.0f",
            hovermode="x unified",
            height=400,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # 待機資金 % + ポジション数
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        cash_pct = cash_s / eq_s * 100
        fig3.add_trace(go.Scatter(
            x=eq_s.index, y=cash_pct.values,
            name=f"待機資金 % (平均 {cash_pct.mean():.1f}%)",
            fill="tozeroy", fillcolor="rgba(240,165,0,0.25)",
            line=dict(color="#f0a500", width=1.5),
        ), secondary_y=False)
        fig3.add_trace(go.Scatter(
            x=eq_s.index, y=npos_s.values,
            name=f"保有銘柄数 (平均 {npos_s.mean():.1f})",
            line=dict(color="#c084fc", width=1.2),
        ), secondary_y=True)
        fig3.add_hline(y=cash_pct.mean(), line_dash="dash",
                       line_color="rgba(240,165,0,0.5)",
                       annotation_text=f"平均 {cash_pct.mean():.1f}%")
        fig3.update_yaxes(title_text="待機資金 (%)", secondary_y=False,
                          tickformat=".0f", ticksuffix="%", range=[0, 100])
        fig3.update_yaxes(title_text="保有銘柄数", secondary_y=True,
                          range=[0, MAX_POS + 3])
        fig3.update_layout(
            title="待機資金 (%) & 保有銘柄数",
            hovermode="x unified",
            height=350,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.subheader("年別 待機資金")
        yr_idle = {}
        for yr in range(2021, 2027):
            mask = (eq_s.index.year == yr)
            if mask.sum() == 0:
                continue
            yr_idle[yr] = {
                "待機率":      f"{(cash_s[mask] / eq_s[mask] * 100).mean():.1f}%",
                "平均待機$":   f"${cash_s[mask].mean():,.0f}",
                "平均投資$":   f"${inv_s[mask].mean():,.0f}",
                "平均銘柄数":  f"{npos_s[mask].mean():.1f}",
            }
        st.dataframe(pd.DataFrame(yr_idle).T, use_container_width=True)

        st.subheader("全期間")
        st.dataframe(pd.DataFrame({
            "指標": ["平均待機率", "平均投資率", "平均保有銘柄数", "最大保有銘柄数"],
            "値":  [
                f"{avg_idle_pct:.1f}% (${cash_s.mean():,.0f})",
                f"{avg_dep_pct:.1f}%  (${inv_s.mean():,.0f})",
                f"{npos_s.mean():.1f} / {MAX_POS}",
                f"{int(npos_s.max())} / {MAX_POS}",
            ]
        }), use_container_width=True, hide_index=True)

# ── Tab 3: 年別 ───────────────────────────────────────────────────
with tab3:
    col_c, col_d = st.columns([3, 2])

    with col_c:
        yrs  = list(yr_stats.keys())
        rets = [yr_stats[y]["ret"] for y in yrs]
        pnls = [yr_stats[y]["pnl"] for y in yrs]
        cols = ["#00d4ff" if r >= 0 else "#ff6b6b" for r in rets]

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=[str(y) for y in yrs], y=rets,
            marker_color=cols,
            text=[f"{r:+.1f}%<br>${p:+,.0f}" for r, p in zip(rets, pnls)],
            textposition="outside",
            name="年間リターン",
        ))
        fig4.add_hline(y=0, line_color="#888", line_width=1)
        fig4.update_layout(
            title="年間リターン & 損益 ($)",
            yaxis_title="リターン (%)",
            yaxis_ticksuffix="%",
            height=420,
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col_d:
        yr_table = []
        for yr, s in yr_stats.items():
            yr_table.append({
                "年":      yr,
                "取引数":  s["trades"],
                "勝ち":    s["wins"],
                "勝率":    f"{s['wins']/s['trades']*100:.1f}%",
                "リターン": f"{s['ret']:+.1f}%",
                "損益$":   f"${s['pnl']:+,.0f}",
                "年末残高": f"${s['end']:,.0f}",
            })
        st.subheader("年別内訳")
        st.dataframe(pd.DataFrame(yr_table), use_container_width=True, hide_index=True)

        # SPY 年別比較
        spy_yr = {}
        for yr in sorted(yrs):
            spy_yr_mask = spy_norm[spy_norm.index.year == yr]
            if spy_yr_mask.empty:
                continue
            spy_yr_mask2 = spy[spy.index.year == yr]
            if spy_yr_mask2.empty:
                continue
            s_ret = (spy_yr_mask2.iloc[-1] - spy_yr_mask2.iloc[0]) / spy_yr_mask2.iloc[0] * 100
            alpha = yr_stats[yr]["ret"] - s_ret
            spy_yr[yr] = {"SPY %": f"{s_ret:+.1f}%", "超過リターン": f"{alpha:+.1f}%"}

        st.subheader("SPY対比（年別）")
        st.dataframe(pd.DataFrame(spy_yr).T, use_container_width=True)

# ── Tab 4: 取引一覧 ───────────────────────────────────────────────
with tab4:
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        result_options = ["全件"] + sorted(df["result"].unique().tolist())
        result_filter = st.selectbox("終了理由フィルター", result_options)
    with col_f2:
        year_options = ["全件"] + sorted(df["date"].dt.year.unique().astype(str).tolist())
        year_filter = st.selectbox("年", year_options)
    with col_f3:
        sort_col = st.selectbox("並び替え", ["date", "pl_pct", "pnl", "ticker"])

    disp = df.copy()
    if result_filter != "全件":
        disp = disp[disp["result"] == result_filter]
    if year_filter != "全件":
        disp = disp[disp["date"].dt.year == int(year_filter)]
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

    # 結果別内訳
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

# ── Tab 5: サマリー ───────────────────────────────────────────────
with tab5:
    c_l, c_r = st.columns(2)

    with c_l:
        st.subheader("パフォーマンスサマリー")
        summary = {
            "初期資金":         f"${INITIAL:,.0f}",
            "最終残高":         f"${final_bal:,.2f}",
            "トータルリターン": f"{total_ret:+.1f}%",
            "合計損益":         f"${final_bal - INITIAL:+,.2f}",
            "SPY比超過":        f"{total_ret - spy_ret:+.1f}%",
            "":                  "",
            "総取引数":         str(n_tr),
            "勝率":             f"{winrate:.1f}%  ({wins}勝 / {losses}敗)",
            "プロフィットファクター": f"{pf:.3f}",
            "最大ドローダウン": f"{max_dd:.1f}%",
            "ピーク残高":       f"${peak:,.2f}",
            " ":                 "",
            "平均待機資金":     f"{avg_idle_pct:.1f}%  (${cash_s.mean():,.0f})",
            "平均投資額":       f"{avg_dep_pct:.1f}%  (${inv_s.mean():,.0f})",
            "平均保有銘柄数":   f"{npos_s.mean():.1f} / {MAX_POS}",
            "  ":                "",
            "バックテスト期間": f"{start_str} → {end_str}",
            "CSV":               selected_csv,
            "シグナルフィルター": signal_filter,
        }
        for k, v in summary.items():
            if k.strip() == "":
                st.write("")
            else:
                col_k, col_v = st.columns([1.2, 1.8])
                col_k.write(f"**{k}**")
                col_v.write(v)

    with c_r:
        st.subheader("終了理由の分布")
        res_counts = df["result"].value_counts()
        fig5 = px.pie(
            values=res_counts.values,
            names=res_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        fig5.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig5, use_container_width=True)

        st.subheader("損益分布")
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

# ── Tab 6: ETFオーバーレイ ────────────────────────────────────────
with tab6:
    st.subheader("ETFオーバーレイ バックテスト結果")
    st.caption(
        "待機資金をモメンタム上位ETF（最大2本）に自動投資する戦略。"
        "個別株シグナル時はETFを一部売却して資金を確保。"
    )

    # ── KPI比較 ──────────────────────────────────────────────────
    ETF_OVERLAY_RESULTS = {
        "ベースライン": {"残高": 11146.92, "リターン": 178.67, "最大DD": 12.45, "PF": 1.701},
        "ETFオーバーレイ": {"残高": 12076.26, "リターン": 201.91, "最大DD": 21.45, "PF": 1.708},
    }
    col1, col2, col3, col4 = st.columns(4)
    diff_ret = ETF_OVERLAY_RESULTS["ETFオーバーレイ"]["リターン"] - ETF_OVERLAY_RESULTS["ベースライン"]["リターン"]
    diff_bal = ETF_OVERLAY_RESULTS["ETFオーバーレイ"]["残高"]    - ETF_OVERLAY_RESULTS["ベースライン"]["残高"]
    diff_dd  = ETF_OVERLAY_RESULTS["ETFオーバーレイ"]["最大DD"]  - ETF_OVERLAY_RESULTS["ベースライン"]["最大DD"]
    col1.metric("最終残高（ETF）",     f"${ETF_OVERLAY_RESULTS['ETFオーバーレイ']['残高']:,.0f}",
                f"+${diff_bal:,.0f} vs Baseline")
    col2.metric("トータルリターン（ETF）", f"+{ETF_OVERLAY_RESULTS['ETFオーバーレイ']['リターン']:.1f}%",
                f"{diff_ret:+.1f}% vs Baseline")
    col3.metric("最大DD（ETF）",       f"{ETF_OVERLAY_RESULTS['ETFオーバーレイ']['最大DD']:.1f}%",
                f"{diff_dd:+.1f}% vs Baseline", delta_color="inverse")
    col4.metric("PF（ETF）",           f"{ETF_OVERLAY_RESULTS['ETFオーバーレイ']['PF']:.3f}",
                f"Baseline: {ETF_OVERLAY_RESULTS['ベースライン']['PF']:.3f}")

    st.divider()

    # ── チャート画像 ──────────────────────────────────────────────
    chart_path = BASE / "output/backtest/etf_overlay_result.png"
    if chart_path.exists():
        st.image(str(chart_path), use_container_width=True)
    else:
        st.warning("チャートが見つかりません。先に `etf_overlay_backtest.py` を実行してください。")
        st.code("python etf_overlay_backtest.py")

    # ── ETFルール説明 ────────────────────────────────────────────
    st.divider()
    st.subheader("ETFオーバーレイ ルール")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
**ユニバース（13本）**
SPY / QQQ / XLC / XLY / XLP / XLE / XLF / XLV / XLI / XLB / XLK / XLRE / XLU

**スコアリング（月1回）**
- 1ヶ月リターン × 30%
- 3ヶ月リターン × 30%
- 6ヶ月リターン × 30%
- 12ヶ月リターン × 10%
- 各期間で順位付け → 加重合計（低スコア = 強い）

**保有本数**: 上位2本（50MA以上のみ）
        """)
    with col_r:
        st.markdown("""
**エントリー**: 毎月第1営業日にリバランス、全待機資金を均等配分

**50MA割れ**: 即日売却、翌月リバランスまでキャッシュ

**全売り**: なし（相場によって輝くセクターがあるため）

**個別株シグナル時**: モメンタムスコアが弱いETFから必要額を売却

**個別株EXIT後**: 翌月リバランスで再デプロイ
        """)
