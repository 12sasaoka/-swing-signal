"""
trade_charts.py — トレード詳細チャート HTML レポート生成

signal_backtest CSV を読み込み、各取引のローソク足チャートに
エントリー / エグジット / SL を重ねたインタラクティブ HTML を出力する。

セクション構成:
  1. Top N 利益トレード（大きい順）
  2. Top N 損失トレード（大きい順）
  3. 銘柄別全トレード一覧（取引数 Top M 銘柄を 1 チャートに集約）

usage:
    python output/backtest/trade_charts.py
    python output/backtest/trade_charts.py --csv output/backtest/signal_backtest_XXX.csv
    python output/backtest/trade_charts.py --top 30 --tickers 15
    python output/backtest/trade_charts.py --filter sl_hit --top 50
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from datetime import timedelta

import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# パス設定
# ============================================================
ROOT = Path(__file__).resolve().parent.parent.parent
CSV_DIR = ROOT / "output" / "backtest"
OUTPUT_HTML = CSV_DIR / "trade_charts.html"

# ============================================================
# 表示設定
# ============================================================
CONTEXT_BEFORE = 30   # エントリー前に表示する営業日数
CONTEXT_AFTER  = 12   # エグジット後に表示する営業日数
MA_SHORT = 20
MA_LONG  = 50

# ダークテーマカラー
C_WIN    = "#00d46a"
C_LOSS   = "#ff4444"
C_ENTRY  = "#00aaff"
C_SL     = "#ff8800"
C_MA20   = "#ffcc00"
C_MA50   = "#cc66ff"
C_BG     = "#16213e"
C_PAPER  = "#1a1a2e"
C_GRID   = "#2a2a4a"

RESULT_LABEL = {
    "sl_hit":        "SL Hit",
    "trailing_stop": "Trail Stop",
    "pre_earnings":  "Pre-Earn",
    "timeout":       "Timeout",
    "timeout_30d":   "TO-30d",
    "timeout_60d":   "TO-60d",
    "timeout_90d":   "TO-90d",
}


# ============================================================
# データ読み込み
# ============================================================

def load_trades(csv_path: str) -> list[dict]:
    trades = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            trades.append({
                "ticker":      row["ticker"],
                "signal":      row["signal"],
                "score":       float(row["score"]),
                "signal_date": row["signal_date"],
                "entry_date":  row["entry_date"],
                "exit_date":   row["exit_date"],
                "entry_price": float(row["entry_price"]),
                "exit_price":  float(row["exit_price"]),
                "result":      row["result"],
                "pl_pct":      float(row["pl_pct"]),
                "sl_price":    float(row["sl_price"]),
            })
    return trades


def find_latest_csv() -> Path:
    """最新の signal_backtest_* CSV を返す。"""
    csvs = sorted(CSV_DIR.glob("signal_backtest_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"signal_backtest_*.csv が {CSV_DIR} に見つかりません")
    return csvs[-1]


# ============================================================
# 株価データ取得（キャッシュ付き）
# ============================================================

_price_cache: dict[str, pd.DataFrame | None] = {}


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance が返す MultiIndex 列を単純列名に変換する。"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def get_price_data(ticker: str, end_date: str) -> pd.DataFrame | None:
    """yfinance でローソク足データを取得（バックテスト全期間 + α を一括取得）。"""
    if ticker in _price_cache:
        return _price_cache[ticker]
    try:
        end_padded = (pd.Timestamp(end_date) + timedelta(days=20)).strftime("%Y-%m-%d")
        df = yf.download(
            ticker,
            start="2020-07-01",   # ウォームアップ分込みで十分長く
            end=end_padded,
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            _price_cache[ticker] = None
        else:
            df = _flatten_columns(df)
            df.index = df.index.tz_localize(None)
            _price_cache[ticker] = df
    except Exception as e:
        print(f"  ⚠ {ticker} データ取得失敗: {e}")
        _price_cache[ticker] = None
    return _price_cache[ticker]


def prefetch(tickers: list[str], end_date: str) -> None:
    """複数銘柄を事前取得（進捗表示付き）。"""
    unique = [t for t in dict.fromkeys(tickers) if t not in _price_cache]
    for i, tk in enumerate(unique, 1):
        print(f"  [{i}/{len(unique)}] {tk}", end="\r", flush=True)
        get_price_data(tk, end_date)
    if unique:
        print(f"  -> {len(unique)} 銘柄取得完了" + " " * 20)


# ============================================================
# 個別トレードチャート（ローソク足 + 出来高）
# ============================================================

def _nearest_idx(index: pd.DatetimeIndex, dt: pd.Timestamp) -> int:
    """dt に最も近いインデックス位置を返す。"""
    loc = index.searchsorted(dt)
    return min(loc, len(index) - 1)


def make_trade_chart(trade: dict, df_full: pd.DataFrame) -> go.Figure | None:
    """1 トレード分のローソク足チャートを作成する。"""
    entry_dt = pd.Timestamp(trade["entry_date"])
    exit_dt  = pd.Timestamp(trade["exit_date"])

    idx       = df_full.index
    entry_loc = _nearest_idx(idx, entry_dt)
    exit_loc  = _nearest_idx(idx, exit_dt)

    start_loc = max(0, entry_loc - CONTEXT_BEFORE)
    end_loc   = min(len(idx), exit_loc + CONTEXT_AFTER + 1)

    df = df_full.iloc[start_loc:end_loc].copy()
    if len(df) < 3:
        return None

    df["ma20"] = df["Close"].rolling(MA_SHORT, min_periods=1).mean()
    df["ma50"] = df["Close"].rolling(MA_LONG,  min_periods=1).mean()

    is_win     = trade["pl_pct"] > 0
    exit_color = C_WIN if is_win else C_LOSS
    result_lbl = RESULT_LABEL.get(trade["result"], trade["result"])
    pl_sign    = "+" if trade["pl_pct"] >= 0 else ""

    # ---- 2段レイアウト（OHLC + Volume）----
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    # ローソク足
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="OHLC",
        increasing=dict(line=dict(color=C_WIN),  fillcolor="#003d20"),
        decreasing=dict(line=dict(color=C_LOSS), fillcolor="#4d0000"),
        showlegend=False,
    ), row=1, col=1)

    # MA20 / MA50
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ma20"],
        line=dict(color=C_MA20, width=1.2), name="MA20", hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ma50"],
        line=dict(color=C_MA50, width=1.2), name="MA50", hoverinfo="skip",
    ), row=1, col=1)

    # SL ライン（エントリー〜エグジット期間）
    actual_entry = idx[entry_loc]
    actual_exit  = idx[exit_loc]
    fig.add_trace(go.Scatter(
        x=[actual_entry, actual_exit],
        y=[trade["sl_price"], trade["sl_price"]],
        line=dict(color=C_SL, width=1.8, dash="dash"),
        mode="lines",
        name=f"SL ${trade['sl_price']:.2f}",
    ), row=1, col=1)

    # エントリーマーカー
    entry_y = trade["entry_price"]
    fig.add_trace(go.Scatter(
        x=[actual_entry], y=[entry_y],
        mode="markers+text",
        marker=dict(symbol="triangle-up", color=C_ENTRY, size=15,
                    line=dict(color="white", width=1.2)),
        text=[f"${entry_y:.2f}"],
        textposition="bottom center",
        textfont=dict(color=C_ENTRY, size=9),
        name=f"Entry ${entry_y:.2f}",
        showlegend=False,
        hovertemplate=f"Entry: ${entry_y:.2f}<extra></extra>",
    ), row=1, col=1)

    # エグジットマーカー
    exit_y = trade["exit_price"]
    fig.add_trace(go.Scatter(
        x=[actual_exit], y=[exit_y],
        mode="markers+text",
        marker=dict(symbol="triangle-down", color=exit_color, size=15,
                    line=dict(color="white", width=1.2)),
        text=[f"{result_lbl} {pl_sign}{trade['pl_pct']:.1f}%"],
        textposition="top center",
        textfont=dict(color=exit_color, size=9),
        name=f"Exit {pl_sign}{trade['pl_pct']:.1f}%",
        showlegend=False,
        hovertemplate=f"Exit: ${exit_y:.2f} | {result_lbl} {pl_sign}{trade['pl_pct']:.1f}%<extra></extra>",
    ), row=1, col=1)

    # 出来高バー
    vol_colors = [
        C_WIN if c >= o else C_LOSS
        for c, o in zip(df["Close"].tolist(), df["Open"].tolist())
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=vol_colors,
        name="Volume", showlegend=False, hoverinfo="skip",
    ), row=2, col=1)

    # 垂直線（エントリー / エグジット）
    for dt, color in [(actual_entry, C_ENTRY), (actual_exit, exit_color)]:
        fig.add_vline(
            x=dt.timestamp() * 1000,
            line=dict(color=color, width=1.0, dash="dot"),
        )

    # タイトル
    hold_days = (pd.Timestamp(trade["exit_date"]) - pd.Timestamp(trade["entry_date"])).days
    title_html = (
        f"<b>{trade['ticker']}</b>  |  "
        f"{trade['signal']} (score={trade['score']:.3f})  |  "
        f"{trade['entry_date']} -> {trade['exit_date']} ({hold_days}日)  |  "
        f"<span style='color:{exit_color}'><b>{pl_sign}{trade['pl_pct']:.1f}%</b></span>  |  {result_lbl}"
    )

    fig.update_layout(
        title=dict(text=title_html, font=dict(size=12, color="#cccccc"), x=0),
        paper_bgcolor=C_PAPER,
        plot_bgcolor=C_BG,
        font=dict(color="#cccccc"),
        xaxis_rangeslider_visible=False,
        height=400,
        margin=dict(l=55, r=20, t=48, b=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
            font=dict(size=9), bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )
    # 土日の隙間を非表示
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],
        gridcolor=C_GRID, showgrid=True,
        tickfont=dict(color="#aaa", size=9),
    )
    fig.update_yaxes(
        gridcolor=C_GRID, showgrid=True,
        tickfont=dict(color="#aaa", size=9),
        tickprefix="$", row=1, col=1,
    )
    fig.update_yaxes(tickprefix="", showticklabels=False, row=2, col=1)

    return fig


# ============================================================
# 銘柄別全トレードチャート（全期間に全エントリー/エグジットを表示）
# ============================================================

def make_ticker_chart(ticker: str, trades: list[dict], df_full: pd.DataFrame) -> go.Figure | None:
    """1 銘柄の全取引を 1 枚のチャートに表示する。"""
    if not trades or df_full is None or df_full.empty:
        return None

    # 表示範囲: 最初のエントリー -CONTEXT_BEFORE 〜 最後のエグジット +CONTEXT_AFTER
    all_entries = [pd.Timestamp(t["entry_date"]) for t in trades]
    all_exits   = [pd.Timestamp(t["exit_date"])  for t in trades]
    range_start = min(all_entries) - timedelta(days=CONTEXT_BEFORE * 1.5)
    range_end   = max(all_exits)   + timedelta(days=CONTEXT_AFTER * 1.5)

    df = df_full[(df_full.index >= range_start) & (df_full.index <= range_end)].copy()
    if len(df) < 5:
        return None

    df["ma20"] = df["Close"].rolling(MA_SHORT, min_periods=1).mean()
    df["ma50"] = df["Close"].rolling(MA_LONG,  min_periods=1).mean()

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.78, 0.22],
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    # ローソク足
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="OHLC",
        increasing=dict(line=dict(color=C_WIN),  fillcolor="#003d20"),
        decreasing=dict(line=dict(color=C_LOSS), fillcolor="#4d0000"),
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["ma20"],
        line=dict(color=C_MA20, width=1.2), name="MA20", hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ma50"],
        line=dict(color=C_MA50, width=1.2), name="MA50", hoverinfo="skip",
    ), row=1, col=1)

    # 各トレードにエントリー / エグジット / SL を追加
    idx = df.index
    for i, trade in enumerate(trades):
        is_win     = trade["pl_pct"] > 0
        exit_color = C_WIN if is_win else C_LOSS
        result_lbl = RESULT_LABEL.get(trade["result"], trade["result"])
        pl_sign    = "+" if trade["pl_pct"] >= 0 else ""

        entry_dt   = pd.Timestamp(trade["entry_date"])
        exit_dt    = pd.Timestamp(trade["exit_date"])
        actual_entry = idx[_nearest_idx(idx, entry_dt)]
        actual_exit  = idx[_nearest_idx(idx, exit_dt)]

        # SL
        fig.add_trace(go.Scatter(
            x=[actual_entry, actual_exit],
            y=[trade["sl_price"], trade["sl_price"]],
            line=dict(color=C_SL, width=1.2, dash="dash"),
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
        ), row=1, col=1)

        show_legend = (i == 0)
        # エントリー
        fig.add_trace(go.Scatter(
            x=[actual_entry], y=[trade["entry_price"]],
            mode="markers",
            marker=dict(symbol="triangle-up", color=C_ENTRY, size=11,
                        line=dict(color="white", width=1)),
            name="Entry" if show_legend else None,
            showlegend=show_legend,
            hovertemplate=(
                f"<b>#{i+1} Entry</b><br>"
                f"Date: {trade['entry_date']}<br>"
                f"Price: ${trade['entry_price']:.2f}<br>"
                f"Signal: {trade['signal']} (score={trade['score']:.3f})"
                "<extra></extra>"
            ),
        ), row=1, col=1)

        # エグジット
        fig.add_trace(go.Scatter(
            x=[actual_exit], y=[trade["exit_price"]],
            mode="markers",
            marker=dict(symbol="triangle-down", color=exit_color, size=11,
                        line=dict(color="white", width=1)),
            name="Win Exit" if (show_legend and is_win) else ("Loss Exit" if show_legend else None),
            showlegend=show_legend,
            hovertemplate=(
                f"<b>#{i+1} Exit</b> {pl_sign}{trade['pl_pct']:.1f}% ({result_lbl})<br>"
                f"Date: {trade['exit_date']}<br>"
                f"Price: ${trade['exit_price']:.2f}<br>"
                f"SL: ${trade['sl_price']:.2f}"
                "<extra></extra>"
            ),
        ), row=1, col=1)

    # 出来高
    vol_colors = [
        C_WIN if c >= o else C_LOSS
        for c, o in zip(df["Close"].tolist(), df["Open"].tolist())
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=vol_colors,
        name="Volume", showlegend=False, hoverinfo="skip",
    ), row=2, col=1)

    # 統計サマリーをタイトルに
    wins      = sum(1 for t in trades if t["pl_pct"] > 0)
    total     = len(trades)
    total_pl  = sum(t["pl_pct"] for t in trades)
    avg_pl    = total_pl / total
    sl_cnt    = sum(1 for t in trades if t["result"] == "sl_hit")
    trail_cnt = sum(1 for t in trades if t["result"] == "trailing_stop")

    title_html = (
        f"<b>{ticker}</b>  |  {total}件  勝率 {wins/total*100:.0f}%  "
        f"平均 {avg_pl:+.1f}%  累計 {total_pl:+.1f}%  |  "
        f"SL:{sl_cnt}件  Trail:{trail_cnt}件"
    )

    fig.update_layout(
        title=dict(text=title_html, font=dict(size=12, color="#cccccc"), x=0),
        paper_bgcolor=C_PAPER,
        plot_bgcolor=C_BG,
        font=dict(color="#cccccc"),
        xaxis_rangeslider_visible=False,
        height=480,
        margin=dict(l=55, r=20, t=48, b=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
            font=dict(size=9), bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],
        gridcolor=C_GRID, showgrid=True,
        tickfont=dict(color="#aaa", size=9),
    )
    fig.update_yaxes(
        gridcolor=C_GRID, showgrid=True,
        tickfont=dict(color="#aaa", size=9),
        tickprefix="$", row=1, col=1,
    )
    fig.update_yaxes(tickprefix="", showticklabels=False, row=2, col=1)

    return fig


# ============================================================
# HTML 生成
# ============================================================

def generate_html(
    sections: list[tuple[str, list[go.Figure]]],
    output_path: Path,
    csv_name: str,
) -> None:
    """セクションリストから 1 本の HTML を生成する。"""

    # 目次用 anchor 生成
    def anchor(title: str) -> str:
        return title.replace(" ", "_").replace("（", "_").replace("）", "").replace("/", "_")

    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="ja"><head>',
        '<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Swing Signal — Trade Charts</title>",
        '<script src="https://cdn.plot.ly/plotly-3.0.0.min.js"></script>',
        "<style>",
        f"* {{ box-sizing:border-box; margin:0; padding:0; }}",
        f"body {{ background:{C_PAPER}; color:#ccc; font-family:'Segoe UI',sans-serif; }}",
        f"header {{ background:{C_BG}; padding:16px 24px; border-bottom:1px solid #2a3a5e; }}",
        f"header h1 {{ color:#00d4ff; font-size:22px; }}",
        f"header p  {{ color:#888; font-size:12px; margin-top:4px; }}",
        ".toc { max-width:1280px; margin:12px auto; padding:10px 20px; "
        f"  background:{C_BG}; border-radius:6px; border:1px solid #2a3a5e; }}",
        ".toc b  { color:#ffaa00; font-size:13px; }",
        ".toc a  { color:#00aaff; margin:0 10px; font-size:13px; text-decoration:none; }",
        ".toc a:hover { text-decoration:underline; }",
        "h2 { max-width:1280px; margin:20px auto 6px; padding:8px 20px; "
        f"  color:#ffaa00; border-left:4px solid #ffaa00; font-size:15px; }}",
        ".chart-wrap { max-width:1280px; margin:0 auto 6px; "
        f"  background:{C_BG}; border-radius:4px; padding:4px; }}",
        "</style>",
        "</head><body>",
        f"<header>",
        f"  <h1>Swing Signal — Trade Chart Report</h1>",
        f"  <p>CSV: {csv_name}</p>",
        f"</header>",
    ]

    # 目次
    html_parts.append('<nav class="toc"><b>目次：</b>')
    for sec_title, figs in sections:
        anc = anchor(sec_title)
        html_parts.append(f'<a href="#{anc}">{sec_title}（{len(figs)}件）</a>')
    html_parts.append("</nav>")

    # セクション
    for sec_title, figs in sections:
        anc = anchor(sec_title)
        html_parts.append(f'<h2 id="{anc}">{sec_title}</h2>')
        for fig in figs:
            chart_div = fig.to_html(
                include_plotlyjs=False,
                full_html=False,
                config={"responsive": True, "displayModeBar": True,
                        "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
            )
            html_parts.append(f'<div class="chart-wrap">{chart_div}</div>')

    html_parts.append("</body></html>")

    output_path.write_text("\n".join(html_parts), encoding="utf-8")


# ============================================================
# メイン
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="トレードチャート HTML レポート生成")
    parser.add_argument("--csv",     default=None, help="signal_backtest CSV パス（省略時は最新）")
    parser.add_argument("--top",     type=int, default=25, help="勝ち/負けセクションの最大件数 (default: 25)")
    parser.add_argument("--tickers", type=int, default=15, help="銘柄別セクションに含める銘柄数 (default: 15)")
    parser.add_argument("--filter",  default="all",
                        choices=["all", "sl_hit", "trailing_stop", "pre_earnings"],
                        help="決済タイプフィルター (default: all)")
    parser.add_argument("--output",  default=str(OUTPUT_HTML), help="出力 HTML パス")
    args = parser.parse_args()

    # ----- CSV ロード -----
    csv_path = args.csv or str(find_latest_csv())
    print(f"[CSV] CSV: {csv_path}")
    all_trades = load_trades(csv_path)

    if args.filter != "all":
        all_trades = [t for t in all_trades if t["result"] == args.filter]

    print(f"   対象トレード: {len(all_trades)} 件")

    # 全体の最後エグジット日（データ取得の end 用）
    global_end = max(t["exit_date"] for t in all_trades) if all_trades else "2026-12-31"

    # ============================================================
    # Section 1 & 2: 個別トレードチャート（Top N wins / losses）
    # ============================================================
    wins_sorted   = sorted([t for t in all_trades if t["pl_pct"] > 0],  key=lambda t: -t["pl_pct"])
    losses_sorted = sorted([t for t in all_trades if t["pl_pct"] <= 0], key=lambda t:  t["pl_pct"])

    top_wins   = wins_sorted[:args.top]
    top_losses = losses_sorted[:args.top]

    trade_sections_spec = [
        (f"^ Top {args.top} 利益トレード（大きい順）", top_wins),
        (f"v Top {args.top} 損失トレード（大きい順）", top_losses),
    ]

    sections_out: list[tuple[str, list[go.Figure]]] = []

    for sec_title, trades_subset in trade_sections_spec:
        print(f"\n[CHARTS] {sec_title}")
        tickers_needed = list(dict.fromkeys(t["ticker"] for t in trades_subset))
        prefetch(tickers_needed, global_end)

        figs = []
        for i, trade in enumerate(trades_subset, 1):
            df = get_price_data(trade["ticker"], global_end)
            if df is None:
                print(f"  [{i}/{len(trades_subset)}] {trade['ticker']}: データなし -> スキップ")
                continue
            fig = make_trade_chart(trade, df)
            if fig is not None:
                figs.append(fig)
                pl_str = f"{'+'if trade['pl_pct']>=0 else ''}{trade['pl_pct']:.1f}%"
                print(f"  [{i}/{len(trades_subset)}] {trade['ticker']:6s} "
                      f"{trade['entry_date']}->{trade['exit_date']} "
                      f"{pl_str:>8} ({trade['result']})")

        sections_out.append((sec_title, figs))
        print(f"  -> {len(figs)} チャート生成")

    # ============================================================
    # Section 3: 銘柄別全トレードチャート（取引数 Top N 銘柄）
    # ============================================================
    from collections import defaultdict
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for t in all_trades:
        by_ticker[t["ticker"]].append(t)

    top_tickers = sorted(by_ticker, key=lambda tk: -len(by_ticker[tk]))[: args.tickers]

    print(f"\n[TICKER] 銘柄別全トレード（Top {args.tickers} 銘柄）")
    prefetch(top_tickers, global_end)

    ticker_figs = []
    for i, ticker in enumerate(top_tickers, 1):
        trades_for_tk = sorted(by_ticker[ticker], key=lambda t: t["entry_date"])
        df = get_price_data(ticker, global_end)
        if df is None:
            print(f"  [{i}/{len(top_tickers)}] {ticker}: データなし -> スキップ")
            continue
        fig = make_ticker_chart(ticker, trades_for_tk, df)
        if fig is not None:
            ticker_figs.append(fig)
            total_pl = sum(t["pl_pct"] for t in trades_for_tk)
            print(f"  [{i}/{len(top_tickers)}] {ticker:6s}  "
                  f"{len(trades_for_tk):2d}件  累計{total_pl:+.1f}%")

    sec_title3 = f"[PIN] 銘柄別全トレード（取引数 Top {args.tickers} 銘柄）"
    sections_out.append((sec_title3, ticker_figs))
    print(f"  -> {len(ticker_figs)} チャート生成")

    # ============================================================
    # HTML 出力
    # ============================================================
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[WRITE]  HTML 生成中... ", end="", flush=True)
    generate_html(sections_out, output_path, Path(csv_path).name)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"完了")
    print(f"\n[DONE] 出力: {output_path}  ({size_mb:.1f} MB)")
    print(f"   ブラウザで開く: file:///{output_path.as_posix()}")


if __name__ == "__main__":
    main()
