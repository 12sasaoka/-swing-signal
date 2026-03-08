"""
Live Trading — 日次LINE通知フォーマット

メッセージ例:
  📊 Daily Signal [2026-02-22]
  SPY: $510.30  MA200: $490.50  ✅ 強気相場

  🔴 EXIT (2件)
    NVDA  SL Hit     @$118.50  -7.8%  -$35.10
    AMD   Trail Stop @$155.30 +12.1%  +$54.20

  🟢 ENTRY (2件)
    AAPL  BUY        @$210.50  SL:-8.2%  Alloc:$500
    TSLA  STRONG_BUY @$280.00  SL:-7.9%  Alloc:$500

  💼 PORTFOLIO
    Balance: $5,230.10 (+30.8%)
    Open: 3/10 | Cash: $3,154.80
"""

from __future__ import annotations

import logging

from output.notifier import notify_custom

logger = logging.getLogger(__name__)

_MAX_LINE_LEN = 1000


def send_daily_notification(
    exits: list[dict],
    entries: list[dict],
    state,
    spy_filter: bool,
    spy_close: float,
    spy_ma200: float,
    final_balance: float,
    total_return_pct: float,
    today: str,
    etf_orders: list[dict] | None = None,
) -> bool:
    """日次ポートフォリオ更新を LINE Notify で送信する。"""
    message = _build_message(
        exits, entries, state, spy_filter,
        spy_close, spy_ma200,
        final_balance, total_return_pct, today,
        etf_orders=etf_orders,
    )
    return notify_custom(message)


def _build_message(
    exits: list[dict],
    entries: list[dict],
    state,
    spy_filter: bool,
    spy_close: float,
    spy_ma200: float,
    final_balance: float,
    total_return_pct: float,
    today: str,
    etf_orders: list[dict] | None = None,
) -> str:
    from backtest.engine import MAX_CONCURRENT_POSITIONS

    lines: list[str] = []

    # ヘッダー
    lines.append(f"\n📊 Daily Signal [{today}]")

    # SPY ステータス
    if spy_close > 0 and spy_ma200 > 0:
        spy_icon = "✅" if spy_filter else "🚫"
        spy_status = "強気相場" if spy_filter else "弱気相場 — エントリーなし"
        lines.append(f"SPY: ${spy_close:.2f}  MA200: ${spy_ma200:.2f}  {spy_icon} {spy_status}")

    # エグジット
    if exits:
        lines.append(f"\n🔴 EXIT ({len(exits)}件)")
        for e in exits[:8]:  # 最大8件
            sign = "+" if e["pnl_dollar"] >= 0 else ""
            label = _exit_label(e["result"])
            lines.append(
                f"  {e['ticker']:<6} {label:<10} @${e['exit_price']:.2f}  "
                f"{sign}{e['pl_pct']:.1f}%  {sign}${e['pnl_dollar']:.2f}"
            )
        if len(exits) > 8:
            lines.append(f"  ... 他{len(exits) - 8}件")
    else:
        lines.append("\n🔴 EXIT: なし")

    # エントリー
    if entries:
        lines.append(f"\n🟢 ENTRY ({len(entries)}件)")
        for e in entries[:6]:  # 最大6件
            sig = "STRONG" if e["signal"] == "STRONG_BUY" else "BUY   "
            lines.append(
                f"  {e['ticker']:<6} {sig}  @${e['entry_price']:.2f}  "
                f"SL:-{e['sl_pct']*100:.1f}%  Alloc:${e['allocated']:.0f}"
            )
        if len(entries) > 6:
            lines.append(f"  ... 他{len(entries) - 6}件")
    elif not spy_filter:
        lines.append("\n🟢 ENTRY: 弱気相場のためスキップ")
    else:
        lines.append("\n🟢 ENTRY: シグナルなし")

    # ETFオーダー
    if etf_orders:
        buys  = [o for o in etf_orders if o["action"] == "BUY"]
        sells = [o for o in etf_orders if o["action"] == "SELL"]
        lines.append(f"\n📊 ETF ({len(etf_orders)}件)")
        for o in sells[:4]:
            lines.append(f"  SELL {o['ticker']:<5} ${o['value']:.0f}  [{o['reason']}]")
        for o in buys[:4]:
            lines.append(f"  BUY  {o['ticker']:<5} ${o['value']:.0f}  [{o['reason']}]")

    # ポートフォリオサマリー
    sign = "+" if total_return_pct >= 0 else ""
    lines.append(f"\n💼 PORTFOLIO")
    lines.append(f"  Balance: ${final_balance:,.2f} ({sign}{total_return_pct:.1f}%)")
    lines.append(f"  Open: {len(state.positions)}/{MAX_CONCURRENT_POSITIONS} | Cash: ${state.cash:,.2f}")

    # 決済履歴サマリー（存在すれば）
    if state.history:
        total = len(state.history)
        wins = sum(1 for h in state.history if h.get("pnl_dollar", 0) > 0)
        winrate = wins / total * 100 if total > 0 else 0
        lines.append(f"  Total: {total}trades  WR: {winrate:.0f}%")

    message = "\n".join(lines)

    # LINE の文字数制限（1000文字）
    if len(message) > _MAX_LINE_LEN:
        message = _truncate(message)

    return message


def _truncate(message: str) -> str:
    """1000文字以内に収まるように末尾を切り詰める。"""
    return message[: _MAX_LINE_LEN - 20] + "\n...(省略)"


def _exit_label(reason: str) -> str:
    labels = {
        "sl_hit": "SL Hit",
        "trailing_stop": "Trail Stop",
        "timeout_30d": "TO-30d",
        "timeout_60d": "TO-60d",
        "timeout_90d": "TO-90d",
    }
    return labels.get(reason, reason)
