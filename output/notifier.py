"""
Swing Trade Signal System — LINE Notify 通知

BUY/SELL シグナルが出た銘柄のサマリーを LINE Notify API で送信する。

LINE_NOTIFY_TOKEN 未設定時はエラーを出さずスキップする。

メッセージ例:
  📊 Swing Signal [2025-02-20 18:00]

  🟢 BUY (2)
  NVDA  BUY  $125.30  SL:$118.66  TP:$130.08
  AMD   BUY  $165.20  SL:$158.10  TP:$172.50

  🔴 SELL (1)
  BAD  STRONG_SELL  $42.10  SL:$38.73  TP:$50.52
"""

from __future__ import annotations

import logging
from datetime import datetime

import requests

from config.settings import LINE_NOTIFY_API_URL, get_line_token
from strategy.scorer import (
    BUY,
    SELL,
    STRONG_BUY,
    STRONG_SELL,
    ScoringResult,
    filter_actionable,
)

logger = logging.getLogger(__name__)

# LINE Notify 1メッセージあたりの最大文字数
_MAX_MESSAGE_LENGTH: int = 1000


# ============================================================
# 公開 API
# ============================================================

def notify_signals(
    results: list[ScoringResult],
    force: bool = False,
) -> bool:
    """BUY/SELL シグナルを LINE Notify で送信する。

    Args:
        results: ScoringResult のリスト。
        force:   True の場合、シグナルがなくても「シグナルなし」を通知。

    Returns:
        送信成功時 True、スキップまたは失敗時 False。
    """
    token = get_line_token()
    if not token:
        logger.debug("LINE Notify: トークン未設定 — スキップ")
        return False

    if not results:
        if force:
            return _send(token, "\n📊 Swing Signal: 分析結果なし")
        return False

    filtered = filter_actionable(results)
    buy_list = filtered["buy"]
    sell_list = filtered["sell"]

    if not buy_list and not sell_list:
        if force:
            msg = _build_no_signal_message(len(results))
            return _send(token, msg)
        logger.info("LINE Notify: アクション対象なし — スキップ")
        return False

    message = _build_signal_message(buy_list, sell_list, len(results))
    return _send(token, message)


def notify_custom(message: str) -> bool:
    """任意のメッセージを LINE Notify で送信する。

    Args:
        message: 送信するメッセージ文字列。

    Returns:
        送信成功時 True。
    """
    token = get_line_token()
    if not token:
        logger.debug("LINE Notify: トークン未設定 — スキップ")
        return False
    return _send(token, message)


# ============================================================
# メッセージ構築
# ============================================================

def _build_signal_message(
    buy_list: list[ScoringResult],
    sell_list: list[ScoringResult],
    total_count: int,
) -> str:
    """BUY/SELL シグナルの通知メッセージを構築する。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [
        f"\n📊 Swing Signal [{now}]",
        f"全{total_count}銘柄分析完了",
        "",
    ]

    # BUY 候補
    if buy_list:
        lines.append(f"🟢 BUY ({len(buy_list)})")
        for r in buy_list:
            lines.append(_format_signal_line(r))
        lines.append("")

    # SELL 候補
    if sell_list:
        lines.append(f"🔴 SELL ({len(sell_list)})")
        for r in sell_list:
            lines.append(_format_signal_line(r))
        lines.append("")

    message = "\n".join(lines)

    # LINE Notify の文字数制限対応
    if len(message) > _MAX_MESSAGE_LENGTH:
        message = _truncate_message(buy_list, sell_list, total_count)

    return message


def _build_no_signal_message(total_count: int) -> str:
    """シグナルなし時のメッセージを構築する。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"\n📊 Swing Signal [{now}]\n"
        f"全{total_count}銘柄分析完了\n"
        f"⚪ アクション対象なし（全銘柄 HOLD）"
    )


def _format_signal_line(result: ScoringResult) -> str:
    """1銘柄のシグナル行をフォーマットする。"""
    if result.risk:
        return (
            f"  {result.ticker} {result.signal} "
            f"${result.risk.current_price:,.2f} "
            f"SL:${result.risk.stop_loss:,.2f} "
            f"TP:${result.risk.take_profit:,.2f}"
        )
    return f"  {result.ticker} {result.signal} (価格情報なし)"


def _truncate_message(
    buy_list: list[ScoringResult],
    sell_list: list[ScoringResult],
    total_count: int,
) -> str:
    """文字数制限に収まるように短縮版メッセージを構築する。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [
        f"\n📊 Swing Signal [{now}]",
        f"全{total_count}銘柄分析",
        "",
    ]

    # 上位5件ずつに制限
    if buy_list:
        lines.append(f"🟢 BUY ({len(buy_list)})")
        for r in buy_list[:5]:
            lines.append(f"  {r.ticker} {r.signal} ${r.risk.current_price:,.0f}" if r.risk else f"  {r.ticker} {r.signal}")
        if len(buy_list) > 5:
            lines.append(f"  ... 他{len(buy_list) - 5}銘柄")
        lines.append("")

    if sell_list:
        lines.append(f"🔴 SELL ({len(sell_list)})")
        for r in sell_list[:5]:
            lines.append(f"  {r.ticker} {r.signal} ${r.risk.current_price:,.0f}" if r.risk else f"  {r.ticker} {r.signal}")
        if len(sell_list) > 5:
            lines.append(f"  ... 他{len(sell_list) - 5}銘柄")

    return "\n".join(lines)


# ============================================================
# HTTP 送信
# ============================================================

def _send(token: str, message: str) -> bool:
    """LINE Notify API にメッセージを POST する。

    Args:
        token:   LINE Notify トークン。
        message: 送信メッセージ。

    Returns:
        HTTP 200 の場合 True。
    """
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": message}

    try:
        response = requests.post(
            LINE_NOTIFY_API_URL,
            headers=headers,
            data=data,
            timeout=10,
        )

        if response.status_code == 200:
            logger.info("LINE Notify 送信成功")
            return True

        logger.warning(
            "LINE Notify 送信失敗: status=%d, body=%s",
            response.status_code,
            response.text[:200],
        )
        return False

    except requests.exceptions.Timeout:
        logger.error("LINE Notify タイムアウト")
        return False
    except requests.exceptions.ConnectionError:
        logger.error("LINE Notify 接続エラー")
        return False
    except Exception:
        logger.error("LINE Notify で予期せぬエラー", exc_info=True)
        return False
