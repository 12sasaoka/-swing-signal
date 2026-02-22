"""
Swing Trade Signal System — Claude API 分析モジュール

Anthropic Claude API を使って2種類の高精度分析を行う:
  1. analyze_earnings()  — ファンダメンタルデータから決算評価
  2. analyze_news()      — ニュースヘッドラインからセンチメント分析

統合: combined_score = earnings_score * 0.7 + news_score * 0.3

APIキー未設定・エラー時はスコア 0.0 にフォールバックし、
システム全体の実行を止めない。
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from config.settings import CLAUDE_API_CONFIG

logger = logging.getLogger(__name__)

# ============================================================
# APIクライアント初期化（遅延ロード）
# ============================================================

_client: Any = None


def _get_client() -> Any:
    """Anthropic クライアントを遅延初期化して返す。

    Returns:
        anthropic.Anthropic インスタンス。APIキー未設定時は None。
    """
    global _client
    if _client is not None:
        return _client

    api_key = CLAUDE_API_CONFIG.api_key
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY が未設定 — Claude分析はスキップされます")
        return None

    try:
        import anthropic
        _client = anthropic.Anthropic(api_key=api_key)
        logger.info("Anthropic クライアント初期化成功")
        return _client
    except ImportError:
        logger.error("anthropic パッケージが未インストール (pip install anthropic)")
        return None
    except Exception:
        logger.error("Anthropic クライアント初期化失敗", exc_info=True)
        return None


# ============================================================
# 公開 API
# ============================================================

def analyze_earnings(
    fundamentals: dict[str, Any],
    ticker: str = "",
) -> dict[str, Any]:
    """ファンダメンタルデータを Claude に送り、決算評価を得る。

    Args:
        fundamentals: ファンダメンタル指標の辞書。キー例:
            trailing_pe, forward_pe, return_on_equity, ev_to_ebitda,
            revenue_growth, debt_to_equity, current_ratio, market_cap,
            free_cashflow 等。
        ticker: ティッカーシンボル（プロンプトの文脈情報として使用）。

    Returns:
        分析結果の辞書:
            - signal (str):  "BUY" / "HOLD" / "SELL"
            - score (float): -1.0 〜 +1.0
            - reason (str):  判定理由の要約
            - risks (list[str]): 特定されたリスク要因
    """
    default = _default_result("HOLD", 0.0, "分析不可（APIスキップ）")

    if not fundamentals:
        return default

    client = _get_client()
    if client is None:
        return default

    prompt = _build_earnings_prompt(fundamentals, ticker)

    try:
        raw = _call_api(client, prompt)
        result = _parse_analysis_response(raw)
        logger.info(
            "決算分析完了: %s signal=%s score=%.2f",
            ticker or "?", result["signal"], result["score"],
        )
        return result
    except Exception:
        logger.error("決算分析でエラー発生: %s", ticker, exc_info=True)
        return default


def analyze_news(
    headlines: list[str],
    ticker: str = "",
) -> dict[str, Any]:
    """ニュースヘッドラインを Claude に送り、センチメント分析を得る。

    Args:
        headlines: ニュースヘッドライン文字列のリスト（最大10件）。
        ticker: ティッカーシンボル（プロンプトの文脈情報として使用）。

    Returns:
        分析結果の辞書:
            - score (float):      -1.0（非常に弱気）〜 +1.0（非常に強気）
            - catalysts (list[str]): 特定されたカタリスト
            - summary (str):       センチメント判定の要約
    """
    default = {"score": 0.0, "catalysts": [], "summary": "分析不可（APIスキップ）"}

    if not headlines:
        return default

    client = _get_client()
    if client is None:
        return default

    prompt = _build_news_prompt(headlines, ticker)

    try:
        raw = _call_api(client, prompt)
        result = _parse_news_response(raw)
        logger.info("ニュース分析完了: %s score=%.2f", ticker or "?", result["score"])
        return result
    except Exception:
        logger.error("ニュース分析でエラー発生: %s", ticker, exc_info=True)
        return default


def analyze_ticker(
    fundamentals: dict[str, Any],
    headlines: list[str],
    ticker: str = "",
) -> dict[str, Any]:
    """1銘柄の決算 + ニュースを統合分析する。

    combined_score = earnings_score * 0.7 + news_score * 0.3

    Args:
        fundamentals: ファンダメンタル指標の辞書。
        headlines:    ニュースヘッドラインのリスト。
        ticker: ティッカーシンボル（プロンプトの文脈情報として使用）。

    Returns:
        統合分析結果:
            - combined_score (float): 統合スコア (-1.0 〜 +1.0)
            - earnings (dict):        決算分析結果
            - news (dict):            ニュース分析結果
    """
    earnings_result = analyze_earnings(fundamentals, ticker)
    news_result = analyze_news(headlines, ticker)

    e_weight = CLAUDE_API_CONFIG.earnings_weight
    n_weight = CLAUDE_API_CONFIG.news_weight
    combined = (
        earnings_result["score"] * e_weight
        + news_result["score"] * n_weight
    )
    combined = max(-1.0, min(1.0, combined))

    return {
        "combined_score": combined,
        "earnings": earnings_result,
        "news": news_result,
    }


def analyze_batch(
    tickers_data: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """複数銘柄を並列で統合分析する。

    Args:
        tickers_data: ティッカー → {"fundamentals": dict, "headlines": list} の辞書。

    Returns:
        ティッカー → 統合分析結果 の辞書。
    """
    if not tickers_data:
        return {}

    client = _get_client()
    if client is None:
        logger.warning("APIクライアントなし — 全銘柄をデフォルトスコアで返却")
        return {
            ticker: {
                "combined_score": 0.0,
                "earnings": _default_result("HOLD", 0.0, "APIスキップ"),
                "news": {"score": 0.0, "catalysts": [], "summary": "APIスキップ"},
            }
            for ticker in tickers_data
        }

    results: dict[str, dict[str, Any]] = {}
    max_workers = CLAUDE_API_CONFIG.max_workers

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(
                analyze_ticker,
                data.get("fundamentals", {}),
                data.get("headlines", []),
                ticker,
            ): ticker
            for ticker, data in tickers_data.items()
        }
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                results[ticker] = future.result()
            except Exception:
                logger.error("バッチ分析エラー: %s", ticker, exc_info=True)
                results[ticker] = {
                    "combined_score": 0.0,
                    "earnings": _default_result("HOLD", 0.0, "エラー"),
                    "news": {"score": 0.0, "catalysts": [], "summary": "エラー"},
                }

    logger.info("バッチ分析完了: %d 銘柄", len(results))
    return results


# ============================================================
# API 呼び出し
# ============================================================

def _call_api(client: Any, user_prompt: str) -> str:
    """Claude API を呼び出し、テキスト応答を返す。

    Args:
        client:      Anthropic クライアント。
        user_prompt: ユーザープロンプト。

    Returns:
        Claude の応答テキスト。
    """
    response = client.messages.create(
        model=CLAUDE_API_CONFIG.model,
        max_tokens=CLAUDE_API_CONFIG.max_tokens,
        messages=[{"role": "user", "content": user_prompt}],
    )
    # response.content は ContentBlock のリスト
    text_parts = []
    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    return "\n".join(text_parts)


# ============================================================
# プロンプト構築
# ============================================================

def _build_earnings_prompt(fundamentals: dict[str, Any], ticker: str = "") -> str:
    """決算分析用のプロンプトを構築する。"""
    metrics_text = "\n".join(
        f"  - {key}: {value}"
        for key, value in fundamentals.items()
        if value is not None
    )

    ticker_line = f"## 銘柄: {ticker}\n\n" if ticker else ""

    return f"""{ticker_line}以下の企業のファンダメンタル指標を分析し、投資判断を行ってください。

## 指標データ
{metrics_text}

## 回答フォーマット（必ずJSON形式で回答してください）
{{
  "signal": "BUY" or "HOLD" or "SELL",
  "score": -1.0 から +1.0 の数値（+1.0が最も強気、-1.0が最も弱気）,
  "reason": "判定理由を1〜2文で簡潔に",
  "risks": ["リスク要因1", "リスク要因2"]
}}

JSONのみを出力し、他のテキストは含めないでください。"""


def _build_news_prompt(headlines: list[str], ticker: str = "") -> str:
    """ニュースセンチメント分析用のプロンプトを構築する。"""
    headlines_text = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(headlines[:10]))

    ticker_line = f"## 銘柄: {ticker}\n\n" if ticker else ""

    return f"""{ticker_line}以下のニュースヘッドラインから、この銘柄に対する市場センチメントを分析してください。

## ニュースヘッドライン
{headlines_text}

## 回答フォーマット（必ずJSON形式で回答してください）
{{
  "score": -1.0 から +1.0 の数値（+1.0が非常に楽観、-1.0が非常に悲観）,
  "catalysts": ["主要カタリスト1", "主要カタリスト2"],
  "summary": "センチメント判定の要約を1文で"
}}

JSONのみを出力し、他のテキストは含めないでください。"""


# ============================================================
# レスポンスパース
# ============================================================

def _parse_analysis_response(raw: str) -> dict[str, Any]:
    """決算分析レスポンスをパースする。

    JSONパース失敗時はデフォルト値にフォールバック。
    """
    parsed = _extract_json(raw)
    if parsed is None:
        logger.warning("決算分析のJSONパース失敗: %s", raw[:200])
        return _default_result("HOLD", 0.0, "パース失敗")

    signal = str(parsed.get("signal", "HOLD")).upper()
    if signal not in ("BUY", "HOLD", "SELL"):
        signal = "HOLD"

    score = _clamp_score(parsed.get("score", 0.0))
    reason = str(parsed.get("reason", ""))
    risks = parsed.get("risks", [])
    if not isinstance(risks, list):
        risks = []

    return {
        "signal": signal,
        "score": score,
        "reason": reason,
        "risks": [str(r) for r in risks],
    }


def _parse_news_response(raw: str) -> dict[str, Any]:
    """ニュース分析レスポンスをパースする。"""
    parsed = _extract_json(raw)
    if parsed is None:
        logger.warning("ニュース分析のJSONパース失敗: %s", raw[:200])
        return {"score": 0.0, "catalysts": [], "summary": "パース失敗"}

    score = _clamp_score(parsed.get("score", 0.0))
    catalysts = parsed.get("catalysts", [])
    if not isinstance(catalysts, list):
        catalysts = []
    summary = str(parsed.get("summary", ""))

    return {
        "score": score,
        "catalysts": [str(c) for c in catalysts],
        "summary": summary,
    }


# ============================================================
# ユーティリティ
# ============================================================

def _extract_json(text: str) -> dict[str, Any] | None:
    """テキストから JSON オブジェクトを抽出する。

    Markdown コードブロック (```json ... ```) にも対応。
    """
    if not text:
        return None

    # コードブロック除去
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.strip()

    # 最初の { から最後の } を抽出
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    json_str = cleaned[start : end + 1]

    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None


def _clamp_score(value: Any) -> float:
    """値を float に変換し、-1.0 〜 +1.0 にクランプする。"""
    try:
        f = float(value)
    except (ValueError, TypeError):
        return 0.0
    if f != f:  # NaN check
        return 0.0
    return max(-1.0, min(1.0, f))


def _default_result(signal: str, score: float, reason: str) -> dict[str, Any]:
    """デフォルトの分析結果を生成する。"""
    return {
        "signal": signal,
        "score": score,
        "reason": reason,
        "risks": [],
    }
