"""
Swing Trade Signal System — 設定・定数定義

システム全体で使用する閾値・ウェイト・パラメータを一元管理する。
値を変更する場合はこのファイルのみを編集すること。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# パス定義
# ============================================================

# プロジェクトルート: swing_signal/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data" / "db"
OUTPUT_DIR: Path = PROJECT_ROOT / "output" / "results"
SQLITE_DB_PATH: Path = DATA_DIR / "trading.db"


# ============================================================
# トレードルール
# ============================================================

@dataclass(frozen=True)
class TradeRules:
    """売買判定に使う基本ルール。"""

    max_position_pct: float = 0.05          # 1銘柄最大ポジション (5%)
    max_sector_pct: float = 0.25            # 1セクター最大ポジション (25%)
    hard_stop_loss_pct: float = -0.06       # ハードストップ損切り (-6%)
    hard_take_profit_pct: float = 0.20      # ハード利確 (+20%)
    buy_threshold: float = 0.7              # BUYシグナル閾値 (スコア ≥ 0.7)
    sell_threshold: float = -0.3            # SELLシグナル閾値 (スコア ≤ -0.3)


# ============================================================
# シグナル判定閾値
# ============================================================

@dataclass(frozen=True)
class SignalThresholds:
    """統合スコアからシグナルを判定する閾値。"""

    strong_buy: float = 0.8     # STRONG_BUY: score ≥ 0.8
    buy: float = 0.65           # BUY:        0.65 ≤ score < 0.8
    hold_upper: float = 0.5     # HOLD:       -0.3 < score < 0.5
    hold_lower: float = -0.3
    sell: float = -0.3          # SELL:        -0.6 ≤ score ≤ -0.3
    strong_sell: float = -0.6   # STRONG_SELL: score < -0.6


# ============================================================
# ファクターウェイト
# ============================================================

@dataclass(frozen=True)
class FactorWeights:
    """4ファクターの統合ウェイト（合計 1.0）。"""

    momentum: float = 0.65
    value: float = 0.00
    quality: float = 0.35
    sentiment: float = 0.00

    def __post_init__(self) -> None:
        total = self.momentum + self.value + self.quality + self.sentiment
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"ファクターウェイトの合計が1.0ではありません: {total}")


# ============================================================
# テクニカル指標パラメータ
# ============================================================

@dataclass(frozen=True)
class TechnicalParams:
    """テクニカル分析で使用する期間・パラメータ。"""

    # 移動平均
    sma_short: int = 20             # 短期SMA (日)
    sma_long: int = 50              # 長期SMA (日)
    sma_200: int = 200              # 超長期SMA (日)

    # RSI
    rsi_period: int = 14            # RSI計算期間 (日)

    # MACD
    macd_fast: int = 12             # MACD Fast EMA
    macd_slow: int = 26             # MACD Slow EMA
    macd_signal: int = 9            # MACD Signal EMA

    # ROC (Rate of Change)
    roc_period: int = 10            # ROC計算期間 (日)

    # データ取得
    price_history_days: int = 365   # 株価履歴ウィンドウ (日)
    news_lookback_days: int = 7     # ニュース参照期間 (日)


# ============================================================
# ATR・リスク管理パラメータ
# ============================================================

@dataclass(frozen=True)
class ATRParams:
    """ATR（Average True Range）ベースのリスク管理設定。"""

    period: int = 14                # ATR計算期間 (日)
    stop_loss_multiplier: float = 2.0   # SL = 現在値 − ATR × この値
    take_profit_multiplier: float = 4.0 # TP = 現在値 + ATR × この値


# ============================================================
# Claude API 設定
# ============================================================

@dataclass(frozen=True)
class ClaudeAPIConfig:
    """Anthropic Claude API の接続設定。"""

    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 1024
    max_workers: int = 4            # 並列APIコール数
    earnings_weight: float = 0.7    # 決算分析スコアのウェイト
    news_weight: float = 0.3        # ニュース分析スコアのウェイト

    @property
    def api_key(self) -> str:
        """環境変数から API キーを取得。未設定時は空文字を返す。"""
        return os.environ.get("ANTHROPIC_API_KEY", "")


# ============================================================
# データ取得設定
# ============================================================

@dataclass(frozen=True)
class FetcherConfig:
    """データ取得の並列処理・バッチ設定。"""

    price_batch_size: int = 100     # yfinance 1バッチあたりの銘柄数
    max_workers: int = 8            # ThreadPoolExecutor ワーカー数
    max_news_articles: int = 10     # 銘柄あたりの最大ニュース件数
    cache_fundamentals_days: int = 7    # ファンダメンタルキャッシュ有効期間 (日)
    cache_price_hours: int = 12         # 株価キャッシュ有効期間 (時間)


# ============================================================
# セクター別ベンチマーク（バリューファクター用）
# ============================================================

SECTOR_BENCHMARKS: dict[str, dict[str, float]] = {
    "semiconductor": {"ev_ebitda": 25.0, "pb": 5.0},
    "energy":        {"ev_ebitda": 8.0,  "pb": 1.8},
    "aerospace":     {"ev_ebitda": 15.0, "pb": 3.5},
    "biotech":       {"ev_ebitda": 20.0, "pb": 4.0},
    "mining":        {"ev_ebitda": 10.0, "pb": 2.0},
    "datacenter":    {"ev_ebitda": 20.0, "pb": 3.0},
    "etf":           {"ev_ebitda": 15.0, "pb": 3.0},
    "default":       {"ev_ebitda": 15.0, "pb": 3.0},
}


# ============================================================
# LINE Notify 設定
# ============================================================

LINE_NOTIFY_API_URL: str = "https://notify-api.line.me/api/notify"


def get_line_token() -> str:
    """環境変数から LINE Notify トークンを取得。"""
    return os.environ.get("LINE_NOTIFY_TOKEN", "")


# ============================================================
# デフォルトインスタンス（インポート用）
# ============================================================

TRADE_RULES = TradeRules()
SIGNAL_THRESHOLDS = SignalThresholds()
FACTOR_WEIGHTS = FactorWeights()
TECHNICAL_PARAMS = TechnicalParams()
ATR_PARAMS = ATRParams()
CLAUDE_API_CONFIG = ClaudeAPIConfig()
FETCHER_CONFIG = FetcherConfig()
