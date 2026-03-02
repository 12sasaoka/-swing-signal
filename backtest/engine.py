"""
Swing Trade Signal System — バックテストエンジン

2種類のバックテストを提供する:

1. シグナルバックテスト (run_signal_backtest)
   - 株価データでウォークフォワード分析（デフォルト5年）
   - BUY/STRONG_BUY シグナル発生後、SLヒットまたは段階的タイムアウトでポジションを決済
   - 固定TPなし — 利益を伸ばすため段階的タイムアウトで保有期間を制御
   - 段階的タイムアウト:
     - 30営業日: 含み損（< 0%）→ 強制決済、含み益なら保有継続
     - 90営業日: 無条件で強制決済（絶対タイムアウト）
   - 勝率・平均損益・Sharpe比・最大ドローダウンを集計
   - 市場環境フィルター: SPY終値 < SPY 200日MA のときはBUYシグナルを停止
   - 同時ポジション上限: 最大10銘柄まで（集中リスク抑制）

2. スクリーニングバックテスト (run_screening_backtest)
   - 月次でモメンタムスコア上位銘柄を選択
   - 翌月の実績リターンで「選ばれた銘柄 vs 選ばれなかった銘柄」を比較

注意事項:
- ルックアヘッドバイアス防止: スコアリング時は「その日まで」のデータのみ使用
- ファンダメンタル: 時系列データが取得困難なため現時点の値を全期間に適用
- Sentiment: バックテスト中はウェイトを0にしてMomentum/Qualityに再分配
- SL: ATRベースの動的SL（ATR×2.0、-8%フロア）
- TP: 固定TPなし — トレーリングストップ（最高値 - ATR×3.5）+ 段階的タイムアウト
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
from bisect import bisect_left
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# Phase1 キャッシュ
# ============================================================

import hashlib
import pickle
from pathlib import Path as _Path

_PHASE1_CACHE_DIR = _Path(__file__).resolve().parent.parent / "data" / "db"
_PHASE1_CACHE_FILE = _PHASE1_CACHE_DIR / "phase1_cache.pkl"


def _compute_phase1_cache_key(tickers: list, has_quarterly: bool) -> str:
    """Phase1キャッシュキーを計算する。

    主要パラメータ + ティッカーリストのハッシュ。
    パラメータ or 銘柄リストが変わると自動的にキャッシュが無効化される。
    新しい株価データを取得した場合はキャッシュファイルを手動削除すること:
      data/db/phase1_cache.pkl
    """
    import config.settings as _cfg

    key_parts = "|".join([
        str(has_quarterly),
        str(WARMUP_DAYS),
        str(VOLUME_FILTER_MULTIPLIER),
        str(VOLUME_FILTER_PERIOD),
        str(DIP_FILTER_ENABLED),
        str(DIP_MIN_PCT),
        str(DIP_MAX_PCT),
        str(OVERBOUGHT_5D_THRESHOLD),
        str(SPY_HOT_5D_THRESHOLD),
        str(POST_EARNINGS_WINDOW_DAYS),
        str(POST_EARNINGS_MIN_DROP),
        str(POST_EARNINGS_SCORE_BONUS),
        str(_cfg.SIGNAL_THRESHOLDS.buy),
        str(_cfg.SIGNAL_THRESHOLDS.strong_buy),
        ",".join(sorted(tickers)),
    ])
    return hashlib.md5(key_parts.encode()).hexdigest()


def _load_phase1_cache(cache_key: str) -> list | None:
    """キャッシュが有効なら signal_candidates を返す。なければ None。"""
    if not _PHASE1_CACHE_FILE.exists():
        return None
    try:
        with open(_PHASE1_CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
        if cached.get("key") == cache_key:
            candidates = cached["candidates"]
            logger.info("Phase1キャッシュヒット: %d 件を読み込み (キー=%s...)",
                        len(candidates), cache_key[:8])
            return candidates
    except Exception:
        pass
    return None


def _save_phase1_cache(cache_key: str, candidates: list) -> None:
    """signal_candidates（df フィールドなし）をキャッシュに保存する。"""
    try:
        # df フィールドはサイズが大きく price_data から復元可能なので除外
        slim = [{k: v for k, v in c.items() if k != "df"} for c in candidates]
        with open(_PHASE1_CACHE_FILE, "wb") as f:
            pickle.dump({"key": cache_key, "candidates": slim}, f, protocol=4)
        logger.info("Phase1キャッシュ保存: %d 件 → %s", len(slim), _PHASE1_CACHE_FILE)
    except Exception as e:
        logger.warning("Phase1キャッシュ保存失敗: %s", e)


# ============================================================
# Phase1 並列処理ワーカー
# ============================================================

_worker_shared: dict = {}


def _init_phase1_worker(
    spy_df, market_ok, warmup_days, earnings_dates,
    quarterly_data=None, spy_not_hot=None, sector_momentum=None,
):
    """Phase1ワーカーの初期化（プロセスごとに1回実行）。

    spy_df と market_ok はワーカー間で共有（spawn時に各プロセスへコピー）。
    quarterly_data が渡された場合は時系列対応のウェイト（M=65%, Q=35%）を設定する。
    quarterly_data が None の場合は Momentum=100% フォールバック。
    earnings_dates: ticker → sorted list of earnings date strings (YYYY-MM-DD)
    spy_not_hot: 日付 → True(エントリーOK)/False(過熱=回避) 【案7】
    sector_momentum: {YYYY-MM: {sector: bonus_score}} 【案4】
    """
    import config.settings as cfg
    if quarterly_data:
        # 時系列 Quality 有効: M=65%, Q=35%
        # Sentiment はバックテストでは 0%（ライブでは Claude AI が担当）
        cfg.FACTOR_WEIGHTS = cfg.FactorWeights(
            momentum=0.65, value=0.00, quality=0.35, sentiment=0.00,
        )
    else:
        # フォールバック: Momentum のみ（ルックアヘッドバイアス回避）
        cfg.FACTOR_WEIGHTS = cfg.FactorWeights(
            momentum=1.00, value=0.00, quality=0.00, sentiment=0.00,
        )
    _worker_shared["spy_df"] = spy_df
    _worker_shared["market_ok"] = market_ok
    _worker_shared["warmup_days"] = warmup_days
    _worker_shared["earnings_dates"] = earnings_dates or {}
    _worker_shared["quarterly_data"] = quarterly_data or {}
    _worker_shared["spy_not_hot"] = spy_not_hot or {}      # 案7
    _worker_shared["sector_momentum"] = sector_momentum or {}  # 案4


def _score_ticker_phase1(args):
    """1銘柄の全日付をスコアリングし、シグナル候補リストを返す。

    メインプロセスの score_ticker と同じロジックだが、
    multiprocessing 用にモジュールレベル関数として定義。
    """
    ticker, df, sector, fundamentals = args
    spy_df = _worker_shared["spy_df"]
    market_ok = _worker_shared["market_ok"]
    warmup_days = _worker_shared["warmup_days"]

    from strategy.scorer import score_ticker as _score_ticker

    BUY_SIG = "BUY"
    STRONG_BUY_SIG = "STRONG_BUY"

    candidates = []

    try:
        if df is None or df.empty or len(df) < warmup_days + 5:
            return candidates

        trading_dates = df.index.tolist()

        for i in range(warmup_days, len(trading_dates)):
            current_date = trading_dates[i]
            current_date_str = current_date.strftime("%Y-%m-%d")

            # 市場環境フィルター
            if market_ok and not market_ok.get(current_date_str, True):
                continue

            # 翌営業日がなければスキップ
            if i + 1 >= len(trading_dates):
                continue

            # スコアリング
            window_df = df.iloc[: i + 1]
            try:
                # 時系列対応: その日時点で公表済みの最新四半期データを使用
                quarterly_data = _worker_shared.get("quarterly_data", {})
                if quarterly_data:
                    from data.quarterly_fetcher import get_fundamentals_as_of
                    hist_fund = get_fundamentals_as_of(ticker, current_date_str, quarterly_data)
                else:
                    hist_fund = fundamentals

                result = _score_ticker(
                    ticker=ticker,
                    sector=sector,
                    price_df=window_df,
                    fundamentals=hist_fund,
                    headlines=None,
                    claude_score=None,  # バックテストは Sentiment=0%（ライブは Claude AI）
                    spy_df=spy_df,
                )

                # ---- 動的重み正規化 ----
                # quality データがない期間（2019年など）は
                # quality の重みをモメンタムに再配分して閾値スケールを維持する
                # quality あり: M=65%, Q=35%  → composite スケール維持
                # quality なし: M=65/65=100%  → Momentum のみ（スケール維持）
                if quarterly_data and hist_fund is None:
                    import numpy as _np
                    from strategy.scorer import determine_signal as _det_sig
                    # quality データなし → Momentum=100% で再計算
                    result.final_score = float(_np.clip(result.momentum_score, -1.0, 1.0))
                    result.signal = _det_sig(result.final_score)
            except Exception:
                continue

            # ---- 案10: 決算後ディップ優先エントリー ----
            # りおぽん: 「決算で売られたが内容は悪くない銘柄は良い押し目」
            # 落ちるナイフ誤買い防止: 反発陽線 + 出来高急増を必須条件とする
            earnings_map = _worker_shared.get("earnings_dates", {})
            ticker_earnings = earnings_map.get(ticker, [])
            if ticker_earnings and len(window_df) >= 2:
                ei_post = bisect_left(ticker_earnings, current_date_str)
                if ei_post > 0:
                    last_earn_str = ticker_earnings[ei_post - 1]
                    days_since = (pd.Timestamp(current_date_str) - pd.Timestamp(last_earn_str)).days
                    if 7 <= days_since <= POST_EARNINGS_WINDOW_DAYS:
                        # 決算前日終値を取得
                        try:
                            earn_dt = pd.Timestamp(last_earn_str)
                            pre_earn_df = df.loc[:earn_dt - pd.Timedelta(days=1)]
                            if len(pre_earn_df) >= 1:
                                pre_earn_close = float(pre_earn_df["Close"].iloc[-1])
                                cur_close = float(window_df["Close"].iloc[-1])
                                drop_pct = (cur_close - pre_earn_close) / pre_earn_close
                                if drop_pct <= POST_EARNINGS_MIN_DROP:
                                    # 反発確認: 本日終値 > 前日終値
                                    is_recovering = cur_close > float(window_df["Close"].iloc[-2])
                                    # 出来高急増確認
                                    is_high_vol = False
                                    if "Volume" in window_df.columns and len(window_df) > VOLUME_FILTER_PERIOD:
                                        cur_vol = float(window_df["Volume"].iloc[-1])
                                        avg_vol_post = float(window_df["Volume"].iloc[-(VOLUME_FILTER_PERIOD + 1):-1].mean())
                                        is_high_vol = avg_vol_post > 0 and cur_vol >= avg_vol_post * POST_EARNINGS_VOL_MULT
                                    if is_recovering and is_high_vol:
                                        # 条件を満たせばスコアボーナス
                                        result.final_score = float(np.clip(
                                            result.final_score + POST_EARNINGS_SCORE_BONUS, -1.0, 1.0
                                        ))
                                        from strategy.scorer import determine_signal as _det_sig_pe
                                        result.signal = _det_sig_pe(result.final_score)
                        except Exception:
                            pass

            # ---- 案4: セクターモメンタムボーナス ----
            # りおぽん: 「勝ち馬セクターに乗る」
            sector_mom_map = _worker_shared.get("sector_momentum", {})
            if sector_mom_map:
                current_month = current_date_str[:7]
                # 最新の月データを逆順検索
                for past_month in sorted(sector_mom_map.keys(), reverse=True):
                    if past_month <= current_month:
                        sector_bonus = sector_mom_map[past_month].get(sector, 0.0)
                        if sector_bonus != 0.0:
                            result.final_score = float(np.clip(
                                result.final_score + sector_bonus, -1.0, 1.0
                            ))
                            from strategy.scorer import determine_signal as _det_sig_sec
                            result.signal = _det_sig_sec(result.final_score)
                        break

            if result.signal not in (BUY_SIG, STRONG_BUY_SIG):
                continue
            if result.risk is None:
                continue

            # ---- 出来高確認フィルター ----
            # シグナル日の出来高が過去N日平均の VOLUME_FILTER_MULTIPLIER 倍以上か確認
            if (
                VOLUME_FILTER_MULTIPLIER > 0
                and "Volume" in window_df.columns
                and len(window_df) > VOLUME_FILTER_PERIOD
            ):
                current_vol = float(window_df["Volume"].iloc[-1])
                avg_vol = float(window_df["Volume"].iloc[-(VOLUME_FILTER_PERIOD + 1):-1].mean())
                if avg_vol > 0 and current_vol < avg_vol * VOLUME_FILTER_MULTIPLIER:
                    continue  # 出来高不足 → スキップ

            # ---- 案7: 市場ディップ日スコアボーナス ----
            # りおぽん: 「好調な上昇相場では買わない。ディップの日に仕込む」
            # ※ハードフィルターではなくスコアボーナスに変更（機会損失を防ぐ）
            # SPYが5日で下落している日はエントリー品質が高い → ボーナス付与
            spy_not_hot_map = _worker_shared.get("spy_not_hot", {})
            if spy_not_hot_map and len(window_df) >= 6:
                # SPYが下落中(not hot)の日はスコアボーナス
                if not spy_not_hot_map.get(current_date_str, True):
                    # 過熱市場(SPY +5%超): 軽くペナルティ（完全ブロックはしない）
                    result.final_score = float(np.clip(result.final_score - 0.05, -1.0, 1.0))
                    from strategy.scorer import determine_signal as _det_sig_spy
                    result.signal = _det_sig_spy(result.final_score)

            # ---- 案9: 過熱回避スコアボーナス（銘柄レベル）----
            # りおぽん: 「急上昇中の銘柄にも触らない」
            # ※ハードフィルターではなくペナルティに変更（完全排除はしない）
            if len(window_df) >= 6:
                close_now = float(window_df["Close"].iloc[-1])
                close_5d = float(window_df["Close"].iloc[-6])
                if close_5d > 0:
                    ret_5d_stock = (close_now - close_5d) / close_5d
                    if ret_5d_stock >= OVERBOUGHT_5D_THRESHOLD:
                        # 急騰中: スコアペナルティ（完全ブロックはしない）
                        result.final_score = float(np.clip(result.final_score - 0.08, -1.0, 1.0))
                        from strategy.scorer import determine_signal as _det_sig_ob
                        result.signal = _det_sig_ob(result.final_score)

            # ---- 案1: ディップ買いスコアボーナス ----
            # りおぽん: 「直近高値から程よく引いた押し目で仕込む」
            # ※ハードフィルターではなくスコアボーナスに変更（機会損失を防ぐ）
            # ディップゾーン(3-15%下落)にある株は「旬を維持しつつ割安感あり」
            if DIP_FILTER_ENABLED and len(window_df) >= DIP_LOOKBACK:
                rolling_high = float(window_df["Close"].iloc[-DIP_LOOKBACK:].max())
                cur_close_dip = float(window_df["Close"].iloc[-1])
                if rolling_high > 0:
                    dip_ratio = (rolling_high - cur_close_dip) / rolling_high
                    if DIP_MIN_PCT <= dip_ratio <= 0.15:
                        # 理想的なディップゾーン: ボーナス
                        result.final_score = float(np.clip(result.final_score + 0.05, -1.0, 1.0))
                        from strategy.scorer import determine_signal as _det_sig_dip
                        result.signal = _det_sig_dip(result.final_score)
                    elif dip_ratio > DIP_MAX_PCT:
                        # 急落しすぎ(20%超): 軽くペナルティ
                        result.final_score = float(np.clip(result.final_score - 0.05, -1.0, 1.0))
                        from strategy.scorer import determine_signal as _det_sig_crash
                        result.signal = _det_sig_crash(result.final_score)

            # スコア調整後に再チェック（ペナルティでHOLDに格下がりした場合はスキップ）
            if result.signal not in (BUY_SIG, STRONG_BUY_SIG):
                continue

            entry_row = df.iloc[i + 1]
            entry_price = float(entry_row["Open"])
            if entry_price <= 0:
                continue

            # 決算前エントリー禁止チェック
            # Phase2の強制決済閾値（5カレンダー日）と一致させ、同日Entry+Exit問題を防ぐ
            if ticker_earnings:
                entry_date_chk = trading_dates[i + 1].strftime("%Y-%m-%d")
                ei = bisect_left(ticker_earnings, entry_date_chk)
                if ei < len(ticker_earnings):
                    days_to_e = (
                        pd.Timestamp(ticker_earnings[ei]) - pd.Timestamp(entry_date_chk)
                    ).days
                    if days_to_e <= 5:  # Phase2強制決済と同じ5カレンダー日以内は禁止
                        continue

            candidates.append({
                "date": current_date_str,
                "entry_date_str": trading_dates[i + 1].strftime("%Y-%m-%d"),
                "ticker": ticker,
                "signal": result.signal,
                "score": result.final_score,
                "atr": result.risk.atr,
                "entry_price": entry_price,
                "df_idx": i,
            })
    except Exception:
        pass  # ワーカーエラーは握り潰し（他銘柄は継続）

    return candidates


@contextmanager
def _backtest_weights():
    """バックテスト中はMomentum 100%に設定する。

    Quality/Sentimentは現時点のファンダメンタルデータに依存するため、
    過去バックテストではルックアヘッドバイアスになる。
    Momentumは純粋な価格データのみで算出されるため唯一クリーンに使える。

    通常: Momentum=0.50, Value=0.00, Quality=0.30, Sentiment=0.20
    BT時: Momentum=1.00, Value=0.00, Quality=0.00, Sentiment=0.00
    """
    import config.settings as cfg

    original = cfg.FACTOR_WEIGHTS
    cfg.FACTOR_WEIGHTS = cfg.FactorWeights(
        momentum=1.00,
        value=0.00,
        quality=0.00,
        sentiment=0.00,
    )
    try:
        yield
    finally:
        cfg.FACTOR_WEIGHTS = original

# 段階的タイムアウト設定
# (経過営業日, 最低含み益%) — 含み益が基準未満なら強制決済
# 【案3: りおぽん「月7%最低ライン」準拠 + 早期パフォーマンスカット追加】
STAGED_TIMEOUT = [
    (30, 0.00),   # 30営業日: 含み損（< 0%）なら決済（継続監視）
    # (15, 0.05) は削除 — 優良トレードを早期に切りすぎて平均損益を悪化させた
]
# 絶対タイムアウト（無条件で強制決済）
# 【案11: 旬の賞味期限は最長3ヶ月 → 90日→60日】
MAX_HOLD_DAYS = 60

# ── STRONG_BUY 専用の拡張保有設定 ──────────────────────────────────
# 地力が強いシグナルは長く持つことで大きな利益を狙う
# BUY は通常ルール、STRONG_BUY だけここで上書き
SB_MAX_HOLD_DAYS          = 90    # 最大保有: 60日 → 90日
SB_TRAILING_ATR_MULT      = 4.0   # トレーリング幅（デフォルト）。run_signal_backtest の sb_trail_mult で上書き可
SB_STAGED_TIMEOUT_ENABLED = True   # 30日タイムアウト: BUY と同様に有効のまま
# ────────────────────────────────────────────────────────────────────

# トレーリングストップ（シャンデリア・エグジット）デフォルト倍率
# 含み益が少ない初期段階ではこの倍率を使用
TRAILING_STOP_ATR_MULTIPLIER = 3.5

# 段階的トレーリングストップ: (最小含み益 ATR倍率, trail 倍率)
# 含み益が閾値以上になったら trail 倍率を縮小して利益をロック
# 降順で定義（大きい含み益から優先評価）
PROGRESSIVE_TRAIL_LEVELS: list[tuple[float, float]] = [
    (6.0, 2.0),  # 含み益 ≥ 6.0 ATR → 最高値 − ATR×2.0（タイト）
    (4.0, 2.5),  # 含み益 ≥ 4.0 ATR → 最高値 − ATR×2.5
    (2.0, 3.0),  # 含み益 ≥ 2.0 ATR → 最高値 − ATR×3.0
]

# ブレイクイーブン移動トリガー: 含み益がこの ATR 倍率以上で SL ≥ エントリー価格を保証
BREAKEVEN_TRIGGER_ATR: float = 1.0
# ブレイクイーブンラチェット: トリガー後のロック水準 = entry_price + atr × この値
# 0.0 = 従来通り 0% ロック、0.5 = +0.5ATR（約+1〜2%）でロック
BREAKEVEN_LOCK_ATR: float = 1.0

# 【案6: 直近高値ベーストレーリング】
# りおぽん: 「直近高値から5%以上下落したら利確検討」
# ユーザー懸念対応: ボラ考慮 → max(5%, ATR×1.5÷高値) のルームを確保
PEAK_TRAIL_MIN_PCT: float = 0.05    # 最低5%ルーム（安定株）
PEAK_TRAIL_ATR_MULT: float = 1.5   # ATRベース: 1.5ATR（高ボラ株はこちらが広くなる）

# 出来高確認フィルター（エントリーシグナル時点）
VOLUME_FILTER_MULTIPLIER: float = 1.35   # 当日出来高 ≥ N日平均 × 1.35
VOLUME_FILTER_PERIOD: int = 20           # 平均算出期間（営業日）

# ウォームアップ日数（200日MAのため最低200日必要）
WARMUP_DAYS = 200

# 同時保有ポジション上限（ドローダウン抑制）
# 【案12: りおぽん「10-15銘柄分散」→ 12に拡大】
MAX_CONCURRENT_POSITIONS = 12

# 市場環境フィルター: SPY 200日MA を下回ったら BUY シグナルを停止
MARKET_FILTER_MA_PERIOD = 200

# 【案1: ディップ買いフィルター】
# りおぽん: 「好調な上昇相場では買わない。ディップの日に仕込む」
# 直近N日高値からの下落率が DIP_MIN_PCT〜DIP_MAX_PCT の範囲のみエントリー許可
DIP_FILTER_ENABLED: bool = True
DIP_LOOKBACK: int = 20          # 高値計算のルックバック（営業日）
DIP_MIN_PCT: float = 0.03       # 最低3%下落（高値近辺はスキップ）
DIP_MAX_PCT: float = 0.20       # 最大20%下落（急落すぎはスキップ）

# 【案7: 市場過熱フィルター】
# りおぽん: 「好調な上昇相場では買わない」
# SPY直近5日リターンがこの閾値超なら過熱市場 → エントリー回避
SPY_HOT_5D_THRESHOLD: float = 0.05   # SPY 5日リターン > 5% は過熱

# 【案9: 過熱回避フィルター】
# りおぽん: 「急上昇中の銘柄にも触らない」
# ユーザー懸念対応: 案1(ディップ)との競合を避けるため5日リターンで判定
OVERBOUGHT_5D_THRESHOLD: float = 0.10   # 5日で10%超上昇はスキップ

# 【案10: 決算後ディップ優先エントリー】
# りおぽん: 「決算で売られたが内容は悪くない銘柄は良い押し目」
# ユーザー懸念対応: 出来高急増 + 反発陽線を必須条件として「落ちるナイフ」誤買いを防止
POST_EARNINGS_WINDOW_DAYS: int = 21    # 決算後21カレンダー日以内
POST_EARNINGS_MIN_DROP: float = -0.08 # 最低8%の決算後下落が必要
POST_EARNINGS_SCORE_BONUS: float = 0.05  # スコアボーナス
POST_EARNINGS_VOL_MULT: float = 1.50   # 出来高1.5倍以上（機関の押し目買い確認）必要


# ============================================================
# データクラス
# ============================================================

@dataclass
class TradeResult:
    """1トレードの結果。"""
    ticker: str
    signal: str             # "STRONG_BUY" | "BUY"
    signal_date: str        # シグナル発生日
    entry_date: str         # エントリー日（翌営業日）
    entry_price: float
    exit_date: str
    exit_price: float
    result: str             # "sl_hit" | "trailing_stop" | "timeout_30d" | "timeout_90d" | "pre_earnings" | "timeout"
    pl_pct: float           # 損益率 (-1.0 〜 +∞)
    sl_price: float
    tp_price: float
    score: float = 0.0      # シグナル発生時のスコア (0.65〜1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "signal": self.signal,
            "score": round(self.score, 4),
            "signal_date": self.signal_date,
            "entry_date": self.entry_date,
            "entry_price": round(self.entry_price, 4),
            "exit_date": self.exit_date,
            "exit_price": round(self.exit_price, 4),
            "result": self.result,
            "pl_pct": round(self.pl_pct * 100, 2),
            "sl_price": round(self.sl_price, 4),
            "tp_price": round(self.tp_price, 4),
        }


@dataclass
class SignalBacktestResult:
    """シグナルバックテストの集計結果。"""
    trades: list[TradeResult] = field(default_factory=list)
    total_trades: int = 0
    win_rate: float = 0.0          # 勝率 (pl_pct > 0 の割合)
    avg_pl_pct: float = 0.0        # 平均損益率 (%)
    total_pl_pct: float = 0.0      # 累計損益率 (%)
    max_drawdown: float = 0.0      # 最大ドローダウン (%)
    sharpe_ratio: float = 0.0      # Sharpe比 (年率換算)
    by_signal: dict[str, dict] = field(default_factory=dict)   # シグナル別集計
    by_year: dict[str, dict] = field(default_factory=dict)     # 年別集計
    by_result: dict[str, int] = field(default_factory=dict)    # sl_hit/timeout_30d/60d/90d 別件数


@dataclass
class MonthlyRecord:
    """スクリーニングバックテストの1ヶ月分記録。"""
    month: str              # "YYYY-MM"
    selected_tickers: list[str]
    selected_return: float  # 選択銘柄の翌月平均リターン
    all_return: float       # 全銘柄の翌月平均リターン
    outperformance: float   # selected - all


@dataclass
class ScreeningBacktestResult:
    """スクリーニングバックテストの集計結果。"""
    monthly_records: list[MonthlyRecord] = field(default_factory=list)
    avg_selected_return: float = 0.0
    avg_all_return: float = 0.0
    avg_outperformance: float = 0.0
    win_months: int = 0     # アウトパフォームした月数
    total_months: int = 0


# ============================================================
# シグナルバックテスト
# ============================================================

def _build_market_filter(spy_df: pd.DataFrame | None) -> dict[str, bool]:
    """SPY の 200日MA フィルターを事前計算する。

    Returns:
        日付文字列 → True(BUY許可) / False(BUY禁止) の辞書。
        SPY データがない場合は空辞書（= フィルターなし）。
    """
    if spy_df is None or spy_df.empty:
        return {}

    close = spy_df["Close"].astype(float)
    ma200 = close.rolling(window=MARKET_FILTER_MA_PERIOD).mean()
    result: dict[str, bool] = {}
    for i in range(len(spy_df)):
        date_str = spy_df.index[i].strftime("%Y-%m-%d")
        if pd.isna(ma200.iloc[i]):
            result[date_str] = True  # MA算出前は許可
        else:
            result[date_str] = float(close.iloc[i]) >= float(ma200.iloc[i])
    return result


def _build_spy_hot_filter(spy_df: pd.DataFrame | None) -> dict[str, bool]:
    """【案7】SPY 5日リターンが過熱かどうかを事前計算する。

    りおぽん: 「好調な上昇相場では買わない」
    SPY が直近5日で SPY_HOT_5D_THRESHOLD(5%) 超上昇した日は過熱市場と判定。

    Returns:
        日付文字列 → True(エントリーOK) / False(過熱=回避) の辞書。
        SPY データがない場合は空辞書（= フィルターなし）。
    """
    if spy_df is None or spy_df.empty:
        return {}

    close = spy_df["Close"].astype(float)
    result: dict[str, bool] = {}
    for i in range(len(spy_df)):
        date_str = spy_df.index[i].strftime("%Y-%m-%d")
        if i < 5:
            result[date_str] = True  # データ不足は許可
        else:
            spy_5d_ret = (close.iloc[i] - close.iloc[i - 5]) / close.iloc[i - 5]
            result[date_str] = spy_5d_ret <= SPY_HOT_5D_THRESHOLD
    return result


def _build_sector_momentum(
    tickers: list[str],
    price_data: dict[str, "pd.DataFrame"],
    fund_data: dict[str, dict],
) -> dict[str, dict[str, float]]:
    """【案4】セクター別モメンタムスコアを月次で事前計算する。

    りおぽん: 「勝ち馬セクターに乗る」
    各月末時点でセクター別の直近3ヶ月中央値リターンを計算し、
    クロスセクター比較でスコア化。好調セクターの銘柄にボーナスを付与。

    Returns:
        {YYYY-MM: {sector: float}} のネストした辞書。
        float は ±0.05 の範囲に正規化されたセクターボーナス。
    """
    from config.universe import get_sector

    # セクター別の月次3ヶ月リターンを収集
    sector_monthly_rets: dict[str, dict[str, list[float]]] = {}  # {YYYY-MM: {sector: [rets]}}

    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None or df.empty or len(df) < 64:
            continue
        sector = get_sector(ticker)
        if not sector:
            continue

        close = df["Close"].astype(float)
        # 月末のみ計算（リサンプリング）— resampleはDataFrameインデックスがDatetimeIndexである必要がある
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.copy()
                df.index = pd.to_datetime(df.index)
            monthly_ends = df.resample("ME").last().index
            if len(monthly_ends) == 0:
                # フォールバック: 月次サンプリング（古いpandasのME対応）
                monthly_ends = df.resample("M").last().index
        except Exception:
            continue

        for month_end in monthly_ends:
            window = close.loc[:month_end]
            if len(window) < 64:
                continue
            ret_3m = float(window.iloc[-1] - window.iloc[-64]) / float(window.iloc[-64]) if float(window.iloc[-64]) > 0 else 0.0
            month_str = month_end.strftime("%Y-%m")
            if month_str not in sector_monthly_rets:
                sector_monthly_rets[month_str] = {}
            if sector not in sector_monthly_rets[month_str]:
                sector_monthly_rets[month_str][sector] = []
            sector_monthly_rets[month_str][sector].append(ret_3m)

    # セクター別中央値を計算し正規化
    sector_scores: dict[str, dict[str, float]] = {}
    for month_str, sector_rets in sector_monthly_rets.items():
        sector_medians = {s: float(np.median(rets)) for s, rets in sector_rets.items() if len(rets) >= 3}
        if len(sector_medians) < 2:
            continue
        all_med = float(np.median(list(sector_medians.values())))
        all_std = float(np.std(list(sector_medians.values()))) or 0.01
        scores = {}
        for sector, med in sector_medians.items():
            z = (med - all_med) / all_std
            # 最大±0.05のボーナス（メインスコアへの影響を控えめに）
            scores[sector] = float(np.clip(z * 0.02, -0.05, 0.05))
        sector_scores[month_str] = scores

    logger.info("セクターモメンタム事前計算完了: %d ヶ月分", len(sector_scores))
    return sector_scores


def _build_rolling_universe(
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
    top_n: int = 100,
    min_history_days: int = 200,
) -> dict[str, set[str]]:
    """ルックアヘッドバイアスなしで毎月の有効ユニバースを構築する。

    各月 M のユニバース = 前月末時点のモメンタムスコア上位 top_n 銘柄。
    例: 2019年2月のシグナルには「2019年1月31日までのデータ」で算出したスコアを使用。

    Args:
        tickers:          Tier1 通過済みティッカーリスト。
        price_data:       ティッカー → OHLCV DataFrame（全期間データ）。
        top_n:            各月で選択するティッカー数上限（デフォルト100）。
        min_history_days: スコア計算に必要な最低データ日数（デフォルト200）。

    Returns:
        "YYYY-MM" → その月に有効なティッカー set のマッピング。
        キーが存在しない月はフィルタ対象外（全銘柄許可）。
    """
    from analysis.momentum import calc_momentum_score

    # 月末日カレンダーを取得（最初に見つかった有効な株価データから）
    sample_df: pd.DataFrame | None = None
    for ticker in tickers:
        df = price_data.get(ticker)
        if df is not None and not df.empty and isinstance(df.index, pd.DatetimeIndex):
            sample_df = df
            break

    if sample_df is None:
        logger.warning("ローリングユニバース: 有効な株価データがありません")
        return {}

    try:
        monthly_ends = sample_df.resample("ME").last().index.tolist()
    except Exception:
        try:
            monthly_ends = sample_df.resample("M").last().index.tolist()
        except Exception:
            logger.warning("ローリングユニバース: 月次リサンプリングに失敗")
            return {}

    if not monthly_ends:
        return {}

    logger.info(
        "ローリングユニバース構築開始: %d ヶ月分 × %d 銘柄",
        len(monthly_ends), len(tickers),
    )

    rolling_universe: dict[str, set[str]] = {}

    for i, month_end in enumerate(monthly_ends):
        # 各銘柄のモメンタムスコアを「月末時点まで」のデータで計算
        scored: list[tuple[str, float]] = []
        for ticker in tickers:
            df = price_data.get(ticker)
            if df is None or df.empty:
                continue
            if not isinstance(df.index, pd.DatetimeIndex):
                continue
            # 月末時点以前のデータに切り取り（ルックアヘッドバイアス防止）
            df_window = df.loc[df.index <= month_end]
            if len(df_window) < min_history_days:
                continue
            try:
                score = calc_momentum_score(df_window)
                scored.append((ticker, score))
            except Exception:
                continue

        # スコア降順でトップ N を選択
        scored.sort(key=lambda x: x[1], reverse=True)
        top_tickers = {t for t, _ in scored[:top_n]}

        # このユニバースを「翌月」のシグナルに適用
        # （月末時点のデータ → 翌月初時点で利用可能 → 翌月のシグナルに使用）
        if i + 1 < len(monthly_ends):
            next_month_str = monthly_ends[i + 1].strftime("%Y-%m")
            rolling_universe[next_month_str] = top_tickers

        if (i + 1) % 12 == 0 or (i + 1) == len(monthly_ends):
            logger.info(
                "ローリングユニバース: %s 完了 (%d/%d ヶ月, 今月有効 %d 銘柄)",
                month_end.strftime("%Y-%m"), i + 1, len(monthly_ends), len(top_tickers),
            )

    logger.info(
        "ローリングユニバース構築完了: %d ヶ月分, 平均 %.1f 銘柄/月",
        len(rolling_universe),
        sum(len(v) for v in rolling_universe.values()) / max(1, len(rolling_universe)),
    )
    return rolling_universe


def run_signal_backtest(
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
    fund_data: dict[str, dict[str, Any]],
    spy_df: pd.DataFrame | None = None,
    earnings_dates: dict[str, list[str]] | None = None,
    quarterly_data: dict[str, dict] | None = None,
) -> SignalBacktestResult:
    """シグナルバックテストを実行する。

    Args:
        tickers:        分析対象のティッカーリスト。
        price_data:     ティッカー → OHLCV DataFrame。
        fund_data:      ティッカー → ファンダメンタル辞書（現時点の値を全期間に適用）。
        spy_df:         SPY の OHLCV DataFrame（市場環境フィルター用、Noneならフィルター無効）。
        earnings_dates:  ティッカー → 決算日リスト（YYYY-MM-DD、昇順）。
                         Noneの場合は決算フィルター無効。
        quarterly_data:  ティッカー → 四半期財務データ辞書。
                         Noneの場合は Momentum=100% フォールバック（ルックアヘッドバイアス回避）。
                         data.quarterly_fetcher.fetch_quarterly_data() の戻り値を渡すこと。

    Returns:
        SignalBacktestResult。
    """
    from config.universe import get_sector
    from strategy.scorer import score_ticker, BUY, STRONG_BUY
    from config.settings import ATR_PARAMS, TRADE_RULES

    # ---- 市場環境フィルター事前計算 ----
    market_ok = _build_market_filter(spy_df)
    if market_ok:
        bearish_days = sum(1 for v in market_ok.values() if not v)
        logger.info("市場環境フィルター: %d日中 %d日がベア判定（BUY禁止）",
                     len(market_ok), bearish_days)
    else:
        logger.info("市場環境フィルター: SPYデータなし → フィルター無効")

    # ---- 【案7】市場過熱フィルター事前計算 ----
    spy_not_hot = _build_spy_hot_filter(spy_df)
    if spy_not_hot:
        hot_days = sum(1 for v in spy_not_hot.values() if not v)
        logger.info("市場過熱フィルター(案7): %d日中 %d日が過熱判定（エントリー回避）",
                     len(spy_not_hot), hot_days)

    # ---- 【案4】セクターモメンタム事前計算 ----
    sector_momentum = _build_sector_momentum(tickers, price_data, fund_data)

    # ---- ローリングユニバース構築（ルックアヘッドバイアス防止） ----
    # 各月末のモメンタムスコアで翌月の有効銘柄セットを決定する。
    # 例: 2019年1月末のスコア上位100 → 2019年2月のシグナルのみ有効
    from config.screening import SCREENING_PARAMS as _sp
    rolling_universe = _build_rolling_universe(
        tickers, price_data,
        top_n=_sp.max_screen_candidates,
        min_history_days=WARMUP_DAYS,
    )
    logger.info("ローリングユニバース: %d ヶ月分を構築完了", len(rolling_universe))

    # ---- SL パラメータ ----
    # ATRベースの動的SL（-8%ハードストップをフロアとして適用）
    sl_multiplier = ATR_PARAMS.stop_loss_multiplier  # 2.0
    hard_stop_pct = TRADE_RULES.hard_stop_loss_pct   # -0.08
    # 固定TPなし — 段階的タイムアウトで利益を伸ばす

    all_trades: list[TradeResult] = []

    # ---- 同時ポジション管理（全銘柄横断） ----
    # 銘柄ごとに直列処理するため、日付→ポジション数のグローバルカウンタを使う
    # まず日付ごとの処理に変換する必要がある
    # → 銘柄ごとにシグナル候補を収集し、日付順にポートフォリオ全体で管理する

    # Phase 1: 銘柄ごとにシグナル候補日とスコアを事前計算（並列処理）
    signal_candidates: list[dict] = []

    # ワーカーに渡すタスクを準備
    tasks = []
    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue
        if len(df) < WARMUP_DAYS + 5:
            continue
        sector = get_sector(ticker)
        fundamentals = fund_data.get(ticker)
        tasks.append((ticker, df, sector, fundamentals))

    # ワーカー数: 論理コア数 - 1（メインプロセス用に1残す）、最低1・最大制限なし
    cpu_count = mp.cpu_count() or 4
    n_workers = max(1, cpu_count - 1)
    total_tasks = len(tasks)
    logger.info("Phase1: %d 銘柄を %d ワーカーで並列処理開始 (CPU=%d)",
                total_tasks, n_workers, cpu_count)

    # chunksize を大きめに設定してプロセス間通信オーバーヘッドを削減
    chunksize = max(1, total_tasks // (n_workers * 2))
    completed = 0

    if quarterly_data:
        logger.info("四半期財務データ: %d 銘柄分 (M=60%%, Q=30%%, S=10%%)", len(quarterly_data))
    else:
        logger.info("四半期財務データなし → Momentum=100%% フォールバック")

    with mp.Pool(
        processes=n_workers,
        initializer=_init_phase1_worker,
        initargs=(spy_df, market_ok, WARMUP_DAYS, earnings_dates or {},
                  quarterly_data or {}, spy_not_hot, sector_momentum),
    ) as pool:
        for ticker_candidates in pool.imap_unordered(
            _score_ticker_phase1, tasks, chunksize=chunksize,
        ):
            completed += 1
            if completed % 50 == 0 or completed == total_tasks:
                logger.info(
                    "Phase1 進捗: %d / %d 銘柄 (%d%%)",
                    completed, total_tasks, int(completed / total_tasks * 100),
                )
            for c in ticker_candidates:
                c["df"] = price_data[c["ticker"]]
                signal_candidates.append(c)

    logger.info("シグナル候補: %d 件を収集 (ローリングユニバースフィルタ前)", len(signal_candidates))

    # ---- ローリングユニバースフィルタ ----
    # 各シグナル候補の発生月に対応するユニバース（前月末時点のトップN）に含まれる銘柄のみ残す。
    # ユニバースが存在しない月（バックテスト開始直後の1ヶ月など）はフィルタを適用しない。
    if rolling_universe:
        before_filter = len(signal_candidates)
        signal_candidates = [
            c for c in signal_candidates
            if c["ticker"] in rolling_universe.get(c["date"][:7], {c["ticker"]})
        ]
        logger.info(
            "ローリングユニバースフィルタ後: %d → %d 件 (除外 %d 件)",
            before_filter, len(signal_candidates), before_filter - len(signal_candidates),
        )

    # Phase 2: 日付順にソート後、ポートフォリオシミュレーション実行
    signal_candidates.sort(key=lambda c: (c["date"], -c["score"]))

    # 全日付を統合
    all_dates: set[str] = set()
    for ticker in tickers:
        df = price_data.get(ticker)
        if df is not None and not df.empty:
            for dt in df.index:
                all_dates.add(dt.strftime("%Y-%m-%d"))
    sorted_dates = sorted(all_dates)

    all_trades = _run_phase2(
        signal_candidates=signal_candidates,
        sorted_dates=sorted_dates,
        earnings_dates=earnings_dates,
        sl_multiplier=sl_multiplier,
        hard_stop_pct=hard_stop_pct,
        sb_trail_mult=None,  # SB_TRAILING_ATR_MULT をデフォルト使用
    )

    logger.info("シグナルバックテスト完了: 合計 %d トレード", len(all_trades))
    return _aggregate_signal_results(all_trades)


def run_signal_backtest_multi_trail(
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
    fund_data: dict[str, dict[str, Any]],
    spy_df: pd.DataFrame | None = None,
    earnings_dates: dict[str, list[str]] | None = None,
    quarterly_data: dict[str, dict] | None = None,
    sb_trail_mults: list[float] | None = None,
) -> dict[float, "SignalBacktestResult"]:
    """Phase1を1回実行し、sb_trail_mults の各値でPhase2を実行して比較する。

    複数のトレーリング幅を比較する際に Phase1（最も時間がかかる部分）を
    使い回せるため、run_signal_backtest を N 回呼ぶより大幅に高速。

    Args:
        sb_trail_mults: 比較するSTRONG_BUYトレーリング倍率リスト。
                        デフォルト: [4.0, 4.5]

    Returns:
        {trail_mult: SignalBacktestResult} の辞書。
    """
    if sb_trail_mults is None:
        sb_trail_mults = [4.0, 4.5]

    from config.universe import get_sector
    from config.settings import ATR_PARAMS, TRADE_RULES

    # ---- 市場環境フィルター事前計算 ----
    market_ok = _build_market_filter(spy_df)
    if market_ok:
        bearish_days = sum(1 for v in market_ok.values() if not v)
        logger.info("市場環境フィルター: %d日中 %d日がベア判定（BUY禁止）",
                     len(market_ok), bearish_days)
    else:
        logger.info("市場環境フィルター: SPYデータなし → フィルター無効")

    # ---- 【案7】市場過熱フィルター事前計算 ----
    spy_not_hot = _build_spy_hot_filter(spy_df)
    if spy_not_hot:
        hot_days = sum(1 for v in spy_not_hot.values() if not v)
        logger.info("市場過熱フィルター(案7): %d日中 %d日が過熱判定（エントリー回避）",
                     len(spy_not_hot), hot_days)

    # ---- 【案4】セクターモメンタム事前計算 ----
    sector_momentum = _build_sector_momentum(tickers, price_data, fund_data)

    # ---- ローリングユニバース構築（ルックアヘッドバイアス防止） ----
    from config.screening import SCREENING_PARAMS as _sp
    rolling_universe = _build_rolling_universe(
        tickers, price_data,
        top_n=_sp.max_screen_candidates,
        min_history_days=WARMUP_DAYS,
    )
    logger.info("ローリングユニバース: %d ヶ月分を構築完了", len(rolling_universe))

    # ---- SL パラメータ ----
    sl_multiplier = ATR_PARAMS.stop_loss_multiplier
    hard_stop_pct = TRADE_RULES.hard_stop_loss_pct

    # ---- Phase 1: 銘柄ごとにシグナル候補を並列スコアリング（キャッシュ対応） ----
    signal_candidates: list[dict] = []

    tasks = []
    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue
        if len(df) < WARMUP_DAYS + 5:
            continue
        sector = get_sector(ticker)
        fundamentals = fund_data.get(ticker)
        tasks.append((ticker, df, sector, fundamentals))

    # キャッシュチェック
    cache_key = _compute_phase1_cache_key(tickers, bool(quarterly_data))
    cached_candidates = _load_phase1_cache(cache_key)

    if cached_candidates is not None:
        # キャッシュヒット: df を price_data から再アタッチして即利用
        for c in cached_candidates:
            df = price_data.get(c["ticker"])
            if df is not None:
                c["df"] = df
                signal_candidates.append(c)
        logger.info("Phase1スキップ（キャッシュ利用）: %d 件", len(signal_candidates))
    else:
        # キャッシュなし: 通常の並列スコアリングを実行
        cpu_count = mp.cpu_count() or 4
        n_workers = max(1, cpu_count - 1)
        total_tasks = len(tasks)
        logger.info("Phase1 (multi-trail): %d 銘柄を %d ワーカーで並列処理開始 (CPU=%d)",
                    total_tasks, n_workers, cpu_count)

        chunksize = max(1, total_tasks // (n_workers * 2))
        completed = 0

        if quarterly_data:
            logger.info("四半期財務データ: %d 銘柄分 (M=65%%, Q=35%%)", len(quarterly_data))
        else:
            logger.info("四半期財務データなし → Momentum=100%% フォールバック")

        with mp.Pool(
            processes=n_workers,
            initializer=_init_phase1_worker,
            initargs=(spy_df, market_ok, WARMUP_DAYS, earnings_dates or {},
                      quarterly_data or {}, spy_not_hot, sector_momentum),
        ) as pool:
            for ticker_candidates in pool.imap_unordered(
                _score_ticker_phase1, tasks, chunksize=chunksize,
            ):
                completed += 1
                if completed % 50 == 0 or completed == total_tasks:
                    logger.info(
                        "Phase1 進捗: %d / %d 銘柄 (%d%%)",
                        completed, total_tasks, int(completed / total_tasks * 100),
                    )
                for c in ticker_candidates:
                    c["df"] = price_data[c["ticker"]]
                    signal_candidates.append(c)

        # キャッシュ保存（df は除外）
        _save_phase1_cache(cache_key, signal_candidates)

    logger.info("シグナル候補: %d 件を収集 (ローリングユニバースフィルタ前)", len(signal_candidates))

    # ---- ローリングユニバースフィルタ ----
    if rolling_universe:
        before_filter = len(signal_candidates)
        signal_candidates = [
            c for c in signal_candidates
            if c["ticker"] in rolling_universe.get(c["date"][:7], {c["ticker"]})
        ]
        logger.info(
            "ローリングユニバースフィルタ後: %d → %d 件 (除外 %d 件)",
            before_filter, len(signal_candidates), before_filter - len(signal_candidates),
        )

    # Phase2 用の準備
    signal_candidates.sort(key=lambda c: (c["date"], -c["score"]))

    all_dates: set[str] = set()
    for ticker in tickers:
        df = price_data.get(ticker)
        if df is not None and not df.empty:
            for dt in df.index:
                all_dates.add(dt.strftime("%Y-%m-%d"))
    sorted_dates = sorted(all_dates)

    # ---- Phase 2: 各トレーリング幅でシミュレーション（Phase1の結果を使い回す） ----
    results: dict[float, SignalBacktestResult] = {}
    for mult in sb_trail_mults:
        logger.info("Phase2 実行中: sb_trail_mult=%.1f", mult)
        trades = _run_phase2(
            signal_candidates=signal_candidates,
            sorted_dates=sorted_dates,
            earnings_dates=earnings_dates,
            sl_multiplier=sl_multiplier,
            hard_stop_pct=hard_stop_pct,
            sb_trail_mult=mult,
        )
        results[mult] = _aggregate_signal_results(trades)
        logger.info(
            "sb_trail_mult=%.1f 完了: %d トレード, 勝率 %.1f%%, 平均 %+.2f%%",
            mult, len(trades),
            results[mult].win_rate * 100,
            results[mult].avg_pl_pct * 100,
        )

    return results


def _run_phase2(
    signal_candidates: list[dict],
    sorted_dates: list[str],
    earnings_dates: dict[str, list[str]] | None,
    sl_multiplier: float,
    hard_stop_pct: float,
    sb_trail_mult: float | None = None,
) -> list[TradeResult]:
    """Phase2: シグナル候補を日付順にポートフォリオシミュレーション。

    sb_trail_mult を指定することで STRONG_BUY のトレーリング幅を上書きできる。
    Phase1 の結果（signal_candidates）を使い回して複数パラメータを比較可能。
    """
    all_trades: list[TradeResult] = []
    active_positions: dict[str, dict] = {}

    candidates_by_date: dict[str, list[dict]] = {}
    for c in signal_candidates:
        candidates_by_date.setdefault(c["date"], []).append(c)

    last_exit_date: dict[str, str] = {}

    for date_str in sorted_dates:
        # Step 1: 既存ポジションの決済判定
        closed_tickers = []
        for ticker, pos in list(active_positions.items()):
            df = pos["df"]
            if date_str not in [d.strftime("%Y-%m-%d") for d in df.index]:
                continue
            date_idx = None
            for idx_i, d in enumerate(df.index):
                if d.strftime("%Y-%m-%d") == date_str:
                    date_idx = idx_i
                    break
            if date_idx is None:
                continue

            row = df.iloc[date_idx]

            # 決算前強制決済チェック
            trade = None
            if earnings_dates:
                ticker_earnings = earnings_dates.get(ticker, [])
                if ticker_earnings:
                    ei = bisect_left(ticker_earnings, date_str)
                    if ei < len(ticker_earnings):
                        days_to_e = (
                            pd.Timestamp(ticker_earnings[ei]) - pd.Timestamp(date_str)
                        ).days
                        if 0 < days_to_e <= 5:
                            close_price = float(row["Close"])
                            tr = pos["trade_result"]
                            pl_pct = (close_price - tr.entry_price) / tr.entry_price
                            trade = TradeResult(
                                ticker=tr.ticker, signal=tr.signal,
                                signal_date=tr.signal_date, entry_date=tr.entry_date,
                                entry_price=tr.entry_price, exit_date=date_str,
                                exit_price=close_price, result="pre_earnings",
                                pl_pct=pl_pct,
                                sl_price=tr.sl_price, tp_price=0.0, score=tr.score,
                            )

            if trade is None:
                trade, new_hh = _check_exit(
                    pos["trade_result"], date_str, row,
                    atr=pos["atr"], highest_high=pos["highest_high"],
                    sb_trail_mult=sb_trail_mult,
                )
                pos["highest_high"] = new_hh

            if trade is not None:
                all_trades.append(trade)
                last_exit_date[ticker] = date_str
                closed_tickers.append(ticker)

        for t in closed_tickers:
            del active_positions[t]

        # Step 2: 新規エントリー
        day_candidates = candidates_by_date.get(date_str, [])
        for c in day_candidates:
            if len(active_positions) >= MAX_CONCURRENT_POSITIONS:
                break
            ticker = c["ticker"]
            if ticker in active_positions:
                continue
            if last_exit_date.get(ticker) == date_str:
                continue

            entry_price = c["entry_price"]
            atr = c["atr"]
            atr_sl   = entry_price - atr * sl_multiplier
            hard_sl  = entry_price * (1.0 + hard_stop_pct)
            sl_price = max(atr_sl, hard_sl)
            if sl_price <= 0:
                sl_price = entry_price * 0.90

            df = c["df"]
            df_idx = c["df_idx"]
            _hold_limit = SB_MAX_HOLD_DAYS if c["signal"] == "STRONG_BUY" else MAX_HOLD_DAYS
            future_start = df_idx + 2
            future_end = min(df_idx + 2 + _hold_limit, len(df))
            future_df = df.iloc[future_start:future_end]

            trade = _simulate_trade(
                ticker=ticker,
                signal=c["signal"],
                signal_date=date_str,
                entry_date=c["entry_date_str"],
                entry_price=entry_price,
                sl_price=sl_price,
                atr=atr,
                future_df=future_df,
                score=c.get("score", 0.0),
                sb_trail_mult=sb_trail_mult,
            )

            active_positions[ticker] = {
                "trade_result": trade,
                "df": df,
                "atr": atr,
                "highest_high": entry_price,
            }

    # 期間終了時の残ポジションを強制決済
    for ticker, pos in active_positions.items():
        tr = pos["trade_result"]
        df = pos["df"]
        last_row = df.iloc[-1]
        last_date = df.index[-1].strftime("%Y-%m-%d")
        pl_pct = (float(last_row["Close"]) - tr.entry_price) / tr.entry_price
        all_trades.append(TradeResult(
            ticker=tr.ticker, signal=tr.signal,
            signal_date=tr.signal_date, entry_date=tr.entry_date,
            entry_price=tr.entry_price, exit_date=last_date,
            exit_price=float(last_row["Close"]), result="timeout",
            pl_pct=pl_pct, sl_price=tr.sl_price, tp_price=tr.tp_price, score=tr.score,
        ))

    return all_trades


def _get_trail_level(
    highest_high: float,
    entry_price: float,
    atr: float,
    trail_mult_override: float | None = None,
) -> float:
    """段階的トレーリングストップの水準を返す。

    含み益（ATR倍率）に応じてトレール倍率を縮小し、利益をロックする。
    BREAKEVEN_TRIGGER_ATR 以上の含み益では SL ≥ エントリー価格を保証（床）。

    【案6: 直近高値ベーストレーリング追加】
    りおぽん: 「直近高値から5%以上下落したら利確検討」
    ユーザー懸念対応: ATR考慮で可変化 → max(5%, ATR×1.5÷高値) のルームを確保
    ATRトレーリングと直近高値%トレーリングの高い方（より保守的な方）を採用。

    Args:
        highest_high: 保有中の最高値。
        entry_price:  エントリー価格。
        atr:          エントリー時点のATR値。

    Returns:
        トレーリングストップ価格。
    """
    # trail_mult_override が指定されている場合（STRONG_BUY 専用）はそれを基準倍率とする
    base_mult = trail_mult_override if trail_mult_override is not None else TRAILING_STOP_ATR_MULTIPLIER

    if atr <= 0:
        return highest_high - atr * base_mult

    peak_profit_atr = (highest_high - entry_price) / atr

    # 段階的倍率: 含み益大きいほどタイトに
    # ただし trail_mult_override が指定された場合（STRONG_BUY）は PROGRESSIVE_TRAIL_LEVELS を
    # バイパスして固定倍率を維持する。これにより trail×4.0 vs trail×4.5 の比較が有効になる。
    # BUY (override=None): 通常の段階的トレーリング（3.5→3.0→2.5→2.0）
    # STRONG_BUY (override=4.0/4.5): 固定幅トレーリング（利益確保よりも大きな上昇を待つ）
    trail_mult = base_mult
    if trail_mult_override is None:
        for min_atr, mult in PROGRESSIVE_TRAIL_LEVELS:
            if peak_profit_atr >= min_atr:
                trail_mult = mult
                break

    trail_atr = highest_high - atr * trail_mult

    # 【案6】直近高値ベーストレーリング（ボラ考慮の可変版）
    # ルーム = max(5%, ATR×1.5÷高値) → 高ボラ株は広め、安定株は5%
    if highest_high > 0:
        pct_room = max(PEAK_TRAIL_MIN_PCT, PEAK_TRAIL_ATR_MULT * atr / highest_high)
        trail_peak = highest_high * (1.0 - pct_room)
    else:
        trail_peak = trail_atr

    # 2つのトレーリングの高い方（より早く発動する方）を採用
    trail = max(trail_atr, trail_peak)

    # ブレイクイーブン床: 含み益 ≥ BREAKEVEN_TRIGGER_ATR → SL はエントリー価格 + ATR×ラチェット以上
    if peak_profit_atr >= BREAKEVEN_TRIGGER_ATR:
        trail = max(trail, entry_price + atr * BREAKEVEN_LOCK_ATR)

    return trail


def _simulate_trade(
    ticker: str,
    signal: str,
    signal_date: str,
    entry_date: str,
    entry_price: float,
    sl_price: float,
    atr: float,
    future_df: pd.DataFrame,
    score: float = 0.0,
    sb_trail_mult: float | None = None,
) -> TradeResult:
    """SL + トレーリングストップ + 段階的タイムアウトでトレードを模擬する。

    - SL: エントリー時の固定SL（ATR×2.0、-8%フロア）
    - トレーリングストップ: 最高値 - ATR×3.5 を下回ったら利確（エントリー価格以上のみ発動）
    - 段階的タイムアウト:
      - 30営業日: 含み損（< 0%）→ timeout_30d
      - 90営業日: 無条件 → timeout_90d

    Args:
        atr:        エントリー時点のATR値（トレーリングストップ計算用）。
        future_df:  エントリー日翌日以降の価格DataFrame（最大MAX_HOLD_DAYS行）。

    Returns:
        TradeResult（ポジションとして仮保存した状態）。
    """
    highest_high = entry_price  # エントリー価格を初期値

    # STRONG_BUY 専用パラメータの適用
    # sb_trail_mult が引数で渡された場合はそれを優先（複数パターン比較用）
    is_strong_buy = (signal == "STRONG_BUY")
    _sb_tm        = sb_trail_mult if sb_trail_mult is not None else SB_TRAILING_ATR_MULT
    _trail_mult   = _sb_tm if is_strong_buy else TRAILING_STOP_ATR_MULTIPLIER
    _max_hold     = SB_MAX_HOLD_DAYS if is_strong_buy else MAX_HOLD_DAYS
    _use_timeout  = SB_STAGED_TIMEOUT_ENABLED if is_strong_buy else True

    for idx in range(len(future_df)):
        row = future_df.iloc[idx]
        date_str = future_df.index[idx].strftime("%Y-%m-%d")
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        holding_days = idx + 1  # 1-indexed（エントリー翌日が1日目）

        # 最高値を更新
        if high > highest_high:
            highest_high = high

        # SL判定（ギャップダウン対応: 始値がSL以下なら始値で約定）
        open_price_row = float(row["Open"]) if "Open" in row.index else float(row["Low"])
        if low <= sl_price or open_price_row <= sl_price:
            exit_price_sl = min(sl_price, open_price_row)
            return TradeResult(
                ticker=ticker, signal=signal, signal_date=signal_date,
                entry_date=entry_date, entry_price=entry_price,
                exit_date=date_str, exit_price=exit_price_sl,
                result="sl_hit",
                pl_pct=(exit_price_sl - entry_price) / entry_price,
                sl_price=sl_price, tp_price=0.0, score=score,
            )

        # トレーリングストップ判定（STRONG_BUY は広めの倍率を使用）
        # 【案1】エントリーから5営業日間はトレーリングストップを停止（SLは通常通り有効）
        trail_level = _get_trail_level(highest_high, entry_price, atr,
                                       trail_mult_override=_trail_mult)
        if holding_days > 5 and trail_level >= entry_price and close <= trail_level:
            pl_pct = (trail_level - entry_price) / entry_price
            return TradeResult(
                ticker=ticker, signal=signal, signal_date=signal_date,
                entry_date=entry_date, entry_price=entry_price,
                exit_date=date_str, exit_price=trail_level,
                result="trailing_stop",
                pl_pct=pl_pct,
                sl_price=sl_price, tp_price=0.0, score=score,
            )

        # 段階的タイムアウト判定（STRONG_BUY は無効化）
        if _use_timeout:
            pl_pct = (close - entry_price) / entry_price
            for threshold_days, min_profit in STAGED_TIMEOUT:
                if holding_days == threshold_days and pl_pct < min_profit:
                    return TradeResult(
                        ticker=ticker, signal=signal, signal_date=signal_date,
                        entry_date=entry_date, entry_price=entry_price,
                        exit_date=date_str, exit_price=close,
                        result=f"timeout_{threshold_days}d",
                        pl_pct=pl_pct,
                        sl_price=sl_price, tp_price=0.0, score=score,
                    )

        # 絶対タイムアウト（STRONG_BUY は 90日、BUY は 60日）
        if holding_days >= _max_hold:
            return TradeResult(
                ticker=ticker, signal=signal, signal_date=signal_date,
                entry_date=entry_date, entry_price=entry_price,
                exit_date=date_str, exit_price=close,
                result=f"timeout_{_max_hold}d",
                pl_pct=(close - entry_price) / entry_price,
                sl_price=sl_price, tp_price=0.0, score=score,
            )

    # future_df が尽きた場合（期間末）
    if len(future_df) > 0:
        last_row = future_df.iloc[-1]
        last_date = future_df.index[-1].strftime("%Y-%m-%d")
        last_close = float(last_row["Close"])
    else:
        last_date = entry_date
        last_close = entry_price

    return TradeResult(
        ticker=ticker, signal=signal, signal_date=signal_date,
        entry_date=entry_date, entry_price=entry_price,
        exit_date=last_date, exit_price=last_close,
        result=f"timeout_{_max_hold}d",
        pl_pct=(last_close - entry_price) / entry_price,
        sl_price=sl_price, tp_price=0.0, score=score,
    )


def _check_exit(
    position: TradeResult,
    current_date_str: str,
    row: pd.Series,
    atr: float,
    highest_high: float,
    sb_trail_mult: float | None = None,
) -> tuple[TradeResult | None, float]:
    """保有中ポジションの決済判定（SL + トレーリングストップ + 段階的タイムアウト）。

    Args:
        atr:          エントリー時点のATR値。
        highest_high: これまでの最高値（呼び出し側で管理・更新する）。

    Returns:
        (決済TradeResult or None, 更新後のhighest_high) のタプル。
    """
    open_price = float(row["Open"]) if "Open" in row.index else float(row["Low"])
    high = float(row["High"])
    low = float(row["Low"])
    close = float(row["Close"])

    # 最高値を更新
    if high > highest_high:
        highest_high = high

    # STRONG_BUY 専用パラメータ（引数で渡された場合はそれを優先）
    is_strong_buy = (position.signal == "STRONG_BUY")
    _sb_tm        = sb_trail_mult if sb_trail_mult is not None else SB_TRAILING_ATR_MULT
    _trail_mult   = _sb_tm if is_strong_buy else None
    _max_hold     = SB_MAX_HOLD_DAYS if is_strong_buy else MAX_HOLD_DAYS
    _use_timeout  = SB_STAGED_TIMEOUT_ENABLED if is_strong_buy else True

    # SL判定（ギャップダウン対応: 始値がSL以下なら始値で約定）
    if low <= position.sl_price or open_price <= position.sl_price:
        exit_price = min(position.sl_price, open_price)
        pl_pct = (exit_price - position.entry_price) / position.entry_price
        return TradeResult(
            ticker=position.ticker, signal=position.signal,
            signal_date=position.signal_date, entry_date=position.entry_date,
            entry_price=position.entry_price, exit_date=current_date_str,
            exit_price=exit_price, result="sl_hit", pl_pct=pl_pct,
            sl_price=position.sl_price, tp_price=0.0, score=position.score,
        ), highest_high

    # トレーリングストップ判定（STRONG_BUY は広めの倍率）
    trail_level = _get_trail_level(highest_high, position.entry_price, atr,
                                   trail_mult_override=_trail_mult)
    if trail_level >= position.entry_price and close <= trail_level:
        pl_pct = (trail_level - position.entry_price) / position.entry_price
        return TradeResult(
            ticker=position.ticker, signal=position.signal,
            signal_date=position.signal_date, entry_date=position.entry_date,
            entry_price=position.entry_price, exit_date=current_date_str,
            exit_price=trail_level, result="trailing_stop", pl_pct=pl_pct,
            sl_price=position.sl_price, tp_price=0.0, score=position.score,
        ), highest_high

    # 段階的タイムアウト判定（STRONG_BUY は無効）
    try:
        entry_dt = pd.Timestamp(position.entry_date)
        current_dt = pd.Timestamp(current_date_str)
        calendar_days = (current_dt - entry_dt).days
        # カレンダー日数 → 営業日近似 (× 5/7)
        approx_biz_days = int(calendar_days * 5 / 7)

        pl_pct = (close - position.entry_price) / position.entry_price

        if _use_timeout:
            for idx_t, (threshold_days, min_profit) in enumerate(STAGED_TIMEOUT):
                is_last_threshold = (idx_t == len(STAGED_TIMEOUT) - 1)
                if is_last_threshold:
                    if approx_biz_days >= threshold_days and pl_pct < min_profit:
                        return TradeResult(
                            ticker=position.ticker, signal=position.signal,
                            signal_date=position.signal_date, entry_date=position.entry_date,
                            entry_price=position.entry_price, exit_date=current_date_str,
                            exit_price=close, result=f"timeout_{threshold_days}d",
                            pl_pct=pl_pct,
                            sl_price=position.sl_price, tp_price=0.0, score=position.score,
                        ), highest_high
                else:
                    if (threshold_days - 2) <= approx_biz_days <= (threshold_days + 2) and pl_pct < min_profit:
                        return TradeResult(
                            ticker=position.ticker, signal=position.signal,
                            signal_date=position.signal_date, entry_date=position.entry_date,
                            entry_price=position.entry_price, exit_date=current_date_str,
                            exit_price=close, result=f"timeout_{threshold_days}d",
                            pl_pct=pl_pct,
                            sl_price=position.sl_price, tp_price=0.0, score=position.score,
                        ), highest_high

        # 絶対タイムアウト（STRONG_BUY: 90日, BUY: 60日）
        if approx_biz_days >= _max_hold:
            return TradeResult(
                ticker=position.ticker, signal=position.signal,
                signal_date=position.signal_date, entry_date=position.entry_date,
                entry_price=position.entry_price, exit_date=current_date_str,
                exit_price=close, result=f"timeout_{_max_hold}d", pl_pct=pl_pct,
                sl_price=position.sl_price, tp_price=0.0, score=position.score,
            ), highest_high
    except Exception:
        pass

    return None, highest_high


def _aggregate_signal_results(trades: list[TradeResult]) -> SignalBacktestResult:
    """トレード結果リストを集計してSignalBacktestResultを返す。"""
    if not trades:
        return SignalBacktestResult()

    result = SignalBacktestResult(trades=trades)
    result.total_trades = len(trades)

    pl_list = [t.pl_pct for t in trades]
    wins = [t for t in trades if t.pl_pct > 0]

    result.win_rate = len(wins) / len(trades) if trades else 0.0
    result.avg_pl_pct = float(np.mean(pl_list)) if pl_list else 0.0
    result.total_pl_pct = float(np.sum(pl_list))

    # 最大ドローダウン（累積損益ベース）
    cumulative = np.cumsum(pl_list)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    result.max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Sharpe比（日次リターンから年率換算）
    if len(pl_list) > 1:
        std = float(np.std(pl_list, ddof=1))
        if std > 0:
            # 取引頻度を考慮した年率換算
            # 平均保有期間を実績から推定
            avg_hold = 45  # 段階的タイムアウトの中央値を近似
            result.sharpe_ratio = (result.avg_pl_pct / std) * math.sqrt(252 / avg_hold)
        else:
            result.sharpe_ratio = 0.0

    # result別件数
    result_types = set(t.result for t in trades)
    result.by_result = {r: sum(1 for t in trades if t.result == r) for r in sorted(result_types)}

    # シグナル別集計
    for sig in ("STRONG_BUY", "BUY"):
        sig_trades = [t for t in trades if t.signal == sig]
        if not sig_trades:
            continue
        sig_pl = [t.pl_pct for t in sig_trades]
        result.by_signal[sig] = {
            "count": len(sig_trades),
            "win_rate": sum(1 for t in sig_trades if t.pl_pct > 0) / len(sig_trades),
            "avg_pl_pct": float(np.mean(sig_pl)),
        }

    # 年別集計
    for trade in trades:
        year = trade.entry_date[:4]
        if year not in result.by_year:
            result.by_year[year] = {"count": 0, "wins": 0, "total_pl": 0.0}
        result.by_year[year]["count"] += 1
        if trade.pl_pct > 0:
            result.by_year[year]["wins"] += 1
        result.by_year[year]["total_pl"] += trade.pl_pct

    for year_data in result.by_year.values():
        cnt = year_data["count"]
        year_data["win_rate"] = year_data["wins"] / cnt if cnt > 0 else 0.0
        year_data["avg_pl_pct"] = year_data["total_pl"] / cnt if cnt > 0 else 0.0

    return result


# ============================================================
# スクリーニングバックテスト
# ============================================================

def run_screening_backtest(
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
) -> ScreeningBacktestResult:
    """スクリーニングバックテストを実行する。

    各月末に momentum スコア上位銘柄を選択し、翌月の実績リターンと比較する。

    Args:
        tickers:    対象ティッカーリスト。
        price_data: ティッカー → OHLCV DataFrame。

    Returns:
        ScreeningBacktestResult。
    """
    from analysis.momentum import calc_momentum_score
    from config.screening import SCREENING_PARAMS

    # 全銘柄の月次リターンを算出（各月の最終営業日の終値ベース）
    monthly_close: dict[str, pd.Series] = {}
    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue
        # 月末終値（各月の最終営業日）
        monthly = df["Close"].resample("ME").last()
        monthly_close[ticker] = monthly

    if not monthly_close:
        return ScreeningBacktestResult()

    # 月次リターン計算
    monthly_returns: dict[str, pd.Series] = {}
    for ticker, series in monthly_close.items():
        monthly_returns[ticker] = series.pct_change()

    # 共通の月インデックスを取得
    all_months = sorted(set().union(*[set(s.index) for s in monthly_close.values()]))

    records: list[MonthlyRecord] = []

    # ウォームアップ後の月から開始（252日 ≈ 12ヶ月）
    start_idx = 12
    max_candidates = SCREENING_PARAMS.max_screen_candidates

    for m_idx in range(start_idx, len(all_months) - 1):
        month_end = all_months[m_idx]
        next_month = all_months[m_idx + 1]
        month_str = month_end.strftime("%Y-%m")

        # 各銘柄のmomentumスコアを月末時点のDataFrameで計算
        scored: list[tuple[str, float]] = []
        for ticker in tickers:
            df = price_data.get(ticker)
            if df is None or df.empty:
                continue
            # 月末日以前のデータのみ使用（ルックアヘッド防止）
            window = df.loc[:month_end]
            if len(window) < 30:
                continue
            try:
                score = calc_momentum_score(window)
                scored.append((ticker, score))
            except Exception:
                continue

        if not scored:
            continue

        # スコア上位を選択
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in scored[:max_candidates]]

        # 翌月のリターンを集計
        def get_next_month_return(ticker: str) -> float | None:
            ret_series = monthly_returns.get(ticker)
            if ret_series is None:
                return None
            if next_month not in ret_series.index:
                return None
            val = ret_series.loc[next_month]
            if pd.isna(val):
                return None
            return float(val)

        selected_rets = [r for t in selected if (r := get_next_month_return(t)) is not None]
        all_rets = [r for t in tickers if (r := get_next_month_return(t)) is not None]

        if not selected_rets or not all_rets:
            continue

        sel_avg = float(np.mean(selected_rets))
        all_avg = float(np.mean(all_rets))
        outperf = sel_avg - all_avg

        records.append(MonthlyRecord(
            month=month_str,
            selected_tickers=selected,
            selected_return=sel_avg,
            all_return=all_avg,
            outperformance=outperf,
        ))

    logger.info("スクリーニングバックテスト完了: %d ヶ月", len(records))
    return _aggregate_screening_results(records)


def _aggregate_screening_results(records: list[MonthlyRecord]) -> ScreeningBacktestResult:
    """月次記録を集計してScreeningBacktestResultを返す。"""
    if not records:
        return ScreeningBacktestResult()

    result = ScreeningBacktestResult(monthly_records=records)
    result.total_months = len(records)
    result.avg_selected_return = float(np.mean([r.selected_return for r in records]))
    result.avg_all_return = float(np.mean([r.all_return for r in records]))
    result.avg_outperformance = result.avg_selected_return - result.avg_all_return
    result.win_months = sum(1 for r in records if r.outperformance > 0)

    return result
