"""
バックテスト実行スクリプト — 2018年1月〜2021年12月

yfinance に直接 start/end を指定して2017-2021年のデータを取得し、
既存のバックテストエンジンを使って 2018-2021 年間のシグナル損益を検証する。

ウォームアップ (200営業日) のため 2017-03-01 からデータ取得する。
2021年末に入ったポジションが90営業日持てるよう 2022-04-30 まで取得する。
"""

import os
import sys
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import logging
import pandas as pd
import yfinance as yf

# プロジェクトルートを path に追加
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backtest.engine import run_signal_backtest, run_screening_backtest
from backtest.report import (
    print_signal_backtest,
    print_screening_backtest,
    save_signal_backtest_csv,
    save_screening_backtest_csv,
)
from config.universe import get_all_tickers
from data.cache import CacheDB
from data.fundamental_fetcher import fetch_fundamentals

# ============================================================
# 設定
# ============================================================
FETCH_START = "2017-03-01"   # 200営業日 ウォームアップのため 2018-01 より前
FETCH_END   = "2022-04-30"   # 2021年末エントリー分の最大90営業日を確保
BATCH_SIZE  = 20
BATCH_SLEEP = 2.5            # レート制限対策 (秒)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# ============================================================
# メイン
# ============================================================

def main() -> None:
    print()
    print("=" * 58)
    print("  バックテスト 2018-01〜2021-12")
    print(f"  データ取得: {FETCH_START} 〜 {FETCH_END}")
    print("=" * 58)
    print()

    # ---- DB 初期化 ----
    db = CacheDB()
    db.initialize()

    # ---- 銘柄リスト ----
    tickers = get_all_tickers()
    all_tickers = sorted(set(tickers) | {"SPY"})
    print(f"対象銘柄: {len(tickers)} 銘柄 + SPY")
    print()

    # ---- 株価取得 (start/end 指定) ----
    print(f"⏳ [1/3] 株価データ取得中（{FETCH_START} 〜 {FETCH_END}）...")
    price_data = _fetch_prices_range(all_tickers)
    print(f"  → {len(price_data)} 銘柄を取得")

    spy_df = price_data.pop("SPY", None)
    if spy_df is not None:
        print(f"  → SPY: {len(spy_df)} 日分 (市場フィルター用)")
    else:
        print("  ⚠ SPY データ取得失敗 — 市場フィルター無効")

    # ---- ファンダメンタル取得 ----
    print("⏳ [2/3] ファンダメンタルデータ取得中...")
    fund_data = fetch_fundamentals(tickers, db=db)
    print(f"  → {len(fund_data)} 銘柄のファンダメンタルを取得")

    # ---- バックテスト実行 ----
    print("⏳ [3/3] バックテスト実行中...")
    start_bt = time.time()
    sig_result = run_signal_backtest(tickers, price_data, fund_data, spy_df=spy_df)
    scr_result = run_screening_backtest(tickers, price_data)
    elapsed = time.time() - start_bt
    print(f"  → 完了（{elapsed:.1f}秒）")

    # ---- 結果表示 ----
    print_signal_backtest(sig_result)
    print_screening_backtest(scr_result)

    # ---- CSV 保存 ----
    sig_path = save_signal_backtest_csv(sig_result)
    if sig_path:
        print(f"\n📁 シグナルバックテスト CSV: {sig_path}")
    scr_path = save_screening_backtest_csv(scr_result)
    if scr_path:
        print(f"📁 スクリーニングバックテスト CSV: {scr_path}")

    print("\n✅ 完了")


# ============================================================
# 株価取得（start/end 指定、バッチ逐次処理）
# ============================================================

def _fetch_prices_range(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """yfinance に直接 start/end を渡して全銘柄を取得する。"""
    results: dict[str, pd.DataFrame] = {}
    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

    for idx, batch in enumerate(batches):
        if idx > 0:
            time.sleep(BATCH_SLEEP)

        batch_result = _download_batch(batch)
        results.update(batch_result)
        print(
            f"  バッチ {idx+1}/{len(batches)}: "
            f"{len(batch_result)}/{len(batch)} 銘柄取得",
            flush=True,
        )

    return results


def _download_batch(tickers: list[str], retries: int = 3) -> dict[str, pd.DataFrame]:
    """バッチダウンロード（リトライ付き）。"""
    ticker_str = " ".join(tickers)
    delay = 10.0

    for attempt in range(retries):
        try:
            raw = yf.download(
                ticker_str,
                start=FETCH_START,
                end=FETCH_END,
                group_by="ticker",
                auto_adjust=True,
                threads=False,
                progress=False,
            )
            if raw.empty:
                return {}
            return _parse_batch(raw, tickers)

        except Exception as exc:
            is_rate_limit = "RateLimit" in type(exc).__name__ or "rate" in str(exc).lower()
            if is_rate_limit and attempt < retries - 1:
                time.sleep(delay)
                delay *= 3
                continue
            # フォールバック: 個別ダウンロード
            return _download_individual(tickers)

    return {}


def _download_individual(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """個別取得フォールバック。"""
    results: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            raw = yf.download(
                ticker,
                start=FETCH_START,
                end=FETCH_END,
                auto_adjust=True,
                progress=False,
            )
            df = _clean(raw)
            if df is not None:
                results[ticker] = df
            time.sleep(0.5)
        except Exception:
            pass
    return results


def _parse_batch(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """バッチダウンロード結果を ticker → DataFrame に変換する。"""
    results: dict[str, pd.DataFrame] = {}

    if len(tickers) == 1:
        df = _clean(raw)
        if df is not None:
            results[tickers[0]] = df
        return results

    # MultiIndex 構造 (ticker, OHLCV)
    level0 = raw.columns.get_level_values(0) if isinstance(raw.columns, pd.MultiIndex) else []
    for ticker in tickers:
        try:
            if ticker not in level0:
                continue
            df = _clean(raw[ticker].copy())
            if df is not None:
                results[ticker] = df
        except Exception:
            pass

    return results


def _clean(df: pd.DataFrame) -> pd.DataFrame | None:
    """DataFrame をクリーンアップする（カラム名統一・NaN除去）。"""
    if df is None or df.empty:
        return None

    # MultiIndex を平坦化
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    rename = {}
    for col in df.columns:
        cl = str(col).lower()
        if cl == "open":   rename[col] = "Open"
        elif cl == "high":  rename[col] = "High"
        elif cl == "low":   rename[col] = "Low"
        elif cl == "close": rename[col] = "Close"
        elif cl == "volume":rename[col] = "Volume"
    if rename:
        df = df.rename(columns=rename)

    need = {"Open", "High", "Low", "Close", "Volume"}
    if not need.issubset(df.columns):
        return None

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df if not df.empty else None


if __name__ == "__main__":
    main()
