"""
Exit Parameter Sweep — 全組み合わせ比較（修正版）

元のベースラインCSVのエントリー信号をそのまま使い、
出口パラメータ（trail / BEロック / 保有日数）だけを48通りで再シミュレーション。
Phase1再実行不要 → 所要時間 約30〜40分。

結果: output/backtest/exit_sweep_{timestamp}.csv
"""
from __future__ import annotations

import csv
import itertools
import logging
import math
import multiprocessing
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent

# ============================================================
# 元のベースラインCSV（エントリー信号のソース）
# ============================================================
BASELINE_CSV = BASE / "output/backtest/signal_backtest_20260305_223046_sb_trail4.0.csv"

# ============================================================
# パラメータグリッド（48通り）
# ============================================================
TRAIL_VARIANTS: dict[str, list[tuple[float, float]]] = {
    "T0_base": [(6.0, 2.0), (4.0, 2.5), (2.0, 3.0)],  # ベースライン
    "T1_mild": [(6.0, 2.5), (4.0, 3.0), (2.0, 3.5)],  # 各段階 +0.5（ゆるめ）
    "T2_high": [(8.0, 2.5), (5.0, 3.0)],               # 閾値引き上げ
    "T3_none": [],                                       # 段階縮小なし（常にベース倍率）
}
BE_TRIGGER_VARIANTS: list[float] = [1.0, 1.5, 2.0]
BE_LOCK_VARIANTS:    list[float] = [0.5, 1.0]
HOLD_VARIANTS: list[tuple[int, int]] = [(60, 90), (90, 120)]  # (BUY, SB)

# ============================================================
# ETFオーバーレイ定数
# ============================================================
INITIAL      = 4000.0
RISK         = 0.005
ALLOC        = 0.13
SB_ALLOC     = 0.14
MAX_POS      = 11
MIN_RR       = 0.005

ETF_UNIVERSE = [
    "SPY", "QQQ", "XLC", "XLY", "XLP",
    "XLE", "XLF", "XLV", "XLI", "XLB",
    "XLK", "XLRE", "XLU",
]
ETF_TOP_N    = 2
ETF_MA       = 50
SPY_TREND_MA = 200
SAFE_TICKER  = "SGOV"
MOMENTUM_PERIODS = {
    "1M":  (21,  0.30),
    "3M":  (63,  0.30),
    "6M":  (126, 0.30),
    "12M": (252, 0.10),
}
SIM_START  = "2021-01-04"
SIM_END    = "2026-02-28"
DATA_START = "2019-10-01"


# ============================================================
# ETF補助関数
# ============================================================
def etf_price(etf_close, ticker, date):
    s = etf_close.get(ticker)
    if s is None: return None
    past = s[s.index <= date]
    return float(past.iloc[-1]) if len(past) > 0 else None

def etf_ma(etf_close, ticker, date, period=ETF_MA):
    s = etf_close.get(ticker)
    if s is None: return None
    past = s[s.index <= date]
    return float(past.iloc[-period:].mean()) if len(past) >= period else None

def spy_regime(etf_close, date):
    s = etf_close.get("SPY")
    if s is None: return True
    past = s[s.index <= date]
    if len(past) < SPY_TREND_MA: return True
    return float(past.iloc[-1]) > float(past.iloc[-SPY_TREND_MA:].mean())

def calc_momentum_scores(etf_close, date):
    rets = {}
    for ticker in ETF_UNIVERSE:
        s = etf_close.get(ticker)
        if s is None: continue
        past = s[s.index <= date]
        if len(past) == 0: continue
        current = float(past.iloc[-1])
        tr = {}; ok = True
        for name, (days, _) in MOMENTUM_PERIODS.items():
            if len(past) < days + 1: ok = False; break
            base = float(past.iloc[-(days + 1)])
            if base <= 0: ok = False; break
            tr[name] = (current - base) / base
        if ok: rets[ticker] = tr
    if len(rets) < 2: return []
    scores = {t: 0.0 for t in rets}
    for name, (_, weight) in MOMENTUM_PERIODS.items():
        ranked = sorted(rets.keys(), key=lambda t: rets[t][name], reverse=True)
        for rank, ticker in enumerate(ranked, 1):
            scores[ticker] += rank * weight
    return sorted(scores.items(), key=lambda x: x[1])

def select_target_etfs(etf_close, date, scored):
    result = []
    for ticker, _ in scored:
        if len(result) >= ETF_TOP_N: break
        p = etf_price(etf_close, ticker, date)
        ma = etf_ma(etf_close, ticker, date)
        if p and ma and p >= ma: result.append(ticker)
    return result

def get_period_key(date):
    return (date.year, date.isocalendar()[1])


# ============================================================
# ETFオーバーレイ シミュレーション
# ============================================================
def run_etf_simulation(trades_raw, etf_close):
    trades_raw = sorted(trades_raw, key=lambda t: t["entry_date"])
    same_day = {i for i, t in enumerate(trades_raw) if t["entry_date"] == t["exit_date"]}

    entries_by_date: dict[str, list[int]] = {}
    exits_by_date:   dict[str, list[int]] = {}
    for i, t in enumerate(trades_raw):
        entries_by_date.setdefault(t["entry_date"], []).append(i)
        exits_by_date.setdefault(t["exit_date"],   []).append(i)

    bdays = pd.bdate_range(SIM_START, SIM_END)
    cash = INITIAL; stock_pos = {}; etf_pos = {}
    peak = INITIAL; max_dd = 0.0
    daily_balances = []; trade_hist = []
    pe: set[int] = set(); px: set[int] = set()
    prev_period = None; current_scores = []; prev_bull = None

    for date in bdays:
        date_str = date.strftime("%Y-%m-%d")
        bull     = spy_regime(etf_close, date)
        reg_chg  = (prev_bull is not None and bull != prev_bull)

        # ① リバランス（週次）
        period_key = get_period_key(date)
        if period_key != prev_period:
            prev_period = period_key
            if bull:
                current_scores = calc_momentum_scores(etf_close, date)
                targets = select_target_etfs(etf_close, date, current_scores)
                for tk in list(etf_pos.keys()):
                    if tk not in targets:
                        p = etf_price(etf_close, tk, date)
                        if p: cash += etf_pos[tk] * p
                        del etf_pos[tk]
                if targets and cash > 1.0:
                    alloc_each = cash / len(targets)
                    for tk in targets:
                        p = etf_price(etf_close, tk, date)
                        if p and p > 0:
                            etf_pos[tk] = etf_pos.get(tk, 0.0) + alloc_each / p
                            cash -= alloc_each
            else:
                for tk in list(etf_pos.keys()):
                    if tk != SAFE_TICKER:
                        p = etf_price(etf_close, tk, date)
                        if p: cash += etf_pos[tk] * p
                        del etf_pos[tk]
                if cash > 1.0:
                    sgov_p = etf_price(etf_close, SAFE_TICKER, date)
                    if sgov_p:
                        etf_pos[SAFE_TICKER] = etf_pos.get(SAFE_TICKER, 0.0) + cash / sgov_p
                        cash = 0.0

        # ② レジーム変化 + 50MA割れ
        if reg_chg and not bull:
            for tk in list(etf_pos.keys()):
                if tk != SAFE_TICKER:
                    p = etf_price(etf_close, tk, date)
                    if p: cash += etf_pos[tk] * p
                    del etf_pos[tk]
            if cash > 1.0:
                sgov_p = etf_price(etf_close, SAFE_TICKER, date)
                if sgov_p:
                    etf_pos[SAFE_TICKER] = etf_pos.get(SAFE_TICKER, 0.0) + cash / sgov_p
                    cash = 0.0
        elif reg_chg and bull:
            for tk in [SAFE_TICKER]:
                if tk in etf_pos:
                    p = etf_price(etf_close, tk, date)
                    if p: cash += etf_pos[tk] * p
                    del etf_pos[tk]
            for tk in list(etf_pos.keys()):
                p = etf_price(etf_close, tk, date); ma = etf_ma(etf_close, tk, date)
                if p and ma and p < ma:
                    cash += etf_pos[tk] * p; del etf_pos[tk]
        elif bull:
            for tk in list(etf_pos.keys()):
                if tk == SAFE_TICKER: continue
                p = etf_price(etf_close, tk, date); ma = etf_ma(etf_close, tk, date)
                if p and ma and p < ma:
                    cash += etf_pos[tk] * p; del etf_pos[tk]

        prev_bull = bull

        # ③ 個別株EXIT
        for idx in exits_by_date.get(date_str, []):
            if idx in same_day: continue
            if idx in pe and idx not in px:
                px.add(idx)
                t = trades_raw[idx]; amt = stock_pos.pop(idx)
                pnl = amt * (t["pl_pct"] / 100.0); cash += amt + pnl
                etf_val = sum(etf_pos[tk] * (etf_price(etf_close, tk, date) or 0) for tk in etf_pos)
                bal = cash + sum(stock_pos.values()) + etf_val
                if bal > peak: peak = bal
                dd = (peak - bal) / peak * 100 if peak > 0 else 0
                if dd > max_dd: max_dd = dd
                trade_hist.append({"pnl": pnl})

        # ④ 個別株ENTRY
        etf_val  = sum(etf_pos[tk] * (etf_price(etf_close, tk, date) or 0) for tk in etf_pos)
        total_eq = cash + sum(stock_pos.values()) + etf_val

        for idx in entries_by_date.get(date_str, []):
            if idx in pe: continue
            if len(stock_pos) >= MAX_POS: continue
            t = trades_raw[idx]
            rr = max((t["entry_price"] - t["sl_price"]) / t["entry_price"]
                     if t["entry_price"] > 0 else 0.08, MIN_RR)
            inv  = (total_eq * RISK) / rr
            ap   = SB_ALLOC if t["signal"] == "STRONG_BUY" else ALLOC
            need = min(inv, total_eq * ap)

            if cash < need and etf_pos:
                shortage  = need - cash
                score_map = {tk: s for tk, s in current_scores}
                ranked = sorted(etf_pos.keys(),
                                key=lambda tk: (0 if tk == SAFE_TICKER else 1,
                                                score_map.get(tk, 999)),
                                reverse=True)
                for tk in ranked:
                    if shortage <= 0: break
                    p = etf_price(etf_close, tk, date)
                    if not p: continue
                    sell = min(shortage, etf_pos[tk] * p)
                    etf_pos[tk] -= sell / p
                    if etf_pos[tk] < 1e-9: del etf_pos[tk]
                    cash += sell; shortage -= sell

            alloc = min(need, cash)
            if t["entry_price"] > 0:
                sh = math.floor(alloc / t["entry_price"])
                alloc = sh * t["entry_price"]
            else:
                sh = 0
            if alloc <= 0 or sh == 0 or cash <= 0: continue
            cash -= alloc; stock_pos[idx] = alloc; pe.add(idx)

            if idx in same_day:
                px.add(idx)
                cash += alloc + alloc * (t["pl_pct"] / 100.0)
                stock_pos.pop(idx, None)

        # ⑤ 日次スナップショット
        etf_val  = sum(etf_pos[tk] * (etf_price(etf_close, tk, date) or 0) for tk in etf_pos)
        total_eq = cash + sum(stock_pos.values()) + etf_val
        if total_eq > peak: peak = total_eq
        dd = (peak - total_eq) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd
        daily_balances.append(total_eq)

    final = daily_balances[-1] if daily_balances else INITIAL
    ret   = (final - INITIAL) / INITIAL * 100
    bal_s = pd.Series(daily_balances)
    dr    = bal_s.pct_change().dropna()
    sharpe  = (dr.mean() / dr.std()) * np.sqrt(252) if dr.std() > 0 else 0.0
    calmar  = ret / max_dd if max_dd > 0 else 0.0
    wins    = sum(1 for h in trade_hist if h["pnl"] > 0)
    gp      = sum(h["pnl"] for h in trade_hist if h["pnl"] > 0)
    gl      = abs(sum(h["pnl"] for h in trade_hist if h["pnl"] < 0))
    pf      = gp / gl if gl > 0 else 9.99
    wr      = wins / len(trade_hist) * 100 if trade_hist else 0.0
    return final, ret, max_dd, sharpe, calmar, pf, wr, len(trade_hist)


# ============================================================
# TradeResult → trades_raw 変換
# ============================================================
def trades_to_dict_list(trades):
    return [{
        "ticker":      t.ticker,
        "signal":      t.signal,
        "entry_date":  t.entry_date,
        "exit_date":   t.exit_date,
        "entry_price": t.entry_price,
        "exit_price":  t.exit_price,
        "pl_pct":      t.pl_pct,
        "sl_price":    t.sl_price,
    } for t in trades]


# ============================================================
# ベースラインCSVからsignal_candidatesを再構築
# ============================================================
def build_candidates_from_csv(csv_path: Path, price_data: dict) -> list[dict]:
    """
    ベースラインCSVのエントリー信号を読み込み、
    Phase2が必要とするsignal_candidate形式に変換する。

    Phase2はこれらのエントリー日・価格を使い、
    出口（trail/BE/timeout）だけをパラメータに従って再シミュレーションする。
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    print(f"  CSV読み込み: {len(rows)} 件")

    # 重複排除: 同一(ticker, entry_date)は1件のみ使用
    seen = set()
    unique_rows = []
    for row in rows:
        key = (row["ticker"], row["entry_date"])
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    print(f"  重複排除後: {len(unique_rows)} 件")

    candidates = []
    skipped = 0

    for row in unique_rows:
        ticker        = row["ticker"]
        signal        = row["signal"]
        signal_date_s = row["signal_date"]
        entry_date_s  = row["entry_date"]
        entry_price   = float(row["entry_price"])
        sl_price_csv  = float(row["sl_price"])

        df = price_data.get(ticker)
        if df is None or df.empty:
            skipped += 1; continue

        signal_ts = pd.Timestamp(signal_date_s)
        if signal_ts not in df.index:
            skipped += 1; continue
        df_idx = df.index.get_loc(signal_ts)
        if df_idx < 20:
            skipped += 1; continue

        # ATRをWilderのEMAで計算（engine.pyと同一ロジック）
        high  = df["High"]
        low   = df["Low"]
        close = df["Close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_series = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        atr = float(atr_series.iloc[df_idx])
        if atr <= 0:
            skipped += 1; continue

        # スコアはシグナル種別から推定（順序付けのみに使用）
        score = 0.80 if signal == "STRONG_BUY" else 0.70

        candidates.append({
            "ticker":        ticker,
            "date":          signal_date_s,
            "signal":        signal,
            "score":         score,
            "entry_price":   entry_price,
            "entry_date_str": entry_date_s,
            "atr":           atr,
            "sl_price":      sl_price_csv,
            "df":            df,
            "df_idx":        df_idx,
        })

    candidates.sort(key=lambda c: (c["date"], -c["score"]))
    print(f"  signal_candidates構築: {len(candidates)} 件（スキップ: {skipped} 件）")
    return candidates


# ============================================================
# メイン
# ============================================================
def main():
    import backtest.engine as eng
    from config.settings import ATR_PARAMS, TRADE_RULES
    from data.cache import CacheDB
    from data.price_fetcher import fetch_prices
    from data.earnings_fetcher import fetch_earnings_dates

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = BASE / f"output/backtest/exit_sweep_{ts}.csv"

    print("=" * 70)
    print("EXIT PARAMETER SWEEP v2（元CSVのエントリー信号を使用）")
    print(f"  入力CSV : {BASELINE_CSV.name}")
    print(f"  グリッド: 4 trail × 3 BE_trigger × 2 BE_lock × 2 hold = 48通り")
    print(f"  開始時刻: {ts}")
    print("=" * 70)

    # ── Step 1: CSVからティッカーリスト取得 ─────────────────────────
    print("\n[1/5] CSVからティッカーリスト取得...")
    csv_tickers = set()
    with open(BASELINE_CSV, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            csv_tickers.add(row["ticker"])
    print(f"  → {len(csv_tickers)} ティッカー")

    # ── Step 2: 株価データ取得（DBキャッシュ優先）────────────────────
    print("[2/5] 株価データ取得中（DBキャッシュ優先）...")
    db = CacheDB()
    db.initialize()
    tickers = sorted(csv_tickers)
    price_data = fetch_prices(tickers, db=db, period="6y", allow_stale=True)
    print(f"  → {len(price_data)} 銘柄")

    # SPY も取得（sorted_dates 用）
    spy_results = fetch_prices(["SPY"], db=db, period="6y", allow_stale=True)
    spy_df = spy_results.get("SPY")

    # ── Step 3: 決算日取得 ────────────────────────────────────────────
    print("[3/5] 決算日データ取得中...")
    earnings_dates = fetch_earnings_dates(tickers, force_refresh=False)
    print(f"  → {sum(1 for v in earnings_dates.values() if v)} 銘柄")

    # ── Step 4: ETFデータ取得 ─────────────────────────────────────────
    print("[4/5] ETFデータ取得中...")
    etf_tickers = ETF_UNIVERSE + [SAFE_TICKER]
    raw_etf = yf.download(etf_tickers, start=DATA_START, end=SIM_END,
                          auto_adjust=True, progress=False)
    etf_close: dict[str, pd.Series] = {}
    for tk in etf_tickers:
        try:
            s = raw_etf["Close"][tk].dropna()
            etf_close[tk] = s
        except Exception:
            pass
    print(f"  → {len(etf_close)} 本: {list(etf_close.keys())}")

    # ── Step 5: signal_candidates 再構築 ─────────────────────────────
    print("[5/5] signal_candidates を元CSVから再構築中...")
    signal_candidates = build_candidates_from_csv(BASELINE_CSV, price_data)

    # sorted_dates（全銘柄の取引日の和集合）
    all_dates: set[str] = set()
    for df in price_data.values():
        if df is not None and not df.empty:
            for dt in df.index:
                all_dates.add(dt.strftime("%Y-%m-%d"))
    if spy_df is not None:
        for dt in spy_df.index:
            all_dates.add(dt.strftime("%Y-%m-%d"))
    sorted_dates = sorted(all_dates)
    print(f"  → 取引日数: {len(sorted_dates)} 日")

    sl_multiplier = ATR_PARAMS.stop_loss_multiplier   # 2.0
    hard_stop_pct = TRADE_RULES.hard_stop_loss_pct    # -0.06

    # ── Phase2 × 48 + ETF sim ────────────────────────────────────────
    print(f"\nPhase2 × 48 + ETF sim 開始...\n")

    combo_list = list(itertools.product(
        TRAIL_VARIANTS.items(),
        BE_TRIGGER_VARIANTS,
        BE_LOCK_VARIANTS,
        HOLD_VARIANTS,
    ))
    total = len(combo_list)
    results = []

    for i, ((trail_name, trail_lvl), be_trig, be_lock, (hold_buy, hold_sb)) in enumerate(combo_list, 1):
        label = f"{trail_name}_BET{be_trig}_BEL{be_lock}_H{hold_buy}_{hold_sb}"
        is_baseline = (trail_name == "T0_base" and be_trig == 1.0 and
                       be_lock == 1.0 and hold_buy == 60 and hold_sb == 90)

        print(f"[{i:>2}/{total}] {label}", end="  ", flush=True)

        # エンジン定数パッチ
        eng.PROGRESSIVE_TRAIL_LEVELS = trail_lvl
        eng.BREAKEVEN_TRIGGER_ATR    = be_trig
        eng.BREAKEVEN_LOCK_ATR       = be_lock
        eng.MAX_HOLD_DAYS            = hold_buy
        eng.SB_MAX_HOLD_DAYS         = hold_sb

        try:
            trades = eng._run_phase2(
                signal_candidates=signal_candidates,
                sorted_dates=sorted_dates,
                earnings_dates=earnings_dates,
                sl_multiplier=sl_multiplier,
                hard_stop_pct=hard_stop_pct,
                sb_trail_mult=4.0,
            )
        except Exception as e:
            print(f"ERROR: {e}"); continue

        n_trades = len(trades)
        wins_raw = sum(1 for t in trades if t.pl_pct > 0)
        wr_raw   = wins_raw / n_trades * 100 if n_trades > 0 else 0
        avg_pl   = sum(t.pl_pct for t in trades) / n_trades * 100 if n_trades > 0 else 0

        trades_dict = trades_to_dict_list(trades)

        try:
            final, ret, max_dd, sharpe, calmar, pf, wr, nt = run_etf_simulation(
                trades_dict, etf_close
            )
        except Exception as e:
            print(f"ETF sim ERROR: {e}"); continue

        print(f"ret={ret:+.1f}%  sharpe={sharpe:.3f}  dd={max_dd:.2f}%  n={n_trades}  avgPL={avg_pl:+.2f}%")

        results.append({
            "label":       label,
            "trail":       trail_name,
            "be_trigger":  be_trig,
            "be_lock":     be_lock,
            "hold_buy":    hold_buy,
            "hold_sb":     hold_sb,
            "is_baseline": is_baseline,
            "n_trades":    n_trades,
            "win_rate":    wr_raw,
            "avg_pl_pct":  avg_pl,
            "return_pct":  ret,
            "max_dd":      max_dd,
            "sharpe":      sharpe,
            "calmar":      calmar,
            "pf":          pf,
            "final":       final,
        })

    # エンジン定数をベースラインに戻す
    eng.PROGRESSIVE_TRAIL_LEVELS = [(6.0, 2.0), (4.0, 2.5), (2.0, 3.0)]
    eng.BREAKEVEN_TRIGGER_ATR    = 1.0
    eng.BREAKEVEN_LOCK_ATR       = 1.0
    eng.MAX_HOLD_DAYS            = 60
    eng.SB_MAX_HOLD_DAYS         = 90

    if not results:
        print("結果なし"); return

    results.sort(key=lambda r: r["sharpe"], reverse=True)
    baseline = next((r for r in results if r["is_baseline"]), None)

    # ── 結果表示 ─────────────────────────────────────────────────────
    print("\n" + "=" * 115)
    print(f"{'RK':<4} {'LABEL':<45} {'Return%':>9} {'MaxDD%':>7} {'Sharpe':>8} "
          f"{'Calmar':>8} {'PF':>6} {'WR%':>6} {'AvgPL%':>7} {'N':>5} {'dSharpe':>8}")
    print("-" * 115)

    base_sharpe = baseline["sharpe"] if baseline else 0
    for rank, r in enumerate(results, 1):
        marker = " *** BASELINE" if r["is_baseline"] else ""
        ds = r["sharpe"] - base_sharpe
        print(f"{rank:<4} {r['label']:<45} {r['return_pct']:>+8.1f}% "
              f"{r['max_dd']:>6.2f}% {r['sharpe']:>8.3f} {r['calmar']:>8.2f} "
              f"{r['pf']:>6.3f} {r['win_rate']:>5.1f}% {r['avg_pl_pct']:>+6.2f}% "
              f"{r['n_trades']:>5} {ds:>+7.3f}{marker}")

    print("=" * 115)
    print(f"\n完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # CSV保存
    fieldnames = ["rank","label","trail","be_trigger","be_lock","hold_buy","hold_sb",
                  "is_baseline","n_trades","win_rate","avg_pl_pct","return_pct",
                  "max_dd","sharpe","calmar","pf","final"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rank, r in enumerate(results, 1):
            w.writerow({"rank": rank, **{k: r[k] for k in fieldnames[1:]}})
    print(f"CSV保存: {out_csv}")

    # サマリー
    best = results[0]
    print(f"\n── TOP 5 ──")
    for r in results[:5]:
        print(f"  {r['label']:<45} sharpe={r['sharpe']:.3f}  ret={r['return_pct']:+.1f}%  dd={r['max_dd']:.2f}%")
    if baseline:
        print(f"\n── BASELINE（rank {next(i for i,r in enumerate(results,1) if r['is_baseline'])}）──")
        print(f"  {baseline['label']:<45} sharpe={baseline['sharpe']:.3f}  ret={baseline['return_pct']:+.1f}%  dd={baseline['max_dd']:.2f}%")
        print(f"\n── BEST vs BASELINE ──")
        print(f"  Sharpe: {best['sharpe']:.3f} vs {baseline['sharpe']:.3f}  ({best['sharpe']-baseline['sharpe']:+.3f})")
        print(f"  Return: {best['return_pct']:+.1f}% vs {baseline['return_pct']:+.1f}%")
        print(f"  MaxDD:  {best['max_dd']:.2f}% vs {baseline['max_dd']:.2f}%")
        print(f"  Calmar: {best['calmar']:.2f} vs {baseline['calmar']:.2f}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
