"""
全バリアント一括比較スクリプト

Case A: レバレッジETF（QLD/SSO）をユニバースに追加
Case B: スクリーニング緩和（DIP/出来高） ← Phase1再実行が必要なため別途実行
Case C: ベア相場インバースETF（SH 50%投入）
Case D: エントリー優先度をリスク調整スコア順に変更
Case E: STRONG_BUY の SL を ATR×2.0 → ATR×2.5 に拡大

実行順序:
  1. Phase2 バリアント（D, E, DE）→ ベースラインと比較
  2. ETF バリアント（A, C, AC）→ ベースラインと比較
  3. 最良 Phase2 × 最良 ETF の組み合わせを確認
"""
import os
import sys
import csv
import math
import multiprocessing
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import yfinance as yf

BASE = Path(__file__).resolve().parent
OUT  = BASE / "output" / "backtest"
OUT.mkdir(parents=True, exist_ok=True)

# ── ベースライン CSV（Phase2 変更なし） ───────────────────────────
BASELINE_CSV = BASE / "output/backtest/signal_backtest_20260305_223046_sb_trail4.0.csv"

# ── シミュレーションパラメータ ─────────────────────────────────────
INITIAL      = 4000.0
RISK         = 0.005
ALLOC        = 0.13
SB_ALLOC     = 0.14
MAX_POS      = 11
MIN_RR       = 0.005
SIM_START    = "2021-01-04"
SIM_END      = "2026-02-28"
DATA_START   = "2019-10-01"

# ── ETF 設定 ──────────────────────────────────────────────────────
ETF_UNIVERSE_BASE = [
    "SPY", "QQQ", "XLC", "XLY", "XLP",
    "XLE", "XLF", "XLV", "XLI", "XLB",
    "XLK", "XLRE", "XLU",
]
LEVERAGE_ETFS = ["QLD", "SSO"]
SAFE_TICKER   = "SGOV"
BEAR_ETF      = "SH"
BEAR_ETF_ALLOC = 0.5
ETF_TOP_N     = 2
ETF_MA        = 50
SPY_MA        = 200
REBAL_FREQ    = "weekly"
MOMENTUM_PERIODS = {
    "1M":  (21,  0.30),
    "3M":  (63,  0.30),
    "6M":  (126, 0.30),
    "12M": (252, 0.10),
}


# ════════════════════════════════════════════════════════════════════
# Part 0: ETF データ取得（共通）
# ════════════════════════════════════════════════════════════════════

print("ETFデータ取得中...")
_all_etf_tickers = ETF_UNIVERSE_BASE + LEVERAGE_ETFS + [SAFE_TICKER, BEAR_ETF]
_raw = yf.download(_all_etf_tickers, start=DATA_START, end=SIM_END,
                   auto_adjust=True, progress=False)
etf_close: dict[str, pd.Series] = {}
for tk in _all_etf_tickers:
    try:
        s = _raw["Close"][tk].dropna()
        etf_close[tk] = s
    except Exception:
        pass
print(f"  取得完了: {list(etf_close.keys())}")


# ════════════════════════════════════════════════════════════════════
# ETF シミュレーター（設定可能版）
# ════════════════════════════════════════════════════════════════════

def _etf_price(ticker, date):
    s = etf_close.get(ticker)
    if s is None: return None
    past = s[s.index <= date]
    return float(past.iloc[-1]) if len(past) > 0 else None

def _etf_ma(ticker, date, period=ETF_MA):
    s = etf_close.get(ticker)
    if s is None: return None
    past = s[s.index <= date]
    return float(past.iloc[-period:].mean()) if len(past) >= period else None

def _spy_regime(date):
    s = etf_close.get("SPY")
    if s is None: return True
    past = s[s.index <= date]
    if len(past) < SPY_MA: return True
    return float(past.iloc[-1]) > float(past.iloc[-SPY_MA:].mean())

def _calc_momentum_scores(date, universe):
    rets = {}
    for ticker in universe:
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

def _get_period_key(date):
    if REBAL_FREQ == "weekly":
        return (date.year, date.isocalendar()[1])
    return (date.year, date.month)


def run_etf_simulation(trades_raw, use_leverage=False, use_bear_etf=False):
    """ETFオーバーレイシミュレーション（フラグで Case A/C を切替）"""
    universe = ETF_UNIVERSE_BASE.copy()
    if use_leverage:
        universe += LEVERAGE_ETFS
    bear_safe = [SAFE_TICKER] + ([BEAR_ETF] if use_bear_etf else [])

    entries_by_date: dict[str, list[int]] = {}
    exits_by_date:   dict[str, list[int]] = {}
    same_day = {i for i, t in enumerate(trades_raw) if t["entry_date"] == t["exit_date"]}
    for i, t in enumerate(trades_raw):
        entries_by_date.setdefault(t["entry_date"], []).append(i)
        exits_by_date.setdefault(t["exit_date"],   []).append(i)

    bdays = pd.bdate_range(SIM_START, SIM_END)
    cash = INITIAL; stock_pos = {}; etf_pos = {}
    peak = INITIAL; max_dd = 0.0
    daily_records = []; trade_history = []
    pe: set[int] = set(); px: set[int] = set()
    prev_period = None; current_scores = []; prev_bull = None

    for date in bdays:
        date_str = date.strftime("%Y-%m-%d")
        bull = _spy_regime(date)
        reg_chg = (prev_bull is not None and bull != prev_bull)

        # ① リバランス
        period_key = _get_period_key(date)
        if period_key != prev_period:
            prev_period = period_key
            if bull:
                current_scores = _calc_momentum_scores(date, universe)
                targets = []
                for tk, _ in current_scores:
                    if len(targets) >= ETF_TOP_N: break
                    p = _etf_price(tk, date); ma = _etf_ma(tk, date)
                    if p and ma and p >= ma: targets.append(tk)
                for tk in list(etf_pos.keys()):
                    if tk not in targets:
                        p = _etf_price(tk, date)
                        if p: cash += etf_pos[tk] * p
                        del etf_pos[tk]
                if targets and cash > 1.0:
                    alloc_each = cash / len(targets)
                    for tk in targets:
                        p = _etf_price(tk, date)
                        if p and p > 0:
                            etf_pos[tk] = etf_pos.get(tk, 0.0) + alloc_each / p
                            cash -= alloc_each
            else:
                for tk in list(etf_pos.keys()):
                    if tk not in bear_safe:
                        p = _etf_price(tk, date)
                        if p: cash += etf_pos[tk] * p
                        del etf_pos[tk]
                if cash > 1.0:
                    if use_bear_etf:
                        bear_p = _etf_price(BEAR_ETF, date)
                        sgov_p = _etf_price(SAFE_TICKER, date)
                        if bear_p and sgov_p:
                            etf_pos[BEAR_ETF]    = etf_pos.get(BEAR_ETF, 0.0)    + cash * BEAR_ETF_ALLOC / bear_p
                            etf_pos[SAFE_TICKER] = etf_pos.get(SAFE_TICKER, 0.0) + cash * (1 - BEAR_ETF_ALLOC) / sgov_p
                            cash = 0.0
                    else:
                        sgov_p = _etf_price(SAFE_TICKER, date)
                        if sgov_p:
                            etf_pos[SAFE_TICKER] = etf_pos.get(SAFE_TICKER, 0.0) + cash / sgov_p
                            cash = 0.0

        # ② 日次: レジーム変化 + 50MA割れ
        if reg_chg and not bull:
            for tk in list(etf_pos.keys()):
                if tk not in bear_safe:
                    p = _etf_price(tk, date)
                    if p: cash += etf_pos[tk] * p
                    del etf_pos[tk]
            if cash > 1.0:
                if use_bear_etf:
                    bear_p = _etf_price(BEAR_ETF, date)
                    sgov_p = _etf_price(SAFE_TICKER, date)
                    if bear_p and sgov_p:
                        etf_pos[BEAR_ETF]    = etf_pos.get(BEAR_ETF, 0.0)    + cash * BEAR_ETF_ALLOC / bear_p
                        etf_pos[SAFE_TICKER] = etf_pos.get(SAFE_TICKER, 0.0) + cash * (1 - BEAR_ETF_ALLOC) / sgov_p
                        cash = 0.0
                else:
                    sgov_p = _etf_price(SAFE_TICKER, date)
                    if sgov_p and cash > 1.0:
                        etf_pos[SAFE_TICKER] = etf_pos.get(SAFE_TICKER, 0.0) + cash / sgov_p
                        cash = 0.0
        elif reg_chg and bull:
            for tk in bear_safe:
                if tk in etf_pos:
                    p = _etf_price(tk, date)
                    if p: cash += etf_pos[tk] * p
                    del etf_pos[tk]
            for tk in list(etf_pos.keys()):
                p = _etf_price(tk, date); ma = _etf_ma(tk, date)
                if p and ma and p < ma:
                    cash += etf_pos[tk] * p; del etf_pos[tk]
        elif bull:
            for tk in list(etf_pos.keys()):
                if tk in bear_safe: continue
                p = _etf_price(tk, date); ma = _etf_ma(tk, date)
                if p and ma and p < ma:
                    cash += etf_pos[tk] * p; del etf_pos[tk]

        prev_bull = bull

        # ③ 個別株 EXIT
        etf_val = sum(etf_pos[tk] * (_etf_price(tk, date) or 0) for tk in etf_pos)
        for idx in exits_by_date.get(date_str, []):
            if idx in same_day: continue
            if idx in pe and idx not in px:
                px.add(idx)
                t = trades_raw[idx]
                amt = stock_pos.pop(idx)
                pnl = amt * (t["pl_pct"] / 100)
                cash += amt + pnl
                etf_val = sum(etf_pos[tk] * (_etf_price(tk, date) or 0) for tk in etf_pos)
                bal = cash + sum(stock_pos.values()) + etf_val
                if bal > peak: peak = bal
                dd = (peak - bal) / peak * 100 if peak > 0 else 0
                if dd > max_dd: max_dd = dd
                trade_history.append({"pnl": pnl, "balance": bal})

        # ④ 個別株 ENTRY
        etf_val  = sum(etf_pos[tk] * (_etf_price(tk, date) or 0) for tk in etf_pos)
        total_eq = cash + sum(stock_pos.values()) + etf_val
        for idx in entries_by_date.get(date_str, []):
            if idx in pe: continue
            if len(stock_pos) >= MAX_POS: continue
            t = trades_raw[idx]
            rr = max((t["entry_price"] - t["sl_price"]) / t["entry_price"]
                     if t["entry_price"] > 0 else 0.08, MIN_RR)
            inv = (total_eq * RISK) / rr
            ap = SB_ALLOC if t["signal"] == "STRONG_BUY" else ALLOC
            need = min(inv, total_eq * ap)

            # ETF売却で資金調達
            score_map = {tk: s for tk, s in current_scores}
            if cash < need and etf_pos:
                shortage = need - cash
                ranked = sorted(etf_pos.keys(),
                                key=lambda tk: (0 if tk == SAFE_TICKER else 1,
                                                score_map.get(tk, 999)), reverse=True)
                for tk in ranked:
                    if shortage <= 0: break
                    p = _etf_price(tk, date)
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
                cash += alloc + alloc * (t["pl_pct"] / 100)
                stock_pos.pop(idx, None)

        # ⑤ 日次スナップショット
        etf_val  = sum(etf_pos[tk] * (_etf_price(tk, date) or 0) for tk in etf_pos)
        total_eq = cash + sum(stock_pos.values()) + etf_val
        if total_eq > peak: peak = total_eq
        dd = (peak - total_eq) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd
        daily_records.append({"date": date, "balance": total_eq})

    df = pd.DataFrame(daily_records)
    return df, trade_history, max_dd


def calc_stats(df, hist, max_dd, label):
    final = float(df.iloc[-1]["balance"])
    ret   = (final - INITIAL) / INITIAL * 100
    wins  = sum(1 for h in hist if h["pnl"] > 0)
    gp    = sum(h["pnl"] for h in hist if h["pnl"] > 0)
    gl    = abs(sum(h["pnl"] for h in hist if h["pnl"] < 0))
    pf    = gp / gl if gl > 0 else 9.99
    wr    = wins / len(hist) * 100 if hist else 0
    r     = df.set_index("date")["balance"].pct_change().dropna()
    sharpe = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0
    calmar = ret / max_dd if max_dd > 0 else 0
    yrs    = (pd.Timestamp(SIM_END) - pd.Timestamp(SIM_START)).days / 365.25
    cagr   = (final / INITIAL) ** (1 / yrs) - 1
    return {
        "label": label, "final": final, "ret": ret, "cagr": cagr * 100,
        "max_dd": max_dd, "sharpe": sharpe, "calmar": calmar,
        "pf": pf, "wr": wr, "n": len(hist),
    }


def load_trades(csv_path):
    trades = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            trades.append({
                "ticker":      row["ticker"],
                "signal":      row["signal"],
                "entry_date":  row["entry_date"],
                "exit_date":   row["exit_date"],
                "entry_price": float(row["entry_price"]),
                "exit_price":  float(row["exit_price"]),
                "pl_pct":      float(row["pl_pct"]),
                "sl_price":    float(row["sl_price"]),
            })
    trades.sort(key=lambda t: t["entry_date"])
    return trades


def print_table(results):
    print()
    print(f"{'Label':<22} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7} {'PF':>6} {'WR':>6}")
    print("-" * 72)
    for r in results:
        print(f"{r['label']:<22} {r['ret']:>+7.1f}%  {r['sharpe']:>6.3f}  "
              f"{r['max_dd']:>6.2f}%  {r['calmar']:>6.2f}  {r['pf']:>5.3f}  {r['wr']:>5.1f}%")
    print()


# ════════════════════════════════════════════════════════════════════
# Part 1: Phase2 バリアント（D, E）
# ════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  PART 1: Phase2 バリアント（Case D / E）")
print("=" * 72)
print("  Phase1 キャッシュを使用してバックテスト実行中...")

# main.py から run_signal_backtest_multi_trail を使う方法では D/E の制御ができないため、
# 独自に Phase2 だけを呼び出す。
# Phase1 は既存の signal CSV をそのまま再解釈して代用（エンジン再実行不要）。
# ただし Phase2 の sort 変更（D）は既存 CSV のシグナルには影響しないため、
# 既存 CSV を使って ETF overlay で比較するにとどめる。
# ※ フル Phase2 実行（D/E）は別途 run_backtest.py を --param-sweep で対応予定。

# 現時点では既存 CSV をベースに ETF バリアントを先に評価する。
print("  ※ Phase2 D/E は engine.py に実装済み（次回バックテスト実行時に有効）")
print("  → 本スクリプトでは ETF バリアント（A/C）を優先評価します")

baseline_trades = load_trades(BASELINE_CSV)

# ════════════════════════════════════════════════════════════════════
# Part 2: ETF バリアント（A, C, AC）
# ════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  PART 2: ETF バリアント（Case A / C）")
print("=" * 72)

etf_results = []

configs = [
    ("Baseline ETF",        False, False),
    ("A: +Leverage(QLD/SSO)", True,  False),
    ("C: +Bear(SH 50%)",    False, True),
    ("AC: Leverage+Bear",   True,  True),
]

for label, use_lev, use_bear in configs:
    print(f"  実行中: {label}...")
    df, hist, max_dd = run_etf_simulation(baseline_trades, use_leverage=use_lev, use_bear_etf=use_bear)
    stats = calc_stats(df, hist, max_dd, label)
    etf_results.append(stats)
    print(f"    → Return: {stats['ret']:+.1f}%  Sharpe: {stats['sharpe']:.3f}  MaxDD: {stats['max_dd']:.2f}%")

print_table(etf_results)

# 最良 ETF 設定を特定
best_etf = max(etf_results, key=lambda r: r["sharpe"])
print(f"  ★ ETF ベスト: {best_etf['label']} (Sharpe {best_etf['sharpe']:.3f}, Return {best_etf['ret']:+.1f}%)")

# ════════════════════════════════════════════════════════════════════
# まとめ
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  参照: ベースライン（ETF v2 weekly）")
print("    Return: +288.3%  Sharpe: 2.394  MaxDD: 13.43%  Calmar: 21.47")
print("=" * 72)
print()
print("  Case D / E の完全検証には:")
print("  python run_backtest.py を実行して新 CSV を生成後、")
print("  etf_overlay_backtest.py の CSV_IN を更新して再比較してください。")
print()
print("  Case B（スクリーニング緩和）は engine.py の DIP_MAX_PCT / VOLUME_FILTER_MULTIPLIER")
print("  を変更後に run_backtest.py で Phase1 から再実行が必要です（数時間）。")
