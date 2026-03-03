# -*- coding: utf-8 -*-
"""ETF v2 取引履歴の詳細出力"""
import csv, math, sys, os
from pathlib import Path
import pandas as pd
import yfinance as yf

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE   = Path(__file__).resolve().parent
CSV_IN = BASE / "output/backtest/signal_backtest_20260303_022802_sb_trail4.0.csv"

INITIAL  = 4000.0
RISK     = 0.005
ALLOC    = 0.13
SB_ALLOC = 0.14
MAX_POS  = 11
MIN_RR   = 0.005

ETF_UNIVERSE = ["SPY","QQQ","XLC","XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLK","XLRE","XLU"]
ETF_TOP_N    = 2
ETF_MA       = 50
SPY_TREND_MA = 200
SAFE_TICKER  = "SGOV"

MOMENTUM_PERIODS = {
    "1M": (21, 0.30), "3M": (63, 0.30), "6M": (126, 0.30), "12M": (252, 0.10),
}
SIM_START  = "2021-01-04"
SIM_END    = "2026-02-28"
DATA_START = "2019-10-01"

# ── 個別株シグナル読み込み ──
trades_raw = []
with open(CSV_IN, "r", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        trades_raw.append({
            "ticker": row["ticker"], "signal": row["signal"],
            "entry_date": row["entry_date"], "exit_date": row["exit_date"],
            "entry_price": float(row["entry_price"]), "sl_price": float(row["sl_price"]),
            "pl_pct": float(row["pl_pct"]),
        })
trades_raw.sort(key=lambda t: t["entry_date"])
same_day = {i for i, t in enumerate(trades_raw) if t["entry_date"] == t["exit_date"]}
entries_by_date: dict = {}
exits_by_date:   dict = {}
for i, t in enumerate(trades_raw):
    entries_by_date.setdefault(t["entry_date"], []).append(i)
    exits_by_date.setdefault(t["exit_date"], []).append(i)

# ── ETFデータ取得 ──
print("データ取得中...")
tickers_dl = ETF_UNIVERSE + [SAFE_TICKER]
raw_all = yf.download(tickers_dl, start=DATA_START, end=SIM_END, auto_adjust=True, progress=False)
etf_close: dict = {}
for tk in tickers_dl:
    try:
        etf_close[tk] = raw_all["Close"][tk].dropna()
    except Exception:
        pass
print(f"  完了: {list(etf_close.keys())}")

def ep(tk, date):
    s = etf_close.get(tk)
    if s is None: return None
    past = s[s.index <= date]
    return float(past.iloc[-1]) if len(past) > 0 else None

def ema(tk, date, period=ETF_MA):
    s = etf_close.get(tk)
    if s is None: return None
    past = s[s.index <= date]
    return float(past.iloc[-period:].mean()) if len(past) >= period else None

def spy_regime(date):
    s = etf_close.get("SPY")
    if s is None: return True
    past = s[s.index <= date]
    if len(past) < SPY_TREND_MA: return True
    return float(past.iloc[-1]) > float(past.iloc[-SPY_TREND_MA:].mean())

def calc_scores(date):
    rets = {}
    for tk in ETF_UNIVERSE:
        s = etf_close.get(tk)
        if s is None: continue
        past = s[s.index <= date]
        if len(past) == 0: continue
        cur = float(past.iloc[-1])
        tr = {}; ok = True
        for name, (days, _) in MOMENTUM_PERIODS.items():
            if len(past) < days + 1: ok = False; break
            base = float(past.iloc[-(days + 1)])
            if base <= 0: ok = False; break
            tr[name] = (cur - base) / base
        if ok: rets[tk] = tr
    if len(rets) < 2: return []
    scores = {t: 0.0 for t in rets}
    for name, (_, w) in MOMENTUM_PERIODS.items():
        ranked = sorted(rets.keys(), key=lambda t: rets[t][name], reverse=True)
        for rank, tk in enumerate(ranked, 1):
            scores[tk] += rank * w
    return sorted(scores.items(), key=lambda x: x[1])

def select_targets(date, scored):
    result = []
    for tk, _ in scored:
        if len(result) >= ETF_TOP_N: break
        p = ep(tk, date); ma = ema(tk, date)
        if p and ma and p >= ma: result.append(tk)
    return result

# ── シミュレーション（ETF取引ログ付き） ──
print("シミュレーション実行中...")
bdays = pd.bdate_range(SIM_START, SIM_END)
cash = INITIAL
stock_pos: dict = {}
etf_pos:   dict = {}
pe: set = set()
px: set = set()
prev_month = None
current_scores = []
prev_bull = None
balance = INITIAL

etf_txns = []

def log(date, tk, action, shares, price, value, reason):
    etf_txns.append({
        "date":   date.strftime("%Y-%m-%d"),
        "ticker": tk,
        "action": action,
        "shares": round(shares, 4),
        "price":  round(price, 4),
        "value":  round(value, 2),
        "reason": reason,
    })

for date in bdays:
    date_str = date.strftime("%Y-%m-%d")
    bull = spy_regime(date)
    regime_changed = (prev_bull is not None and bull != prev_bull)

    # ① 月初リバランス
    month_key = (date.year, date.month)
    if month_key != prev_month:
        prev_month = month_key
        if bull:
            current_scores = calc_scores(date)
            targets = select_targets(date, current_scores)
            score_map = {t: s for t, s in current_scores}
            for tk in list(etf_pos.keys()):
                if tk not in targets:
                    p = ep(tk, date)
                    if p:
                        val = etf_pos[tk] * p
                        cash += val
                        log(date, tk, "SELL", etf_pos[tk], p, val, "月初リバランス（非ターゲット）")
                    del etf_pos[tk]
            if targets and cash > 1.0:
                alloc_each = cash / len(targets)
                for tk in targets:
                    p = ep(tk, date)
                    if p and p > 0:
                        sh = alloc_each / p
                        etf_pos[tk] = etf_pos.get(tk, 0.0) + sh
                        cash -= alloc_each
                        sc = score_map.get(tk, 0)
                        # モメンタム詳細
                        s_obj = etf_close.get(tk)
                        past  = s_obj[s_obj.index <= date] if s_obj is not None else None
                        mom_strs = []
                        if past is not None and len(past) > 0:
                            cur = float(past.iloc[-1])
                            for nm, (days, _) in MOMENTUM_PERIODS.items():
                                if len(past) >= days + 1:
                                    base = float(past.iloc[-(days + 1)])
                                    if base > 0:
                                        mom_strs.append(f"{nm}:{(cur-base)/base*100:+.1f}%")
                        detail = " ".join(mom_strs)
                        log(date, tk, "BUY", sh, p, alloc_each,
                            f"月初リバランス（スコア{sc:.1f}, {detail}）")
        else:
            for tk in list(etf_pos.keys()):
                if tk != SAFE_TICKER:
                    p = ep(tk, date)
                    if p:
                        val = etf_pos[tk] * p
                        cash += val
                        log(date, tk, "SELL", etf_pos[tk], p, val, "月初（ベア相場）→SGOV待避")
                    del etf_pos[tk]
            sgov_p = ep(SAFE_TICKER, date)
            if sgov_p and cash > 1.0:
                sh = cash / sgov_p
                etf_pos[SAFE_TICKER] = etf_pos.get(SAFE_TICKER, 0.0) + sh
                log(date, SAFE_TICKER, "BUY", sh, sgov_p, sh * sgov_p, "月初（ベア）SGOV待避")
                cash = 0.0

    # ② 日次レジーム変化
    if regime_changed:
        if not bull:
            sgov_p = ep(SAFE_TICKER, date)
            for tk in list(etf_pos.keys()):
                if tk != SAFE_TICKER:
                    p = ep(tk, date)
                    if p:
                        val = etf_pos[tk] * p
                        cash += val
                        log(date, tk, "SELL", etf_pos[tk], p, val, "レジーム転換 ブル→ベア（SPY<200MA）")
                    del etf_pos[tk]
            if sgov_p and cash > 1.0:
                sh = cash / sgov_p
                etf_pos[SAFE_TICKER] = etf_pos.get(SAFE_TICKER, 0.0) + sh
                log(date, SAFE_TICKER, "BUY", sh, sgov_p, sh * sgov_p, "レジーム転換→SGOV待避")
                cash = 0.0
        else:
            if SAFE_TICKER in etf_pos:
                p = ep(SAFE_TICKER, date)
                if p:
                    val = etf_pos[SAFE_TICKER] * p
                    cash += val
                    log(date, SAFE_TICKER, "SELL", etf_pos[SAFE_TICKER], p, val,
                        "レジーム転換 ベア→ブル（SPY>200MA）")
                del etf_pos[SAFE_TICKER]
            for tk in list(etf_pos.keys()):
                p = ep(tk, date); ma = ema(tk, date)
                if p and ma and p < ma:
                    val = etf_pos[tk] * p
                    cash += val
                    log(date, tk, "SELL", etf_pos[tk], p, val, "50MA割れ")
                    del etf_pos[tk]
    elif bull:
        for tk in list(etf_pos.keys()):
            if tk == SAFE_TICKER: continue
            p = ep(tk, date); ma = ema(tk, date)
            if p and ma and p < ma:
                val = etf_pos[tk] * p
                cash += val
                log(date, tk, "SELL", etf_pos[tk], p, val, "50MA割れ")
                del etf_pos[tk]

    prev_bull = bull

    # ③ 個別株EXIT
    for idx in exits_by_date.get(date_str, []):
        if idx in same_day: continue
        if idx in pe and idx not in px:
            px.add(idx)
            t = trades_raw[idx]
            amt = stock_pos.pop(idx)
            cash += amt + amt * (t["pl_pct"] / 100)

    # ④ 個別株ENTRY
    etf_val  = sum(etf_pos[tk] * (ep(tk, date) or 0) for tk in etf_pos)
    total_eq = cash + sum(stock_pos.values()) + etf_val
    for idx in entries_by_date.get(date_str, []):
        if idx in pe: continue
        if len(stock_pos) >= MAX_POS: continue
        t = trades_raw[idx]
        rr   = max((t["entry_price"] - t["sl_price"]) / t["entry_price"]
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
                p = ep(tk, date)
                if not p: continue
                avail = etf_pos[tk] * p
                sell  = min(shortage, avail)
                sh    = sell / p
                etf_pos[tk] -= sh
                if etf_pos[tk] < 1e-9: del etf_pos[tk]
                cash += sell; shortage -= sell
                log(date, tk, "SELL", sh, p, sell, f"個別株資金捻出（{t['ticker']}）")
        alloc = min(need, cash)
        if t["entry_price"] > 0:
            sh    = math.floor(alloc / t["entry_price"])
            alloc = sh * t["entry_price"]
        else:
            sh = 0
        if alloc <= 0 or sh == 0 or cash <= 0: continue
        cash -= alloc; stock_pos[idx] = alloc; pe.add(idx)
        if idx in same_day:
            px.add(idx)
            cash += alloc + alloc * (t["pl_pct"] / 100)
            stock_pos.pop(idx, None)

    # ⑤ 残高記録
    etf_val = sum(etf_pos[tk] * (ep(tk, date) or 0) for tk in etf_pos)
    balance = cash + sum(stock_pos.values()) + etf_val

# ── 出力 ──
df = pd.DataFrame(etf_txns)
print(f"\n全ETF取引件数: {len(df)}件")

print("\n=== 理由別集計 ===")
summary = df.groupby(["reason", "action"]).agg(
    件数=("value", "count"),
    合計金額=("value", "sum"),
    平均金額=("value", "mean"),
).round(0)
print(summary.to_string())

print("\n=== Ticker別 BUY回数・平均保有額 ===")
buy_df = df[df["action"] == "BUY"]
print(f"{'Ticker':<8} {'BUY回数':>8} {'合計投資額':>12} {'平均投資額':>12}")
print("-" * 45)
for tk in sorted(buy_df["ticker"].unique()):
    td = buy_df[buy_df["ticker"] == tk]
    print(f"{tk:<8} {len(td):>8} ${td['value'].sum():>10,.0f} ${td['value'].mean():>10,.0f}")

# CSV保存
out_csv = BASE / "output/backtest/etf_v2_txn_history.csv"
df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\nCSV保存: {out_csv}")

# 全取引履歴表示
print("\n=== ETF v2 全取引履歴 ===")
print(f"{'#':>4}  {'日付':<12} {'Tk':<6} {'売買':>5} {'単価':>8} {'金額':>10}  理由")
print("-" * 90)
for i, row in df.iterrows():
    print(f"{i+1:>4}  {row['date']:<12} {row['ticker']:<6} {row['action']:>5} "
          f"${row['price']:>7.2f} ${row['value']:>9,.0f}  {row['reason']}")
