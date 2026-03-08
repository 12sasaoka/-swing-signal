"""Case D / E / DE のバックテスト（Phase1キャッシュ活用・高速）"""
import os, sys, multiprocessing, csv as csvlib, math
import datetime
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    from pathlib import Path
    import numpy as np
    import pandas as pd
    import yfinance as yf

    BASE = Path(__file__).resolve().parent
    OUT  = BASE / "output" / "backtest"
    OUT.mkdir(parents=True, exist_ok=True)

    # ── データ読み込み（main.py の _run_backtest と同じ流れ）─────────
    from data.cache import CacheDB
    from data.screener import fetch_iwv_holdings, screen_tier1
    from data.price_fetcher import fetch_prices
    from data.fundamental_fetcher import fetch_fundamentals
    from data.earnings_fetcher import fetch_earnings_dates
    from data.quarterly_fetcher import fetch_quarterly_data

    print("データ読み込み中（キャッシュ使用）...")
    db = CacheDB(); db.initialize()
    iwv = fetch_iwv_holdings(db)
    tickers = sorted(set(screen_tier1(iwv, db)))
    print(f"  Tier1: {len(tickers)} 銘柄")
    price_data  = fetch_prices(tickers, db=db, period="6y", allow_stale=True)
    spy_results = fetch_prices(["SPY"], db=db, period="6y", allow_stale=True)
    spy_df      = spy_results.get("SPY")
    fund_data   = fetch_fundamentals(tickers, db=db)
    earnings    = fetch_earnings_dates(tickers, force_refresh=False)
    quarterly   = fetch_quarterly_data(tickers, db=db)
    print(f"  価格: {len(price_data)} 銘柄、四半期: {sum(1 for v in quarterly.values() if v.get('quarters'))} 銘柄")

    # ── Phase1 キャッシュ読み込み + df 補完 ─────────────────────────
    from backtest.engine import (
        _load_phase1_cache, _compute_phase1_cache_key,
        _build_market_filter, _build_spy_hot_filter,
        _build_sector_momentum, _build_rolling_universe,
        _run_phase2, _aggregate_signal_results, WARMUP_DAYS,
    )
    from config.settings import ATR_PARAMS, TRADE_RULES
    from config.screening import SCREENING_PARAMS as _sp

    sl_mult    = ATR_PARAMS.stop_loss_multiplier
    hard_stop  = TRADE_RULES.hard_stop_loss_pct
    market_ok  = _build_market_filter(spy_df)
    spy_hot    = _build_spy_hot_filter(spy_df)
    sec_mom    = _build_sector_momentum(tickers, price_data, fund_data)
    rolling    = _build_rolling_universe(tickers, price_data,
                                         top_n=_sp.max_screen_candidates,
                                         min_history_days=WARMUP_DAYS)

    cache_key  = _compute_phase1_cache_key(tickers, bool(quarterly))
    candidates = _load_phase1_cache(cache_key)
    if candidates is None:
        print("Phase1 キャッシュなし → 終了（run_backtest.py を先に実行してください）")
        sys.exit(1)
    print(f"Phase1 キャッシュヒット: {len(candidates)} 件")

    for c in candidates:
        c["df"] = price_data[c["ticker"]]

    before = len(candidates)
    candidates = [c for c in candidates
                  if c["ticker"] in rolling.get(c["date"][:7], {c["ticker"]})]
    print(f"ローリングユニバースフィルタ後: {before} → {len(candidates)} 件")
    candidates.sort(key=lambda c: (c["date"], -c["score"]))

    all_dates: set[str] = set()
    for df in price_data.values():
        for d in df.index: all_dates.add(d.strftime("%Y-%m-%d"))
    sorted_dates = sorted(all_dates)

    # ── ETF オーバーレイ（簡易版・ベースライン設定のみ）────────────
    ETF_UNIVERSE = ["SPY","QQQ","XLC","XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLK","XLRE","XLU"]
    SAFE = "SGOV"; ETF_TOP_N = 2; ETF_MA_P = 50; SPY_MA_P = 200
    SIM_START = "2021-01-04"; SIM_END = "2026-02-28"; DATA_START = "2019-10-01"
    INITIAL = 4000.0; RISK = 0.005; ALLOC = 0.13; SB_ALLOC = 0.14
    MAX_POS = 11; MIN_RR = 0.005
    MOM = {"1M":(21,0.30),"3M":(63,0.30),"6M":(126,0.30),"12M":(252,0.10)}

    print("ETFデータ取得中...")
    _raw = yf.download(ETF_UNIVERSE + [SAFE], start=DATA_START, end=SIM_END,
                       auto_adjust=True, progress=False)
    ec: dict[str, pd.Series] = {}
    for tk in ETF_UNIVERSE + [SAFE]:
        try: ec[tk] = _raw["Close"][tk].dropna()
        except: pass

    def ep(t, d):
        s = ec.get(t); past = s[s.index <= d] if s is not None else None
        return float(past.iloc[-1]) if past is not None and len(past) > 0 else None

    def em(t, d):
        s = ec.get(t); past = s[s.index <= d] if s is not None else None
        return float(past.iloc[-ETF_MA_P:].mean()) if past is not None and len(past) >= ETF_MA_P else None

    def regime(d):
        s = ec.get("SPY"); past = s[s.index <= d] if s is not None else None
        if past is None or len(past) < SPY_MA_P: return True
        return float(past.iloc[-1]) > float(past.iloc[-SPY_MA_P:].mean())

    def calc_scores(d):
        rets = {}
        for tk in ETF_UNIVERSE:
            s = ec.get(tk)
            if s is None: continue
            past = s[s.index <= d]
            if not len(past): continue
            cur = float(past.iloc[-1]); tr = {}; ok = True
            for nm, (days, _) in MOM.items():
                if len(past) < days+1: ok=False; break
                base = float(past.iloc[-(days+1)])
                if base <= 0: ok=False; break
                tr[nm] = (cur-base)/base
            if ok: rets[tk] = tr
        if len(rets) < 2: return []
        sc = {t: 0.0 for t in rets}
        for nm, (_, w) in MOM.items():
            ranked = sorted(rets.keys(), key=lambda t: rets[t][nm], reverse=True)
            for rank, tk in enumerate(ranked, 1): sc[tk] += rank * w
        return sorted(sc.items(), key=lambda x: x[1])

    def run_etf(trades_raw):
        same_day = {i for i,t in enumerate(trades_raw) if t["entry_date"]==t["exit_date"]}
        eby = {}; exby = {}
        for i, t in enumerate(trades_raw):
            eby.setdefault(t["entry_date"],[]).append(i)
            exby.setdefault(t["exit_date"],[]).append(i)
        bdays = pd.bdate_range(SIM_START, SIM_END)
        cash = INITIAL; sp2 = {}; ep2 = {}; peak = INITIAL; max_dd = 0.0
        hist = []; pe2 = set(); px2 = set(); pp = None; cs = []; pb = None
        for date in bdays:
            ds = date.strftime("%Y-%m-%d"); bull = regime(date); rc = (pb is not None and bull != pb)
            pk = (date.year, date.isocalendar()[1])
            if pk != pp:
                pp = pk
                if bull:
                    cs = calc_scores(date); tgts = []
                    for tk, _ in cs:
                        if len(tgts) >= ETF_TOP_N: break
                        pv = ep(tk,date); mv = em(tk,date)
                        if pv and mv and pv >= mv: tgts.append(tk)
                    for tk in list(ep2.keys()):
                        if tk not in tgts:
                            pv = ep(tk,date)
                            if pv: cash += ep2[tk]*pv
                            del ep2[tk]
                    if tgts and cash > 1.0:
                        ae = cash/len(tgts)
                        for tk in tgts:
                            pv = ep(tk,date)
                            if pv and pv > 0: ep2[tk]=ep2.get(tk,0.0)+ae/pv; cash-=ae
                else:
                    for tk in list(ep2.keys()):
                        if tk != SAFE:
                            pv = ep(tk,date)
                            if pv: cash += ep2[tk]*pv
                            del ep2[tk]
                    sv = ep(SAFE,date)
                    if sv and cash > 1.0: ep2[SAFE]=ep2.get(SAFE,0.0)+cash/sv; cash=0.0
            if rc and not bull:
                for tk in list(ep2.keys()):
                    if tk != SAFE:
                        pv = ep(tk,date)
                        if pv: cash += ep2[tk]*pv
                        del ep2[tk]
                sv = ep(SAFE,date)
                if sv and cash > 1.0: ep2[SAFE]=ep2.get(SAFE,0.0)+cash/sv; cash=0.0
            elif rc and bull:
                if SAFE in ep2:
                    pv = ep(SAFE,date)
                    if pv: cash += ep2[SAFE]*pv
                    del ep2[SAFE]
                for tk in list(ep2.keys()):
                    pv=ep(tk,date); mv=em(tk,date)
                    if pv and mv and pv < mv: cash+=ep2[tk]*pv; del ep2[tk]
            elif bull:
                for tk in list(ep2.keys()):
                    if tk == SAFE: continue
                    pv=ep(tk,date); mv=em(tk,date)
                    if pv and mv and pv < mv: cash+=ep2[tk]*pv; del ep2[tk]
            pb = bull
            ev = sum(ep2[tk]*(ep(tk,date) or 0) for tk in ep2)
            for idx in exby.get(ds, []):
                if idx in same_day: continue
                if idx in pe2 and idx not in px2:
                    px2.add(idx); t=trades_raw[idx]; amt=sp2.pop(idx)
                    pnl=amt*(t["pl_pct"]/100); cash+=amt+pnl
                    ev=sum(ep2[tk]*(ep(tk,date) or 0) for tk in ep2)
                    bal=cash+sum(sp2.values())+ev
                    if bal>peak: peak=bal
                    dd=(peak-bal)/peak*100 if peak>0 else 0
                    if dd>max_dd: max_dd=dd
                    hist.append({"pnl":pnl,"bal":bal})
            ev=sum(ep2[tk]*(ep(tk,date) or 0) for tk in ep2)
            te=cash+sum(sp2.values())+ev; sm={tk:s for tk,s in cs}
            for idx in eby.get(ds, []):
                if idx in pe2: continue
                if len(sp2) >= MAX_POS: continue
                t=trades_raw[idx]
                rr=max((t["entry_price"]-t["sl_price"])/t["entry_price"] if t["entry_price"]>0 else 0.08, MIN_RR)
                inv=(te*RISK)/rr; ap=SB_ALLOC if t["signal"]=="STRONG_BUY" else ALLOC
                need=min(inv,te*ap)
                if cash<need and ep2:
                    sh3=need-cash
                    rk=sorted(ep2.keys(),key=lambda tk:(0 if tk==SAFE else 1,sm.get(tk,999)),reverse=True)
                    for tk in rk:
                        if sh3<=0: break
                        pv=ep(tk,date)
                        if not pv: continue
                        sell=min(sh3,ep2[tk]*pv); ep2[tk]-=sell/pv
                        if ep2[tk]<1e-9: del ep2[tk]
                        cash+=sell; sh3-=sell
                alloc=min(need,cash)
                if t["entry_price"]>0: shr=math.floor(alloc/t["entry_price"]); alloc=shr*t["entry_price"]
                else: shr=0
                if alloc<=0 or shr==0 or cash<=0: continue
                cash-=alloc; sp2[idx]=alloc; pe2.add(idx)
                if idx in same_day:
                    px2.add(idx); cash+=alloc+alloc*(t["pl_pct"]/100); sp2.pop(idx,None)
            ev=sum(ep2[tk]*(ep(tk,date) or 0) for tk in ep2)
            te=cash+sum(sp2.values())+ev
            if te>peak: peak=te
            dd=(peak-te)/peak*100 if peak>0 else 0
            if dd>max_dd: max_dd=dd
        final=cash+sum(sp2.values())+sum(ep2.get(tk,0)*(ep(tk,bdays[-1]) or 0) for tk in ep2)
        ret=(final-INITIAL)/INITIAL*100
        wins=sum(1 for h in hist if h["pnl"]>0); gp=sum(h["pnl"] for h in hist if h["pnl"]>0)
        gl=abs(sum(h["pnl"] for h in hist if h["pnl"]<0)); pf=gp/gl if gl>0 else 9.99
        wr=wins/len(hist)*100 if hist else 0
        bals=[h["bal"] for h in hist]
        if len(bals)>1:
            r=pd.Series(bals).pct_change().dropna()
            sharpe=(r.mean()/r.std())*np.sqrt(252) if r.std()>0 else 0
        else: sharpe=0.0
        calmar=ret/max_dd if max_dd>0 else 0
        return ret, sharpe, max_dd, calmar, pf, wr, len(hist)

    # ── バリアント実行 ─────────────────────────────────────────────
    print("\n" + "="*75)
    print("  Case D / E / DE バリアント vs ベースライン")
    print("="*75)
    print(f"{'Label':<22} {'Trades':>7} {'WR':>6} {'AvgPL':>7} {'ETF Ret':>9} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7}")
    print("-"*75)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    variants = [
        ("Baseline",       False, None),
        ("D: RiskAdjSort", True,  None),
        ("E: SB SL×2.5",   False, 2.5),
        ("DE: D+E",        True,  2.5),
    ]

    for label, sort_risk, sb_sl in variants:
        trades_p2 = _run_phase2(
            signal_candidates=candidates,
            sorted_dates=sorted_dates,
            earnings_dates=earnings,
            sl_multiplier=sl_mult,
            hard_stop_pct=hard_stop,
            sb_trail_mult=4.0,
            sort_by_risk_adj=sort_risk,
            sb_sl_mult=sb_sl,
        )
        agg = _aggregate_signal_results(trades_p2)

        slug = label.replace(":","").replace(" ","_").replace("+","").replace("×","x").replace(".","")
        csv_out = OUT / f"signal_backtest_{ts}_{slug}.csv"
        rows = [t.to_dict() for t in trades_p2]
        if rows:
            with open(csv_out, "w", encoding="utf-8", newline="") as f:
                w = csvlib.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader(); w.writerows(rows)

        trades_raw_etf = []
        for row in rows:
            trades_raw_etf.append({
                "ticker": row["ticker"], "signal": row["signal"],
                "entry_date": row["entry_date"], "exit_date": row["exit_date"],
                "entry_price": float(row["entry_price"]), "exit_price": float(row["exit_price"]),
                "pl_pct": float(row["pl_pct"]), "sl_price": float(row["sl_price"]),
            })
        trades_raw_etf.sort(key=lambda t: t["entry_date"])
        ret_e, sh_e, dd_e, cal_e, pf_e, wr_e, n_e = run_etf(trades_raw_etf)

        print(f"{label:<22} {agg.total_trades:>7}  {agg.win_rate*100:>5.1f}%  "
              f"{agg.avg_pl_pct*100:>+6.2f}%  {ret_e:>+8.1f}%  {sh_e:>6.3f}  {dd_e:>6.2f}%  {cal_e:>6.2f}")

    print()
    print("参照ベースライン（2026-03-07 確定値）:")
    print("  Return +288.3%  Sharpe 2.394  MaxDD 13.43%  Calmar 21.47")
