"""3パターンのパラメータスイープ実行スクリプト"""
import subprocess, sys, re, os

patterns = [
    ("0.70/9.5/11", 0.0070, 0.095, 11),
    ("0.65/9.0/11", 0.0065, 0.090, 11),
    ("0.60/8.0/10", 0.0060, 0.080, 10),
]

with open("simulate_portfolio.py", encoding="utf-8") as f:
    original = f.read()

results = []
for label, risk, alloc, pos in patterns:
    src = original
    src = re.sub(r"MAX_POSITIONS\s*=\s*\d+",    f"MAX_POSITIONS = {pos}",    src)
    src = re.sub(r"RISK_PER_TRADE\s*=\s*[\d.]+", f"RISK_PER_TRADE = {risk}", src)
    src = re.sub(r"MAX_ALLOC_PCT\s*=\s*[\d.]+",  f"MAX_ALLOC_PCT = {alloc}", src)

    with open("simulate_portfolio.py", "w", encoding="utf-8") as f:
        f.write(src)

    out = subprocess.check_output(
        [sys.executable, "simulate_portfolio.py"],
        stderr=subprocess.DEVNULL
    ).decode("utf-8", errors="replace")

    def g(pat): m = re.search(pat, out); return m.group(1) if m else "?"

    results.append({
        "label":  label,
        "final":  g(r"Final Balance:\s+\$\s*([\d,\.]+)"),
        "ret":    g(r"Total Return:\s+([+-][\d\.]+%)"),
        "wr":     g(r"Wins:\s+\d+\s+\((\d+\.\d+)%\)"),
        "pf":     g(r"Profit Factor:\s+([\d\.]+)"),
        "plr":    g(r"P/L Ratio:\s+([\d\.]+)"),
        "mdd":    g(r"Max Drawdown:\s+\$[\d,\.]+ \(([\d\.]+)%\)"),
        "skip":   g(r"Skipped \(full\):\s+(\d+)"),
        "trades": g(r"Total Trades:\s+(\d+)"),
    })
    print(f"  [{label}] {results[-1]['ret']}  PF={results[-1]['pf']}  skip={results[-1]['skip']}")

# ベースラインに戻す
with open("simulate_portfolio.py", "w", encoding="utf-8") as f:
    f.write(original)
print("\n  -> ベースラインに戻しました\n")

# 結果表示
baseline = {
    "label":"Baseline(0.75/10/12)",
    "final":"7,491.77","ret":"+87.29%","wr":"38.2",
    "pf":"1.323","plr":"2.137","mdd":"21.14","skip":"12","trades":"680"
}
all_r = [baseline] + results

print("=" * 92)
print(f"  {'Pattern':<22} {'Final $':>12} {'Return':>9} {'WinRate':>8} {'PF':>7} {'PLR':>7} {'MaxDD':>7} {'Skip':>6} {'Trades':>7}")
print("  " + "-" * 88)
for r in all_r:
    marker = " <-- baseline" if "Baseline" in r["label"] else ""
    print(f"  {r['label']:<22} ${r['final']:>12} {r['ret']:>9} {r['wr']:>7}% {r['pf']:>7} {r['plr']:>7} {r['mdd']:>6}% {r['skip']:>6} {r['trades']:>7}{marker}")
print("=" * 92)
