"""批量回测：3 只股票 × 40 天 → 自动回测"""
import subprocess, sys, os, time
from pathlib import Path

STOCKS = ["AAPL", "MSFT", "GOOGL"]
START, END = "2026-03-02", "2026-04-30"
PROJECT = Path(__file__).parent
VENV = str(PROJECT / "venv" / "Scripts" / "python.exe")

t0 = time.time()
print(f"PrimoAgent 批量回测 — {len(STOCKS)} 只股票, {START} ~ {END}")
print("=" * 60)

for i, sym in enumerate(STOCKS, 1):
    print(f"\n[{i}/{len(STOCKS)}] {sym} 分析中...")
    env = {**os.environ, "PRIMO_SYMBOL": sym, "PRIMO_START_DATE": START, "PRIMO_END_DATE": END}
    r = subprocess.run([VENV, "-u", "main.py"], cwd=str(PROJECT), env=env,
                       capture_output=False, timeout=None)
    if r.returncode != 0:
        print(f"[FAIL] {sym} 返回值 {r.returncode}")
    else:
        print(f"[OK] {sym} 完成")

print(f"\n分析总耗时: {(time.time()-t0)/60:.1f} 分钟")
print("\n开始多股票回测...")
subprocess.run([VENV, "-u", "backtest.py"], cwd=str(PROJECT),
               input=b"2\n1\nn\nn\n", timeout=300)
print(f"\n全部完成! 总耗时: {(time.time()-t0)/60:.1f} 分钟")
