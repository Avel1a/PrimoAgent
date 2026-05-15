"""批量测试：牛/熊/震荡 三组对比 — 6只股票"""
import subprocess, sys, os, time
from pathlib import Path

GROUPS = [
    ("NVDA", "2024-01-02", "2024-03-08", "Bull"),
    ("MSFT", "2024-01-02", "2024-03-08", "Bull"),
    ("TSLA", "2022-09-01", "2022-11-10", "Bear"),
    ("AAPL", "2022-08-01", "2022-10-15", "Bear"),
    ("JNJ",  "2023-06-01", "2023-08-15", "Sideways"),
    ("XOM",  "2023-09-01", "2023-11-15", "Sideways"),
]

PROJECT = Path(__file__).parent
VENV = str(PROJECT / "venv" / "Scripts" / "python.exe")
BACKTEST_SCRIPT = str(PROJECT / "_backtest_single.py")

t0 = time.time()
print(f"PrimoAgent 市场状态对比测试 — {len(GROUPS)} 只股票, 3 组 (Bull / Bear / Sideways)")
print("=" * 60)

# ── Step 1: 逐只跑分析流水线 ──
for i, (sym, start, end, regime) in enumerate(GROUPS, 1):
    print(f"\n[{i}/{len(GROUPS)}] [{regime}] {sym} | {start} ~ {end}")
    env = {**os.environ, "PRIMO_SYMBOL": sym, "PRIMO_START_DATE": start, "PRIMO_END_DATE": end}
    r = subprocess.run([VENV, "-u", "main.py"], cwd=str(PROJECT), env=env,
                       capture_output=False, timeout=None)
    if r.returncode != 0:
        print(f"  [FAIL] {sym} 返回值 {r.returncode}")
    else:
        print(f"  [OK] {sym} 分析完成")

print(f"\n分析总耗时: {(time.time()-t0)/60:.1f} 分钟")

# ── Step 2: 逐只跑回测 ──
print("\n" + "=" * 60)
print("开始逐只回测...\n")

for sym, start, end, regime in GROUPS:
    label = f"[{regime}] {sym}"
    print(f"{label} 回测中...")
    r = subprocess.run(
        [VENV, "-u", BACKTEST_SCRIPT, sym],
        cwd=str(PROJECT), capture_output=False, timeout=300,
    )
    if r.returncode != 0:
        print(f"  [FAIL] {sym} 回测返回值 {r.returncode}")
    else:
        print(f"  [OK] {sym} 回测完成")

elapsed = (time.time() - t0) / 60
print(f"\n{'=' * 60}")
print(f"全部完成! 总耗时: {elapsed:.1f} 分钟")
print(f"输出目录: output/csv/ | output/backtests/")
