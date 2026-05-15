"""一键分析+回测模板 — 改 STOCKS 和日期直接跑"""
import subprocess, sys, os, time
from pathlib import Path

# ============================================================
# 修改这里：股票列表、起止日期
# ============================================================
STOCKS = [
    ("NVDA", "2026-03-02", "2026-04-30"),
    ("JPM",  "2026-03-02", "2026-04-30"),
    ("XOM",  "2026-03-02", "2026-04-30"),
]
# ============================================================

PROJECT = Path(__file__).parent
VENV = str(PROJECT / "venv" / "Scripts" / "python.exe")

t0 = time.time()
print(f"PrimoAgent 批量分析 + 回测 — {len(STOCKS)} 只股票")
print("=" * 60)

# ── Step 1: 串行跑分析（避免 API 限流）──
for i, (sym, start, end) in enumerate(STOCKS, 1):
    print(f"\n[{i}/{len(STOCKS)}] {sym} | {start} ~ {end}")
    env = {**os.environ, "PRIMO_SYMBOL": sym, "PRIMO_START_DATE": start, "PRIMO_END_DATE": end}
    r = subprocess.run(
        [VENV, "-u", "main.py"], cwd=str(PROJECT), env=env,
        capture_output=False, timeout=None,
    )
    if r.returncode != 0:
        print(f"  [FAIL] {sym} 返回值 {r.returncode}")
    else:
        print(f"  [OK] {sym} 完成")

elapsed_analysis = (time.time() - t0) / 60
print(f"\n分析耗时: {elapsed_analysis:.1f} 分钟")

# ── Step 2: 多股票回测 ──
print("\n" + "=" * 60)
print("多股票回测中...\n")

r = subprocess.run(
    [VENV, "-u", "backtest.py"], cwd=str(PROJECT),
    input=b"y\n2\ny\nn\n", timeout=300,
)
if r.returncode != 0:
    print(f"[FAIL] 回测返回值 {r.returncode}")

elapsed = (time.time() - t0) / 60
print(f"\n{'=' * 60}")
print(f"全部完成! 总耗时: {elapsed:.1f} 分钟")
print(f"CSV:   output/csv/")
print(f"图表:  output/backtests/")
