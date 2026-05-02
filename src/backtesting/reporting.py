from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def generate_markdown_report(
    all_results: Dict[str, Dict[str, Any]],
    report_path: Path,
    spy_metrics: Dict[str, Any] | None = None,
    ew_metrics: Dict[str, Any] | None = None,
) -> None:
    """Generate markdown report with benchmarks."""
    lines = []
    lines.append("# PrimoAgent Multi-Stock Backtest Results")
    lines.append(f"\n*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    lines.append("\n## Performance Metrics")
    lines.append("\n| Stock | Strategy | Final Value | Return % | Volatility % | Sharpe | Max DD % | Trades |")
    lines.append("|-------|----------|-------------|----------|--------------|--------|----------|--------|")

    for symbol in sorted(all_results.keys()):
        primo = all_results[symbol]["primo"]
        buyhold = all_results[symbol]["buyhold"]
        lines.append(
            f"| {symbol} | PrimoAgent | ${primo['Final Value']:,.0f} | {primo['Cumulative Return [%]']:+.2f}% | "
            f"{primo['Annual Volatility [%]']:.2f}% | {primo['Sharpe Ratio']:.3f} | {primo['Max Drawdown [%]']:.2f}% | {primo['Total Trades']} |"
        )
        lines.append(
            f"| {symbol} | Buy & Hold | ${buyhold['Final Value']:,.0f} | {buyhold['Cumulative Return [%]']:+.2f}% | "
            f"{buyhold['Annual Volatility [%]']:.2f}% | {buyhold['Sharpe Ratio']:.3f} | {buyhold['Max Drawdown [%]']:.2f}% | {buyhold['Total Trades']} |"
        )
        lines.append("| | | | | | | | |")

    # Benchmark rows
    if spy_metrics:
        lines.append(
            f"| **S&P 500** | **SPY** | ${spy_metrics['Final Value']:,.0f} | {spy_metrics['Cumulative Return [%]']:+.2f}% | "
            f"{spy_metrics['Annual Volatility [%]']:.2f}% | {spy_metrics['Sharpe Ratio']:.3f} | {spy_metrics['Max Drawdown [%]']:.2f}% | {spy_metrics['Total Trades']} |"
        )
    if ew_metrics:
        lines.append(
            f"| **Equal Weight** | **Portfolio** | ${ew_metrics['Final Value']:,.0f} | {ew_metrics['Cumulative Return [%]']:+.2f}% | "
            f"{ew_metrics['Annual Volatility [%]']:.2f}% | {ew_metrics['Sharpe Ratio']:.3f} | {ew_metrics['Max Drawdown [%]']:.2f}% | {ew_metrics['Total Trades']} |"
        )

    lines.append("\n## Summary")
    total = len(all_results)
    wins = sum(1 for r in all_results.values() if r["primo"]["Cumulative Return [%]"] > r["buyhold"]["Cumulative Return [%]"])
    avg_primo = sum(r["primo"]["Cumulative Return [%]"] for r in all_results.values()) / total
    avg_bh = sum(r["buyhold"]["Cumulative Return [%]"] for r in all_results.values()) / total
    lines.append(f"- PrimoAgent beats Buy & Hold: **{wins}/{total}** ({wins/total*100:.1f}%)")
    lines.append(f"- Avg PrimoAgent return: **{avg_primo:+.2f}%** vs Buy & Hold: **{avg_bh:+.2f}%**")
    if spy_metrics:
        spy_ret = spy_metrics["Cumulative Return [%]"]
        lines.append(f"- S&P 500 (SPY) return: **{spy_ret:+.2f}%**")
        lines.append(f"- PrimoAgent vs S&P 500: **{avg_primo - spy_ret:+.2f}%** (alpha)")
    if ew_metrics:
        ew_ret = ew_metrics["Cumulative Return [%]"]
        lines.append(f"- Equal Weight return: **{ew_ret:+.2f}%**")
        lines.append(f"- PrimoAgent vs Equal Weight: **{avg_primo - ew_ret:+.2f}%**")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

