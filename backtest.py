from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.backtesting import (
    run_backtest,
    PrimoAgentStrategy,
    BuyAndHoldStrategy,
)
from src.backtesting.data import load_stock_data, load_all_data, list_available_stocks, load_spy_data, compute_equal_weight_benchmark
from src.backtesting.plotting import plot_single_stock, plot_returns_bar_chart
from src.backtesting.reporting import generate_markdown_report
from src.config import config


def _prompt(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def yes_no(question: str, default: bool = True) -> bool:
    suffix = "[Y/n] " if default else "[y/N] "
    while True:
        ans = _prompt(f"{question} {suffix}").strip().lower()
        if ans == "" and default is not None:
            return default
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please enter 'y' for yes or 'n' for no.")


def choose_mode() -> str:
    print("\nSelect backtest mode:")
    print("  1) Single stock")
    print("  2) Multiple stocks")
    print("  q) Quit")
    while True:
        choice = _prompt("> ").strip().lower()
        if choice in {"1", "2", "q"}:
            return choice
        print("Invalid selection. Enter 1, 2 or q.")


def choose_symbol(available: list[str]) -> str | None:
    print("\nAvailable symbols:")
    for i, s in enumerate(available, 1):
        print(f"  {i:>2}) {s}")
    print("Enter index or symbol (blank to cancel):")
    while True:
        val = _prompt("> ").strip()
        if val == "":
            return None
        if val.isdigit():
            idx = int(val)
            if 1 <= idx <= len(available):
                return available[idx - 1]
        else:
            up = val.upper()
            if up in available:
                return up
        print("Invalid input. Try again.")


def choose_symbols_multi(available: list[str]) -> list[str] | None:
    print("\nAvailable symbols:")
    for i, s in enumerate(available, 1):
        print(f"  {i:>2}) {s}")
    if yes_no("Run for ALL symbols in the list?", default=True):
        return available
    print("Enter comma-separated indices (e.g. 1,3,5), or blank to cancel:")
    while True:
        val = _prompt("> ").strip()
        if val == "":
            return None
        try:
            idxs = [int(x) for x in val.split(",") if x.strip()]
            picked = []
            for i in idxs:
                if 1 <= i <= len(available):
                    picked.append(available[i - 1])
            if picked:
                # Ukloni duplikate uz očuvanje redoslijeda
                seen = set()
                uniq = []
                for s in picked:
                    if s not in seen:
                        seen.add(s)
                        uniq.append(s)
                return uniq
        except ValueError:
            pass
        print("Invalid input. Try again.")


def pick_paths() -> tuple[Path, Path]:
    # Determine data dir: prefer ./output/csv, fallback to ./data, then ./tests/data
    preferred = Path("output/csv")
    if preferred.exists():
        data_dir = preferred
    else:
        alt1 = Path("data")
        alt2 = Path("tests/data")
        if alt1.exists():
            print("'output/csv' not found. Using 'data/'.")
            data_dir = alt1
        elif alt2.exists():
            print("'output/csv' and 'data/' not found. Using 'tests/data/'.")
            data_dir = alt2
        else:
            print("No data directory found. Expected one of: output/csv, data, tests/data.")
            raise SystemExit(1)

    # Output directory (default: output/backtests)
    default_output = Path("output/backtests")
    use_default = yes_no(f"Use the default output directory '{default_output}'?", default=True)
    if use_default:
        output_dir = default_output
    else:
        # allow user to enter custom path
        while True:
            entered = _prompt("Enter output directory path: ").strip()
            if entered:
                output_dir = Path(entered)
                break
            print("Path cannot be empty.")

    output_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, output_dir


def run_single_interactive(data_dir: Path, output_dir: Path) -> int:
    available = list_available_stocks(str(data_dir))
    if not available:
        print(f"No CSV files available in '{data_dir}'.")
        return 1

    symbol = choose_symbol(available)
    if not symbol:
        print("Cancelled.")
        return 0

    printlog = yes_no("Enable detailed strategy logs?", default=False)

    ohlc_data, signals_df = load_stock_data(symbol, str(data_dir))
    if ohlc_data is None or signals_df is None:
        return 1

    # S&P 500 benchmark
    spy_portfolio = None
    spy_dates = None
    spy_metrics = None
    start_date = signals_df["date"].min()
    end_date = signals_df["date"].max()
    spy_ohlc = load_spy_data(start_date, end_date)
    if spy_ohlc is not None and not spy_ohlc.empty:
        spy_ohlc_reset = spy_ohlc.reset_index()
        spy_prices = spy_ohlc_reset["Close"].tolist()
        spy_dates_list = spy_ohlc_reset["Date"].dt.date.tolist() if "Date" in spy_ohlc_reset.columns else spy_ohlc_reset["index"].dt.date.tolist()
        if spy_prices:
            spy_shares = 100000 / spy_prices[0]
            spy_portfolio = [spy_shares * p for p in spy_prices]
            spy_dates = spy_dates_list
            spy_returns_series = pd.Series(spy_prices).pct_change().dropna()
            spy_cum_ret = (spy_portfolio[-1] / spy_portfolio[0] - 1) * 100
            spy_vol = spy_returns_series.std() * (252 ** 0.5) * 100
            spy_peak = pd.Series(spy_portfolio).cummax()
            spy_dd = abs((pd.Series(spy_portfolio) / spy_peak - 1).min()) * 100
            excess = spy_returns_series - 0.02 / 252
            spy_sharpe = excess.mean() / spy_returns_series.std() * (252 ** 0.5) if spy_returns_series.std() != 0 else 0
            spy_metrics = {
                "Final Value": spy_portfolio[-1],
                "Cumulative Return [%]": spy_cum_ret,
                "Annual Volatility [%]": spy_vol,
                "Max Drawdown [%]": spy_dd,
                "Sharpe Ratio": spy_sharpe,
                "Total Trades": 1,
                "Strategy": "S&P 500",
            }

    print(f"\nPRIMOAGENT SINGLE STOCK BACKTEST - {symbol}")
    print("=" * 60)

    primo_results, primo_cerebro = run_backtest(
        ohlc_data, PrimoAgentStrategy, "PrimoAgent", signals_df=signals_df, printlog=printlog, trailing_stop_pct=config.risk_stop_loss_pct, take_profit_pct=config.risk_take_profit_pct
    )
    buyhold_results, buyhold_cerebro = run_backtest(
        ohlc_data, BuyAndHoldStrategy, "Buy & Hold"
    )

    print("\nPerformance comparison")
    print("-" * 65)
    if spy_metrics:
        print(f"{'Metric':<22} {'PrimoAgent':>12} {'Buy & Hold':>12} {'S&P 500':>12}")
    else:
        print(f"{'Metric':<22} {'PrimoAgent':>12} {'Buy & Hold':>12} {'Difference':>12}")
    metrics = [
        "Cumulative Return [%]",
        "Annual Volatility [%]",
        "Max Drawdown [%]",
        "Sharpe Ratio",
        "Total Trades",
    ]
    for m in metrics:
        pv, bv = primo_results[m], buyhold_results[m]
        if spy_metrics:
            sv = spy_metrics[m]
            print(f"{m:<22} {pv:>12.2f} {bv:>12.2f} {sv:>12.2f}")
        else:
            diff = pv - bv
            if "[%]" in m or "Ratio" in m:
                print(f"{m:<22} {pv:>12.2f} {bv:>12.2f} {diff:>+12.2f}")
            else:
                print(f"{m:<22} {pv:>12.0f} {bv:>12.0f} {diff:>+12.0f}")

    rel = primo_results["Cumulative Return [%]"] - buyhold_results["Cumulative Return [%]"]
    vs_spy = ""
    if spy_metrics:
        rel_spy = primo_results["Cumulative Return [%]"] - spy_metrics["Cumulative Return [%]"]
        vs_spy = f" | vs S&P 500: {rel_spy:+.2f}%"
    if rel > 0:
        print(f"\nPrimoAgent OUTPERFORMED Buy & Hold by {rel:+.2f}%!{vs_spy}")
    else:
        print(f"\nPrimoAgent underperformed Buy & Hold by {abs(rel):.2f}%{vs_spy}")

    chart_path = plot_single_stock(
        symbol, primo_cerebro, buyhold_cerebro, str(output_dir),
        spy_portfolio=spy_portfolio, spy_dates=spy_dates,
    )
    print(f"Chart saved: {chart_path}")
    return 0


def run_multi_interactive(data_dir: Path, output_dir: Path) -> int:
    available = list_available_stocks(str(data_dir))
    if not available:
        print(f"No CSV files available in '{data_dir}'.")
        return 1

    symbols = choose_symbols_multi(available)
    if not symbols:
        print("Cancelled.")
        return 0

    printlog = yes_no("Enable detailed strategy logs?", default=False)

    print("\nPRIMOAGENT MULTI-STOCK BACKTEST")
    print("=" * 50)
    print(f"Selected: {', '.join(symbols)}")

    all_results = {}
    all_ohlc = {}
    spy_portfolio = None
    spy_dates = None
    spy_metrics = None
    ew_metrics = None

    for symbol in symbols:
        print(f"\n{'=' * 60}\nProcessing {symbol}\n{'=' * 60}")
        ohlc_data, signals_df = load_stock_data(symbol, str(data_dir))
        if ohlc_data is None or signals_df is None:
            print(f"Skipping {symbol} (no data)")
            continue
        try:
            all_ohlc[symbol] = ohlc_data.copy()
            primo_results, primo_cerebro = run_backtest(
                ohlc_data, PrimoAgentStrategy, f"{symbol} PrimoAgent", signals_df=signals_df, printlog=printlog, trailing_stop_pct=config.risk_stop_loss_pct, take_profit_pct=config.risk_take_profit_pct
            )
            buyhold_results, buyhold_cerebro = run_backtest(
                ohlc_data, BuyAndHoldStrategy, f"{symbol} Buy & Hold"
            )
            all_results[symbol] = {"primo": primo_results, "buyhold": buyhold_results}

            # individual chart
            _ = plot_single_stock(symbol, primo_cerebro, buyhold_cerebro, str(output_dir), f"backtest_results_{symbol}.png")

            # quick comparison
            primo_return = primo_results["Cumulative Return [%]"]
            buyhold_return = buyhold_results["Cumulative Return [%]"]
            rel = primo_return - buyhold_return
            if rel > 0:
                print(f"{symbol}: PrimoAgent +{rel:.2f}% ( {primo_return:.2f}% vs {buyhold_return:.2f}% )")
            else:
                print(f"{symbol}: PrimoAgent -{abs(rel):.2f}% ( {primo_return:.2f}% vs {buyhold_return:.2f}% )")
        except Exception as e:
            print(f"Error for {symbol}: {e}")

    if not all_results:
        print("No successful backtests.")
        return 1

    # --- S&P 500 benchmark (SPY) ---
    # Determine date range from already-loaded OHLC data
    all_starts = []
    all_ends = []
    for ohlc_df in all_ohlc.values():
        dates = ohlc_df["Date"] if "Date" in ohlc_df.columns else ohlc_df.index
        all_starts.append(pd.to_datetime(dates).min())
        all_ends.append(pd.to_datetime(dates).max())
    all_start = min(all_starts)
    all_end = max(all_ends)
    spy_ohlc = load_spy_data(all_start, all_end)
    if spy_ohlc is not None and not spy_ohlc.empty:
        spy_ohlc_reset = spy_ohlc.reset_index()
        spy_prices = spy_ohlc_reset["Close"].tolist()
        spy_dates_list = spy_ohlc_reset["Date"].dt.date.tolist() if "Date" in spy_ohlc_reset.columns else spy_ohlc_reset["index"].dt.date.tolist()
        if spy_prices:
            spy_shares = 100000 / spy_prices[0]
            spy_portfolio = [spy_shares * p for p in spy_prices]
            spy_dates = spy_dates_list
            spy_returns_series = pd.Series(spy_prices).pct_change().dropna()
            spy_metrics = {
                "Final Value": spy_portfolio[-1],
                "Cumulative Return [%]": (spy_portfolio[-1] / spy_portfolio[0] - 1) * 100,
                "Annual Volatility [%]": spy_returns_series.std() * (252 ** 0.5) * 100,
                "Max Drawdown [%]": abs((pd.Series(spy_portfolio) / pd.Series(spy_portfolio).cummax() - 1).min()) * 100,
                "Sharpe Ratio": (spy_returns_series.mean() - 0.02 / 252) / spy_returns_series.std() * (252 ** 0.5) if spy_returns_series.std() != 0 else 0,
                "Total Trades": 1,
                "Strategy": "S&P 500",
            }
            print(f"S&P 500 (SPY) return: {spy_metrics['Cumulative Return [%]']:+.2f}%")

    # --- Equal Weight benchmark ---
    if len(all_ohlc) >= 2:
        ew_result = compute_equal_weight_benchmark(all_ohlc, all_start, all_end)
        if ew_result is not None:
            _, ew_metrics = ew_result
            print(f"Equal Weight return: {ew_metrics['Cumulative Return [%]']:+.2f}%")

    # aggregate chart and report
    bar_chart_path = output_dir / "returns_comparison.png"
    plot_returns_bar_chart(all_results, bar_chart_path, spy_metrics=spy_metrics, ew_metrics=ew_metrics)
    print(f"Returns comparison chart saved: {bar_chart_path}")

    report_path = output_dir / "backtest_analysis_report.md"
    generate_markdown_report(all_results, report_path, spy_metrics=spy_metrics, ew_metrics=ew_metrics)
    print(f"Report saved: {report_path}")

    total = len(all_results)
    wins = sum(1 for r in all_results.values() if r["primo"]["Cumulative Return [%]"] > r["buyhold"]["Cumulative Return [%]"])
    avg_primo = sum(r["primo"]["Cumulative Return [%]"] for r in all_results.values()) / total
    avg_bh = sum(r["buyhold"]["Cumulative Return [%]"] for r in all_results.values()) / total
    print("\nCOMPLETE")
    print("=" * 50)
    print(f"Stocks: {total}")
    print(f"PrimoAgent wins vs Buy & Hold: {wins}/{total} ({wins/total*100:.1f}%)")
    print(f"Avg PrimoAgent: {avg_primo:.2f}% | Buy & Hold: {avg_bh:.2f}% | Relative: {avg_primo - avg_bh:+.2f}%")
    if spy_metrics:
        spy_ret = spy_metrics["Cumulative Return [%]"]
        wins_spy = sum(1 for r in all_results.values() if r["primo"]["Cumulative Return [%]"] > spy_ret)
        print(f"PrimoAgent wins vs S&P 500: {wins_spy}/{total} ({wins_spy/total*100:.1f}%)")
        print(f"Alpha vs S&P 500: {avg_primo - spy_ret:+.2f}%")
    if ew_metrics:
        ew_ret = ew_metrics["Cumulative Return [%]"]
        wins_ew = sum(1 for r in all_results.values() if r["primo"]["Cumulative Return [%]"] > ew_ret)
        print(f"PrimoAgent wins vs Equal Weight: {wins_ew}/{total} ({wins_ew/total*100:.1f}%)")
    print(f"Outputs: {output_dir.resolve()}")
    return 0


def main() -> int:
    print("PrimoAgent Backtest (interactive mode)")
    data_dir, output_dir = pick_paths()

    mode = choose_mode()
    if mode == "q":
        print("Goodbye.")
        return 0
    if mode == "1":
        return run_single_interactive(data_dir, output_dir)
    if mode == "2":
        return run_multi_interactive(data_dir, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
