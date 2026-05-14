"""非交互式批量回测 - 使用可用 CSV 数据"""
from pathlib import Path
from src.backtesting import run_backtest, PrimoAgentStrategy, BuyAndHoldStrategy
from src.backtesting.data import load_stock_data, list_available_stocks, load_spy_data, compute_equal_weight_benchmark
from src.backtesting.plotting import plot_single_stock, plot_returns_bar_chart
from src.backtesting.reporting import generate_markdown_report
from src.config import config
import pandas as pd

data_dir = Path("output/csv")
output_dir = Path("output/backtests")
output_dir.mkdir(parents=True, exist_ok=True)

available = list_available_stocks(str(data_dir))
print(f"Available stocks: {available}")

if not available:
    print("No CSV data found!")
    exit(1)

all_results = {}
all_ohlc = {}

for symbol in available:
    print(f"\n{'='*60}\nProcessing {symbol}\n{'='*60}")
    ohlc_data, signals_df = load_stock_data(symbol, str(data_dir))
    if ohlc_data is None or signals_df is None:
        print(f"  Skipping {symbol} (no data)")
        continue

    try:
        all_ohlc[symbol] = ohlc_data.copy()

        primo_results, primo_cerebro = run_backtest(
            ohlc_data, PrimoAgentStrategy, f"{symbol} PrimoAgent",
            signals_df=signals_df, printlog=False,
            trailing_stop_pct=config.risk_stop_loss_pct,
            take_profit_pct=config.risk_take_profit_pct,
        )
        buyhold_results, buyhold_cerebro = run_backtest(
            ohlc_data, BuyAndHoldStrategy, f"{symbol} Buy & Hold"
        )
        all_results[symbol] = {"primo": primo_results, "buyhold": buyhold_results}

        plot_single_stock(symbol, primo_cerebro, buyhold_cerebro, str(output_dir), f"backtest_results_{symbol}.png")

        primo_ret = primo_results["Cumulative Return [%]"]
        bh_ret = buyhold_results["Cumulative Return [%]"]
        rel = primo_ret - bh_ret
        print(f"  PrimoAgent: {primo_ret:+.2f}% | Buy&Hold: {bh_ret:+.2f}% | Diff: {rel:+.2f}%")
    except Exception as e:
        print(f"  Error: {e}")

if not all_results:
    print("No results!")
    exit(1)

# SPY benchmark
all_starts, all_ends = [], []
for df in all_ohlc.values():
    dates = df["Date"] if "Date" in df.columns else df.index
    all_starts.append(pd.to_datetime(dates).min())
    all_ends.append(pd.to_datetime(dates).max())

spy_metrics = None
spy_ohlc = load_spy_data(min(all_starts), max(all_ends))
if spy_ohlc is not None and not spy_ohlc.empty:
    spy_reset = spy_ohlc.reset_index()
    spy_prices = spy_reset["Close"].tolist()
    spy_shares = 100000 / spy_prices[0]
    spy_portfolio = [spy_shares * p for p in spy_prices]
    spy_returns = pd.Series(spy_prices).pct_change().dropna()
    spy_metrics = {
        "Final Value": spy_portfolio[-1],
        "Cumulative Return [%]": (spy_portfolio[-1] / spy_portfolio[0] - 1) * 100,
        "Annual Volatility [%]": spy_returns.std() * (252 ** 0.5) * 100,
        "Max Drawdown [%]": abs((pd.Series(spy_portfolio) / pd.Series(spy_portfolio).cummax() - 1).min()) * 100,
        "Sharpe Ratio": (spy_returns.mean() - 0.02 / 252) / spy_returns.std() * (252 ** 0.5) if spy_returns.std() != 0 else 0,
        "Total Trades": 1,
        "Strategy": "S&P 500",
    }

ew_metrics = None
if len(all_ohlc) >= 2:
    ew_result = compute_equal_weight_benchmark(all_ohlc, min(all_starts), max(all_ends))
    if ew_result:
        _, ew_metrics = ew_result

# Charts and report
plot_returns_bar_chart(all_results, output_dir / "returns_comparison.png", spy_metrics=spy_metrics, ew_metrics=ew_metrics)
generate_markdown_report(all_results, output_dir / "backtest_analysis_report.md", spy_metrics=spy_metrics, ew_metrics=ew_metrics)

total = len(all_results)
wins = sum(1 for r in all_results.values() if r["primo"]["Cumulative Return [%]"] > r["buyhold"]["Cumulative Return [%]"])
avg_primo = sum(r["primo"]["Cumulative Return [%]"] for r in all_results.values()) / total
avg_bh = sum(r["buyhold"]["Cumulative Return [%]"] for r in all_results.values()) / total

print(f"\n{'='*60}")
print(f"COMPLETE: {total} stocks, PrimoAgent wins {wins}/{total} ({wins/total*100:.1f}%)")
print(f"Avg PrimoAgent: {avg_primo:+.2f}% | Buy&Hold: {avg_bh:+.2f}% | Diff: {avg_primo - avg_bh:+.2f}%")
