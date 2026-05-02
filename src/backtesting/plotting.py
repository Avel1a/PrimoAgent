from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt

from .engine import DEFAULT_CASH


def calculate_trade_execution(
    signals_df, dates: List, prices: List[float], starting_cash: float
) -> Tuple[List, List[float], List, List[float]]:
    """Calculate executed buy/sell points to overlay on portfolio chart."""
    buy_dates, buy_values, sell_dates, sell_values = [], [], [], []
    current_shares = 0
    current_cash = starting_cash

    for i, date in enumerate(dates):
        matching_signals = signals_df[signals_df["date"].dt.date == date]
        if matching_signals.empty:
            continue
        signal_row = matching_signals.iloc[0]
        current_price = prices[i]

        if signal_row["trading_signal"] == "BUY":
            position_percent = signal_row["position_size"] / 100.0
            target_cash = current_cash * position_percent
            shares_bought = int(target_cash / current_price) if current_cash > current_price else 0
            if shares_bought > 0:
                current_shares += shares_bought
                current_cash -= shares_bought * current_price
                buy_dates.append(date)
                buy_values.append(current_shares * current_price + current_cash)

        elif signal_row["trading_signal"] == "SELL":
            position_percent = signal_row["position_size"] / 100.0
            shares_sold = int(current_shares * position_percent) if current_shares > 0 else 0
            if shares_sold > 0:
                current_shares -= shares_sold
                current_cash += shares_sold * current_price
                sell_dates.append(date)
                sell_values.append(current_shares * current_price + current_cash)

    return buy_dates, buy_values, sell_dates, sell_values


def _extract_dates_prices(cerebro) -> Tuple[List, List[float]]:
    data_feed = cerebro.datas[0]
    data_len = len(data_feed.close.array)
    dates, prices = [], []
    for i in range(data_len):
        dt_val = data_feed.datetime.array[i]
        date_obj = datetime.fromordinal(int(dt_val)).date()
        dates.append(date_obj)
        prices.append(data_feed.close.array[i])
    return dates, prices


def plot_single_stock(
    symbol: str,
    primo_cerebro,
    buyhold_cerebro,
    output_dir: str,
    filename: str | None = None,
    spy_portfolio: list | None = None,
    spy_dates: list | None = None,
) -> Path:
    """Create single-stock portfolio comparison chart with S&P 500 benchmark."""
    dates, prices = _extract_dates_prices(primo_cerebro)

    primo_strategy = primo_cerebro.runstrats[0][0]
    primo_portfolio = getattr(primo_strategy, "portfolio_values", [])

    buyhold_portfolio = [DEFAULT_CASH]
    if prices:
        buyhold_shares = int(DEFAULT_CASH / prices[0])
        buyhold_cash_left = DEFAULT_CASH - (buyhold_shares * prices[0])
        for price in prices[1:]:
            buyhold_portfolio.append(buyhold_shares * price + buyhold_cash_left)

    if len(primo_portfolio) != len(dates):
        if len(primo_portfolio) < len(dates):
            last_value = primo_portfolio[-1] if primo_portfolio else DEFAULT_CASH
            primo_portfolio.extend([last_value] * (len(dates) - len(primo_portfolio)))
        else:
            primo_portfolio = primo_portfolio[: len(dates)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(dates, primo_portfolio, color="blue", linewidth=2, label="PrimoAgent Portfolio")
    ax1.plot(dates, buyhold_portfolio, color="red", linewidth=2, label=f"Buy & Hold {symbol}")
    if spy_portfolio is not None and spy_dates is not None and len(spy_portfolio) > 0:
        ax1.plot(spy_dates, spy_portfolio, color="orange", linewidth=2, linestyle="--", label="S&P 500 (SPY)")
    ax1.set_ylabel("Portfolio Value ($)")

    signals_df = getattr(primo_strategy, "signals_df", None)
    if signals_df is not None:
        buy_dates, buy_values, sell_dates, sell_values = calculate_trade_execution(
            signals_df, dates, prices, DEFAULT_CASH
        )
        if buy_dates:
            ax1.scatter(buy_dates, buy_values, color="green", marker="^", s=100, alpha=0.8, label="BUY Executed", zorder=5)
        if sell_dates:
            ax1.scatter(sell_dates, sell_values, color="red", marker="v", s=100, alpha=0.8, label="SELL Executed", zorder=5)

    ax1.legend(loc="upper left")
    ax1.set_title(f"{symbol}: PrimoAgent vs Buy & Hold Performance")
    ax1.grid(True, alpha=0.3)

    if signals_df is not None:
        buy_volumes, sell_volumes = [], []
        current_shares, current_cash = 0, DEFAULT_CASH
        for i, date in enumerate(dates):
            matching_signals = signals_df[signals_df["date"].dt.date == date]
            if not matching_signals.empty:
                signal_row = matching_signals.iloc[0]
                price = prices[i]
                if signal_row["trading_signal"] == "BUY":
                    position_percent = signal_row["position_size"] / 100.0
                    target_cash = current_cash * position_percent
                    shares_bought = int(target_cash / price) if current_cash > price else 0
                    if shares_bought > 0:
                        current_shares += shares_bought
                        current_cash -= shares_bought * price
                        buy_volumes.append(shares_bought)
                        sell_volumes.append(0)
                    else:
                        buy_volumes.append(0)
                        sell_volumes.append(0)
                elif signal_row["trading_signal"] == "SELL":
                    position_percent = signal_row["position_size"] / 100.0
                    shares_sold = int(current_shares * position_percent) if current_shares > 0 else 0
                    if shares_sold > 0:
                        current_shares -= shares_sold
                        current_cash += shares_sold * price
                        buy_volumes.append(0)
                        sell_volumes.append(-shares_sold)
                    else:
                        buy_volumes.append(0)
                        sell_volumes.append(0)
                else:
                    buy_volumes.append(0)
                    sell_volumes.append(0)
            else:
                buy_volumes.append(0)
                sell_volumes.append(0)

        ax2.bar(dates, buy_volumes, color="green", alpha=0.7, label="BUY Shares")
        ax2.bar(dates, sell_volumes, color="red", alpha=0.7, label="SELL Shares")
        ax2.set_ylabel("Number of Shares")
        ax2.set_xlabel("Date")
        ax2.set_title("Trading Volume")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / (filename or f"single_backtest_{symbol}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


def plot_returns_bar_chart(
    all_results: Dict[str, Dict[str, Any]],
    save_path: Path,
    spy_metrics: Dict[str, Any] | None = None,
    ew_metrics: Dict[str, Any] | None = None,
) -> None:
    """Create bar chart comparing PrimoAgent, Buy & Hold, S&P 500, and Equal Weight."""
    n_stocks = len(all_results)
    has_spy = spy_metrics is not None
    has_ew = ew_metrics is not None
    n_benchmarks = 1 + has_spy + has_ew  # Buy&Hold + optional SPY + optional EW

    fig, ax = plt.subplots(figsize=(14, 8))

    symbols = sorted(all_results.keys())
    primo_returns = [all_results[s]["primo"]["Cumulative Return [%]"] for s in symbols]
    buyhold_returns = [all_results[s]["buyhold"]["Cumulative Return [%]"] for s in symbols]

    x = range(len(symbols))
    width = 0.8 / (n_benchmarks + 1)  # +1 for PrimoAgent

    # PrimoAgent bars
    offset = -n_benchmarks * width / 2
    bars_p = ax.bar([i + offset for i in x], primo_returns, width, label="PrimoAgent", color="#1f77b4", alpha=0.9)
    offset += width

    # Buy & Hold bars
    bars_bh = ax.bar([i + offset for i in x], buyhold_returns, width, label="Buy & Hold", color="#ff7f0e", alpha=0.9)
    offset += width

    # S&P 500 benchmark (same value for all stocks, shown as horizontal reference)
    if has_spy:
        spy_return = spy_metrics["Cumulative Return [%]"]
        spy_values = [spy_return] * n_stocks
        bars_spy = ax.bar([i + offset for i in x], spy_values, width, label="S&P 500 (SPY)", color="#2ca02c", alpha=0.9)
        offset += width

    # Equal Weight benchmark
    if has_ew:
        ew_return = ew_metrics["Cumulative Return [%]"]
        ew_values = [ew_return] * n_stocks
        bars_ew = ax.bar([i + offset for i in x], ew_values, width, label="Equal Weight", color="#9467bd", alpha=0.9)

    ax.set_xlabel("Stocks")
    ax.set_ylabel("Cumulative Return (%)")
    title = "Performance Comparison"
    if has_spy:
        title += " (vs S&P 500)"
    ax.set_title(title)
    ax.set_xticks(list(x))
    ax.set_xticklabels(symbols)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Value labels on bars
    def label_bars(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (0.5 if height >= 0 else -1.5),
                f"{value:.1f}%",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=9,
                fontweight="bold",
            )

    label_bars(bars_p, primo_returns)
    label_bars(bars_bh, buyhold_returns)
    if has_spy:
        label_bars(bars_spy, spy_values)
    if has_ew:
        label_bars(bars_ew, ew_values)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

