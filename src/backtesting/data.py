from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List
import asyncio

import pandas as pd
import asyncio
from ..tools.yfinance_tool import fetch_daily_dataframe


def _fetch_sync(symbol: str, outputsize: str = "compact") -> pd.DataFrame:
    """Synchronous wrapper around the async fetch_daily_dataframe."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(fetch_daily_dataframe(symbol, outputsize=outputsize))
    # Running in async context: use thread pool
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(lambda: asyncio.run(fetch_daily_dataframe(symbol, outputsize=outputsize)))
        return future.result(timeout=30)


_SPY_CACHE: Optional[pd.DataFrame] = None


def load_spy_data(start_date, end_date) -> Optional[pd.DataFrame]:
    """Fetch SPY (S&P 500 ETF) OHLC data; cached for reuse across symbols."""
    global _SPY_CACHE
    if _SPY_CACHE is not None:
        cached = _SPY_CACHE
        cached_dates = cached.index.date if hasattr(cached.index, 'date') else cached.index
        if min(cached_dates) <= start_date.date() and max(cached_dates) >= end_date.date():
            return cached[(cached.index.date >= start_date.date()) & (cached.index.date <= end_date.date())]

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(_fetch_spy_sync)
                df = future.result(timeout=30)
        else:
            df = _fetch_spy_sync()
    except Exception:
        df = None

    if df is not None and not df.empty:
        _SPY_CACHE = df
        return df[(df.index.date >= start_date.date()) & (df.index.date <= end_date.date())]
    return None


def _fetch_spy_sync() -> Optional[pd.DataFrame]:
    """Synchronous SPY fetch (Alpha Vantage blocking call)."""
    import requests
    import os
    import time
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return None
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": "SPY",
        "outputsize": "compact",
        "apikey": api_key,
    }
    resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
    try:
        data = resp.json()
    except Exception:
        return None
    time.sleep(2)
    series_key = "Time Series (Daily)"
    if series_key not in data:
        return None
    rows = []
    for date_str, fields in data[series_key].items():
        rows.append({
            "Date": date_str,
            "Open": float(fields["1. open"]),
            "High": float(fields["2. high"]),
            "Low": float(fields["3. low"]),
            "Close": float(fields["4. close"]),
            "Volume": float(fields["5. volume"]),
        })
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df


def compute_equal_weight_benchmark(all_ohlc: Dict[str, pd.DataFrame], start_date, end_date) -> Optional[Tuple[pd.Series, Dict[str, Any]]]:
    """Compute equal-weight portfolio: each stock gets equal allocation at start, no rebalancing.

    Returns (portfolio_value_series, metrics_dict) or None.
    """
    if not all_ohlc:
        return None

    # Align all stocks to common date range
    date_range = pd.date_range(start_date, end_date, freq="B")
    aligned = {}
    for sym, df in all_ohlc.items():
        closes = df.set_index("Date")["Close"] if "Date" in df.columns else df["Close"]
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
        aligned[sym] = closes

    if not aligned:
        return None

    prices_df = pd.DataFrame(aligned).reindex(date_range).ffill().dropna(axis=0, how="all")
    if prices_df.empty:
        return None

    # Drop dates where no stock has data
    prices_df = prices_df.dropna(how="all")
    # Forward-fill within the available range
    prices_df = prices_df.ffill().dropna()

    if prices_df.empty or len(prices_df.columns) < 2:
        return None

    # Equal allocation: buy equal dollar amount at day 0
    n_stocks = len(prices_df.columns)
    allocation = 100000 / n_stocks
    shares = {}
    first_prices = prices_df.iloc[0]
    for col in prices_df.columns:
        price = first_prices[col]
        if price > 0:
            shares[col] = allocation / price
        else:
            shares[col] = 0

    portfolio_values = []
    for _, row in prices_df.iterrows():
        val = sum(shares[col] * row[col] for col in prices_df.columns if col in shares)
        portfolio_values.append(val)

    portfolio_series = pd.Series(portfolio_values, index=prices_df.index)

    # Calculate metrics
    returns = portfolio_series.pct_change().dropna()
    cumulative_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1) * 100
    annual_vol = returns.std() * (252 ** 0.5) * 100
    running_max = portfolio_series.cummax()
    drawdown = portfolio_series / running_max - 1
    max_dd = abs(drawdown.min()) * 100
    excess = returns - 0.02 / 252
    sharpe = excess.mean() / returns.std() * (252 ** 0.5) if returns.std() != 0 else 0

    metrics = {
        "Final Value": portfolio_series.iloc[-1],
        "Cumulative Return [%]": cumulative_return,
        "Annual Volatility [%]": annual_vol,
        "Max Drawdown [%]": max_dd,
        "Sharpe Ratio": sharpe,
        "Total Trades": n_stocks,
        "Strategy": "Equal Weight",
    }

    return portfolio_series, metrics


def _load_from_tiingo_cache(symbol: str) -> Optional[pd.DataFrame]:
    """从 Tiingo Parquet 缓存加载 OHLC 数据 (5 年日线)。"""
    cache_path = Path("output/cache/tiingo") / f"{symbol.upper()}_ohlcv.parquet"
    if not cache_path.exists():
        return None
    try:
        df = pd.read_parquet(cache_path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.rename(columns={"date": "Date"}).set_index("Date")
        return df
    except Exception as e:
        print(f"Tiingo cache load failed for {symbol}: {e}")
        return None


def load_stock_data(symbol: str, data_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """加载信号 CSV 并获取 OHLC 数据 (优先 Tiingo 缓存，回退 Alpha Vantage)。

    Returns (ohlc_df, signals_df) or (None, None) on error.
    """
    data_path = Path(data_dir)
    csv_file = data_path / f"daily_analysis_{symbol}.csv"

    if not csv_file.exists():
        print(f"Error: File {csv_file} not found!")
        return None, None

    signals_df = pd.read_csv(csv_file)
    signals_df["date"] = pd.to_datetime(signals_df["date"])
    signals_df = signals_df.sort_values("date").reset_index(drop=True)

    start_date = signals_df["date"].min()
    end_date = signals_df["date"].max()

    df = _load_from_tiingo_cache(symbol)
    if df is None:
        df = _fetch_sync(symbol)

    ohlc_data = df[(df.index.date >= start_date.date()) & (df.index.date <= (end_date + pd.Timedelta(days=0)).date())]

    if ohlc_data.empty:
        print(f"No OHLC data available for {symbol}")
        return None, None

    return ohlc_data.reset_index(), signals_df


def load_all_data(data_dir: str) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Load all symbols from data_dir as mapping symbol -> (ohlc_df, signals_df)."""
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("daily_analysis_*.csv"))
    all_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

    for csv_file in csv_files:
        symbol = csv_file.stem.replace("daily_analysis_", "")
        try:
            signals_df = pd.read_csv(csv_file)
            signals_df["date"] = pd.to_datetime(signals_df["date"])
            signals_df = signals_df.sort_values("date").reset_index(drop=True)

            start_date = signals_df["date"].min()
            end_date = signals_df["date"].max()

            df = _load_from_tiingo_cache(symbol)
            if df is None:
                df = _fetch_sync(symbol)

            ohlc_data = df[(df.index.date >= start_date.date()) & (df.index.date <= (end_date + pd.Timedelta(days=0)).date())]
            if not ohlc_data.empty:
                all_data[symbol] = (ohlc_data.reset_index(), signals_df)
            else:
                print(f"✗ {symbol}: No OHLC data available")
        except Exception as e:
            print(f"✗ {symbol}: Failed to load - {e}")

    return all_data


def list_available_stocks(data_dir: str) -> list[str]:
    """List stock symbols available as CSVs in data_dir."""
    path = Path(data_dir)
    if not path.exists():
        return []
    return sorted([p.stem.replace("daily_analysis_", "") for p in path.glob("daily_analysis_*.csv")])
