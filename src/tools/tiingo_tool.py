import os
import asyncio
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from .utils import ToolResult

TIINGO_BASE_URL = "https://api.tiingo.com"
TIINGO_DAILY_PRICES_ENDPOINT = "/tiingo/daily/{ticker}/prices"
TIINGO_METADATA_ENDPOINT = "/tiingo/daily/{ticker}"

# Disk cache for historical OHLCV data (immutable, fetched once, reused forever)
OHLCV_CACHE_DIR = Path("./output/cache/tiingo")
OHLCV_CACHE_FETCH_YEARS = 5  # Fetch this many years on first call to maximize cache coverage


def _get_ohlcv_cache_path(symbol: str) -> Path:
    return OHLCV_CACHE_DIR / f"{symbol.upper()}_ohlcv.parquet"


def _read_ohlcv_cache(symbol: str) -> Optional[pd.DataFrame]:
    """Read cached OHLCV DataFrame from disk. Returns None on cache miss or corruption."""
    path = _get_ohlcv_cache_path(symbol)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception:
        return None


def _write_ohlcv_cache(symbol: str, df: pd.DataFrame) -> None:
    """Write merged OHLCV DataFrame to disk cache."""
    path = _get_ohlcv_cache_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_to_save = df.reset_index()
    df_to_save["date"] = df_to_save["date"].dt.strftime("%Y-%m-%d")
    df_to_save.to_parquet(path, index=False)

def _get_tiingo_api_key() -> Optional[str]:
    """Get Tiingo API key from environment variables."""
    return os.getenv('TIINGO_API_KEY')

def _convert_symbol_to_tiingo(symbol: str) -> str:
    """Convert symbol to Tiingo format (replace '.' with '-')."""
    # Tiingo uses dashes instead of periods for share classes
    # e.g., BRK-A instead of BRK.A
    return symbol.replace('.', '-')

def _make_tiingo_request(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Make a request to Tiingo API with authentication."""
    api_key = _get_tiingo_api_key()
    if not api_key:
        print("[Tiingo] No API key found, skipping request")
        return None

    url = TIINGO_BASE_URL + endpoint
    headers = {
        'Authorization': f'Token {api_key}',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[Tiingo] API request failed: {e}")
        return None

async def _apply_rate_limiting():
    """Rate limit Tiingo API calls (free tier: 50 req/hr, ~1 req per 72s)."""
    await asyncio.sleep(5)


async def get_market_data(symbol: str, analysis_date: Optional[str] = None, period: str = "1y") -> ToolResult:
    """
    Get market data for a symbol from Tiingo with disk caching.

    First call fetches a wide date range (5 years) and caches to Parquet.
    Subsequent calls hit the disk cache and only fetch new data if the cache
    doesn't cover the requested end_date. This eliminates redundant API calls.

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        analysis_date: Optional date for analysis (YYYY-MM-DD format)
        period: Period for historical data slice (default "1y")

    Returns:
        ToolResult with market data
    """
    try:
        tiingo_symbol = _convert_symbol_to_tiingo(symbol)

        # Determine desired date range
        end_date = datetime.now().date()
        if analysis_date:
            end_date = datetime.strptime(analysis_date, "%Y-%m-%d").date()

        period_days = {"1y": 365, "6mo": 180, "3mo": 90, "1mo": 30, "1wk": 7}
        delta_days = period_days.get(period, 365)
        desired_start = end_date - timedelta(days=delta_days)

        # --- Disk cache lookup ---
        cached_df = _read_ohlcv_cache(symbol)
        cache_hit = False
        if cached_df is not None:
            cache_min = cached_df.index.min().date()
            cache_max = cached_df.index.max().date()
            if cache_min <= desired_start and cache_max >= end_date:
                cache_hit = True
                df = cached_df.copy()

        # --- API fetch (only when cache misses or needs newer data) ---
        if not cache_hit:
            # Fetch wide range on first call to maximize future cache coverage
            if cached_df is None:
                fetch_start = end_date - timedelta(days=OHLCV_CACHE_FETCH_YEARS * 365)
            else:
                # Only fetch data newer than what's cached
                cache_max = cached_df.index.max().date()
                fetch_start = cache_max + timedelta(days=1)
                # But also ensure we cover the desired period
                if desired_start < fetch_start:
                    fetch_start = desired_start

            start_str = fetch_start.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            endpoint = TIINGO_DAILY_PRICES_ENDPOINT.format(ticker=tiingo_symbol)
            params = {"startDate": start_str, "endDate": end_str, "format": "json"}

            await _apply_rate_limiting()
            data = _make_tiingo_request(endpoint, params)

            if data is None or not data:
                # If API fails but we have stale cache, use it as fallback
                if cached_df is not None:
                    df = cached_df.copy()
                else:
                    return ToolResult(
                        success=False,
                        error=f"Failed to fetch data from Tiingo for {symbol}",
                    )

            new_df = pd.DataFrame(data)
            if not new_df.empty:
                new_df["date"] = pd.to_datetime(new_df["date"]).dt.tz_localize(None)
                new_df.set_index("date", inplace=True)
                new_df.sort_index(inplace=True)

                # Merge with cache
                if cached_df is not None:
                    df = pd.concat([cached_df, new_df])
                    df = df[~df.index.duplicated(keep="last")]
                    df.sort_index(inplace=True)
                else:
                    df = new_df

                _write_ohlcv_cache(symbol, df)
            elif cached_df is not None:
                df = cached_df.copy()
            else:
                return ToolResult(
                    success=False, error=f"No data available for {symbol}"
                )

        # --- Slice to requested period ---
        df = df[(df.index.date >= desired_start) & (df.index.date <= end_date)]

        if df.empty:
            return ToolResult(
                success=False,
                error=f"No data available for {symbol} in requested range",
            )

        latest = df.iloc[-1]
        previous_close = None
        price_change = None
        price_change_pct = None

        if len(df) >= 2:
            previous = df.iloc[-2]
            previous_close = float(previous["close"])
            current_close = float(latest["close"])
            price_change = current_close - previous_close
            price_change_pct = (price_change / previous_close) * 100 if previous_close != 0 else 0

        # Prepare historical data
        historical_clean = df.reset_index()
        historical_clean["date"] = historical_clean["date"].dt.strftime("%Y-%m-%d")
        for col in ["open", "high", "low", "close", "volume"]:
            if col in historical_clean.columns:
                historical_clean[col] = historical_clean[col].astype(float)

        historical_dict = historical_clean.to_dict("records")

        result_date = analysis_date if analysis_date else str(df.index[-1].date())
        result_data = {
            "symbol": symbol.upper(),
            "date": result_date,
            "current_price": float(latest["close"]),
            "price_data": {
                "open": float(latest["open"]),
                "high": float(latest["high"]),
                "low": float(latest["low"]),
                "close": float(latest["close"]),
                "volume": int(latest["volume"]),
                "previous_close": previous_close,
                "price_change": price_change,
                "price_change_pct": price_change_pct,
            },
            "historical_data": historical_dict,
        }

        return ToolResult(success=True, data=result_data)

    except Exception as e:
        return ToolResult(
            success=False,
            error=f"Error getting market data for {symbol} from Tiingo: {str(e)}",
        )

async def get_company_info(symbol: str) -> ToolResult:
    """
    Get company information for a symbol from Tiingo.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
    
    Returns:
        ToolResult with company information
    """
    try:
        tiingo_symbol = _convert_symbol_to_tiingo(symbol)
        endpoint = TIINGO_METADATA_ENDPOINT.format(ticker=tiingo_symbol)
        
        data = _make_tiingo_request(endpoint)
        if data is None:
            return ToolResult(success=False, error=f"Failed to fetch company info from Tiingo for {symbol}")
        
        if not data:
            return ToolResult(success=False, error=f"No company info available for {symbol}")
        
        # Extract relevant fields
        company_data = {
            "symbol": symbol.upper(),
            "name": data.get("name", "N/A"),
            "description": data.get("description", "N/A"),
            "exchange": data.get("exchangeCode", "N/A"),
            "start_date": data.get("startDate", "N/A"),
            "end_date": data.get("endDate", "N/A"),
        }
        
        return ToolResult(success=True, data=company_data)
        
    except Exception as e:
        return ToolResult(success=False, error=f"Error getting company info for {symbol} from Tiingo: {str(e)}")
