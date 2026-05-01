import os
import asyncio
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from .utils import ToolResult
from dotenv import load_dotenv

load_dotenv()

TIINGO_BASE_URL = "https://api.tiingo.com"
TIINGO_DAILY_PRICES_ENDPOINT = "/tiingo/daily/{ticker}/prices"
TIINGO_METADATA_ENDPOINT = "/tiingo/daily/{ticker}"

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
    print(f"[DEBUG Tiingo] API key available: {'YES' if api_key else 'NO'}")
    if api_key:
        print(f"[DEBUG Tiingo] API key (first 8 chars): {api_key[:8]}...")
    if not api_key:
        print("[DEBUG Tiingo] No API key found, returning None")
        return None
    
    url = TIINGO_BASE_URL + endpoint
    headers = {
        'Authorization': f'Token {api_key}',
        'Content-Type': 'application/json'
    }
    
    print(f"[DEBUG Tiingo] Making request to: {url}")
    print(f"[DEBUG Tiingo] Params: {params}")
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        print(f"[DEBUG Tiingo] Response status: {response.status_code}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG Tiingo] API request failed: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"[DEBUG Tiingo] Response text: {e.response.text[:200]}")
        return None

async def get_market_data(symbol: str, analysis_date: Optional[str] = None, period: str = "1y") -> ToolResult:
    """
    Get market data for a symbol from Tiingo.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
        analysis_date: Optional date for analysis (YYYY-MM-DD format)
        period: Period for historical data (default "1y")
    
    Returns:
        ToolResult with market data
    """
    print(f"[DEBUG Tiingo] Getting market data for {symbol}")
    try:
        tiingo_symbol = _convert_symbol_to_tiingo(symbol)
        
        # Determine date range
        end_date = datetime.now().date()
        if analysis_date:
            end_date = datetime.strptime(analysis_date, "%Y-%m-%d").date()
        
        # Calculate start date based on period
        if period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "6mo":
            start_date = end_date - timedelta(days=180)
        elif period == "3mo":
            start_date = end_date - timedelta(days=90)
        elif period == "1mo":
            start_date = end_date - timedelta(days=30)
        elif period == "1wk":
            start_date = end_date - timedelta(days=7)
        else:
            start_date = end_date - timedelta(days=365)  # default to 1 year
        
        # Format dates for API
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        # Build endpoint and parameters
        endpoint = TIINGO_DAILY_PRICES_ENDPOINT.format(ticker=tiingo_symbol)
        params = {
            'startDate': start_date_str,
            'endDate': end_date_str,
            'format': 'json'
        }
        
        data = _make_tiingo_request(endpoint, params)
        if data is None:
            return ToolResult(success=False, error=f"Failed to fetch data from Tiingo for {symbol}")
        
        if not data:
            return ToolResult(success=False, error=f"No data available for {symbol}")
        
        # Convert to DataFrame for processing
        df = pd.DataFrame(data)
        if df.empty:
            return ToolResult(success=False, error=f"No data available for {symbol}")
        
        # Parse dates and set as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # Filter by analysis_date if provided
        if analysis_date:
            ad = datetime.strptime(analysis_date, "%Y-%m-%d").date()
            df = df[df.index.date <= ad]
            if df.empty:
                return ToolResult(success=False, error=f"No data available for {symbol} on or before {analysis_date}")
        else:
            # Use most recent data
            end_dt = df.index.max().date()
            start_dt = end_dt - timedelta(days=365)
            df = df[df.index.date >= start_dt]
            if df.empty:
                return ToolResult(success=False, error=f"No recent data available for {symbol}")
        
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
        # Ensure numeric columns
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
        return ToolResult(success=False, error=f"Error getting market data for {symbol} from Tiingo: {str(e)}")

async def get_company_info(symbol: str) -> ToolResult:
    """
    Get company information for a symbol from Tiingo.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
    
    Returns:
        ToolResult with company information
    """
    print(f"[DEBUG Tiingo] Getting company info for {symbol}")
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

# Add rate limiting if needed
async def _apply_rate_limiting():
    """Apply rate limiting for Tiingo API calls."""
    # Tiingo Starter tier: 50 requests per hour, ~0.83 requests per minute
    # Use 5 seconds delay to stay well within limits
    await asyncio.sleep(5)