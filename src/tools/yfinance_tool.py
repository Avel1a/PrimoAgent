from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import os
import asyncio
import requests
import pandas as pd
from .utils import ToolResult

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

def _is_crypto_symbol(symbol: str) -> bool:
    s = symbol.upper()
    return s.endswith("USD") or "-" in s or s.endswith("USDT")

def _empty_ohlcv_df() -> pd.DataFrame:
    df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df.index = pd.DatetimeIndex([], name="Date")
    return df

def _parse_stock_timeseries(json_obj: Dict[str, Any]) -> pd.DataFrame:
    series_key = "Time Series (Daily)"
    if series_key not in json_obj:
        raise ValueError(json_obj.get("Note") or json_obj.get("Information") or json_obj.get("Error Message") or "Alpha Vantage response missing Time Series (Daily)")
    rows: List[Dict[str, Any]] = []
    for date_str, fields in json_obj[series_key].items():
        rows.append({
            "Date": date_str,
            "Open": float(fields.get("1. open", "nan")),
            "High": float(fields.get("2. high", "nan")),
            "Low": float(fields.get("3. low", "nan")),
            "Close": float(fields.get("4. close", "nan")),
            "Volume": float(fields.get("5. volume", "nan")),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df

def _parse_crypto_timeseries(json_obj: Dict[str, Any]) -> pd.DataFrame:
    series_key = "Time Series (Digital Currency Daily)"
    if series_key not in json_obj:
        raise ValueError(json_obj.get("Note") or json_obj.get("Information") or json_obj.get("Error Message") or "Alpha Vantage response missing Digital Currency Daily")
    rows: List[Dict[str, Any]] = []
    for date_str, fields in json_obj[series_key].items():
        def _getf(*keys):
            for k in keys:
                v = fields.get(k)
                if v is not None:
                    try:
                        return float(v)
                    except Exception:
                        continue
            return float("nan")
        rows.append({
            "Date": date_str,
            "Open": _getf("1a. open (USD)", "1b. open (USD)", "1. open"),
            "High": _getf("2a. high (USD)", "2b. high (USD)", "2. high"),
            "Low":  _getf("3a. low (USD)", "3b. low (USD)", "3. low"),
            "Close": _getf("4a. close (USD)", "4b. close (USD)", "4. close"),
            "Volume": _getf("5. volume", "5. Volume"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df

async def fetch_daily_dataframe(symbol: str, outputsize: str = "compact", market: str = "USD") -> pd.DataFrame:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ALPHA_VANTAGE_API_KEY in environment")
    is_crypto = _is_crypto_symbol(symbol)
    params = {}
    if is_crypto:
        base = symbol.upper().split("-")[0] if "-" in symbol else symbol.upper().replace("USD", "")
        params = {
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": base,
            "market": market,
            "apikey": api_key,
        }
    else:
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol.upper(),
            "outputsize": outputsize,
            "apikey": api_key,
        }
    resp = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
    try:
        data = resp.json()
    except Exception:
        print("Alpha Vantage Raw Response: <non-JSON payload>")
        return _empty_ohlcv_df()
    print("Alpha Vantage Raw Response:", data)
    await asyncio.sleep(5)
    if isinstance(data, dict) and (("Error Message" in data) or ("Information" in data) or ("Note" in data)):
        msg = data.get("Error Message") or data.get("Information") or data.get("Note")
        print("Alpha Vantage Error:", msg)
        return _empty_ohlcv_df()
    if is_crypto:
        try:
            return _parse_crypto_timeseries(data)
        except Exception as e:
            print("Alpha Vantage Parse Error (crypto):", str(e))
            return _empty_ohlcv_df()
    try:
        return _parse_stock_timeseries(data)
    except Exception as e:
        print("Alpha Vantage Parse Error (stock):", str(e))
        return _empty_ohlcv_df()

async def get_market_data(symbol: str, analysis_date: Optional[str] = None, period: str = "1y") -> ToolResult:
    try:
        df = await fetch_daily_dataframe(symbol, outputsize="compact")
        if df.empty:
            return ToolResult(success=False, error=f"No data available for {symbol}")
        if analysis_date:
            ad = datetime.strptime(analysis_date, "%Y-%m-%d").date()
            df = df[df.index.date <= ad]
            if df.empty:
                return ToolResult(success=False, error=f"No data available for {symbol} on or before {analysis_date}")
        else:
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
            previous_close = float(previous["Close"])
            current_close = float(latest["Close"])
            price_change = current_close - previous_close
            price_change_pct = (price_change / previous_close) * 100 if previous_close != 0 else 0
        historical_clean = df.reset_index()
        historical_clean["Date"] = historical_clean["Date"].dt.strftime("%Y-%m-%d")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in historical_clean.columns:
                historical_clean[col] = historical_clean[col].astype(float)
        historical_dict = historical_clean.to_dict("records")
        result_date = analysis_date if analysis_date else str(df.index[-1].date())
        result_data = {
            "symbol": symbol.upper(),
            "date": result_date,
            "current_price": float(latest["Close"]),
            "price_data": {
                "open": float(latest["Open"]),
                "high": float(latest["High"]),
                "low": float(latest["Low"]),
                "close": float(latest["Close"]),
                "volume": int(latest["Volume"]),
                "previous_close": previous_close,
                "price_change": price_change,
                "price_change_pct": price_change_pct,
            },
            "historical_data": historical_dict,
        }
        return ToolResult(success=True, data=result_data)
    except Exception as e:
        return ToolResult(success=False, error=f"Error getting market data for {symbol}: {str(e)}")

async def get_company_info(symbol: str) -> ToolResult:
    try:
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            return ToolResult(success=False, error="Missing ALPHA_VANTAGE_API_KEY in environment")
        params = {
            "function": "OVERVIEW",
            "symbol": symbol.upper(),
            "apikey": api_key,
        }
        resp = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
        try:
            info = resp.json()
        except Exception:
            print("Alpha Vantage Raw Response: <non-JSON payload>")
            return ToolResult(success=False, error=f"No company info available for {symbol}")
        print("Alpha Vantage Raw Response:", info)
        await asyncio.sleep(5)
        if (isinstance(info, dict) and (("Error Message" in info) or ("Information" in info) or ("Note" in info))):
            msg = info.get("Error Message") or info.get("Information") or info.get("Note")
            print("Alpha Vantage Error:", msg)
            return ToolResult(success=False, error=f"No company info available for {symbol}: {msg}")
        if not info or "Symbol" not in info:
            return ToolResult(success=False, error=f"No company info available for {symbol}")
        company_data = {
            "symbol": symbol.upper(),
            "name": info.get("Name", "N/A"),
            "sector": info.get("Sector", "N/A"),
            "industry": info.get("Industry", "N/A"),
            "country": info.get("Country", "N/A"),
            "exchange": info.get("Exchange", "N/A"),
            "market_cap": info.get("MarketCapitalization", "N/A"),
            "website": info.get("Website", "N/A"),
            "description": (info.get("Description") or "")[:500],
        }
        return ToolResult(success=True, data=company_data)
    except Exception as e:
        return ToolResult(success=False, error=f"Error getting company info for {symbol}: {str(e)}")
