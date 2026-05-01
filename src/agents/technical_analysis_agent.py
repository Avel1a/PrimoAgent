from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
from ..workflows.state import AgentState
from ..tools.technical_indicators_tool import calculate_technical_indicators
from ..tools.yfinance_tool import fetch_daily_dataframe


async def analyze_technical(symbol: str, analysis_date: Optional[str] = None, market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Calculate technical indicators for a symbol up to a specific date.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        analysis_date: Date for analysis in YYYY-MM-DD format (optional)
        market_data: Optional market data from previous agent
        
    Returns:
        Dict with technical indicators or error info
    """
    try:
        symbol = symbol.upper()
        df = fetch_daily_dataframe(symbol, outputsize="compact")
        if df is None or df.empty:
            return {
                'symbol': symbol,
                'success': False,
                'error': 'No historical data available'
            }
        if analysis_date:
            target_date = datetime.strptime(analysis_date, "%Y-%m-%d")
            start_date = target_date - timedelta(days=60)
            hist_data = df[(df.index.date >= start_date.date()) & (df.index.date <= target_date.date())]
        else:
            end_date = df.index.max().date()
            start_date = end_date - timedelta(days=180)
            hist_data = df[df.index.date >= start_date]
        
        if hist_data.empty:
            return {
                'symbol': symbol,
                'success': False,
                'error': 'No historical data available'
            }
        
        # Simple date filtering - keep only data up to analysis_date
        if analysis_date and not hist_data.empty:
            # Filter by converting both to date strings for comparison
            mask = [str(date.date()) <= analysis_date for date in hist_data.index]
            hist_data = hist_data[mask]
            
            if hist_data.empty:
                return {
                    'symbol': symbol,
                    'success': False,
                    'error': f'No historical data available up to {analysis_date}'
                }
            
            # Check if we have enough data for technical indicators
            if len(hist_data) < 20:  # Minimum required for most indicators
                return {
                    'symbol': symbol,
                    'success': False,
                    'error': f'Insufficient historical data for {analysis_date}. Need at least 20 days, got {len(hist_data)} days'
                }
                
            print(f"Historical data for {symbol} up to {analysis_date}: {len(hist_data)} days")
        
        # Calculate technical indicators using dedicated tool
        indicators = ['SMA', 'EMA', 'RSI', 'MACD', 'BBANDS', 'ADX', 'CCI']
        # Ensure hist_data is DataFrame type for linter
        hist_data_df = pd.DataFrame(hist_data) if not isinstance(hist_data, pd.DataFrame) else hist_data
        result = await calculate_technical_indicators(hist_data_df, indicators, symbol, analysis_date)
        
        if not result.success:
            return {
                'symbol': symbol,
                'success': False,
                'error': result.error or 'Technical analysis failed'
            }
        
        # Get current price from market data or last close price from filtered data
        current_price = 0.0
        if market_data and isinstance(market_data, dict) and 'current_price' in market_data:
            current_price = market_data['current_price']
        elif not hist_data_df.empty:
            current_price = float(hist_data_df['Close'].iloc[-1])
        
        # Add current price to indicators
        indicators_data = result.data if result.data else {}
        indicators_data['current_price'] = current_price
        
        return {
            'symbol': symbol,
            'indicators': indicators_data,
            'success': True
        }
        
    except Exception as e:
        print(f"Error in technical analysis for {symbol}: {e}")
        return {
            'symbol': symbol,
            'success': False,
            'error': str(e)
        }


async def technical_analysis_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node for technical analysis.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with technical analysis results
    """
    try:
        # Get symbol, analysis_date and market data from previous agent
        symbol = state['symbols'][0] if state['symbols'] else 'AAPL'
        analysis_date = state['analysis_date']
        data_collection_results = state.get('data_collection_results')
        market_data = data_collection_results.get('market_data') if data_collection_results else None
        
        # Perform technical analysis with analysis_date
        result = await analyze_technical(symbol, analysis_date, market_data)
        
        # Update state
        state['technical_analysis_results'] = result
        state['current_step'] = 'technical_analysis_complete'
        
        if not result['success']:
            state['error'] = result.get('error', 'Technical analysis failed')
            
        return state
        
    except Exception as e:
        print(f"Technical analysis node error: {e}")
        state['error'] = str(e)
        state['current_step'] = 'error'
        return state 
