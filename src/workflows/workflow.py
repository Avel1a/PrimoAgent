import asyncio
from copy import deepcopy
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from .state import AgentState, create_initial_state
from ..agents.data_collection_agent import data_collection_agent_node
from ..agents.technical_analysis_agent import technical_analysis_agent_node
from ..agents.news_intelligence_agent import news_intelligence_agent_node
from ..agents.portfolio_manager_agent import portfolio_manager_agent_node
from ..agents.risk_manager_agent import risk_manager_agent_node


def debug_state(state: AgentState, agent_name: str) -> AgentState:
    """Debug function to log state after each agent."""
    print(f"\n{agent_name} Agent Complete:")

    analysis_date = state.get('analysis_date', 'N/A')
    symbol = state['symbols'][0] if state.get('symbols') else 'N/A'
    print(f"Date: {analysis_date} | Symbol: {symbol}")

    data_results = state.get('data_collection_results')
    if data_results and agent_name == "Data Collection":
        market_data = data_results.get('market_data') or {}
        current_price = market_data.get('current_price', 'N/A')
        print(f"Current Price: ${current_price}")

    tech_results = state.get('technical_analysis_results')
    if tech_results and agent_name == "Parallel Analysis":
        success = tech_results.get('success', False)
        print(f"Technical Success: {success}")

    news_results = state.get('news_intelligence_results')
    if news_results and agent_name == "Parallel Analysis":
        success = news_results.get('success', False)
        print(f"News Success: {success}")

    portfolio_results = state.get('portfolio_manager_results')
    if portfolio_results and agent_name == "Portfolio Manager":
        symbol_data = portfolio_results.get(symbol, {})
        if symbol_data and symbol_data.get('success'):
            signal = symbol_data.get('trading_signal', 'N/A')
            confidence = symbol_data.get('confidence_level', 'N/A')
            print(f"Signal: {signal} | Confidence: {confidence}")

    if state.get('error'):
        print(f"Error: {state.get('error')}")

    return state


async def debug_data_collection_node(state: AgentState) -> AgentState:
    result = await data_collection_agent_node(state)
    return debug_state(result, "Data Collection")


async def debug_parallel_analysis_node(state: AgentState) -> AgentState:
    """Run technical analysis and news intelligence in parallel via asyncio."""
    state_tech = deepcopy(state)
    state_news = deepcopy(state)

    tech_task = asyncio.create_task(technical_analysis_agent_node(state_tech))
    news_task = asyncio.create_task(news_intelligence_agent_node(state_news))

    tech_result, news_result = await asyncio.gather(tech_task, news_task)

    # Merge results back into the original state
    state['technical_analysis_results'] = tech_result.get('technical_analysis_results')
    state['news_intelligence_results'] = news_result.get('news_intelligence_results')

    # Only propagate technical errors — news failures are non-fatal
    # (Finnhub free tier doesn't cover historical news, but PM/RM can still work)
    if tech_result.get('error') and not state.get('error'):
        state['error'] = tech_result['error']

    # Always advance to portfolio manager even when news is unavailable
    if not state.get('error'):
        state['current_step'] = 'analysis_complete'

    return debug_state(state, "Parallel Analysis")


async def debug_portfolio_manager_node(state: AgentState) -> AgentState:
    result = await portfolio_manager_agent_node(state)
    return debug_state(result, "Portfolio Manager")


async def debug_risk_manager_node(state: AgentState) -> AgentState:
    result = await risk_manager_agent_node(state)
    risk_results = result.get("risk_manager_results", {})
    if risk_results:
        action = risk_results.get("action", "unknown")
        reason = risk_results.get("reason", "")
        print(f"Risk Manager: {action} — {reason}")
    return debug_state(result, "Risk Manager")


def create_workflow() -> StateGraph:
    """
    Create LangGraph workflow with parallel technical + news analysis.

    Flow:
        data_collection → [technical_analysis ∥ news_intelligence] → portfolio_manager → risk_manager → END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("data_collection", debug_data_collection_node)
    workflow.add_node("parallel_analysis", debug_parallel_analysis_node)
    workflow.add_node("portfolio_manager", debug_portfolio_manager_node)
    workflow.add_node("risk_manager", debug_risk_manager_node)

    workflow.set_entry_point("data_collection")

    workflow.add_conditional_edges("data_collection", should_continue, {
        "parallel_analysis": "parallel_analysis",
        END: END,
    })
    workflow.add_conditional_edges("parallel_analysis", should_continue, {
        "portfolio_manager": "portfolio_manager",
        END: END,
    })
    workflow.add_conditional_edges("portfolio_manager", should_continue, {
        "risk_manager": "risk_manager",
        END: END,
    })
    workflow.add_conditional_edges("risk_manager", should_continue, {
        END: END,
    })

    return workflow


async def run_analysis(symbols: list[str], session_id: str = "default", analysis_date: Optional[str] = None, cached_company_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run complete analysis workflow for symbols.

    Args:
        symbols: List of stock symbols to analyze
        session_id: Session identifier
        analysis_date: Date for analysis in YYYY-MM-DD format (optional, defaults to today)
        cached_company_info: Pre-fetched company info reused across days

    Returns:
        Dict with analysis results
    """
    try:
        workflow = create_workflow()
        app = workflow.compile()

        initial_state = create_initial_state(session_id, symbols, analysis_date)
        if cached_company_info:
            initial_state['_cached_company_info'] = cached_company_info

        result = await app.ainvoke(initial_state)

        return {
            'success': True,
            'session_id': session_id,
            'symbols': symbols,
            'analysis_date': analysis_date,
            'results': {
                'data_collection': result.get('data_collection_results'),
                'technical_analysis': result.get('technical_analysis_results'),
                'news_intelligence': result.get('news_intelligence_results'),
                'portfolio_manager': result.get('portfolio_manager_results'),
                'risk_manager': result.get('risk_manager_results'),
            },
            'final_step': result.get('current_step'),
            '_cached_company_info': result.get('_cached_company_info'),
            'error': result.get('error')
        }

    except Exception as e:
        print(f"Workflow error: {e}")
        return {
            'success': False,
            'error': str(e),
            'symbols': symbols,
            'session_id': session_id,
            'analysis_date': analysis_date
        }


def should_continue(state: AgentState) -> str:
    """Conditional routing: errors skip to END."""
    if state.get('error'):
        return END

    current_step = state.get('current_step', '')

    if current_step == 'data_collection_complete':
        return 'parallel_analysis'
    elif current_step == 'analysis_complete':
        return 'portfolio_manager'
    elif current_step == 'portfolio_management_complete':
        return 'risk_manager'
    elif current_step == 'risk_management_complete':
        return END
    else:
        return END
