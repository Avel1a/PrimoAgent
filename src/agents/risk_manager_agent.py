from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from ..workflows.state import AgentState
from ..config import config


def _daily_returns(close_prices: pd.Series) -> pd.Series:
    return close_prices.pct_change().dropna()


def _historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR: the return at the (1 - confidence) percentile."""
    if returns.empty:
        return 0.0
    return abs(float(np.percentile(returns, (1 - confidence) * 100)))


def _recent_drawdown(historical_data: list) -> float:
    """Current drawdown from the historical peak close."""
    if not historical_data:
        return 0.0
    closes = [float(d.get("close", 0)) for d in historical_data if d.get("close")]
    if not closes:
        return 0.0
    peak = max(closes)
    current = closes[-1]
    return (peak - current) / peak * 100 if peak > 0 else 0.0


async def risk_manager_agent_node(state: AgentState) -> AgentState:
    """
    Validate and adjust portfolio manager output against risk constraints.

    Constraints enforced:
    1. Max position size capped by VaR: position_value ≤ max_daily_var_pct * portfolio
    2. Max single-position allocation: position_size ≤ max_position_pct
    3. Cash reserve: at least min_cash_reserve_pct stays uninvested
    4. Stop-loss signal override: if open drawdown exceeds max_drawdown_pct, force HOLD
    """
    try:
        symbol = state["symbols"][0] if state["symbols"] else "UNKNOWN"
        pm_results = state.get("portfolio_manager_results", {})
        symbol_data = pm_results.get(symbol, {})

        if not symbol_data.get("success"):
            state["risk_manager_results"] = {
                "symbol": symbol,
                "action": "skip",
                "reason": "Portfolio manager did not produce a valid decision",
            }
            state["current_step"] = "risk_management_complete"
            return state

        signal = symbol_data.get("trading_signal", "HOLD")
        confidence = symbol_data.get("confidence_level", 0.5)
        position_size = symbol_data.get("position_size", 10)

        # --- Extract historical data for VaR ---
        dc_results = state.get("data_collection_results", {})
        market_data = dc_results.get("market_data", {}) or {}
        historical = market_data.get("historical_data", []) or []

        var_value = 0.0
        drawdown = 0.0

        if historical:
            closes = pd.Series([float(d["close"]) for d in historical if d.get("close")])
            if len(closes) >= 20:
                returns = _daily_returns(closes).tail(config.risk_var_lookback_days)
                var_value = _historical_var(returns, config.risk_var_confidence)
                drawdown = _recent_drawdown(historical)

        # --- Risk checks ---

        # 1. Drawdown guard: if portfolio drawdown exceeds limit, force HOLD
        if drawdown > config.risk_max_drawdown_pct:
            symbol_data["trading_signal"] = "HOLD"
            symbol_data["confidence_level"] = 0.0
            symbol_data["position_size"] = 0
            symbol_data["risk_adjusted"] = True
            symbol_data["risk_metrics"] = {
                "var_daily_pct": round(var_value * 100, 2),
                "drawdown_pct": round(drawdown, 2),
                "max_drawdown_limit_pct": config.risk_max_drawdown_pct,
                "override": "max_drawdown_breached",
            }
            state["portfolio_manager_results"][symbol] = symbol_data
            state["risk_manager_results"] = {
                "symbol": symbol,
                "action": "override_to_hold",
                "reason": f"Drawdown {drawdown:.1f}% exceeds limit {config.risk_max_drawdown_pct}%",
                "risk_metrics": symbol_data["risk_metrics"],
            }
            state["current_step"] = "risk_management_complete"
            return state

        # 2. VaR-based position capping
        if var_value > 0 and signal == "BUY":
            # Max position value = (max_daily_var_pct / var) * portfolio_pct
            # If daily VaR is 2% and max allowed is 2%, we allow full position.
            # If daily VaR is 4% and max allowed is 2%, we cap at 50% of intended.
            risk_ratio = config.risk_max_daily_var_pct / max(var_value * 100, 0.01)
            var_capped_size = int(position_size * min(risk_ratio, 1.0))
            # Round down to nearest 10
            var_capped_size = max(10, (var_capped_size // 10) * 10)
            position_size = min(position_size, var_capped_size)

        # 3. Max single-position cap
        position_size = min(position_size, int(config.risk_max_position_pct))

        # 4. Cash reserve: if position uses more than (100 - cash_reserve), cap it
        max_allocated = 100 - int(config.risk_min_cash_reserve_pct)
        position_size = min(position_size, max_allocated)

        # Round to nearest 10, floor at 10
        position_size = max(10, (position_size // 10) * 10)

        # --- Apply adjustments ---
        risk_metrics = {
            "var_daily_pct": round(var_value * 100, 2),
            "drawdown_pct": round(drawdown, 2),
            "max_position_pct_limit": config.risk_max_position_pct,
            "min_cash_reserve_pct": config.risk_min_cash_reserve_pct,
            "override": "none",
        }

        original_size = symbol_data.get("position_size", position_size)
        if position_size != original_size:
            symbol_data["position_size"] = position_size
            symbol_data["risk_adjusted"] = True
            risk_metrics["override"] = "position_reduced"
            risk_metrics["original_position_size"] = original_size
        else:
            symbol_data["risk_adjusted"] = False
            symbol_data["risk_metrics"] = risk_metrics

        state["portfolio_manager_results"][symbol] = symbol_data
        state["risk_manager_results"] = {
            "symbol": symbol,
            "action": "validated",
            "reason": f"Position size {position_size}% passes all risk checks",
            "risk_metrics": risk_metrics,
        }
        state["current_step"] = "risk_management_complete"
        return state

    except Exception as e:
        state["error"] = f"Risk manager failed: {e}"
        state["current_step"] = "error"
        return state
