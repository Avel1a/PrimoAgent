from typing import Dict, Any, Optional
import numpy as np
import math
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


def _compute_rolling_drawdown(closes: pd.Series, window: int = 60) -> tuple:
    """Rolling drawdown from the peak within the last `window` days.

    Returns (drawdown_pct, score) where score in [0, 1]: 1 = no drawdown, 0 = 30%+ drawdown.
    """
    if len(closes) < 2:
        return 0.0, 1.0
    recent = closes.tail(window) if len(closes) >= window else closes
    peak = recent.max()
    current = recent.iloc[-1]
    if peak <= 0:
        return 0.0, 1.0
    dd_pct = (peak - current) / peak * 100
    score = 1.0 - min(dd_pct / 30.0, 1.0)
    return dd_pct, score


def _compute_trend_score(closes: pd.Series) -> float:
    """Compute trend strength score in [-1, 1] from SMA slope, MACD direction, and ADX.

    Requires at least 40 bars. Falls back to 0.0 if insufficient data.
    Uses close-to-close ranges as ADX proxy since we only have close prices.
    """
    n = len(closes)
    if n < 40:
        return 0.0

    # --- SMA20 slope ---
    sma20 = closes.rolling(window=20).mean()
    sma_recent = sma20.iloc[-1]
    sma_past = sma20.iloc[-20]
    if sma_past > 0:
        slope = (sma_recent - sma_past) / sma_past
    else:
        slope = 0.0
    slope_score = math.tanh(slope * 50)

    # --- MACD direction ---
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    hist_val = macd_hist.iloc[-1]
    macd_score = 1.0 if hist_val > 0 else -1.0
    macd_score *= min(abs(hist_val) / 2.0, 1.0)

    # --- ADX (close-to-close approximation) ---
    high_est = closes.rolling(window=2).max()
    low_est = closes.rolling(window=2).min()
    prev_closes = closes.shift(1)
    tr_approx = pd.concat([
        high_est - low_est,
        (high_est - prev_closes).abs(),
        (low_est - prev_closes).abs()
    ], axis=1).max(axis=1)
    atr14 = tr_approx.ewm(span=14, adjust=False).mean()
    up_move = closes.diff().clip(lower=0)
    down_move = (-closes.diff()).clip(lower=0)
    atr_val = atr14.iloc[-1]
    if atr_val > 0:
        di_plus = up_move.ewm(span=14, adjust=False).mean().iloc[-1] / atr_val * 100
        di_minus = down_move.ewm(span=14, adjust=False).mean().iloc[-1] / atr_val * 100
        denom = di_plus + di_minus
        if denom > 0:
            dx = abs(di_plus - di_minus) / denom * 100
        else:
            dx = 0.0
    else:
        dx = 0.0
    # Smooth DX into ADX (simple EMA of DX values... we only have one value, use as-is)
    adx_val = dx
    adx_score = min(adx_val / 50.0, 1.0)
    if slope < 0:
        adx_score = -adx_score

    return (slope_score + macd_score + adx_score) / 3.0


def _compute_volatility_score(closes: pd.Series, atr_period: int = 14, lookback: int = 60) -> float:
    """Compute volatility state score in [-1, 1]. High vol -> negative, low vol -> positive.

    Uses ATR ratio: current ATR / mean ATR over lookback period.
    Close-to-close range used as True Range proxy.
    """
    n = len(closes)
    if n < atr_period + 2:
        return 0.0

    high_est = closes.rolling(window=2).max()
    low_est = closes.rolling(window=2).min()
    prev_closes = closes.shift(1)
    tr = pd.concat([
        high_est - low_est,
        (high_est - prev_closes).abs(),
        (low_est - prev_closes).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False).mean()

    if len(atr.dropna()) < lookback:
        return 0.0

    current_atr = atr.iloc[-1]
    hist_atr_mean = atr.tail(lookback).mean()
    if hist_atr_mean <= 0:
        return 0.0

    atr_ratio = current_atr / hist_atr_mean
    score = 1.0 - (atr_ratio - 1.0)
    return max(-1.0, min(1.0, score))


def _compute_regime_score(closes: pd.Series) -> dict:
    """Compute regime score and return regime metadata.

    Returns dict with keys: regime_score, regime, rolling_dd_pct, trend_score, vol_score.
    Falls back to Neutral if insufficient data or no regime config.
    """
    regime_cfg = config.risk_regime
    if not regime_cfg:
        return {
            "regime_score": 0.0, "regime": "neutral",
            "rolling_dd_pct": 0.0, "trend_score": 0.0, "vol_score": 0.0
        }

    window = regime_cfg.get("rolling_drawdown_window", 60)
    atr_period = regime_cfg.get("volatility_atr_period", 14)
    vol_lookback = regime_cfg.get("volatility_lookback", 60)
    weights = regime_cfg.get("weights", {"trend": 0.4, "rolling_dd": 0.3, "volatility": 0.3})
    bull_th = regime_cfg.get("bull_threshold", 0.3)
    bear_th = regime_cfg.get("bear_threshold", -0.3)

    if len(closes) < max(window, 40):
        return {
            "regime_score": 0.0, "regime": "neutral",
            "rolling_dd_pct": 0.0, "trend_score": 0.0, "vol_score": 0.0
        }

    dd_pct, dd_score = _compute_rolling_drawdown(closes, window)
    trend_score = _compute_trend_score(closes)
    vol_score = _compute_volatility_score(closes, atr_period, vol_lookback)

    regime_score = (
        weights["trend"] * trend_score
        + weights["rolling_dd"] * dd_score
        + weights["volatility"] * vol_score
    )

    if regime_score > bull_th:
        regime = "bull"
    elif regime_score < bear_th:
        regime = "bear"
    else:
        regime = "neutral"

    return {
        "regime_score": round(regime_score, 4),
        "regime": regime,
        "rolling_dd_pct": round(dd_pct, 2),
        "trend_score": round(trend_score, 4),
        "vol_score": round(vol_score, 4),
    }


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
