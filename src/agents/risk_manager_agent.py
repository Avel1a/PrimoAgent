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


def _compute_rolling_drawdown(closes: pd.Series, window: int = 60) -> tuple[float, float]:
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


def _compute_true_range(closes: pd.Series) -> pd.Series:
    """Estimate true range from close prices only (close-to-close proxy)."""
    high_est = closes.rolling(window=2).max()
    low_est = closes.rolling(window=2).min()
    prev = closes.shift(1)
    return pd.concat([
        high_est - low_est,
        (high_est - prev).abs(),
        (low_est - prev).abs()
    ], axis=1).max(axis=1)


def _compute_atr(closes: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range via EMA smoothing of true range estimates."""
    tr = _compute_true_range(closes)
    # Drop NaN from first row before EWM to avoid propagation issues
    return tr.iloc[1:].ewm(span=period, adjust=False).mean()


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
    atr14 = _compute_atr(closes, 14)
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

    atr = _compute_atr(closes, atr_period)

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

    min_bars = max(window, 40, atr_period + 2)
    if len(closes) < min_bars:
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
    regime_score = max(-1.0, min(1.0, regime_score))

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
    """Validate and adjust portfolio manager output against adaptive risk constraints.

    Detection order:
    1. Compute regime (Bear/Neutral/Bull) from multi-signal detector
    2. Get regime-specific risk parameters
    3. Drawdown guard: force HOLD if rolling drawdown exceeds regime threshold
    4. VaR-based position capping with regime multiplier
    5. Max single-position cap (regime-specific)
    6. Cash reserve (regime-specific)
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
        position_size = symbol_data.get("position_size", 10)

        # --- Extract historical data ---
        dc_results = state.get("data_collection_results", {})
        market_data = dc_results.get("market_data", {}) or {}
        historical = market_data.get("historical_data", []) or []

        # --- Compute regime ---
        regime_info = {
            "regime_score": 0.0, "regime": "neutral",
            "rolling_dd_pct": 0.0, "trend_score": 0.0, "vol_score": 0.0,
        }
        closes_full = pd.Series(dtype=float)
        if historical:
            closes_full = pd.Series(
                [float(d["close"]) for d in historical if d.get("close")]
            )
            if len(closes_full) >= 60:
                regime_info = _compute_regime_score(closes_full)

        regime = regime_info["regime"]
        regime_cfg = config.risk_regime
        params = regime_cfg.get(regime, regime_cfg.get("neutral", {}))
        max_dd = params.get("max_drawdown_pct", 20)
        pos_min = int(params.get("position_min", 10))
        pos_max = int(params.get("position_max", 100))
        var_mult = params.get("var_multiplier", 1.0)
        cash_reserve = int(params.get("cash_reserve_pct", 10))
        rolling_dd = regime_info["rolling_dd_pct"]

        # --- Compute VaR (unchanged methodology) ---
        var_value = 0.0
        if len(closes_full) >= 20:
            returns = _daily_returns(closes_full).tail(config.risk_var_lookback_days)
            var_value = _historical_var(returns, config.risk_var_confidence)

        # --- 1. Drawdown guard (regime-aware) ---
        force_hold = False
        override_reason = ""
        if regime == "bear" and rolling_dd > max_dd:
            force_hold = True
            override_reason = (
                f"Bear regime: drawdown {rolling_dd:.1f}% > {max_dd}%"
            )
        elif regime == "neutral" and rolling_dd > max_dd and regime_info["trend_score"] < 0:
            force_hold = True
            override_reason = (
                f"Neutral regime: drawdown {rolling_dd:.1f}% > {max_dd}% "
                f"and trend {regime_info['trend_score']:.2f} < 0"
            )
        elif regime == "bull":
            if rolling_dd > max_dd and regime_info["trend_score"] < 0:
                force_hold = True
                override_reason = (
                    f"Bull regime: drawdown {rolling_dd:.1f}% > {max_dd}% "
                    f"and trend {regime_info['trend_score']:.2f} < 0"
                )

        if force_hold:
            symbol_data["trading_signal"] = "HOLD"
            symbol_data["confidence_level"] = 0.0
            symbol_data["position_size"] = 0
            symbol_data["risk_adjusted"] = True
            symbol_data["risk_metrics"] = {
                "var_daily_pct": round(var_value * 100, 2),
                "drawdown_pct": round(rolling_dd, 2),
                "max_drawdown_limit_pct": max_dd,
                "regime": regime,
                "regime_score": regime_info["regime_score"],
                "override": "max_drawdown_breached",
            }
            state["portfolio_manager_results"][symbol] = symbol_data
            state["risk_manager_results"] = {
                "symbol": symbol,
                "action": "override_to_hold",
                "reason": override_reason,
                "risk_metrics": symbol_data["risk_metrics"],
            }
            state["current_step"] = "risk_management_complete"
            return state

        # --- 2. VaR-based position capping with regime multiplier ---
        if var_value > 0 and signal == "BUY":
            effective_var_limit = config.risk_max_daily_var_pct * var_mult
            risk_ratio = effective_var_limit / max(var_value * 100, 0.01)
            var_capped_size = int(position_size * min(risk_ratio, 1.0))
            var_capped_size = max(10, (var_capped_size // 10) * 10)
            position_size = min(position_size, var_capped_size)

        # --- 3. Regime-scaled position cap ---
        # Map regime_score to [pos_min, pos_max] linearly within the regime's range
        regime_score = regime_info["regime_score"]
        bear_th = regime_cfg.get("bear_threshold", -0.3)
        bull_th = regime_cfg.get("bull_threshold", 0.3)
        if regime == "bull":
            t = (regime_score - bull_th) / (1.0 - bull_th)
            regime_cap = round(pos_min + t * (pos_max - pos_min))
        elif regime == "bear":
            t = (regime_score - (-1.0)) / (bear_th - (-1.0))
            regime_cap = round(pos_min + t * (pos_max - pos_min))
        else:  # neutral
            t = (regime_score - bear_th) / (bull_th - bear_th)
            regime_cap = round(pos_min + t * (pos_max - pos_min))
        regime_cap = max(pos_min, min(pos_max, (regime_cap // 10) * 10))
        position_size = min(position_size, regime_cap)

        # --- 4. Cash reserve (regime-specific) ---
        max_allocated = 100 - cash_reserve
        position_size = min(position_size, max_allocated)

        # Round to nearest 10, floor at 10
        position_size = max(10, (position_size // 10) * 10)

        # --- Apply adjustments ---
        risk_metrics = {
            "var_daily_pct": round(var_value * 100, 2),
            "drawdown_pct": round(rolling_dd, 2),
            "regime_position_range": f"{pos_min}-{pos_max}",
            "regime_position_cap": regime_cap,
            "min_cash_reserve_pct": cash_reserve,
            "regime": regime,
            "regime_score": regime_info["regime_score"],
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
            "reason": (
                f"Position size {position_size}% passes all risk checks "
                f"(regime: {regime})"
            ),
            "risk_metrics": risk_metrics,
        }
        state["current_step"] = "risk_management_complete"
        return state

    except Exception as e:
        state["error"] = f"Risk manager failed: {e}"
        state["current_step"] = "error"
        return state
