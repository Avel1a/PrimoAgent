# Adaptive Risk Regime Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed-parameter drawdown guard with an adaptive multi-signal Risk Regime Detector that dynamically adjusts risk parameters based on rolling drawdown, trend strength, and volatility state.

**Architecture:** Three new signal-computation functions feed into a single `_compute_regime_score()` that outputs a score in [-1, 1]. This score maps to Bear/Neutral/Bull regimes with distinct risk parameters from config. The existing risk check pipeline (drawdown guard → VaR cap → position cap → cash reserve) is preserved but each step reads regime-specific thresholds.

**Tech Stack:** Python 3.12, pandas, numpy (already in project)

**Files:**
- Modify: `src/config/config.json` — add `risk.regime` section
- Modify: `src/config/config.py` — add `risk_regime` property
- Modify: `src/agents/risk_manager_agent.py` — core regime detection + refactored risk checks
- Modify: `src/tools/daily_csv_tool.py` — add `regime` and `regime_score` columns

---

### Task 1: Add regime configuration to config.json

**Files:**
- Modify: `src/config/config.json`

- [ ] **Step 1: Add `regime` section under `risk`**

Modify `src/config/config.json` — add the `"regime"` key inside the existing `"risk"` object, after `"min_cash_reserve_pct"`:

```json
{
  "risk": {
    "max_position_pct": 25,
    "stop_loss_pct": 5,
    "take_profit_pct": 15,
    "max_drawdown_pct": 20,
    "max_daily_var_pct": 2.0,
    "var_confidence": 0.95,
    "var_lookback_days": 252,
    "min_cash_reserve_pct": 10,
    "regime": {
      "rolling_drawdown_window": 60,
      "trend_sma_period": 20,
      "volatility_atr_period": 14,
      "volatility_lookback": 60,
      "weights": {
        "trend": 0.4,
        "rolling_dd": 0.3,
        "volatility": 0.3
      },
      "bull_threshold": 0.3,
      "bear_threshold": -0.3,
      "bear": {
        "max_drawdown_pct": 12,
        "max_position_pct": 10,
        "var_multiplier": 0.5,
        "cash_reserve_pct": 20
      },
      "neutral": {
        "max_drawdown_pct": 20,
        "max_position_pct": 25,
        "var_multiplier": 1.0,
        "cash_reserve_pct": 10
      },
      "bull": {
        "max_drawdown_pct": 30,
        "max_position_pct": 30,
        "var_multiplier": 1.5,
        "cash_reserve_pct": 5
      }
    }
  }
}
```

- [ ] **Step 2: Verify JSON is valid**

Run: `python -c "import json; json.load(open('src/config/config.json')); print('Valid JSON')"`

Expected: `Valid JSON`

- [ ] **Step 3: Commit**

```bash
git add src/config/config.json
git commit -m "feat: add risk regime configuration section to config.json"
```

---

### Task 2: Add regime config property to config.py

**Files:**
- Modify: `src/config/config.py`

- [ ] **Step 1: Add `risk_regime` property**

Add after the `risk_min_cash_reserve_pct` property (line 147):

```python
    @property
    def risk_regime(self) -> dict:
        """Get risk regime configuration for adaptive risk management."""
        risk = self._config_data.get('risk', {})
        return risk.get('regime', {})
```

- [ ] **Step 2: Verify the property resolves correctly**

Run: `python -c "from src.config.config import config; r = config.risk_regime; print('weights:', r.get('weights')); print('bull_threshold:', r.get('bull_threshold')); print('bear:', r['bear']['max_drawdown_pct'])"`

Expected output:
```
weights: {'trend': 0.4, 'rolling_dd': 0.3, 'volatility': 0.3}
bull_threshold: 0.3
bear: 12
```

- [ ] **Step 3: Commit**

```bash
git add src/config/config.py
git commit -m "feat: add risk_regime config property"
```

---

### Task 3: Implement signal computation functions in risk_manager_agent.py

**Files:**
- Modify: `src/agents/risk_manager_agent.py`

- [ ] **Step 1: Add import for math**

At the top of the file, the existing imports are:
```python
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from ..workflows.state import AgentState
from ..config import config
```

Add `import math` after `import numpy as np`:
```python
from typing import Dict, Any, Optional
import numpy as np
import math
import pandas as pd
from ..workflows.state import AgentState
from ..config import config
```

- [ ] **Step 2: Add `_compute_rolling_drawdown` function**

Replace the existing `_recent_drawdown` function (lines 19-28) with:

```python
def _compute_rolling_drawdown(closes: pd.Series, window: int = 60) -> tuple[float, float]:
    """Rolling drawdown from the peak within the last `window` days.

    Returns (drawdown_pct, score) where score ∈ [0, 1]: 1 = no drawdown, 0 = 30%+ drawdown.
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
    """Compute trend strength score ∈ [-1, 1] from SMA slope, MACD direction, and ADX.

    Requires at least 40 bars.
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

    # --- ADX ---
    tr_list = []
    dm_plus_list = []
    dm_minus_list = []
    highs = pd.Series()  # We'll compute TR/ADX from closes only as approximation
    # Simple ADX approximation from closes: use close-to-close range as TR proxy
    high_est = closes.rolling(window=2).max()
    low_est = closes.rolling(window=2).min()
    prev_closes = closes.shift(1)
    tr_approx = pd.concat([high_est - low_est, (high_est - prev_closes).abs(), (low_est - prev_closes).abs()], axis=1).max(axis=1)
    atr14 = tr_approx.ewm(span=14, adjust=False).mean()
    # Directional movement: use close changes as proxy
    up_move = closes.diff().clip(lower=0)
    down_move = (-closes.diff()).clip(lower=0)
    atr_val = atr14.iloc[-1]
    if atr_val > 0:
        di_plus = up_move.ewm(span=14, adjust=False).mean().iloc[-1] / atr_val * 100
        di_minus = down_move.ewm(span=14, adjust=False).mean().iloc[-1] / atr_val * 100
        dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100 if (di_plus + di_minus) > 0 else 0
        adx_val = pd.Series(dx).ewm(span=14, adjust=False).mean().iloc[-1] if isinstance(dx, (int, float)) else 0
    else:
        adx_val = 0.0

    adx_score = min(adx_val / 50.0, 1.0)
    if slope < 0:
        adx_score = -adx_score

    return (slope_score + macd_score + adx_score) / 3.0


def _compute_volatility_score(closes: pd.Series, atr_period: int = 14, lookback: int = 60) -> float:
    """Compute volatility state score ∈ [-1, 1]. High vol → negative, low vol → positive.

    Uses average true range (ATR) ratio: current ATR / mean ATR over lookback.
    """
    n = len(closes)
    if n < atr_period + 2:
        return 0.0

    # True Range from closes only (close-to-close proxy)
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
    return max(-1.0, min(1.0, score))  # clamp to [-1, 1]
```

- [ ] **Step 3: Add `_compute_regime_score` function**

Add after the signal functions:

```python
def _compute_regime_score(closes: pd.Series) -> dict:
    """Compute regime score and return regime metadata.

    Returns dict with keys: regime_score, regime, rolling_dd_pct, trend_score, vol_score.
    Falls back to Neutral if insufficient data.
    """
    regime_cfg = config.risk_regime
    if not regime_cfg:
        return {"regime_score": 0.0, "regime": "neutral", "rolling_dd_pct": 0.0,
                "trend_score": 0.0, "vol_score": 0.0}

    window = regime_cfg.get("rolling_drawdown_window", 60)
    atr_period = regime_cfg.get("volatility_atr_period", 14)
    vol_lookback = regime_cfg.get("volatility_lookback", 60)
    weights = regime_cfg.get("weights", {"trend": 0.4, "rolling_dd": 0.3, "volatility": 0.3})
    bull_th = regime_cfg.get("bull_threshold", 0.3)
    bear_th = regime_cfg.get("bear_threshold", -0.3)

    if len(closes) < max(window, 40):
        return {"regime_score": 0.0, "regime": "neutral", "rolling_dd_pct": 0.0,
                "trend_score": 0.0, "vol_score": 0.0}

    _, dd_score = _compute_rolling_drawdown(closes, window)
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

    # Recompute rolling dd for the drawdown guard value
    dd_pct, _ = _compute_rolling_drawdown(closes, window)

    return {
        "regime_score": round(regime_score, 4),
        "regime": regime,
        "rolling_dd_pct": round(dd_pct, 2),
        "trend_score": round(trend_score, 4),
        "vol_score": round(vol_score, 4),
    }
```

- [ ] **Step 4: Verify signal functions are importable**

Run: `python -c "from src.agents.risk_manager_agent import _compute_rolling_drawdown, _compute_trend_score, _compute_volatility_score, _compute_regime_score; print('All imported')"`

Expected: `All imported`

- [ ] **Step 5: Commit**

```bash
git add src/agents/risk_manager_agent.py
git commit -m "feat: add regime detection signal functions to risk manager"
```

---

### Task 4: Refactor risk_manager_agent_node to use regime detection

**Files:**
- Modify: `src/agents/risk_manager_agent.py`

- [ ] **Step 1: Rewrite `risk_manager_agent_node`**

Replace the entire `risk_manager_agent_node` function (lines 31-151) with:

```python
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
        regime_info = {"regime_score": 0.0, "regime": "neutral",
                       "rolling_dd_pct": 0.0, "trend_score": 0.0, "vol_score": 0.0}
        if historical:
            closes = pd.Series([float(d["close"]) for d in historical if d.get("close")])
            if len(closes) >= 60:
                regime_info = _compute_regime_score(closes)

        regime = regime_info["regime"]
        regime_cfg = config.risk_regime
        params = regime_cfg.get(regime, regime_cfg.get("neutral", {}))
        max_dd = params.get("max_drawdown_pct", 20)
        max_pos = int(params.get("max_position_pct", 25))
        var_mult = params.get("var_multiplier", 1.0)
        cash_reserve = int(params.get("cash_reserve_pct", 10))
        rolling_dd = regime_info["rolling_dd_pct"]

        # --- Compute VaR (unchanged methodology) ---
        var_value = 0.0
        if historical:
            closes_full = pd.Series([float(d["close"]) for d in historical if d.get("close")])
            if len(closes_full) >= 20:
                returns = _daily_returns(closes_full).tail(config.risk_var_lookback_days)
                var_value = _historical_var(returns, config.risk_var_confidence)

        # --- 1. Drawdown guard (regime-aware) ---
        force_hold = False
        override_reason = ""
        if regime == "bear" and rolling_dd > max_dd:
            force_hold = True
            override_reason = f"Bear regime: drawdown {rolling_dd:.1f}% > {max_dd}%"
        elif regime == "neutral" and rolling_dd > max_dd:
            force_hold = True
            override_reason = f"Neutral regime: drawdown {rolling_dd:.1f}% > {max_dd}%"
        elif regime == "bull":
            if rolling_dd > max_dd and regime_info["trend_score"] < 0:
                force_hold = True
                override_reason = f"Bull regime: drawdown {rolling_dd:.1f}% > {max_dd}% and trend {regime_info['trend_score']:.2f} < 0"

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

        # --- 3. Max single-position cap (regime-specific) ---
        position_size = min(position_size, max_pos)

        # --- 4. Cash reserve (regime-specific) ---
        max_allocated = 100 - cash_reserve
        position_size = min(position_size, max_allocated)

        # Round to nearest 10, floor at 10
        position_size = max(10, (position_size // 10) * 10)

        # --- Apply adjustments ---
        risk_metrics = {
            "var_daily_pct": round(var_value * 100, 2),
            "drawdown_pct": round(rolling_dd, 2),
            "max_position_pct_limit": max_pos,
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
            "reason": f"Position size {position_size}% passes all risk checks (regime: {regime})",
            "risk_metrics": risk_metrics,
        }
        state["current_step"] = "risk_management_complete"
        return state

    except Exception as e:
        state["error"] = f"Risk manager failed: {e}"
        state["current_step"] = "error"
        return state
```

Note: The old `_recent_drawdown` function is already replaced in Task 3. Remove it completely if not already removed.

- [ ] **Step 2: Verify the module loads without syntax errors**

Run: `python -c "from src.agents.risk_manager_agent import risk_manager_agent_node; print('Loaded OK')"`

Expected: `Loaded OK`

- [ ] **Step 3: Commit**

```bash
git add src/agents/risk_manager_agent.py
git commit -m "feat: refactor risk manager to use adaptive regime detection"
```

---

### Task 5: Add regime columns to CSV output

**Files:**
- Modify: `src/tools/daily_csv_tool.py`

- [ ] **Step 1: Add `regime` and `regime_score` to `save_workflow_to_csv`**

In `save_workflow_to_csv`, after extracting `nlp_features` (around line 115), add extraction of risk metrics. Then add the new columns to `csv_data`.

After line 115 (`nlp_features = ...`), add:

```python
        # Get risk manager results for regime info
        risk_results = results.get('risk_manager', {})
        risk_metrics = risk_results.get('risk_metrics', {}) if isinstance(risk_results, dict) else {}
        regime = risk_metrics.get('regime', 'neutral')
        regime_score = risk_metrics.get('regime_score', 0.0)
```

Then in the `csv_data` dict (line 119), add after `'risk_profile_change'`:

```python
            'regime': regime,
            'regime_score': regime_score,
```

- [ ] **Step 2: Add `regime` and `regime_score` to `save_workflow_to_symbol_csv`**

Same changes. After extracting `nlp_features` (around line 262), add:

```python
        risk_results = results.get('risk_manager', {})
        risk_metrics = risk_results.get('risk_metrics', {}) if isinstance(risk_results, dict) else {}
        regime = risk_metrics.get('regime', 'neutral')
        regime_score = risk_metrics.get('regime_score', 0.0)
```

Then in the `csv_row` dict (line 264), add after `'risk_profile_change'`:

```python
            'regime': regime,
            'regime_score': regime_score,
```

- [ ] **Step 3: Verify CSV tool imports correctly**

Run: `python -c "from src.tools.daily_csv_tool import save_workflow_to_csv, save_workflow_to_symbol_csv; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/tools/daily_csv_tool.py
git commit -m "feat: add regime and regime_score columns to CSV output"
```

---

### Task 6: Verification — re-run TSLA 30-day backtest

**Files:** None (verification only)

- [ ] **Step 1: Run TSLA analysis with new regime detection**

```bash
source venv/Scripts/activate && PRIMO_SYMBOL=TSLA PRIMO_START_DATE=2026-04-08 PRIMO_END_DATE=2026-05-08 python main.py
```

Expected: Analysis completes with regime info in risk manager output. Regime transitions should appear in the log (e.g., "regime: bear → neutral → bull" as recovery progresses).

- [ ] **Step 2: Check CSV has regime columns**

Run: `python -c "import pandas as pd; df = pd.read_csv('output/csv/daily_analysis_TSLA.csv'); print(df[['date', 'trading_signal', 'position_size']].tail(10)); print(); print('Regime values:'); print(df['regime'].value_counts())"`

Expected: `regime` column contains 'bear', 'neutral', and/or 'bull' values. Early dates should be 'bear', later dates should transition.

- [ ] **Step 3: Run backtest**

```bash
source venv/Scripts/activate && python -c "
import sys
from pathlib import Path
import pandas as pd
from src.backtesting import run_backtest, PrimoAgentStrategy, BuyAndHoldStrategy
from src.backtesting.data import load_stock_data
from src.backtesting.plotting import plot_single_stock

data_dir = 'output/csv'
output_dir = 'output/backtests'
symbol = 'TSLA'

ohlc_data, signals_df = load_stock_data(symbol, data_dir)
primo_results, primo_cerebro = run_backtest(ohlc_data, PrimoAgentStrategy, 'PrimoAgent', signals_df=signals_df, printlog=True)
buyhold_results, buyhold_cerebro = run_backtest(ohlc_data, BuyAndHoldStrategy, 'Buy & Hold')

print()
print(f\"{'Metric':<22} {'PrimoAgent':>12} {'Buy & Hold':>12}\")
print('-' * 50)
for m in ['Cumulative Return [%]', 'Annual Volatility [%]', 'Max Drawdown [%]', 'Sharpe Ratio', 'Total Trades']:
    pv, bv = primo_results[m], buyhold_results[m]
    print(f'{m:<22} {pv:>12.2f} {bv:>12.2f}')

rel = primo_results['Cumulative Return [%]'] - buyhold_results['Cumulative Return [%]']
print(f\"\\nPrimoAgent vs Buy & Hold: {rel:+.2f}%\")

chart_path = plot_single_stock(symbol, primo_cerebro, buyhold_cerebro, str(output_dir))
print(f'Chart: {chart_path}')
"
```

Expected: PrimoAgent should have more than 4 trades (previous result) because the regime should allow BUY signals during the recovery. Cumulative return should be positive and closer to Buy & Hold than the previous +0.46% vs +18.82%.

- [ ] **Step 4: Verify regime transitions make sense**

Check that:
- Early dates (when TSLA was at $343, near the bottom) are Bear or Neutral
- Middle dates (as price recovers past $380) transition to Neutral or Bull
- Late dates (price > $400, trend established) are Bull
- In Bull regime, BUY signals are not overridden by drawdown guard

- [ ] **Step 5: Commit (if satisfied)**

No code changes to commit — verification only. If adjustments needed, fix and commit those.
