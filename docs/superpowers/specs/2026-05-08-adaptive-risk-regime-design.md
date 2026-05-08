# Adaptive Risk Regime Detector — Design Spec

**Date:** 2026-05-08  
**Status:** approved  
**Scope:** Risk Manager refactor to replace fixed-parameter drawdown guard with adaptive multi-signal regime detection

---

## Problem

`_recent_drawdown()` uses the all-time peak from historical data (up to 5 years via Tiingo cache). When a stock drops significantly from a distant peak (e.g., TSLA $490 → $343, -30%), the risk manager forces HOLD on every subsequent day, including during a recovery rally (+20%). The stale peak never decays, permanently locking the strategy out of trades.

## Solution

Replace the single-threshold drawdown guard with a **Risk Regime Detector** that combines three signals into a regime score, then dynamically adjusts risk parameters per regime.

---

## Architecture

```
_data_collection_results (historical OHLCV)
         │
         ▼
┌─────────────────────────┐
│   Risk Regime Detector   │
│                         │
│  1. Rolling Drawdown    │──┐
│     (60d peak, not all- │  │
│      time)              │  │
│  2. Trend Strength      │──┤──► regime_score ∈ [-1, 1]
│     (SMA slope + MACD   │  │
│      direction + ADX)   │  │
│  3. Volatility State    │──┘
│     (ATR14 vs ATR60)    │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Dynamic Risk Adjuster   │
│                         │
│  Bear  (< -0.3): tight  │
│  Neutral (-0.3~0.3)     │
│  Bull  (> 0.3): loose   │
└─────────┬───────────────┘
          │
          ▼
   Risk checks applied
   (VaR cap, position cap,
    cash reserve, drawdown
    override with regime-
    specific thresholds)
```

---

## Three Input Signals

### 1. Rolling Drawdown (replaces global-peak drawdown)

```
peak_60d = max(closes[-60:])
rolling_dd = (peak_60d - current_close) / peak_60d × 100
rolling_dd_score = 1 - min(rolling_dd / 30, 1)   # 0% dd → 1.0, 30% dd → 0.0
```

A 60-day rolling window means a recovery rally naturally shrinks the drawdown. After TSLA rises from $343 to $400, the 60d peak updates as old $490 data exits the window.

### 2. Trend Strength

Three sub-components, equally weighted:

| Sub-signal | Computation | Normalization |
|-----------|-------------|---------------|
| SMA slope | `(sma20[-1] - sma20[-20]) / sma20[-20]` | `tanh(slope × 50)` → [-1, 1] |
| MACD direction | `sign(macd_hist)` × `min(abs(macd_hist)/2, 1)` | [-1, 1] |
| ADX strength | `adx / 50` | capped at 1.0, combined with SMA slope sign |

```
trend_score = (sma_slope_score + macd_direction_score + adx_score) / 3
```

### 3. Volatility State

```
atr_ratio = ATR(14) / mean(ATR(14) over last 60 days)
vol_score = 1 - (atr_ratio - 1)   # 1.0 = normal vol, >1 = low vol (good), <1 = high vol (bad)
vol_score = clamp(vol_score, -1, 1)
```

### Regime Score

```
regime_score = 0.4 × trend_score + 0.3 × rolling_dd_score + 0.3 × vol_score
```

---

## Dynamic Parameter Tables

| Parameter | Bear (score < -0.3) | Neutral (-0.3 to 0.3) | Bull (score > 0.3) |
|-----------|--------------------|-----------------------|--------------------|
| `max_drawdown_pct` | 12% | 20% | 30% |
| `max_position_pct` | 10% | 25% | 30% |
| `var_multiplier` | 0.5 | 1.0 | 1.5 |
| HOLD override | On any BUY/SELL | Only if dd > threshold | Only Bear-level dd |
| Cash reserve | 20% | 10% | 5% |

**HOLD override logic per regime:**

- **Bear:** If rolling_dd > 12%, force HOLD on all signals (tight defense)
- **Neutral:** If rolling_dd > 20%, force HOLD (current behavior, but with rolling dd)
- **Bull:** Only force HOLD if rolling_dd > 30% AND trend_score < 0 (trend disagrees with the rally)

Key insight: In Bull regime, a BUY signal is allowed to pass even if there is some residual drawdown — as long as trend confirms the recovery.

---

## Implementation Plan

### Files to change

1. **`src/agents/risk_manager_agent.py`** — Core changes:
   - Add `_compute_regime_score(historical_closes)` function
   - Add `_compute_rolling_drawdown(closes, window=60)` replacing `_recent_drawdown`
   - Add `_compute_trend_score(closes)` 
   - Add `_compute_volatility_score(closes)`
   - Refactor `risk_manager_agent_node` to compute regime first, then apply dynamic thresholds
   - Add `regime` and `regime_score` fields to `risk_metrics`

2. **`src/config/config.json`** — Add `risk.regime` section:
   ```json
   "regime": {
     "rolling_drawdown_window": 60,
     "trend_sma_period": 20,
     "volatility_atr_period": 14,
     "volatility_lookback": 60,
     "weights": { "trend": 0.4, "rolling_dd": 0.3, "volatility": 0.3 },
     "bull_threshold": 0.3,
     "bear_threshold": -0.3,
     "bear": { "max_drawdown_pct": 12, "max_position_pct": 10, "var_multiplier": 0.5, "cash_reserve_pct": 20 },
     "neutral": { "max_drawdown_pct": 20, "max_position_pct": 25, "var_multiplier": 1.0, "cash_reserve_pct": 10 },
     "bull": { "max_drawdown_pct": 30, "max_position_pct": 30, "var_multiplier": 1.5, "cash_reserve_pct": 5 }
   }
   ```

3. **`src/config/config.py`** — Add `RegimeConfig` dataclass or properties for the nested regime config.

### Edge Cases

- **Insufficient history (< 60 days):** Default to Neutral regime with current fixed thresholds. Skip regime detection entirely.
- **Flat market (all scores ≈ 0):** Falls into Neutral band. No oscillation between regimes.
- **Conflicting signals:** If trend is strong bullish but rolling dd is high (early recovery), trend weight (0.4) slightly dominates dd (0.3). The combined score determines regime — no hard-coded conflict resolution needed.

### Backward Compatibility

- Default (Neutral) regime parameters match current hardcoded values exactly
- Existing `risk_*` config keys preserved as Neutral defaults
- CSV output unchanged (adds `regime` → `regime_score` column)

### Verification

- Re-run TSLA 30-day backtest: strategy should participate in the recovery
- Regime should transition from Bear → Neutral → Bull as price recovers and trend improves
- In a downtrend (e.g., a stock steadily declining), regime should stay Bear and continue protecting
