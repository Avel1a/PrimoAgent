# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

所有回复、解释、代码注释和文档请使用中文。

## Commands

```bash
# 激活虚拟环境（运行任何 Python 命令前必须先执行）
source venv/Scripts/activate

# Install dependencies (Python 3.12+)
pip install -r requirements.txt
pip install pyarrow          # required for Parquet disk caching

# Run analysis pipeline — interactive mode
python main.py

# Run analysis — non-interactive (env vars)
PRIMO_SYMBOL=AAPL PRIMO_START_DATE=2026-03-02 PRIMO_END_DATE=2026-04-30 python main.py

# Run backtest (interactive: 1=single, 2=multi-stock)
python backtest.py

# 市场状态对比测试 — 牛/熊/震荡三组 (分析 + 回测全自动)
python run_batch_market_regime.py

# 单股票回测
python _backtest_single.py AAPL

# 非交互式多股票综合回测 (产出对比报告)
python run_backtest_quick.py
```

No lint, test, or type-check scripts are configured in this project.

## Architecture

This is a **LangGraph-based multi-agent stock analysis system** with 5 LLM-powered agents in a DAG pipeline, plus a Backtrader backtesting engine.

### Pipeline Flow

```
data_collection → [technical_analysis ∥ news_intelligence] → portfolio_manager → risk_manager → CSV output
                                                                                       ↑
                                                          Adaptive Regime Detection ───┘
                                                          (rolling dd + trend + vol → Bear/Neutral/Bull)
```

Errors at any stage short-circuit to END via conditional routing (`should_continue` in `src/workflows/workflow.py`). The parallel stage uses `asyncio.gather` with deep-copied state — results are merged back into the original state after both agents complete.

### Core State

`AgentState` (TypedDict in `src/workflows/state.py`) carries all data between nodes: session metadata, analysis date, agent results dicts, an optional `_cached_company_info` for cross-day reuse, and an `error` field that triggers conditional routing to END.

### Data Sources

- **Tiingo** (primary): OHLCV prices + company info. Uses a **Parquet disk cache** in `output/cache/tiingo/` — first call fetches 5 years of data, subsequent calls hit cache only. Metadata has no cache but is cached in-memory across days by `main.py`.
- **Finnhub**: Company news, profiles, basic financials, market holidays. News is source-filtered (`config.json` → `news.valid_sources`) and trading-session-filtered.
- **Alpha Vantage**: SPY data for S&P 500 benchmark (backtest). yfinance is a fallback for OHLCV in backtesting data loading.

### LLM Configuration

All models are configured in `src/config/config.json` under `models.*` with `provider`, `model`, and `temperature` keys. The default provider is `openai` pointing to `deepseek-chat`. A bridge in `ModelFactory` maps `OPENAI_BASE_URL` to LangChain's `OPENAI_API_BASE` env var.

### Risk Manager (Adaptive Regime)

`src/agents/risk_manager_agent.py` first detects the market regime, then enforces 4 constraints with regime-specific thresholds:

**Regime Detection** — combines 3 signals into `regime_score ∈ [-1, 1]`:
- **Rolling drawdown** (60d window, replaces global-peak): `1 - min(dd/30, 1)`
- **Trend strength** (SMA20 slope + MACD direction + ADX): `(slope_score + macd_score + adx_score) / 3`
- **Volatility state** (ATR14 / ATR60 mean): `1 - (atr_ratio - 1)`, clamped to [-1, 1]
- Weighted: `0.4 × trend + 0.3 × dd + 0.3 × vol`

**Regime Thresholds** (configured in `config.json` → `risk.regime`):

| Regime | Condition | Max DD | Max Pos | VaR Mult | Cash Res | HOLD Override |
|--------|-----------|--------|---------|----------|----------|---------------|
| Bear | < -0.3 | 12% | 10% | 0.5x | 20% | All signals |
| Neutral | -0.3~0.3 | 20% | 25% | 1.0x | 10% | DD > threshold |
| Bull | > 0.3 | 30% | 30% | 1.5x | 5% | DD > threshold AND trend < 0 |

**Constraints** (applied after regime detection):
1. **Drawdown guard** — regime-aware HOLD override
2. **VaR position capping** — scaled by `regime.var_multiplier`
3. **Single-position cap** — `regime.max_position_pct`
4. **Cash reserve** — `regime.cash_reserve_pct`

Falls back to Neutral if < 60 bars of historical data. Position sizes rounded to nearest 10 (floor at 10). The risk manager **mutates** `portfolio_manager_results` in the state directly.

### Backtesting

Backtesting reads per-symbol CSVs from `output/csv/daily_analysis_{SYMBOL}.csv` (produced by the analysis pipeline). Key components in `src/backtesting/`:

- `engine.py`: Wraps Backtrader Cerebro with configurable commission (0.1%) and slippage (0.1%)
- `strategies.py`: `PrimoAgentStrategy` executes BUY/SELL signals day-by-day; `BuyAndHoldStrategy` buys once at start
- `data.py`: Loads OHLCV via yfinance, signals from CSVs; SPY and equal-weight benchmarks
- `plotting.py` / `reporting.py`: Output charts and markdown reports

### 市场状态对比测试

`run_batch_market_regime.py` 按牛/熊/震荡三组跑 6 只股票的完整分析+回测。修改 `GROUPS` 列表即可换股票和时间段：

```python
GROUPS = [
    ("NVDA", "2024-01-02", "2024-03-08", "Bull"),     # 牛市组
    ("TSLA", "2022-09-01", "2022-11-10", "Bear"),     # 熊市组
    ("JNJ",  "2023-06-01", "2023-08-15", "Sideways"), # 震荡组
]
```

每组可以放多只股票，`_backtest_single.py` 是单股票回测辅助脚本，也可单独调用 `python _backtest_single.py <SYMBOL>`。

### Output Artifacts

- `output/csv/daily_analysis.csv` — combined CSV (all symbols, sorted newest first)
- `output/csv/daily_analysis_{SYMBOL}.csv` — per-symbol CSVs (sorted ascending by date)
- `output/backtests/` — backtest PNG charts and markdown reports
- `output/cache/tiingo/{SYMBOL}_ohlcv.parquet` — OHLCV disk cache

### Key Patterns

- All API tools return a `ToolResult` dataclass (`success`, `data`, `error`, `timestamp`) from `src/tools/utils.py`
- Company info is fetched once then passed via `cached_company_info` across all days in a multi-day run — avoids redundant Tiingo calls
- NLP features from News Intelligence are integer scores in `{-2, -1, 0, 1, 2}` for 7 dimensions (relevance, sentiment, price impact, trend, earnings, investor confidence, risk change)
- Portfolio Manager decision uses discrete values: confidence ∈ {0.1, 0.2, ..., 1.0}, position ∈ {10, 20, ..., 100}
- The `n/` directory appears unused/empty
