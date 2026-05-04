# PrimoAgent：多智能体 AI 股票分析系统

## 概述

PrimoAgent 是一个基于 **LangGraph** 的多智能体 AI 股票分析系统，由 5 个专业化 Agent 组成流水线，整合技术分析、NLP 新闻分析和风险管理，每日输出交易信号（BUY/SELL/HOLD）及置信度。回测引擎支持多基准对比和真实交易成本模拟。

## 核心架构

### 并行流水线

```
数据采集 Agent
      ↓
[技术分析 Agent ∥ 新闻情报 Agent]   ← 并行执行（asyncio.gather）
      ↓
投资组合管理 Agent
      ↓
风险管理 Agent
      ↓
CSV 输出 + 回测引擎
```

### 五个 Agent

| Agent | 职责 | 数据来源 |
|-------|------|---------|
| **数据采集** | 拉取 OHLCV 行情、公司信息、新闻 | Tiingo（主）+ Finnhub（新闻/财报） |
| **技术分析** | SMA、RSI、MACD、布林带、ADX、CCI | Tiingo 历史数据 |
| **新闻情报** | 7 维 NLP 特征提取（情绪、相关性、价格影响等） | Finnhub 新闻 + Firecrawl 抓取 |
| **投资组合管理** | LLM 综合所有数据产出交易信号 + 仓位 | DeepSeek LLM |
| **风险管理** | VaR 仓位上限、最大回撤熔断、现金储备、集中度限制 | 历史价格数据 |

### 关键特性

- **Tiingo 磁盘缓存**：OHLCV 历史数据以 Parquet 格式缓存至 `output/cache/tiingo/`，首次拉取 5 年全量，后续当天回测零 API 调用
- **并行分析**：技术分析和新闻情报通过 `asyncio.gather` 并发执行
- **条件路由**：Agent 出错时自动跳过后续节点，避免无效 LLM 调用
- **风险预算**：4 道风控（VaR 仓位上限、最大回撤熔断、单仓集中度、现金储备）
- **真实回测成本**：佣金 0.1% + 滑点 0.1%，Backtrader 引擎
- **多基准对比**：S&P 500 (SPY) + 等权重组合 + Buy & Hold

## 项目结构

```
src/
  agents/
    data_collection_agent.py     # Agent 1：数据采集
    technical_analysis_agent.py  # Agent 2：技术分析
    news_intelligence_agent.py   # Agent 3：新闻情报（NLP）
    portfolio_manager_agent.py   # Agent 4：投资组合管理
    risk_manager_agent.py        # Agent 5：风险管理
  backtesting/
    engine.py                    # Backtrader Cerebro 配置
    strategies.py                # PrimoAgent / BuyAndHold 策略
    data.py                      # 数据加载 + SPY 基准 + 等权组合
    plotting.py                  # 图表生成
    reporting.py                 # Markdown 报告
  config/
    config.json                  # 交易参数、模型配置、风控阈值
    config.py                    # 配置读取 + 属性访问
    model_factory.py             # LLM 工厂（OpenAI/Anthropic）
  prompts/                       # LLM Prompt 模板
  tools/
    tiingo_tool.py               # Tiingo API + Parquet 磁盘缓存
    finnhub_tool.py              # Finnhub 新闻/财报/节假日
    yfinance_tool.py             # Alpha Vantage OHLCV（备用）
    technical_indicators_tool.py # ta 库技术指标计算
    daily_csv_tool.py            # 日频分析 CSV 持久化
  workflows/
    state.py                     # AgentState TypedDict 定义
    workflow.py                  # LangGraph 图定义 + 执行

output/
  cache/tiingo/                  # OHLCV Parquet 缓存
  csv/                           # 每日分析结果
  backtests/                     # 回测图表 + 报告
```

## 快速开始

### 环境配置

```bash
# 创建虚拟环境（Python 3.12+）
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
pip install pyarrow         # Parquet 缓存支持
```

### API Key 配置

在项目根目录创建 `.env` 文件（已加入 `.gitignore`）：

```bash
OPENAI_API_KEY=    # DeepSeek 或 OpenAI API Key
OPENAI_BASE_URL=   # API Base URL（如 https://api.deepseek.com/v1）
TIINGO_API_KEY=    # Tiingo（免费版 50次/小时）
FINNHUB_API_KEY=   # Finnhub（新闻 + 财报）
ALPHA_VANTAGE_API_KEY=  # Alpha Vantage（备用数据源 + SPY 基准）
```

### 两步工作流

**Step 1：运行分析管线**

```bash
# 交互模式
python main.py

# 或使用环境变量（非交互）
PRIMO_SYMBOL=AAPL PRIMO_START_DATE=2026-03-02 PRIMO_END_DATE=2026-04-30 python main.py
```

**Step 2：回测**

```bash
python backtest.py
# 1 = 单股回测（含 S&P 500 对比）
# 2 = 多股回测（含 S&P 500 + 等权组合）
```

## 最新回测结果

> AAPL 单股，2026-03-02 ~ 2026-03-20（15 个交易日），初始资金 $100,000

| 策略 | 累计收益 | 年化波动 | 最大回撤 | Sharpe | 交易次数 |
|------|---------|---------|---------|--------|---------|
| PrimoAgent | -5.90% | 15.14% | 7.82% | -1.55 | 2 |
| Buy & Hold | 0.00% | 0.00% | 0.00% | 0.00 | 1 |

> 注：15 天数据量较小，该周期 AAPL 处于下跌震荡行情。风险模块将最大回撤控制在 7.82%。

## 配置参考

### 风险管理 (`config.json` → `risk`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_position_pct` | 25 | 单仓最大占比 |
| `stop_loss_pct` | 5 | 止损线（%） |
| `take_profit_pct` | 15 | 止盈线（%） |
| `max_drawdown_pct` | 20 | 最大回撤熔断线 |
| `max_daily_var_pct` | 2.0 | 日 VaR 限额（95% 置信） |

### 回测成本 (`config.json` → `backtesting`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `initial_cash` | 100000 | 初始资金 |
| `commission_pct` | 0.1 | 佣金费率 |
| `slippage_pct` | 0.1 | 滑点 |

## 引用

Botunac, I. (2025). Implementation of a multi-agent artificial intelligence system for financial trading decision-making. *Oeconomica Jadertina*, 15(2), 90-115.

```bibtex
@article{botunac2025multiagent,
  title={Implementacija vi\v{s}eagentnog sustava umjetne inteligencije
         za dono\v{s}enje financijskih trgova\v{c}kih odluka},
  author={Botunac, Ive},
  journal={Oeconomica Jadertina},
  volume={15},
  number={2},
  pages={90--115},
  year={2025},
  doi={10.15291/oec.4863},
  url={https://hrcak.srce.hr/341777}
}
```

## 免责声明

本项目为学术研究代码，仅供教育目的。所有交易策略均为实验性质，不构成任何投资建议。交易涉及重大损失风险，做出投资决策前请咨询专业金融顾问。
