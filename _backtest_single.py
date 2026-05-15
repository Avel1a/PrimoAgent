"""单股票非交互回测 — 供 run_batch_market_regime.py 调用"""
import sys
from src.backtesting import run_backtest, PrimoAgentStrategy, BuyAndHoldStrategy
from src.backtesting.data import load_stock_data
from src.backtesting.plotting import plot_single_stock

sym = sys.argv[1]
ohlc, sig = load_stock_data(sym, "output/csv")
if ohlc is None or ohlc.empty:
    print(f"  [SKIP] {sym} 无 OHLC 数据")
    sys.exit(0)

primo, cerebro = run_backtest(ohlc, PrimoAgentStrategy, "PrimoAgent", signals_df=sig)
bh, _ = run_backtest(ohlc, BuyAndHoldStrategy, "Buy & Hold")
print(f"  PrimoAgent: {primo['Cumulative Return [%]']:.2f}% | Buy&Hold: {bh['Cumulative Return [%]']:.2f}%")
plot_single_stock(sym, cerebro, _, "output/backtests")
