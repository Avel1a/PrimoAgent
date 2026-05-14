from __future__ import annotations

from typing import Any, List, Optional

import backtrader as bt
import pandas as pd


class PrimoAgentStrategy(bt.Strategy):
    """AI-driven trading strategy using PrimoAgent signals with trailing stop."""

    params: tuple = (
        ("signals_df", None),
        ("printlog", False),
        ("trailing_stop_pct", 5.0),
        ("take_profit_pct", 15.0),
    )

    signals_df: Optional[pd.DataFrame]
    portfolio_values: List[float]
    order_count: int
    highest_price: float

    def __init__(self) -> None:
        self.signals_df = self.p.signals_df
        self.portfolio_values = []
        self.order_count = 0
        self.highest_price = 0.0

    def log(self, txt: str, dt: Any = None) -> None:
        if self.p.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}: {txt}")

    def notify_order(self, order) -> None:
        if order.status == order.Completed and order.isbuy():
            self.highest_price = self.data.close[0]

    def next(self) -> None:
        current_date = self.data.datetime.date(0)
        current_price = self.data.close[0]

        portfolio_value = self.broker.getvalue()
        self.portfolio_values.append(portfolio_value)

        # --- trailing stop / take profit while in position ---
        if self.position:
            self.highest_price = max(self.highest_price, current_price)

            # Trailing stop: sell if price drops below stop threshold from peak
            stop_price = self.highest_price * (1.0 - self.p.trailing_stop_pct / 100.0)
            if current_price <= stop_price:
                self.sell(size=self.position.size)
                self.order_count += 1
                self.log(
                    f"   TRAILING STOP: SOLD {self.position.size} shares @ ${current_price:.2f} "
                    f"(peak: ${self.highest_price:.2f}, stop: ${stop_price:.2f})"
                )
                self.highest_price = 0.0
                return

            # Take profit: sell if price reaches profit target
            avg_entry = self.position.price
            if avg_entry > 0 and current_price >= avg_entry * (1.0 + self.p.take_profit_pct / 100.0):
                self.sell(size=self.position.size)
                self.order_count += 1
                self.log(
                    f"   TAKE PROFIT: SOLD {self.position.size} shares @ ${current_price:.2f} "
                    f"(entry: ${avg_entry:.2f}, target: +{self.p.take_profit_pct}%)"
                )
                self.highest_price = 0.0
                return

        if self.signals_df is None:
            return

        current_signal_row = self.signals_df[
            self.signals_df["date"].dt.date == current_date
        ]

        if current_signal_row.empty:
            return

        signal = current_signal_row.iloc[0]["trading_signal"]
        position_percent = current_signal_row.iloc[0]["position_size"] / 100.0

        self.log(
            f"{current_date} | Signal: {signal} | Price: ${current_price:.2f} | "
            f"Position: {self.position.size} shares"
        )

        if signal == "BUY":
            available_cash = self.broker.getcash()
            target_cash = available_cash * position_percent
            size = int(target_cash / current_price)

            if size >= 1:
                self.buy(size=size)
                self.order_count += 1
                self.log(f"   BOUGHT {size} shares @ ${current_price:.2f}")
            else:
                self.log(
                    f"   Not enough cash for 1 share (need ${current_price:.2f}, "
                    f"have ${available_cash:.2f})"
                )

        elif signal == "SELL" and self.position:
            size = int(self.position.size * position_percent)
            if size >= 1:
                self.sell(size=size)
                self.order_count += 1
                self.log(f"   SOLD {size} shares @ ${current_price:.2f}")
                if not self.position:
                    self.highest_price = 0.0
            else:
                self.log("   Less than 1 share to sell")


class BuyAndHoldStrategy(bt.Strategy):
    """Simple buy and hold strategy for comparison."""

    bought: bool
    portfolio_values: List[float]
    order_count: int

    def __init__(self) -> None:
        self.bought = False
        self.portfolio_values = []
        self.order_count = 0

    def notify_order(self, order) -> None:
        if order.status in [order.Rejected, order.Margin]:
            # 跨 bar 价格跳变可能导致资金不足，自动缩小仓位重试
            smaller = max(1, int(order.created.size * 0.9))
            if smaller < order.created.size:
                self.buy(size=smaller)
                self.order_count += 1

    def next(self) -> None:
        portfolio_value = self.broker.getvalue()
        self.portfolio_values.append(portfolio_value)

        if not self.bought:
            # 预留 5% 缓冲：佣金 0.2% + 跨 bar 价格波动 ~4.8%
            size = int(self.broker.getcash() / (self.data.close[0] * 1.05))
            if size > 0:
                self.buy(size=size)
                self.order_count += 1
                self.bought = True

