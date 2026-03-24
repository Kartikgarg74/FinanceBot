"""Alpaca Markets client for US stock trading (free paper trading)."""

import logging
from typing import Optional

import pandas as pd

from ..base_trader import BaseTrader, OrderSide, OrderType, TradeRecord, Position

logger = logging.getLogger(__name__)


class AlpacaClient(BaseTrader):
    """
    Alpaca Markets wrapper.
    - Free paper trading, commission-free live trading
    - Historical data via yfinance (free) or Alpaca Data API
    """

    def __init__(self, config: dict, mode: str = "paper"):
        super().__init__(config, mode)
        self.alpaca_config = config.get("alpaca", {})
        self.trading_config = config.get("trading", {})
        self._api = None

    def connect(self) -> bool:
        if self.mode == "paper" and not self.alpaca_config.get("api_key"):
            logger.info("[PAPER] Alpaca client in offline paper mode")
            self._paper_capital = float(self.trading_config.get("max_capital_per_trade", 1000)) * 15
            return True

        api_key = self.alpaca_config.get("api_key", "")
        api_secret = self.alpaca_config.get("api_secret", "")
        base_url = self.alpaca_config.get("base_url", "https://paper-api.alpaca.markets")

        if not api_key or not api_secret:
            logger.error("Alpaca API key/secret not configured")
            return False

        try:
            from alpaca.trading.client import TradingClient
            self._api = TradingClient(api_key, api_secret, paper=("paper" in base_url))
            account = self._api.get_account()
            logger.info("Alpaca connected. Buying power: $%s", account.buying_power)
            return True
        except Exception as e:
            logger.error("Alpaca connection failed: %s", e)
            return False

    def get_positions(self) -> list[Position]:
        if self.mode == "paper" and not self._api:
            return list(self._paper_portfolio.values())

        try:
            positions = self._api.get_all_positions()
            return [
                Position(
                    ticker=p.symbol,
                    quantity=float(p.qty),
                    avg_price=float(p.avg_entry_price),
                    current_price=float(p.current_price),
                    pnl=float(p.unrealized_pl),
                    pnl_pct=float(p.unrealized_plpc) * 100,
                    exchange="ALPACA",
                )
                for p in positions
            ]
        except Exception as e:
            logger.error("Failed to get Alpaca positions: %s", e)
            return []

    def get_balance(self) -> float:
        if self.mode == "paper" and not self._api:
            return self._paper_capital

        try:
            account = self._api.get_account()
            return float(account.buying_power)
        except Exception as e:
            logger.error("Failed to get Alpaca balance: %s", e)
            return 0

    def place_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int | float,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> TradeRecord:
        if self.mode == "paper" and not self._api:
            from ..base_trader import Signal, SignalAction
            signal = Signal(
                ticker=ticker,
                action=SignalAction.BUY if side == OrderSide.BUY else SignalAction.SELL,
                confidence=100,
                price=price or 0,
            )
            return self._paper_trade(signal, quantity)

        try:
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide as AlpSide, TimeInForce

            alp_side = AlpSide.BUY if side == OrderSide.BUY else AlpSide.SELL

            if order_type == OrderType.MARKET:
                req = MarketOrderRequest(
                    symbol=ticker,
                    qty=int(quantity),
                    side=alp_side,
                    time_in_force=TimeInForce.DAY,
                )
            else:
                req = LimitOrderRequest(
                    symbol=ticker,
                    qty=int(quantity),
                    side=alp_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=price,
                )

            order = self._api.submit_order(req)
            return TradeRecord(
                id=str(order.id),
                ticker=ticker,
                side=side.value,
                quantity=int(quantity),
                price=float(order.filled_avg_price or price or 0),
                order_type=order_type.value,
                status="executed",
                exchange="ALPACA",
            )
        except Exception as e:
            logger.error("Alpaca order failed: %s", e)
            return TradeRecord(
                id="failed", ticker=ticker, side=side.value,
                quantity=int(quantity), price=price or 0,
                order_type=order_type.value, status="failed",
                reason=str(e), exchange="ALPACA",
            )

    def get_quote(self, ticker: str) -> dict:
        from ..analysis.data_fetcher import DataFetcher
        df = DataFetcher().fetch_us_stock(ticker, interval="1d", days=5)
        if df is not None and not df.empty:
            row = df.iloc[-1]
            return {
                "ticker": ticker,
                "price": float(row["Close"]),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "volume": int(row["Volume"]) if "Volume" in row else 0,
            }
        return {"ticker": ticker, "price": 0}

    def get_historical_data(self, ticker: str, interval: str = "1d", days: int = 365):
        from ..analysis.data_fetcher import DataFetcher
        return DataFetcher().fetch_us_stock(ticker, interval, days)
