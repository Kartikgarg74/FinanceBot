"""Binance crypto trading client using CCXT."""

import logging
from typing import Optional

import pandas as pd

from ..base_trader import BaseTrader, OrderSide, OrderType, TradeRecord, Position

logger = logging.getLogger(__name__)


class BinanceClient(BaseTrader):
    """
    Binance trading client via CCXT.
    - Testnet for paper trading
    - WebSocket for real-time tick data + kline close events
    - REST for order placement
    """

    def __init__(self, config: dict, mode: str = "dry_run"):
        super().__init__(config, mode)
        self.binance_config = config.get("binance", {})
        self.trading_config = config.get("trading", {})
        self.pairs = config.get("pairs", ["BTC/USDT", "ETH/USDT"])
        self._exchange = None
        self.ws = None  # WebSocket streamer (initialized in start_websocket)

    def connect(self) -> bool:
        api_key = self.binance_config.get("api_key", "")
        api_secret = self.binance_config.get("api_secret", "")
        use_testnet = self.binance_config.get("testnet", True)

        if self.mode == "dry_run" and not api_key:
            logger.info("[DRY_RUN] Binance client in offline dry-run mode")
            self._paper_capital = float(self.trading_config.get("stake_amount", 100)) * 50
            return True

        try:
            import ccxt
            exchange_options = {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }

            if use_testnet:
                exchange_options["sandbox"] = True

            self._exchange = ccxt.binance(exchange_options)

            # Test connection
            balance = self._exchange.fetch_balance()
            usdt = balance.get("USDT", {}).get("free", 0)
            logger.info("Binance connected%s. USDT balance: %.2f",
                        " (TESTNET)" if use_testnet else "", usdt)
            return True
        except Exception as e:
            logger.error("Binance connection failed: %s", e)
            return False

    def get_positions(self) -> list[Position]:
        if self.mode == "dry_run" and not self._exchange:
            return list(self._paper_portfolio.values())

        try:
            balance = self._exchange.fetch_balance()
            positions = []
            for currency, info in balance.items():
                if isinstance(info, dict) and info.get("total", 0) > 0 and currency != "USDT":
                    qty = info["total"]
                    pair = f"{currency}/USDT"
                    try:
                        ticker = self._exchange.fetch_ticker(pair)
                        price = ticker["last"]
                        # Estimate avg price from current (can't get from exchange easily)
                        positions.append(Position(
                            ticker=pair,
                            quantity=qty,
                            avg_price=price,  # Approximation
                            current_price=price,
                            pnl=0,  # Can't calculate without entry price
                            pnl_pct=0,
                            exchange="BINANCE",
                        ))
                    except Exception:
                        pass
            return positions
        except Exception as e:
            logger.error("Failed to get Binance positions: %s", e)
            return []

    def get_balance(self) -> float:
        if self.mode == "dry_run" and not self._exchange:
            return self._paper_capital

        try:
            balance = self._exchange.fetch_balance()
            return float(balance.get("USDT", {}).get("free", 0))
        except Exception as e:
            logger.error("Failed to get Binance balance: %s", e)
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
        if self.mode == "dry_run" and not self._exchange:
            from ..base_trader import Signal, SignalAction
            signal = Signal(
                ticker=ticker,
                action=SignalAction.BUY if side == OrderSide.BUY else SignalAction.SELL,
                confidence=100,
                price=price or 0,
            )
            return self._paper_trade(signal, quantity)

        try:
            side_str = "buy" if side == OrderSide.BUY else "sell"

            if order_type == OrderType.MARKET:
                order = self._exchange.create_market_order(ticker, side_str, quantity)
            elif order_type == OrderType.LIMIT:
                order = self._exchange.create_limit_order(ticker, side_str, quantity, price)
            else:
                # Stop-loss order
                order = self._exchange.create_order(
                    ticker, "stop_loss_limit", side_str, quantity,
                    price, params={"stopPrice": stop_loss}
                )

            filled_price = order.get("average") or order.get("price") or price or 0

            trade = TradeRecord(
                id=str(order.get("id", "")),
                ticker=ticker,
                side=side.value,
                quantity=quantity,
                price=float(filled_price),
                order_type=order_type.value,
                status="executed" if order.get("status") == "closed" else "pending",
                exchange="BINANCE",
            )

            # Place stop-loss as separate order if specified
            if stop_loss and side == OrderSide.BUY:
                try:
                    self._exchange.create_order(
                        ticker, "stop_loss_limit", "sell", quantity,
                        stop_loss * 0.99, params={"stopPrice": stop_loss}
                    )
                    trade.stop_loss = stop_loss
                except Exception as e:
                    logger.warning("Failed to place SL for %s: %s", ticker, e)

            return trade
        except Exception as e:
            logger.error("Binance order failed: %s", e)
            return TradeRecord(
                id="failed", ticker=ticker, side=side.value,
                quantity=quantity, price=price or 0,
                order_type=order_type.value, status="failed",
                reason=str(e), exchange="BINANCE",
            )

    def get_quote(self, ticker: str) -> dict:
        # Priority 1: WebSocket real-time data (sub-second)
        if self.ws and self.ws.is_connected:
            price = self.ws.get_price(ticker)
            if price:
                tick = self.ws.latest_ticks.get(ticker)
                return {
                    "ticker": ticker,
                    "price": price,
                    "open": tick.open if tick else 0,
                    "high": tick.high if tick else 0,
                    "low": tick.low if tick else 0,
                    "volume": tick.volume if tick else 0,
                    "source": "websocket",
                }

        # Priority 2: CCXT REST
        if self._exchange:
            try:
                t = self._exchange.fetch_ticker(ticker)
                return {
                    "ticker": ticker,
                    "price": t.get("last", 0),
                    "open": t.get("open", 0),
                    "high": t.get("high", 0),
                    "low": t.get("low", 0),
                    "volume": t.get("baseVolume", 0),
                    "change_pct": t.get("percentage", 0),
                    "source": "rest",
                }
            except Exception as e:
                logger.error("Binance quote failed for %s: %s", ticker, e)

        # Priority 3: yfinance fallback
        from ..analysis.data_fetcher import DataFetcher
        symbol = ticker.split("/")[0]
        df = DataFetcher().fetch_crypto_yfinance(symbol, interval="1d", days=5)
        if df is not None and not df.empty:
            row = df.iloc[-1]
            return {"ticker": ticker, "price": float(row["Close"]), "source": "yfinance"}
        return {"ticker": ticker, "price": 0, "source": "none"}

    def get_historical_data(self, ticker: str, interval: str = "1h", days: int = 90):
        """Get historical data via CCXT or yfinance fallback."""
        if self._exchange:
            from ..analysis.data_fetcher import DataFetcher
            return DataFetcher().fetch_crypto_ccxt(ticker, timeframe=interval)

        symbol = ticker.split("/")[0]
        from ..analysis.data_fetcher import DataFetcher
        return DataFetcher().fetch_crypto_yfinance(symbol, interval, days)

    async def start_websocket(self, on_kline_close=None):
        """Start WebSocket streaming for all configured pairs."""
        from .websocket_stream import BinanceWebSocket

        intervals = [self.config.get("strategy", {}).get("primary_timeframe", "5m")]
        testnet = self.binance_config.get("testnet", True)

        self.ws = BinanceWebSocket(
            pairs=self.pairs,
            intervals=intervals,
            testnet=testnet,
            on_kline_close=on_kline_close,
        )
        await self.ws.start()
        logger.info("Binance WebSocket started for %d pairs", len(self.pairs))

    async def stop_websocket(self):
        """Stop WebSocket streaming."""
        if self.ws:
            await self.ws.stop()
            self.ws = None

    def get_all_pair_quotes(self) -> dict[str, dict]:
        """Get quotes for all configured trading pairs."""
        quotes = {}
        for pair in self.pairs:
            quotes[pair] = self.get_quote(pair)
        return quotes
