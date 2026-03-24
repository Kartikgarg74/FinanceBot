"""Binance WebSocket streamer — real-time kline, ticker, and user data streams."""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class TickData:
    """Real-time tick from WebSocket."""
    pair: str
    price: float
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    change_pct: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class KlineData:
    """Real-time kline/candlestick close event."""
    pair: str
    interval: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool  # True when candle is finalized
    timestamp: datetime = field(default_factory=datetime.utcnow)


class BinanceWebSocket:
    """
    Manages WebSocket connections to Binance for real-time data.
    Uses the websockets library (falls back to aiohttp if unavailable).

    Streams:
      - Kline (candlestick) streams per pair/interval
      - Mini ticker for all subscribed pairs
      - User data stream (order fills, balance changes)

    Features:
      - Auto reconnection with exponential backoff
      - Heartbeat/keepalive
      - Callbacks for kline close events (trigger analysis)
    """

    MAINNET_WS = "wss://stream.binance.com:9443/ws"
    TESTNET_WS = "wss://testnet.binance.vision/ws"

    def __init__(self, pairs: list[str], intervals: list[str] | None = None,
                 testnet: bool = True, on_kline_close: Callable | None = None,
                 on_tick: Callable | None = None):
        self.pairs = pairs
        self.intervals = intervals or ["5m"]
        self.testnet = testnet
        self.base_url = self.TESTNET_WS if testnet else self.MAINNET_WS

        # Callbacks
        self.on_kline_close = on_kline_close  # Called when a candle closes
        self.on_tick = on_tick  # Called on every ticker update

        # State
        self._ws = None
        self._running = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 20
        self._base_backoff = 1  # seconds
        self._max_backoff = 120

        # Latest data (accessible by the trading engine)
        self.latest_ticks: dict[str, TickData] = {}
        self.latest_klines: dict[str, KlineData] = {}

    def _build_stream_url(self) -> str:
        """Build combined stream URL for all pairs and intervals."""
        streams = []
        for pair in self.pairs:
            symbol = pair.replace("/", "").lower()  # BTC/USDT -> btcusdt
            # Kline streams
            for interval in self.intervals:
                streams.append(f"{symbol}@kline_{interval}")
            # Mini ticker
            streams.append(f"{symbol}@miniTicker")

        combined = "/".join(streams)
        return f"{self.base_url}/{combined}" if len(streams) == 1 else \
               f"{self.base_url.replace('/ws', '/stream')}?streams={combined}"

    async def start(self):
        """Start the WebSocket connection in the background."""
        self._running = True
        asyncio.create_task(self._connect_loop())
        logger.info("Binance WebSocket starting for %d pairs (%s)",
                     len(self.pairs), "TESTNET" if self.testnet else "MAINNET")

    async def stop(self):
        """Stop the WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("Binance WebSocket stopped")

    async def _connect_loop(self):
        """Main connection loop with auto-reconnect."""
        while self._running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                if not self._running:
                    break
                self._reconnect_attempts += 1
                if self._reconnect_attempts > self._max_reconnect_attempts:
                    logger.error("Max reconnection attempts reached. Stopping WebSocket.")
                    self._running = False
                    break

                backoff = min(
                    self._base_backoff * (2 ** self._reconnect_attempts),
                    self._max_backoff,
                )
                logger.warning(
                    "WebSocket disconnected: %s. Reconnecting in %ds (attempt %d/%d)",
                    str(e)[:100], backoff, self._reconnect_attempts, self._max_reconnect_attempts,
                )
                await asyncio.sleep(backoff)

    async def _connect_and_listen(self):
        """Connect to WebSocket and process messages."""
        url = self._build_stream_url()
        logger.info("Connecting to Binance WebSocket: %s", url[:100])

        try:
            import websockets
            async with websockets.connect(url, ping_interval=30, ping_timeout=10) as ws:
                self._ws = ws
                self._reconnect_attempts = 0  # Reset on successful connect
                logger.info("Binance WebSocket connected (%d streams)", len(self.pairs) * (len(self.intervals) + 1))

                async for message in ws:
                    try:
                        await self._handle_message(json.loads(message))
                    except json.JSONDecodeError:
                        logger.debug("Non-JSON WebSocket message: %s", message[:100])
                    except Exception as e:
                        logger.error("Error processing WebSocket message: %s", e)

        except ImportError:
            # Fallback: try aiohttp
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url, heartbeat=30) as ws:
                        self._ws = ws
                        self._reconnect_attempts = 0
                        logger.info("Binance WebSocket connected via aiohttp")

                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    await self._handle_message(json.loads(msg.data))
                                except Exception as e:
                                    logger.error("Error processing message: %s", e)
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                raise ConnectionError(f"WebSocket error: {ws.exception()}")

            except ImportError:
                logger.error("Neither 'websockets' nor 'aiohttp' installed. Cannot start WebSocket.")
                logger.error("Install with: pip install websockets")
                self._running = False
                return

    async def _handle_message(self, data: dict):
        """Process incoming WebSocket messages."""
        # Combined stream format: {"stream": "btcusdt@kline_5m", "data": {...}}
        if "stream" in data:
            stream_name = data["stream"]
            payload = data["data"]
        else:
            stream_name = data.get("e", "")
            payload = data

        event_type = payload.get("e", "")

        if event_type == "kline":
            await self._handle_kline(payload)
        elif event_type == "24hrMiniTicker":
            await self._handle_ticker(payload)

    async def _handle_kline(self, data: dict):
        """Process kline/candlestick event."""
        k = data.get("k", {})
        symbol = data.get("s", "")  # BTCUSDT
        pair = self._symbol_to_pair(symbol)

        kline = KlineData(
            pair=pair,
            interval=k.get("i", ""),
            open=float(k.get("o", 0)),
            high=float(k.get("h", 0)),
            low=float(k.get("l", 0)),
            close=float(k.get("c", 0)),
            volume=float(k.get("v", 0)),
            is_closed=k.get("x", False),
        )

        self.latest_klines[f"{pair}:{kline.interval}"] = kline

        # Trigger callback when candle closes (this is when we should analyze)
        if kline.is_closed and self.on_kline_close:
            try:
                if asyncio.iscoroutinefunction(self.on_kline_close):
                    await self.on_kline_close(kline)
                else:
                    self.on_kline_close(kline)
            except Exception as e:
                logger.error("Kline close callback error for %s: %s", pair, e)

    async def _handle_ticker(self, data: dict):
        """Process mini ticker event."""
        symbol = data.get("s", "")
        pair = self._symbol_to_pair(symbol)

        tick = TickData(
            pair=pair,
            price=float(data.get("c", 0)),  # Close price
            open=float(data.get("o", 0)),
            high=float(data.get("h", 0)),
            low=float(data.get("l", 0)),
            close=float(data.get("c", 0)),
            volume=float(data.get("v", 0)),
        )

        self.latest_ticks[pair] = tick

        if self.on_tick:
            try:
                if asyncio.iscoroutinefunction(self.on_tick):
                    await self.on_tick(tick)
                else:
                    self.on_tick(tick)
            except Exception as e:
                logger.error("Tick callback error for %s: %s", pair, e)

    def _symbol_to_pair(self, symbol: str) -> str:
        """Convert BTCUSDT -> BTC/USDT."""
        for pair in self.pairs:
            if pair.replace("/", "") == symbol:
                return pair
        # Fallback: assume USDT quote
        if symbol.endswith("USDT"):
            return f"{symbol[:-4]}/USDT"
        return symbol

    def get_price(self, pair: str) -> float | None:
        """Get latest price for a pair from WebSocket data."""
        tick = self.latest_ticks.get(pair)
        if tick:
            return tick.price
        kline = self.latest_klines.get(f"{pair}:{self.intervals[0]}")
        if kline:
            return kline.close
        return None

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._running
