"""Abstract base class for all trading modules."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "SL"
    STOP_LOSS_MARKET = "SL-M"


class SignalAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    ticker: str
    action: SignalAction
    confidence: float  # 0-100
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    technical_score: float = 0.0
    sentiment_score: float = 0.0
    chart_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TradeRecord:
    id: str
    ticker: str
    side: str
    quantity: float
    price: float
    order_type: str
    status: str  # "executed", "failed", "pending"
    reason: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    pnl: Optional[float] = None
    exchange: str = ""


@dataclass
class Position:
    ticker: str
    quantity: float
    avg_price: float
    current_price: float
    pnl: float
    pnl_pct: float
    exchange: str = ""


class BaseTrader(ABC):
    """Abstract base for all market traders (Zerodha, Alpaca, Binance)."""

    def __init__(self, config: dict, mode: str = "paper"):
        self.config = config
        self.mode = mode  # "paper" or "live"
        self.trades: list[TradeRecord] = []
        self._paper_portfolio: dict[str, Position] = {}
        self._paper_capital: float = 100000.0
        self._daily_pnl: float = 0.0

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the broker/exchange. Return True on success."""
        ...

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get current open positions."""
        ...

    @abstractmethod
    def get_balance(self) -> float:
        """Get available capital."""
        ...

    @abstractmethod
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
        """Place an order. Returns trade record."""
        ...

    @abstractmethod
    def get_quote(self, ticker: str) -> dict:
        """Get current quote for a ticker."""
        ...

    @abstractmethod
    def get_historical_data(self, ticker: str, interval: str = "1d", days: int = 365):
        """Get historical OHLCV data as a pandas DataFrame."""
        ...

    def execute_signal(self, signal: Signal, position_size: int | float) -> TradeRecord | None:
        """Execute a trading signal with the calculated position size."""
        if self.mode == "paper":
            return self._paper_trade(signal, position_size)

        side = OrderSide.BUY if signal.action == SignalAction.BUY else OrderSide.SELL
        try:
            trade = self.place_order(
                ticker=signal.ticker,
                side=side,
                quantity=position_size,
                order_type=OrderType.MARKET,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )
            trade.reason = signal.reasoning
            self.trades.append(trade)
            logger.info("Trade executed: %s %s x%s @ %s", side.value, signal.ticker, position_size, trade.price)
            return trade
        except Exception as e:
            logger.error("Order failed for %s: %s", signal.ticker, e)
            return None

    def _paper_trade(self, signal: Signal, position_size: int | float) -> TradeRecord:
        """Simulate a trade in paper mode."""
        import uuid
        trade = TradeRecord(
            id=str(uuid.uuid4())[:8],
            ticker=signal.ticker,
            side=signal.action.value,
            quantity=position_size,
            price=signal.price,
            order_type="MARKET",
            status="executed",
            reason=signal.reasoning,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            exchange=self.__class__.__name__,
        )

        # Update paper portfolio
        if signal.action == SignalAction.BUY:
            cost = signal.price * position_size
            if cost <= self._paper_capital:
                self._paper_capital -= cost
                existing = self._paper_portfolio.get(signal.ticker)
                if existing:
                    total_qty = existing.quantity + position_size
                    existing.avg_price = (
                        (existing.avg_price * existing.quantity + signal.price * position_size) / total_qty
                    )
                    existing.quantity = total_qty
                else:
                    self._paper_portfolio[signal.ticker] = Position(
                        ticker=signal.ticker,
                        quantity=position_size,
                        avg_price=signal.price,
                        current_price=signal.price,
                        pnl=0.0,
                        pnl_pct=0.0,
                        exchange=self.__class__.__name__,
                    )
            else:
                trade.status = "failed"
                trade.reason = "Insufficient paper capital"
        elif signal.action == SignalAction.SELL:
            existing = self._paper_portfolio.get(signal.ticker)
            if existing and existing.quantity >= position_size:
                pnl = (signal.price - existing.avg_price) * position_size
                self._paper_capital += signal.price * position_size
                self._daily_pnl += pnl
                existing.quantity -= position_size
                trade.pnl = pnl
                if existing.quantity == 0:
                    del self._paper_portfolio[signal.ticker]
            else:
                trade.status = "failed"
                trade.reason = "No position to sell"

        self.trades.append(trade)
        logger.info("[PAPER] %s %s x%s @ %.2f | PnL: %s",
                     signal.action.value, signal.ticker, position_size, signal.price,
                     f"₹{trade.pnl:.2f}" if trade.pnl else "N/A")
        return trade
