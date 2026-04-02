"""Paper trading engine — simulates realistic order execution.

Works for any market (India/US/Crypto). Tracks positions, P&L, fees,
and generates performance reports. Designed to plug into ML signal generators.

For Indian stocks: uses exact Zerodha fee structure.
For US stocks: designed to work with Alpaca paper API as well.
For Crypto: works standalone or with Binance testnet.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .cost_model import get_cost_model, TradeCost
from .base_trader import Signal, SignalAction, TradeRecord, Position

logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    """A simulated open position."""
    ticker: str
    quantity: float
    avg_price: float
    entry_time: datetime
    entry_fee: float = 0.0
    side: str = "long"


@dataclass
class PaperTradeRecord:
    """Record of a completed paper trade."""
    id: str
    ticker: str
    side: str  # BUY or SELL
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    gross_pnl: float
    entry_fee: float
    exit_fee: float
    slippage_cost: float
    net_pnl: float
    hold_duration: str
    signal_confidence: float = 0.0
    signal_reasoning: str = ""


class PaperTradingEngine:
    """Simulates trading with realistic fee deduction and slippage."""

    def __init__(
        self,
        broker: str = "zerodha",
        trade_type: str = "intraday",
        initial_capital: float = 100000,
        slippage_bps: float = 10,
        max_position_pct: float = 10.0,
        max_positions: int = 5,
    ):
        self.broker = broker
        self.cost_model = get_cost_model(broker, trade_type=trade_type, slippage_bps=slippage_bps)
        self.slippage_bps = slippage_bps

        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_position_pct = max_position_pct
        self.max_positions = max_positions

        self.positions: dict[str, PaperPosition] = {}
        self.trade_history: list[PaperTradeRecord] = []
        self.daily_snapshots: list[dict] = []

        self.peak_capital = initial_capital
        self.total_fees = 0.0
        self.total_slippage = 0.0

    def execute_signal(self, signal: Signal, current_price: float | None = None) -> PaperTradeRecord | None:
        """Execute a trading signal with realistic simulation."""
        price = current_price or signal.price
        if price <= 0:
            return None

        if signal.action == SignalAction.BUY:
            return self._execute_buy(signal, price)
        elif signal.action == SignalAction.SELL:
            return self._execute_sell(signal, price)
        return None

    def _execute_buy(self, signal: Signal, price: float) -> PaperTradeRecord | None:
        """Simulate a BUY order."""
        # Check if already holding
        if signal.ticker in self.positions:
            logger.debug("Already holding %s, skipping BUY", signal.ticker)
            return None

        # Check max positions
        if len(self.positions) >= self.max_positions:
            logger.debug("Max positions reached (%d), skipping", self.max_positions)
            return None

        # Position sizing: max % of capital
        max_value = self.capital * (self.max_position_pct / 100)
        quantity = int(max_value / price)
        if quantity <= 0:
            return None

        trade_value = quantity * price

        # Apply slippage (worse fill for buy = higher price)
        fill_price = price * (1 + self.slippage_bps / 10000)
        actual_cost = quantity * fill_price

        # Calculate fees
        fee = self.cost_model.calculate(actual_cost, "buy")

        # Check if we can afford it
        total_cost = actual_cost + fee.total
        if total_cost > self.capital:
            quantity = int((self.capital - fee.total) / fill_price)
            if quantity <= 0:
                return None
            actual_cost = quantity * fill_price
            fee = self.cost_model.calculate(actual_cost, "buy")
            total_cost = actual_cost + fee.total

        # Execute
        self.capital -= total_cost
        self.total_fees += fee.total - fee.slippage
        self.total_slippage += fee.slippage

        self.positions[signal.ticker] = PaperPosition(
            ticker=signal.ticker,
            quantity=quantity,
            avg_price=fill_price,
            entry_time=datetime.utcnow(),
            entry_fee=fee.total,
        )

        logger.info("[PAPER BUY] %s x%d @ %.2f (fee=%.2f, slip=%.2f) | Capital: %.2f",
                     signal.ticker, quantity, fill_price, fee.total, fee.slippage, self.capital)
        return None  # Trade record created on exit

    def _execute_sell(self, signal: Signal, price: float) -> PaperTradeRecord | None:
        """Simulate a SELL order (close position)."""
        pos = self.positions.get(signal.ticker)
        if not pos:
            logger.debug("No position in %s to sell", signal.ticker)
            return None

        # Apply slippage (worse fill for sell = lower price)
        fill_price = price * (1 - self.slippage_bps / 10000)
        proceeds = pos.quantity * fill_price

        # Calculate fees
        fee = self.cost_model.calculate(proceeds, "sell")
        net_proceeds = proceeds - fee.total

        # P&L calculation
        gross_pnl = (fill_price - pos.avg_price) * pos.quantity
        net_pnl = gross_pnl - pos.entry_fee - fee.total

        self.capital += net_proceeds
        self.total_fees += fee.total - fee.slippage
        self.total_slippage += fee.slippage

        # Track peak capital
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital

        # Create trade record
        now = datetime.utcnow()
        duration = now - pos.entry_time
        record = PaperTradeRecord(
            id=str(uuid.uuid4())[:8],
            ticker=signal.ticker,
            side="BUY→SELL",
            quantity=pos.quantity,
            entry_price=pos.avg_price,
            exit_price=fill_price,
            entry_time=pos.entry_time,
            exit_time=now,
            gross_pnl=round(gross_pnl, 2),
            entry_fee=round(pos.entry_fee, 2),
            exit_fee=round(fee.total, 2),
            slippage_cost=round(fee.slippage + pos.avg_price * pos.quantity * self.slippage_bps / 10000, 2),
            net_pnl=round(net_pnl, 2),
            hold_duration=str(duration),
            signal_confidence=signal.confidence,
            signal_reasoning=signal.reasoning[:200],
        )
        self.trade_history.append(record)

        # Remove position
        del self.positions[signal.ticker]

        logger.info("[PAPER SELL] %s x%d @ %.2f | PnL: %.2f (gross) → %.2f (net) | Capital: %.2f",
                     signal.ticker, pos.quantity, fill_price, gross_pnl, net_pnl, self.capital)
        return record

    def take_snapshot(self, prices: dict[str, float] | None = None):
        """Record daily portfolio snapshot for tracking."""
        unrealized = 0.0
        if prices:
            for ticker, pos in self.positions.items():
                if ticker in prices:
                    unrealized += (prices[ticker] - pos.avg_price) * pos.quantity

        total_value = self.capital + unrealized
        drawdown = (self.peak_capital - total_value) / self.peak_capital * 100 if self.peak_capital > 0 else 0

        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "capital": round(self.capital, 2),
            "unrealized_pnl": round(unrealized, 2),
            "total_value": round(total_value, 2),
            "positions": len(self.positions),
            "total_trades": len(self.trade_history),
            "total_fees": round(self.total_fees, 2),
            "drawdown_pct": round(drawdown, 2),
        }
        self.daily_snapshots.append(snapshot)
        return snapshot

    def get_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        trades = self.trade_history
        if not trades:
            return f"Paper Trading Report — No trades executed. Capital: {self.capital:,.2f}"

        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]
        returns = np.array([t.net_pnl / self.initial_capital for t in trades])

        net_pnl = sum(t.net_pnl for t in trades)
        gross_pnl = sum(t.gross_pnl for t in trades)
        total_fees = sum(t.entry_fee + t.exit_fee for t in trades)
        total_slip = sum(t.slippage_cost for t in trades)

        win_rate = len(wins) / len(trades) * 100
        avg_win = np.mean([t.net_pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.net_pnl for t in losses]) if losses else 0
        profit_factor = sum(t.net_pnl for t in wins) / abs(sum(t.net_pnl for t in losses)) if losses else float("inf")

        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        drawdown = (self.peak_capital - self.capital) / self.peak_capital * 100

        lines = [
            f"Paper Trading Report — {self.broker.upper()}",
            "=" * 50,
            f"Initial Capital:   {self.initial_capital:>15,.2f}",
            f"Current Capital:   {self.capital:>15,.2f}",
            f"Net P&L:           {net_pnl:>+15,.2f} ({net_pnl/self.initial_capital*100:+.1f}%)",
            f"Gross P&L:         {gross_pnl:>+15,.2f}",
            f"Total Fees:        {total_fees:>15,.2f}",
            f"Total Slippage:    {total_slip:>15,.2f}",
            f"Fee Drag:          {total_fees/abs(gross_pnl)*100 if gross_pnl else 0:>14.1f}%",
            "",
            f"Total Trades:      {len(trades):>15}",
            f"Win Rate:          {win_rate:>14.1f}%",
            f"Avg Win:           {avg_win:>+15,.2f}",
            f"Avg Loss:          {avg_loss:>+15,.2f}",
            f"Profit Factor:     {profit_factor:>15.2f}",
            f"Sharpe Ratio:      {sharpe:>15.2f}",
            f"Max Drawdown:      {drawdown:>14.1f}%",
            f"Open Positions:    {len(self.positions):>15}",
        ]
        return "\n".join(lines)

    def save_trades(self, path: str | Path):
        """Save trade history to CSV."""
        if not self.trade_history:
            return
        records = []
        for t in self.trade_history:
            records.append({
                "id": t.id, "ticker": t.ticker, "side": t.side,
                "quantity": t.quantity, "entry_price": t.entry_price,
                "exit_price": t.exit_price, "entry_time": t.entry_time,
                "exit_time": t.exit_time, "gross_pnl": t.gross_pnl,
                "entry_fee": t.entry_fee, "exit_fee": t.exit_fee,
                "slippage": t.slippage_cost, "net_pnl": t.net_pnl,
                "hold_duration": t.hold_duration,
                "confidence": t.signal_confidence,
            })
        df = pd.DataFrame(records)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Trades saved: %s (%d records)", path, len(records))
