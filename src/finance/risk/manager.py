"""Risk management system — position sizing, circuit breakers, daily limits."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date

from ..base_trader import Signal, SignalAction, Position

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    max_risk_per_trade_pct: float = 2.0
    max_position_size_pct: float = 10.0
    max_daily_loss_pct: float = 5.0
    max_trades_per_day: int = 20
    max_drawdown_pct: float = 15.0
    max_correlated_positions: int = 3
    always_use_stop_loss: bool = True
    min_risk_reward_ratio: float = 1.5
    consecutive_loss_limit: int = 5
    pause_duration_hours: int = 24


@dataclass
class RiskState:
    """Tracks current risk state for circuit breaker logic."""
    daily_trades: int = 0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    peak_capital: float = 0.0
    current_capital: float = 0.0
    is_paused: bool = False
    pause_until: datetime | None = None
    last_trade_date: date | None = None

    def reset_daily(self):
        self.daily_trades = 0
        self.daily_pnl = 0.0

    def check_new_day(self):
        today = date.today()
        if self.last_trade_date != today:
            self.reset_daily()
            self.last_trade_date = today


class RiskManager:
    """Enforces risk rules and calculates position sizes."""

    def __init__(self, config: dict):
        trading = config.get("trading", {})
        risk_config = config.get("risk_management", {})

        self.limits = RiskLimits(
            max_risk_per_trade_pct=risk_config.get("max_risk_per_trade_pct", 2.0),
            max_position_size_pct=risk_config.get("max_position_size_pct", 10.0),
            max_daily_loss_pct=risk_config.get("max_daily_loss_pct", 5.0),
            max_trades_per_day=risk_config.get("max_trades_per_day", 20),
            max_drawdown_pct=risk_config.get("max_drawdown_pct", 15.0),
            consecutive_loss_limit=risk_config.get("consecutive_loss_limit", 5),
        )
        self.max_capital_per_trade = trading.get("max_capital_per_trade", 50000)
        self.max_daily_loss = trading.get("max_daily_loss", 10000)
        self.max_positions = trading.get("max_positions", 10)

        self.state = RiskState()

    def can_trade(self, capital: float, positions: list[Position]) -> tuple[bool, str]:
        """Check if trading is allowed based on current risk state."""
        self.state.check_new_day()

        # Check pause (circuit breaker)
        if self.state.is_paused:
            if self.state.pause_until and datetime.utcnow() < self.state.pause_until:
                return False, f"Trading paused until {self.state.pause_until}"
            else:
                self.state.is_paused = False
                self.state.consecutive_losses = 0
                logger.info("Circuit breaker reset, trading resumed")

        # Check daily trade limit
        if self.state.daily_trades >= self.limits.max_trades_per_day:
            return False, f"Daily trade limit reached ({self.limits.max_trades_per_day})"

        # Check daily loss limit
        if abs(self.state.daily_pnl) > self.max_daily_loss:
            return False, f"Daily loss limit reached (₹{self.state.daily_pnl:.2f})"

        daily_loss_pct = abs(self.state.daily_pnl) / capital * 100 if capital > 0 else 0
        if daily_loss_pct > self.limits.max_daily_loss_pct:
            return False, f"Daily loss % limit reached ({daily_loss_pct:.1f}%)"

        # Check max positions
        if len(positions) >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"

        # Check max drawdown
        if self.state.peak_capital > 0:
            drawdown = (self.state.peak_capital - capital) / self.state.peak_capital * 100
            if drawdown > self.limits.max_drawdown_pct:
                self._trigger_circuit_breaker("Max drawdown exceeded")
                return False, f"Max drawdown reached ({drawdown:.1f}%)"

        # Check consecutive losses
        if self.state.consecutive_losses >= self.limits.consecutive_loss_limit:
            self._trigger_circuit_breaker("Consecutive loss limit")
            return False, f"Consecutive losses ({self.state.consecutive_losses})"

        return True, "OK"

    def calculate_position_size(
        self,
        signal: Signal,
        capital: float,
    ) -> int | float:
        """
        Calculate position size using risk-based sizing.
        Risk only max_risk_per_trade_pct of capital per trade.
        """
        if signal.action == SignalAction.HOLD:
            return 0

        # Risk amount = percentage of capital
        risk_pct = self.limits.max_risk_per_trade_pct / 100
        risk_amount = capital * risk_pct

        # Cap by max capital per trade
        risk_amount = min(risk_amount, self.max_capital_per_trade)

        # Calculate risk per share (distance to stop loss)
        if signal.stop_loss and signal.price > 0:
            risk_per_share = abs(signal.price - signal.stop_loss)
        else:
            # Default: use 3% of price as risk per share
            risk_per_share = signal.price * 0.03

        if risk_per_share == 0:
            return 0

        position_size = int(risk_amount / risk_per_share)

        # Cap by max position size % of portfolio
        max_position_value = capital * (self.limits.max_position_size_pct / 100)
        max_shares = int(max_position_value / signal.price) if signal.price > 0 else 0
        position_size = min(position_size, max_shares)

        # Ensure at least 1 share if we can afford it
        if position_size == 0 and signal.price <= capital:
            position_size = 1

        # Check risk-reward ratio if take_profit is set
        if signal.take_profit and signal.stop_loss and self.limits.min_risk_reward_ratio > 0:
            reward = abs(signal.take_profit - signal.price)
            risk = abs(signal.price - signal.stop_loss)
            if risk > 0:
                rr_ratio = reward / risk
                if rr_ratio < self.limits.min_risk_reward_ratio:
                    logger.info(
                        "Skipping %s: R:R ratio %.2f < %.2f minimum",
                        signal.ticker, rr_ratio, self.limits.min_risk_reward_ratio,
                    )
                    return 0

        # Final sanity check via security guardrail
        from src.utils.security import validate_trade_amount
        position_size = validate_trade_amount(
            position_size, signal.price, capital, self.limits.max_position_size_pct,
        )

        return position_size

    def record_trade_result(self, pnl: float):
        """Record trade result for circuit breaker tracking."""
        self.state.daily_trades += 1
        self.state.daily_pnl += pnl

        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

    def update_capital(self, capital: float):
        """Update capital tracking for drawdown calculation."""
        self.state.current_capital = capital
        if capital > self.state.peak_capital:
            self.state.peak_capital = capital

    def _trigger_circuit_breaker(self, reason: str):
        """Pause all trading."""
        from datetime import timedelta
        self.state.is_paused = True
        self.state.pause_until = datetime.utcnow() + timedelta(hours=self.limits.pause_duration_hours)
        logger.warning("CIRCUIT BREAKER triggered: %s. Paused until %s", reason, self.state.pause_until)

    def get_risk_summary(self, capital: float, positions: list[Position]) -> str:
        """Human-readable risk summary for Telegram."""
        drawdown = 0
        if self.state.peak_capital > 0:
            drawdown = (self.state.peak_capital - capital) / self.state.peak_capital * 100

        return (
            f"Risk Summary\n"
            f"{'=' * 24}\n"
            f"Capital: {capital:,.2f}\n"
            f"Peak: {self.state.peak_capital:,.2f}\n"
            f"Drawdown: {drawdown:.1f}% (max {self.limits.max_drawdown_pct}%)\n"
            f"Daily P&L: {self.state.daily_pnl:+,.2f}\n"
            f"Daily Trades: {self.state.daily_trades}/{self.limits.max_trades_per_day}\n"
            f"Consecutive Losses: {self.state.consecutive_losses}/{self.limits.consecutive_loss_limit}\n"
            f"Positions: {len(positions)}/{self.max_positions}\n"
            f"Status: {'PAUSED' if self.state.is_paused else 'ACTIVE'}"
        )
