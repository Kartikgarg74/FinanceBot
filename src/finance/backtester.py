"""Backtesting framework — replay historical data through the signal generator."""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from .analysis.technical import TechnicalAnalyzer
from .analysis.signals import SignalGenerator
from .base_trader import SignalAction

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    ticker: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    pnl_pct: float
    max_drawdown_pct: float
    win_rate: float
    avg_win: float
    avg_loss: float
    sharpe_ratio: float
    profit_factor: float
    trades: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Backtest: {self.ticker}\n"
            f"Period: {self.start_date} to {self.end_date}\n"
            f"{'=' * 30}\n"
            f"Initial Capital: {self.initial_capital:,.2f}\n"
            f"Final Capital:   {self.final_capital:,.2f}\n"
            f"Total P&L:       {self.total_pnl:+,.2f} ({self.pnl_pct:+.1f}%)\n"
            f"Total Trades:    {self.total_trades}\n"
            f"Win Rate:        {self.win_rate:.1f}%\n"
            f"Avg Win:         {self.avg_win:+,.2f}\n"
            f"Avg Loss:        {self.avg_loss:+,.2f}\n"
            f"Max Drawdown:    {self.max_drawdown_pct:.1f}%\n"
            f"Sharpe Ratio:    {self.sharpe_ratio:.2f}\n"
            f"Profit Factor:   {self.profit_factor:.2f}"
        )


class Backtester:
    """Replays historical data through technical analysis and signal generation."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.technical = TechnicalAnalyzer(config)
        self.signal_gen = SignalGenerator(config)

    def run(
        self,
        df: pd.DataFrame,
        ticker: str,
        initial_capital: float = 100000,
        risk_per_trade_pct: float = 2.0,
        lookback: int = 50,  # Minimum candles before first signal
    ) -> BacktestResult:
        """
        Run a backtest on historical OHLCV data.
        Slides a window through the data, generates signals, simulates trades.
        """
        if df is None or len(df) < lookback + 10:
            return BacktestResult(
                ticker=ticker, start_date="N/A", end_date="N/A",
                initial_capital=initial_capital, final_capital=initial_capital,
                total_trades=0, winning_trades=0, losing_trades=0,
                total_pnl=0, pnl_pct=0, max_drawdown_pct=0, win_rate=0,
                avg_win=0, avg_loss=0, sharpe_ratio=0, profit_factor=0,
            )

        capital = initial_capital
        peak_capital = initial_capital
        max_drawdown = 0
        position = None  # (entry_price, quantity, entry_idx)
        trades = []
        daily_returns = []

        for i in range(lookback, len(df)):
            window = df.iloc[:i + 1]
            price = float(window["Close"].iloc[-1])

            # Generate signal on the window
            tech = self.technical.analyze(window, ticker)
            signal = self.signal_gen.generate(ticker=ticker, tech_signal=tech)

            # Position management
            if position is None and signal.action == SignalAction.BUY:
                # Enter position
                risk_amount = capital * (risk_per_trade_pct / 100)
                stop_distance = price * 0.03  # 3% default stop
                qty = int(risk_amount / stop_distance) if stop_distance > 0 else 0
                if qty > 0 and qty * price <= capital:
                    position = (price, qty, i)

            elif position is not None and signal.action == SignalAction.SELL:
                # Exit position
                entry_price, qty, entry_idx = position
                pnl = (price - entry_price) * qty
                capital += pnl
                daily_returns.append(pnl / initial_capital)

                trades.append({
                    "entry_price": entry_price,
                    "exit_price": price,
                    "quantity": qty,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round((price / entry_price - 1) * 100, 2),
                    "hold_bars": i - entry_idx,
                })
                position = None

            # Track drawdown
            if capital > peak_capital:
                peak_capital = capital
            dd = (peak_capital - capital) / peak_capital * 100
            if dd > max_drawdown:
                max_drawdown = dd

        # Close any open position at last price
        if position is not None:
            entry_price, qty, entry_idx = position
            last_price = float(df["Close"].iloc[-1])
            pnl = (last_price - entry_price) * qty
            capital += pnl
            trades.append({
                "entry_price": entry_price, "exit_price": last_price,
                "quantity": qty, "pnl": round(pnl, 2),
                "pnl_pct": round((last_price / entry_price - 1) * 100, 2),
                "hold_bars": len(df) - 1 - entry_idx,
            })

        # Compute stats
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]

        total_pnl = capital - initial_capital
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe ratio (annualized, assuming daily returns)
        import numpy as np
        if daily_returns:
            returns_array = np.array(daily_returns)
            sharpe = (returns_array.mean() / returns_array.std() * np.sqrt(252)) if returns_array.std() > 0 else 0
        else:
            sharpe = 0

        start_date = str(df.index[lookback].date()) if hasattr(df.index[lookback], "date") else str(df.index[lookback])
        end_date = str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])

        return BacktestResult(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=round(capital, 2),
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            total_pnl=round(total_pnl, 2),
            pnl_pct=round(total_pnl / initial_capital * 100, 2),
            max_drawdown_pct=round(max_drawdown, 2),
            win_rate=round(win_rate, 1),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            sharpe_ratio=round(sharpe, 2),
            profit_factor=round(profit_factor, 2),
            trades=trades,
        )
