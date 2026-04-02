"""Backtesting framework — replay historical data through signal generators.

Supports both rule-based and ML-based signal generation with realistic
transaction cost modeling.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis.technical import TechnicalAnalyzer
from .analysis.signals import SignalGenerator
from .base_trader import SignalAction
from .cost_model import get_cost_model

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
    # Cost breakdown
    total_fees: float = 0.0
    total_slippage: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    trades: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Backtest: {self.ticker}\n"
            f"Period: {self.start_date} to {self.end_date}\n"
            f"{'=' * 40}\n"
            f"Initial Capital: {self.initial_capital:,.2f}\n"
            f"Final Capital:   {self.final_capital:,.2f}\n"
            f"Gross P&L:       {self.gross_pnl:+,.2f}\n"
            f"Total Fees:      {self.total_fees:,.2f}\n"
            f"Total Slippage:  {self.total_slippage:,.2f}\n"
            f"Net P&L:         {self.net_pnl:+,.2f} ({self.pnl_pct:+.1f}%)\n"
            f"{'=' * 40}\n"
            f"Total Trades:    {self.total_trades}\n"
            f"Win Rate:        {self.win_rate:.1f}%\n"
            f"Avg Win:         {self.avg_win:+,.2f}\n"
            f"Avg Loss:        {self.avg_loss:+,.2f}\n"
            f"Max Drawdown:    {self.max_drawdown_pct:.1f}%\n"
            f"Sharpe Ratio:    {self.sharpe_ratio:.2f}\n"
            f"Sortino Ratio:   {self.sortino_ratio:.2f}\n"
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
        lookback: int = 50,
        broker: str | None = None,
        trade_type: str = "intraday",
        slippage_bps: float = 10,
    ) -> BacktestResult:
        """
        Run a backtest on historical OHLCV data with realistic cost modeling.

        broker: 'zerodha', 'alpaca', or 'binance' — enables cost deduction
        """
        if df is None or len(df) < lookback + 10:
            return self._empty_result(ticker, initial_capital)

        # Setup cost model
        cost_model = None
        if broker:
            cost_model = get_cost_model(broker, trade_type=trade_type, slippage_bps=slippage_bps)

        capital = initial_capital
        peak_capital = initial_capital
        max_drawdown = 0
        position = None  # (entry_price, quantity, entry_idx, entry_cost)
        trades = []
        daily_returns = []
        total_fees = 0.0
        total_slippage = 0.0

        for i in range(lookback, len(df)):
            window = df.iloc[:i + 1]
            price = float(window["Close"].iloc[-1])

            # Generate signal on the window
            tech = self.technical.analyze(window, ticker)
            signal = self.signal_gen.generate(ticker=ticker, tech_signal=tech)

            # Position management
            if position is None and signal.action == SignalAction.BUY:
                # Calculate position size
                risk_amount = capital * (risk_per_trade_pct / 100)
                stop_distance = price * 0.03
                qty = int(risk_amount / stop_distance) if stop_distance > 0 else 0
                if qty > 0 and qty * price <= capital:
                    # Apply entry costs
                    trade_value = qty * price
                    entry_fee = 0.0
                    entry_slip = 0.0
                    if cost_model:
                        cost = cost_model.calculate(trade_value, "buy")
                        entry_fee = cost.total - cost.slippage
                        entry_slip = cost.slippage
                        # Adjust entry price for slippage (worse fill)
                        price += price * (slippage_bps / 10000)

                    capital -= entry_fee
                    total_fees += entry_fee
                    total_slippage += entry_slip
                    position = (price, qty, i, entry_fee)

            elif position is not None and signal.action == SignalAction.SELL:
                entry_price, qty, entry_idx, entry_fee = position
                trade_value = qty * price

                # Apply exit costs
                exit_fee = 0.0
                exit_slip = 0.0
                if cost_model:
                    cost = cost_model.calculate(trade_value, "sell")
                    exit_fee = cost.total - cost.slippage
                    exit_slip = cost.slippage
                    # Adjust exit price for slippage (worse fill)
                    price -= price * (slippage_bps / 10000)

                gross_pnl = (price - entry_price) * qty
                net_pnl = gross_pnl - exit_fee
                capital += net_pnl
                total_fees += exit_fee
                total_slippage += exit_slip
                daily_returns.append(net_pnl / initial_capital)

                trades.append({
                    "entry_price": entry_price,
                    "exit_price": price,
                    "quantity": qty,
                    "gross_pnl": round(gross_pnl, 2),
                    "fees": round(entry_fee + exit_fee, 2),
                    "net_pnl": round(net_pnl, 2),
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
            entry_price, qty, entry_idx, entry_fee = position
            last_price = float(df["Close"].iloc[-1])
            trade_value = qty * last_price

            exit_fee = 0.0
            if cost_model:
                cost = cost_model.calculate(trade_value, "sell")
                exit_fee = cost.total
                total_fees += exit_fee

            gross_pnl = (last_price - entry_price) * qty
            net_pnl = gross_pnl - exit_fee
            capital += net_pnl

            trades.append({
                "entry_price": entry_price, "exit_price": last_price,
                "quantity": qty, "gross_pnl": round(gross_pnl, 2),
                "fees": round(entry_fee + exit_fee, 2),
                "net_pnl": round(net_pnl, 2),
                "pnl_pct": round((last_price / entry_price - 1) * 100, 2),
                "hold_bars": len(df) - 1 - entry_idx,
            })

        # Compute stats
        wins = [t for t in trades if t["net_pnl"] > 0]
        losses = [t for t in trades if t["net_pnl"] <= 0]

        gross_total = sum(t["gross_pnl"] for t in trades)
        net_total = capital - initial_capital
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = sum(t["net_pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["net_pnl"] for t in losses) / len(losses) if losses else 0
        gross_profit = sum(t["net_pnl"] for t in wins)
        gross_loss = abs(sum(t["net_pnl"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe, Sortino, Calmar
        sharpe = sortino = calmar = 0.0
        if daily_returns:
            returns_array = np.array(daily_returns)
            if returns_array.std() > 0:
                sharpe = returns_array.mean() / returns_array.std() * np.sqrt(252)
            # Sortino
            downside = returns_array[returns_array < 0]
            if len(downside) > 0 and downside.std() > 0:
                sortino = returns_array.mean() / downside.std() * np.sqrt(252)
            # Calmar
            annual_return = returns_array.mean() * 252
            if max_drawdown > 0:
                calmar = annual_return / (max_drawdown / 100)

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
            total_pnl=round(net_total, 2),
            pnl_pct=round(net_total / initial_capital * 100, 2),
            max_drawdown_pct=round(max_drawdown, 2),
            win_rate=round(win_rate, 1),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            sharpe_ratio=round(sharpe, 2),
            profit_factor=round(profit_factor, 2),
            total_fees=round(total_fees, 2),
            total_slippage=round(total_slippage, 2),
            gross_pnl=round(gross_total, 2),
            net_pnl=round(net_total, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            trades=trades,
        )

    def run_ml(
        self,
        df: pd.DataFrame,
        ticker: str,
        model_path: str | Path,
        initial_capital: float = 100000,
        risk_per_trade_pct: float = 2.0,
        lookback: int = 60,
        broker: str | None = None,
        trade_type: str = "intraday",
        slippage_bps: float = 10,
    ) -> BacktestResult:
        """
        Run a backtest using an ML model for signal generation.
        """
        from ..ml.signal_generator import MLSignalGenerator

        if df is None or len(df) < lookback + 10:
            return self._empty_result(ticker, initial_capital)

        ml_gen = MLSignalGenerator(model_path)
        cost_model = get_cost_model(broker, trade_type=trade_type, slippage_bps=slippage_bps) if broker else None

        capital = initial_capital
        peak_capital = initial_capital
        max_drawdown = 0
        position = None
        trades = []
        daily_returns = []
        total_fees = 0.0
        total_slippage = 0.0

        for i in range(lookback, len(df)):
            window = df.iloc[:i + 1]
            price = float(window["Close"].iloc[-1])

            signal = ml_gen.generate(window, ticker)

            if position is None and signal.action == SignalAction.BUY:
                risk_amount = capital * (risk_per_trade_pct / 100)
                stop_distance = abs(price - signal.stop_loss) if signal.stop_loss else price * 0.03
                qty = int(risk_amount / stop_distance) if stop_distance > 0 else 0

                if qty > 0 and qty * price <= capital:
                    trade_value = qty * price
                    entry_fee = 0.0
                    if cost_model:
                        cost = cost_model.calculate(trade_value, "buy")
                        entry_fee = cost.total
                        price += price * (slippage_bps / 10000)

                    capital -= entry_fee
                    total_fees += entry_fee
                    position = (price, qty, i, entry_fee)

            elif position is not None and signal.action == SignalAction.SELL:
                entry_price, qty, entry_idx, entry_fee = position
                trade_value = qty * price

                exit_fee = 0.0
                if cost_model:
                    cost = cost_model.calculate(trade_value, "sell")
                    exit_fee = cost.total
                    price -= price * (slippage_bps / 10000)

                gross_pnl = (price - entry_price) * qty
                net_pnl = gross_pnl - exit_fee
                capital += net_pnl
                total_fees += exit_fee
                daily_returns.append(net_pnl / initial_capital)

                trades.append({
                    "entry_price": entry_price, "exit_price": price,
                    "quantity": qty, "gross_pnl": round(gross_pnl, 2),
                    "fees": round(entry_fee + exit_fee, 2),
                    "net_pnl": round(net_pnl, 2),
                    "pnl_pct": round((price / entry_price - 1) * 100, 2),
                    "hold_bars": i - entry_idx,
                })
                position = None

            if capital > peak_capital:
                peak_capital = capital
            dd = (peak_capital - capital) / peak_capital * 100
            if dd > max_drawdown:
                max_drawdown = dd

        # Close open position
        if position is not None:
            entry_price, qty, entry_idx, entry_fee = position
            last_price = float(df["Close"].iloc[-1])
            exit_fee = cost_model.calculate(qty * last_price, "sell").total if cost_model else 0
            total_fees += exit_fee
            gross_pnl = (last_price - entry_price) * qty
            net_pnl = gross_pnl - exit_fee
            capital += net_pnl
            trades.append({
                "entry_price": entry_price, "exit_price": last_price,
                "quantity": qty, "gross_pnl": round(gross_pnl, 2),
                "fees": round(entry_fee + exit_fee, 2),
                "net_pnl": round(net_pnl, 2),
                "pnl_pct": round((last_price / entry_price - 1) * 100, 2),
                "hold_bars": len(df) - 1 - entry_idx,
            })

        # Stats
        wins = [t for t in trades if t["net_pnl"] > 0]
        losses = [t for t in trades if t["net_pnl"] <= 0]
        gross_total = sum(t["gross_pnl"] for t in trades)
        net_total = capital - initial_capital
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = sum(t["net_pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["net_pnl"] for t in losses) / len(losses) if losses else 0
        gp = sum(t["net_pnl"] for t in wins)
        gl = abs(sum(t["net_pnl"] for t in losses))
        pf = gp / gl if gl > 0 else float("inf")

        sharpe = sortino = calmar = 0.0
        if daily_returns:
            ra = np.array(daily_returns)
            if ra.std() > 0:
                sharpe = ra.mean() / ra.std() * np.sqrt(252)
            down = ra[ra < 0]
            if len(down) > 0 and down.std() > 0:
                sortino = ra.mean() / down.std() * np.sqrt(252)
            ann_ret = ra.mean() * 252
            if max_drawdown > 0:
                calmar = ann_ret / (max_drawdown / 100)

        start_date = str(df.index[lookback].date()) if hasattr(df.index[lookback], "date") else str(df.index[lookback])
        end_date = str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])

        return BacktestResult(
            ticker=ticker, start_date=start_date, end_date=end_date,
            initial_capital=initial_capital, final_capital=round(capital, 2),
            total_trades=len(trades), winning_trades=len(wins), losing_trades=len(losses),
            total_pnl=round(net_total, 2), pnl_pct=round(net_total / initial_capital * 100, 2),
            max_drawdown_pct=round(max_drawdown, 2), win_rate=round(win_rate, 1),
            avg_win=round(avg_win, 2), avg_loss=round(avg_loss, 2),
            sharpe_ratio=round(sharpe, 2), profit_factor=round(pf, 2),
            total_fees=round(total_fees, 2), total_slippage=round(total_slippage, 2),
            gross_pnl=round(gross_total, 2), net_pnl=round(net_total, 2),
            sortino_ratio=round(sortino, 2), calmar_ratio=round(calmar, 2),
            trades=trades,
        )

    def _empty_result(self, ticker, capital):
        return BacktestResult(
            ticker=ticker, start_date="N/A", end_date="N/A",
            initial_capital=capital, final_capital=capital,
            total_trades=0, winning_trades=0, losing_trades=0,
            total_pnl=0, pnl_pct=0, max_drawdown_pct=0, win_rate=0,
            avg_win=0, avg_loss=0, sharpe_ratio=0, profit_factor=0,
        )
