"""Finance-specific Telegram command handlers — wired into the main Telegram bot."""

import logging

logger = logging.getLogger(__name__)


class FinanceTelegramCommands:
    """
    Provides finance command handlers to register with the existing TelegramBot.
    Call register(telegram_bot, engines) to add all finance commands.
    """

    def __init__(self, engines: dict, trade_logger=None):
        """
        engines: dict of {market_name: TradingEngine}
        """
        self.engines = engines
        self.trade_logger = trade_logger

    def register(self, bot) -> None:
        """Register all finance command handlers with the Telegram bot."""
        from telegram.ext import CommandHandler

        commands = {
            "portfolio": self._cmd_portfolio,
            "zerodha": self._cmd_zerodha,
            "crypto": self._cmd_crypto,
            "alpaca": self._cmd_alpaca,
            "trades": self._cmd_trades,
            "signals": self._cmd_signals,
            "risk": self._cmd_risk,
            "news": self._cmd_news,
            "health": self._cmd_health,
            "backtest": self._cmd_backtest,
            "markets": self._cmd_markets,
        }

        for cmd, handler in commands.items():
            bot.app.add_handler(CommandHandler(cmd, handler))

        logger.info("Finance Telegram commands registered: %s", ", ".join(f"/{c}" for c in commands))

    async def _cmd_portfolio(self, update, context):
        """Show portfolio for all markets."""
        lines = []
        for name, engine in self.engines.items():
            summary = await engine.get_portfolio_summary()
            lines.append(summary)
        await update.message.reply_text("\n\n".join(lines) if lines else "No engines running")

    async def _cmd_zerodha(self, update, context):
        engine = self.engines.get("zerodha")
        if not engine:
            await update.message.reply_text("Zerodha module not active")
            return
        summary = await engine.get_portfolio_summary()
        await update.message.reply_text(summary)

    async def _cmd_crypto(self, update, context):
        engine = self.engines.get("binance")
        if not engine:
            await update.message.reply_text("Binance module not active")
            return
        summary = await engine.get_portfolio_summary()
        await update.message.reply_text(summary)

    async def _cmd_alpaca(self, update, context):
        engine = self.engines.get("alpaca")
        if not engine:
            await update.message.reply_text("Alpaca module not active")
            return
        summary = await engine.get_portfolio_summary()
        await update.message.reply_text(summary)

    async def _cmd_trades(self, update, context):
        """Show recent trades from database."""
        if not self.trade_logger:
            await update.message.reply_text("Trade logger not available")
            return

        # Parse optional exchange filter
        exchange = None
        if context.args:
            exchange = context.args[0].upper()

        trades = self.trade_logger.get_recent_trades(exchange=exchange, limit=10)
        if not trades:
            await update.message.reply_text("No trades recorded yet")
            return

        lines = ["Recent Trades", "=" * 28]
        for t in trades:
            pnl_str = f" P&L:{t['pnl']:+.2f}" if t['pnl'] else ""
            lines.append(
                f"[{t['mode']}] {t['side']} {t['ticker']} x{t['qty']} @ {t['price']:.2f}{pnl_str} ({t['time']})"
            )
        await update.message.reply_text("\n".join(lines))

    async def _cmd_signals(self, update, context):
        """Run a quick scan and show current signals."""
        lines = ["Current Signals", "=" * 28]
        for name, engine in self.engines.items():
            if not engine.market_hours.is_open():
                lines.append(f"[{name.upper()}] Market closed")
                continue

            tickers = engine.config.get("watchlist", engine.config.get("pairs", []))[:5]
            for ticker in tickers:
                df = engine.trader.get_historical_data(ticker)
                if df is not None and not df.empty:
                    tech = engine.technical.analyze(df, ticker)
                    lines.append(f"  {ticker}: {tech.action} (conf={tech.confidence:.0f}%)")

        await update.message.reply_text("\n".join(lines))

    async def _cmd_risk(self, update, context):
        """Show risk status for all markets."""
        lines = []
        for name, engine in self.engines.items():
            capital = engine.trader.get_balance()
            positions = engine.trader.get_positions()
            risk_text = engine.risk_mgr.get_risk_summary(capital, positions)
            lines.append(f"[{name.upper()}]\n{risk_text}")
        await update.message.reply_text("\n\n".join(lines) if lines else "No engines running")

    async def _cmd_news(self, update, context):
        """Show latest scored news."""
        if not self.trade_logger:
            await update.message.reply_text("Trade logger not available")
            return

        from src.database.models import NewsLog, get_session
        try:
            session = get_session()
            news = session.query(NewsLog).order_by(NewsLog.scored_at.desc()).limit(10).all()
            session.close()

            if not news:
                await update.message.reply_text("No scored news yet")
                return

            lines = ["Latest News Sentiment", "=" * 28]
            for n in news:
                score_indicator = "+" if n.sentiment_score > 0 else ""
                lines.append(f"  [{score_indicator}{n.sentiment_score:.1f}] {n.title[:80]}")
            await update.message.reply_text("\n".join(lines))
        except Exception as e:
            await update.message.reply_text(f"Error fetching news: {e}")

    async def _cmd_health(self, update, context):
        """Show API health status."""
        lines = []
        for name, engine in self.engines.items():
            lines.append(engine.get_health_status())
        cache = self.engines[list(self.engines.keys())[0]].cache if self.engines else None
        if cache:
            lines.append(f"\nCache: {cache.stats}")
        await update.message.reply_text("\n\n".join(lines) if lines else "No engines running")

    async def _cmd_backtest(self, update, context):
        """Run a quick backtest. Usage: /backtest RELIANCE 365"""
        if not context.args:
            await update.message.reply_text("Usage: /backtest <TICKER> [days]\nExample: /backtest RELIANCE 365")
            return

        ticker = context.args[0].upper()
        days = int(context.args[1]) if len(context.args) > 1 else 365

        await update.message.reply_text(f"Running backtest for {ticker} ({days} days)...")

        # Find an engine that can fetch this ticker
        engine = None
        for name, eng in self.engines.items():
            engine = eng
            break

        if not engine:
            await update.message.reply_text("No engines available")
            return

        from .backtester import Backtester
        from .analysis.data_fetcher import DataFetcher

        # Try Indian stock first, then US, then crypto
        fetcher = DataFetcher()
        df = fetcher.fetch_indian_stock(ticker, "1d", days)
        if df is None or df.empty:
            df = fetcher.fetch_us_stock(ticker, "1d", days)
        if df is None or df.empty:
            df = fetcher.fetch_crypto_yfinance(ticker, "1d", days)

        if df is None or df.empty:
            await update.message.reply_text(f"No data found for {ticker}")
            return

        bt = Backtester(engine.config)
        result = bt.run(df, ticker, initial_capital=100000)
        await update.message.reply_text(result.summary())

    async def _cmd_markets(self, update, context):
        """Show status of all configured markets."""
        lines = ["Market Status", "=" * 28]
        for name, engine in self.engines.items():
            is_open = engine.market_hours.is_open()
            status = "OPEN" if is_open else f"CLOSED (next: {engine.market_hours.next_open()})"
            mode = engine.trader.mode
            lines.append(f"  {name.upper()}: {status} | Mode: {mode}")
        await update.message.reply_text("\n".join(lines))
