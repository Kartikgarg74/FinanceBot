"""Trading engine — orchestrates the full analysis-signal-trade loop with all integrations."""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from .base_trader import BaseTrader, Signal, SignalAction
from .analysis.technical import TechnicalAnalyzer
from .analysis.sentiment import NewsFetcher, SentimentScorer
from .analysis.signals import SignalGenerator
from .risk.manager import RiskManager
from .trade_logger import TradeLogger
from .market_hours import MarketHours
from .cache import get_cache, cached_history, set_cached_history, SENTIMENT_TTL
from .health import get_health_monitor, resilient_call

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Orchestrates the full trading loop for any market:
    1. Check market hours
    2. Fetch data (cached)
    3. Run technical analysis
    4. Fetch & score news (cached)
    5. Chart vision (on schedule)
    6. Generate combined signal
    7. Risk check
    8. Execute trade
    9. Persist to DB
    10. Notify via Telegram
    """

    def __init__(
        self,
        trader: BaseTrader,
        ai_router,
        config: dict,
        news_config: dict,
        telegram_bot=None,
    ):
        self.trader = trader
        self.ai = ai_router
        self.config = config
        self.telegram = telegram_bot

        self.technical = TechnicalAnalyzer(config)
        self.news_fetcher = NewsFetcher(news_config)
        self.sentiment_scorer = SentimentScorer(ai_router, news_config)
        self.signal_gen = SignalGenerator(config)
        self.risk_mgr = RiskManager(config)
        self.trade_logger = TradeLogger()
        self.market_name = config.get("_market_name", "unknown")
        self.market_hours = MarketHours(self.market_name, config)
        self.cache = get_cache()
        self.health = get_health_monitor()

        # Chart vision (lazy init)
        self._chart_analyzer = None
        self._last_chart_analysis: dict[str, dict] = {}  # ticker -> last analysis result

        # Register APIs for health monitoring
        self.health.register(f"{self.market_name}_data")
        self.health.register(f"{self.market_name}_orders")
        self.health.register("news_api")

    async def run_scan(self, tickers: list[str]) -> list[Signal]:
        """Run a full scan cycle on all tickers. Returns generated signals."""
        signals = []

        # 0. Check market hours
        if not self.market_hours.is_open():
            next_open = self.market_hours.next_open()
            logger.info("[%s] Market closed. Next open: %s", self.market_name.upper(), next_open)
            return signals

        # 1. Check if we can trade (with retry)
        max_retries = 3
        capital = None
        positions = None
        for attempt in range(max_retries):
            try:
                capital = await resilient_call(
                    f"{self.market_name}_data", self.trader.get_balance
                )
                positions = await resilient_call(
                    f"{self.market_name}_data", self.trader.get_positions
                )
                break  # Success
            except ConnectionError as e:
                if attempt < max_retries - 1:
                    backoff = 2 ** attempt
                    logger.warning("[%s] API failed (attempt %d/%d). Retrying in %ds...",
                                   self.market_name, attempt + 1, max_retries, backoff)
                    await asyncio.sleep(backoff)
                else:
                    logger.error("[%s] API down after %d retries: %s", self.market_name, max_retries, e)
                    if self.telegram:
                        await self.telegram.send_message(
                            f"[{self.market_name.upper()}] API down after {max_retries} retries — no trades executed"
                        )
                    return signals

        if capital is None:
            return signals

        self.risk_mgr.update_capital(capital)
        can_trade, reason = self.risk_mgr.can_trade(capital, positions)
        if not can_trade:
            logger.warning("[%s] Trading blocked: %s", self.market_name, reason)
            return signals

        # 2. Fetch news (shared across all tickers, cached)
        sentiment_map = self._get_sentiment(tickers)

        # 3. Analyze each ticker
        for ticker in tickers:
            try:
                signal = self._analyze_ticker(ticker, sentiment_map)
                if signal and signal.action != SignalAction.HOLD:
                    signals.append(signal)
            except Exception as e:
                logger.error("[%s] Error analyzing %s: %s", self.market_name, ticker, e)

        # 4. Execute actionable signals
        for signal in signals:
            await self._execute_signal(signal, capital, positions)

        # 5. Save daily stats
        self.trade_logger.save_daily_stats(self.market_name.upper(), self.trader.trades)

        # 6. Cleanup cache
        self.cache.cleanup()

        return signals

    def _analyze_ticker(self, ticker: str, sentiment_map: dict) -> Optional[Signal]:
        """Full analysis pipeline for one ticker."""
        # Get historical data (with caching)
        cache_key = f"{self.market_name}:{ticker}"
        df = cached_history(ticker, "1d")
        if df is None:
            df = self.trader.get_historical_data(ticker)
            if df is not None and not df.empty:
                set_cached_history(ticker, "1d", df)

        if df is None or df.empty:
            logger.debug("No data for %s, skipping", ticker)
            return None

        # Technical analysis
        tech_signal = self.technical.analyze(df, ticker)

        # Get sentiment for this ticker
        sentiment = sentiment_map.get(ticker)

        # Get last chart analysis (if available)
        chart_analysis = self._last_chart_analysis.get(ticker)

        # Generate combined signal
        signal = self.signal_gen.generate(
            ticker=ticker,
            tech_signal=tech_signal,
            sentiment=sentiment,
            chart_analysis=chart_analysis,
        )

        return signal

    def _get_sentiment(self, tickers: list[str]) -> dict:
        """Fetch and score news, return sentiment map."""
        # Check cache first
        cache_key = f"sentiment_map:{self.market_name}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        articles = []

        # Fetch from sources with health checking
        if self.health.can_call("news_api"):
            try:
                articles.extend(self.news_fetcher.fetch_finnhub_news("general"))
                articles.extend(self.news_fetcher.fetch_alpha_vantage_news(tickers[:5]))
                for ticker in tickers[:5]:
                    articles.extend(self.news_fetcher.fetch_finnhub_company_news(ticker))
                self.health.record_success("news_api")
            except Exception as e:
                self.health.record_failure("news_api", str(e))

        # Reddit sentiment
        reddit_config = self.config.get("news", {}).get("sources", {}).get("reddit", {})
        if reddit_config.get("enabled"):
            for sub in reddit_config.get("subreddits", [])[:2]:
                articles.extend(self.news_fetcher.fetch_reddit_posts(sub, limit=10))

        # Score articles
        if articles:
            articles = self.sentiment_scorer.score_batch(articles)
            # Persist scored news to DB
            for article in articles[:50]:  # Cap to avoid DB bloat
                self.trade_logger.save_news(
                    title=article.title, source=article.source, url=article.url,
                    score=article.sentiment_score, confidence=article.confidence,
                    category=article.category, tickers=article.tickers,
                    published_at=article.published_at,
                )

        # Aggregate per ticker
        asset_type = "crypto" if self.market_name == "binance" else "stock"
        result = self.sentiment_scorer.aggregate_sentiment(articles, asset_type)

        # Cache the result
        self.cache.set(cache_key, result, SENTIMENT_TTL)

        logger.info("[%s] Sentiment updated: %d articles, %d tickers scored",
                     self.market_name, len(articles), len(result))
        return result

    async def run_chart_analysis(self, tickers: list[str]) -> dict[str, dict]:
        """Run chart vision analysis (scheduled separately, less frequent)."""
        if self._chart_analyzer is None:
            from .analysis.chart_vision import ChartVisionAnalyzer
            self._chart_analyzer = ChartVisionAnalyzer(self.ai)

        results = {}
        for ticker in tickers:
            df = self.trader.get_historical_data(ticker, interval="1d", days=60)
            if df is not None and not df.empty:
                analysis = self._chart_analyzer.analyze_chart(df, ticker)
                if analysis:
                    results[ticker] = analysis
                    self._last_chart_analysis[ticker] = analysis

        if self.telegram and results:
            summary_lines = [f"[{self.market_name.upper()}] Chart Analysis"]
            for ticker, a in results.items():
                summary_lines.append(
                    f"  {ticker}: {a.get('recommendation', '?')} "
                    f"(conf={a.get('confidence', 0)}%, "
                    f"trend={a.get('trend', '?')}, "
                    f"patterns={a.get('patterns', [])})"
                )
            await self.telegram.send_message("\n".join(summary_lines))

        return results

    async def _execute_signal(self, signal: Signal, capital: float, positions: list):
        """Execute a signal after risk checks."""
        position_size = self.risk_mgr.calculate_position_size(signal, capital)

        if position_size <= 0:
            logger.info("[%s] %s signal for %s rejected by risk manager (size=0)",
                         self.market_name, signal.action.value, signal.ticker)
            return

        # Don't double up
        existing = [p for p in positions if p.ticker == signal.ticker]
        if existing and signal.action == SignalAction.BUY:
            logger.info("[%s] Already holding %s, skipping BUY signal", self.market_name, signal.ticker)
            return

        # Execute with health tracking
        try:
            trade = await resilient_call(
                f"{self.market_name}_orders",
                self.trader.execute_signal, signal, position_size,
            )
        except ConnectionError as e:
            logger.error("[%s] Order API down: %s", self.market_name, e)
            if self.telegram:
                await self.telegram.send_message(
                    f"[{self.market_name.upper()}] ORDER FAILED - API down\n"
                    f"Signal: {signal.action.value} {signal.ticker}\n"
                    f"Error: {e}"
                )
            return

        if trade and trade.status == "executed":
            self.risk_mgr.record_trade_result(trade.pnl or 0)

            # Persist to database
            self.trade_logger.save_trade(trade, signal, mode=self.trader.mode)

            # Notify via Telegram
            if self.telegram:
                mode_tag = "[PAPER]" if self.trader.mode in ("paper", "dry_run") else "[LIVE]"
                await self.telegram.send_message(
                    f"{mode_tag} {self.market_name.upper()}\n"
                    f"{trade.side} {trade.ticker} x{trade.quantity} @ {trade.price:,.2f}\n"
                    f"Reason: {signal.reasoning[:200]}\n"
                    f"SL: {signal.stop_loss or 'N/A'} | TP: {signal.take_profit or 'N/A'}\n"
                    f"Confidence: {signal.confidence:.0f}%"
                )

    async def get_portfolio_summary(self) -> str:
        """Generate portfolio summary for Telegram."""
        positions = self.trader.get_positions()
        capital = self.trader.get_balance()

        if not positions:
            return f"[{self.market_name.upper()}] No open positions. Capital: {capital:,.2f}"

        total_pnl = sum(p.pnl for p in positions)
        lines = [
            f"[{self.market_name.upper()}] Portfolio",
            f"Capital: {capital:,.2f}",
            f"Positions: {len(positions)}",
            f"Total P&L: {total_pnl:+,.2f}",
            "---",
        ]
        for p in positions:
            lines.append(
                f"  {p.ticker}: {p.quantity} @ {p.avg_price:.2f} "
                f"| Now: {p.current_price:.2f} | P&L: {p.pnl:+,.2f} ({p.pnl_pct:+.1f}%)"
            )
        return "\n".join(lines)

    async def get_daily_summary(self) -> dict:
        """Generate daily summary stats."""
        executed = [t for t in self.trader.trades if t.status == "executed"]
        today_trades = [t for t in executed if t.timestamp.date() == datetime.utcnow().date()]
        total_pnl = sum(t.pnl or 0 for t in today_trades)

        return {
            "market": self.market_name,
            "trades_today": len(today_trades),
            "total_pnl": total_pnl,
            "capital": self.trader.get_balance(),
            "positions": len(self.trader.get_positions()),
            "mode": self.trader.mode,
        }

    def get_health_status(self) -> str:
        """Get health status of all APIs."""
        return self.health.get_summary()
