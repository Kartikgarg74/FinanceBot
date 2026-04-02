"""FinanceBot — AI-Powered Trading Bot — Main Entry Point."""

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logging, get_logger
from src.utils.config import ConfigLoader
from src.database.models import init_db
from src.ai.router import AIRouter
from src.notifications.telegram_bot import TelegramBot

logger = get_logger("financebot")


def parse_args():
    parser = argparse.ArgumentParser(description="FinanceBot — AI-Powered Trading Bot")
    parser.add_argument("--market", default=None,
                        choices=["zerodha", "alpaca", "binance", "all"],
                        help="Finance market to trade (default: all configured)")
    parser.add_argument("--user", default=os.environ.get("FINANCEBOT_USER", "kartik"),
                        help="User profile to load (default: $FINANCEBOT_USER or kartik)")
    parser.add_argument("--mode", default=None,
                        choices=["paper", "live", "dry_run"],
                        help="Mode override: paper/live/dry_run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Alias for --mode paper")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Log level (default: INFO)")
    parser.add_argument("--scan-once", action="store_true",
                        help="Run a single scan cycle and exit (for testing)")
    return parser.parse_args()


async def run_finance(args, config_loader, ai_router, telegram_bot):
    """Run the finance trading bot."""
    from src.finance.engine import TradingEngine

    news_config = config_loader.load_news_config()
    engines = []

    markets = []
    if args.market == "all" or args.market is None:
        for m in ["zerodha", "binance", "alpaca"]:
            try:
                cfg = config_loader.load_finance_config(m)
                markets.append((m, cfg))
            except FileNotFoundError:
                pass
    else:
        cfg = config_loader.load_finance_config(args.market)
        markets.append((args.market, cfg))

    if not markets:
        logger.error("No finance markets configured. Create config/zerodha.yaml, config/binance.yaml, or config/alpaca.yaml")
        return

    for market_name, market_config in markets:
        mode = args.mode or ("paper" if args.dry_run else market_config.get("trading", {}).get("mode", "paper"))
        market_config["_market_name"] = market_name

        if market_name == "zerodha":
            from src.finance.zerodha.client import ZerodhaClient
            trader = ZerodhaClient(market_config, mode=mode)
        elif market_name == "alpaca":
            from src.finance.alpaca.client import AlpacaClient
            trader = AlpacaClient(market_config, mode=mode)
        elif market_name == "binance":
            from src.finance.binance.client import BinanceClient
            trader = BinanceClient(market_config, mode=mode)
        else:
            logger.warning("Unknown market: %s", market_name)
            continue

        if not trader.connect():
            logger.error("Failed to connect to %s — skipping", market_name)
            continue

        engine = TradingEngine(
            trader=trader,
            ai_router=ai_router,
            config=market_config,
            news_config=news_config,
            telegram_bot=telegram_bot,
        )
        engines.append((market_name, engine, market_config, trader))
        logger.info("[%s] Engine ready (mode=%s)", market_name.upper(), mode)

    if not engines:
        logger.error("No trading engines could be started")
        return

    if telegram_bot:
        from src.finance.telegram_commands import FinanceTelegramCommands
        from src.finance.trade_logger import TradeLogger
        engine_map = {n: e for n, e, _, _ in engines}
        finance_cmds = FinanceTelegramCommands(engine_map, trade_logger=TradeLogger())
        finance_cmds.register(telegram_bot)
        logger.info("Finance Telegram commands registered")

    for name, engine, cfg, trader in engines:
        if name == "binance" and hasattr(trader, "start_websocket"):
            try:
                await trader.start_websocket()
                logger.info("[BINANCE] WebSocket streaming started")
            except Exception as e:
                logger.warning("[BINANCE] WebSocket failed to start: %s (falling back to REST polling)", e)

    if telegram_bot:
        market_list = ", ".join(f"{n.upper()}({e.trader.mode})" for n, e, _, _ in engines)
        await telegram_bot.send_message(f"FinanceBot started!\nMarkets: {market_list}")

    if args.scan_once:
        for name, engine, cfg, _ in engines:
            tickers = cfg.get("watchlist", cfg.get("pairs", []))
            logger.info("[%s] Running single scan on %d tickers...", name.upper(), len(tickers))
            signals = await engine.run_scan(tickers)
            logger.info("[%s] Scan complete: %d actionable signals", name.upper(), len(signals))
        return

    from src.utils.scheduler import JobScheduler
    scheduler = JobScheduler()

    for name, engine, cfg in engines:
        tickers = cfg.get("watchlist", cfg.get("pairs", []))
        schedule = cfg.get("schedule", {})

        if name == "binance":
            interval = schedule.get("scan_interval_minutes", 5)
            scheduler.add_interval_job(
                func=engine.run_scan, args=[tickers],
                minutes=interval, job_id=f"scan_{name}",
            )
            logger.info("[%s] Scheduled: scan every %d min (24/7)", name.upper(), interval)
        else:
            interval = schedule.get("scan_interval_minutes", 15)
            scheduler.add_interval_job(
                func=engine.run_scan, args=[tickers],
                minutes=interval, job_id=f"scan_{name}",
            )
            logger.info("[%s] Scheduled: scan every %d min", name.upper(), interval)

        summary_time = schedule.get("daily_summary_time", "16:00")
        hour, minute = map(int, summary_time.split(":"))
        tz = schedule.get("timezone", "UTC")

        async def send_summary(eng=engine, nm=name):
            summary = await eng.get_daily_summary()
            portfolio = await eng.get_portfolio_summary()
            if telegram_bot:
                await telegram_bot.send_message(f"Daily Summary — {nm.upper()}\n{portfolio}")

        scheduler.add_cron_job(
            func=send_summary, hour=hour, minute=minute,
            timezone=tz, job_id=f"summary_{name}",
        )

        chart_hours = schedule.get("chart_analysis_interval_hours", 6)
        if chart_hours > 0:
            async def run_charts(eng=engine, t=tickers):
                await eng.run_chart_analysis(t[:5])

            scheduler.add_interval_job(
                func=run_charts, hours=chart_hours, job_id=f"chart_{name}",
            )
            logger.info("[%s] Chart analysis: every %d hours", name.upper(), chart_hours)

    scheduler.start()
    logger.info("All schedules started. Ctrl+C to stop.")

    stop_event = asyncio.Event()

    def signal_handler():
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await stop_event.wait()
    finally:
        scheduler.shutdown()


async def main():
    args = parse_args()

    setup_logging(log_dir=str(PROJECT_ROOT / "data" / "logs"), level=args.log_level)
    logger.info("FinanceBot starting...")

    config_loader = ConfigLoader(project_root=PROJECT_ROOT)
    config = config_loader.load(user=args.user)
    logger.info("Configuration loaded for user: %s", args.user)

    db_path = config.get("tracking", {}).get("local_db", "data/financebot.db")
    init_db(str(PROJECT_ROOT / db_path))
    logger.info("Database initialized: %s", db_path)

    ai_router = AIRouter(config)
    providers = []
    if ai_router.claude:
        providers.append("Claude")
    if ai_router.groq:
        providers.append("Groq")
    logger.info("AI Router ready: [%s]", ", ".join(providers))

    telegram_config = config.get("notifications", {}).get("telegram", {})
    bot_token = telegram_config.get("bot_token", "")
    chat_id = telegram_config.get("chat_id", "")

    telegram_bot = None
    if bot_token and chat_id and telegram_config.get("enabled", False):
        telegram_bot = TelegramBot(bot_token=bot_token, chat_id=chat_id)
        await telegram_bot.start()
        logger.info("Telegram bot started")
    else:
        logger.warning("Telegram bot disabled (missing token/chat_id or disabled in config)")

    try:
        await run_finance(args, config_loader, ai_router, telegram_bot)
    finally:
        if telegram_bot:
            await telegram_bot.send_message("FinanceBot shutting down.")
            await telegram_bot.stop()
        logger.info("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
