"""Persists trades, news logs, and daily stats to SQLite."""

import json
import logging
import uuid
from datetime import datetime, date

from src.database.models import Trade, TradeDailyStat, NewsLog, get_session
from .base_trader import TradeRecord, Signal

logger = logging.getLogger(__name__)


class TradeLogger:
    """Persists all trade data to the SQLite database."""

    def save_trade(self, record: TradeRecord, signal: Signal | None = None, mode: str = "paper") -> None:
        """Save a trade record to the database."""
        try:
            session = get_session()
            trade = Trade(
                id=record.id or str(uuid.uuid4())[:12],
                ticker=record.ticker,
                exchange=record.exchange,
                side=record.side,
                quantity=record.quantity,
                price=record.price,
                order_type=record.order_type,
                status=record.status,
                reason=record.reason,
                stop_loss=record.stop_loss,
                take_profit=record.take_profit,
                pnl=record.pnl,
                confidence=signal.confidence if signal else None,
                technical_score=signal.technical_score if signal else None,
                sentiment_score=signal.sentiment_score if signal else None,
                mode=mode,
                timestamp=record.timestamp,
            )
            session.add(trade)
            session.commit()
            session.close()
            logger.debug("Trade saved: %s %s %s @ %.2f", record.side, record.ticker, record.exchange, record.price)
        except Exception as e:
            logger.error("Failed to save trade: %s", e)

    def save_news(self, title: str, source: str, url: str, score: float,
                  confidence: float, category: str, tickers: list[str],
                  published_at: datetime | None = None) -> None:
        """Save a scored news article to the database."""
        try:
            session = get_session()
            news = NewsLog(
                title=title[:500],
                source=source,
                url=url,
                sentiment_score=score,
                confidence=confidence,
                category=category,
                affected_tickers=json.dumps(tickers),
                published_at=published_at,
            )
            session.add(news)
            session.commit()
            session.close()
        except Exception as e:
            logger.error("Failed to save news log: %s", e)

    def save_daily_stats(self, exchange: str, trades: list[TradeRecord]) -> None:
        """Compute and save daily stats."""
        today = date.today()
        today_trades = [t for t in trades if t.timestamp.date() == today and t.status == "executed"]

        if not today_trades:
            return

        winning = sum(1 for t in today_trades if t.pnl and t.pnl > 0)
        losing = sum(1 for t in today_trades if t.pnl and t.pnl < 0)
        total_pnl = sum(t.pnl or 0 for t in today_trades)

        try:
            session = get_session()
            # Upsert
            existing = session.query(TradeDailyStat).filter_by(date=today, exchange=exchange).first()
            if existing:
                existing.total_trades = len(today_trades)
                existing.winning_trades = winning
                existing.losing_trades = losing
                existing.total_pnl = total_pnl
            else:
                stat = TradeDailyStat(
                    date=today,
                    exchange=exchange,
                    total_trades=len(today_trades),
                    winning_trades=winning,
                    losing_trades=losing,
                    total_pnl=total_pnl,
                )
                session.add(stat)
            session.commit()
            session.close()
        except Exception as e:
            logger.error("Failed to save daily stats: %s", e)

    def get_recent_trades(self, exchange: str | None = None, limit: int = 20) -> list[dict]:
        """Get recent trades from DB."""
        try:
            session = get_session()
            query = session.query(Trade).order_by(Trade.timestamp.desc())
            if exchange:
                query = query.filter(Trade.exchange == exchange)
            trades = query.limit(limit).all()
            result = [
                {
                    "id": t.id, "ticker": t.ticker, "exchange": t.exchange,
                    "side": t.side, "qty": t.quantity, "price": t.price,
                    "pnl": t.pnl, "status": t.status, "mode": t.mode,
                    "time": t.timestamp.strftime("%Y-%m-%d %H:%M"),
                }
                for t in trades
            ]
            session.close()
            return result
        except Exception as e:
            logger.error("Failed to get recent trades: %s", e)
            return []

    def get_performance_stats(self, exchange: str | None = None, days: int = 30) -> dict:
        """Get aggregate performance stats."""
        try:
            from datetime import timedelta
            session = get_session()
            cutoff = datetime.utcnow() - timedelta(days=days)
            query = session.query(Trade).filter(Trade.timestamp >= cutoff, Trade.status == "executed")
            if exchange:
                query = query.filter(Trade.exchange == exchange)
            trades = query.all()
            session.close()

            if not trades:
                return {"total_trades": 0, "win_rate": 0, "total_pnl": 0}

            wins = sum(1 for t in trades if t.pnl and t.pnl > 0)
            total_pnl = sum(t.pnl or 0 for t in trades)

            return {
                "total_trades": len(trades),
                "wins": wins,
                "losses": len(trades) - wins,
                "win_rate": round(wins / len(trades) * 100, 1) if trades else 0,
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(total_pnl / len(trades), 2) if trades else 0,
            }
        except Exception as e:
            logger.error("Failed to get performance stats: %s", e)
            return {}
