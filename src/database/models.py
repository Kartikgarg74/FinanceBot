"""SQLAlchemy 2.0 database models for FinanceBot."""

from datetime import datetime, date
from pathlib import Path

from sqlalchemy import create_engine, String, Float, Integer, Text, DateTime, Date
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class Trade(Base):
    """Executed trade record (all markets)."""
    __tablename__ = "trades"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    ticker: Mapped[str] = mapped_column(String, nullable=False)
    exchange: Mapped[str] = mapped_column(String, default="")  # ZERODHA, ALPACA, BINANCE
    side: Mapped[str] = mapped_column(String, nullable=False)  # BUY, SELL
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    order_type: Mapped[str] = mapped_column(String, default="MARKET")
    status: Mapped[str] = mapped_column(String, default="executed")
    reason: Mapped[str] = mapped_column(Text, default="")
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    technical_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    mode: Mapped[str] = mapped_column(String, default="paper")  # paper, live
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TradeDailyStat(Base):
    """Daily trading statistics per market."""
    __tablename__ = "trade_daily_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    exchange: Mapped[str] = mapped_column(String, nullable=False)
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0)
    losing_trades: Mapped[int] = mapped_column(Integer, default=0)
    total_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    capital_end: Mapped[float] = mapped_column(Float, default=0.0)


class NewsLog(Base):
    """Log of processed news articles and their sentiment scores."""
    __tablename__ = "news_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String, default="")
    url: Mapped[str] = mapped_column(String, default="")
    sentiment_score: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    category: Mapped[str] = mapped_column(String, default="")
    affected_tickers: Mapped[str] = mapped_column(Text, default="[]")  # JSON
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    scored_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


_engine = None
_SessionLocal = None


def init_db(db_path: str = "data/financebot.db") -> None:
    """Initialize the database engine and create tables."""
    global _engine, _SessionLocal
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(f"sqlite:///{db_path}", echo=False)
    _SessionLocal = sessionmaker(bind=_engine)
    Base.metadata.create_all(_engine)


def get_session() -> Session:
    """Get a new database session."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _SessionLocal()
