"""Free historical data fetcher using yfinance (Indian .NS, US, Crypto via CCXT)."""

import logging
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches historical OHLCV data from free sources."""

    # NSE ticker suffix for yfinance
    NSE_SUFFIX = ".NS"
    BSE_SUFFIX = ".BO"

    # yfinance valid intervals
    VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m",
                       "1h", "1d", "5d", "1wk", "1mo", "3mo"]

    def __init__(self):
        self._yf = None

    def _get_yf(self):
        if self._yf is None:
            import yfinance as yf
            self._yf = yf
        return self._yf

    def fetch_indian_stock(
        self,
        ticker: str,
        interval: str = "1d",
        days: int = 365,
        exchange: str = "NSE",
    ) -> pd.DataFrame | None:
        """
        Fetch Indian stock data via yfinance. FREE, no API key needed.
        ticker: NSE symbol like "RELIANCE", "TCS", "INFY"
        """
        suffix = self.NSE_SUFFIX if exchange.upper() == "NSE" else self.BSE_SUFFIX
        yf_ticker = f"{ticker}{suffix}"
        return self._fetch_yfinance(yf_ticker, interval, days)

    def fetch_us_stock(
        self,
        ticker: str,
        interval: str = "1d",
        days: int = 365,
    ) -> pd.DataFrame | None:
        """Fetch US stock data via yfinance. FREE."""
        return self._fetch_yfinance(ticker, interval, days)

    def fetch_crypto_ccxt(
        self,
        pair: str = "BTC/USDT",
        timeframe: str = "1h",
        limit: int = 500,
        exchange_name: str = "binance",
    ) -> pd.DataFrame | None:
        """Fetch crypto data via CCXT (from any exchange). FREE."""
        try:
            import ccxt
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({"enableRateLimit": True})
            ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)

            df = pd.DataFrame(ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df.index.name = "Date"
            logger.info("Fetched %d candles for %s from %s (%s)", len(df), pair, exchange_name, timeframe)
            return df
        except Exception as e:
            logger.error("CCXT fetch failed for %s: %s", pair, e)
            return None

    def fetch_crypto_yfinance(
        self,
        symbol: str = "BTC",
        interval: str = "1d",
        days: int = 365,
    ) -> pd.DataFrame | None:
        """Fetch crypto data via yfinance (BTC-USD format). FREE."""
        yf_ticker = f"{symbol}-USD"
        return self._fetch_yfinance(yf_ticker, interval, days)

    def _fetch_yfinance(
        self,
        ticker: str,
        interval: str = "1d",
        days: int = 365,
    ) -> pd.DataFrame | None:
        """Core yfinance fetch logic."""
        yf = self._get_yf()
        try:
            # yfinance limits: intraday data only available for last 60 days
            if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
                days = min(days, 59)

            end = datetime.now()
            start = end - timedelta(days=days)

            stock = yf.Ticker(ticker)
            df = stock.history(start=start, end=end, interval=interval)

            if df is None or df.empty:
                logger.warning("No data returned for %s (interval=%s, days=%d)", ticker, interval, days)
                return None

            # Standardize column names
            df.index.name = "Date"
            # yfinance returns Open, High, Low, Close, Volume already

            logger.info("Fetched %d candles for %s (%s, %d days)", len(df), ticker, interval, days)
            return df
        except Exception as e:
            logger.error("yfinance fetch failed for %s: %s", ticker, e)
            return None

    def fetch_multiple_indian(
        self,
        tickers: list[str],
        interval: str = "1d",
        days: int = 365,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple Indian stocks."""
        results = {}
        for ticker in tickers:
            df = self.fetch_indian_stock(ticker, interval, days)
            if df is not None and not df.empty:
                results[ticker] = df
        return results

    def fetch_multiple_us(
        self,
        tickers: list[str],
        interval: str = "1d",
        days: int = 365,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple US stocks."""
        results = {}
        for ticker in tickers:
            df = self.fetch_us_stock(ticker, interval, days)
            if df is not None and not df.empty:
                results[ticker] = df
        return results
