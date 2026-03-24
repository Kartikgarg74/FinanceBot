"""Market hours enforcement — dynamic holiday calendar via exchange_calendars + Finnhub fallback."""

import logging
from datetime import datetime, time, date, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


class HolidayCalendar:
    """
    Dynamic holiday calendar — fetches holidays automatically.
    Priority:
      1. exchange_calendars library (offline, covers NSE/NYSE/etc for decades)
      2. Finnhub API (free, 60/min, real-time market status)
      3. Weekend-only fallback (no holidays, just weekdays)
    """

    def __init__(self, finnhub_api_key: str = ""):
        self._finnhub_key = finnhub_api_key
        self._cache: dict[str, set[str]] = {}  # exchange -> set of "YYYY-MM-DD"
        self._exchange_cal = None
        self._tried_exchange_cal = False

    def _load_exchange_calendars(self):
        """Try to load exchange_calendars library (pip install exchange_calendars)."""
        if self._tried_exchange_cal:
            return self._exchange_cal
        self._tried_exchange_cal = True
        try:
            import exchange_calendars
            self._exchange_cal = exchange_calendars
            logger.info("exchange_calendars loaded — dynamic holiday support active")
        except ImportError:
            logger.info("exchange_calendars not installed. Install with: pip install exchange_calendars")
            self._exchange_cal = None
        return self._exchange_cal

    def get_holidays(self, exchange_code: str, year: int | None = None) -> set[str]:
        """
        Get set of holiday date strings ("YYYY-MM-DD") for a given exchange.
        Exchange codes: XBOM (NSE/BSE India), XNYS (NYSE), XNAS (NASDAQ)
        """
        year = year or date.today().year
        cache_key = f"{exchange_code}:{year}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        holidays = set()

        # Method 1: exchange_calendars (best — offline, accurate)
        ec = self._load_exchange_calendars()
        if ec:
            try:
                cal = ec.get_calendar(exchange_code)
                start = f"{year}-01-01"
                end = f"{year}-12-31"
                # Get all sessions (trading days)
                sessions = cal.sessions_in_range(start, end)
                # All business days minus sessions = holidays
                import pandas as pd
                all_bdays = pd.bdate_range(start, end)
                holiday_dates = all_bdays.difference(sessions)
                holidays = {d.strftime("%Y-%m-%d") for d in holiday_dates}
                self._cache[cache_key] = holidays
                logger.info("Loaded %d holidays for %s (%d) via exchange_calendars", len(holidays), exchange_code, year)
                return holidays
            except Exception as e:
                logger.warning("exchange_calendars failed for %s: %s", exchange_code, e)

        # Method 2: Finnhub API (online, free)
        if self._finnhub_key:
            try:
                holidays = self._fetch_finnhub_holidays(exchange_code, year)
                if holidays:
                    self._cache[cache_key] = holidays
                    return holidays
            except Exception as e:
                logger.warning("Finnhub holiday fetch failed: %s", e)

        # Method 3: Empty set (weekend-only fallback)
        logger.warning("No holiday data for %s — weekday-only mode", exchange_code)
        self._cache[cache_key] = holidays
        return holidays

    def _fetch_finnhub_holidays(self, exchange_code: str, year: int) -> set[str]:
        """Fetch market holidays from Finnhub (free API)."""
        import httpx

        # Finnhub uses different exchange codes
        finnhub_exchange = {
            "XBOM": "BSE",
            "XNSE": "NSE",
            "XNYS": "US",
            "XNAS": "US",
        }.get(exchange_code, "US")

        try:
            resp = httpx.get(
                "https://finnhub.io/api/v1/stock/market-holiday",
                params={"exchange": finnhub_exchange, "token": self._finnhub_key},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            holidays = set()
            for item in data.get("data", []):
                event_date = item.get("atDate", "")
                if event_date.startswith(str(year)):
                    holidays.add(event_date)

            logger.info("Finnhub: %d holidays for %s (%d)", len(holidays), finnhub_exchange, year)
            return holidays
        except Exception as e:
            logger.error("Finnhub holiday API error: %s", e)
            return set()

    def is_holiday(self, exchange_code: str, check_date: date | None = None) -> bool:
        """Check if a specific date is a holiday."""
        check_date = check_date or date.today()
        year_holidays = self.get_holidays(exchange_code, check_date.year)
        return check_date.strftime("%Y-%m-%d") in year_holidays


# Global singleton
_calendar = None


def get_holiday_calendar(finnhub_key: str = "") -> HolidayCalendar:
    global _calendar
    if _calendar is None:
        _calendar = HolidayCalendar(finnhub_key)
    return _calendar


# ===== EXCHANGE CODE MAPPING =====
MARKET_EXCHANGE_CODES = {
    "zerodha": "XBOM",   # BSE/NSE India
    "alpaca": "XNYS",    # NYSE
    "binance": None,      # 24/7, no holidays
}


class MarketHours:
    """Checks whether a market is currently open. Uses dynamic holiday calendar."""

    MARKET_SCHEDULES = {
        "zerodha": {
            "open": time(9, 15),
            "close": time(15, 30),
            "timezone": "Asia/Kolkata",
            "weekdays": [0, 1, 2, 3, 4],  # Mon-Fri
        },
        "alpaca": {
            "open": time(9, 30),
            "close": time(16, 0),
            "timezone": "America/New_York",
            "weekdays": [0, 1, 2, 3, 4],
        },
        "binance": {
            "open": time(0, 0),
            "close": time(23, 59),
            "timezone": "UTC",
            "weekdays": [0, 1, 2, 3, 4, 5, 6],  # 24/7
        },
    }

    def __init__(self, market: str, config: dict | None = None):
        self.market = market
        schedule = config.get("schedule", {}) if config else {}

        if market in self.MARKET_SCHEDULES:
            self.schedule = dict(self.MARKET_SCHEDULES[market])
        else:
            self.schedule = dict(self.MARKET_SCHEDULES["binance"])

        # Override from config if provided
        hours_str = schedule.get("market_hours", "")
        if hours_str and "-" in hours_str:
            open_str, close_str = hours_str.split("-")
            oh, om = map(int, open_str.split(":"))
            ch, cm = map(int, close_str.split(":"))
            self.schedule["open"] = time(oh, om)
            self.schedule["close"] = time(ch, cm)

        tz_str = schedule.get("timezone")
        if tz_str:
            self.schedule["timezone"] = tz_str

        # Get exchange code for holiday lookups
        self.exchange_code = MARKET_EXCHANGE_CODES.get(market)

        # Init holiday calendar with Finnhub key if available
        news_config = config.get("news", {}).get("sources", {}).get("finnhub", {}) if config else {}
        finnhub_key = news_config.get("api_key", "")
        self._holidays = get_holiday_calendar(finnhub_key)

    def is_open(self) -> bool:
        """Check if the market is currently open."""
        if self.market == "binance":
            return True

        tz = ZoneInfo(self.schedule["timezone"])
        now = datetime.now(tz)

        # Weekend check
        if now.weekday() not in self.schedule["weekdays"]:
            return False

        # Holiday check (dynamic!)
        if self.exchange_code and self._holidays.is_holiday(self.exchange_code, now.date()):
            return False

        # Time check
        current_time = now.time()
        return self.schedule["open"] <= current_time <= self.schedule["close"]

    def next_open(self) -> str:
        """Return when the market next opens."""
        if self.is_open():
            return "NOW (market is open)"

        tz = ZoneInfo(self.schedule["timezone"])
        now = datetime.now(tz)

        for i in range(1, 15):  # Look up to 2 weeks ahead
            next_day = now + timedelta(days=i)
            if next_day.weekday() not in self.schedule["weekdays"]:
                continue
            if self.exchange_code and self._holidays.is_holiday(self.exchange_code, next_day.date()):
                continue
            open_time = datetime.combine(next_day.date(), self.schedule["open"], tzinfo=tz)
            return open_time.strftime("%Y-%m-%d %H:%M %Z")

        return "Unknown"

    def time_until_close(self) -> int:
        """Minutes until market close. Returns 0 if closed."""
        if not self.is_open():
            return 0

        tz = ZoneInfo(self.schedule["timezone"])
        now = datetime.now(tz)
        close_dt = datetime.combine(now.date(), self.schedule["close"], tzinfo=tz)
        delta = (close_dt - now).total_seconds() / 60
        return max(int(delta), 0)

    def get_holidays_this_year(self) -> list[str]:
        """Get all holidays for this year (useful for /markets command)."""
        if not self.exchange_code:
            return []
        return sorted(self._holidays.get_holidays(self.exchange_code))
