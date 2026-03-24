"""In-memory TTL cache for quotes, news, and indicators to avoid API hammering."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    value: Any
    expires_at: float


class TTLCache:
    """Simple thread-safe in-memory cache with TTL expiration."""

    def __init__(self):
        self._store: dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Get a value from cache. Returns None if expired or missing."""
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        if time.time() > entry.expires_at:
            del self._store[key]
            self._misses += 1
            return None
        self._hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl_seconds: int = 60) -> None:
        """Set a value with TTL."""
        self._store[key] = CacheEntry(value=value, expires_at=time.time() + ttl_seconds)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def cleanup(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        expired = [k for k, v in self._store.items() if now > v.expires_at]
        for k in expired:
            del self._store[k]
        return len(expired)

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{(self._hits / total * 100):.1f}%" if total > 0 else "N/A",
        }


# Global cache instance with preset TTLs
_cache = TTLCache()

# TTL presets (seconds)
QUOTE_TTL = 30        # Quotes refresh every 30 seconds
NEWS_TTL = 600        # News cached for 10 minutes
INDICATOR_TTL = 300   # Indicators cached for 5 minutes
SENTIMENT_TTL = 600   # Sentiment cached for 10 minutes
HISTORY_TTL = 900     # Historical data cached for 15 minutes


def get_cache() -> TTLCache:
    return _cache


def cached_quote(ticker: str) -> Any | None:
    return _cache.get(f"quote:{ticker}")


def set_cached_quote(ticker: str, data: dict) -> None:
    _cache.set(f"quote:{ticker}", data, QUOTE_TTL)


def cached_history(ticker: str, interval: str) -> Any | None:
    return _cache.get(f"history:{ticker}:{interval}")


def set_cached_history(ticker: str, interval: str, data: Any) -> None:
    _cache.set(f"history:{ticker}:{interval}", data, HISTORY_TTL)


def cached_sentiment(ticker: str) -> Any | None:
    return _cache.get(f"sentiment:{ticker}")


def set_cached_sentiment(ticker: str, data: Any) -> None:
    _cache.set(f"sentiment:{ticker}", data, SENTIMENT_TTL)
