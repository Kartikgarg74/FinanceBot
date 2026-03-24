"""Health monitoring and error recovery for trading connections."""

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class APIHealth:
    """Tracks health of an external API connection."""
    name: str
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_success: float = 0.0
    last_failure: float = 0.0
    is_healthy: bool = True
    backoff_until: float = 0.0  # Don't retry until this timestamp

    MAX_CONSECUTIVE_FAILURES = 5
    BASE_BACKOFF_SECONDS = 2
    MAX_BACKOFF_SECONDS = 300  # 5 minutes max

    def record_success(self):
        self.consecutive_failures = 0
        self.total_successes += 1
        self.last_success = time.time()
        self.is_healthy = True
        self.backoff_until = 0

    def record_failure(self, error: str = ""):
        self.consecutive_failures += 1
        self.total_failures += 1
        self.last_failure = time.time()

        if self.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            self.is_healthy = False
            backoff = min(
                self.BASE_BACKOFF_SECONDS * (2 ** self.consecutive_failures),
                self.MAX_BACKOFF_SECONDS,
            )
            self.backoff_until = time.time() + backoff
            logger.warning(
                "%s: %d consecutive failures. Backing off for %ds. Error: %s",
                self.name, self.consecutive_failures, backoff, error[:200],
            )

    def can_retry(self) -> bool:
        """Check if we should retry (backoff expired)."""
        if self.backoff_until == 0:
            return True
        if time.time() >= self.backoff_until:
            self.backoff_until = 0
            return True
        return False

    @property
    def status(self) -> str:
        if self.is_healthy:
            return "HEALTHY"
        if self.can_retry():
            return "RECOVERING"
        return f"BACKING_OFF (retry in {int(self.backoff_until - time.time())}s)"


class HealthMonitor:
    """Monitors health of all trading connections."""

    def __init__(self):
        self._apis: dict[str, APIHealth] = {}

    def register(self, name: str) -> APIHealth:
        """Register an API to monitor."""
        health = APIHealth(name=name)
        self._apis[name] = health
        return health

    def get(self, name: str) -> APIHealth:
        """Get health for an API."""
        if name not in self._apis:
            return self.register(name)
        return self._apis[name]

    def record_success(self, name: str):
        self.get(name).record_success()

    def record_failure(self, name: str, error: str = ""):
        self.get(name).record_failure(error)

    def can_call(self, name: str) -> bool:
        """Check if an API call should be attempted."""
        return self.get(name).can_retry()

    def get_summary(self) -> str:
        """Get health summary for all APIs."""
        lines = ["API Health Monitor", "=" * 30]
        for name, health in self._apis.items():
            uptime = health.total_successes / max(health.total_successes + health.total_failures, 1) * 100
            lines.append(
                f"  {name}: {health.status} "
                f"(success={health.total_successes}, fail={health.total_failures}, "
                f"uptime={uptime:.0f}%)"
            )
        return "\n".join(lines)

    @property
    def all_healthy(self) -> bool:
        return all(h.is_healthy for h in self._apis.values())


# Global health monitor
_monitor = HealthMonitor()


def get_health_monitor() -> HealthMonitor:
    return _monitor


async def resilient_call(name: str, func, *args, **kwargs):
    """
    Execute a function with health tracking and exponential backoff.
    Raises the original exception if all retries exhausted.
    """
    monitor = get_health_monitor()
    health = monitor.get(name)

    if not health.can_retry():
        raise ConnectionError(f"{name} is in backoff (retry in {int(health.backoff_until - time.time())}s)")

    try:
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        health.record_success()
        return result
    except Exception as e:
        health.record_failure(str(e))
        raise
