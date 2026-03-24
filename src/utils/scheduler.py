"""APScheduler wrapper for scheduling recurring tasks."""

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)


class JobScheduler:
    """Wrapper around APScheduler for managing scheduled tasks."""

    def __init__(self):
        self.scheduler = AsyncIOScheduler()

    def add_cron_job(self, func, hour: int, minute: int = 0,
                     timezone: str = "Asia/Kolkata", job_id: str | None = None, **kwargs):
        """Add a cron-triggered job (runs at specific time daily)."""
        trigger = CronTrigger(hour=hour, minute=minute, timezone=timezone)
        self.scheduler.add_job(func, trigger, id=job_id, replace_existing=True, **kwargs)
        logger.info("Scheduled cron job '%s' at %02d:%02d %s", job_id or func.__name__, hour, minute, timezone)

    def add_interval_job(self, func, minutes: int = 60, job_id: str | None = None, **kwargs):
        """Add an interval-triggered job (runs every N minutes)."""
        trigger = IntervalTrigger(minutes=minutes)
        self.scheduler.add_job(func, trigger, id=job_id, replace_existing=True, **kwargs)
        logger.info("Scheduled interval job '%s' every %d min", job_id or func.__name__, minutes)

    def start(self):
        """Start the scheduler."""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Scheduler started with %d jobs", len(self.scheduler.get_jobs()))

    def shutdown(self):
        """Gracefully shut down the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("Scheduler shut down")

    @property
    def jobs(self) -> list:
        """List all scheduled jobs."""
        return self.scheduler.get_jobs()
