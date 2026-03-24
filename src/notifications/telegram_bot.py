"""Telegram bot for notifications and interactive commands."""

import asyncio
import logging
from pathlib import Path

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

from src.utils.security import sign_callback_data, verify_callback_data, sanitize_error

logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot for KartikAI notifications and commands."""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        self._allowed_chat_ids: set[str] = {self.chat_id}
        self.app: Application | None = None
        self._pipeline_paused = False

        # Callbacks for approve/reject (set by the orchestrator)
        self._approval_callbacks: dict[str, asyncio.Event] = {}
        self._approval_decisions: dict[str, str] = {}

    def _is_authorized(self, update: Update) -> bool:
        """Check if the message is from an authorized chat."""
        chat_id = str(update.effective_chat.id)
        if chat_id not in self._allowed_chat_ids:
            logger.warning("Unauthorized Telegram access from chat_id: %s", chat_id)
            return False
        return True

    async def start(self) -> None:
        """Initialize and start the Telegram bot."""
        self.app = Application.builder().token(self.bot_token).build()

        # Register command handlers
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("help", self._cmd_help))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("test", self._cmd_test))
        self.app.add_handler(CommandHandler("pause", self._cmd_pause))
        self.app.add_handler(CommandHandler("resume", self._cmd_resume))
        self.app.add_handler(CommandHandler("analytics", self._cmd_analytics_stub))
        self.app.add_handler(CommandHandler("scrape", self._cmd_scrape_stub))
        self.app.add_handler(CommandHandler("cost", self._cmd_cost_stub))
        self.app.add_handler(CallbackQueryHandler(self._handle_callback))

        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        logger.info("Telegram bot started")

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("Telegram bot stopped")

    # --- Command Handlers (all auth-gated) ---

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            await update.message.reply_text("Unauthorized.")
            return
        await update.message.reply_text(
            "Welcome to KartikAI Auto Job Applier!\n\n"
            "Commands:\n"
            "/status - Bot status\n"
            "/scrape - Trigger job scrape\n"
            "/analytics - View stats\n"
            "/pause - Pause pipeline\n"
            "/resume - Resume pipeline\n"
            "/cost - API cost estimate\n"
            "/help - Show this help"
        )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._cmd_start(update, context)

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        status = "PAUSED" if self._pipeline_paused else "RUNNING"
        pending = len(self._approval_callbacks)
        await update.message.reply_text(
            f"Pipeline: {status}\n"
            f"Pending approvals: {pending}"
        )

    async def _cmd_test(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        await update.message.reply_text("KartikAI bot is working!")

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        self._pipeline_paused = True
        await update.message.reply_text("Pipeline PAUSED. Use /resume to continue.")

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        self._pipeline_paused = False
        await update.message.reply_text("Pipeline RESUMED.")

    async def _cmd_analytics_stub(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        await update.message.reply_text("Analytics will be available after Phase 6 implementation.")

    async def _cmd_scrape_stub(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        await update.message.reply_text("Scraping will be available after Phase 2 implementation.")

    async def _cmd_cost_stub(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        await update.message.reply_text("Cost tracking will be available after AI layer is active.")

    # --- Callback Query Handler (HMAC-signed) ---

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()

        if not self._is_authorized(update):
            return

        # Verify HMAC signature
        raw_data = verify_callback_data(query.data)
        if raw_data is None:
            logger.warning("Invalid callback signature: %s", query.data[:30])
            await query.edit_message_text(
                query.message.text + "\n\n--- Invalid signature ---"
            )
            return

        if ":" not in raw_data:
            return

        action, job_id = raw_data.split(":", 1)

        if action not in ("approve", "reject"):
            logger.warning("Invalid callback action: %s", action)
            return

        if job_id in self._approval_callbacks:
            self._approval_decisions[job_id] = action
            self._approval_callbacks[job_id].set()
            await query.edit_message_reply_markup(reply_markup=None)
            label = "Approved" if action == "approve" else "Rejected"
            await query.edit_message_text(
                query.message.text + f"\n\n--- {label} ---"
            )
        else:
            await query.edit_message_text(
                query.message.text + "\n\n--- Expired (no pending action) ---"
            )

    # --- Notification Methods ---

    async def send_message(self, text: str) -> None:
        """Send a plain text message."""
        if not self.app:
            logger.warning("Telegram bot not initialized, skipping message")
            return
        await self.app.bot.send_message(chat_id=self.chat_id, text=text)

    async def send_photo(self, photo_path: str, caption: str = "") -> None:
        """Send a photo (screenshot)."""
        if not self.app:
            return
        with open(photo_path, "rb") as f:
            await self.app.bot.send_photo(
                chat_id=self.chat_id, photo=f, caption=caption
            )

    async def send_job_card(self, job_id: str, title: str, company: str,
                            score: float, matching_skills: list[str],
                            missing_skills: list[str], recommendation: str,
                            screenshot_path: str | None = None) -> None:
        """Send a rich job card with HMAC-signed approve/reject buttons."""
        text = (
            f"Ready to apply:\n"
            f"Role: {title} at {company}\n"
            f"Score: {score:.0f}/100 ({recommendation})\n"
            f"Matching Skills: {', '.join(matching_skills[:5])}\n"
            f"Missing: {', '.join(missing_skills[:3]) if missing_skills else 'None'}"
        )

        # Sign callback data with HMAC
        approve_data = sign_callback_data(f"approve:{job_id}")
        reject_data = sign_callback_data(f"reject:{job_id}")

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Approve", callback_data=approve_data),
                InlineKeyboardButton("Reject", callback_data=reject_data),
            ]
        ])

        if screenshot_path and Path(screenshot_path).exists():
            with open(screenshot_path, "rb") as f:
                await self.app.bot.send_photo(
                    chat_id=self.chat_id,
                    photo=f,
                    caption=text,
                    reply_markup=keyboard,
                )
        else:
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                reply_markup=keyboard,
            )

    async def send_application_confirmation(self, title: str, company: str,
                                             score: float, app_number: int) -> None:
        """Send confirmation after a successful application."""
        await self.send_message(
            f"Applied:\n"
            f"Role: {title} at {company}\n"
            f"Score: {score:.0f}/100\n"
            f"Application #{app_number}"
        )

    async def send_daily_summary(self, stats: dict) -> None:
        """Send end-of-day summary."""
        text = (
            f"Daily Summary\n"
            f"{'=' * 24}\n"
            f"Jobs Scraped: {stats.get('scraped', 0)}\n"
            f"Jobs Scored: {stats.get('scored', 0)}\n"
            f"Applications: {stats.get('applied', 0)}\n"
            f"Skipped: {stats.get('skipped', 0)}\n"
            f"Avg Score: {stats.get('avg_score', 0):.0f}/100"
        )
        await self.send_message(text)

    # --- Approval Flow ---

    async def wait_for_approval(self, job_id: str, timeout_minutes: int = 30) -> str:
        """Wait for user to approve or reject a job. Returns 'approve' or 'reject'."""
        event = asyncio.Event()
        self._approval_callbacks[job_id] = event

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_minutes * 60)
            decision = self._approval_decisions.pop(job_id, "reject")
            return decision
        except asyncio.TimeoutError:
            logger.info("Approval timeout for job %s", job_id)
            return "timeout"
        finally:
            self._approval_callbacks.pop(job_id, None)

    @property
    def is_paused(self) -> bool:
        return self._pipeline_paused
