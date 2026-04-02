"""Shared test fixtures for FinanceBot tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_config():
    """Minimal config for testing."""
    return {
        "ai": {
            "primary_provider": "anthropic",
            "fallback_provider": "groq",
            "anthropic": {
                "api_key": "test-key-anthropic",
                "models": {
                    "cheap": "claude-haiku-4-5-20251001",
                    "quality": "claude-sonnet-4-6",
                },
            },
            "groq": {
                "api_key": "test-key-groq",
                "model": "llama-3.3-70b-versatile",
                "rate_limit": 14400,
            },
        },
        "trading": {
            "risk_per_trade_pct": 2,
            "max_capital": 100000,
        },
        "notifications": {
            "telegram": {"enabled": False},
        },
    }


@pytest.fixture
def mock_telegram():
    bot = MagicMock()
    bot.send_message = AsyncMock()
    return bot
