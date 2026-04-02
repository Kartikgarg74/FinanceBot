"""Tests for AI Router — startup validation, routing, token capping, and output validation."""

import pytest
from unittest.mock import MagicMock, patch

from src.ai.router import AIRouter, MAX_TOKENS_PER_TASK, MAX_TOKENS_ABSOLUTE


class TestStartupValidation:

    def test_raises_when_no_providers(self):
        config = {"ai": {"anthropic": {"api_key": ""}, "groq": {"api_key": ""}}}
        with pytest.raises(RuntimeError, match="No AI providers configured"):
            AIRouter(config)

    def test_works_with_groq_only(self, sample_config):
        sample_config["ai"]["anthropic"]["api_key"] = ""
        with patch("src.ai.router.GroqClient"):
            router = AIRouter(sample_config)
            assert router.claude is None
            assert router.groq is not None

    def test_works_with_both(self, sample_config):
        with patch("src.ai.router.ClaudeClient"), patch("src.ai.router.GroqClient"):
            router = AIRouter(sample_config)
            assert router.claude is not None
            assert router.groq is not None


class TestTokenCapping:

    def test_caps_news_sentiment(self, sample_config):
        with patch("src.ai.router.ClaudeClient"), patch("src.ai.router.GroqClient"):
            router = AIRouter(sample_config)
            assert router._cap_tokens("news_sentiment", 9999) == 512

    def test_unknown_task_uses_absolute(self, sample_config):
        with patch("src.ai.router.ClaudeClient"), patch("src.ai.router.GroqClient"):
            router = AIRouter(sample_config)
            assert router._cap_tokens("unknown", 99999) == MAX_TOKENS_ABSOLUTE


class TestOutputValidation:

    def test_clamps_confidence(self, sample_config):
        with patch("src.ai.router.ClaudeClient"), patch("src.ai.router.GroqClient"):
            router = AIRouter(sample_config)
            result = router._validate_json_output("test", {"confidence": 999})
            assert result["confidence"] == 100.0

    def test_clamps_negative_score(self, sample_config):
        with patch("src.ai.router.ClaudeClient"), patch("src.ai.router.GroqClient"):
            router = AIRouter(sample_config)
            result = router._validate_json_output("test", {"score": -999})
            assert result["score"] == -100.0

    def test_invalid_action_defaults_to_hold(self, sample_config):
        with patch("src.ai.router.ClaudeClient"), patch("src.ai.router.GroqClient"):
            router = AIRouter(sample_config)
            result = router._validate_json_output("test", {"action": "YOLO_BUY"})
            assert result["action"] == "hold"

    def test_valid_action_preserved(self, sample_config):
        with patch("src.ai.router.ClaudeClient"), patch("src.ai.router.GroqClient"):
            router = AIRouter(sample_config)
            result = router._validate_json_output("test", {"action": "buy", "confidence": 85})
            assert result["action"] == "buy"
            assert result["confidence"] == 85.0

    def test_non_dict_returns_empty(self, sample_config):
        with patch("src.ai.router.ClaudeClient"), patch("src.ai.router.GroqClient"):
            router = AIRouter(sample_config)
            assert router._validate_json_output("test", "not a dict") == {}

    def test_non_numeric_confidence_zeroed(self, sample_config):
        with patch("src.ai.router.ClaudeClient"), patch("src.ai.router.GroqClient"):
            router = AIRouter(sample_config)
            result = router._validate_json_output("test", {"confidence": "high"})
            assert result["confidence"] == 0.0


class TestTaskRouting:

    def test_sentiment_routes_to_cheap(self, sample_config):
        with patch("src.ai.router.ClaudeClient"), patch("src.ai.router.GroqClient"):
            router = AIRouter(sample_config)
            _, model = router._get_client_and_model("news_sentiment")
            assert model == "claude-haiku-4-5-20251001"

    def test_chart_routes_to_quality(self, sample_config):
        with patch("src.ai.router.ClaudeClient"), patch("src.ai.router.GroqClient"):
            router = AIRouter(sample_config)
            _, model = router._get_client_and_model("chart_analysis")
            assert model == "claude-sonnet-4-6"

    def test_fallback_routes_to_groq(self, sample_config):
        with patch("src.ai.router.ClaudeClient"), patch("src.ai.router.GroqClient"):
            router = AIRouter(sample_config)
            _, model = router._get_client_and_model("news_sentiment", use_fallback=True)
            assert model == "llama-3.3-70b-versatile"
