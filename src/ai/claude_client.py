"""Claude API client wrapper with retry, token tracking, and JSON parsing."""

import json
import logging
import time

import anthropic

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Wrapper around the Anthropic Claude API."""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def complete(
        self,
        prompt: str,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 1024,
        system_prompt: str = "",
        temperature: float = 0.7,
    ) -> str:
        """Send a completion request to Claude with retry logic."""
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        for attempt in range(3):
            try:
                response = self.client.messages.create(**kwargs)
                self.total_input_tokens += response.usage.input_tokens
                self.total_output_tokens += response.usage.output_tokens
                logger.debug(
                    "Claude %s: %d in / %d out tokens",
                    model, response.usage.input_tokens, response.usage.output_tokens,
                )
                return response.content[0].text
            except anthropic.RateLimitError:
                wait = 2 ** attempt
                logger.warning("Rate limited, retrying in %ds...", wait)
                time.sleep(wait)
            except anthropic.APIError as e:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                logger.warning("API error (%s), retrying in %ds...", e, wait)
                time.sleep(wait)

        raise RuntimeError("Claude API failed after 3 retries")

    def complete_json(
        self,
        prompt: str,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 2048,
        system_prompt: str = "",
        temperature: float = 0.3,
    ) -> dict:
        """Send a completion request and parse the response as JSON.

        Uses multi-strategy parsing with fallback to prevent crashes from malformed output.
        """
        if "JSON" not in system_prompt and "json" not in prompt.lower():
            prompt += "\n\nRespond with valid JSON only, no other text."

        text = self.complete(prompt, model, max_tokens, system_prompt, temperature)

        from src.utils.security import safe_parse_json
        result = safe_parse_json(text)
        if result is None:
            logger.error("Failed to parse JSON from Claude response (all strategies failed)")
            logger.debug("Raw response: %s", text[:300])
            raise ValueError("Claude returned unparseable JSON")
        return result

    @property
    def estimated_cost(self) -> float:
        """Estimate cost based on token usage (rough approximation)."""
        # Haiku: $0.25/M input, $1.25/M output
        # Sonnet: $3/M input, $15/M output
        # Using Haiku rates as default approximation
        input_cost = (self.total_input_tokens / 1_000_000) * 0.25
        output_cost = (self.total_output_tokens / 1_000_000) * 1.25
        return input_cost + output_cost
