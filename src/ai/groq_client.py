"""Groq API client wrapper - free LLM fallback with rate limit tracking."""

import json
import logging
import time
from datetime import date

from groq import Groq

logger = logging.getLogger(__name__)


class GroqClient:
    """Wrapper around the Groq API (free Llama 3.3 70B)."""

    def __init__(self, api_key: str, daily_limit: int = 14400):
        self.client = Groq(api_key=api_key)
        self.daily_limit = daily_limit
        self._request_count = 0
        self._count_date = date.today()

    def _check_rate_limit(self):
        """Reset counter on new day, check if under limit."""
        today = date.today()
        if today != self._count_date:
            self._request_count = 0
            self._count_date = today

        if self._request_count >= self.daily_limit:
            raise RuntimeError(
                f"Groq daily rate limit reached ({self.daily_limit} requests). "
                "Falling back to Claude API."
            )

    def complete(
        self,
        prompt: str,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 1024,
        system_prompt: str = "",
        temperature: float = 0.7,
    ) -> str:
        """Send a completion request to Groq."""
        self._check_rate_limit()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                self._request_count += 1
                return response.choices[0].message.content
            except Exception as e:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                logger.warning("Groq error (%s), retrying in %ds...", e, wait)
                time.sleep(wait)

        raise RuntimeError("Groq API failed after 3 retries")

    def complete_json(
        self,
        prompt: str,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 2048,
        system_prompt: str = "",
        temperature: float = 0.3,
    ) -> dict:
        """Send a completion request and parse the response as JSON."""
        if "JSON" not in system_prompt and "json" not in prompt.lower():
            prompt += "\n\nRespond with valid JSON only, no other text."

        text = self.complete(prompt, model, max_tokens, system_prompt, temperature)

        from src.utils.security import safe_parse_json
        result = safe_parse_json(text)
        if result is None:
            logger.error("Failed to parse JSON from Groq response (all strategies failed)")
            raise ValueError("Groq returned unparseable JSON")
        return result

    @property
    def requests_remaining(self) -> int:
        """How many requests left today."""
        if date.today() != self._count_date:
            return self.daily_limit
        return max(0, self.daily_limit - self._request_count)
