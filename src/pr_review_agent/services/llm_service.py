"""LLM service wrapping Google Gemini via LangChain.

Provides:
- invoke_structured: call LLM and parse result into a Pydantic model
- invoke_raw: call LLM and return plain text
- token usage tracking across calls
- automatic retry on rate-limit / transient errors
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, TypeVar

from google.api_core.exceptions import ResourceExhausted
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pr_review_agent.config.settings import Settings, get_settings
from pr_review_agent.utils.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class TokenUsage(BaseModel):
    """Token usage for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMService:
    """Wrapper around ChatGoogleGenerativeAI with retry, tracking, and structured output.

    Args:
        settings: Application settings. Defaults to global singleton.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = ChatGoogleGenerativeAI(
            model=self._settings.llm_model,
            temperature=self._settings.llm_temperature,
            max_output_tokens=self._settings.llm_max_tokens,
            google_api_key=self._settings.gemini_api_key,
        )
        self._total_tokens = TokenUsage()
        self._call_count = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def total_tokens(self) -> TokenUsage:
        """Accumulated token usage across all calls in this service instance."""
        return self._total_tokens

    @property
    def call_count(self) -> int:
        """Total number of LLM calls made."""
        return self._call_count

    # ── Retry wrapper ─────────────────────────────────────────────────────────

    def _retry(self) -> Any:
        return retry(
            retry=retry_if_exception_type((ResourceExhausted, IOError)),
            stop=stop_after_attempt(self._settings.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            reraise=True,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _track_usage(self, response: Any) -> None:
        """Extract and accumulate token usage from an AIMessage response."""
        usage = getattr(response, "usage_metadata", None) or {}
        if isinstance(usage, dict):
            self._total_tokens = TokenUsage(
                input_tokens=self._total_tokens.input_tokens
                + usage.get("input_tokens", 0),
                output_tokens=self._total_tokens.output_tokens
                + usage.get("output_tokens", 0),
            )
        self._call_count += 1

    def _extract_json(self, text: str) -> Any:
        """Robustly extract JSON from LLM output that may contain prose.

        Tries:
        1. Direct JSON parse
        2. Extract from markdown code fence
        3. Scan for the first {...} or [...] block
        """
        text = text.strip()

        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Markdown code fence
        fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                pass

        # First brace block
        brace_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if brace_match:
            try:
                return json.loads(brace_match.group(1))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract valid JSON from LLM response:\n{text[:500]}")

    # ── Public API ────────────────────────────────────────────────────────────

    def invoke_raw(self, prompt: str) -> str:
        """Call the LLM and return the raw text response.

        Args:
            prompt: The full prompt to send.

        Returns:
            The LLM's text output.
        """
        start = time.perf_counter()
        log.debug("invoking LLM (raw)", prompt_length=len(prompt))

        @self._retry()
        def _call() -> str:
            response = self._client.invoke([HumanMessage(content=prompt)])
            self._track_usage(response)
            return str(response.content)

        result = _call()
        log.debug(
            "LLM raw response received",
            response_length=len(result),
            total_tokens=self._total_tokens.total,
            elapsed=round(time.perf_counter() - start, 2),
        )
        return result

    def invoke_structured(self, prompt: str, response_model: type[T]) -> T:
        """Call the LLM and parse the JSON response into a Pydantic model.

        Uses LangChain's `with_structured_output` when possible, falling back
        to raw JSON extraction if the model doesn't support tool-calling.

        Args:
            prompt: The full prompt to send.
            response_model: A Pydantic BaseModel subclass to parse into.

        Returns:
            An instance of response_model populated from the LLM output.

        Raises:
            ValueError: If the LLM output cannot be parsed into response_model.
        """
        start = time.perf_counter()
        log.debug(
            "invoking LLM (structured)",
            model=response_model.__name__,
            prompt_length=len(prompt),
        )

        @self._retry()
        def _call() -> T:
            structured_client = self._client.with_structured_output(response_model)
            response = structured_client.invoke([HumanMessage(content=prompt)])
            self._call_count += 1
            return response  # type: ignore[return-value]

        try:
            result = _call()
        except Exception as exc:
            log.warning(
                "structured output failed, falling back to raw JSON extraction",
                error=str(exc),
                model=response_model.__name__,
            )
            raw = self.invoke_raw(prompt)
            data = self._extract_json(raw)
            result = response_model.model_validate(data)

        log.debug(
            "LLM structured response received",
            model=response_model.__name__,
            total_tokens=self._total_tokens.total,
            elapsed=round(time.perf_counter() - start, 2),
        )
        return result

    def invoke_json_list(self, prompt: str, item_model: type[T]) -> list[T]:
        """Call the LLM expecting a JSON array and parse each element.

        Args:
            prompt: The full prompt to send.
            item_model: Pydantic model for each array element.

        Returns:
            List of item_model instances. Returns empty list if parsing fails.
        """
        start = time.perf_counter()
        log.debug(
            "invoking LLM (json list)",
            item_model=item_model.__name__,
            prompt_length=len(prompt),
        )

        raw = self.invoke_raw(prompt)

        try:
            data = self._extract_json(raw)
            if not isinstance(data, list):
                data = [data]
            items = [item_model.model_validate(item) for item in data]
        except Exception as exc:
            log.warning(
                "failed to parse LLM JSON list response",
                item_model=item_model.__name__,
                error=str(exc),
                raw_preview=raw[:200],
            )
            items = []

        log.debug(
            "LLM json list response parsed",
            item_model=item_model.__name__,
            item_count=len(items),
            elapsed=round(time.perf_counter() - start, 2),
        )
        return items
