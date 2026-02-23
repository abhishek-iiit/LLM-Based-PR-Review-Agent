"""Application configuration via Pydantic Settings.

All settings are loaded from environment variables (or a .env file).
Import the singleton `settings` object rather than instantiating Settings directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the PR Review Agent.

    All fields map directly to environment variables.
    Required fields have no default; the app will fail fast on startup if they are missing.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Required ──────────────────────────────────────────────────────────────
    gemini_api_key: str = Field(
        ...,
        description="Google Gemini API key. Get one at https://aistudio.google.com/apikey",
    )
    github_token: str = Field(
        ...,
        description="GitHub personal access token with `repo` and `pull_requests` scopes.",
    )

    # ── LLM settings ──────────────────────────────────────────────────────────
    llm_model: str = Field(
        default="gemini-2.0-flash-lite",
        description="Google Gemini model ID to use for all LLM calls.",
    )
    llm_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Sampling temperature. Lower = more deterministic.",
    )
    llm_max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum number of output tokens per LLM call.",
    )

    # ── Retry / resilience ────────────────────────────────────────────────────
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for GitHub API and LLM calls.",
    )

    # ── Webhook server ────────────────────────────────────────────────────────
    webhook_secret: str = Field(
        default="",
        description="HMAC-SHA256 secret configured in GitHub webhook settings. "
        "Leave empty to skip signature verification (not recommended in production).",
    )
    port: int = Field(
        default=8080,
        ge=1024,
        le=65535,
        description="Port for the FastAPI webhook server.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Application log level.",
    )
    log_format: Literal["json", "console"] = Field(
        default="json",
        description="Log renderer. Use 'console' for local development.",
    )

    # ── Review behaviour ──────────────────────────────────────────────────────
    max_diff_chars: int = Field(
        default=8000,
        gt=0,
        description="Maximum characters of a file diff sent to the LLM. "
        "Diffs longer than this are truncated with a note.",
    )
    github_comment_max_chars: int = Field(
        default=65535,
        description="GitHub PR comment character limit.",
    )

    @field_validator("gemini_api_key")
    @classmethod
    def validate_gemini_key(cls, v: str) -> str:
        """Ensure the key looks like a Google AI Studio key."""
        if not v.startswith("AIza"):
            raise ValueError(
                "GEMINI_API_KEY must start with 'AIza'. "
                "Get a key at https://aistudio.google.com/apikey"
            )
        return v

    @field_validator("github_token")
    @classmethod
    def validate_github_token(cls, v: str) -> str:
        """Ensure the token is non-empty and has a plausible prefix."""
        if not v or len(v) < 10:
            raise ValueError("GITHUB_TOKEN appears invalid. It must be a non-empty PAT.")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton.

    Using lru_cache means Settings is instantiated once and reused everywhere.
    In tests, call `get_settings.cache_clear()` after patching env vars.
    """
    return Settings()  # type: ignore[call-arg]


# Convenience alias — import this in the rest of the codebase
settings: Settings = get_settings()
