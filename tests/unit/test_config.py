"""Tests for configuration management."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pr_review_agent.config.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """Clear the lru_cache after each test to avoid state leakage."""
    yield
    get_settings.cache_clear()


def _valid_env(overrides: dict | None = None) -> dict:
    base = {
        "GEMINI_API_KEY": "AIzaTestKey1234567890abcdef",
        "GITHUB_TOKEN": "ghp_testtoken1234",
    }
    if overrides:
        base.update(overrides)
    return base


def test_settings_loads_required_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _valid_env().items():
        monkeypatch.setenv(k, v)
    settings = Settings()
    assert settings.gemini_api_key == "AIzaTestKey1234567890abcdef"
    assert settings.github_token == "ghp_testtoken1234"


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _valid_env().items():
        monkeypatch.setenv(k, v)
    settings = Settings()
    assert settings.llm_model == "gemini-2.0-flash-lite"
    assert settings.llm_temperature == 0.2
    assert settings.llm_max_tokens == 4096
    assert settings.max_retries == 3
    assert settings.log_level == "INFO"
    assert settings.log_format == "json"
    assert settings.port == 8080


def test_missing_gemini_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test1234567890")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises((ValidationError, Exception)):
        Settings()


def test_missing_github_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaTestKey1234567890abcdef")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises((ValidationError, Exception)):
        Settings()


def test_invalid_gemini_key_prefix_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _valid_env({"GEMINI_API_KEY": "invalid-key-format"}).items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValidationError):
        Settings()


def test_temperature_out_of_range_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _valid_env({"LLM_TEMPERATURE": "2.5"}).items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValidationError):
        Settings()


def test_log_level_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _valid_env({"LOG_LEVEL": "VERBOSE"}).items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValidationError):
        Settings()


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _valid_env().items():
        monkeypatch.setenv(k, v)
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
