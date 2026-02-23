"""Tests for the LLM service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from pr_review_agent.services.llm_service import LLMService


@pytest.fixture()
def mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.gemini_api_key = "AIzaTestKey1234567890abcdef"
    settings.llm_model = "gemini-2.0-flash-lite"
    settings.llm_temperature = 0.2
    settings.llm_max_tokens = 100
    settings.max_retries = 2
    return settings


@pytest.fixture()
def llm_service(mock_settings: MagicMock) -> LLMService:
    with patch("pr_review_agent.services.llm_service.ChatGoogleGenerativeAI"):
        svc = LLMService(settings=mock_settings)
        return svc


class _SimpleModel(BaseModel):
    name: str
    value: int


class TestInvokeRaw:
    def test_returns_string_content(self, llm_service: LLMService) -> None:
        mock_response = MagicMock()
        mock_response.content = "Hello from Gemini"
        mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
        llm_service._client.invoke.return_value = mock_response

        result = llm_service.invoke_raw("test prompt")
        assert result == "Hello from Gemini"

    def test_tracks_token_usage(self, llm_service: LLMService) -> None:
        mock_response = MagicMock()
        mock_response.content = "response"
        mock_response.usage_metadata = {"input_tokens": 20, "output_tokens": 10}
        llm_service._client.invoke.return_value = mock_response

        llm_service.invoke_raw("prompt")
        assert llm_service.total_tokens.input_tokens == 20
        assert llm_service.total_tokens.output_tokens == 10
        assert llm_service.call_count == 1


class TestExtractJson:
    def test_plain_json_string(self, llm_service: LLMService) -> None:
        result = llm_service._extract_json('{"name": "test", "value": 42}')
        assert result == {"name": "test", "value": 42}

    def test_json_in_markdown_fence(self, llm_service: LLMService) -> None:
        text = '```json\n{"name": "test", "value": 42}\n```'
        result = llm_service._extract_json(text)
        assert result["name"] == "test"

    def test_json_array_extracted(self, llm_service: LLMService) -> None:
        text = 'Here is the result:\n```json\n[{"a": 1}, {"a": 2}]\n```'
        result = llm_service._extract_json(text)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_raises_on_no_json(self, llm_service: LLMService) -> None:
        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            llm_service._extract_json("This is just plain text with no JSON.")

    def test_json_embedded_in_prose(self, llm_service: LLMService) -> None:
        text = 'Sure! Here: {"name": "result", "value": 7}'
        result = llm_service._extract_json(text)
        assert result["value"] == 7


class TestInvokeJsonList:
    def test_returns_empty_list_on_failure(self, llm_service: LLMService) -> None:
        mock_response = MagicMock()
        mock_response.content = "Not JSON at all!"
        mock_response.usage_metadata = {}
        llm_service._client.invoke.return_value = mock_response

        result = llm_service.invoke_json_list("prompt", _SimpleModel)
        assert result == []

    def test_parses_valid_json_list(self, llm_service: LLMService) -> None:
        mock_response = MagicMock()
        mock_response.content = '[{"name": "a", "value": 1}, {"name": "b", "value": 2}]'
        mock_response.usage_metadata = {}
        llm_service._client.invoke.return_value = mock_response

        result = llm_service.invoke_json_list("prompt", _SimpleModel)
        assert len(result) == 2
        assert result[0].name == "a"
        assert result[1].value == 2
