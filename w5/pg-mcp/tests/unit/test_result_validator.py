"""Unit tests for the ResultValidator service.

These tests use a mocked OpenAI client so no API key or network is required.
They cover the confidence-threshold logic (which now keys off
``min_confidence_score``), JSON parsing robustness, and error handling.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from pg_mcp.config.settings import OpenAIConfig, ValidationConfig
from pg_mcp.models.errors import LLMError, LLMTimeoutError
from pg_mcp.services.result_validator import ResultValidator


def make_validator(
    *,
    enabled: bool = True,
    min_confidence_score: int = 70,
) -> ResultValidator:
    """Build a ResultValidator with a test OpenAI config."""
    openai_config = OpenAIConfig(api_key=SecretStr("sk-test-key-12345"))
    validation_config = ValidationConfig(
        enabled=enabled,
        min_confidence_score=min_confidence_score,
    )
    return ResultValidator(openai_config, validation_config)


def mock_completion(content: str) -> MagicMock:
    """Build a mock ChatCompletion whose message content is ``content``."""
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


class TestResultValidatorDisabled:
    """When validation is disabled, a high-confidence result is returned."""

    @pytest.mark.asyncio
    async def test_disabled_returns_high_confidence(self) -> None:
        validator = make_validator(enabled=False)
        result = await validator.validate(
            question="count", sql="SELECT 1", results=[{"n": 1}], row_count=1
        )
        assert result.confidence == 100
        assert result.is_acceptable is True


class TestResultValidatorThreshold:
    """is_acceptable must key off min_confidence_score."""

    @pytest.mark.asyncio
    async def test_confidence_above_threshold_is_acceptable(self) -> None:
        validator = make_validator(min_confidence_score=70)
        payload = json.dumps({"confidence": 85, "explanation": "good"})

        with patch.object(
            validator.client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_completion(payload)),
        ):
            result = await validator.validate(
                question="q", sql="SELECT 1", results=[{"n": 1}], row_count=1
            )
        assert result.confidence == 85
        assert result.is_acceptable is True

    @pytest.mark.asyncio
    async def test_confidence_below_threshold_not_acceptable(self) -> None:
        validator = make_validator(min_confidence_score=70)
        payload = json.dumps({"confidence": 50, "explanation": "weak"})

        with patch.object(
            validator.client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_completion(payload)),
        ):
            result = await validator.validate(
                question="q", sql="SELECT 1", results=[{"n": 1}], row_count=1
            )
        assert result.confidence == 50
        assert result.is_acceptable is False


class TestResultValidatorParsing:
    """The validator must be robust to malformed LLM output."""

    @pytest.mark.asyncio
    async def test_invalid_json_returns_moderate_confidence(self) -> None:
        validator = make_validator()
        with patch.object(
            validator.client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_completion("not-json-at-all")),
        ):
            result = await validator.validate(
                question="q", sql="SELECT 1", results=[{"n": 1}], row_count=1
            )
        assert result.confidence == 60
        assert result.is_acceptable is False

    @pytest.mark.asyncio
    async def test_out_of_range_confidence_is_clamped(self) -> None:
        validator = make_validator()
        payload = json.dumps({"confidence": 150, "explanation": "overshoot"})
        with patch.object(
            validator.client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_completion(payload)),
        ):
            result = await validator.validate(
                question="q", sql="SELECT 1", results=[{"n": 1}], row_count=1
            )
        assert 0 <= result.confidence <= 100


class TestResultValidatorErrors:
    """Error handling paths."""

    @pytest.mark.asyncio
    async def test_timeout_raises_llm_timeout(self) -> None:
        validator = make_validator()
        with (
            patch.object(
                validator.client.chat.completions,
                "create",
                new=AsyncMock(side_effect=TimeoutError("slow")),
            ),
            pytest.raises(LLMTimeoutError),
        ):
            await validator.validate(question="q", sql="SELECT 1", results=[{"n": 1}], row_count=1)

    @pytest.mark.asyncio
    async def test_empty_choices_raises_llm_error(self) -> None:
        validator = make_validator()
        empty = MagicMock()
        empty.choices = []
        with (
            patch.object(
                validator.client.chat.completions,
                "create",
                new=AsyncMock(return_value=empty),
            ),
            pytest.raises(LLMError),
        ):
            await validator.validate(question="q", sql="SELECT 1", results=[{"n": 1}], row_count=1)
