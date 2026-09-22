"""Integration-style unit tests for orchestrator wiring.

These tests cover the behaviours that the codex review flagged as designed but
not integrated:

- Multi-database routing: each database executes against its OWN executor.
- Metrics: the pipeline records query/LLM/database/security metrics.
- Rate limiting: LLM and query execution are gated; exhaustion is surfaced.
- Retry/backoff: transient LLM errors are retried with exponential backoff.
- Question length: the configured maximum is enforced.

They use lightweight mocks so no database or OpenAI API is required.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    LLMTimeoutError,
    QuestionTooLongError,
    SecurityViolationError,
)
from pg_mcp.models.query import QueryRequest, ResultValidationResult, ReturnType
from pg_mcp.models.schema import ColumnInfo, DatabaseSchema, TableInfo
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.orchestrator import QueryOrchestrator


def make_schema(name: str = "test_db") -> DatabaseSchema:
    """Build a minimal single-table schema for tests."""
    return DatabaseSchema(
        database_name=name,
        tables=[
            TableInfo(
                schema_name="public",
                table_name="users",
                columns=[
                    ColumnInfo(name="id", data_type="integer", is_nullable=False),
                ],
            )
        ],
        version="16.0",
    )


def make_cache(schema: DatabaseSchema) -> MagicMock:
    """Build a schema-cache mock that returns ``schema`` and a fixed age."""
    cache = MagicMock()
    cache.get.return_value = schema
    cache.get_cache_age.return_value = 12.5
    return cache


def make_generator(sql: str = "SELECT id FROM users;") -> AsyncMock:
    """Build a generator mock that returns ``sql`` and reports token usage."""
    generator = AsyncMock()
    generator.generate.return_value = sql
    generator.last_tokens_used = 123
    return generator


def passing_validator() -> MagicMock:
    """Build a validator mock that accepts any SQL."""
    validator = MagicMock()
    validator.validate_or_raise.return_value = None
    return validator


class TestMultiDatabaseRouting:
    """Each database must be executed against its own executor."""

    @pytest.mark.asyncio
    async def test_query_routes_to_correct_executor(self) -> None:
        """Querying db2 must call db2's executor, never db1's."""
        exec1 = AsyncMock()
        exec1.execute.return_value = ([{"id": 1}], 1)
        exec2 = AsyncMock()
        exec2.execute.return_value = ([{"id": 2}], 1)

        orchestrator = QueryOrchestrator(
            sql_generator=make_generator(),
            sql_validator=passing_validator(),
            sql_executor=exec1,  # fallback (primary)
            result_validator=MagicMock(),
            schema_cache=make_cache(make_schema("db2")),
            pools={"db1": MagicMock(), "db2": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=False),
            sql_executors={"db1": exec1, "db2": exec2},
        )

        response = await orchestrator.execute_query(
            QueryRequest(question="get user", database="db2", return_type=ReturnType.RESULT)
        )

        assert response.success is True
        exec2.execute.assert_awaited_once()
        exec1.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_to_single_executor(self) -> None:
        """When no per-database executors are given, the fallback is used."""
        fallback = AsyncMock()
        fallback.execute.return_value = ([{"id": 1}], 1)

        orchestrator = QueryOrchestrator(
            sql_generator=make_generator(),
            sql_validator=passing_validator(),
            sql_executor=fallback,
            result_validator=MagicMock(),
            schema_cache=make_cache(make_schema("only")),
            pools={"only": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=False),
        )

        response = await orchestrator.execute_query(
            QueryRequest(question="get user", return_type=ReturnType.RESULT)
        )

        assert response.success is True
        fallback.execute.assert_awaited_once()


class TestQuestionLengthEnforcement:
    """The configured maximum question length must be enforced."""

    @pytest.mark.asyncio
    async def test_question_too_long_rejected(self) -> None:
        """A question longer than max_question_length yields a QUESTION_TOO_LONG error."""
        orchestrator = QueryOrchestrator(
            sql_generator=make_generator(),
            sql_validator=passing_validator(),
            sql_executor=AsyncMock(),
            result_validator=MagicMock(),
            schema_cache=make_cache(make_schema()),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(max_question_length=10),
        )

        response = await orchestrator.execute_query(
            QueryRequest(question="x" * 50, database="test_db", return_type=ReturnType.SQL)
        )

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "question_too_long"

    @pytest.mark.asyncio
    async def test_check_question_length_raises(self) -> None:
        """The internal helper raises QuestionTooLongError with useful details."""
        orchestrator = QueryOrchestrator(
            sql_generator=make_generator(),
            sql_validator=passing_validator(),
            sql_executor=AsyncMock(),
            result_validator=MagicMock(),
            schema_cache=make_cache(make_schema()),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(max_question_length=5),
        )
        with pytest.raises(QuestionTooLongError) as exc_info:
            orchestrator._check_question_length("way too long")
        assert exc_info.value.details["max_length"] == 5


class TestMetricsIntegration:
    """The orchestrator must record metrics throughout the pipeline."""

    @pytest.mark.asyncio
    async def test_success_metrics_recorded(self) -> None:
        """A successful query records request, duration, LLM and DB metrics."""
        metrics = MagicMock()
        executor = AsyncMock()
        executor.execute.return_value = ([{"id": 1}], 1)

        orchestrator = QueryOrchestrator(
            sql_generator=make_generator(),
            sql_validator=passing_validator(),
            sql_executor=executor,
            result_validator=MagicMock(),
            schema_cache=make_cache(make_schema()),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=False),
            metrics=metrics,
        )

        await orchestrator.execute_query(
            QueryRequest(question="get user", database="test_db", return_type=ReturnType.RESULT)
        )

        metrics.increment_query_request.assert_called_once()
        assert metrics.increment_query_request.call_args.kwargs["status"] == "success"
        metrics.query_duration.observe.assert_called_once()
        metrics.increment_llm_call.assert_any_call("generate_sql")
        metrics.observe_db_query_duration.assert_called_once()
        metrics.increment_llm_tokens.assert_called_once_with("generate_sql", 123)
        metrics.set_schema_cache_age.assert_called_once()

    @pytest.mark.asyncio
    async def test_security_violation_records_rejection(self) -> None:
        """A blocked query increments the SQL rejection counter and error status."""
        metrics = MagicMock()
        validator = MagicMock()
        validator.validate_or_raise.side_effect = SecurityViolationError("DELETE not allowed")

        orchestrator = QueryOrchestrator(
            sql_generator=make_generator("DELETE FROM users;"),
            sql_validator=validator,
            sql_executor=AsyncMock(),
            result_validator=MagicMock(),
            schema_cache=make_cache(make_schema()),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=0),
            validation_config=ValidationConfig(),
            metrics=metrics,
        )

        response = await orchestrator.execute_query(
            QueryRequest(question="delete", database="test_db", return_type=ReturnType.SQL)
        )

        assert response.success is False
        metrics.increment_sql_rejected.assert_called()
        assert metrics.increment_query_request.call_args.kwargs["status"] == "security_violation"


class TestRateLimiting:
    """LLM and query execution must be gated by the rate limiter."""

    @pytest.mark.asyncio
    async def test_llm_rate_limit_exhaustion_surfaced(self) -> None:
        """When no LLM slot is available in time, a RATE_LIMIT_EXCEEDED error is returned."""
        rate_limiter = MultiRateLimiter(query_limit=1, llm_limit=1)
        # Saturate the LLM limiter so generation cannot acquire a slot.
        await rate_limiter.llm_limiter.acquire()

        orchestrator = QueryOrchestrator(
            sql_generator=make_generator(),
            sql_validator=passing_validator(),
            sql_executor=AsyncMock(),
            result_validator=MagicMock(),
            schema_cache=make_cache(make_schema()),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=0, rate_limit_timeout=0.1),
            validation_config=ValidationConfig(),
            rate_limiter=rate_limiter,
        )

        response = await orchestrator.execute_query(
            QueryRequest(question="get user", database="test_db", return_type=ReturnType.SQL)
        )

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "rate_limit_exceeded"

    @pytest.mark.asyncio
    async def test_query_rate_limit_exhaustion_surfaced(self) -> None:
        """When no query slot is available in time, execution is rejected."""
        rate_limiter = MultiRateLimiter(query_limit=1, llm_limit=5)
        # Saturate the query limiter; LLM generation will still succeed.
        await rate_limiter.query_limiter.acquire()

        executor = AsyncMock()
        executor.execute.return_value = ([{"id": 1}], 1)

        orchestrator = QueryOrchestrator(
            sql_generator=make_generator(),
            sql_validator=passing_validator(),
            sql_executor=executor,
            result_validator=MagicMock(),
            schema_cache=make_cache(make_schema()),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=0, rate_limit_timeout=0.1),
            validation_config=ValidationConfig(enabled=False),
            rate_limiter=rate_limiter,
        )

        response = await orchestrator.execute_query(
            QueryRequest(question="get user", database="test_db", return_type=ReturnType.RESULT)
        )

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "rate_limit_exceeded"
        executor.execute.assert_not_called()


class TestRetryWithBackoff:
    """Transient LLM errors must be retried with exponential backoff."""

    @pytest.mark.asyncio
    async def test_transient_llm_error_retried_then_succeeds(self) -> None:
        """A transient timeout on the first attempt is retried and then succeeds."""
        generator = AsyncMock()
        generator.generate.side_effect = [
            LLMTimeoutError("temporary timeout"),
            "SELECT id FROM users;",
        ]
        generator.last_tokens_used = None

        orchestrator = QueryOrchestrator(
            sql_generator=generator,
            sql_validator=passing_validator(),
            sql_executor=AsyncMock(),
            result_validator=MagicMock(),
            schema_cache=make_cache(make_schema()),
            pools={"test_db": MagicMock()},
            # retry_delay is clamped to >= 0.1 by config; keep it at the minimum.
            resilience_config=ResilienceConfig(max_retries=2, retry_delay=0.1, backoff_factor=2.0),
            validation_config=ValidationConfig(),
        )

        sql, validation, _tokens = await orchestrator._generate_sql_with_retry(
            question="get user", schema=make_schema(), request_id="req-1"
        )

        assert sql == "SELECT id FROM users;"
        assert validation.is_valid is True
        assert generator.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_transient_llm_error_exhausts_retries(self) -> None:
        """Persistent transient errors eventually propagate after retries."""
        generator = AsyncMock()
        generator.generate.side_effect = LLMTimeoutError("always timing out")
        generator.last_tokens_used = None

        orchestrator = QueryOrchestrator(
            sql_generator=generator,
            sql_validator=passing_validator(),
            sql_executor=AsyncMock(),
            result_validator=MagicMock(),
            schema_cache=make_cache(make_schema()),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=1, retry_delay=0.1),
            validation_config=ValidationConfig(),
        )

        with pytest.raises(LLMTimeoutError):
            await orchestrator._generate_sql_with_retry(
                question="get user", schema=make_schema(), request_id="req-2"
            )
        assert generator.generate.call_count == 2  # initial + one retry


class TestResultValidationThreshold:
    """min_confidence_score is the operative acceptability threshold."""

    @pytest.mark.asyncio
    async def test_confidence_flows_from_result_validator(self) -> None:
        """The confidence returned by the result validator is used in the response."""
        executor = AsyncMock()
        executor.execute.return_value = ([{"id": 1}], 1)

        result_validator = AsyncMock()
        result_validator.validate.return_value = ResultValidationResult(
            confidence=82,
            explanation="looks right",
            suggestion=None,
            is_acceptable=True,
        )

        orchestrator = QueryOrchestrator(
            sql_generator=make_generator(),
            sql_validator=passing_validator(),
            sql_executor=executor,
            result_validator=result_validator,
            schema_cache=make_cache(make_schema()),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=True, min_confidence_score=70),
        )

        response = await orchestrator.execute_query(
            QueryRequest(question="get user", database="test_db", return_type=ReturnType.RESULT)
        )

        assert response.success is True
        assert response.confidence == 82
