"""Unit tests for the MCP server's ``query`` tool.

These tests exercise the tool's request handling and error branches WITHOUT a
database or OpenAI API by mocking the module-level orchestrator. They cover the
paths the e2e tests (which require a live database) cannot run in CI.
"""

from unittest.mock import AsyncMock, patch

import pytest

import pg_mcp.server as server
from pg_mcp.models.query import (
    ErrorDetail,
    QueryResponse,
    QueryResult,
    ValidationResult,
)


@pytest.fixture
def restore_orchestrator():
    """Save and restore the module-level orchestrator around each test."""
    original = server._orchestrator
    yield
    server._orchestrator = original


class TestQueryToolGuards:
    """Guard clauses that don't require an orchestrator to be wired."""

    @pytest.mark.asyncio
    async def test_server_not_initialized(self, restore_orchestrator) -> None:
        """When the orchestrator is missing, a clear error is returned."""
        server._orchestrator = None
        result = await server.query(question="anything")
        assert result["success"] is False
        assert result["error"]["code"] == "SERVER_NOT_INITIALIZED"

    @pytest.mark.asyncio
    async def test_invalid_return_type(self, restore_orchestrator) -> None:
        """An unsupported return_type is rejected before any work is done."""
        server._orchestrator = AsyncMock()  # present, but should not be called
        result = await server.query(question="anything", return_type="csv")
        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_PARAMETER"

    @pytest.mark.asyncio
    async def test_empty_question_is_invalid_request(self, restore_orchestrator) -> None:
        """A blank question fails request construction with INVALID_REQUEST."""
        server._orchestrator = AsyncMock()
        result = await server.query(question="   ")
        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_REQUEST"


class TestQueryToolDelegation:
    """The tool delegates to the orchestrator and serializes the response."""

    @pytest.mark.asyncio
    async def test_successful_sql_only(self, restore_orchestrator) -> None:
        """A successful SQL-only response is returned as a dict with tokens_used."""
        mock_orch = AsyncMock()
        mock_orch.execute_query.return_value = QueryResponse(
            success=True,
            generated_sql="SELECT 1;",
            validation=ValidationResult(is_valid=True, is_select=True),
            confidence=100,
            tokens_used=42,
        )
        server._orchestrator = mock_orch

        result = await server.query(question="count", return_type="sql")

        assert result["success"] is True
        assert result["generated_sql"] == "SELECT 1;"
        assert result["tokens_used"] == 42
        mock_orch.execute_query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_successful_result_has_tokens_used_default(self, restore_orchestrator) -> None:
        """Even without token metadata, tokens_used is present (0)."""
        mock_orch = AsyncMock()
        mock_orch.execute_query.return_value = QueryResponse(
            success=True,
            generated_sql="SELECT id FROM users;",
            data=QueryResult(columns=["id"], rows=[{"id": 1}], row_count=1),
            confidence=90,
            tokens_used=None,
        )
        server._orchestrator = mock_orch

        result = await server.query(question="users", return_type="result")

        assert result["success"] is True
        assert result["tokens_used"] == 0
        assert result["data"]["row_count"] == 1

    @pytest.mark.asyncio
    async def test_error_response_serialized(self, restore_orchestrator) -> None:
        """An orchestrator error response is serialized with its code."""
        mock_orch = AsyncMock()
        mock_orch.execute_query.return_value = QueryResponse(
            success=False,
            error=ErrorDetail(code="security_violation", message="DELETE not allowed"),
        )
        server._orchestrator = mock_orch

        result = await server.query(question="delete everything")

        assert result["success"] is False
        assert result["error"]["code"] == "security_violation"
        assert result["tokens_used"] == 0

    @pytest.mark.asyncio
    async def test_unexpected_exception_is_caught(self, restore_orchestrator) -> None:
        """An unexpected exception from the orchestrator becomes INTERNAL_ERROR."""
        mock_orch = AsyncMock()
        mock_orch.execute_query.side_effect = RuntimeError("kaboom")
        server._orchestrator = mock_orch

        result = await server.query(question="boom")

        assert result["success"] is False
        assert result["error"]["code"] == "INTERNAL_ERROR"
        assert result["tokens_used"] == 0

    @pytest.mark.asyncio
    async def test_database_parameter_forwarded(self, restore_orchestrator) -> None:
        """The database parameter is forwarded to the orchestrator request."""
        mock_orch = AsyncMock()
        mock_orch.execute_query.return_value = QueryResponse(
            success=True, generated_sql="SELECT 1;", confidence=100, tokens_used=0
        )
        server._orchestrator = mock_orch

        await server.query(question="count", database="analytics", return_type="sql")

        request = mock_orch.execute_query.call_args.args[0]
        assert request.database == "analytics"


class TestLifespanShutdownSafety:
    """The lifespan shutdown path is defensive when startup fails early."""

    @pytest.mark.asyncio
    async def test_lifespan_startup_failure_still_cleans_up(self, restore_orchestrator) -> None:
        """If pool creation fails, shutdown still runs without masking the error."""
        # Reset module globals so the finally block sees a clean slate.
        server._pools = None
        server._schema_cache = None

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key-12345"}),
            patch.object(server, "create_pool", new=AsyncMock(side_effect=RuntimeError("no db"))),
            pytest.raises(RuntimeError, match="no db"),
        ):
            async with server.lifespan(server.mcp):
                pass
