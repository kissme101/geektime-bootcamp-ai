"""Unit tests for request tracing and context propagation."""

import pytest

from pg_mcp.observability.tracing import (
    TraceContext,
    TracingLogger,
    clear_request_id,
    generate_request_id,
    get_request_id,
    get_tracing_logger,
    request_context,
    set_request_id,
    trace_async,
    trace_sync,
)


class TestRequestId:
    """Request ID generation and context-variable management."""

    def test_generate_request_id_is_unique(self) -> None:
        assert generate_request_id() != generate_request_id()

    def test_set_and_get_request_id(self) -> None:
        set_request_id("abc-123")
        assert get_request_id() == "abc-123"

    def test_clear_request_id(self) -> None:
        set_request_id("to-clear")
        clear_request_id()
        assert get_request_id() is None


class TestRequestContext:
    """The async request_context manager sets and restores the request id."""

    @pytest.mark.asyncio
    async def test_generates_id_when_none_given(self) -> None:
        clear_request_id()
        async with request_context() as req_id:
            assert req_id is not None
            assert get_request_id() == req_id
        # Restored to previous (None) afterwards.
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_uses_provided_id(self) -> None:
        async with request_context("fixed-id") as req_id:
            assert req_id == "fixed-id"
            assert get_request_id() == "fixed-id"

    @pytest.mark.asyncio
    async def test_restores_previous_id(self) -> None:
        set_request_id("outer")
        async with request_context("inner"):
            assert get_request_id() == "inner"
        assert get_request_id() == "outer"


class TestTraceDecorators:
    """trace_async / trace_sync run the wrapped function and pass through results."""

    @pytest.mark.asyncio
    async def test_trace_async_with_context(self) -> None:
        @trace_async(operation="op")
        async def add(a: int, b: int) -> int:
            assert get_request_id() == "ctx-1"
            return a + b

        async with request_context("ctx-1"):
            assert await add(2, 3) == 5

    @pytest.mark.asyncio
    async def test_trace_async_without_context(self) -> None:
        clear_request_id()

        @trace_async()
        async def echo(x: str) -> str:
            return x

        assert await echo("hi") == "hi"

    def test_trace_sync_with_context(self) -> None:
        @trace_sync(operation="sync-op")
        def mul(a: int, b: int) -> int:
            return a * b

        set_request_id("ctx-2")
        try:
            assert mul(4, 5) == 20
        finally:
            clear_request_id()

    def test_trace_sync_without_context(self) -> None:
        clear_request_id()

        @trace_sync()
        def const() -> int:
            return 42

        assert const() == 42


class TestTraceContextModel:
    """The TraceContext pydantic model."""

    def test_minimal(self) -> None:
        ctx = TraceContext(request_id="r1")
        assert ctx.request_id == "r1"
        assert ctx.parent_id is None

    def test_full(self) -> None:
        ctx = TraceContext(request_id="r1", parent_id="p0", operation="query", metadata={"db": "x"})
        assert ctx.operation == "query"
        assert ctx.metadata == {"db": "x"}


class TestTracingLogger:
    """The TracingLogger wrapper injects request context into log calls."""

    def test_logger_methods_do_not_raise(self) -> None:
        logger = get_tracing_logger("test.logger")
        assert isinstance(logger, TracingLogger)
        set_request_id("log-ctx")
        try:
            logger.debug("debug")
            logger.info("info")
            logger.warning("warning")
            logger.error("error")
            logger.critical("critical")
        finally:
            clear_request_id()

    def test_exception_logging(self) -> None:
        logger = get_tracing_logger("test.logger.exc")
        try:
            raise ValueError("boom")
        except ValueError:
            # Should attach exc_info and not raise.
            logger.exception("handled")
