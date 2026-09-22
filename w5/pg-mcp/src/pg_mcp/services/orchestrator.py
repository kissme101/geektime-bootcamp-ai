"""Query orchestrator for coordinating the complete query flow.

This module provides the QueryOrchestrator class that coordinates all components
of the query processing pipeline: SQL generation, validation, execution, and result
validation. It implements retry logic with exponential backoff, rate limiting,
metrics collection, request tracing, and error handling.
"""

import asyncio
import logging
from typing import Any

from asyncpg import Pool

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseError,
    ErrorCode,
    LLMError,
    LLMTimeoutError,
    LLMUnavailableError,
    PgMcpError,
    QuestionTooLongError,
    RateLimitExceededError,
    SchemaLoadError,
    SecurityViolationError,
    SQLParseError,
)
from pg_mcp.models.query import (
    ErrorDetail,
    QueryRequest,
    QueryResponse,
    QueryResult,
    ReturnType,
    ValidationResult,
)
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.observability.tracing import generate_request_id, set_request_id
from pg_mcp.resilience.circuit_breaker import CircuitBreaker
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_generator import SQLGenerator
from pg_mcp.services.sql_validator import SQLValidator

logger = logging.getLogger(__name__)


class QueryOrchestrator:
    """Orchestrates the complete query processing pipeline.

    This class coordinates SQL generation, validation, execution, and result
    validation across one or more databases. It implements retry logic with
    error feedback and exponential backoff, the circuit breaker pattern for
    fault tolerance, rate limiting to protect shared resources, Prometheus
    metrics collection, and comprehensive error handling.

    Example:
        >>> orchestrator = QueryOrchestrator(
        ...     sql_generator=generator,
        ...     sql_validator=validator,
        ...     result_validator=result_validator,
        ...     schema_cache=cache,
        ...     pools={"mydb": pool},
        ...     resilience_config=resilience_config,
        ...     validation_config=validation_config,
        ...     sql_executors={"mydb": executor},
        ...     metrics=metrics,
        ...     rate_limiter=rate_limiter,
        ... )
        >>> response = await orchestrator.execute_query(QueryRequest(
        ...     question="How many users?",
        ...     database="mydb"
        ... ))
    """

    def __init__(
        self,
        sql_generator: SQLGenerator,
        sql_validator: SQLValidator,
        sql_executor: SQLExecutor | None,
        result_validator: ResultValidator,
        schema_cache: SchemaCache,
        pools: dict[str, Pool],
        resilience_config: ResilienceConfig,
        validation_config: ValidationConfig,
        sql_executors: dict[str, SQLExecutor] | None = None,
        metrics: MetricsCollector | None = None,
        rate_limiter: MultiRateLimiter | None = None,
    ) -> None:
        """Initialize query orchestrator.

        Args:
            sql_generator: SQL generation service.
            sql_validator: SQL validation service.
            sql_executor: Fallback SQL execution service used when ``sql_executors``
                does not contain the resolved database. Kept for backward
                compatibility and single-database deployments; may be ``None`` when
                ``sql_executors`` covers every database.
            result_validator: Result validation service.
            schema_cache: Schema cache instance.
            pools: Dictionary mapping database names to connection pools.
            resilience_config: Resilience configuration for retries, backoff,
                circuit breaker, and rate limiting.
            validation_config: Validation configuration including thresholds.
            sql_executors: Mapping of database name to its dedicated SQL executor.
                This is what makes multi-database routing correct: each database is
                executed against its own pool. When omitted, ``sql_executor`` is
                used for every database (single-database mode).
            metrics: Optional metrics collector. When provided, query, LLM, database,
                and security metrics are recorded throughout the pipeline.
            rate_limiter: Optional rate limiter. When provided, LLM generation and
                query execution are gated to protect shared resources.
        """
        self.sql_generator = sql_generator
        self.sql_validator = sql_validator
        self.result_validator = result_validator
        self.schema_cache = schema_cache
        self.pools = pools
        self.resilience_config = resilience_config
        self.validation_config = validation_config

        # Per-database executors make multi-database routing correct. A single
        # fallback executor keeps single-database deployments and existing callers
        # working unchanged.
        self.sql_executors: dict[str, SQLExecutor] = dict(sql_executors or {})
        self._fallback_executor = sql_executor
        # Backward-compatible alias: some callers/tests reference ``sql_executor``.
        self.sql_executor = sql_executor

        self.metrics = metrics
        self.rate_limiter = rate_limiter

        # Create circuit breaker for LLM calls
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=resilience_config.circuit_breaker_threshold,
            recovery_timeout=resilience_config.circuit_breaker_timeout,
        )

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Execute complete query flow from question to results.

        This method orchestrates the entire pipeline:
        1. Generate request_id for tracing (propagated via contextvar)
        2. Enforce the configured maximum question length
        3. Resolve and validate the database name
        4. Load schema from cache (recording cache age)
        5. Generate and validate SQL with retry logic and backoff
        6. Execute SQL against the database's own executor (if return_type == RESULT)
        7. Validate results (optional)
        8. Record metrics and return a structured response

        Args:
            request: Query request containing question and parameters.

        Returns:
            QueryResponse: Complete response with SQL, results, or error information.
        """
        # Generate request_id for full-chain tracing and publish it to the
        # tracing context so downstream logs are automatically correlated.
        request_id = generate_request_id()
        set_request_id(request_id)
        start_time = self._get_current_time_ms()
        database_name = request.database or "unknown"

        logger.info(
            "Starting query execution",
            extra={"request_id": request_id, "question": request.question[:100]},
        )

        try:
            # Step 1: Enforce configured maximum question length.
            self._check_question_length(request.question)

            # Step 2: Resolve database name
            database_name = self._resolve_database(request.database)
            logger.debug(
                "Resolved database",
                extra={"request_id": request_id, "database": database_name},
            )

            # Step 3: Get schema from cache
            schema = await self._get_schema(database_name, request_id)

            # Step 4: Generate and validate SQL with retry logic
            generated_sql, validation_result, tokens_used = await self._generate_sql_with_retry(
                question=request.question,
                schema=schema,
                request_id=request_id,
            )

            # Step 5: If return_type is SQL, return early
            if request.return_type == ReturnType.SQL:
                logger.info(
                    "Returning SQL only",
                    extra={"request_id": request_id, "sql_length": len(generated_sql)},
                )
                self._record_query_result("success", database_name, start_time)
                return QueryResponse(
                    success=True,
                    generated_sql=generated_sql,
                    validation=validation_result,
                    data=None,
                    error=None,
                    confidence=100,
                    tokens_used=tokens_used,
                )

            # Step 6: Execute SQL against the database's own executor
            logger.debug("Executing SQL", extra={"request_id": request_id})
            results, total_count, execution_time_ms = await self._execute_sql(
                database_name=database_name,
                sql=generated_sql,
                request_id=request_id,
            )

            # Step 7: Validate results (non-blocking, failures don't fail the request)
            result_confidence = await self._validate_results_safely(
                question=request.question,
                sql=generated_sql,
                results=results,
                row_count=total_count,
                request_id=request_id,
            )

            # Step 8: Build successful response
            query_result = QueryResult(
                columns=list(results[0].keys()) if results else [],
                rows=results,
                row_count=len(results),  # Limited row count (after max_rows applied)
                execution_time_ms=execution_time_ms,
            )

            self._record_query_result("success", database_name, start_time)
            return QueryResponse(
                success=True,
                generated_sql=generated_sql,
                validation=validation_result,
                data=query_result,
                error=None,
                confidence=result_confidence,
                tokens_used=tokens_used,
            )

        except PgMcpError as e:
            # Handle known application errors
            logger.warning(
                "Query execution failed with known error",
                extra={
                    "request_id": request_id,
                    "error_code": e.code,
                    "error_message": str(e),
                },
            )
            self._record_query_result(e.code.value, database_name, start_time)
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=e.code.value,
                    message=e.message,
                    details=e.details,
                ),
                confidence=0,
                tokens_used=None,
            )
        except Exception as e:
            # Handle unexpected errors
            logger.exception(
                "Query execution failed with unexpected error",
                extra={"request_id": request_id},
            )
            self._record_query_result("internal_error", database_name, start_time)
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=ErrorCode.INTERNAL_ERROR.value,
                    message=f"Internal server error: {e!s}",
                    details={"error_type": type(e).__name__},
                ),
                confidence=0,
                tokens_used=None,
            )

    def _check_question_length(self, question: str) -> None:
        """Enforce the configured maximum question length.

        Args:
            question: The user's natural language question.

        Raises:
            QuestionTooLongError: If the question exceeds ``max_question_length``.
        """
        max_length = self.validation_config.max_question_length
        if len(question) > max_length:
            raise QuestionTooLongError(
                message=(
                    f"Question length {len(question)} exceeds the maximum "
                    f"allowed length of {max_length} characters"
                ),
                details={"length": len(question), "max_length": max_length},
            )

    async def _get_schema(self, database_name: str, request_id: str) -> Any:
        """Load a database schema from cache (loading on miss) and record its age.

        Args:
            database_name: Resolved database name.
            request_id: Request ID for tracking.

        Returns:
            The database schema.

        Raises:
            DatabaseError: If no connection pool exists for the database.
            SchemaLoadError: If schema introspection fails.
        """
        schema = self.schema_cache.get(database_name)
        if schema is None:
            pool = self.pools.get(database_name)
            if pool is None:
                raise DatabaseError(
                    message=f"No connection pool available for database '{database_name}'",
                    details={"database": database_name},
                )
            try:
                schema = await self.schema_cache.load(database_name, pool)
            except Exception as e:
                raise SchemaLoadError(
                    message=f"Failed to load schema for database '{database_name}': {e!s}",
                    details={"database": database_name, "error": str(e)},
                ) from e

        # Record schema cache age for observability.
        if self.metrics is not None:
            age = self.schema_cache.get_cache_age(database_name)
            if age is not None:
                self.metrics.set_schema_cache_age(database_name, age)

        logger.debug(
            "Schema loaded",
            extra={
                "request_id": request_id,
                "database": database_name,
                "tables": len(schema.tables),
            },
        )
        return schema

    def _resolve_database(self, database: str | None) -> str:
        """Resolve database name from request or auto-select.

        If database is specified, validate it exists.
        If not specified and only one database available, auto-select it.

        Args:
            database: Database name from request (optional).

        Returns:
            str: Resolved database name.

        Raises:
            DatabaseError: If database is invalid or cannot be auto-selected.
        """
        if database is not None:
            # Validate specified database exists
            if database not in self.pools:
                raise DatabaseError(
                    message=f"Database '{database}' not found",
                    details={
                        "requested_database": database,
                        "available_databases": list(self.pools.keys()),
                    },
                )
            return database

        # Auto-select if only one database available
        available_dbs = list(self.pools.keys())
        if len(available_dbs) == 0:
            raise DatabaseError(
                message="No databases configured",
                details={},
            )
        if len(available_dbs) == 1:
            return available_dbs[0]

        # Multiple databases, must specify
        raise DatabaseError(
            message="Multiple databases available, please specify which to query",
            details={"available_databases": available_dbs},
        )

    def _resolve_executor(self, database_name: str) -> SQLExecutor:
        """Return the SQL executor bound to the given database.

        This is the key to correct multi-database routing: each database is
        executed against its own connection pool, never the primary pool.

        Args:
            database_name: Resolved database name.

        Returns:
            SQLExecutor: The executor for that database.

        Raises:
            DatabaseError: If no executor is available for the database.
        """
        executor = self.sql_executors.get(database_name, self._fallback_executor)
        if executor is None:
            raise DatabaseError(
                message=f"No SQL executor available for database '{database_name}'",
                details={
                    "database": database_name,
                    "available_executors": list(self.sql_executors.keys()),
                },
            )
        return executor

    async def _execute_sql(
        self,
        database_name: str,
        sql: str,
        request_id: str,
    ) -> tuple[list[dict[str, Any]], int, float]:
        """Execute SQL against the database's own executor, with rate limiting.

        Args:
            database_name: Resolved database name.
            sql: Validated SQL to execute.
            request_id: Request ID for tracking.

        Returns:
            tuple: (results, total_count, execution_time_ms)

        Raises:
            RateLimitExceededError: If a query slot cannot be acquired in time.
            DatabaseError / ExecutionTimeoutError: On execution failures.
        """
        executor = self._resolve_executor(database_name)
        start = self._get_current_time_ms()

        async def _run() -> tuple[list[dict[str, Any]], int]:
            return await executor.execute(sql)

        if self.rate_limiter is not None:
            try:
                async with self.rate_limiter.for_queries(
                    timeout=self.resilience_config.rate_limit_timeout
                ):
                    results, total_count = await _run()
            except TimeoutError as e:
                raise RateLimitExceededError(
                    message="Query rate limit exceeded; please retry shortly",
                    details={"resource": "queries"},
                ) from e
        else:
            results, total_count = await _run()

        execution_time_ms = self._get_current_time_ms() - start

        if self.metrics is not None:
            self.metrics.observe_db_query_duration(execution_time_ms / 1000.0)

        logger.info(
            "SQL executed successfully",
            extra={
                "request_id": request_id,
                "database": database_name,
                "row_count": total_count,
                "execution_time_ms": execution_time_ms,
            },
        )
        return results, total_count, execution_time_ms

    async def _call_generator(
        self,
        question: str,
        schema: Any,
        previous_sql: str | None,
        error_feedback: str | None,
    ) -> str:
        """Call the SQL generator, applying LLM rate limiting and metrics.

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            previous_sql: Previously generated SQL that failed (for retry).
            error_feedback: Error message from the previous attempt (for retry).

        Returns:
            str: Generated SQL.

        Raises:
            RateLimitExceededError: If an LLM slot cannot be acquired in time.
        """
        if self.metrics is not None:
            self.metrics.increment_llm_call("generate_sql")

        llm_start = self._get_current_time_ms()

        async def _run() -> str:
            return await self.sql_generator.generate(
                question=question,
                schema=schema,
                previous_attempt=previous_sql,
                error_feedback=error_feedback,
            )

        if self.rate_limiter is not None:
            try:
                async with self.rate_limiter.for_llm(
                    timeout=self.resilience_config.rate_limit_timeout
                ):
                    generated_sql = await _run()
            except TimeoutError as e:
                raise RateLimitExceededError(
                    message="LLM rate limit exceeded; please retry shortly",
                    details={"resource": "llm"},
                ) from e
        else:
            generated_sql = await _run()

        if self.metrics is not None:
            self.metrics.observe_llm_latency(
                "generate_sql", (self._get_current_time_ms() - llm_start) / 1000.0
            )

        return generated_sql

    async def _generate_sql_with_retry(
        self,
        question: str,
        schema: Any,
        request_id: str,
    ) -> tuple[str, ValidationResult, int | None]:
        """Generate and validate SQL with retry logic and exponential backoff.

        This method implements a retry loop that:
        1. Checks circuit breaker state (fails fast if open)
        2. Generates SQL using the LLM (rate limited, metered)
        3. Retries transient LLM errors with exponential backoff
        4. Validates the generated SQL
        5. On validation failure, retries with error feedback
        6. Records success/failure to the circuit breaker

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            request_id: Request ID for tracking.

        Returns:
            tuple: (generated_sql, validation_result, tokens_used)

        Raises:
            LLMError: If circuit breaker is open or generation fails.
            SecurityViolationError: If SQL fails validation after all retries.
            SQLParseError: If SQL cannot be parsed after all retries.
            RateLimitExceededError: If an LLM slot cannot be acquired in time.
        """
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise LLMError(
                message="SQL generation service is temporarily unavailable (circuit breaker open)",
                details={
                    "circuit_state": self.circuit_breaker.state,
                    "failure_count": self.circuit_breaker.failure_count,
                },
            )

        previous_sql: str | None = None
        error_feedback: str | None = None
        max_retries = self.resilience_config.max_retries
        delay = self.resilience_config.retry_delay
        backoff = self.resilience_config.backoff_factor
        tokens_used: int | None = None
        last_transient: LLMError | None = None

        for attempt in range(max_retries + 1):
            logger.debug(
                "Generating SQL",
                extra={
                    "request_id": request_id,
                    "attempt": attempt + 1,
                    "max_retries": max_retries + 1,
                },
            )

            # --- Generate SQL (handle transient vs fatal errors) ---
            try:
                generated_sql = await self._call_generator(
                    question, schema, previous_sql, error_feedback
                )
            except (LLMTimeoutError, LLMUnavailableError) as transient_error:
                # Transient LLM error: record failure, back off, and retry.
                self.circuit_breaker.record_failure()
                last_transient = transient_error
                if attempt < max_retries:
                    logger.warning(
                        "Transient LLM error, retrying with backoff",
                        extra={
                            "request_id": request_id,
                            "attempt": attempt + 1,
                            "delay_seconds": delay,
                            "error": str(transient_error),
                        },
                    )
                    await asyncio.sleep(delay)
                    delay *= backoff
                    continue
                logger.error(
                    "Transient LLM error persisted after all retries",
                    extra={"request_id": request_id, "attempts": attempt + 1},
                )
                raise
            except (RateLimitExceededError, LLMError):
                # Rate-limit rejections and non-transient LLM errors fail fast.
                self.circuit_breaker.record_failure()
                raise
            except Exception as e:
                # Unexpected error during generation.
                self.circuit_breaker.record_failure()
                logger.exception(
                    "Unexpected error during SQL generation",
                    extra={"request_id": request_id},
                )
                raise LLMError(
                    message=f"SQL generation failed unexpectedly: {e!s}",
                    details={"error_type": type(e).__name__},
                ) from e

            # Capture token usage reported by the generator (if any).
            reported = getattr(self.sql_generator, "last_tokens_used", None)
            if isinstance(reported, int):
                tokens_used = reported
                if self.metrics is not None:
                    self.metrics.increment_llm_tokens("generate_sql", reported)

            logger.debug(
                "SQL generated",
                extra={"request_id": request_id, "sql_length": len(generated_sql)},
            )

            # --- Validate SQL ---
            try:
                self.sql_validator.validate_or_raise(generated_sql)
            except (SecurityViolationError, SQLParseError) as validation_error:
                if self.metrics is not None:
                    reason = getattr(getattr(validation_error, "code", None), "value", "validation")
                    self.metrics.increment_sql_rejected(reason)
                if attempt < max_retries:
                    # Record as failure and retry with feedback
                    logger.warning(
                        "SQL validation failed, retrying with feedback",
                        extra={
                            "request_id": request_id,
                            "attempt": attempt + 1,
                            "error": str(validation_error),
                        },
                    )
                    previous_sql = generated_sql
                    error_feedback = str(validation_error)
                    continue
                # Out of retries, record failure and raise
                self.circuit_breaker.record_failure()
                logger.error(
                    "SQL validation failed after all retries",
                    extra={
                        "request_id": request_id,
                        "attempts": attempt + 1,
                        "error": str(validation_error),
                    },
                )
                raise

            # --- Validation successful ---
            self.circuit_breaker.record_success()
            logger.info(
                "SQL generated and validated successfully",
                extra={"request_id": request_id, "attempts": attempt + 1},
            )
            validation_result = ValidationResult(
                is_valid=True,
                is_select=True,
                allows_data_modification=False,
                uses_blocked_functions=[],
                error_message=None,
            )
            return generated_sql, validation_result, tokens_used

        # Loop exhausted (only reachable if every attempt was a transient error
        # that was retried, then the final one continued instead of raising).
        self.circuit_breaker.record_failure()
        if last_transient is not None:
            raise last_transient
        raise LLMError(
            message="SQL generation failed after all retry attempts",
            details={"max_retries": max_retries},
        )

    async def _validate_results_safely(
        self,
        question: str,
        sql: str,
        results: list[dict[str, Any]],
        row_count: int,
        request_id: str,
    ) -> int:
        """Validate query results with error handling (non-blocking).

        This method attempts to validate results using the LLM, but failures
        don't cause the overall query to fail. Returns a confidence score.

        Args:
            question: User's original question.
            sql: Generated SQL query.
            results: Query results.
            row_count: Total row count.
            request_id: Request ID for tracking.

        Returns:
            int: Confidence score (0-100). Returns 100 if validation disabled/fails.
        """
        if not self.validation_config.enabled:
            return 100

        try:
            logger.debug("Validating results", extra={"request_id": request_id})

            if self.metrics is not None:
                self.metrics.increment_llm_call("validate_result")

            validation_result = await self.result_validator.validate(
                question=question,
                sql=sql,
                results=results,
                row_count=row_count,
            )

            logger.info(
                "Result validation completed",
                extra={
                    "request_id": request_id,
                    "confidence": validation_result.confidence,
                    "is_acceptable": validation_result.is_acceptable,
                },
            )

            return validation_result.confidence

        except Exception as e:
            # Log but don't fail the query
            logger.warning(
                "Result validation failed, continuing with default confidence",
                extra={"request_id": request_id, "error": str(e)},
            )
            return 100  # Default to high confidence if validation fails

    def _record_query_result(
        self,
        status: str,
        database: str,
        start_time_ms: float,
    ) -> None:
        """Record query request count and total duration metrics.

        Args:
            status: Outcome status label (e.g. "success" or an error code).
            database: Database name (or "unknown" if resolution failed).
            start_time_ms: Query start time in milliseconds.
        """
        if self.metrics is None:
            return
        duration_seconds = (self._get_current_time_ms() - start_time_ms) / 1000.0
        self.metrics.increment_query_request(status=status, database=database)
        self.metrics.query_duration.observe(duration_seconds)

    @staticmethod
    def _get_current_time_ms() -> float:
        """Get current time in milliseconds.

        Returns:
            float: Current time in milliseconds since epoch.
        """
        import time

        return time.time() * 1000
