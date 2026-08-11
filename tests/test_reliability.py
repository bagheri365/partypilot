"""Tests for explicit timeout and bounded retry policies."""

from __future__ import annotations

import pytest

from partypilot.adapters.reliability import (
    ExternalCallPolicy,
    RetryExhaustedError,
    RetryPolicy,
    call_with_retry,
)


class TransientError(RuntimeError):
    pass


class PermanentError(RuntimeError):
    pass


def test_success_receives_explicit_timeout_once() -> None:
    timeouts: list[float] = []

    def operation(timeout_seconds: float) -> str:
        timeouts.append(timeout_seconds)
        return "ok"

    result = call_with_retry(
        operation,
        policy=ExternalCallPolicy(timeout_seconds=2.5, retry=RetryPolicy(max_retries=3)),
        retryable_exceptions=(TransientError,),
    )

    assert result == "ok"
    assert timeouts == [2.5]


def test_transient_failure_then_success_retries_with_same_timeout() -> None:
    attempts = 0
    timeouts: list[float] = []

    def operation(timeout_seconds: float) -> str:
        nonlocal attempts
        attempts += 1
        timeouts.append(timeout_seconds)
        if attempts == 1:
            raise TransientError("temporary")
        return "recovered"

    result = call_with_retry(
        operation,
        policy=ExternalCallPolicy(timeout_seconds=1.25, retry=RetryPolicy(max_retries=2)),
        retryable_exceptions=(TransientError,),
    )

    assert result == "recovered"
    assert attempts == 2
    assert timeouts == [1.25, 1.25]


def test_permanent_failure_stops_immediately() -> None:
    attempts = 0

    def operation(_: float) -> str:
        nonlocal attempts
        attempts += 1
        raise PermanentError("do not retry")

    with pytest.raises(PermanentError, match="do not retry"):
        call_with_retry(
            operation,
            policy=ExternalCallPolicy(timeout_seconds=1, retry=RetryPolicy(max_retries=5)),
            retryable_exceptions=(TransientError,),
        )

    assert attempts == 1


def test_retry_exhaustion_is_typed_and_bounded() -> None:
    attempts = 0

    def operation(_: float) -> str:
        nonlocal attempts
        attempts += 1
        raise TransientError("still unavailable")

    with pytest.raises(RetryExhaustedError) as exc_info:
        call_with_retry(
            operation,
            policy=ExternalCallPolicy(timeout_seconds=3, retry=RetryPolicy(max_retries=2)),
            retryable_exceptions=(TransientError,),
        )

    assert attempts == 3
    assert exc_info.value.attempts == 3
    assert isinstance(exc_info.value.last_error, TransientError)


def test_policy_rejects_missing_or_unbounded_timeout_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        ExternalCallPolicy(timeout_seconds=0)

    with pytest.raises(ValueError):
        RetryPolicy(max_retries=11)
