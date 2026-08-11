"""Reusable bounded retry and timeout policies for infrastructure calls."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field


class ExternalCallError(RuntimeError):
    """Base typed error for PartyPilot external-call reliability failures."""


class RetryExhaustedError(ExternalCallError):
    """Raised after all configured attempts fail with retryable errors."""

    def __init__(self, *, attempts: int, last_error: Exception) -> None:
        super().__init__(
            f"external call failed after {attempts} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        )
        self.attempts = attempts
        self.last_error = last_error


class RetryPolicy(BaseModel):
    """Validated retry policy with a bounded number of retries."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_retries: int = Field(default=0, ge=0, le=10)

    @property
    def max_attempts(self) -> int:
        """Total call attempts, including the initial attempt."""
        return self.max_retries + 1


@dataclass(frozen=True, slots=True)
class ExternalCallPolicy:
    """Explicit timeout and retry behavior for one external-call boundary."""

    timeout_seconds: float
    retry: RetryPolicy = field(default_factory=RetryPolicy)

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


def call_with_retry[T](
    operation: Callable[[float], T],
    *,
    policy: ExternalCallPolicy,
    retryable_exceptions: tuple[type[Exception], ...],
) -> T:
    """Execute an external call with an explicit timeout and bounded retries.

    ``operation`` receives the timeout for every attempt. Exceptions not listed
    in ``retryable_exceptions`` propagate immediately. Exhaustion of a retryable
    failure sequence is translated into ``RetryExhaustedError``.
    """

    last_error: Exception | None = None
    for attempt in range(1, policy.retry.max_attempts + 1):
        try:
            return operation(policy.timeout_seconds)
        except retryable_exceptions as error:
            last_error = error
            if attempt == policy.retry.max_attempts:
                raise RetryExhaustedError(attempts=attempt, last_error=error) from error

    # The loop always returns or raises. This is retained as a defensive guard.
    if last_error is not None:  # pragma: no cover
        raise RetryExhaustedError(
            attempts=policy.retry.max_attempts,
            last_error=last_error,
        ) from last_error
    raise ExternalCallError("external call did not execute")  # pragma: no cover
