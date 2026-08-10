"""Provider-neutral LLM generation contracts for PartyPilot."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


class StructuredOutputExpectation(BaseModel):
    """Expected structured-output shape for a generation request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_name: str
    json_schema: dict[str, JsonValue]

    @field_validator("schema_name")
    @classmethod
    def validate_schema_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("schema_name cannot be blank")
        return normalized


class GenerationRequest(BaseModel):
    """Provider-neutral generation input."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: str
    system_prompt: str | None = None
    structured_output: StructuredOutputExpectation | None = None

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt cannot be blank")
        return value

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("system_prompt cannot be blank")
        return value


class UsageMetadata(BaseModel):
    """Optional usage details reported by a provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class GenerationResponse(BaseModel):
    """Provider-neutral generation result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    structured_output: JsonValue | None = None
    usage: UsageMetadata | None = None


class LLMProvider(Protocol):
    """Port implemented by concrete LLM providers."""

    def generate(
        self,
        request: GenerationRequest,
        *,
        timeout_seconds: float,
    ) -> GenerationResponse:
        """Generate a response within the supplied timeout."""
        ...


class FakeLLMProvider:
    """Deterministic queued-response provider for tests."""

    def __init__(self, responses: Iterable[GenerationResponse]) -> None:
        self._responses = deque(responses)
        self.requests: list[tuple[GenerationRequest, float]] = []

    def generate(
        self,
        request: GenerationRequest,
        *,
        timeout_seconds: float,
    ) -> GenerationResponse:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.requests.append((request, timeout_seconds))
        if not self._responses:
            raise RuntimeError("fake provider has no queued response")
        return self._responses.popleft()


class FailingFakeLLMProvider:
    """Deterministic provider that always raises a configured exception."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.requests: list[tuple[GenerationRequest, float]] = []

    def generate(
        self,
        request: GenerationRequest,
        *,
        timeout_seconds: float,
    ) -> GenerationResponse:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.requests.append((request, timeout_seconds))
        raise self._error
