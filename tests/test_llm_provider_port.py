"""Tests for the provider-neutral LLM port and deterministic fakes."""

from typing import assert_type

import pytest
from pydantic import ValidationError

from partypilot.ports import (
    FailingFakeLLMProvider,
    FakeLLMProvider,
    GenerationRequest,
    GenerationResponse,
    LLMProvider,
    StructuredOutputExpectation,
    UsageMetadata,
)


def test_generation_request_supports_structured_output_expectation() -> None:
    expectation = StructuredOutputExpectation(
        schema_name="party_plan",
        json_schema={
            "type": "object",
            "properties": {"venue_id": {"type": "string"}},
            "required": ["venue_id"],
        },
    )
    request = GenerationRequest(
        prompt="Plan a party",
        system_prompt="Return only valid structured output.",
        structured_output=expectation,
    )

    assert request.structured_output == expectation


def test_usage_metadata_supports_optional_token_counts() -> None:
    usage = UsageMetadata(input_tokens=10, output_tokens=4, total_tokens=14)
    response = GenerationResponse(text='{"venue_id":"venue-1"}', usage=usage)

    assert response.usage == usage


def test_usage_metadata_rejects_negative_tokens() -> None:
    with pytest.raises(ValidationError):
        UsageMetadata(input_tokens=-1)


def test_request_rejects_blank_prompt() -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="   ")


def test_structured_output_rejects_blank_schema_name() -> None:
    with pytest.raises(ValidationError):
        StructuredOutputExpectation(schema_name=" ", json_schema={"type": "object"})


def test_fake_provider_returns_responses_deterministically() -> None:
    first = GenerationResponse(text="first", structured_output={"value": 1})
    second = GenerationResponse(text="second")
    provider = FakeLLMProvider([first, second])
    request = GenerationRequest(prompt="hello")

    assert provider.generate(request, timeout_seconds=2.5) == first
    assert provider.generate(request, timeout_seconds=2.5) == second
    assert provider.requests == [(request, 2.5), (request, 2.5)]


def test_fake_provider_rejects_non_positive_timeout() -> None:
    provider = FakeLLMProvider([GenerationResponse(text="unused")])

    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        provider.generate(GenerationRequest(prompt="hello"), timeout_seconds=0)


def test_fake_provider_fails_when_queue_is_exhausted() -> None:
    provider = FakeLLMProvider([])

    with pytest.raises(RuntimeError, match="no queued response"):
        provider.generate(GenerationRequest(prompt="hello"), timeout_seconds=1)


def test_failing_fake_provider_raises_configured_error() -> None:
    error = TimeoutError("simulated timeout")
    provider = FailingFakeLLMProvider(error)
    request = GenerationRequest(prompt="hello")

    with pytest.raises(TimeoutError, match="simulated timeout"):
        provider.generate(request, timeout_seconds=3)
    assert provider.requests == [(request, 3)]


def test_fake_provider_structurally_satisfies_protocol() -> None:
    provider: LLMProvider = FakeLLMProvider([GenerationResponse(text="ok")])
    assert_type(provider, LLMProvider)
    assert provider.generate(GenerationRequest(prompt="hello"), timeout_seconds=1).text == "ok"
