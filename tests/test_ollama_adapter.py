"""Tests for the Ollama provider adapter without a live Ollama server."""

from __future__ import annotations

import json
from collections import deque

import pytest
from pydantic import ValidationError

from partypilot.adapters.ollama import (
    HttpResponse,
    OllamaAdapter,
    OllamaConfig,
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from partypilot.ports import GenerationRequest, StructuredOutputExpectation


class FakeTransport:
    def __init__(self, results: list[HttpResponse | Exception]) -> None:
        self._results = deque(results)
        self.calls: list[tuple[str, bytes, float]] = []

    def post_json(self, url: str, payload: bytes, *, timeout_seconds: float) -> HttpResponse:
        self.calls.append((url, payload, timeout_seconds))
        result = self._results.popleft()
        if isinstance(result, Exception):
            raise result
        return result


def config(**overrides: object) -> OllamaConfig:
    values: dict[str, object] = {
        "base_url": "http://ollama.test:11434",
        "model": "llama-test",
        "timeout_seconds": 12.0,
        "max_retries": 2,
    }
    values.update(overrides)
    return OllamaConfig.model_validate(values)


def test_config_loads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARTYPILOT_OLLAMA_MODEL", "qwen-test")
    monkeypatch.setenv("PARTYPILOT_OLLAMA_BASE_URL", "http://example.test:11434/")
    monkeypatch.setenv("PARTYPILOT_OLLAMA_TIMEOUT_SECONDS", "9")
    monkeypatch.setenv("PARTYPILOT_OLLAMA_MAX_RETRIES", "1")

    loaded = OllamaConfig.from_env()

    assert loaded.model == "qwen-test"
    assert loaded.base_url == "http://example.test:11434"
    assert loaded.timeout_seconds == 9
    assert loaded.max_retries == 1


def test_config_requires_explicit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PARTYPILOT_OLLAMA_MODEL", raising=False)

    with pytest.raises(ValueError, match="PARTYPILOT_OLLAMA_MODEL"):
        OllamaConfig.from_env()


def test_config_rejects_invalid_timeout_and_retry_bounds() -> None:
    with pytest.raises(ValidationError):
        config(timeout_seconds=0)
    with pytest.raises(ValidationError):
        config(max_retries=6)


def test_generate_sends_model_prompts_schema_and_uses_bounded_timeout() -> None:
    response_body = json.dumps(
        {
            "response": '{"venue":"v1"}',
            "prompt_eval_count": 11,
            "eval_count": 7,
        }
    ).encode()
    transport = FakeTransport([HttpResponse(200, response_body)])
    adapter = OllamaAdapter(config(), transport)
    request = GenerationRequest(
        prompt="Plan a party",
        system_prompt="Return JSON",
        structured_output=StructuredOutputExpectation(
            schema_name="PartyPlan",
            json_schema={"type": "object"},
        ),
    )

    result = adapter.generate(request, timeout_seconds=30)

    assert result.text == '{"venue":"v1"}'
    assert result.structured_output == {"venue": "v1"}
    assert result.usage is not None
    assert result.usage.total_tokens == 18
    assert len(transport.calls) == 1
    url, raw_payload, timeout = transport.calls[0]
    assert url == "http://ollama.test:11434/api/generate"
    assert timeout == 12.0
    payload = json.loads(raw_payload)
    assert payload == {
        "model": "llama-test",
        "prompt": "Plan a party",
        "stream": False,
        "system": "Return JSON",
        "format": "json",
    }


def test_generate_retries_connection_failure_then_succeeds() -> None:
    transport = FakeTransport(
        [
            OllamaConnectionError("offline"),
            HttpResponse(200, b'{"response":"ok"}'),
        ]
    )
    adapter = OllamaAdapter(config(max_retries=1), transport)

    result = adapter.generate(GenerationRequest(prompt="hello"), timeout_seconds=5)

    assert result.text == "ok"
    assert len(transport.calls) == 2


def test_generate_stops_after_bounded_retries() -> None:
    transport = FakeTransport(
        [OllamaTimeoutError("slow"), OllamaTimeoutError("slow"), OllamaTimeoutError("slow")]
    )
    adapter = OllamaAdapter(config(max_retries=2), transport)

    with pytest.raises(OllamaTimeoutError, match="slow"):
        adapter.generate(GenerationRequest(prompt="hello"), timeout_seconds=5)

    assert len(transport.calls) == 3


def test_generate_does_not_retry_provider_response_errors() -> None:
    transport = FakeTransport([HttpResponse(500, b'{"error":"model unavailable"}')])
    adapter = OllamaAdapter(config(max_retries=2), transport)

    with pytest.raises(OllamaResponseError, match=r"model unavailable.*HTTP 500"):
        adapter.generate(GenerationRequest(prompt="hello"), timeout_seconds=5)

    assert len(transport.calls) == 1


def test_generate_rejects_invalid_json_response() -> None:
    transport = FakeTransport([HttpResponse(200, b"not-json")])
    adapter = OllamaAdapter(config(), transport)

    with pytest.raises(OllamaResponseError, match="invalid JSON"):
        adapter.generate(GenerationRequest(prompt="hello"), timeout_seconds=5)


def test_generate_requires_response_text() -> None:
    transport = FakeTransport([HttpResponse(200, b"{}")])
    adapter = OllamaAdapter(config(), transport)

    with pytest.raises(OllamaResponseError, match="missing text output"):
        adapter.generate(GenerationRequest(prompt="hello"), timeout_seconds=5)


def test_generate_requires_positive_call_timeout() -> None:
    adapter = OllamaAdapter(config(), FakeTransport([]))

    with pytest.raises(ValueError, match="positive"):
        adapter.generate(GenerationRequest(prompt="hello"), timeout_seconds=0)
