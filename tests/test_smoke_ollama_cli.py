from __future__ import annotations

from types import SimpleNamespace

import pytest

from partypilot.cli import smoke_ollama
from partypilot.ports.llm_provider import GenerationResponse


class FakeAdapter:
    def __init__(self, response: GenerationResponse) -> None:
        self.response = response
        self.requests: list[tuple[object, float]] = []

    def generate(self, request: object, *, timeout_seconds: float) -> GenerationResponse:
        self.requests.append((request, timeout_seconds))
        return self.response


def test_smoke_ollama_reports_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_config = SimpleNamespace(model="fake-model", timeout_seconds=12.0)
    adapter = FakeAdapter(GenerationResponse(text="OK"))
    monkeypatch.setattr(smoke_ollama, "_ollama_config", lambda **kwargs: fake_config)
    monkeypatch.setattr(smoke_ollama, "OllamaAdapter", lambda config, transport: adapter)

    exit_code = smoke_ollama.main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Ollama smoke test passed." in captured.out
    assert "fake-model" in captured.out
    assert "OK" in captured.out
    assert len(adapter.requests) == 1


def test_smoke_ollama_reports_empty_response(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_config = SimpleNamespace(model="fake-model", timeout_seconds=12.0)
    adapter = FakeAdapter(GenerationResponse(text="   "))
    monkeypatch.setattr(smoke_ollama, "_ollama_config", lambda **kwargs: fake_config)
    monkeypatch.setattr(smoke_ollama, "OllamaAdapter", lambda config, transport: adapter)

    exit_code = smoke_ollama.main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "empty response" in captured.err
