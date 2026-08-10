"""Ollama LLM adapter implemented behind PartyPilot's provider-neutral port."""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, field_validator

from partypilot.ports.llm_provider import (
    GenerationRequest,
    GenerationResponse,
    LLMProvider,
    UsageMetadata,
)


class OllamaProviderError(RuntimeError):
    """Base typed error for Ollama provider failures."""


class OllamaTimeoutError(OllamaProviderError):
    """Raised when the Ollama request exceeds its timeout."""


class OllamaConnectionError(OllamaProviderError):
    """Raised when Ollama cannot be reached."""


class OllamaResponseError(OllamaProviderError):
    """Raised when Ollama returns an invalid or unsuccessful response."""


class OllamaConfig(BaseModel):
    """Environment-driven configuration for the Ollama adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str = "http://localhost:11434"
    model: str
    timeout_seconds: float = Field(gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)

    @field_validator("base_url", "model")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("configuration values cannot be blank")
        return (
            normalized.rstrip("/")
            if value == normalized and value.startswith("http")
            else normalized
        )

    @classmethod
    def from_env(cls) -> OllamaConfig:
        """Load configuration from environment variables."""
        model = os.environ.get("PARTYPILOT_OLLAMA_MODEL")
        if model is None:
            raise ValueError("PARTYPILOT_OLLAMA_MODEL is required")

        return cls(
            base_url=os.environ.get("PARTYPILOT_OLLAMA_BASE_URL", "http://localhost:11434"),
            model=model,
            timeout_seconds=float(os.environ.get("PARTYPILOT_OLLAMA_TIMEOUT_SECONDS", "30")),
            max_retries=int(os.environ.get("PARTYPILOT_OLLAMA_MAX_RETRIES", "2")),
        )


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Minimal HTTP response used by the adapter transport boundary."""

    status_code: int
    body: bytes


class HttpTransport(Protocol):
    """Minimal typed transport required by the Ollama adapter."""

    def post_json(self, url: str, payload: bytes, *, timeout_seconds: float) -> HttpResponse:
        """POST JSON bytes and return a response."""
        ...


class UrllibHttpTransport:
    """Standard-library HTTP transport used at the composition boundary."""

    def post_json(self, url: str, payload: bytes, *, timeout_seconds: float) -> HttpResponse:
        request = Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return HttpResponse(status_code=response.status, body=response.read())
        except HTTPError as error:
            return HttpResponse(status_code=error.code, body=error.read())
        except TimeoutError as error:
            raise OllamaTimeoutError("Ollama request timed out") from error
        except URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                raise OllamaTimeoutError("Ollama request timed out") from error
            raise OllamaConnectionError("Could not connect to Ollama") from error
        except OSError as error:
            raise OllamaConnectionError("Could not connect to Ollama") from error


class OllamaAdapter(LLMProvider):
    """Provider-neutral PartyPilot LLM adapter for Ollama's generate API."""

    def __init__(self, config: OllamaConfig, transport: HttpTransport) -> None:
        self._config = config
        self._transport = transport

    def generate(
        self,
        request: GenerationRequest,
        *,
        timeout_seconds: float,
    ) -> GenerationResponse:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        effective_timeout = min(timeout_seconds, self._config.timeout_seconds)
        payload = self._build_payload(request)
        attempts = self._config.max_retries + 1

        last_error: OllamaProviderError | None = None
        for attempt in range(attempts):
            try:
                response = self._transport.post_json(
                    f"{self._config.base_url}/api/generate",
                    payload,
                    timeout_seconds=effective_timeout,
                )
                return self._parse_response(response)
            except (OllamaTimeoutError, OllamaConnectionError) as error:
                last_error = error
                if attempt == attempts - 1:
                    raise

        if last_error is not None:  # pragma: no cover - defensive safeguard
            raise last_error
        raise OllamaProviderError("Ollama generation failed")  # pragma: no cover

    def _build_payload(self, request: GenerationRequest) -> bytes:
        body: dict[str, object] = {
            "model": self._config.model,
            "prompt": request.prompt,
            "stream": False,
        }
        if request.system_prompt is not None:
            body["system"] = request.system_prompt
        if request.structured_output is not None:
            body["format"] = "json"
        return json.dumps(body, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _parse_response(response: HttpResponse) -> GenerationResponse:
        try:
            payload = json.loads(response.body)
        except (JSONDecodeError, UnicodeDecodeError) as error:
            raise OllamaResponseError("Ollama returned invalid JSON") from error

        if not isinstance(payload, dict):
            raise OllamaResponseError("Ollama response must be a JSON object")

        if response.status_code < 200 or response.status_code >= 300:
            detail = payload.get("error")
            message = detail if isinstance(detail, str) and detail else "Ollama request failed"
            raise OllamaResponseError(f"{message} (HTTP {response.status_code})")

        text = payload.get("response")
        if not isinstance(text, str):
            raise OllamaResponseError("Ollama response is missing text output")

        structured_output = None
        if text:
            try:
                structured_output = json.loads(text)
            except JSONDecodeError:
                structured_output = None

        input_tokens = payload.get("prompt_eval_count")
        output_tokens = payload.get("eval_count")
        usage = None
        if isinstance(input_tokens, int) or isinstance(output_tokens, int):
            safe_input = (
                input_tokens if isinstance(input_tokens, int) and input_tokens >= 0 else None
            )
            safe_output = (
                output_tokens if isinstance(output_tokens, int) and output_tokens >= 0 else None
            )
            total = None
            if safe_input is not None and safe_output is not None:
                total = safe_input + safe_output
            usage = UsageMetadata(
                input_tokens=safe_input,
                output_tokens=safe_output,
                total_tokens=total,
            )

        return GenerationResponse(text=text, structured_output=structured_output, usage=usage)
