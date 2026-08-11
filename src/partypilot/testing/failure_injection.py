"""Deterministic failure-injection doubles for PartyPilot tests.

These helpers model repeatable infrastructure and data-quality failures without
network access, sleeping, randomness, or hidden mutable global state.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from datetime import date

from partypilot.domain.evidence_corpus import EvidenceDocumentStatus, EvidenceDocumentType
from partypilot.ports.evidence_retriever import (
    EvidenceRetrievalQuery,
    EvidenceRetrievalResult,
    EvidenceVersionMetadata,
    RetrievalMethod,
)
from partypilot.ports.llm_provider import GenerationRequest, GenerationResponse, LLMProvider


class InjectedProviderUnavailableError(RuntimeError):
    """Deterministic stand-in for a provider/service outage."""


class InjectedToolError(RuntimeError):
    """Deterministic stand-in for an unexpected external tool failure."""


class ScriptedLLMProvider(LLMProvider):
    """Provider double that replays a fixed sequence of responses/exceptions."""

    def __init__(self, outcomes: Iterable[GenerationResponse | Exception]) -> None:
        self._outcomes = deque(outcomes)
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
        if not self._outcomes:
            raise RuntimeError("scripted provider has no queued outcome")
        outcome = self._outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class StaticEvidenceRetriever:
    """Retriever double that always returns the same deterministic results."""

    def __init__(self, results: Iterable[EvidenceRetrievalResult]) -> None:
        self._results = tuple(results)
        self.queries: list[EvidenceRetrievalQuery] = []

    def retrieve(self, query: EvidenceRetrievalQuery) -> tuple[EvidenceRetrievalResult, ...]:
        self.queries.append(query)
        return self._results[: query.top_k]


class RaisingEvidenceRetriever:
    """Retriever double that deterministically raises the configured exception."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.queries: list[EvidenceRetrievalQuery] = []

    def retrieve(self, query: EvidenceRetrievalQuery) -> tuple[EvidenceRetrievalResult, ...]:
        self.queries.append(query)
        raise self._error


def timeout_provider(
    *, attempts: int = 1, message: str = "injected timeout"
) -> ScriptedLLMProvider:
    """Return a provider that times out exactly ``attempts`` times."""

    if attempts <= 0:
        raise ValueError("attempts must be positive")
    return ScriptedLLMProvider(TimeoutError(message) for _ in range(attempts))


def unavailable_provider(
    *, attempts: int = 1, message: str = "injected provider unavailable"
) -> ScriptedLLMProvider:
    """Return a provider that reports deterministic service unavailability."""

    if attempts <= 0:
        raise ValueError("attempts must be positive")
    return ScriptedLLMProvider(InjectedProviderUnavailableError(message) for _ in range(attempts))


def malformed_structured_output_provider(*, attempts: int = 1) -> ScriptedLLMProvider:
    """Return responses whose structured payload is intentionally malformed."""

    if attempts <= 0:
        raise ValueError("attempts must be positive")
    response = GenerationResponse(text="malformed", structured_output={"unexpected": True})
    return ScriptedLLMProvider(response for _ in range(attempts))


def repeated_invalid_output_provider(*, attempts: int) -> ScriptedLLMProvider:
    """Return the same invalid structured response for a bounded number of calls."""

    return malformed_structured_output_provider(attempts=attempts)


def empty_retriever() -> StaticEvidenceRetriever:
    """Return a retriever that deterministically finds no evidence."""

    return StaticEvidenceRetriever(())


def tool_exception_retriever(message: str = "injected tool exception") -> RaisingEvidenceRetriever:
    """Return a retriever that raises a deterministic external-tool error."""

    return RaisingEvidenceRetriever(InjectedToolError(message))


def wrong_vendor_evidence(
    *,
    expected_resource_id: str = "venue-alpha",
    actual_resource_id: str = "venue-distractor",
) -> EvidenceRetrievalResult:
    """Create current evidence that belongs to the wrong vendor/resource."""

    if expected_resource_id == actual_resource_id:
        raise ValueError("actual_resource_id must differ from expected_resource_id")
    return EvidenceRetrievalResult(
        document_id="failure-wrong-vendor-doc",
        chunk_id="failure-wrong-vendor-doc#chunk-1",
        resource_id=actual_resource_id,
        version=EvidenceVersionMetadata(
            version="1.0",
            effective_date=date(2026, 1, 1),
            status=EvidenceDocumentStatus.CURRENT,
        ),
        document_type=EvidenceDocumentType.VENUE_POLICY,
        text=f"Distractor policy for {actual_resource_id}; expected {expected_resource_id}.",
        score=1.0,
        rank=1,
        retrieval_method=RetrievalMethod.FAKE,
    )


def stale_evidence(*, resource_id: str = "venue-alpha") -> EvidenceRetrievalResult:
    """Create deterministic outdated evidence for version-sensitivity tests."""

    return EvidenceRetrievalResult(
        document_id="failure-stale-doc",
        chunk_id="failure-stale-doc#chunk-1",
        resource_id=resource_id,
        version=EvidenceVersionMetadata(
            version="0.9",
            effective_date=date(2024, 1, 1),
            status=EvidenceDocumentStatus.OUTDATED,
        ),
        document_type=EvidenceDocumentType.VENUE_POLICY,
        text="Outdated policy retained only for deterministic failure testing.",
        score=1.0,
        rank=1,
        retrieval_method=RetrievalMethod.FAKE,
    )
