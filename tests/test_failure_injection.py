from __future__ import annotations

import pytest

from partypilot.domain.evidence_corpus import EvidenceDocumentStatus
from partypilot.ports.evidence_retriever import EvidenceRetrievalQuery
from partypilot.ports.llm_provider import GenerationRequest
from partypilot.testing.failure_injection import (
    InjectedProviderUnavailableError,
    InjectedToolError,
    empty_retriever,
    repeated_invalid_output_provider,
    stale_evidence,
    timeout_provider,
    tool_exception_retriever,
    unavailable_provider,
    wrong_vendor_evidence,
)


def _request() -> GenerationRequest:
    return GenerationRequest(prompt="failure injection")


def test_timeout_failure_is_reproducible() -> None:
    provider = timeout_provider(attempts=2)
    for _ in range(2):
        with pytest.raises(TimeoutError, match="injected timeout"):
            provider.generate(_request(), timeout_seconds=1.0)
    assert len(provider.requests) == 2


def test_provider_unavailable_failure_is_reproducible() -> None:
    provider = unavailable_provider(attempts=2)
    for _ in range(2):
        with pytest.raises(InjectedProviderUnavailableError):
            provider.generate(_request(), timeout_seconds=1.0)


def test_repeated_invalid_output_is_bounded_and_identical() -> None:
    provider = repeated_invalid_output_provider(attempts=3)
    outputs = [provider.generate(_request(), timeout_seconds=1.0) for _ in range(3)]
    assert [output.structured_output for output in outputs] == [
        {"unexpected": True},
        {"unexpected": True},
        {"unexpected": True},
    ]
    with pytest.raises(RuntimeError, match="no queued outcome"):
        provider.generate(_request(), timeout_seconds=1.0)


def test_empty_retrieval_is_reproducible() -> None:
    retriever = empty_retriever()
    query = EvidenceRetrievalQuery(text="allergen policy")
    assert retriever.retrieve(query) == ()
    assert retriever.retrieve(query) == ()


def test_wrong_vendor_fixture_is_explicit() -> None:
    result = wrong_vendor_evidence(
        expected_resource_id="venue-alpha",
        actual_resource_id="venue-beta",
    )
    assert result.resource_id == "venue-beta"
    assert "venue-alpha" in result.text
    assert result.version.status is EvidenceDocumentStatus.CURRENT


def test_stale_evidence_fixture_is_outdated() -> None:
    result = stale_evidence(resource_id="venue-alpha")
    assert result.resource_id == "venue-alpha"
    assert result.version.status is EvidenceDocumentStatus.OUTDATED
    assert result.version.version == "0.9"


def test_tool_exception_is_reproducible() -> None:
    retriever = tool_exception_retriever()
    query = EvidenceRetrievalQuery(text="policy")
    for _ in range(2):
        with pytest.raises(InjectedToolError, match="injected tool exception"):
            retriever.retrieve(query)
    assert retriever.queries == [query, query]


def test_invalid_failure_fixture_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="attempts"):
        timeout_provider(attempts=0)
    with pytest.raises(ValueError, match="attempts"):
        unavailable_provider(attempts=0)
    with pytest.raises(ValueError, match="differ"):
        wrong_vendor_evidence(expected_resource_id="same", actual_resource_id="same")
