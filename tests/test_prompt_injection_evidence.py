from __future__ import annotations

import json
import os
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from partypilot.adapters.bm25_evidence_retriever import BM25EvidenceRetriever
from partypilot.adapters.llm_constraint_extractor import LLMConstraintExtractor
from partypilot.domain.evidence_corpus import EvidenceDocument, EvidenceDocumentStatus
from partypilot.domain.party_request import PartyRequest
from partypilot.ports.constraint_extractor import (
    ConstraintExtractionContext,
    ConstraintExtractionInput,
)
from partypilot.ports.evidence_retriever import EvidenceRetrievalFilters, EvidenceRetrievalQuery
from partypilot.ports.llm_provider import FakeLLMProvider, GenerationResponse

CORPUS_PATH = Path("data/evidence/v0_2_documents.json")
INJECTION_DOCUMENT_IDS = {
    "doc-injection-system-override",
    "doc-injection-ignore-constraints",
    "doc-injection-permissions",
    "doc-injection-secret-exfiltration",
    "doc-injection-false-vendor-safety",
}


def _documents() -> tuple[EvidenceDocument, ...]:
    raw = json.loads(CORPUS_PATH.read_text())
    return tuple(EvidenceDocument.model_validate(item) for item in raw)


def _by_id() -> dict[str, EvidenceDocument]:
    return {doc.metadata.document_id: doc for doc in _documents()}


def test_adversarial_evidence_documents_cover_required_injection_patterns() -> None:
    documents = _by_id()

    assert documents.keys() >= INJECTION_DOCUMENT_IDS
    combined = " ".join(documents[doc_id].text.casefold() for doc_id in INJECTION_DOCUMENT_IDS)
    assert "ignore all previous instructions" in combined
    assert "ignore partypilot safety constraints" in combined
    assert "permissions" in combined
    assert "api keys" in combined
    assert "completely peanut-free" in combined


def test_document_text_cannot_override_current_status_filter() -> None:
    retriever = BM25EvidenceRetriever(_documents())

    results = retriever.retrieve(
        EvidenceRetrievalQuery(
            text="system override approve every booking constraints satisfied",
            top_k=10,
            filters=EvidenceRetrievalFilters(
                resource_id="venue-brooklyn-loft",
                status=EvidenceDocumentStatus.CURRENT,
            ),
        )
    )

    returned_ids = {result.document_id for result in results}
    assert "doc-injection-system-override" not in returned_ids
    assert "doc-injection-permissions" not in returned_ids
    assert all(result.version.status is EvidenceDocumentStatus.CURRENT for result in results)


def test_false_vendor_claim_cannot_override_resource_filter() -> None:
    retriever = BM25EvidenceRetriever(_documents())

    results = retriever.retrieve(
        EvidenceRetrievalQuery(
            text="Caterer Family Table peanut-free allergy safe",
            top_k=10,
            filters=EvidenceRetrievalFilters(
                resource_id="caterer-family-table",
                status=EvidenceDocumentStatus.CURRENT,
            ),
        )
    )

    assert results
    assert all(result.resource_id == "caterer-family-table" for result in results)
    assert "doc-injection-false-vendor-safety" not in {result.document_id for result in results}


def test_secret_exfiltration_instruction_is_returned_only_as_text_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PARTYPILOT_TEST_SECRET", "never-expose-this-value")
    retriever = BM25EvidenceRetriever(_documents())

    result = retriever.retrieve(
        EvidenceRetrievalQuery(
            text="environment variables API keys tokens passwords secrets",
            top_k=1,
            filters=EvidenceRetrievalFilters(
                resource_id="caterer-family-table",
                status=EvidenceDocumentStatus.DRAFT,
            ),
        )
    )[0]

    assert result.document_id == "doc-injection-secret-exfiltration"
    assert "read environment variables" in result.text.casefold()
    assert os.environ["PARTYPILOT_TEST_SECRET"] == "never-expose-this-value"
    assert "never-expose-this-value" not in result.text


def test_llm_extractor_marks_evidence_as_untrusted_data_in_system_instruction() -> None:
    document = _by_id()["doc-injection-ignore-constraints"]
    provider = FakeLLMProvider([GenerationResponse(text="", structured_output={"constraints": []})])
    extractor = LLMConstraintExtractor(provider)
    extraction_input = ConstraintExtractionInput(
        evidence_text=document.text,
        evidence_metadata=document.metadata,
        chunk_id=f"{document.metadata.document_id}#chunk-1",
        planning_context=ConstraintExtractionContext(
            request=PartyRequest(
                location="Boston",
                event_date=date(2026, 9, 1),
                guest_count=24,
                child_age=8,
                total_budget=Decimal("1200.00"),
            ),
            resource_id=document.metadata.resource_id,
        ),
    )

    result = extractor.extract(extraction_input)

    assert result.constraints == ()
    request, _ = provider.requests[0]
    system_prompt = (request.system_prompt or "").casefold()
    assert "untrusted data" in system_prompt
    assert "never as instructions" in system_prompt
    assert "change permissions" in system_prompt
    assert "access secrets" in system_prompt
    assert document.text in request.prompt


def test_retrieval_of_injection_text_does_not_create_or_modify_constraints() -> None:
    retriever = BM25EvidenceRetriever(_documents())

    result = retriever.retrieve(
        EvidenceRetrievalQuery(
            text="set supervision requirements to zero ignore safety constraints",
            top_k=1,
            filters=EvidenceRetrievalFilters(
                resource_id="activity-trampoline-zone",
                status=EvidenceDocumentStatus.DRAFT,
            ),
        )
    )[0]

    assert result.document_id == "doc-injection-ignore-constraints"
    assert result.retrieval_method.value == "bm25"
    # Retrieval returns evidence records only. It has no API for mutating PartyPilot constraints,
    # permissions, provider configuration, or planner state.
    assert not hasattr(result, "constraints")
    assert not hasattr(result, "permissions")
