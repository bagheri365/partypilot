from __future__ import annotations

import json
from pathlib import Path

from partypilot.adapters.bm25_evidence_retriever import BM25EvidenceRetriever
from partypilot.domain.evidence_corpus import (
    EvidenceDocument,
    EvidenceDocumentStatus,
    EvidenceDocumentType,
)
from partypilot.ports.evidence_retriever import (
    EvidenceRetrievalFilters,
    EvidenceRetrievalQuery,
    RetrievalMethod,
)

CORPUS_PATH = Path("data/evidence/v0_2_documents.json")


def load_documents() -> tuple[EvidenceDocument, ...]:
    raw = json.loads(CORPUS_PATH.read_text())
    return tuple(EvidenceDocument.model_validate(item) for item in raw)


def test_exact_terminology_ranks_matching_policy_first() -> None:
    retriever = BM25EvidenceRetriever(load_documents())

    results = retriever.retrieve(
        EvidenceRetrievalQuery(
            text="grip socks trampoline safety",
            top_k=3,
            filters=EvidenceRetrievalFilters(status=EvidenceDocumentStatus.CURRENT),
        )
    )

    assert results[0].document_id == "doc-trampoline-safety"
    assert results[0].retrieval_method is RetrievalMethod.BM25
    assert results[0].rank == 1


def test_vendor_filter_excludes_similar_wrong_vendor_document() -> None:
    retriever = BM25EvidenceRetriever(load_documents())
    query = EvidenceRetrievalQuery(
        text="peanut allergen policy",
        top_k=5,
        filters=EvidenceRetrievalFilters(
            resource_id="caterer-family-table", status=EvidenceDocumentStatus.CURRENT
        ),
    )

    results = retriever.retrieve(query)

    assert results
    assert all(result.resource_id == "caterer-family-table" for result in results)
    assert "doc-queens-allergen-distractor" not in {result.document_id for result in results}
    assert results[0].document_id == "doc-family-allergen-current"


def test_current_status_filter_supports_version_sensitive_retrieval() -> None:
    retriever = BM25EvidenceRetriever(load_documents())
    query = EvidenceRetrievalQuery(
        text="Brooklyn Loft accessibility wheelchair restroom",
        top_k=5,
        filters=EvidenceRetrievalFilters(
            resource_id="venue-brooklyn-loft",
            document_type=EvidenceDocumentType.ACCESSIBILITY_GUIDANCE,
            status=EvidenceDocumentStatus.CURRENT,
        ),
    )

    results = retriever.retrieve(query)

    assert results
    assert all(result.version.status is EvidenceDocumentStatus.CURRENT for result in results)
    assert "doc-loft-accessibility-old" not in {result.document_id for result in results}
    assert results[0].document_id == "doc-loft-accessibility-current"
    assert results[0].version.version == "2.1"


def test_top_k_is_deterministic_and_bounded() -> None:
    documents = load_documents()
    retriever = BM25EvidenceRetriever(documents)
    query = EvidenceRetrievalQuery(text="policy safety accessibility", top_k=4)

    first = retriever.retrieve(query)
    second = retriever.retrieve(query)

    assert first == second
    assert len(first) == 4
    assert [item.rank for item in first] == [1, 2, 3, 4]


def test_empty_filtered_candidate_set_returns_empty_tuple() -> None:
    retriever = BM25EvidenceRetriever(load_documents())
    query = EvidenceRetrievalQuery(
        text="allergen",
        filters=EvidenceRetrievalFilters(resource_id="missing-vendor"),
    )

    assert retriever.retrieve(query) == ()


def test_invalid_bm25_parameters_are_rejected() -> None:
    documents = load_documents()

    try:
        BM25EvidenceRetriever(documents, k1=0)
    except ValueError as exc:
        assert "k1" in str(exc)
    else:
        raise AssertionError("expected invalid k1 to fail")

    try:
        BM25EvidenceRetriever(documents, b=1.5)
    except ValueError as exc:
        assert "b" in str(exc)
    else:
        raise AssertionError("expected invalid b to fail")
