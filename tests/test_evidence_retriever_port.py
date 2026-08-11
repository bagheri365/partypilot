from __future__ import annotations

from datetime import date

from partypilot.domain.evidence_corpus import EvidenceDocumentStatus
from partypilot.ports.evidence_retriever import (
    EvidenceRetrievalFilters,
    EvidenceRetrievalQuery,
    EvidenceRetrievalResult,
    EvidenceVersionMetadata,
    RetrievalMethod,
)


def test_retrieval_query_supports_resource_filter_and_top_k() -> None:
    query = EvidenceRetrievalQuery(
        text="peanut policy",
        top_k=3,
        filters=EvidenceRetrievalFilters(resource_id="caterer-family-table"),
    )
    assert query.top_k == 3
    assert query.filters.resource_id == "caterer-family-table"


def test_result_preserves_version_metadata() -> None:
    result = EvidenceRetrievalResult(
        document_id="doc-1",
        chunk_id="doc-1#chunk-1",
        resource_id="vendor-1",
        version=EvidenceVersionMetadata(
            version="2.0",
            effective_date=date(2026, 1, 1),
            status=EvidenceDocumentStatus.CURRENT,
        ),
        text="Policy text",
        score=4.2,
        rank=1,
        retrieval_method=RetrievalMethod.BM25,
    )
    assert result.version.version == "2.0"
