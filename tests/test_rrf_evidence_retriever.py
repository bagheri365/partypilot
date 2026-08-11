from __future__ import annotations

from datetime import date

import pytest

from partypilot.adapters.rrf_evidence_retriever import RRFConfig, RRFEvidenceRetriever
from partypilot.domain.evidence_corpus import EvidenceDocumentStatus
from partypilot.ports.evidence_retriever import (
    EvidenceRetrievalFilters,
    EvidenceRetrievalQuery,
    EvidenceRetrievalResult,
    EvidenceVersionMetadata,
    RetrievalMethod,
)


class _FakeRetriever:
    def __init__(self, results: tuple[EvidenceRetrievalResult, ...]) -> None:
        self.results = results
        self.queries: list[EvidenceRetrievalQuery] = []

    def retrieve(self, query: EvidenceRetrievalQuery) -> tuple[EvidenceRetrievalResult, ...]:
        self.queries.append(query)
        return self.results[: query.top_k]


def _result(
    document_id: str,
    *,
    rank: int,
    method: RetrievalMethod,
    resource_id: str = "venue-1",
    text: str | None = None,
    version: str = "2.0",
) -> EvidenceRetrievalResult:
    return EvidenceRetrievalResult(
        document_id=document_id,
        chunk_id=f"{document_id}#chunk-1",
        resource_id=resource_id,
        version=EvidenceVersionMetadata(
            version=version,
            effective_date=date(2026, 1, 1),
            status=EvidenceDocumentStatus.CURRENT,
        ),
        text=text or f"Policy text for {document_id}",
        score=1.0,
        rank=rank,
        retrieval_method=method,
    )


def test_rrf_fuses_rankings_and_deduplicates_identical_chunks() -> None:
    shared_bm25 = _result("doc-shared", rank=2, method=RetrievalMethod.BM25)
    shared_semantic = shared_bm25.model_copy(
        update={"rank": 1, "retrieval_method": RetrievalMethod.SEMANTIC, "score": 0.8}
    )
    bm25 = _FakeRetriever(
        (
            _result("doc-lexical", rank=1, method=RetrievalMethod.BM25),
            shared_bm25,
        )
    )
    semantic = _FakeRetriever(
        (
            shared_semantic,
            _result("doc-semantic", rank=2, method=RetrievalMethod.SEMANTIC),
        )
    )
    retriever = RRFEvidenceRetriever(
        bm25_retriever=bm25,
        semantic_retriever=semantic,
        config=RRFConfig(rank_constant=10),
    )

    results = retriever.retrieve(EvidenceRetrievalQuery(text="allergen policy", top_k=3))

    assert [result.document_id for result in results] == [
        "doc-shared",
        "doc-lexical",
        "doc-semantic",
    ]
    assert len([result for result in results if result.document_id == "doc-shared"]) == 1
    assert all(result.retrieval_method is RetrievalMethod.HYBRID_RRF for result in results)
    assert [result.rank for result in results] == [1, 2, 3]
    assert results[0].score == pytest.approx(1 / 12 + 1 / 11)


def test_rrf_propagates_filters_and_expands_component_top_k() -> None:
    bm25 = _FakeRetriever(())
    semantic = _FakeRetriever(())
    retriever = RRFEvidenceRetriever(
        bm25_retriever=bm25,
        semantic_retriever=semantic,
        config=RRFConfig(candidate_multiplier=4),
    )
    query = EvidenceRetrievalQuery(
        text="outside food",
        top_k=2,
        filters=EvidenceRetrievalFilters(resource_id="venue-1"),
    )

    assert retriever.retrieve(query) == ()
    for observed in (bm25.queries[0], semantic.queries[0]):
        assert observed.top_k == 8
        assert observed.filters.resource_id == "venue-1"
        assert observed.text == query.text


def test_rrf_weights_are_configurable() -> None:
    bm25 = _FakeRetriever((_result("doc-b", rank=1, method=RetrievalMethod.BM25),))
    semantic = _FakeRetriever((_result("doc-s", rank=1, method=RetrievalMethod.SEMANTIC),))
    retriever = RRFEvidenceRetriever(
        bm25_retriever=bm25,
        semantic_retriever=semantic,
        config=RRFConfig(rank_constant=0, bm25_weight=3.0, semantic_weight=1.0),
    )

    results = retriever.retrieve(EvidenceRetrievalQuery(text="policy", top_k=2))

    assert [result.document_id for result in results] == ["doc-b", "doc-s"]
    assert results[0].score == pytest.approx(3.0)
    assert results[1].score == pytest.approx(1.0)


def test_rrf_uses_deterministic_tie_breaking() -> None:
    bm25 = _FakeRetriever((_result("doc-b", rank=1, method=RetrievalMethod.BM25),))
    semantic = _FakeRetriever((_result("doc-a", rank=1, method=RetrievalMethod.SEMANTIC),))
    retriever = RRFEvidenceRetriever(bm25_retriever=bm25, semantic_retriever=semantic)

    results = retriever.retrieve(EvidenceRetrievalQuery(text="policy", top_k=2))

    assert [result.document_id for result in results] == ["doc-a", "doc-b"]


def test_rrf_rejects_conflicting_duplicate_metadata() -> None:
    bm25_result = _result("doc-1", rank=1, method=RetrievalMethod.BM25, text="first")
    semantic_result = _result("doc-1", rank=1, method=RetrievalMethod.SEMANTIC, text="different")
    retriever = RRFEvidenceRetriever(
        bm25_retriever=_FakeRetriever((bm25_result,)),
        semantic_retriever=_FakeRetriever((semantic_result,)),
    )

    with pytest.raises(ValueError, match="conflicting metadata"):
        retriever.retrieve(EvidenceRetrievalQuery(text="policy"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rank_constant": -1}, "rank_constant"),
        ({"bm25_weight": -1.0}, "bm25_weight"),
        ({"semantic_weight": -1.0}, "semantic_weight"),
        ({"bm25_weight": 0.0, "semantic_weight": 0.0}, "at least one"),
        ({"candidate_multiplier": 0}, "candidate_multiplier"),
    ],
)
def test_rrf_config_validation(kwargs: dict[str, float | int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RRFConfig(**kwargs)  # type: ignore[arg-type]
