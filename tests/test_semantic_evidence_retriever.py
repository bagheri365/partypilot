from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pytest

from partypilot.adapters.semantic_evidence_retriever import (
    InvalidEmbeddingError,
    SemanticEvidenceRetriever,
)
from partypilot.domain.evidence_corpus import (
    EvidenceDocument,
    EvidenceDocumentMetadata,
    EvidenceDocumentStatus,
    EvidenceDocumentType,
)
from partypilot.ports.evidence_retriever import (
    EvidenceRetrievalFilters,
    EvidenceRetrievalQuery,
    RetrievalMethod,
)


class DeterministicEmbeddingProvider:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def embed(
        self,
        texts: Sequence[str],
        *,
        timeout_seconds: float,
    ) -> tuple[tuple[float, ...], ...]:
        self.timeouts.append(timeout_seconds)
        return tuple(self._vector(text) for text in texts)

    @staticmethod
    def _vector(text: str) -> tuple[float, ...]:
        lowered = text.casefold()
        return (
            float(lowered.count("allergy") + lowered.count("allergen")),
            float(lowered.count("wheelchair") + lowered.count("accessible")),
            float(lowered.count("adult") + lowered.count("supervision")),
            float(lowered.count("cancel")),
        )


def _document(
    document_id: str,
    resource_id: str,
    text: str,
    *,
    status: EvidenceDocumentStatus = EvidenceDocumentStatus.CURRENT,
    version: str = "2.0",
    document_type: EvidenceDocumentType = EvidenceDocumentType.ALLERGEN_POLICY,
) -> EvidenceDocument:
    return EvidenceDocument(
        metadata=EvidenceDocumentMetadata(
            document_id=document_id,
            resource_id=resource_id,
            document_type=document_type,
            version=version,
            effective_date=date(2026, 1, 1),
            status=status,
        ),
        text=text,
    )


def test_semantic_ranking_uses_deterministic_embeddings() -> None:
    provider = DeterministicEmbeddingProvider()
    retriever = SemanticEvidenceRetriever(
        (
            _document("allergy", "venue-a", "Allergy allergen handling guidance."),
            _document("access", "venue-a", "Wheelchair accessible entrance."),
        ),
        embedding_provider=provider,
        timeout_seconds=3.5,
    )

    results = retriever.retrieve(EvidenceRetrievalQuery(text="allergy safety", top_k=2))

    assert [result.document_id for result in results] == ["allergy", "access"]
    assert [result.rank for result in results] == [1, 2]
    assert all(result.retrieval_method is RetrievalMethod.SEMANTIC for result in results)
    assert provider.timeouts == [3.5, 3.5]


def test_vendor_filter_excludes_semantically_similar_wrong_vendor() -> None:
    provider = DeterministicEmbeddingProvider()
    retriever = SemanticEvidenceRetriever(
        (
            _document("a", "venue-a", "Allergy allergen policy."),
            _document("b", "venue-b", "Allergy allergen policy allergy."),
        ),
        embedding_provider=provider,
    )

    results = retriever.retrieve(
        EvidenceRetrievalQuery(
            text="allergy policy",
            filters=EvidenceRetrievalFilters(resource_id="venue-a"),
        )
    )

    assert [result.document_id for result in results] == ["a"]
    assert results[0].resource_id == "venue-a"


def test_status_and_version_metadata_are_preserved() -> None:
    provider = DeterministicEmbeddingProvider()
    retriever = SemanticEvidenceRetriever(
        (
            _document(
                "old",
                "venue-a",
                "Allergy policy.",
                status=EvidenceDocumentStatus.OUTDATED,
                version="1.0",
            ),
            _document("current", "venue-a", "Allergy policy.", version="2.0"),
        ),
        embedding_provider=provider,
    )

    results = retriever.retrieve(
        EvidenceRetrievalQuery(
            text="allergy",
            filters=EvidenceRetrievalFilters(status=EvidenceDocumentStatus.CURRENT),
        )
    )

    assert [result.document_id for result in results] == ["current"]
    assert results[0].version.version == "2.0"
    assert results[0].version.status is EvidenceDocumentStatus.CURRENT


def test_document_type_filter_and_top_k() -> None:
    provider = DeterministicEmbeddingProvider()
    retriever = SemanticEvidenceRetriever(
        (
            _document(
                "a",
                "venue-a",
                "Wheelchair accessible.",
                document_type=EvidenceDocumentType.ACCESSIBILITY_GUIDANCE,
            ),
            _document(
                "b",
                "venue-a",
                "Wheelchair accessible accessible.",
                document_type=EvidenceDocumentType.ACCESSIBILITY_GUIDANCE,
            ),
            _document(
                "c", "venue-a", "Wheelchair.", document_type=EvidenceDocumentType.VENUE_POLICY
            ),
        ),
        embedding_provider=provider,
    )

    results = retriever.retrieve(
        EvidenceRetrievalQuery(
            text="wheelchair accessible",
            top_k=1,
            filters=EvidenceRetrievalFilters(
                document_type=EvidenceDocumentType.ACCESSIBILITY_GUIDANCE
            ),
        )
    )

    assert len(results) == 1
    assert results[0].document_id == "a"


def test_zero_similarity_ties_are_deterministic_by_document_id() -> None:
    provider = DeterministicEmbeddingProvider()
    retriever = SemanticEvidenceRetriever(
        (
            _document("z-doc", "venue-a", "Unrelated wording."),
            _document("a-doc", "venue-a", "Different wording."),
        ),
        embedding_provider=provider,
    )

    results = retriever.retrieve(EvidenceRetrievalQuery(text="unknown concept"))

    assert [result.document_id for result in results] == ["a-doc", "z-doc"]


def test_empty_filtered_candidates_do_not_embed_query() -> None:
    provider = DeterministicEmbeddingProvider()
    retriever = SemanticEvidenceRetriever(
        (_document("a", "venue-a", "Allergy policy."),),
        embedding_provider=provider,
    )

    results = retriever.retrieve(
        EvidenceRetrievalQuery(
            text="allergy",
            filters=EvidenceRetrievalFilters(resource_id="missing"),
        )
    )

    assert results == ()
    assert len(provider.timeouts) == 1


def test_invalid_configuration_is_rejected() -> None:
    provider = DeterministicEmbeddingProvider()
    doc = _document("a", "venue-a", "Allergy policy.")

    with pytest.raises(ValueError, match="timeout_seconds"):
        SemanticEvidenceRetriever((doc,), embedding_provider=provider, timeout_seconds=0)
    with pytest.raises(ValueError, match="documents"):
        SemanticEvidenceRetriever((), embedding_provider=provider)


class BadEmbeddingProvider:
    def __init__(self, vectors: tuple[tuple[float, ...], ...]) -> None:
        self.vectors = vectors

    def embed(
        self,
        texts: Sequence[str],
        *,
        timeout_seconds: float,
    ) -> tuple[tuple[float, ...], ...]:
        del texts, timeout_seconds
        return self.vectors


def test_malformed_embedding_batch_is_rejected() -> None:
    doc = _document("a", "venue-a", "Allergy policy.")

    with pytest.raises(InvalidEmbeddingError):
        SemanticEvidenceRetriever((doc,), embedding_provider=BadEmbeddingProvider(()))
    with pytest.raises(InvalidEmbeddingError):
        SemanticEvidenceRetriever(
            (doc,), embedding_provider=BadEmbeddingProvider(((1.0, float("nan")),))
        )


def test_query_dimension_mismatch_is_rejected() -> None:
    class ChangingProvider:
        def __init__(self) -> None:
            self.calls = 0

        def embed(
            self,
            texts: Sequence[str],
            *,
            timeout_seconds: float,
        ) -> tuple[tuple[float, ...], ...]:
            del timeout_seconds
            self.calls += 1
            dim = 2 if self.calls == 1 else 3
            return tuple(tuple(1.0 for _ in range(dim)) for _ in texts)

    retriever = SemanticEvidenceRetriever(
        (_document("a", "venue-a", "Allergy policy."),),
        embedding_provider=ChangingProvider(),
    )

    with pytest.raises(InvalidEmbeddingError, match="dimension"):
        retriever.retrieve(EvidenceRetrievalQuery(text="allergy"))
