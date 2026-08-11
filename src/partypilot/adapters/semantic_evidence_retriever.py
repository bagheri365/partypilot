from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from partypilot.domain.evidence_corpus import EvidenceDocument
from partypilot.ports.embedding_provider import EmbeddingProvider
from partypilot.ports.evidence_retriever import (
    EvidenceRetrievalQuery,
    EvidenceRetrievalResult,
    EvidenceVersionMetadata,
    RetrievalMethod,
)


class SemanticRetrievalError(RuntimeError):
    """Base error for semantic retrieval failures."""


class InvalidEmbeddingError(SemanticRetrievalError):
    """Raised when an embedding provider returns malformed vectors."""


@dataclass(frozen=True, slots=True)
class _EmbeddedDocument:
    document: EvidenceDocument
    vector: tuple[float, ...]


class SemanticEvidenceRetriever:
    """Experimental in-memory semantic retriever using injected embedding infrastructure."""

    def __init__(
        self,
        documents: Iterable[EvidenceDocument],
        *,
        embedding_provider: EmbeddingProvider,
        timeout_seconds: float = 10.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        materialized = tuple(documents)
        if not materialized:
            raise ValueError("documents must not be empty")

        self._embedding_provider = embedding_provider
        self._timeout_seconds = timeout_seconds
        texts = tuple(self._searchable_text(document) for document in materialized)
        vectors = embedding_provider.embed(texts, timeout_seconds=timeout_seconds)
        self._validate_batch(vectors, expected_count=len(materialized))
        self._documents = tuple(
            _EmbeddedDocument(document=document, vector=vector)
            for document, vector in zip(materialized, vectors, strict=True)
        )

    @staticmethod
    def _searchable_text(document: EvidenceDocument) -> str:
        metadata = document.metadata
        return " ".join(
            (
                document.text,
                metadata.resource_id,
                metadata.document_type.value,
                metadata.version,
                metadata.status.value,
                metadata.effective_date.isoformat(),
            )
        )

    def retrieve(self, query: EvidenceRetrievalQuery) -> tuple[EvidenceRetrievalResult, ...]:
        candidates = tuple(item for item in self._documents if self._matches_filters(item, query))
        if not candidates:
            return ()

        query_vectors = self._embedding_provider.embed(
            (query.text,),
            timeout_seconds=self._timeout_seconds,
        )
        self._validate_batch(query_vectors, expected_count=1)
        query_vector = query_vectors[0]
        if len(query_vector) != len(candidates[0].vector):
            raise InvalidEmbeddingError("query embedding dimension does not match index dimension")

        scored = [
            (
                self._cosine_similarity(query_vector, item.vector),
                item.document.metadata.document_id,
                item,
            )
            for item in candidates
        ]
        scored.sort(key=lambda row: (-row[0], row[1]))

        return tuple(
            EvidenceRetrievalResult(
                document_id=item.document.metadata.document_id,
                chunk_id=f"{item.document.metadata.document_id}#chunk-1",
                resource_id=item.document.metadata.resource_id,
                version=EvidenceVersionMetadata(
                    version=item.document.metadata.version,
                    effective_date=item.document.metadata.effective_date,
                    status=item.document.metadata.status,
                ),
                document_type=item.document.metadata.document_type,
                text=item.document.text,
                score=score,
                rank=rank,
                retrieval_method=RetrievalMethod.SEMANTIC,
            )
            for rank, (score, _, item) in enumerate(scored[: query.top_k], start=1)
        )

    @staticmethod
    def _matches_filters(item: _EmbeddedDocument, query: EvidenceRetrievalQuery) -> bool:
        metadata = item.document.metadata
        filters = query.filters
        return (
            (filters.resource_id is None or metadata.resource_id == filters.resource_id)
            and (filters.document_type is None or metadata.document_type == filters.document_type)
            and (filters.status is None or metadata.status == filters.status)
        )

    @classmethod
    def _validate_batch(
        cls,
        vectors: tuple[tuple[float, ...], ...],
        *,
        expected_count: int,
    ) -> None:
        if len(vectors) != expected_count:
            raise InvalidEmbeddingError("embedding provider returned an unexpected vector count")
        if not vectors:
            raise InvalidEmbeddingError("embedding provider returned no vectors")
        dimension = len(vectors[0])
        if dimension == 0:
            raise InvalidEmbeddingError("embedding vectors must not be empty")
        for vector in vectors:
            if len(vector) != dimension:
                raise InvalidEmbeddingError("embedding vectors must have consistent dimensions")
            if any(not math.isfinite(value) for value in vector):
                raise InvalidEmbeddingError("embedding vectors must contain finite values")

    @staticmethod
    def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        if len(left) != len(right):
            raise InvalidEmbeddingError("embedding dimensions do not match")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
