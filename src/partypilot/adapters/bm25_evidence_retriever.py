from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from partypilot.domain.evidence_corpus import EvidenceDocument
from partypilot.ports.evidence_retriever import (
    EvidenceRetrievalQuery,
    EvidenceRetrievalResult,
    EvidenceVersionMetadata,
    RetrievalMethod,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(text.casefold()))


@dataclass(frozen=True, slots=True)
class _IndexedDocument:
    document: EvidenceDocument
    tokens: tuple[str, ...]
    frequencies: Counter[str]


class BM25EvidenceRetriever:
    """Experimental deterministic BM25 retriever over an in-memory evidence corpus."""

    def __init__(
        self,
        documents: Iterable[EvidenceDocument],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        indexed = tuple(self._index_document(document) for document in documents)
        if not indexed:
            raise ValueError("documents must not be empty")
        self._documents = indexed
        self._k1 = k1
        self._b = b
        self._average_length = sum(len(item.tokens) for item in indexed) / len(indexed)
        self._document_frequency = self._build_document_frequency(indexed)

    @staticmethod
    def _index_document(document: EvidenceDocument) -> _IndexedDocument:
        metadata = document.metadata
        searchable = " ".join(
            (
                document.text,
                metadata.resource_id,
                metadata.document_type.value,
                metadata.version,
                metadata.status.value,
                metadata.effective_date.isoformat(),
            )
        )
        tokens = _tokenize(searchable)
        return _IndexedDocument(document=document, tokens=tokens, frequencies=Counter(tokens))

    @staticmethod
    def _build_document_frequency(
        indexed: tuple[_IndexedDocument, ...],
    ) -> Counter[str]:
        frequency: Counter[str] = Counter()
        for item in indexed:
            frequency.update(set(item.tokens))
        return frequency

    def retrieve(self, query: EvidenceRetrievalQuery) -> tuple[EvidenceRetrievalResult, ...]:
        candidates = tuple(item for item in self._documents if self._matches_filters(item, query))
        if not candidates:
            return ()

        query_terms = _tokenize(query.text)
        scored = [
            (self._score(item, query_terms), item.document.metadata.document_id, item)
            for item in candidates
        ]
        scored.sort(key=lambda row: (-row[0], row[1]))

        results: list[EvidenceRetrievalResult] = []
        for rank, (score, _, item) in enumerate(scored[: query.top_k], start=1):
            metadata = item.document.metadata
            results.append(
                EvidenceRetrievalResult(
                    document_id=metadata.document_id,
                    chunk_id=f"{metadata.document_id}#chunk-1",
                    resource_id=metadata.resource_id,
                    version=EvidenceVersionMetadata(
                        version=metadata.version,
                        effective_date=metadata.effective_date,
                        status=metadata.status,
                    ),
                    document_type=metadata.document_type,
                    text=item.document.text,
                    score=score,
                    rank=rank,
                    retrieval_method=RetrievalMethod.BM25,
                )
            )
        return tuple(results)

    @staticmethod
    def _matches_filters(item: _IndexedDocument, query: EvidenceRetrievalQuery) -> bool:
        metadata = item.document.metadata
        filters = query.filters
        return (
            (filters.resource_id is None or metadata.resource_id == filters.resource_id)
            and (filters.document_type is None or metadata.document_type == filters.document_type)
            and (filters.status is None or metadata.status == filters.status)
        )

    def _score(self, item: _IndexedDocument, query_terms: tuple[str, ...]) -> float:
        score = 0.0
        corpus_size = len(self._documents)
        length = len(item.tokens)
        for term in query_terms:
            term_frequency = item.frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            document_frequency = self._document_frequency.get(term, 0)
            inverse_document_frequency = math.log(
                1 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = term_frequency + self._k1 * (
                1 - self._b + self._b * length / self._average_length
            )
            score += inverse_document_frequency * (term_frequency * (self._k1 + 1)) / denominator
        return score
