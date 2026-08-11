from __future__ import annotations

from dataclasses import dataclass

from partypilot.ports.evidence_retriever import (
    EvidenceRetrievalQuery,
    EvidenceRetrievalResult,
    EvidenceRetriever,
    RetrievalMethod,
)


@dataclass(frozen=True, slots=True)
class RRFConfig:
    """Configuration for reciprocal-rank fusion.

    ``rank_constant`` is the traditional RRF k parameter. ``candidate_multiplier``
    controls how deeply each component retriever is queried relative to the
    requested final top-k. Source weights permit controlled experiments while
    keeping fusion deterministic and transparent.
    """

    rank_constant: int = 60
    bm25_weight: float = 1.0
    semantic_weight: float = 1.0
    candidate_multiplier: int = 3

    def __post_init__(self) -> None:
        if self.rank_constant < 0:
            raise ValueError("rank_constant must be non-negative")
        if self.bm25_weight < 0:
            raise ValueError("bm25_weight must be non-negative")
        if self.semantic_weight < 0:
            raise ValueError("semantic_weight must be non-negative")
        if self.bm25_weight == 0 and self.semantic_weight == 0:
            raise ValueError("at least one retrieval weight must be positive")
        if self.candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")


@dataclass(slots=True)
class _FusedCandidate:
    representative: EvidenceRetrievalResult
    score: float = 0.0


class RRFEvidenceRetriever:
    """Experimental reciprocal-rank fusion over BM25 and semantic retrievers."""

    def __init__(
        self,
        *,
        bm25_retriever: EvidenceRetriever,
        semantic_retriever: EvidenceRetriever,
        config: RRFConfig | None = None,
    ) -> None:
        self._bm25_retriever = bm25_retriever
        self._semantic_retriever = semantic_retriever
        self._config = config or RRFConfig()

    def retrieve(self, query: EvidenceRetrievalQuery) -> tuple[EvidenceRetrievalResult, ...]:
        candidate_top_k = query.top_k * self._config.candidate_multiplier
        component_query = query.model_copy(update={"top_k": candidate_top_k})

        bm25_results = self._bm25_retriever.retrieve(component_query)
        semantic_results = self._semantic_retriever.retrieve(component_query)

        fused: dict[tuple[str, str], _FusedCandidate] = {}
        self._accumulate(fused, bm25_results, weight=self._config.bm25_weight)
        self._accumulate(fused, semantic_results, weight=self._config.semantic_weight)

        ranked = sorted(
            fused.values(),
            key=lambda candidate: (
                -candidate.score,
                candidate.representative.document_id,
                candidate.representative.chunk_id,
            ),
        )

        return tuple(
            candidate.representative.model_copy(
                update={
                    "score": candidate.score,
                    "rank": rank,
                    "retrieval_method": RetrievalMethod.HYBRID_RRF,
                }
            )
            for rank, candidate in enumerate(ranked[: query.top_k], start=1)
        )

    def _accumulate(
        self,
        fused: dict[tuple[str, str], _FusedCandidate],
        results: tuple[EvidenceRetrievalResult, ...],
        *,
        weight: float,
    ) -> None:
        if weight == 0:
            return
        for result in results:
            key = (result.document_id, result.chunk_id)
            candidate = fused.get(key)
            if candidate is None:
                candidate = _FusedCandidate(representative=result)
                fused[key] = candidate
            else:
                self._validate_duplicate_metadata(candidate.representative, result)

            candidate.score += weight / (self._config.rank_constant + result.rank)

    @staticmethod
    def _validate_duplicate_metadata(
        left: EvidenceRetrievalResult,
        right: EvidenceRetrievalResult,
    ) -> None:
        if (
            left.resource_id != right.resource_id
            or left.version != right.version
            or left.text != right.text
        ):
            raise ValueError("retrievers returned conflicting metadata for the same document/chunk")
