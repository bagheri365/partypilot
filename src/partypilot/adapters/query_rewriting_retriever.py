"""Evidence-retriever decorator for conditional query rewriting."""

from __future__ import annotations

from partypilot.application.query_rewriting_experiment import QueryRewriter
from partypilot.ports.evidence_retriever import (
    EvidenceRetrievalQuery,
    EvidenceRetrievalResult,
    EvidenceRetriever,
)


class ConditionalQueryRewritingEvidenceRetriever:
    """Rewrite evidence queries conditionally before delegating to another retriever."""

    def __init__(self, retriever: EvidenceRetriever, rewriter: QueryRewriter) -> None:
        self._retriever = retriever
        self._rewriter = rewriter

    def retrieve(self, query: EvidenceRetrievalQuery) -> tuple[EvidenceRetrievalResult, ...]:
        rewritten = self._rewriter.rewrite(query.text)
        forwarded = query.model_copy(update={"text": rewritten})
        return self._retriever.retrieve(forwarded)
