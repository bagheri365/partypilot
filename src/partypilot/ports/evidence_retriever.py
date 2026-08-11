from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field

from partypilot.domain.evidence_corpus import EvidenceDocumentStatus, EvidenceDocumentType

NonEmptyString = Annotated[str, Field(min_length=1)]


class RetrievalMethod(StrEnum):
    BM25 = "bm25"
    SEMANTIC = "semantic"
    HYBRID_RRF = "hybrid_rrf"
    FAKE = "fake"


class EvidenceVersionMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: NonEmptyString
    effective_date: date
    status: EvidenceDocumentStatus


class EvidenceRetrievalFilters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    resource_id: NonEmptyString | None = None
    document_type: EvidenceDocumentType | None = None
    status: EvidenceDocumentStatus | None = None


class EvidenceRetrievalQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: NonEmptyString
    top_k: Annotated[int, Field(gt=0)] = 5
    filters: EvidenceRetrievalFilters = Field(default_factory=EvidenceRetrievalFilters)


class EvidenceRetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: NonEmptyString
    chunk_id: NonEmptyString
    resource_id: NonEmptyString
    version: EvidenceVersionMetadata
    document_type: EvidenceDocumentType | None = None
    text: NonEmptyString
    score: float
    rank: Annotated[int, Field(gt=0)]
    retrieval_method: RetrievalMethod


class EvidenceRetriever(Protocol):
    def retrieve(self, query: EvidenceRetrievalQuery) -> tuple[EvidenceRetrievalResult, ...]: ...
