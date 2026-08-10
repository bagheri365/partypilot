from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonEmptyString = Annotated[str, Field(min_length=1)]


class EvidenceState(StrEnum):
    """Support state for an evidence-grounded claim or constraint."""

    SUPPORTED = "SUPPORTED"
    CONFLICTED = "CONFLICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNSUPPORTED = "UNSUPPORTED"


class DerivationMethod(StrEnum):
    """How a fact, claim, or constraint was derived from source material."""

    DETERMINISTIC = "deterministic"
    LLM_EXTRACTED = "llm_extracted"
    LLM_INFERRED = "llm_inferred"


class Provenance(BaseModel):
    """Traceable source and derivation metadata for evidence-backed domain data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_document_id: NonEmptyString | None = None
    source_chunk_id: NonEmptyString | None = None
    resource_id: NonEmptyString | None = None
    source_version: NonEmptyString | None = None
    effective_date: date | None = None
    derivation_method: DerivationMethod
    derivation_explanation: NonEmptyString

    @model_validator(mode="after")
    def require_source_reference(self) -> Provenance:
        if (
            self.source_document_id is None
            and self.source_chunk_id is None
            and self.resource_id is None
        ):
            raise ValueError(
                "provenance must reference a source document, source chunk, or resource"
            )
        if self.source_chunk_id is not None and self.source_document_id is None:
            raise ValueError("source_chunk_id requires source_document_id")
        return self


class EvidenceReference(BaseModel):
    """Evidence state and provenance attached to a stable evidence identifier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: NonEmptyString
    state: EvidenceState
    provenance: tuple[Provenance, ...] = Field(min_length=1)
