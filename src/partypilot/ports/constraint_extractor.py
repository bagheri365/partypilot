from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field

from partypilot.domain.constraints import Constraint
from partypilot.domain.evidence import Provenance
from partypilot.domain.evidence_corpus import EvidenceDocumentMetadata
from partypilot.domain.party_request import PartyRequest

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class ConstraintExtractionContext(BaseModel):
    """Planning context supplied to evidence-policy extraction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request: PartyRequest
    resource_id: str | None = None


class ConstraintExtractionInput(BaseModel):
    """Evidence plus metadata and planning context for extraction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_text: Annotated[str, Field(min_length=1)]
    evidence_metadata: EvidenceDocumentMetadata
    planning_context: ConstraintExtractionContext
    chunk_id: Annotated[str, Field(min_length=1)]


class ExtractedConstraint(BaseModel):
    """Typed constraint with source provenance and optional extractor confidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    constraint: Constraint
    provenance: Provenance
    confidence: Confidence | None = None


class ConstraintExtractionResult(BaseModel):
    """A deterministic container for extractor output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    constraints: tuple[ExtractedConstraint, ...] = ()


class ConstraintExtractor(Protocol):
    """Port for extracting policy constraints from evidence.

    Implementations extract source-backed constraints only. They must not execute
    PartyPilot's deterministic validation rules.
    """

    def extract(
        self, extraction_input: ConstraintExtractionInput
    ) -> ConstraintExtractionResult: ...


class FakeConstraintExtractor:
    """Deterministic queued-result extractor for unit/integration tests."""

    def __init__(self, results: Iterable[ConstraintExtractionResult]) -> None:
        self._results = deque(results)
        self.inputs: list[ConstraintExtractionInput] = []

    def extract(self, extraction_input: ConstraintExtractionInput) -> ConstraintExtractionResult:
        self.inputs.append(extraction_input)
        if not self._results:
            raise RuntimeError("fake constraint extractor has no queued result")
        return self._results.popleft()


class FailingFakeConstraintExtractor:
    """Deterministic extractor that always raises the configured error."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.inputs: list[ConstraintExtractionInput] = []

    def extract(self, extraction_input: ConstraintExtractionInput) -> ConstraintExtractionResult:
        self.inputs.append(extraction_input)
        raise self._error
