from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from partypilot.application.derived_constraints import DerivedConstraint
from partypilot.domain.evidence import EvidenceReference, EvidenceState, Provenance
from partypilot.domain.evidence_corpus import EvidenceDocument, EvidenceDocumentStatus


class CitationViolationCode(StrEnum):
    NONEXISTENT_SOURCE = "nonexistent_source"
    WRONG_RESOURCE = "wrong_resource"
    VERSION_MISMATCH = "version_mismatch"
    OUTDATED_VERSION = "outdated_version"
    DERIVED_PROVENANCE_INVALID = "derived_provenance_invalid"
    UNSUPPORTED_AS_SUPPORTED = "unsupported_as_supported"


class CitationViolation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: CitationViolationCode
    message: str
    evidence_id: str | None = None
    constraint_id: str | None = None


class CitationValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    violations: tuple[CitationViolation, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.violations


def validate_citations(
    *,
    corpus: tuple[EvidenceDocument, ...],
    evidence_references: tuple[EvidenceReference, ...] = (),
    derived_constraints: tuple[DerivedConstraint, ...] = (),
) -> CitationValidationResult:
    """Validate evidence citations and derived provenance against the source corpus."""

    documents = {document.metadata.document_id: document for document in corpus}
    violations: list[CitationViolation] = []

    for reference in evidence_references:
        reference_violations: list[CitationViolation] = []
        for provenance in reference.provenance:
            reference_violations.extend(
                _validate_provenance(
                    provenance,
                    documents=documents,
                    evidence_id=reference.evidence_id,
                )
            )
        violations.extend(reference_violations)
        if reference.state is EvidenceState.SUPPORTED and reference_violations:
            violations.append(
                CitationViolation(
                    code=CitationViolationCode.UNSUPPORTED_AS_SUPPORTED,
                    evidence_id=reference.evidence_id,
                    message=(
                        "Evidence marked SUPPORTED has invalid source provenance and cannot "
                        "masquerade as a supported claim."
                    ),
                )
            )

    for derived in derived_constraints:
        provenance_violations: list[CitationViolation] = []
        for provenance in derived.provenance:
            provenance_violations.extend(
                _validate_provenance(
                    provenance,
                    documents=documents,
                    constraint_id=derived.constraint.identifier,
                )
            )
        if (
            not derived.constraint.provenance
            or not derived.constraint.provenance.source_constraint_ids
        ):
            provenance_violations.append(
                CitationViolation(
                    code=CitationViolationCode.DERIVED_PROVENANCE_INVALID,
                    constraint_id=derived.constraint.identifier,
                    message="Derived constraint does not link to a source constraint.",
                )
            )
        if not derived.provenance or any(p.source_document_id is None for p in derived.provenance):
            provenance_violations.append(
                CitationViolation(
                    code=CitationViolationCode.DERIVED_PROVENANCE_INVALID,
                    constraint_id=derived.constraint.identifier,
                    message="Derived constraint does not link to supporting source evidence.",
                )
            )
        violations.extend(provenance_violations)

    return CitationValidationResult(violations=tuple(violations))


def _validate_provenance(
    provenance: Provenance,
    *,
    documents: dict[str, EvidenceDocument],
    evidence_id: str | None = None,
    constraint_id: str | None = None,
) -> list[CitationViolation]:
    if provenance.source_document_id is None:
        return [
            CitationViolation(
                code=CitationViolationCode.NONEXISTENT_SOURCE,
                evidence_id=evidence_id,
                constraint_id=constraint_id,
                message="Citation does not identify a source document.",
            )
        ]

    document = documents.get(provenance.source_document_id)
    if document is None:
        return [
            CitationViolation(
                code=CitationViolationCode.NONEXISTENT_SOURCE,
                evidence_id=evidence_id,
                constraint_id=constraint_id,
                message=f"Source document {provenance.source_document_id!r} does not exist.",
            )
        ]

    metadata = document.metadata
    result: list[CitationViolation] = []
    if provenance.resource_id != metadata.resource_id:
        result.append(
            CitationViolation(
                code=CitationViolationCode.WRONG_RESOURCE,
                evidence_id=evidence_id,
                constraint_id=constraint_id,
                message=(
                    f"Citation resource {provenance.resource_id!r} does not match source "
                    f"resource {metadata.resource_id!r}."
                ),
            )
        )
    if provenance.source_version != metadata.version:
        result.append(
            CitationViolation(
                code=CitationViolationCode.VERSION_MISMATCH,
                evidence_id=evidence_id,
                constraint_id=constraint_id,
                message=(
                    f"Citation version {provenance.source_version!r} does not match source "
                    f"version {metadata.version!r}."
                ),
            )
        )
    if metadata.status is not EvidenceDocumentStatus.CURRENT:
        result.append(
            CitationViolation(
                code=CitationViolationCode.OUTDATED_VERSION,
                evidence_id=evidence_id,
                constraint_id=constraint_id,
                message=(
                    f"Source document {metadata.document_id!r} has status "
                    f"{metadata.status.value!r}, not current."
                ),
            )
        )
    return result
