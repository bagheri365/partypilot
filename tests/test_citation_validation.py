from datetime import date

from partypilot.application.citation_validation import (
    CitationViolationCode,
    validate_citations,
)
from partypilot.application.derived_constraints import DerivedConstraint
from partypilot.domain.constraints import (
    Constraint,
    ConstraintOperator,
    ConstraintProvenance,
    ConstraintType,
)
from partypilot.domain.evidence import (
    DerivationMethod,
    EvidenceReference,
    EvidenceState,
    Provenance,
)
from partypilot.domain.evidence_corpus import (
    EvidenceDocument,
    EvidenceDocumentMetadata,
    EvidenceDocumentStatus,
    EvidenceDocumentType,
)


def _document(
    *, status: EvidenceDocumentStatus = EvidenceDocumentStatus.CURRENT
) -> EvidenceDocument:
    return EvidenceDocument(
        metadata=EvidenceDocumentMetadata(
            document_id="doc-1",
            resource_id="venue-1",
            document_type=EvidenceDocumentType.SUPERVISION_REQUIREMENTS,
            version="v2",
            effective_date=date(2026, 1, 1),
            status=status,
        ),
        text="One adult is required for every five children.",
    )


def _provenance(**overrides: object) -> Provenance:
    values = {
        "source_document_id": "doc-1",
        "source_chunk_id": "doc-1:chunk-1",
        "resource_id": "venue-1",
        "source_version": "v2",
        "effective_date": date(2026, 1, 1),
        "derivation_method": DerivationMethod.LLM_EXTRACTED,
        "derivation_explanation": "Extracted supervision rule.",
    }
    values.update(overrides)
    return Provenance(**values)  # type: ignore[arg-type]


def test_correct_source_is_valid() -> None:
    reference = EvidenceReference(
        evidence_id="constraint:adult_child_ratio",
        state=EvidenceState.SUPPORTED,
        provenance=(_provenance(),),
    )
    assert validate_citations(corpus=(_document(),), evidence_references=(reference,)).valid


def test_wrong_vendor_is_rejected_and_supported_claim_is_flagged() -> None:
    reference = EvidenceReference(
        evidence_id="constraint:adult_child_ratio",
        state=EvidenceState.SUPPORTED,
        provenance=(_provenance(resource_id="venue-2"),),
    )
    result = validate_citations(corpus=(_document(),), evidence_references=(reference,))
    codes = {violation.code for violation in result.violations}
    assert CitationViolationCode.WRONG_RESOURCE in codes
    assert CitationViolationCode.UNSUPPORTED_AS_SUPPORTED in codes


def test_outdated_version_is_rejected() -> None:
    reference = EvidenceReference(
        evidence_id="constraint:adult_child_ratio",
        state=EvidenceState.SUPPORTED,
        provenance=(_provenance(),),
    )
    result = validate_citations(
        corpus=(_document(status=EvidenceDocumentStatus.SUPERSEDED),),
        evidence_references=(reference,),
    )
    assert CitationViolationCode.OUTDATED_VERSION in {v.code for v in result.violations}


def test_nonexistent_source_is_rejected() -> None:
    reference = EvidenceReference(
        evidence_id="constraint:adult_child_ratio",
        state=EvidenceState.SUPPORTED,
        provenance=(_provenance(source_document_id="missing"),),
    )
    result = validate_citations(corpus=(_document(),), evidence_references=(reference,))
    assert CitationViolationCode.NONEXISTENT_SOURCE in {v.code for v in result.violations}


def test_version_mismatch_is_rejected() -> None:
    reference = EvidenceReference(
        evidence_id="constraint:adult_child_ratio",
        state=EvidenceState.SUPPORTED,
        provenance=(_provenance(source_version="v1"),),
    )
    result = validate_citations(corpus=(_document(),), evidence_references=(reference,))
    assert CitationViolationCode.VERSION_MISMATCH in {v.code for v in result.violations}


def test_valid_derived_provenance_links_to_source_evidence() -> None:
    constraint = Constraint(
        identifier="derived:minimum_adults",
        key="minimum_adults",
        operator=ConstraintOperator.GTE,
        value=5,
        constraint_type=ConstraintType.DERIVED,
        description="At least five adults are required.",
        provenance=ConstraintProvenance(
            source_constraint_ids=("extracted:ratio",),
            derivation_explanation="Applied 1/5 to 24 children and rounded up.",
        ),
    )
    derived = DerivedConstraint(
        constraint=constraint,
        provenance=(
            _provenance(
                derivation_method=DerivationMethod.DETERMINISTIC,
                derivation_explanation="Applied 1/5 to 24 children and rounded up.",
            ),
        ),
    )
    assert validate_citations(corpus=(_document(),), derived_constraints=(derived,)).valid
