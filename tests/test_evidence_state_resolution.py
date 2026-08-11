from __future__ import annotations

from datetime import date

from partypilot.application.evidence_state_resolution import (
    EvidenceAssessment,
    resolve_evidence_state,
)
from partypilot.domain.constraints import Constraint, ConstraintOperator, ConstraintType
from partypilot.domain.evidence import DerivationMethod, EvidenceState, Provenance
from partypilot.domain.evidence_corpus import EvidenceDocumentStatus


def provenance(document_id: str) -> Provenance:
    return Provenance(
        source_document_id=document_id,
        source_chunk_id=f"{document_id}:chunk:1",
        resource_id="venue-alpha",
        source_version="2.0",
        effective_date=date(2026, 1, 1),
        derivation_method=DerivationMethod.LLM_EXTRACTED,
        derivation_explanation="Extracted directly from policy text.",
    )


def rule(identifier: str, value: str) -> Constraint:
    return Constraint(
        identifier=identifier,
        key="adult_child_ratio",
        operator=ConstraintOperator.EQ,
        value=value,
        constraint_type=ConstraintType.HARD,
        description="Adult supervision ratio.",
    )


def assessment(
    document_id: str,
    constraint: Constraint | None,
    *,
    status: EvidenceDocumentStatus = EvidenceDocumentStatus.CURRENT,
    ambiguous: bool = False,
    safety_sensitive: bool = False,
    applicable: bool = True,
) -> EvidenceAssessment:
    return EvidenceAssessment(
        provenance=provenance(document_id),
        document_status=status,
        constraint=constraint,
        applicable=applicable,
        ambiguous=ambiguous,
        safety_sensitive=safety_sensitive,
        explanation="Assessment fixture.",
    )


def test_clear_applicable_policy_is_supported() -> None:
    resolved = resolve_evidence_state((assessment("doc-current", rule("ratio", "1/5")),))

    assert resolved.state is EvidenceState.SUPPORTED
    assert resolved.constraints[0].value == "1/5"
    assert resolved.provenance[0].source_document_id == "doc-current"


def test_identical_current_policies_remain_supported() -> None:
    resolved = resolve_evidence_state(
        (
            assessment("doc-a", rule("ratio-a", "1/5")),
            assessment("doc-b", rule("ratio-b", "1/5")),
        )
    )

    assert resolved.state is EvidenceState.SUPPORTED
    assert len(resolved.provenance) == 2


def test_two_current_policies_that_disagree_are_conflicted() -> None:
    resolved = resolve_evidence_state(
        (
            assessment("doc-a", rule("ratio-a", "1/5")),
            assessment("doc-b", rule("ratio-b", "1/6")),
        )
    )

    assert resolved.state is EvidenceState.CONFLICTED
    assert {item.value for item in resolved.constraints} == {"1/5", "1/6"}


def test_contact_us_allergy_language_is_insufficient() -> None:
    resolved = resolve_evidence_state(
        (
            assessment(
                "allergen-policy",
                None,
                ambiguous=True,
                safety_sensitive=True,
            ),
        )
    )

    assert resolved.state is EvidenceState.INSUFFICIENT_EVIDENCE
    assert "Safety-sensitive" in resolved.explanation


def test_safety_sensitive_ambiguity_blocks_supported_even_with_clear_policy() -> None:
    resolved = resolve_evidence_state(
        (
            assessment("clear-safety", rule("ratio", "1/5"), safety_sensitive=True),
            assessment(
                "ambiguous-safety",
                None,
                ambiguous=True,
                safety_sensitive=True,
            ),
        )
    )

    assert resolved.state is EvidenceState.INSUFFICIENT_EVIDENCE


def test_only_outdated_evidence_is_unsupported() -> None:
    resolved = resolve_evidence_state(
        (
            assessment(
                "old-policy",
                rule("old-ratio", "1/4"),
                status=EvidenceDocumentStatus.OUTDATED,
            ),
        )
    )

    assert resolved.state is EvidenceState.UNSUPPORTED
    assert not resolved.constraints


def test_outdated_conflict_does_not_override_current_policy() -> None:
    resolved = resolve_evidence_state(
        (
            assessment("current", rule("current-ratio", "1/5")),
            assessment(
                "old",
                rule("old-ratio", "1/4"),
                status=EvidenceDocumentStatus.SUPERSEDED,
            ),
        )
    )

    assert resolved.state is EvidenceState.SUPPORTED
    assert resolved.constraints[0].value == "1/5"


def test_non_applicable_current_document_does_not_support_requirement() -> None:
    resolved = resolve_evidence_state(
        (assessment("other-vendor", rule("ratio", "1/5"), applicable=False),)
    )

    assert resolved.state is EvidenceState.UNSUPPORTED
