from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from partypilot.domain.constraints import Constraint
from partypilot.domain.evidence import EvidenceState, Provenance
from partypilot.domain.evidence_corpus import EvidenceDocumentStatus

NonEmptyString = Annotated[str, Field(min_length=1)]


class EvidenceAssessment(BaseModel):
    """One applicable-or-candidate policy interpretation considered for resolution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provenance: Provenance
    document_status: EvidenceDocumentStatus
    constraint: Constraint | None = None
    applicable: bool = True
    ambiguous: bool = False
    safety_sensitive: bool = False
    explanation: NonEmptyString


class EvidenceResolution(BaseModel):
    """Resolved support state for one policy requirement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: EvidenceState
    constraints: tuple[Constraint, ...] = ()
    provenance: tuple[Provenance, ...] = ()
    explanation: NonEmptyString


def resolve_evidence_state(assessments: tuple[EvidenceAssessment, ...]) -> EvidenceResolution:
    """Resolve evidence support using deterministic, conservative rules.

    Only current, applicable evidence can support a requirement. Clear current policies
    that disagree are conflicted. Ambiguous current evidence is insufficient, and for
    safety-sensitive requirements ambiguity always prevents a SUPPORTED state.
    """

    current = tuple(
        assessment
        for assessment in assessments
        if assessment.applicable and assessment.document_status is EvidenceDocumentStatus.CURRENT
    )
    if not current:
        return EvidenceResolution(
            state=EvidenceState.UNSUPPORTED,
            explanation="No applicable current evidence supports this requirement.",
        )

    clear = tuple(
        assessment
        for assessment in current
        if not assessment.ambiguous and assessment.constraint is not None
    )
    ambiguous = tuple(
        assessment
        for assessment in current
        if assessment.ambiguous or assessment.constraint is None
    )

    signatures = {
        _constraint_signature(item.constraint) for item in clear if item.constraint is not None
    }
    if len(signatures) > 1:
        return EvidenceResolution(
            state=EvidenceState.CONFLICTED,
            constraints=tuple(item.constraint for item in clear if item.constraint is not None),
            provenance=tuple(item.provenance for item in current),
            explanation="Applicable current policies disagree on the requirement.",
        )

    if ambiguous:
        safety_ambiguity = any(item.safety_sensitive for item in ambiguous) or any(
            item.safety_sensitive for item in current
        )
        explanation = (
            "Safety-sensitive evidence is ambiguous and cannot be treated as supported."
            if safety_ambiguity
            else "Applicable current evidence is ambiguous or incomplete."
        )
        return EvidenceResolution(
            state=EvidenceState.INSUFFICIENT_EVIDENCE,
            constraints=tuple(item.constraint for item in clear if item.constraint is not None),
            provenance=tuple(item.provenance for item in current),
            explanation=explanation,
        )

    if clear:
        representative = clear[0].constraint
        assert representative is not None
        return EvidenceResolution(
            state=EvidenceState.SUPPORTED,
            constraints=(representative,),
            provenance=tuple(item.provenance for item in clear),
            explanation=(
                "Applicable current evidence clearly and consistently supports the requirement."
            ),
        )

    return EvidenceResolution(
        state=EvidenceState.INSUFFICIENT_EVIDENCE,
        provenance=tuple(item.provenance for item in current),
        explanation="Current evidence exists but does not establish a definitive requirement.",
    )


def _constraint_signature(constraint: Constraint | None) -> tuple[object, ...]:
    if constraint is None:
        return (None,)
    return (constraint.key, constraint.operator.value, repr(constraint.value))
