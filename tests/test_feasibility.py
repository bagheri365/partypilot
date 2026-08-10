import pytest
from pydantic import ValidationError

from partypilot.domain import (
    Constraint,
    ConstraintOperator,
    ConstraintType,
    DerivationMethod,
    EvidenceReference,
    EvidenceState,
    FeasibilityOutcome,
    FeasibilityResult,
    Provenance,
    ValidationResult,
)


def _constraint(
    identifier: str, constraint_type: ConstraintType = ConstraintType.HARD
) -> Constraint:
    return Constraint(
        identifier=identifier,
        key="guest_count",
        operator=ConstraintOperator.LTE,
        value=20,
        constraint_type=constraint_type,
        description="Guest count must not exceed capacity.",
    )


def _evidence() -> EvidenceReference:
    return EvidenceReference(
        evidence_id="evidence-capacity",
        state=EvidenceState.SUPPORTED,
        provenance=(
            Provenance(
                resource_id="venue-1",
                derivation_method=DerivationMethod.DETERMINISTIC,
                derivation_explanation="Read from structured venue capacity.",
            ),
        ),
    )


def test_feasibility_outcomes_are_stable() -> None:
    assert {outcome.value for outcome in FeasibilityOutcome} == {
        "FEASIBLE",
        "NO_FEASIBLE_PLAN",
        "HUMAN_REVIEW_REQUIRED",
    }


def test_validation_result_supports_all_requested_fields() -> None:
    satisfied = _constraint("capacity")
    unresolved = _constraint("allergy", ConstraintType.SOFT)

    result = ValidationResult(
        satisfied_hard_constraints=(satisfied,),
        unresolved_constraints=(unresolved,),
        warnings=("Vendor policy is stale.",),
        evidence_references=(_evidence(),),
        reasons=("Capacity is supported by structured venue data.",),
    )

    assert result.satisfied_hard_constraints == (satisfied,)
    assert result.unresolved_constraints == (unresolved,)
    assert result.warnings == ("Vendor policy is stale.",)
    assert result.evidence_references[0].evidence_id == "evidence-capacity"
    assert result.reasons


def test_hard_constraint_buckets_reject_non_hard_constraints() -> None:
    soft = _constraint("theme", ConstraintType.SOFT)

    with pytest.raises(ValidationError, match="only HARD constraints"):
        ValidationResult(satisfied_hard_constraints=(soft,))


def test_constraint_cannot_be_both_satisfied_and_violated() -> None:
    capacity = _constraint("capacity")

    with pytest.raises(ValidationError, match="both satisfied and violated"):
        ValidationResult(
            satisfied_hard_constraints=(capacity,),
            violated_hard_constraints=(capacity,),
        )


def test_resolved_constraint_cannot_also_be_unresolved() -> None:
    capacity = _constraint("capacity")

    with pytest.raises(ValidationError, match="cannot also be unresolved"):
        ValidationResult(
            satisfied_hard_constraints=(capacity,),
            unresolved_constraints=(capacity,),
        )


def test_feasible_result_requires_feasible_plan_and_clean_validation() -> None:
    result = FeasibilityResult(
        outcome=FeasibilityOutcome.FEASIBLE,
        plan_feasible=True,
        validation=ValidationResult(satisfied_hard_constraints=(_constraint("capacity"),)),
    )

    assert result.plan_feasible is True

    with pytest.raises(ValidationError, match="requires plan_feasible=True"):
        FeasibilityResult(
            outcome=FeasibilityOutcome.FEASIBLE,
            plan_feasible=False,
            validation=ValidationResult(),
        )

    with pytest.raises(ValidationError, match="violated hard constraints"):
        FeasibilityResult(
            outcome=FeasibilityOutcome.FEASIBLE,
            plan_feasible=True,
            validation=ValidationResult(violated_hard_constraints=(_constraint("capacity"),)),
        )

    with pytest.raises(ValidationError, match="unresolved constraints"):
        FeasibilityResult(
            outcome=FeasibilityOutcome.FEASIBLE,
            plan_feasible=True,
            validation=ValidationResult(unresolved_constraints=(_constraint("allergy"),)),
        )


def test_no_feasible_plan_cannot_be_marked_feasible() -> None:
    with pytest.raises(ValidationError, match="requires plan_feasible=False"):
        FeasibilityResult(
            outcome=FeasibilityOutcome.NO_FEASIBLE_PLAN,
            plan_feasible=True,
            validation=ValidationResult(violated_hard_constraints=(_constraint("capacity"),)),
        )

    result = FeasibilityResult(
        outcome=FeasibilityOutcome.NO_FEASIBLE_PLAN,
        plan_feasible=False,
        validation=ValidationResult(violated_hard_constraints=(_constraint("capacity"),)),
    )
    assert result.outcome is FeasibilityOutcome.NO_FEASIBLE_PLAN


def test_human_review_requires_unresolved_plan_state() -> None:
    unresolved = _constraint("accessibility")
    result = FeasibilityResult(
        outcome=FeasibilityOutcome.HUMAN_REVIEW_REQUIRED,
        plan_feasible=None,
        validation=ValidationResult(unresolved_constraints=(unresolved,)),
    )
    assert result.plan_feasible is None

    with pytest.raises(ValidationError, match="requires plan_feasible=None"):
        FeasibilityResult(
            outcome=FeasibilityOutcome.HUMAN_REVIEW_REQUIRED,
            plan_feasible=False,
            validation=ValidationResult(unresolved_constraints=(unresolved,)),
        )


def test_feasibility_models_are_frozen_and_reject_unknown_fields() -> None:
    result = ValidationResult()

    with pytest.raises(ValidationError):
        result.warnings = ("changed",)

    with pytest.raises(ValidationError):
        ValidationResult.model_validate({"unexpected": True})
