from __future__ import annotations

from datetime import date, time
from decimal import Decimal

import pytest
from pydantic import ValidationError

from partypilot.adapters import DEFAULT_RESOURCES
from partypilot.application.capability_boundary_benchmark import load_capability_boundary_scenarios
from partypilot.application.state_invalidation import apply_updates
from partypilot.application.targeted_replanning import compare_replanning_strategies
from partypilot.domain import (
    Constraint,
    ConstraintOperator,
    ConstraintProvenance,
    ConstraintType,
    PartyRequest,
    PlanningDecision,
    PlanningDecisionCategory,
    PlanningDependency,
    PlanningDependencyKind,
    PlanningState,
    PlanningStateSummary,
    PlanningUpdate,
    PlanningUpdateKind,
    Resource,
)


def _request() -> PartyRequest:
    return PartyRequest(
        location="Brooklyn, NY",
        event_date=date(2026, 9, 20),
        event_time=time(18, 0),
        guest_count=60,
        total_budget=Decimal("2500"),
        theme_preferences=("garden",),
        allergies=("peanut",),
        dietary_restrictions=("vegan",),
        accessibility_needs=("wheelchair_accessible",),
    )


def _dependencies() -> tuple[PlanningDependency, ...]:
    return (
        PlanningDependency(
            dependency_id="dep-guest-capacity",
            kind=PlanningDependencyKind.GUEST_COUNT_TO_VENUE_CAPACITY,
            source="guest_count",
            target="venue_capacity",
            description="Guest count must fit venue capacity",
        ),
        PlanningDependency(
            dependency_id="dep-guest-catering",
            kind=PlanningDependencyKind.GUEST_COUNT_TO_CATERING_COST,
            source="guest_count",
            target="catering_cost",
            description="Guest count affects catering cost",
        ),
        PlanningDependency(
            dependency_id="dep-accessibility",
            kind=PlanningDependencyKind.ACCESSIBILITY_TO_VENUE,
            source="accessibility_needs",
            target="venue_accessibility",
            description="Accessibility requirement affects venue selection",
        ),
        PlanningDependency(
            dependency_id="dep-dietary",
            kind=PlanningDependencyKind.DIETARY_TO_CATERING_EVIDENCE,
            source="dietary_restrictions",
            target="catering_evidence",
            description="Dietary restriction affects catering evidence",
        ),
        PlanningDependency(
            dependency_id="dep-schedule",
            kind=PlanningDependencyKind.SCHEDULE_TO_VENDOR_AVAILABILITY,
            source="event_time",
            target="vendor_availability",
            description="Event timing affects vendor availability",
        ),
        PlanningDependency(
            dependency_id="dep-budget",
            kind=PlanningDependencyKind.BUDGET_TO_TOTAL_COST,
            source="total_budget",
            target="total_cost",
            description="Budget affects total cost",
        ),
        PlanningDependency(
            dependency_id="dep-fees",
            kind=PlanningDependencyKind.FEES_TO_TOTAL_COST,
            source="fee_rules",
            target="total_cost",
            description="Fee rules affect total cost",
        ),
    )


def _selected_resources() -> tuple[Resource, ...]:
    return DEFAULT_RESOURCES[:3]


def _decisions() -> tuple[PlanningDecision, ...]:
    return (
        PlanningDecision(
            decision_id="venue-selection",
            category=PlanningDecisionCategory.RESOURCE_SELECTION,
            summary="Select an accessible venue with sufficient capacity",
            dependency_ids=("dep-guest-capacity", "dep-accessibility", "dep-schedule"),
            resource_ids=("venue-brooklyn-loft",),
            evidence_ids=("doc-loft-accessibility-current",),
        ),
        PlanningDecision(
            decision_id="caterer-selection",
            category=PlanningDecisionCategory.RESOURCE_SELECTION,
            summary="Select a caterer compatible with guest count and diet",
            dependency_ids=("dep-guest-catering", "dep-dietary", "dep-fees"),
            resource_ids=("caterer-family-table",),
            evidence_ids=("doc-family-allergen-current",),
        ),
        PlanningDecision(
            decision_id="accessibility-clearance",
            category=PlanningDecisionCategory.ACCESSIBILITY,
            summary="Accessibility requirements are satisfied",
            dependency_ids=("dep-accessibility",),
            resource_ids=("venue-brooklyn-loft",),
        ),
        PlanningDecision(
            decision_id="dietary-safety",
            category=PlanningDecisionCategory.DIETARY,
            summary="Dietary restrictions are safe for the selected caterer",
            dependency_ids=("dep-dietary",),
            resource_ids=("caterer-family-table",),
            evidence_ids=("doc-family-allergen-current",),
        ),
        PlanningDecision(
            decision_id="budget-confirmation",
            category=PlanningDecisionCategory.BUDGET,
            summary="Total cost remains within budget",
            dependency_ids=("dep-guest-catering", "dep-budget", "dep-fees"),
            resource_ids=("venue-brooklyn-loft", "caterer-family-table"),
        ),
        PlanningDecision(
            decision_id="schedule-confirmation",
            category=PlanningDecisionCategory.SCHEDULE,
            summary="Selected vendors are available for the event time",
            dependency_ids=("dep-schedule",),
            resource_ids=("venue-brooklyn-loft", "caterer-family-table"),
        ),
        PlanningDecision(
            decision_id="theme-preference",
            category=PlanningDecisionCategory.PREFERENCE,
            summary="Garden theme preference is preserved",
            dependency_ids=(),
        ),
    )


def _state() -> PlanningState:
    evidence_constraint = Constraint(
        identifier="hard-accessibility",
        key="accessibility",
        operator=ConstraintOperator.EQ,
        value="wheelchair_accessible",
        constraint_type=ConstraintType.HARD,
        description="Venue must be wheelchair accessible",
    )
    derived_constraint = Constraint(
        identifier="derived-budget-ceiling",
        key="estimated_total_cost",
        operator=ConstraintOperator.LTE,
        value=Decimal("2500"),
        constraint_type=ConstraintType.DERIVED,
        description="Derived total cost ceiling",
        provenance=ConstraintProvenance(
            source_constraint_ids=("hard-accessibility",),
            derivation_explanation="Budget ceiling derived from the current plan.",
        ),
    )
    return PlanningState(
        revision_number=1,
        request=_request(),
        selected_resources=_selected_resources(),
        evidence_backed_constraints=(evidence_constraint,),
        derived_constraints=(derived_constraint,),
        unresolved_uncertainties=("caterer cross-contact policy needs review",),
        decisions=_decisions(),
        assumptions=("No weather disruption is expected.",),
        dependency_relationships=_dependencies(),
        notes=("Initial deterministic planning state for replanning research.",),
    )


def test_planning_state_tracks_revision_history_and_summary() -> None:
    state = _state()
    summary = PlanningStateSummary.from_state(state)

    assert summary.revision_number == 1
    assert summary.selected_resource_ids == tuple(
        resource.resource_id for resource in _selected_resources()
    )
    assert state.active_decisions
    assert state.preserved_decisions == ()
    assert state.invalidated_decisions == ()
    assert state.evidence_backed_constraints[0].constraint_type is ConstraintType.HARD
    assert state.derived_constraints[0].constraint_type is ConstraintType.DERIVED


def test_planning_state_rejects_derived_evidence_backed_constraints() -> None:
    with pytest.raises(ValidationError, match="evidence-backed constraints must not be derived"):
        PlanningState(
            revision_number=1,
            request=_request(),
            evidence_backed_constraints=(
                Constraint(
                    identifier="bad-derived",
                    key="x",
                    operator=ConstraintOperator.EQ,
                    value="y",
                    constraint_type=ConstraintType.DERIVED,
                    description="Invalid evidence-backed constraint",
                    provenance=ConstraintProvenance(
                        source_constraint_ids=("hard-accessibility",),
                        derivation_explanation="bad",
                    ),
                ),
            ),
        )


def test_guest_count_update_requires_guest_count() -> None:
    with pytest.raises(ValidationError):
        PlanningUpdate.model_validate(
            {
                "update_id": "bad-guest-count",
                "kind": PlanningUpdateKind.GUEST_COUNT_CHANGED,
                "description": "guest count changed",
            }
        )


def test_allergy_update_requires_allergies() -> None:
    with pytest.raises(ValidationError):
        PlanningUpdate.model_validate(
            {
                "update_id": "bad-allergy",
                "kind": PlanningUpdateKind.NEW_ALLERGY_ADDED,
                "description": "allergy added",
            }
        )


def test_no_op_update_rejects_payload() -> None:
    with pytest.raises(ValidationError):
        PlanningUpdate.model_validate(
            {
                "update_id": "bad-no-op",
                "kind": PlanningUpdateKind.NO_OP,
                "description": "no change",
                "guest_count": 1,
            }
        )


def test_guest_count_increase_invalidates_capacity_catering_and_budget_dependencies() -> None:
    result = apply_updates(
        _state(),
        (
            PlanningUpdate(
                update_id="update-guest-count",
                kind=PlanningUpdateKind.GUEST_COUNT_CHANGED,
                description="Guest count increased from 60 to 85",
                guest_count=85,
            ),
        ),
    )

    assert result.affected_dependency_kinds == (
        PlanningDependencyKind.GUEST_COUNT_TO_VENUE_CAPACITY,
        PlanningDependencyKind.GUEST_COUNT_TO_CATERING_QUANTITY,
        PlanningDependencyKind.GUEST_COUNT_TO_CATERING_COST,
        PlanningDependencyKind.GUEST_COUNT_TO_SEATING,
        PlanningDependencyKind.GUEST_COUNT_TO_PARKING,
        PlanningDependencyKind.BUDGET_TO_RESOURCE_SELECTION,
        PlanningDependencyKind.BUDGET_TO_TOTAL_COST,
    )
    assert result.invalidated_decision_ids == (
        "venue-selection",
        "caterer-selection",
        "budget-confirmation",
    )
    assert "theme-preference" in result.preserved_decision_ids
    assert "dietary-safety" in result.preserved_decision_ids
    assert "accessibility-clearance" in result.preserved_decision_ids
    assert result.updated_state.revision_number == 2
    assert result.updated_state.transition_log[-1].from_revision_number == 1
    assert result.updated_state.transition_log[-1].to_revision_number == 2


def test_new_sesame_allergy_invalidates_catering_safety_and_marks_evidence_recheck() -> None:
    result = apply_updates(
        _state(),
        (
            PlanningUpdate(
                update_id="update-sesame",
                kind=PlanningUpdateKind.NEW_ALLERGY_ADDED,
                description="A severe sesame allergy was disclosed after planning",
                added_allergies=("sesame",),
            ),
        ),
    )

    assert result.affected_dependency_kinds == (
        PlanningDependencyKind.DIETARY_TO_CATERING_EVIDENCE,
        PlanningDependencyKind.NEW_EVIDENCE_TO_POLICY_VALIDITY,
    )
    assert result.invalidated_decision_ids == ("caterer-selection", "dietary-safety")
    assert "venue-selection" in result.preserved_decision_ids
    assert "theme-preference" in result.preserved_decision_ids
    assert "recheck dietary evidence" in result.recompute_steps
    assert "recheck evidence-backed policy validity" in result.recompute_steps


def test_budget_reduction_invalidates_cost_dependent_selections_but_preserves_safety() -> None:
    result = apply_updates(
        _state(),
        (
            PlanningUpdate(
                update_id="update-budget",
                kind=PlanningUpdateKind.BUDGET_CHANGED,
                description="Budget reduced after planning",
                total_budget=Decimal("1800"),
            ),
        ),
    )

    assert result.invalidated_decision_ids == ("caterer-selection", "budget-confirmation")
    assert "accessibility-clearance" in result.preserved_decision_ids
    assert "dietary-safety" in result.preserved_decision_ids
    assert "theme-preference" in result.preserved_decision_ids


def test_date_time_change_invalidates_schedule_related_decisions_and_preserves_policy_logic() -> (
    None
):
    result = apply_updates(
        _state(),
        (
            PlanningUpdate(
                update_id="update-date-time",
                kind=PlanningUpdateKind.DATE_TIME_CHANGED,
                description="Event moved to a later evening slot",
                event_date=date(2026, 9, 21),
                event_time=time(20, 0),
            ),
        ),
    )

    assert result.invalidated_decision_ids == ("venue-selection", "schedule-confirmation")
    assert "dietary-safety" in result.preserved_decision_ids
    assert "accessibility-clearance" in result.preserved_decision_ids
    assert "theme-preference" in result.preserved_decision_ids


def test_no_op_update_does_not_invalidate_decisions() -> None:
    result = apply_updates(
        _state(),
        (
            PlanningUpdate(
                update_id="update-no-op",
                kind=PlanningUpdateKind.NO_OP,
                description="No effective change",
            ),
        ),
    )

    assert result.invalidated_decision_ids == ()
    assert result.preserved_decision_ids == tuple(decision.decision_id for decision in _decisions())
    assert result.updated_state.revision_number == 2


def test_targeted_replanning_comparison_reports_full_and_targeted_counts() -> None:
    comparison = compare_replanning_strategies(
        _state(),
        (
            PlanningUpdate(
                update_id="update-guest-count",
                kind=PlanningUpdateKind.GUEST_COUNT_CHANGED,
                description="Guest count increased from 60 to 85",
                guest_count=85,
            ),
        ),
        expected_invalidated_decision_ids=(
            "venue-selection",
            "caterer-selection",
            "budget-confirmation",
        ),
        expected_preserved_decision_ids=(
            "accessibility-clearance",
            "dietary-safety",
            "schedule-confirmation",
            "theme-preference",
        ),
    )

    assert comparison.metrics.correctness == 1.0
    assert comparison.full_replan.recomputed_decision_count == len(_decisions())
    assert comparison.targeted_replan.recomputed_decision_count == 3
    assert comparison.metrics.correctness == 1.0
    assert comparison.metrics.preserved_decision_accuracy == 1.0
    assert comparison.metrics.invalidation_accuracy == 1.0
    assert comparison.metrics.unnecessary_recomputed_decision_count == 0
    assert comparison.metrics.full_replan_latency_ms >= 0
    assert comparison.metrics.targeted_replan_latency_ms >= 0


def test_capability_boundary_replanning_scenarios_are_representable() -> None:
    scenarios = load_capability_boundary_scenarios()
    scenario_map = {scenario.scenario.scenario_id: scenario for scenario in scenarios}

    assert scenario_map["cap-boundary-51-incremental-replanning"].metadata.requires_state_replanning
    assert scenario_map[
        "cap-boundary-52-new-safety-constraint-after-planning"
    ].metadata.requires_state_replanning
    assert scenario_map["cap-boundary-55-cascading-failure"].metadata.requires_state_replanning

    updates = (
        PlanningUpdate(
            update_id="cap-51-guest-count",
            kind=PlanningUpdateKind.GUEST_COUNT_CHANGED,
            description="Guest count increases after planning",
            guest_count=85,
        ),
        PlanningUpdate(
            update_id="cap-52-sesame",
            kind=PlanningUpdateKind.NEW_ALLERGY_ADDED,
            description="A new severe sesame allergy is introduced after planning",
            added_allergies=("sesame",),
        ),
        PlanningUpdate(
            update_id="cap-55-rain",
            kind=PlanningUpdateKind.DATE_TIME_CHANGED,
            description="Rain forces a time and setup revision",
            event_date=date(2026, 9, 21),
            event_time=time(19, 30),
        ),
    )

    assert all(isinstance(update.kind, PlanningUpdateKind) for update in updates)
