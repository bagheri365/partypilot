"""Deterministic invalidation logic for PartyPilot v0.3 planning state."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict

from partypilot.domain.planning_state import (
    PlanningDecision,
    PlanningDecisionStatus,
    PlanningDependencyKind,
    PlanningState,
    PlanningStateTransition,
    PlanningUpdate,
    PlanningUpdateKind,
    ReplanningComparisonMetrics,
)


class StateInvalidationResult(BaseModel):
    """Result of applying one or more planning updates to a state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    previous_state: PlanningState
    updated_state: PlanningState
    updates: tuple[PlanningUpdate, ...]
    affected_dependency_kinds: tuple[PlanningDependencyKind, ...]
    affected_dependency_ids: tuple[str, ...]
    invalidated_decision_ids: tuple[str, ...]
    preserved_decision_ids: tuple[str, ...]
    recompute_steps: tuple[str, ...]
    cycle_detected: bool = False
    cycle_decision_ids: tuple[str, ...] = ()
    cycle_error: str | None = None


def _unique_ordered(items: Iterable[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return tuple(ordered)


def _build_decision_graph(
    state: PlanningState,
) -> tuple[dict[str, PlanningDecision], dict[str, tuple[str, ...]]]:
    decision_by_id = {decision.decision_id: decision for decision in state.decisions}
    dependents: dict[str, list[str]] = {decision_id: [] for decision_id in decision_by_id}
    for decision in state.decisions:
        for prerequisite_id in decision.prerequisite_decision_ids:
            if prerequisite_id in decision_by_id:
                dependents[prerequisite_id].append(decision.decision_id)
    return decision_by_id, {
        decision_id: tuple(dict.fromkeys(dependents[decision_id])) for decision_id in dependents
    }


def _detect_cycle(
    state: PlanningState,
) -> tuple[bool, tuple[str, ...], str | None]:
    decision_by_id = {decision.decision_id: decision for decision in state.decisions}
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def visit(decision_id: str) -> tuple[bool, tuple[str, ...], str | None]:
        if decision_id in visiting:
            start = stack.index(decision_id)
            cycle = tuple([*stack[start:], decision_id])
            return True, cycle, f"dependency cycle detected: {' -> '.join(cycle)}"
        if decision_id in visited:
            return False, tuple(), None

        visiting.add(decision_id)
        stack.append(decision_id)
        decision = decision_by_id[decision_id]
        for prerequisite_id in decision.prerequisite_decision_ids:
            if prerequisite_id not in decision_by_id:
                continue
            detected, cycle_ids, error = visit(prerequisite_id)
            if detected:
                return True, cycle_ids, error
        stack.pop()
        visiting.remove(decision_id)
        visited.add(decision_id)
        return False, tuple(), None

    for decision in state.decisions:
        detected, cycle_ids, error = visit(decision.decision_id)
        if detected:
            return detected, cycle_ids, error
    return False, tuple(), None


def _propagate_invalidation(
    seed_ids: Iterable[str],
    dependents_by_id: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    invalidated: list[str] = []
    seen: set[str] = set()
    queue = deque(dict.fromkeys(seed_ids))
    while queue:
        decision_id = queue.popleft()
        if decision_id in seen:
            continue
        seen.add(decision_id)
        invalidated.append(decision_id)
        for dependent_id in dependents_by_id.get(decision_id, ()):
            if dependent_id not in seen:
                queue.append(dependent_id)
    return tuple(invalidated)


def affected_dependency_kinds_for_updates(
    updates: Sequence[PlanningUpdate],
) -> tuple[PlanningDependencyKind, ...]:
    kinds: list[PlanningDependencyKind] = []
    for update in updates:
        kinds.extend(_affected_dependency_kinds_for_update(update))
    return tuple(dict.fromkeys(kinds))


def _affected_dependency_kinds_for_update(
    update: PlanningUpdate,
) -> tuple[PlanningDependencyKind, ...]:
    if update.kind is PlanningUpdateKind.GUEST_COUNT_CHANGED:
        return (
            PlanningDependencyKind.GUEST_COUNT_TO_VENUE_CAPACITY,
            PlanningDependencyKind.GUEST_COUNT_TO_CATERING_QUANTITY,
            PlanningDependencyKind.GUEST_COUNT_TO_CATERING_COST,
            PlanningDependencyKind.GUEST_COUNT_TO_SEATING,
            PlanningDependencyKind.GUEST_COUNT_TO_PARKING,
            PlanningDependencyKind.BUDGET_TO_RESOURCE_SELECTION,
            PlanningDependencyKind.BUDGET_TO_TOTAL_COST,
        )
    if update.kind is PlanningUpdateKind.BUDGET_CHANGED:
        return (
            PlanningDependencyKind.BUDGET_TO_RESOURCE_SELECTION,
            PlanningDependencyKind.BUDGET_TO_TOTAL_COST,
            PlanningDependencyKind.FEES_TO_TOTAL_COST,
            PlanningDependencyKind.GUEST_COUNT_TO_CATERING_COST,
        )
    if update.kind is PlanningUpdateKind.DATE_TIME_CHANGED:
        return (
            PlanningDependencyKind.SCHEDULE_TO_VENDOR_AVAILABILITY,
            PlanningDependencyKind.SCHEDULE_TO_SETUP_WINDOW,
        )
    if update.kind is PlanningUpdateKind.NEW_ALLERGY_ADDED:
        return (
            PlanningDependencyKind.DIETARY_TO_CATERING_EVIDENCE,
            PlanningDependencyKind.NEW_EVIDENCE_TO_POLICY_VALIDITY,
        )
    if update.kind is PlanningUpdateKind.ACCESSIBILITY_REQUIREMENT_ADDED:
        return (
            PlanningDependencyKind.ACCESSIBILITY_TO_VENUE,
            PlanningDependencyKind.ACCESSIBILITY_TO_PATH,
            PlanningDependencyKind.ACCESSIBILITY_TO_ROOM,
            PlanningDependencyKind.ACCESSIBILITY_TO_RESTROOM,
        )
    if update.kind is PlanningUpdateKind.VENDOR_UNAVAILABLE:
        return (
            PlanningDependencyKind.VENUE_TO_APPROVED_CATERERS,
            PlanningDependencyKind.VENUE_TO_ACTIVITY_SPACE,
            PlanningDependencyKind.SCHEDULE_TO_VENDOR_AVAILABILITY,
        )
    if update.kind is PlanningUpdateKind.NEW_EVIDENCE_DISCOVERED:
        return (PlanningDependencyKind.NEW_EVIDENCE_TO_POLICY_VALIDITY,)
    if update.kind is PlanningUpdateKind.FEE_RULE_CHANGED:
        return (
            PlanningDependencyKind.FEES_TO_TOTAL_COST,
            PlanningDependencyKind.BUDGET_TO_TOTAL_COST,
        )
    return ()


def _step_labels_for_dependency_kind(kind: PlanningDependencyKind) -> tuple[str, ...]:
    mapping = {
        PlanningDependencyKind.GUEST_COUNT_TO_VENUE_CAPACITY: ("recheck venue capacity",),
        PlanningDependencyKind.GUEST_COUNT_TO_CATERING_QUANTITY: ("recheck catering quantity",),
        PlanningDependencyKind.GUEST_COUNT_TO_CATERING_COST: ("recheck catering cost",),
        PlanningDependencyKind.GUEST_COUNT_TO_SEATING: ("recheck seating capacity",),
        PlanningDependencyKind.GUEST_COUNT_TO_PARKING: ("recheck parking capacity",),
        PlanningDependencyKind.VENUE_TO_APPROVED_CATERERS: ("recheck approved caterers",),
        PlanningDependencyKind.VENUE_TO_ACTIVITY_SPACE: ("recheck activity space",),
        PlanningDependencyKind.ACCESSIBILITY_TO_VENUE: ("recheck venue accessibility",),
        PlanningDependencyKind.ACCESSIBILITY_TO_PATH: ("recheck accessible path",),
        PlanningDependencyKind.ACCESSIBILITY_TO_ROOM: ("recheck accessible room",),
        PlanningDependencyKind.ACCESSIBILITY_TO_RESTROOM: ("recheck accessible restroom",),
        PlanningDependencyKind.DIETARY_TO_CATERING_EVIDENCE: ("recheck dietary evidence",),
        PlanningDependencyKind.SCHEDULE_TO_VENDOR_AVAILABILITY: ("recheck vendor availability",),
        PlanningDependencyKind.SCHEDULE_TO_SETUP_WINDOW: ("recheck setup window",),
        PlanningDependencyKind.BUDGET_TO_RESOURCE_SELECTION: ("recheck budgeted selections",),
        PlanningDependencyKind.BUDGET_TO_TOTAL_COST: ("recheck total cost",),
        PlanningDependencyKind.FEES_TO_TOTAL_COST: ("recheck total cost for new fees",),
        PlanningDependencyKind.NEW_EVIDENCE_TO_POLICY_VALIDITY: (
            "recheck evidence-backed policy validity",
        ),
    }
    return mapping[kind]


def _decision_matches_dependency_kind(
    decision: PlanningDecision,
    affected_dependency_ids: set[str],
    unavailable_resource_ids: set[str],
    evidence_document_ids: set[str],
) -> bool:
    if affected_dependency_ids.intersection(decision.dependency_ids):
        return True
    if unavailable_resource_ids.intersection(decision.resource_ids):
        return True
    return bool(evidence_document_ids.intersection(decision.evidence_ids))


def apply_updates(
    state: PlanningState,
    updates: Sequence[PlanningUpdate],
) -> StateInvalidationResult:
    """Apply updates while invalidating only impacted decisions."""

    update_sequence = tuple(updates)
    affected_kinds = affected_dependency_kinds_for_updates(update_sequence)
    cycle_detected, cycle_decision_ids, cycle_error = _detect_cycle(state)
    decision_by_id, dependents_by_id = _build_decision_graph(state)
    affected_dependencies = tuple(
        dependency
        for dependency in state.dependency_relationships
        if dependency.kind in affected_kinds
    )
    affected_dependency_ids = _unique_ordered(
        dependency.dependency_id for dependency in affected_dependencies
    )
    unavailable_resource_ids = {
        resource_id for update in update_sequence for resource_id in update.unavailable_resource_ids
    }
    evidence_document_ids = {
        document_id for update in update_sequence for document_id in update.evidence_document_ids
    }
    direct_invalidated_ids: list[str] = []

    for decision in state.decisions:
        if _decision_matches_dependency_kind(
            decision,
            set(affected_dependency_ids),
            unavailable_resource_ids,
            evidence_document_ids,
        ):
            direct_invalidated_ids.append(decision.decision_id)

    invalidated_ids = list(_propagate_invalidation(direct_invalidated_ids, dependents_by_id))
    if cycle_detected:
        invalidated_ids = list(_unique_ordered((*invalidated_ids, *cycle_decision_ids)))
    invalidated_id_set = set(invalidated_ids)

    updated_decisions: list[PlanningDecision] = []
    preserved_ids: list[str] = []
    for decision_id in decision_by_id:
        decision = decision_by_id[decision_id]
        if decision_id in invalidated_id_set:
            updated_decisions.append(
                decision.model_copy(update={"status": PlanningDecisionStatus.INVALIDATED})
            )
        else:
            preserved_ids.append(decision_id)
            updated_decisions.append(
                decision.model_copy(update={"status": PlanningDecisionStatus.PRESERVED})
            )

    recompute_steps = _unique_ordered(
        step for kind in affected_kinds for step in _step_labels_for_dependency_kind(kind)
    )
    transition = PlanningStateTransition(
        from_revision_number=state.revision_number,
        to_revision_number=state.revision_number + 1,
        updates=update_sequence,
        affected_dependency_kinds=affected_kinds,
        affected_dependency_ids=affected_dependency_ids,
        invalidated_decision_ids=tuple(invalidated_ids),
        preserved_decision_ids=tuple(preserved_ids),
        recompute_steps=recompute_steps,
        cycle_detected=cycle_detected,
        cycle_decision_ids=cycle_decision_ids,
        cycle_error=cycle_error,
    )
    updated_state = state.model_copy(
        update={
            "revision_number": state.revision_number + 1,
            "decisions": tuple(updated_decisions),
            "invalidated_decision_ids": tuple(invalidated_ids),
            "transition_log": (*state.transition_log, transition),
        }
    )
    return StateInvalidationResult(
        previous_state=state,
        updated_state=updated_state,
        updates=update_sequence,
        affected_dependency_kinds=affected_kinds,
        affected_dependency_ids=affected_dependency_ids,
        invalidated_decision_ids=tuple(invalidated_ids),
        preserved_decision_ids=tuple(preserved_ids),
        recompute_steps=recompute_steps,
        cycle_detected=cycle_detected,
        cycle_decision_ids=cycle_decision_ids,
        cycle_error=cycle_error,
    )


def _accuracy(actual: Sequence[str], expected: Sequence[str]) -> float:
    expected_set = set(expected)
    actual_set = set(actual)
    if not expected_set:
        return 1.0 if not actual_set else 0.0
    return len(actual_set & expected_set) / len(expected_set)


def build_replanning_metrics(
    *,
    full_replan_decision_count: int,
    targeted_replan_decision_count: int,
    full_replan_latency_ms: float,
    targeted_replan_latency_ms: float,
    expected_invalidated_decision_ids: Sequence[str],
    expected_preserved_decision_ids: Sequence[str],
    actual_invalidated_decision_ids: Sequence[str],
    actual_preserved_decision_ids: Sequence[str],
) -> ReplanningComparisonMetrics:
    invalidation_accuracy = _accuracy(
        actual_invalidated_decision_ids,
        expected_invalidated_decision_ids,
    )
    preserved_decision_accuracy = _accuracy(
        actual_preserved_decision_ids,
        expected_preserved_decision_ids,
    )
    correctness = (invalidation_accuracy + preserved_decision_accuracy) / 2.0
    expected_invalidated = set(expected_invalidated_decision_ids)
    actual_invalidated = set(actual_invalidated_decision_ids)
    missed_recomputation_count = len(expected_invalidated - actual_invalidated)
    unnecessary_recomputed = len(actual_invalidated - expected_invalidated)
    reduction_ratio = (
        1.0 - (targeted_replan_decision_count / full_replan_decision_count)
        if full_replan_decision_count > 0
        else 0.0
    )
    final_state_correctness = float(
        set(actual_invalidated_decision_ids) == expected_invalidated
        and set(actual_preserved_decision_ids) == set(expected_preserved_decision_ids)
    )
    return ReplanningComparisonMetrics(
        correctness=correctness,
        final_state_correctness=final_state_correctness,
        preserved_decision_accuracy=preserved_decision_accuracy,
        invalidation_accuracy=invalidation_accuracy,
        missed_recomputation_count=missed_recomputation_count,
        full_replan_decision_count=full_replan_decision_count,
        targeted_replan_decision_count=targeted_replan_decision_count,
        unnecessary_recomputed_decision_count=unnecessary_recomputed,
        recomputation_reduction_ratio=reduction_ratio,
        full_replan_latency_ms=full_replan_latency_ms,
        targeted_replan_latency_ms=targeted_replan_latency_ms,
    )
