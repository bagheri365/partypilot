"""Deterministic replanning comparison for PartyPilot v0.3 research."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from partypilot.application.state_invalidation import (
    StateInvalidationResult,
    apply_updates,
    build_replanning_metrics,
)
from partypilot.domain.planning_state import (
    PlanningState,
    PlanningStateSummary,
    PlanningUpdate,
    ReplanningComparisonMetrics,
)


class ReplanningStrategy(StrEnum):
    """Supported deterministic replanning strategies."""

    FULL_REPLAN = "full_replan"
    TARGETED_REPLAN = "targeted_replan"


class PlanningReplanResult(BaseModel):
    """A concrete replanning outcome for one strategy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: ReplanningStrategy
    invalidation: StateInvalidationResult
    recomputed_decision_ids: tuple[str, ...]
    recomputed_decision_count: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    cycle_detected: bool = False
    cycle_decision_ids: tuple[str, ...] = ()
    cycle_error: str | None = None


def _full_replan(
    state: PlanningState, updates: Sequence[PlanningUpdate]
) -> StateInvalidationResult:
    invalidation = apply_updates(state, updates)
    transition = invalidation.updated_state.transition_log[-1].model_copy(
        update={
            "recompute_steps": ("recompute full plan",),
            "affected_dependency_kinds": tuple(),
            "affected_dependency_ids": tuple(),
        }
    )
    updated_state = invalidation.updated_state.model_copy(
        update={"transition_log": (*state.transition_log, transition)}
    )
    return invalidation.model_copy(
        update={
            "updated_state": updated_state,
            "recompute_steps": ("recompute full plan",),
        }
    )


def apply_targeted_replanning(
    state: PlanningState, updates: Sequence[PlanningUpdate]
) -> PlanningReplanResult:
    """Apply dependency-aware targeted replanning to a planning state."""

    latency_ms, invalidation = _measure(lambda: apply_updates(state, updates))
    return PlanningReplanResult(
        strategy=ReplanningStrategy.TARGETED_REPLAN,
        invalidation=invalidation,
        recomputed_decision_ids=invalidation.invalidated_decision_ids,
        recomputed_decision_count=len(invalidation.invalidated_decision_ids),
        latency_ms=latency_ms,
        cycle_detected=invalidation.cycle_detected,
        cycle_decision_ids=invalidation.cycle_decision_ids,
        cycle_error=invalidation.cycle_error,
    )


def apply_full_replanning(
    state: PlanningState, updates: Sequence[PlanningUpdate]
) -> PlanningReplanResult:
    """Recompute the full plan from scratch after applying the same update set."""

    latency_ms, invalidation = _measure(lambda: _full_replan(state, updates))
    return PlanningReplanResult(
        strategy=ReplanningStrategy.FULL_REPLAN,
        invalidation=invalidation,
        recomputed_decision_ids=tuple(decision.decision_id for decision in state.decisions),
        recomputed_decision_count=len(state.decisions),
        latency_ms=latency_ms,
        cycle_detected=invalidation.cycle_detected,
        cycle_decision_ids=invalidation.cycle_decision_ids,
        cycle_error=invalidation.cycle_error,
    )


class PlanningReplanningComparison(BaseModel):
    """Comparison between full replanning and targeted replanning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_state: PlanningStateSummary
    updates: tuple[PlanningUpdate, ...]
    full_replan: PlanningReplanResult
    targeted_replan: PlanningReplanResult
    metrics: ReplanningComparisonMetrics
    notes: tuple[str, ...] = ()


def compare_replanning_strategies(
    state: PlanningState,
    updates: Sequence[PlanningUpdate],
    *,
    expected_invalidated_decision_ids: Sequence[str] | None = None,
    expected_preserved_decision_ids: Sequence[str] | None = None,
) -> PlanningReplanningComparison:
    """Run both replanning strategies and summarize the deterministic comparison."""

    targeted = apply_targeted_replanning(state, updates)
    full = apply_full_replanning(state, updates)
    expected_invalidated = (
        tuple(expected_invalidated_decision_ids)
        if expected_invalidated_decision_ids is not None
        else targeted.invalidation.invalidated_decision_ids
    )
    expected_preserved = (
        tuple(expected_preserved_decision_ids)
        if expected_preserved_decision_ids is not None
        else targeted.invalidation.preserved_decision_ids
    )
    metrics = build_replanning_metrics(
        full_replan_decision_count=full.recomputed_decision_count,
        targeted_replan_decision_count=targeted.recomputed_decision_count,
        full_replan_latency_ms=full.latency_ms,
        targeted_replan_latency_ms=targeted.latency_ms,
        expected_invalidated_decision_ids=expected_invalidated,
        expected_preserved_decision_ids=expected_preserved,
        actual_invalidated_decision_ids=targeted.invalidation.invalidated_decision_ids,
        actual_preserved_decision_ids=targeted.invalidation.preserved_decision_ids,
    )
    return PlanningReplanningComparison(
        initial_state=PlanningStateSummary.from_state(state),
        updates=tuple(updates),
        full_replan=full,
        targeted_replan=targeted,
        metrics=metrics,
    )


def _measure[T](func: Callable[[], T]) -> tuple[float, T]:
    started = perf_counter()
    result = func()
    return max(0.0, (perf_counter() - started) * 1000.0), result
