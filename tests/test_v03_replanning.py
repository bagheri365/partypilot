from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from partypilot.application import v03_replanning as v03
from partypilot.application.targeted_replanning import (
    apply_full_replanning,
    apply_targeted_replanning,
)
from partypilot.cli import eval_v03_replanning as eval_v03
from partypilot.domain import (
    PartyRequest,
    PlanningDecision,
    PlanningDecisionCategory,
    PlanningState,
    PlanningUpdate,
    PlanningUpdateKind,
)


def test_v03_benchmark_loads_expected_fixture_set() -> None:
    scenarios = v03.load_v03_replanning_benchmark()

    assert len(scenarios) == 5
    assert {
        "cap-boundary-51-incremental-replanning",
        "cap-boundary-52-new-safety-constraint-after-planning",
        "cap-boundary-55-cascading-failure",
        "v0-3-control-no-op-update",
        "v0-3-control-broad-update",
    } == {scenario.scenario_id for scenario in scenarios}
    assert any(
        scenario.scenario_id == "cap-boundary-51-incremental-replanning" for scenario in scenarios
    )
    assert any(
        scenario.scenario_id == "cap-boundary-52-new-safety-constraint-after-planning"
        for scenario in scenarios
    )
    assert any(
        scenario.scenario_id == "cap-boundary-55-cascading-failure" for scenario in scenarios
    )


def test_v03_replanning_experiment_prefers_targeted_replanning() -> None:
    report = v03.run_v03_replanning_experiment()

    assert report.metrics.scenario_count == 5
    assert report.metrics.full_replan.invalidation_accuracy == 1.0
    assert report.metrics.full_replan.preserved_decision_accuracy == 1.0
    assert report.metrics.full_replan.final_state_correctness == 1.0
    assert report.metrics.targeted_replan.invalidation_accuracy == 1.0
    assert report.metrics.targeted_replan.preserved_decision_accuracy == 1.0
    assert report.metrics.targeted_replan.final_state_correctness == 1.0
    assert report.metrics.targeted_replan.missed_recomputation_count == 0
    assert report.metrics.recomputation_reduction_ratio > 0.25
    assert report.metrics.retention_rule_passed is True
    assert (
        report.metrics.full_replan.recomputed_decision_count
        > report.metrics.targeted_replan.recomputed_decision_count
    )
    assert (
        report.metrics.full_replan.unnecessary_recomputation_count
        > report.metrics.targeted_replan.unnecessary_recomputation_count
    )
    assert (
        report.metrics.full_replan.final_state_correctness
        == report.metrics.targeted_replan.final_state_correctness
    )

    scenario_55 = next(
        scenario
        for scenario in report.scenarios
        if scenario.scenario_id == "cap-boundary-55-cascading-failure"
    )
    assert scenario_55.failure_stage is None
    assert scenario_55.targeted_replan.invalidated_decision_ids == (
        "rain_contingency",
        "indoor_move",
        "indoor_setup_space",
        "staffing_adjustment",
        "cost_recalculation",
        "budget_confirmation",
    )
    assert scenario_55.targeted_replan.preserved_decision_ids == (
        "theme",
        "dietary_policy",
        "accessibility",
    )


def test_full_replan_recomputes_every_decision_and_remains_correct() -> None:
    scenario = next(
        scenario
        for scenario in v03.load_v03_replanning_benchmark()
        if scenario.scenario_id == "cap-boundary-51-incremental-replanning"
    )

    result = apply_full_replanning(scenario.initial_state, scenario.updates)

    assert result.recomputed_decision_count == len(scenario.initial_state.decisions)
    assert result.invalidation.invalidated_decision_ids == (
        scenario.expected_invalidated_decision_ids
    )
    assert result.invalidation.preserved_decision_ids == scenario.expected_preserved_decision_ids
    assert result.invalidation.updated_state.revision_number == (
        scenario.initial_state.revision_number + 1
    )
    assert result.cycle_detected is False


def test_full_replan_has_higher_unnecessary_recomputation_count_than_targeted() -> None:
    scenario = next(
        scenario
        for scenario in v03.load_v03_replanning_benchmark()
        if scenario.scenario_id == "cap-boundary-52-new-safety-constraint-after-planning"
    )

    targeted = apply_targeted_replanning(scenario.initial_state, scenario.updates)
    full = apply_full_replanning(scenario.initial_state, scenario.updates)

    assert full.recomputed_decision_count > targeted.recomputed_decision_count
    assert len(full.recomputed_decision_ids) - len(
        scenario.expected_invalidated_decision_ids
    ) > len(targeted.recomputed_decision_ids) - len(scenario.expected_invalidated_decision_ids)
    assert full.invalidation.invalidated_decision_ids == (
        targeted.invalidation.invalidated_decision_ids
    )


def test_targeted_replanning_matches_full_final_state_with_less_work() -> None:
    scenario = next(
        scenario
        for scenario in v03.load_v03_replanning_benchmark()
        if scenario.scenario_id == "cap-boundary-55-cascading-failure"
    )

    targeted = apply_targeted_replanning(scenario.initial_state, scenario.updates)
    full = apply_full_replanning(scenario.initial_state, scenario.updates)

    assert targeted.invalidation.invalidated_decision_ids == (
        full.invalidation.invalidated_decision_ids
    )
    assert targeted.invalidation.preserved_decision_ids == full.invalidation.preserved_decision_ids
    assert targeted.recomputed_decision_count < full.recomputed_decision_count
    assert targeted.invalidation.updated_state.decisions == (
        full.invalidation.updated_state.decisions
    )
    assert targeted.invalidation.updated_state.invalidated_decision_ids == (
        full.invalidation.updated_state.invalidated_decision_ids
    )
    assert targeted.invalidation.updated_state.revision_number == (
        full.invalidation.updated_state.revision_number
    )
    assert targeted.invalidation.updated_state.transition_log[-1].recompute_steps


def test_targeted_metrics_fail_when_a_stale_decision_is_retained() -> None:
    scenario = next(
        scenario
        for scenario in v03.load_v03_replanning_benchmark()
        if scenario.scenario_id == "cap-boundary-51-incremental-replanning"
    )
    targeted = apply_targeted_replanning(scenario.initial_state, scenario.updates)
    stale_invalidation = targeted.invalidation.model_copy(
        update={
            "invalidated_decision_ids": ("venue_capacity",),
            "preserved_decision_ids": (
                "catering_quantity",
                "seating",
                "parking",
                "total_cost",
                "theme",
                "dietary_policies",
                "entertainment",
                "accessibility_requirements",
            ),
        }
    )
    stale_result = targeted.model_copy(
        update={
            "invalidation": stale_invalidation,
            "recomputed_decision_ids": ("venue_capacity",),
            "recomputed_decision_count": 1,
        }
    )

    metrics = v03._scenario_strategy_metrics(
        stale_result,
        scenario.expected_invalidated_decision_ids,
        scenario.expected_preserved_decision_ids,
    )

    assert metrics.final_state_correctness == 0.0
    assert metrics.invalidation_accuracy < 1.0
    assert metrics.preserved_decision_accuracy == 1.0


def test_targeted_metrics_fail_when_transitive_dependency_is_missed() -> None:
    scenario = next(
        scenario
        for scenario in v03.load_v03_replanning_benchmark()
        if scenario.scenario_id == "cap-boundary-55-cascading-failure"
    )
    targeted = apply_targeted_replanning(scenario.initial_state, scenario.updates)
    missed_invalidation = targeted.invalidation.model_copy(
        update={
            "invalidated_decision_ids": (
                "rain_contingency",
                "indoor_move",
                "indoor_setup_space",
                "staffing_adjustment",
                "cost_recalculation",
            ),
            "preserved_decision_ids": (
                "budget_confirmation",
                "theme",
                "dietary_policy",
                "accessibility",
            ),
        }
    )
    missed_result = targeted.model_copy(
        update={
            "invalidation": missed_invalidation,
            "recomputed_decision_ids": (
                "rain_contingency",
                "indoor_move",
                "indoor_setup_space",
                "staffing_adjustment",
                "cost_recalculation",
            ),
            "recomputed_decision_count": 5,
        }
    )

    metrics = v03._scenario_strategy_metrics(
        missed_result,
        scenario.expected_invalidated_decision_ids,
        scenario.expected_preserved_decision_ids,
    )

    assert metrics.missed_recomputation_count == 1
    assert metrics.final_state_correctness == 0.0
    assert metrics.invalidation_accuracy < 1.0


def test_v03_cycle_detection_is_safe() -> None:
    request = PartyRequest(
        location="Boston",
        event_date=date(2027, 3, 1),
        guest_count=20,
        total_budget=Decimal("500.00"),
    )
    decisions = (
        PlanningDecision(
            decision_id="decision-a",
            category=PlanningDecisionCategory.OTHER,
            summary="Decision A depends on B",
            prerequisite_decision_ids=("decision-b",),
        ),
        PlanningDecision(
            decision_id="decision-b",
            category=PlanningDecisionCategory.OTHER,
            summary="Decision B depends on A",
            prerequisite_decision_ids=("decision-a",),
        ),
    )
    state = PlanningState(
        revision_number=1,
        request=request,
        decisions=decisions,
        dependency_relationships=(),
    )
    update = PlanningUpdate(
        update_id="noop-cycle",
        kind=PlanningUpdateKind.NO_OP,
        description="No-op update that still exercises cycle detection",
    )

    targeted = apply_targeted_replanning(state, (update,))
    full = apply_full_replanning(state, (update,))

    assert targeted.cycle_detected is True
    assert full.cycle_detected is True
    assert targeted.invalidation.cycle_detected is True
    assert full.invalidation.cycle_detected is True
    assert targeted.invalidation.cycle_error is not None
    assert full.invalidation.cycle_error is not None
    assert set(targeted.invalidation.invalidated_decision_ids) == {"decision-a", "decision-b"}
    assert set(full.invalidation.invalidated_decision_ids) == {"decision-a", "decision-b"}


def test_v03_replanning_cli_writes_artifacts_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixed_timestamp = datetime(2026, 8, 11, 12, 34, tzinfo=UTC)

    def fake_git_metadata() -> tuple[str | None, bool | None, str | None]:
        return "abc123", False, None

    monkeypatch.setattr(v03, "_git_metadata", fake_git_metadata)
    monkeypatch.setattr(
        eval_v03,
        "datetime",
        SimpleNamespace(now=lambda tz: fixed_timestamp),
    )

    exit_code = eval_v03.main(["--output-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Scenario count: 5" in captured.out
    assert "Retention rule passed: True" in captured.out
    assert "Targeted-vs-full recomputation reduction ratio:" in captured.out
    assert "Failure cases: none" in captured.out

    json_path = tmp_path / "v0_3_replanning.json"
    markdown_path = tmp_path / "v0_3_replanning.md"
    assert json_path.exists()
    assert markdown_path.exists()
    assert "PartyPilot v0.3 Replanning Experiment" in markdown_path.read_text(encoding="utf-8")
    assert "Targeted-vs-full recomputation reduction ratio" in markdown_path.read_text(
        encoding="utf-8"
    )
