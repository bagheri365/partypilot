from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from partypilot.domain.evaluation import DatasetSplit, EvaluationScenario, ScenarioCategory

DATASET_PATH = Path(__file__).parents[1] / "data" / "evaluation" / "core_scenarios.json"


def load_scenarios() -> tuple[EvaluationScenario, ...]:
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return TypeAdapter(tuple[EvaluationScenario, ...]).validate_python(payload)


def test_initial_dataset_contains_at_least_twenty_valid_scenarios() -> None:
    scenarios = load_scenarios()

    assert len(scenarios) >= 20
    assert len({scenario.scenario_id for scenario in scenarios}) == len(scenarios)


def test_initial_dataset_uses_all_required_splits() -> None:
    splits = {scenario.dataset_split for scenario in load_scenarios()}

    assert splits == set(DatasetSplit)


def test_initial_dataset_covers_required_categories() -> None:
    categories = {scenario.scenario_category for scenario in load_scenarios()}
    required = {
        ScenarioCategory.FEASIBLE,
        ScenarioCategory.BUDGET,
        ScenarioCategory.CAPACITY,
        ScenarioCategory.AVAILABILITY,
        ScenarioCategory.AGE_RESTRICTION,
        ScenarioCategory.ACCESSIBILITY,
        ScenarioCategory.TEMPORAL,
        ScenarioCategory.RESOURCE_CONFLICT,
        ScenarioCategory.MULTIPLE_CHOICES,
        ScenarioCategory.IMPOSSIBLE_COMBINATION,
        ScenarioCategory.SAFETY_EVIDENCE,
    }

    assert required <= categories


def test_safety_evidence_cases_require_human_review_and_reference_evidence() -> None:
    scenarios = [
        scenario
        for scenario in load_scenarios()
        if scenario.scenario_category is ScenarioCategory.SAFETY_EVIDENCE
    ]

    assert scenarios
    assert all(
        scenario.expected_feasibility.value == "HUMAN_REVIEW_REQUIRED" for scenario in scenarios
    )
    assert all(scenario.relevant_evidence_ids for scenario in scenarios)


def test_dataset_contains_simple_and_complex_cases() -> None:
    scenarios = load_scenarios()
    counts = {scenario.complexity.hard_constraint_count for scenario in scenarios}

    assert min(counts) <= 2
    assert max(counts) >= 5
