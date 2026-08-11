from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from partypilot.application.constraint_engine import ConstraintEngineResult
from partypilot.application.evidence_grounded_planner import (
    EvidenceGroundedPlanCandidate,
    EvidenceGroundedPlanningResult,
)
from partypilot.application.v02_evaluation import (
    V02EvaluationReport,
    V02EvaluationRunner,
    load_retrieval_snapshots,
    load_v01_baseline_snapshot,
    render_v02_evaluation_markdown,
    save_v02_evaluation_reports,
)
from partypilot.domain.evaluation import (
    ComplexityMetadata,
    DatasetSplit,
    EvaluationScenario,
    RetrievalGroundTruthLabel,
    ScenarioCategory,
)
from partypilot.domain.evidence_corpus import (
    EvidenceDocument,
    EvidenceDocumentMetadata,
    EvidenceDocumentStatus,
    EvidenceDocumentType,
)
from partypilot.domain.feasibility import FeasibilityOutcome
from partypilot.domain.party_request import PartyRequest


class StaticPlanner:
    def __init__(self, result: EvidenceGroundedPlanningResult) -> None:
        self.result = result

    def plan(self, request: PartyRequest) -> EvidenceGroundedPlanningResult:
        return self.result


def _request() -> PartyRequest:
    return PartyRequest(
        location="Brooklyn, NY",
        event_date=date(2026, 9, 20),
        guest_count=10,
        total_budget=Decimal("1600"),
    )


def _scenario(*, expected: FeasibilityOutcome = FeasibilityOutcome.FEASIBLE) -> EvaluationScenario:
    return EvaluationScenario(
        scenario_id="s1",
        request=_request(),
        expected_feasibility=expected,
        retrieval_ground_truth=(
            RetrievalGroundTruthLabel(
                expected_document_ids=("doc-1",),
                resource_id="venue-1",
                expected_version="1.0",
                expected_status=EvidenceDocumentStatus.CURRENT,
                policy_type=EvidenceDocumentType.VENUE_POLICY,
            ),
        ),
        scenario_category=ScenarioCategory.FEASIBLE,
        complexity=ComplexityMetadata(),
        dataset_split=DatasetSplit.DEVELOPMENT,
    )


def _corpus() -> tuple[EvidenceDocument, ...]:
    return (
        EvidenceDocument(
            metadata=EvidenceDocumentMetadata(
                document_id="doc-1",
                resource_id="venue-1",
                document_type=EvidenceDocumentType.VENUE_POLICY,
                version="1.0",
                effective_date=date(2026, 1, 1),
                status=EvidenceDocumentStatus.CURRENT,
            ),
            text="Current venue policy.",
        ),
    )


def test_runner_calculates_outcome_and_grounded_decision_accuracy() -> None:
    planner = StaticPlanner(
        EvidenceGroundedPlanningResult(outcome=FeasibilityOutcome.NO_FEASIBLE_PLAN)
    )
    ticks = iter([1.0, 1.002])
    runner = V02EvaluationRunner(planner, corpus=_corpus(), clock=lambda: next(ticks))

    metrics, scenarios = runner.run((_scenario(expected=FeasibilityOutcome.NO_FEASIBLE_PLAN),))

    assert metrics.feasibility_accuracy == 1.0
    assert metrics.grounded_decision_accuracy == 1.0
    assert metrics.no_feasible_plan_accuracy == 1.0
    assert metrics.hard_constraint_validity == 1.0
    assert scenarios[0].latency_ms == pytest.approx(2.0)


def test_source_attribution_accuracy_is_zero_when_expected_source_is_not_cited() -> None:
    planner = StaticPlanner(EvidenceGroundedPlanningResult(outcome=FeasibilityOutcome.FEASIBLE))
    ticks = iter([1.0, 1.0])
    metrics, _ = V02EvaluationRunner(planner, corpus=_corpus(), clock=lambda: next(ticks)).run(
        (_scenario(),)
    )

    assert metrics.source_attribution_accuracy == 0.0


def test_hard_constraint_validity_reflects_candidate_validation() -> None:
    candidate = EvidenceGroundedPlanCandidate(
        resources=(),
        total_cost=Decimal("0"),
        validation=ConstraintEngineResult(
            feasible=False,
            satisfied_constraint_ids=(),
            violations=(),
            unresolved_constraint_ids=("missing",),
        ),
    )
    planner = StaticPlanner(
        EvidenceGroundedPlanningResult(
            outcome=FeasibilityOutcome.FEASIBLE,
            candidates=(candidate,),
        )
    )
    ticks = iter([1.0, 1.0])
    metrics, _ = V02EvaluationRunner(planner, corpus=_corpus(), clock=lambda: next(ticks)).run(
        (_scenario(),)
    )

    assert metrics.hard_constraint_validity == 0.0


def test_loads_existing_baseline_and_retrieval_reports() -> None:
    baseline = load_v01_baseline_snapshot(Path("evals/results/v0_1/deterministic_baseline.json"))
    retrieval = load_retrieval_snapshots(Path("evals/results/v0_2/retrieval_benchmark.json"))

    assert baseline.name == "v0.1 deterministic baseline"
    assert baseline.feasibility_accuracy == 0.875
    assert {item.variant for item in retrieval} == {"bm25", "semantic", "bm25_semantic_rrf"}


def test_reports_are_machine_readable_and_keep_retrieval_separate(tmp_path: Path) -> None:
    planner = StaticPlanner(EvidenceGroundedPlanningResult(outcome=FeasibilityOutcome.FEASIBLE))
    ticks = iter([1.0, 1.0])
    metrics, scenarios = V02EvaluationRunner(
        planner, corpus=_corpus(), clock=lambda: next(ticks)
    ).run((_scenario(),))
    report = V02EvaluationReport(
        evaluation_variant="test-fixture",
        metrics=metrics,
        scenarios=scenarios,
        retrieval_metrics=load_retrieval_snapshots(
            Path("evals/results/v0_2/retrieval_benchmark.json")
        ),
    )

    json_path, md_path = save_v02_evaluation_reports(report, tmp_path)
    markdown = render_v02_evaluation_markdown(report)

    assert json_path.exists()
    assert md_path.exists()
    assert "## Planning and grounding metrics" in markdown
    assert "## Retrieval metrics (separate)" in markdown
    assert "bm25" in markdown
