from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from partypilot.application.v02_evaluation import (
    V02EvaluationMetrics,
    V02EvaluationReport,
    V02ScenarioEvaluation,
)
from partypilot.cli import eval_v02
from partypilot.domain.evaluation import DatasetSplit
from partypilot.domain.experiment import ExperimentConfig, ExperimentResultMetadata
from partypilot.domain.feasibility import FeasibilityOutcome


def test_eval_v02_defaults_to_development_and_writes_timestamped_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixed_timestamp = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
    fixed_output_dir = tmp_path / "development" / "20260811T120000Z"
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, planner: object, *, corpus: object) -> None:
            captured["planner"] = planner
            captured["corpus"] = corpus

        def run(
            self, scenarios: object
        ) -> tuple[V02EvaluationMetrics, tuple[V02ScenarioEvaluation, ...]]:
            captured["scenarios"] = scenarios
            metrics = V02EvaluationMetrics(
                scenario_count=1,
                feasibility_accuracy=1.0,
                hard_constraint_validity=1.0,
                grounded_decision_accuracy=1.0,
                source_attribution_accuracy=1.0,
                derived_constraint_accuracy=1.0,
                unsupported_claim_rate=0.0,
                wrong_source_version_rate=0.0,
                no_feasible_plan_accuracy=1.0,
                mean_latency_ms=12.5,
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                estimated_cost_usd=None,
            )
            scenario = V02ScenarioEvaluation(
                scenario_id="scenario-1",
                expected_outcome=FeasibilityOutcome.FEASIBLE,
                predicted_outcome=FeasibilityOutcome.FEASIBLE,
                outcome_correct=True,
                hard_constraints_valid=True,
                grounded_decision_correct=None,
                latency_ms=12.5,
            )
            return metrics, (scenario,)

    def fake_config(**kwargs: object) -> SimpleNamespace:
        captured["config_kwargs"] = kwargs
        return SimpleNamespace(model="fake-model")

    def fake_documents(path: Path) -> tuple[object, ...]:
        captured["documents_path"] = path
        return ("doc-1",)

    def fake_scenarios(split: DatasetSplit, dataset_path: Path) -> tuple[object, ...]:
        captured["split"] = split
        captured["dataset_path"] = dataset_path
        return ("scenario-1",)

    def fake_planner(*, corpus: tuple[object, ...], config: object) -> object:
        captured["planner_corpus"] = corpus
        captured["planner_config"] = config
        return object()

    def fake_metadata(
        *, split: DatasetSplit, model_name: str | None, timestamp: datetime
    ) -> ExperimentResultMetadata:
        captured["metadata_args"] = (split, model_name, timestamp)
        return ExperimentResultMetadata(
            config=ExperimentConfig(
                experiment_id="v0.2-evidence-grounded-development-20260811T120000Z",
                code_commit_sha="abc123",
                working_tree_dirty=False,
                dataset_version="v0.2",
                architecture_variant="bm25_plus_live_ollama_constraint_extractor",
                model_provider="ollama",
                model_name=model_name,
                timestamp=timestamp,
            ),
            evaluation_split=split.value,
        )

    def fake_report(
        *,
        root: Path,
        metrics: V02EvaluationMetrics,
        scenario_results: tuple[V02ScenarioEvaluation, ...],
        metadata: ExperimentResultMetadata | None = None,
    ) -> V02EvaluationReport:
        captured["report_args"] = (root, metrics, scenario_results, metadata)
        return V02EvaluationReport(
            evaluation_variant="bm25 + live_ollama_constraint_extractor",
            metrics=metrics,
            scenarios=scenario_results,
            metadata=metadata,
        )

    def fake_save(report: V02EvaluationReport, output_directory: Path) -> tuple[Path, Path]:
        captured["saved_report"] = report
        captured["output_directory"] = output_directory
        json_path = output_directory / "v0_2_evidence_grounded_evaluation.json"
        md_path = output_directory / "v0_2_evidence_grounded_evaluation.md"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text("{}", encoding="utf-8")
        md_path.write_text("# report", encoding="utf-8")
        return json_path, md_path

    monkeypatch.setattr(eval_v02, "_ollama_config", fake_config)
    monkeypatch.setattr(eval_v02, "load_documents", fake_documents)
    monkeypatch.setattr(eval_v02, "load_scenarios", fake_scenarios)
    monkeypatch.setattr(eval_v02, "build_v02_planner", fake_planner)
    monkeypatch.setattr(eval_v02, "V02EvaluationRunner", FakeRunner)
    monkeypatch.setattr(eval_v02, "build_release_metadata", fake_metadata)
    monkeypatch.setattr(eval_v02, "build_v02_evaluation_report", fake_report)
    monkeypatch.setattr(eval_v02, "save_v02_evaluation_reports", fake_save)
    monkeypatch.setattr(eval_v02, "default_output_dir", lambda split, timestamp: fixed_output_dir)
    monkeypatch.setattr(
        eval_v02,
        "datetime",
        SimpleNamespace(now=lambda tz: fixed_timestamp),
    )

    exit_code = eval_v02.main([])

    captured_output = capsys.readouterr()
    assert exit_code == 0
    assert "Split: development" in captured_output.out
    assert "Model: fake-model" in captured_output.out
    assert "20260811T120000Z" in captured_output.out
    assert captured["split"] is DatasetSplit.DEVELOPMENT
    assert captured["output_directory"] == fixed_output_dir
    report = captured["saved_report"]
    assert isinstance(report, V02EvaluationReport)
    assert report.metadata is not None
    assert report.metadata.evaluation_split == "development"
    assert report.metadata.config.model_name == "fake-model"
