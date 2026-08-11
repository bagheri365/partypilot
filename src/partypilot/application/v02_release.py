"""Canonical PartyPilot v0.2 release helpers."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

from pydantic import TypeAdapter

from partypilot.application.v02_evaluation import (
    V02EvaluationMetrics,
    V02EvaluationReport,
    V02ScenarioEvaluation,
    load_retrieval_snapshots,
    load_v01_baseline_snapshot,
)
from partypilot.domain.evaluation import DatasetSplit, EvaluationScenario
from partypilot.domain.evidence_corpus import EvidenceDocument
from partypilot.domain.experiment import ExperimentConfig, ExperimentResultMetadata

DATASET_PATH = Path(__file__).resolve().parents[3] / "data" / "evaluation" / "core_scenarios.json"
DEFAULT_OUTPUT_ROOT = Path("evals") / "results" / "v0_2"
ARCHITECTURE_VARIANT = "bm25_plus_live_ollama_constraint_extractor"


def load_documents(path: Path) -> tuple[EvidenceDocument, ...]:
    return tuple(
        EvidenceDocument.model_validate(item)
        for item in json.loads(path.read_text(encoding="utf-8"))
    )


def load_scenarios(
    split: DatasetSplit, dataset_path: Path = DATASET_PATH
) -> tuple[EvaluationScenario, ...]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    scenarios = TypeAdapter(tuple[EvaluationScenario, ...]).validate_python(payload)
    return tuple(scenario for scenario in scenarios if scenario.dataset_split is split)


def _git_metadata() -> tuple[str | None, bool | None, str | None]:
    project_root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        working_tree_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except Exception as exc:
        return None, None, f"Git metadata unavailable: {type(exc).__name__}: {exc}"
    return commit or None, working_tree_dirty, None


def build_release_metadata(
    *,
    split: DatasetSplit,
    model_name: str | None,
    timestamp: datetime,
) -> ExperimentResultMetadata:
    commit_sha, working_tree_dirty, git_metadata_error = _git_metadata()
    config = ExperimentConfig(
        experiment_id=f"v0.2-evidence-grounded-{split.value}-{timestamp.strftime('%Y%m%dT%H%M%SZ')}",
        code_commit_sha=commit_sha,
        working_tree_dirty=working_tree_dirty,
        git_metadata_error=git_metadata_error,
        dataset_version="v0.2",
        architecture_variant=ARCHITECTURE_VARIANT,
        model_provider="ollama" if model_name is not None else None,
        model_name=model_name,
        retrieval_configuration={"retriever": "bm25", "top_k": 5},
        timestamp=timestamp,
    )
    return ExperimentResultMetadata(config=config, evaluation_split=split.value)


def build_v02_evaluation_report(
    *,
    root: Path,
    metrics: V02EvaluationMetrics,
    scenario_results: tuple[V02ScenarioEvaluation, ...],
    metadata: ExperimentResultMetadata | None = None,
) -> V02EvaluationReport:
    return V02EvaluationReport(
        evaluation_variant="bm25 + live_ollama_constraint_extractor",
        metrics=metrics,
        scenarios=scenario_results,
        metadata=metadata,
        v01_baselines=(
            load_v01_baseline_snapshot(root / "evals/results/v0_1/deterministic_baseline.json"),
        ),
        retrieval_metrics=load_retrieval_snapshots(
            root / "evals/results/v0_2/retrieval_benchmark.json"
        ),
        notes=(
            "The retained v0.2 runtime uses plain BM25, a live Ollama-backed constraint "
            "extractor, deterministic request-specific interpretation, and explicit citation "
            "validation.",
            "Token and model-cost metrics are not collected by this evaluation report and are "
            "not fabricated.",
            "The conditional query-rewriting comparison is preserved separately as an experiment "
            "artifact and is not part of the retained runtime.",
            "The controlled live comparison showed identical downstream metrics for conditional "
            "rewriting, so the simpler BM25 runtime was retained.",
        ),
    )


def default_output_dir(split: DatasetSplit, timestamp: datetime) -> Path:
    return DEFAULT_OUTPUT_ROOT / split.value / timestamp.strftime("%Y%m%dT%H%M%SZ")
