"""Compatibility entry point for the canonical live PartyPilot v0.2 evaluation."""

from __future__ import annotations

from partypilot.application.v02_evaluation import (
    V02EvaluationMetrics,
    V02EvaluationReport,
    V02EvaluationRunner,
    V02ScenarioEvaluation,
    save_v02_evaluation_reports,
)
from partypilot.application.v02_release import (
    ARCHITECTURE_VARIANT,
    build_release_metadata,
    build_v02_evaluation_report,
    default_output_dir,
    load_documents,
    load_scenarios,
)
from partypilot.cli.eval_v02 import build_live_constraint_extractor, build_v02_planner, main

__all__ = [
    "ARCHITECTURE_VARIANT",
    "V02EvaluationMetrics",
    "V02EvaluationReport",
    "V02EvaluationRunner",
    "V02ScenarioEvaluation",
    "build_live_constraint_extractor",
    "build_release_metadata",
    "build_v02_evaluation_report",
    "build_v02_planner",
    "default_output_dir",
    "load_documents",
    "load_scenarios",
    "main",
    "save_v02_evaluation_reports",
]


if __name__ == "__main__":
    raise SystemExit(main())
