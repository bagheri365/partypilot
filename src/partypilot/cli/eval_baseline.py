"""Canonical v0.1 baseline evaluation CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import TypeAdapter

from partypilot.adapters import (
    InMemoryResourceStore,
    OllamaAdapter,
    OllamaConfig,
    UrllibHttpTransport,
)
from partypilot.application.baseline_experiment import (
    BaselineExperimentResult,
    run_baseline_experiment,
    save_baseline_experiment_reports,
)
from partypilot.application.baseline_metrics import BaselineFailureLabel
from partypilot.application.deterministic_planner import DeterministicPlanner
from partypilot.application.single_pass_llm_planner import (
    SINGLE_PASS_PROMPT_VERSION,
    SinglePassLLMPlanner,
    SinglePassPlannerProviderError,
)
from partypilot.domain.evaluation import DatasetSplit, EvaluationScenario
from partypilot.domain.experiment import ExperimentConfig, ExperimentResultMetadata

DATASET_PATH = Path(__file__).resolve().parents[3] / "data" / "evaluation" / "core_scenarios.json"
DEFAULT_OUTPUT_ROOT = Path("evals") / "results" / "v0_1"
ARCHITECTURE_VARIANT = "deterministic_plus_single_pass_llm"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PartyPilot v0.1 baseline comparison on a benchmark split."
    )
    parser.add_argument(
        "--split",
        choices=[split.value for split in DatasetSplit],
        default=DatasetSplit.DEVELOPMENT.value,
        help="Benchmark split to evaluate (default: development).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON and Markdown artifacts.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override PARTYPILOT_OLLAMA_BASE_URL for the live Ollama provider.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override PARTYPILOT_OLLAMA_MODEL for the live Ollama provider.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Override PARTYPILOT_OLLAMA_TIMEOUT_SECONDS for the live Ollama provider.",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Override PARTYPILOT_OLLAMA_NUM_CTX for the live Ollama provider.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Override PARTYPILOT_OLLAMA_MAX_RETRIES for the live Ollama provider.",
    )
    return parser


def load_scenarios(
    split: DatasetSplit, dataset_path: Path = DATASET_PATH
) -> tuple[EvaluationScenario, ...]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    scenarios = TypeAdapter(tuple[EvaluationScenario, ...]).validate_python(payload)
    return tuple(scenario for scenario in scenarios if scenario.dataset_split is split)


def build_live_single_pass_planner(
    *,
    model: str | None,
    base_url: str | None,
    timeout_seconds: float | None,
    max_retries: int | None,
    num_ctx: int | None,
) -> tuple[SinglePassLLMPlanner, str | None]:
    config = _ollama_config(
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        num_ctx=num_ctx,
    )
    return SinglePassLLMPlanner(OllamaAdapter(config, UrllibHttpTransport())), config.model


def _ollama_config(
    *,
    model: str | None,
    base_url: str | None,
    timeout_seconds: float | None,
    max_retries: int | None,
    num_ctx: int | None = None,
) -> OllamaConfig:
    try:
        env_config = OllamaConfig.from_env()
    except ValueError as exc:
        if model is None:
            raise ValueError(
                "PARTYPILOT_OLLAMA_MODEL is required for the live baseline. "
                "Set it in the environment or pass --model."
            ) from exc
        env_config = OllamaConfig(
            model=model,
            base_url=base_url or "http://localhost:11434",
            timeout_seconds=timeout_seconds or 30.0,
            num_ctx=num_ctx if num_ctx is not None else 8192,
            max_retries=max_retries if max_retries is not None else 2,
        )
    else:
        env_config = env_config.model_copy(
            update={
                **({"model": model} if model is not None else {}),
                **({"base_url": base_url} if base_url is not None else {}),
                **({"timeout_seconds": timeout_seconds} if timeout_seconds is not None else {}),
                **({"num_ctx": num_ctx} if num_ctx is not None else {}),
                **({"max_retries": max_retries} if max_retries is not None else {}),
            }
        )
    return env_config


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


def _experiment_metadata(
    *,
    split: DatasetSplit,
    model_name: str | None,
    timestamp: datetime,
) -> ExperimentResultMetadata:
    commit_sha, working_tree_dirty, git_metadata_error = _git_metadata()
    config = ExperimentConfig(
        experiment_id=f"baseline-v0.1-{split.value}-{timestamp.strftime('%Y%m%dT%H%M%SZ')}",
        code_commit_sha=commit_sha,
        working_tree_dirty=working_tree_dirty,
        git_metadata_error=git_metadata_error,
        dataset_version="v0.1",
        architecture_variant=ARCHITECTURE_VARIANT,
        model_provider="ollama",
        model_name=model_name,
        prompt_version=SINGLE_PASS_PROMPT_VERSION,
        timestamp=timestamp,
    )
    return ExperimentResultMetadata(config=config)


def _default_output_dir(split: DatasetSplit, timestamp: datetime) -> Path:
    return DEFAULT_OUTPUT_ROOT / split.value / timestamp.strftime("%Y%m%dT%H%M%SZ")


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _format_token_count(value: int | None) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _count_failure_labels(result: BaselineExperimentResult) -> Counter[BaselineFailureLabel]:
    labels: Counter[BaselineFailureLabel] = Counter()
    for scenario in result.single_pass_scenarios:
        labels.update(scenario.failure_labels)
    return labels


def _print_summary(result: BaselineExperimentResult, json_path: Path, markdown_path: Path) -> None:
    comparison = result.comparison
    deterministic = comparison.deterministic
    single_pass = comparison.single_pass_llm
    failure_counts = _count_failure_labels(result)
    print("# PartyPilot v0.1 Baseline Comparison")
    print(f"Split: {result.dataset_split.value}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    print()
    print(
        "| Architecture | Scenarios | Feasibility Accuracy | Structured Valid | "
        "Hard Valid | Mean Latency (ms) | Median Latency (ms) | Input Tokens | "
        "Output Tokens | Total Tokens |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    print(
        f"| Deterministic baseline | {deterministic.scenario_count} | "
        f"{_format_metric(deterministic.feasibility_accuracy)} | "
        f"{_format_metric(deterministic.structured_output_validity)} | "
        f"{_format_metric(deterministic.hard_constraint_validity)} | "
        f"{_format_metric(deterministic.mean_latency_ms)} | "
        f"{_format_metric(deterministic.median_latency_ms)} | n/a | n/a | n/a |"
    )
    print(
        f"| Single-pass LLM baseline | {single_pass.scenario_count} | "
        f"{_format_metric(single_pass.feasibility_accuracy)} | "
        f"{_format_metric(single_pass.structured_output_validity)} | "
        f"{_format_metric(single_pass.hard_constraint_validity)} | "
        f"{_format_metric(single_pass.mean_latency_ms)} | "
        f"{_format_metric(single_pass.median_latency_ms)} | "
        f"{_format_token_count(single_pass.total_input_tokens)} | "
        f"{_format_token_count(single_pass.total_output_tokens)} | "
        f"{_format_token_count(single_pass.total_tokens)} |"
    )
    print()
    if result.metadata.config.code_commit_sha is None:
        print(f"Git metadata error: {result.metadata.config.git_metadata_error or 'n/a'}")
        print("Commit SHA: unavailable")
        print("Working tree dirty: unknown")
    else:
        print(f"Commit SHA: {result.metadata.config.code_commit_sha}")
        print(f"Working tree dirty: {result.metadata.config.working_tree_dirty}")
    print(f"Prompt version: {result.metadata.config.prompt_version or 'n/a'}")
    print()
    print("Failure summary:")
    print(f"- Valid: {failure_counts[BaselineFailureLabel.VALID]}")
    print(f"- Malformed JSON: {failure_counts[BaselineFailureLabel.MALFORMED_JSON]}")
    print(f"- Schema invalid: {failure_counts[BaselineFailureLabel.SCHEMA_INVALID]}")
    print(
        f"- Hard-constraint failures: "
        f"{failure_counts[BaselineFailureLabel.HARD_CONSTRAINT_VIOLATION]}"
    )
    print(
        f"- Feasibility errors: "
        f"{failure_counts[BaselineFailureLabel.FEASIBILITY_MISCLASSIFICATION]}"
    )
    print(f"- Arithmetic errors: {failure_counts[BaselineFailureLabel.ARITHMETIC_ERROR]}")
    print(f"- Hallucinated resources: {failure_counts[BaselineFailureLabel.HALLUCINATED_RESOURCE]}")
    print(
        f"- Unsupported assumptions: {failure_counts[BaselineFailureLabel.UNSUPPORTED_ASSUMPTION]}"
    )
    print(f"- Provider failures: {failure_counts[BaselineFailureLabel.PROVIDER_FAILURE]}")
    print()
    print("Unsupported-claim: N/A (not evaluated in v0.1)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    split = DatasetSplit(args.split)
    timestamp = datetime.now(UTC)

    try:
        scenarios = load_scenarios(split)
        deterministic = DeterministicPlanner(InMemoryResourceStore())
        single_pass, model_name = build_live_single_pass_planner(
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            num_ctx=args.num_ctx,
        )
        metadata = _experiment_metadata(
            split=split,
            model_name=model_name,
            timestamp=timestamp,
        )
        result = run_baseline_experiment(
            scenarios,
            deterministic,
            single_pass,
            metadata=metadata,
            dataset_split=split,
        )
        output_dir = args.output_dir or _default_output_dir(split, timestamp)
        json_path, markdown_path = save_baseline_experiment_reports(result, output_dir)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except SinglePassPlannerProviderError as exc:
        print(
            "ERROR: live Ollama baseline failed. "
            "Ensure Ollama is running, PARTYPILOT_OLLAMA_MODEL is set, and the "
            "model name is valid.\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 1

    _print_summary(result, json_path, markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
