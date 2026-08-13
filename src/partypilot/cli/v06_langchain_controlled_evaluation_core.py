"""Three-way controlled LangChain evaluation for PartyPilot v0.6d."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean, median
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from partypilot.adapters.ollama import OllamaAdapter, OllamaConfig, UrllibHttpTransport
from partypilot.application import v04_multi_agent as v04
from partypilot.application.multi_agent_runtime import (
    MultiAgentLiveReport,
    load_v05_multi_agent_benchmark,
    render_v05_multi_agent_markdown,
    run_v05_multi_agent_experiment,
)
from partypilot.composition.multi_agent_runtime import (
    SpecialistAdapterKind,
    build_live_multi_agent_runtime,
)
from partypilot.domain import (
    CandidateEvaluationResult,
    CapabilityBoundaryScenario,
    ExperimentConfig,
    ExperimentResultMetadata,
    MultiAgentPlanningRuntimeResult,
    SpecialistFailureKind,
)

V06_BENCHMARK_NAME = "PartyPilot v0.6d three-way LangChain controlled evaluation"
V06_BENCHMARK_VERSION = "1.0"
V06_EVALUATION_VARIANT = "three_way_langchain_controlled"
V06_VARIANT_ORDER = ("native_ollama", "langchain_chatollama", "langchain_agent")
V06_RUN_ORDER_BLOCKS: tuple[tuple[SpecialistAdapterKind, ...], ...] = (
    (
        SpecialistAdapterKind.NATIVE,
        SpecialistAdapterKind.LANGCHAIN,
        SpecialistAdapterKind.LANGCHAIN_AGENT,
    ),
    (
        SpecialistAdapterKind.LANGCHAIN,
        SpecialistAdapterKind.LANGCHAIN_AGENT,
        SpecialistAdapterKind.NATIVE,
    ),
    (
        SpecialistAdapterKind.LANGCHAIN_AGENT,
        SpecialistAdapterKind.NATIVE,
        SpecialistAdapterKind.LANGCHAIN,
    ),
)


class NumericSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mean: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    values: tuple[float, ...] = ()


class EvaluationEnvironment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    python_version: str
    langchain_version: str | None = None
    langchain_core_version: str | None = None
    langchain_ollama_version: str | None = None
    langgraph_version: str | None = None
    model: str
    adapter_variant: str
    benchmark_version: str
    provider_io_timeout_seconds: float
    ollama_context_budget: int
    agent_execution_bound: int | None = None
    structured_output_strategy: str


class GitSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    git_sha: str | None = None
    working_tree_dirty: bool | None = None
    git_metadata_error: str | None = None


class V06EvaluationProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_start_git_sha: str | None = None
    experiment_start_working_tree_dirty: bool | None = None
    experiment_start_git_metadata_error: str | None = None
    artifact_git_sha: str | None = None
    artifact_working_tree_dirty: bool | None = None
    artifact_git_metadata_error: str | None = None
    canonical_start_guard_enforced: bool = True
    exploratory_mode: bool = False


class V06RunReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    variant: str
    repetition_index: int = Field(ge=1)
    order_block_index: int = Field(ge=1)
    order_position: int = Field(ge=1)
    scenario_count: int = Field(ge=0)
    provenance: V06EvaluationProvenance
    environment: EvaluationEnvironment
    report: MultiAgentLiveReport


class VariantAggregateReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    variant: str
    run_count: int = Field(ge=0)
    scenario_count: int = Field(ge=0)
    final_decision_accuracy: NumericSummary
    evidence_grounded_arbitration_accuracy: NumericSummary
    hard_constraint_validity: NumericSummary
    cross_domain_compatibility_accuracy: NumericSummary
    global_optimum_accuracy: NumericSummary
    human_review_calibration: NumericSummary
    specialist_success_rate: NumericSummary
    mean_successful_specialist_latency_ms: NumericSummary
    median_successful_specialist_latency_ms: NumericSummary
    p95_successful_specialist_latency_ms: float | None = None
    mean_scenario_wall_clock_latency_ms: NumericSummary
    maximum_specialist_latency_ms: float | None = None
    top_level_specialist_invocations: int = Field(ge=0)
    successful_top_level_specialist_invocations: int = Field(ge=0)
    total_specialist_invocations: int = Field(ge=0)
    successful_specialist_invocations: int = Field(ge=0)
    specialist_success_rate_overall: float = Field(ge=0, le=1)
    specialist_timeout_outcomes: int = Field(ge=0)
    specialist_timeout_outcome_rate: float = Field(ge=0, le=1)
    provider_timeout_count: int = Field(ge=0)
    provider_timeout_rate: float = Field(ge=0, le=1)
    provider_connection_failure_count: int = Field(ge=0)
    provider_response_failure_count: int = Field(ge=0)
    structured_output_validation_failure_count: int = Field(ge=0)
    specialist_domain_validation_failure_count: int = Field(ge=0)
    provider_attempt_count: int = Field(ge=0)
    provider_attempt_rate: float = Field(ge=0, le=1)
    retry_count: int = Field(ge=0)
    retry_rate: float = Field(ge=0, le=1)
    terminal_stability_rate: float = Field(ge=0, le=1)
    terminal_stable_scenario_ids: tuple[str, ...] = ()
    terminal_unstable_scenario_ids: tuple[str, ...] = ()
    total_tool_calls: int | None = None
    candidate_specialist_invocations: int | None = None
    candidate_provider_timeout_count: int | None = None
    candidate_provider_timeout_rate: float | None = None
    candidate_retry_count: int | None = None
    candidate_retry_rate: float | None = None
    tool_calls_by_specialist: tuple[tuple[str, int], ...] | None = None
    tool_calls_by_scenario: tuple[tuple[str, int], ...] | None = None
    tool_call_success_count: int | None = None
    tool_call_failure_count: int | None = None
    no_tool_specialist_completions: int | None = None
    scenarios_with_tool_use: tuple[str, ...] | None = None
    specialist_domains_with_tool_use: tuple[str, ...] | None = None
    execution_limit_hits: int | None = None
    total_model_invocations: int | None = None
    disposition: str | None = None


class V06ControlledEvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    system_name: str = "PartyPilot v0.6d three-way LangChain controlled evaluation"
    benchmark_name: str = V06_BENCHMARK_NAME
    benchmark_version: str = V06_BENCHMARK_VERSION
    evaluation_variant: str = V06_EVALUATION_VARIANT
    run_order_blocks: tuple[tuple[str, ...], ...]
    scenario_count: int = Field(ge=0)
    runs: tuple[V06RunReport, ...]
    variant_summaries: tuple[VariantAggregateReport, ...]
    provenance: V06EvaluationProvenance
    metadata: ExperimentResultMetadata | None = None
    notes: tuple[str, ...] = ()


def load_v06_controlled_scenarios(
    scenario_ids: Sequence[str] | None = None,
) -> tuple[CapabilityBoundaryScenario, ...]:
    scenarios = load_v05_multi_agent_benchmark()
    if not scenario_ids:
        return scenarios
    scenarios_by_id = {scenario.scenario.scenario_id: scenario for scenario in scenarios}
    filtered = []
    for scenario_id in scenario_ids:
        scenario = scenarios_by_id.get(scenario_id)
        if scenario is None:
            raise ValueError(f"unknown evaluation scenario ID: {scenario_id}")
        filtered.append(scenario)
    return tuple(filtered)


def run_v06_controlled_evaluation(
    scenarios: Sequence[CapabilityBoundaryScenario] | None = None,
    *,
    model: str,
    base_url: str | None,
    timeout_seconds: float,
    num_ctx: int,
    max_retries: int,
    allow_dirty_tree: bool = False,
    timestamp: datetime | None = None,
) -> V06ControlledEvaluationReport:
    benchmark = tuple(scenarios) if scenarios is not None else load_v05_multi_agent_benchmark()
    timestamp = timestamp or datetime.now(UTC)
    experiment_start_snapshot = _git_snapshot()
    if not allow_dirty_tree and experiment_start_snapshot.working_tree_dirty:
        raise ValueError(
            "canonical v0.6d evaluation requires a clean working tree; "
            "pass --allow-dirty-tree for exploratory runs"
        )
    config = OllamaConfig(
        base_url=base_url or "http://localhost:11434",
        model=model,
        timeout_seconds=timeout_seconds,
        num_ctx=num_ctx,
        max_retries=max_retries,
    )

    runs: list[V06RunReport] = []
    for block_index, order in enumerate(V06_RUN_ORDER_BLOCKS, start=1):
        for position, adapter_kind in enumerate(order, start=1):
            runtime, strategy = _build_runtime(
                adapter_kind=adapter_kind,
                config=config,
                timeout_seconds=timeout_seconds,
            )
            report = run_v05_multi_agent_experiment(
                benchmark,
                runtime=runtime,
                timestamp=timestamp,
            )
            runs.append(
                V06RunReport(
                    run_id=f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{block_index}-{position}",
                    variant=_variant_name(adapter_kind),
                    repetition_index=block_index,
                    order_block_index=block_index,
                    order_position=position,
                    scenario_count=len(benchmark),
                    provenance=V06EvaluationProvenance(
                        experiment_start_git_sha=experiment_start_snapshot.git_sha,
                        experiment_start_working_tree_dirty=(
                            experiment_start_snapshot.working_tree_dirty
                        ),
                        experiment_start_git_metadata_error=(
                            experiment_start_snapshot.git_metadata_error
                        ),
                        canonical_start_guard_enforced=not allow_dirty_tree,
                        exploratory_mode=allow_dirty_tree,
                    ),
                    environment=_environment_from_config(
                        config=config,
                        adapter_variant=_variant_name(adapter_kind),
                        structured_output_strategy=strategy,
                    ),
                    report=_retag_report(report, adapter_kind),
                )
            )

    variant_summaries = _aggregate_variant_summaries(runs)
    variant_summaries = _apply_dispositions(tuple(variant_summaries))
    metadata = _build_metadata(timestamp=timestamp, git_snapshot=experiment_start_snapshot)
    return V06ControlledEvaluationReport(
        run_order_blocks=tuple(
            tuple(_variant_name(kind) for kind in order) for order in V06_RUN_ORDER_BLOCKS
        ),
        scenario_count=len(benchmark),
        runs=tuple(runs),
        variant_summaries=variant_summaries,
        provenance=V06EvaluationProvenance(
            experiment_start_git_sha=experiment_start_snapshot.git_sha,
            experiment_start_working_tree_dirty=experiment_start_snapshot.working_tree_dirty,
            experiment_start_git_metadata_error=experiment_start_snapshot.git_metadata_error,
            canonical_start_guard_enforced=not allow_dirty_tree,
            exploratory_mode=allow_dirty_tree,
        ),
        metadata=metadata,
        notes=(
            (
                "The benchmark scenarios, coordinator, timeout, and context budget are "
                "frozen across all runs."
            ),
            "Tool use is an experimental result and is not forced by the harness.",
        ),
    )


def save_v06_controlled_evaluation_reports(
    report: V06ControlledEvaluationReport,
    output_dir: Path,
    *,
    artifact_snapshot: GitSnapshot | None = None,
) -> tuple[Path, Path]:
    artifact_snapshot = artifact_snapshot or _git_snapshot()
    report = report.model_copy(
        update={
            "provenance": report.provenance.model_copy(
                update={
                    "artifact_git_sha": artifact_snapshot.git_sha,
                    "artifact_working_tree_dirty": artifact_snapshot.working_tree_dirty,
                    "artifact_git_metadata_error": artifact_snapshot.git_metadata_error,
                }
            )
        }
    )
    metadata = report.metadata
    assert metadata is not None
    timestamp_dir = metadata.config.timestamp.strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_dir / timestamp_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "v0_6_langchain_controlled_evaluation.json"
    markdown_path = output_dir / "v0_6_langchain_controlled_evaluation.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    markdown_path.write_text(render_v06_controlled_evaluation_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def save_v06_controlled_run_report(
    report: V06RunReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "run.json"
    markdown_path = output_dir / "run.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    markdown_path.write_text(render_v06_controlled_run_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def save_v06_controlled_run_reports(
    report: V06ControlledEvaluationReport,
    output_dir: Path,
) -> tuple[Path, Path, tuple[tuple[Path, Path], ...]]:
    artifact_snapshot = _git_snapshot()
    report = report.model_copy(
        update={
            "provenance": report.provenance.model_copy(
                update={
                    "artifact_git_sha": artifact_snapshot.git_sha,
                    "artifact_working_tree_dirty": artifact_snapshot.working_tree_dirty,
                    "artifact_git_metadata_error": artifact_snapshot.git_metadata_error,
                }
            ),
            "runs": tuple(
                run.model_copy(
                    update={
                        "provenance": run.provenance.model_copy(
                            update={
                                "artifact_git_sha": artifact_snapshot.git_sha,
                                "artifact_working_tree_dirty": artifact_snapshot.working_tree_dirty,
                                "artifact_git_metadata_error": artifact_snapshot.git_metadata_error,
                            }
                        )
                    }
                )
                for run in report.runs
            ),
        }
    )
    aggregate_json_path, aggregate_markdown_path = save_v06_controlled_evaluation_reports(
        report,
        output_dir,
        artifact_snapshot=artifact_snapshot,
    )
    metadata = report.metadata
    assert metadata is not None
    timestamp_dir = metadata.config.timestamp.strftime("%Y%m%dT%H%M%SZ")
    run_paths = tuple(
        save_v06_controlled_run_report(
            run,
            output_dir / run.variant / timestamp_dir / run.run_id,
        )
        for run in report.runs
    )
    return aggregate_json_path, aggregate_markdown_path, run_paths


def default_output_dir(timestamp: datetime) -> Path:
    return Path("evals") / "results" / "v0_6" / "langchain"


def render_v06_controlled_run_markdown(report: V06RunReport) -> str:
    lines = [
        "# PartyPilot v0.6d Controlled LangChain Run",
        "",
        f"- Run ID: `{report.run_id}`",
        f"- Variant: `{report.variant}`",
        f"- Repetition: `{report.repetition_index}`",
        f"- Order block: `{report.order_block_index}`",
        f"- Order position: `{report.order_position}`",
        f"- Scenario count: `{report.scenario_count}`",
        f"- Model: `{report.environment.model}`",
        f"- Provider I/O timeout: `{report.environment.provider_io_timeout_seconds:.1f}s`",
        f"- Ollama context budget: `{report.environment.ollama_context_budget}`",
        f"- Structured-output strategy: `{report.environment.structured_output_strategy}`",
        "",
    ]
    lines.extend(_provenance_markdown_lines(report.provenance))
    if report.environment.langchain_version is not None:
        lines.extend(
            [
                "## Environment",
                "",
                f"- Python: `{report.environment.python_version}`",
                f"- LangChain: `{report.environment.langchain_version}`",
                f"- langchain-core: `{report.environment.langchain_core_version or 'n/a'}`",
                f"- langchain-ollama: `{report.environment.langchain_ollama_version or 'n/a'}`",
                f"- LangGraph: `{report.environment.langgraph_version or 'n/a'}`",
                "",
            ]
        )
    lines.append(render_v05_multi_agent_markdown(report.report))
    return "\n".join(lines)


def render_v06_controlled_evaluation_markdown(report: V06ControlledEvaluationReport) -> str:
    lines = [
        "# PartyPilot v0.6d Three-Way LangChain Controlled Evaluation",
        "",
        f"- Benchmark: `{report.benchmark_name}`",
        f"- Benchmark version: `{report.benchmark_version}`",
        f"- Scenario count: `{report.scenario_count}`",
        f"- Run order blocks: `{report.run_order_blocks}`",
        "",
    ]
    lines.extend(_provenance_markdown_lines(report.provenance))
    if report.metadata is not None:
        config = report.metadata.config
        python_version = report.runs[0].environment.python_version if report.runs else "n/a"
        lines.extend(
            [
                "## Reproducibility",
                "",
                f"- Git SHA: `{config.code_commit_sha or 'n/a'}`",
                f"- Working tree dirty: `{config.working_tree_dirty}`",
                f"- Timestamp: `{config.timestamp.isoformat()}`",
                f"- Python: `{python_version}`",
                f"- Model: `{config.model_name or 'n/a'}`",
                "",
            ]
        )
    lines.extend(["## Variant Summaries", ""])
    for summary in report.variant_summaries:
        final_decision = _summary_text(summary.final_decision_accuracy)
        evidence_grounded = _summary_text(summary.evidence_grounded_arbitration_accuracy)
        hard_constraints = _summary_text(summary.hard_constraint_validity)
        cross_domain = _summary_text(summary.cross_domain_compatibility_accuracy)
        global_optimum = _summary_text(summary.global_optimum_accuracy)
        human_review = _summary_text(summary.human_review_calibration)
        specialist_success = _summary_text(summary.specialist_success_rate)
        mean_successful_latency = _summary_text(summary.mean_successful_specialist_latency_ms)
        median_successful_latency = _summary_text(summary.median_successful_specialist_latency_ms)
        p95_successful_latency = (
            str(summary.p95_successful_specialist_latency_ms)
            if summary.p95_successful_specialist_latency_ms is not None
            else "n/a"
        )
        mean_wall_clock_latency = _summary_text(summary.mean_scenario_wall_clock_latency_ms)
        max_specialist_latency = (
            str(summary.maximum_specialist_latency_ms)
            if summary.maximum_specialist_latency_ms is not None
            else "n/a"
        )
        total_specialist_invocations = str(summary.total_specialist_invocations)
        successful_specialist_invocations = str(summary.successful_specialist_invocations)
        top_level_specialist_invocations = str(summary.top_level_specialist_invocations)
        successful_top_level_specialist_invocations = str(
            summary.successful_top_level_specialist_invocations
        )
        provider_timeout_count = str(summary.provider_timeout_count)
        provider_connection_failure_count = str(summary.provider_connection_failure_count)
        provider_response_failure_count = str(summary.provider_response_failure_count)
        structured_output_failures = str(summary.structured_output_validation_failure_count)
        specialist_domain_failures = str(summary.specialist_domain_validation_failure_count)
        candidate_specialist_invocations = str(summary.candidate_specialist_invocations or "n/a")
        candidate_provider_timeout_count = str(
            summary.candidate_provider_timeout_count
            if summary.candidate_provider_timeout_count is not None
            else "n/a"
        )
        candidate_retry_count = str(
            summary.candidate_retry_count if summary.candidate_retry_count is not None else "n/a"
        )
        total_tool_calls = (
            str(summary.total_tool_calls) if summary.total_tool_calls is not None else "n/a"
        )
        no_tool_specialist_completions = (
            str(summary.no_tool_specialist_completions)
            if summary.no_tool_specialist_completions is not None
            else "n/a"
        )
        scenarios_with_tool_use = ", ".join(summary.scenarios_with_tool_use or ()) or "none"
        specialist_domains_with_tool_use = (
            ", ".join(summary.specialist_domains_with_tool_use or ()) or "none"
        )
        lines.extend(
            [
                f"### {summary.variant}",
                "",
                f"- Runs: `{summary.run_count}`",
                f"- Final decision accuracy: `{final_decision}`",
                f"- Evidence-grounded arbitration: `{evidence_grounded}`",
                f"- Hard-constraint validity: `{hard_constraints}`",
                f"- Cross-domain compatibility: `{cross_domain}`",
                f"- Global-optimum accuracy: `{global_optimum}`",
                f"- Human-review calibration: `{human_review}`",
                f"- Specialist success rate: `{specialist_success}`",
                f"- Top-level specialist invocations: `{top_level_specialist_invocations}`",
                (
                    "- Successful top-level specialist invocations: "
                    f"`{successful_top_level_specialist_invocations}`"
                ),
                f"- Mean successful specialist latency (ms): `{mean_successful_latency}`",
                f"- Median successful specialist latency (ms): `{median_successful_latency}`",
                f"- p95 successful specialist latency (ms): `{p95_successful_latency}`",
                f"- Mean scenario wall-clock latency (ms): `{mean_wall_clock_latency}`",
                f"- Maximum specialist latency (ms): `{max_specialist_latency}`",
                f"- Total specialist invocations: `{total_specialist_invocations}`",
                f"- Successful specialist invocations: `{successful_specialist_invocations}`",
                f"- Retry count: `{summary.retry_count}`",
                f"- Retry rate: `{summary.retry_rate:.3f}`",
                f"- Terminal stability rate: `{summary.terminal_stability_rate:.3f}`",
                f"- Specialist timeout outcomes: `{provider_timeout_count}`",
                (
                    f"- Specialist timeout outcome rate: "
                    f"`{summary.specialist_timeout_outcome_rate:.3f}`"
                ),
                f"- Candidate specialist invocations: `{candidate_specialist_invocations}`",
                f"- Candidate provider timeout outcomes: `{candidate_provider_timeout_count}`",
                f"- Candidate retry count: `{candidate_retry_count}`",
                f"- Provider connection failure count: `{provider_connection_failure_count}`",
                f"- Provider response failure count: `{provider_response_failure_count}`",
                f"- Structured-output validation failures: `{structured_output_failures}`",
                f"- Specialist-domain validation failures: `{specialist_domain_failures}`",
                f"- Provider attempts: `{summary.provider_attempt_count}`",
                f"- Provider attempt rate: `{summary.provider_attempt_rate:.3f}`",
                f"- Tool calls: `{total_tool_calls}`",
                f"- No-tool specialist completions: `{no_tool_specialist_completions}`",
                f"- Scenarios with tool use: `{scenarios_with_tool_use}`",
                f"- Specialist domains with tool use: `{specialist_domains_with_tool_use}`",
                f"- Disposition: `{summary.disposition or 'n/a'}`",
                "",
            ]
        )
    lines.extend(["## Runs", ""])
    for run in report.runs:
        lines.extend(
            [
                f"### {run.run_id}",
                "",
                f"- Variant: `{run.variant}`",
                f"- Repetition: `{run.repetition_index}`",
                f"- Order block: `{run.order_block_index}`",
                f"- Order position: `{run.order_position}`",
                "- Run report path: `run.json` / `run.md`",
                "",
            ]
        )
    return "\n".join(lines)


def _provenance_markdown_lines(provenance: V06EvaluationProvenance) -> list[str]:
    experiment_start_working_tree_dirty = provenance.experiment_start_working_tree_dirty
    experiment_start_git_metadata_error = provenance.experiment_start_git_metadata_error
    return [
        "## Provenance",
        "",
        f"- Experiment start Git SHA: `{provenance.experiment_start_git_sha or 'n/a'}`",
        f"- Experiment start working tree dirty: `{experiment_start_working_tree_dirty}`",
        f"- Experiment start git metadata error: `{experiment_start_git_metadata_error or 'n/a'}`",
        f"- Artifact Git SHA: `{provenance.artifact_git_sha or 'n/a'}`",
        f"- Artifact working tree dirty: `{provenance.artifact_working_tree_dirty}`",
        f"- Artifact git metadata error: `{provenance.artifact_git_metadata_error or 'n/a'}`",
        f"- Canonical start guard enforced: `{provenance.canonical_start_guard_enforced}`",
        f"- Exploratory mode: `{provenance.exploratory_mode}`",
        "",
    ]


def _build_runtime(
    *,
    adapter_kind: SpecialistAdapterKind,
    config: OllamaConfig,
    timeout_seconds: float,
) -> tuple[Any, str]:
    if adapter_kind is SpecialistAdapterKind.NATIVE:
        provider = OllamaAdapter(config, UrllibHttpTransport())
        runtime = build_live_multi_agent_runtime(
            provider,
            timeout_seconds=timeout_seconds,
            model_name=config.model,
            adapter_kind=adapter_kind,
            ollama_config=config,
        )
        return runtime, "native_provider_json_schema"
    if adapter_kind is SpecialistAdapterKind.LANGCHAIN:
        runtime = build_live_multi_agent_runtime(
            timeout_seconds=timeout_seconds,
            model_name=config.model,
            adapter_kind=adapter_kind,
            ollama_config=config,
        )
        return runtime, "with_structured_output"
    runtime = build_live_multi_agent_runtime(
        timeout_seconds=timeout_seconds,
        model_name=config.model,
        adapter_kind=adapter_kind,
        ollama_config=config,
    )
    return runtime, "ProviderStrategy(SpecialistDecisionEnvelope)"


def _variant_name(adapter_kind: SpecialistAdapterKind) -> str:
    return {
        SpecialistAdapterKind.NATIVE: "native_ollama",
        SpecialistAdapterKind.LANGCHAIN: "langchain_chatollama",
        SpecialistAdapterKind.LANGCHAIN_AGENT: "langchain_agent",
    }[adapter_kind]


def _retag_report(
    report: MultiAgentLiveReport,
    adapter_kind: SpecialistAdapterKind,
) -> MultiAgentLiveReport:
    variant_name = _variant_name(adapter_kind)
    metadata = report.metadata
    if metadata is not None:
        metadata = metadata.model_copy(
            update={
                "config": metadata.config.model_copy(update={"architecture_variant": variant_name}),
            }
        )
    return report.model_copy(
        update={
            "live_architecture": variant_name,
            "evaluation_variant": f"{variant_name}_vs_deterministic",
            "metadata": metadata,
        }
    )


def _environment_from_config(
    *,
    config: OllamaConfig,
    adapter_variant: str,
    structured_output_strategy: str,
) -> EvaluationEnvironment:
    return EvaluationEnvironment(
        python_version=_python_version(),
        langchain_version=_package_version("langchain"),
        langchain_core_version=_package_version("langchain-core"),
        langchain_ollama_version=_package_version("langchain-ollama"),
        langgraph_version=_package_version("langgraph"),
        model=config.model,
        adapter_variant=adapter_variant,
        benchmark_version=V06_BENCHMARK_VERSION,
        provider_io_timeout_seconds=config.timeout_seconds,
        ollama_context_budget=config.num_ctx,
        agent_execution_bound=8 if adapter_variant == "langchain_agent" else None,
        structured_output_strategy=structured_output_strategy,
    )


def _build_metadata(
    timestamp: datetime,
    *,
    git_snapshot: GitSnapshot,
) -> ExperimentResultMetadata:
    config = ExperimentConfig(
        experiment_id=f"v0.6d-three-way-langchain-controlled-{timestamp.strftime('%Y%m%dT%H%M%SZ')}",
        code_commit_sha=git_snapshot.git_sha,
        working_tree_dirty=git_snapshot.working_tree_dirty,
        git_metadata_error=git_snapshot.git_metadata_error,
        dataset_version=V06_BENCHMARK_VERSION,
        architecture_variant=V06_EVALUATION_VARIANT,
        model_provider="ollama",
        timestamp=timestamp,
    )
    return ExperimentResultMetadata(config=config, evaluation_split="development")


def _aggregate_variant_summaries(
    runs: Sequence[V06RunReport],
) -> tuple[VariantAggregateReport, ...]:
    grouped: dict[str, list[V06RunReport]] = defaultdict(list)
    for run in runs:
        grouped[run.variant].append(run)
    return tuple(
        _summarize_variant(variant=variant, variant_runs=tuple(variant_runs))
        for variant in V06_VARIANT_ORDER
        if (variant_runs := grouped.get(variant)) is not None
    )


def _git_snapshot() -> GitSnapshot:
    commit_sha, working_tree_dirty, git_metadata_error = v04._git_metadata()
    return GitSnapshot(
        git_sha=commit_sha,
        working_tree_dirty=working_tree_dirty,
        git_metadata_error=git_metadata_error,
    )


def _selected_candidate(
    runtime: MultiAgentPlanningRuntimeResult,
) -> CandidateEvaluationResult | None:
    selected_resource_ids = tuple(runtime.final_result.selected_resource_ids)
    for candidate in runtime.candidate_results:
        if tuple(candidate.candidate_resource_ids) == selected_resource_ids:
            return candidate
    return runtime.candidate_results[0] if runtime.candidate_results else None


def _collect_variant_outcomes(
    live_reports: Sequence[MultiAgentLiveReport],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    selected_outcomes: list[Any] = []
    candidate_outcomes: list[Any] = []
    for report in live_reports:
        for scenario in report.scenarios:
            selected_candidate = _selected_candidate(scenario.runtime)
            if selected_candidate is not None:
                selected_outcomes.extend(selected_candidate.specialist_outcomes)
            for candidate in scenario.runtime.candidate_results:
                candidate_outcomes.extend(candidate.specialist_outcomes)
    return tuple(selected_outcomes), tuple(candidate_outcomes)


def _selected_candidate_success_rate(report: MultiAgentLiveReport) -> float:
    selected_outcomes, _ = _collect_variant_outcomes((report,))
    total = len(selected_outcomes)
    if total == 0:
        return 1.0
    return sum(1 for outcome in selected_outcomes if outcome.trace.validation_succeeded) / total


def _summarize_variant(
    *,
    variant: str,
    variant_runs: tuple[V06RunReport, ...],
) -> VariantAggregateReport:
    run_count = len(variant_runs)
    scenario_count = variant_runs[0].scenario_count if variant_runs else 0
    live_reports: list[MultiAgentLiveReport] = [run.report for run in variant_runs]
    final_decision_accuracy = _summary(
        report.metrics.live.final_decision_accuracy for report in live_reports
    )
    evidence_grounded_arbitration_accuracy = _summary(
        report.metrics.live.evidence_grounded_arbitration_accuracy for report in live_reports
    )
    hard_constraint_validity = _summary(
        report.metrics.live.hard_constraint_validity for report in live_reports
    )
    cross_domain_compatibility_accuracy = _summary(
        report.metrics.live.cross_domain_compatibility_accuracy for report in live_reports
    )
    global_optimum_accuracy = _summary(
        report.metrics.live.global_optimum_accuracy for report in live_reports
    )
    human_review_calibration = _summary(
        report.metrics.live.human_review_calibration for report in live_reports
    )
    specialist_success_rate = _summary(
        _selected_candidate_success_rate(report) for report in live_reports
    )
    scenario_wall_clock_latency = _summary(
        report.metrics.runtime.mean_latency_ms for report in live_reports
    )
    selected_outcomes, candidate_outcomes = _collect_variant_outcomes(live_reports)
    selected_total = len(selected_outcomes)
    candidate_total = len(candidate_outcomes)
    selected_success_total = sum(
        1 for outcome in selected_outcomes if outcome.trace.validation_succeeded
    )
    selected_failure_kind_counts: Counter[str] = Counter(
        outcome.failure_kind.value
        for outcome in selected_outcomes
        if outcome.failure_kind is not None
    )
    candidate_failure_kind_counts: Counter[str] = Counter(
        outcome.failure_kind.value
        for outcome in candidate_outcomes
        if outcome.failure_kind is not None
    )
    selected_retry_count = sum(outcome.trace.retry_count for outcome in selected_outcomes)
    candidate_retry_count = sum(outcome.trace.retry_count for outcome in candidate_outcomes)
    selected_retry_rate = selected_retry_count / selected_total if selected_total else 0.0
    candidate_retry_rate = candidate_retry_count / candidate_total if candidate_total else 0.0
    successful_latencies = [
        outcome.trace.latency_ms
        for outcome in selected_outcomes
        if outcome.trace.validation_succeeded
    ]
    all_latencies = [outcome.trace.latency_ms for outcome in selected_outcomes]
    selected_attempt_count = sum(outcome.trace.retry_count + 1 for outcome in selected_outcomes)
    selected_attempt_rate = selected_attempt_count / selected_total if selected_total else 0.0
    stability = _terminal_stability(variant_runs)
    tool_totals = _tool_aggregate(variant_runs) if variant == "langchain_agent" else None
    disposition = _variant_disposition(
        variant=variant,
        summary=final_decision_accuracy,
        evidence_summary=evidence_grounded_arbitration_accuracy,
        success_summary=specialist_success_rate,
        validation_failure_rate=(
            selected_failure_kind_counts.get(
                SpecialistFailureKind.STRUCTURED_OUTPUT_VALIDATION_ERROR.value, 0
            )
            / selected_total
            if selected_total
            else 0.0
        ),
        tool_totals=tool_totals,
    )
    return VariantAggregateReport(
        variant=variant,
        run_count=run_count,
        scenario_count=scenario_count,
        final_decision_accuracy=final_decision_accuracy,
        evidence_grounded_arbitration_accuracy=evidence_grounded_arbitration_accuracy,
        hard_constraint_validity=hard_constraint_validity,
        cross_domain_compatibility_accuracy=cross_domain_compatibility_accuracy,
        global_optimum_accuracy=global_optimum_accuracy,
        human_review_calibration=human_review_calibration,
        specialist_success_rate=specialist_success_rate,
        mean_successful_specialist_latency_ms=_summary(successful_latencies),
        median_successful_specialist_latency_ms=_summary(successful_latencies, use_median=True),
        p95_successful_specialist_latency_ms=(
            _percentile(successful_latencies, 0.95) if successful_latencies else None
        ),
        mean_scenario_wall_clock_latency_ms=scenario_wall_clock_latency,
        maximum_specialist_latency_ms=max(all_latencies) if all_latencies else None,
        top_level_specialist_invocations=selected_total,
        successful_top_level_specialist_invocations=selected_success_total,
        total_specialist_invocations=selected_total,
        successful_specialist_invocations=selected_success_total,
        specialist_success_rate_overall=(
            selected_success_total / selected_total if selected_total else 1.0
        ),
        specialist_timeout_outcomes=selected_failure_kind_counts.get(
            SpecialistFailureKind.PROVIDER_TIMEOUT.value, 0
        ),
        specialist_timeout_outcome_rate=(
            selected_failure_kind_counts.get(SpecialistFailureKind.PROVIDER_TIMEOUT.value, 0)
            / selected_total
            if selected_total
            else 0.0
        ),
        provider_timeout_count=selected_failure_kind_counts.get(
            SpecialistFailureKind.PROVIDER_TIMEOUT.value, 0
        ),
        provider_timeout_rate=(
            selected_failure_kind_counts.get(SpecialistFailureKind.PROVIDER_TIMEOUT.value, 0)
            / selected_total
            if selected_total
            else 0.0
        ),
        provider_connection_failure_count=selected_failure_kind_counts.get(
            SpecialistFailureKind.PROVIDER_CONNECTION_ERROR.value, 0
        ),
        provider_response_failure_count=selected_failure_kind_counts.get(
            SpecialistFailureKind.PROVIDER_RESPONSE_ERROR.value, 0
        ),
        structured_output_validation_failure_count=selected_failure_kind_counts.get(
            SpecialistFailureKind.STRUCTURED_OUTPUT_VALIDATION_ERROR.value, 0
        ),
        specialist_domain_validation_failure_count=selected_failure_kind_counts.get(
            SpecialistFailureKind.SPECIALIST_DOMAIN_VALIDATION_ERROR.value, 0
        ),
        provider_attempt_count=selected_attempt_count,
        provider_attempt_rate=selected_attempt_rate,
        retry_count=selected_retry_count,
        retry_rate=selected_retry_rate,
        candidate_specialist_invocations=candidate_total,
        candidate_provider_timeout_count=candidate_failure_kind_counts.get(
            SpecialistFailureKind.PROVIDER_TIMEOUT.value, 0
        ),
        candidate_provider_timeout_rate=(
            candidate_failure_kind_counts.get(SpecialistFailureKind.PROVIDER_TIMEOUT.value, 0)
            / candidate_total
            if candidate_total
            else 0.0
        ),
        candidate_retry_count=candidate_retry_count,
        candidate_retry_rate=candidate_retry_rate,
        terminal_stability_rate=stability["rate"],
        terminal_stable_scenario_ids=stability["stable"],
        terminal_unstable_scenario_ids=stability["unstable"],
        total_tool_calls=tool_totals["selected_total_tool_calls"]
        if tool_totals is not None
        else None,
        tool_calls_by_specialist=(
            tool_totals["selected_by_specialist"] if tool_totals is not None else None
        ),
        tool_calls_by_scenario=(
            tool_totals["selected_by_scenario"] if tool_totals is not None else None
        ),
        tool_call_success_count=tool_totals["selected_success"]
        if tool_totals is not None
        else None,
        tool_call_failure_count=tool_totals["selected_failure"]
        if tool_totals is not None
        else None,
        no_tool_specialist_completions=(
            tool_totals["selected_no_tool_specialist_completions"]
            if tool_totals is not None
            else None
        ),
        scenarios_with_tool_use=(
            tool_totals["selected_scenarios_with_tool_use"] if tool_totals is not None else None
        ),
        specialist_domains_with_tool_use=(
            tool_totals["selected_specialist_domains_with_tool_use"]
            if tool_totals is not None
            else None
        ),
        execution_limit_hits=(
            tool_totals["selected_execution_limit_hits"] if tool_totals is not None else None
        ),
        total_model_invocations=selected_attempt_count,
        disposition=disposition,
    )


def _terminal_stability(variant_runs: tuple[V06RunReport, ...]) -> dict[str, Any]:
    outcomes_by_scenario: dict[str, set[str]] = defaultdict(set)
    expected_counts: dict[str, int] = {}
    for run in variant_runs:
        for scenario in run.report.scenarios:
            outcomes_by_scenario[scenario.scenario_id].add(
                scenario.live_result.feasibility_outcome.value
            )
            expected_counts[scenario.scenario_id] = expected_counts.get(scenario.scenario_id, 0) + 1
    stable = tuple(
        scenario_id
        for scenario_id, outcomes in sorted(outcomes_by_scenario.items())
        if len(outcomes) == 1 and expected_counts.get(scenario_id, 0) == len(variant_runs)
    )
    unstable = tuple(
        scenario_id
        for scenario_id, outcomes in sorted(outcomes_by_scenario.items())
        if scenario_id not in stable
    )
    total = len(outcomes_by_scenario)
    return {
        "rate": (len(stable) / total if total else 1.0),
        "stable": stable,
        "unstable": unstable,
    }


def _tool_aggregate(variant_runs: tuple[V06RunReport, ...]) -> dict[str, Any]:
    selected_total_tool_calls = 0
    selected_by_specialist: Counter[str] = Counter()
    selected_by_scenario: Counter[str] = Counter()
    selected_success = 0
    selected_failure = 0
    selected_no_tool_specialist_completions = 0
    selected_scenarios_with_tool_use: set[str] = set()
    selected_specialist_domains_with_tool_use: set[str] = set()
    selected_execution_limit_hits = 0

    candidate_total_tool_calls = 0
    candidate_by_specialist: Counter[str] = Counter()
    candidate_by_scenario: Counter[str] = Counter()
    candidate_success = 0
    candidate_failure = 0
    candidate_no_tool_specialist_completions = 0
    candidate_scenarios_with_tool_use: set[str] = set()
    candidate_specialist_domains_with_tool_use: set[str] = set()
    candidate_execution_limit_hits = 0
    for run in variant_runs:
        for scenario in run.report.scenarios:
            selected_candidate = _selected_candidate(scenario.runtime)
            selected_outcomes = (
                selected_candidate.specialist_outcomes if selected_candidate is not None else ()
            )
            candidate_outcomes = [
                outcome
                for candidate in scenario.runtime.candidate_results
                for outcome in candidate.specialist_outcomes
            ]
            selected_scenario_tool_calls = 0
            candidate_scenario_tool_calls = 0
            for outcome in selected_outcomes:
                selected_by_specialist[outcome.trace.specialist_id] += outcome.trace.tool_call_count
                selected_total_tool_calls += outcome.trace.tool_call_count
                selected_scenario_tool_calls += outcome.trace.tool_call_count
                selected_success += outcome.trace.tool_call_success_count
                selected_failure += outcome.trace.tool_call_failure_count
                selected_execution_limit_hits += int(outcome.trace.agent_execution_limit_hit)
                if outcome.trace.tool_call_count > 0:
                    selected_specialist_domains_with_tool_use.add(outcome.trace.domain.value)
                if outcome.decision is not None and outcome.trace.tool_call_count == 0:
                    selected_no_tool_specialist_completions += 1
            for outcome in candidate_outcomes:
                candidate_by_specialist[outcome.trace.specialist_id] += (
                    outcome.trace.tool_call_count
                )
                candidate_total_tool_calls += outcome.trace.tool_call_count
                candidate_scenario_tool_calls += outcome.trace.tool_call_count
                candidate_success += outcome.trace.tool_call_success_count
                candidate_failure += outcome.trace.tool_call_failure_count
                candidate_execution_limit_hits += int(outcome.trace.agent_execution_limit_hit)
                if outcome.trace.tool_call_count > 0:
                    candidate_specialist_domains_with_tool_use.add(outcome.trace.domain.value)
                if outcome.decision is not None and outcome.trace.tool_call_count == 0:
                    candidate_no_tool_specialist_completions += 1
            if selected_scenario_tool_calls > 0:
                selected_scenarios_with_tool_use.add(scenario.scenario_id)
            if candidate_scenario_tool_calls > 0:
                candidate_scenarios_with_tool_use.add(scenario.scenario_id)
            selected_by_scenario[scenario.scenario_id] += selected_scenario_tool_calls
            candidate_by_scenario[scenario.scenario_id] += candidate_scenario_tool_calls
    return {
        "total_tool_calls": selected_total_tool_calls,
        "selected_total_tool_calls": selected_total_tool_calls,
        "selected_by_specialist": tuple(sorted(selected_by_specialist.items())),
        "selected_by_scenario": tuple(sorted(selected_by_scenario.items())),
        "selected_success": selected_success,
        "selected_failure": selected_failure,
        "selected_no_tool_specialist_completions": selected_no_tool_specialist_completions,
        "selected_scenarios_with_tool_use": tuple(sorted(selected_scenarios_with_tool_use)),
        "selected_specialist_domains_with_tool_use": tuple(
            sorted(selected_specialist_domains_with_tool_use)
        ),
        "selected_execution_limit_hits": selected_execution_limit_hits,
        "candidate_total_tool_calls": candidate_total_tool_calls,
        "candidate_by_specialist": tuple(sorted(candidate_by_specialist.items())),
        "candidate_by_scenario": tuple(sorted(candidate_by_scenario.items())),
        "candidate_success": candidate_success,
        "candidate_failure": candidate_failure,
        "candidate_no_tool_specialist_completions": candidate_no_tool_specialist_completions,
        "candidate_scenarios_with_tool_use": tuple(sorted(candidate_scenarios_with_tool_use)),
        "candidate_specialist_domains_with_tool_use": tuple(
            sorted(candidate_specialist_domains_with_tool_use)
        ),
        "candidate_execution_limit_hits": candidate_execution_limit_hits,
    }


def _variant_disposition(
    *,
    variant: str,
    summary: NumericSummary,
    evidence_summary: NumericSummary,
    success_summary: NumericSummary,
    validation_failure_rate: float,
    tool_totals: dict[str, Any] | None,
) -> str:
    if variant == "native_ollama":
        return "BASELINE"
    if summary.mean is None or evidence_summary.mean is None or success_summary.mean is None:
        return "RETAIN_EXPERIMENTALLY"
    if variant == "langchain_chatollama":
        if (
            summary.mean >= 0.8
            and evidence_summary.mean >= 0.8
            and success_summary.mean >= 0.8
            and validation_failure_rate <= 0.05
        ):
            return "RETAIN"
        return "RETAIN_EXPERIMENTALLY"
    if tool_totals is None or tool_totals["total_tool_calls"] == 0:
        return "REJECT_AS_DEFAULT"
    if (
        summary.mean >= 0.8
        and evidence_summary.mean >= 0.8
        and success_summary.mean >= 0.8
        and validation_failure_rate <= 0.05
    ):
        return "RETAIN_EXPERIMENTALLY"
    return "REJECT_AS_DEFAULT"


def _apply_dispositions(
    summaries: tuple[VariantAggregateReport, ...],
) -> tuple[VariantAggregateReport, ...]:
    native = next((summary for summary in summaries if summary.variant == "native_ollama"), None)
    if native is None:
        return summaries
    updated: list[VariantAggregateReport] = []
    for summary in summaries:
        if summary.variant == "native_ollama":
            updated.append(summary.model_copy(update={"disposition": "BASELINE"}))
            continue
        if summary.variant == "langchain_chatollama":
            disposition = (
                "RETAIN"
                if _comparative_ok(summary, native, strict=False)
                else "RETAIN_EXPERIMENTALLY"
            )
        else:
            if summary.total_tool_calls == 0:
                disposition = "REJECT_AS_DEFAULT"
            else:
                disposition = (
                    "RETAIN_EXPERIMENTALLY"
                    if _comparative_ok(summary, native, strict=True)
                    else "REJECT_AS_DEFAULT"
                )
        updated.append(summary.model_copy(update={"disposition": disposition}))
    return tuple(updated)


def _comparative_ok(
    candidate: VariantAggregateReport,
    native: VariantAggregateReport,
    *,
    strict: bool,
) -> bool:
    def _within(
        candidate_value: float | None, native_value: float | None, tolerance: float
    ) -> bool:
        if candidate_value is None or native_value is None:
            return False
        return candidate_value >= native_value - tolerance

    tolerance = 0.02 if not strict else 0.0
    return (
        _within(
            candidate.final_decision_accuracy.mean, native.final_decision_accuracy.mean, tolerance
        )
        and _within(
            candidate.evidence_grounded_arbitration_accuracy.mean,
            native.evidence_grounded_arbitration_accuracy.mean,
            tolerance,
        )
        and _within(
            candidate.specialist_success_rate.mean,
            native.specialist_success_rate.mean,
            0.05 if not strict else 0.0,
        )
        and candidate.structured_output_validation_failure_count
        <= native.structured_output_validation_failure_count + (1 if not strict else 0)
    )


def _summary(
    values: Iterable[float],
    *,
    use_median: bool = False,
) -> NumericSummary:
    items = sorted(values)
    if not items:
        return NumericSummary()
    center = median(items) if use_median else mean(items)
    return NumericSummary(
        mean=float(center),
        minimum=min(items),
        maximum=max(items),
        values=tuple(items),
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    items = sorted(values)
    if not items:
        return 0.0
    index = max(0, min(len(items) - 1, round((len(items) - 1) * percentile)))
    return float(items[index])


def _summary_text(summary: NumericSummary) -> str:
    if summary.mean is None:
        return "n/a"
    return f"mean={summary.mean:.3f}, range={summary.minimum:.3f}..{summary.maximum:.3f}"


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _python_version() -> str:
    import sys

    return sys.version.split()[0]
