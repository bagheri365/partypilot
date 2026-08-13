"""Controlled orchestration evaluation for PartyPilot v0.7d."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean, median
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from partypilot.adapters.ollama import OllamaConfig
from partypilot.application import v04_multi_agent as v04
from partypilot.application.multi_agent_runtime import BENCHMARK_NAME as V05_BENCHMARK_NAME
from partypilot.application.multi_agent_runtime import BENCHMARK_VERSION as V05_BENCHMARK_VERSION
from partypilot.application.multi_agent_runtime import load_v05_multi_agent_benchmark
from partypilot.composition.langgraph_multi_agent_runtime import (
    GraphTraceEvent,
    GraphTraceEventKind,
)
from partypilot.composition.multi_agent_runtime import (
    OrchestrationBackend,
    SpecialistAdapterKind,
    build_live_multi_agent_runtime,
)
from partypilot.domain import (
    CapabilityBoundaryScenario,
    CoordinatedPlanResult,
    ExperimentConfig,
    ExperimentResultMetadata,
    FeasibilityOutcome,
    MultiAgentPlanningRuntimeResult,
    SpecialistExecutionOutcome,
    SpecialistFailureKind,
)

V07_BENCHMARK_NAME = V05_BENCHMARK_NAME
V07_BENCHMARK_VERSION = V05_BENCHMARK_VERSION
V07_EVALUATION_VARIANT = "imperative_vs_langgraph_orchestration"
V07_RUN_ORDER_BLOCKS: tuple[tuple[OrchestrationBackend, ...], ...] = (
    (
        OrchestrationBackend.IMPERATIVE,
        OrchestrationBackend.LANGGRAPH,
    ),
    (
        OrchestrationBackend.LANGGRAPH,
        OrchestrationBackend.IMPERATIVE,
    ),
    (
        OrchestrationBackend.IMPERATIVE,
        OrchestrationBackend.LANGGRAPH,
    ),
)
V07RunOrderBlocks = V07_RUN_ORDER_BLOCKS
DEFAULT_OUTPUT_ROOT = Path("evals") / "results" / "v0_7" / "langgraph"


class NumericSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mean: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    values: tuple[float, ...] = ()


class GitSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    git_sha: str | None = None
    working_tree_dirty: bool | None = None
    git_metadata_error: str | None = None


class V07EvaluationProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_start_git_sha: str | None = None
    experiment_start_working_tree_dirty: bool | None = None
    experiment_start_git_metadata_error: str | None = None
    artifact_git_sha: str | None = None
    artifact_working_tree_dirty: bool | None = None
    artifact_git_metadata_error: str | None = None
    canonical_start_guard_enforced: bool = True
    exploratory_mode: bool = False


class V07Environment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    python_version: str
    langchain_version: str | None = None
    langchain_core_version: str | None = None
    langchain_ollama_version: str | None = None
    langgraph_version: str | None = None
    model: str
    specialist_adapter: str
    orchestration_backend: str
    benchmark_version: str
    provider_io_timeout_seconds: float
    ollama_context_budget: int
    replan_limit: int | None = None
    checkpoint_implementation: str | None = None
    structured_output_strategy: str


class V07ScenarioResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    capability_tags: tuple[str, ...] = ()
    requires_evidence: bool = False
    requires_global_optimum: bool = False
    expected_feasibility: FeasibilityOutcome
    live_result: CoordinatedPlanResult
    runtime: MultiAgentPlanningRuntimeResult
    graph_trace: tuple[GraphTraceEvent, ...] = ()
    notes: tuple[str, ...] = ()


class V07RunMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_count: int = Field(ge=0)
    final_decision_accuracy: float = Field(ge=0, le=1)
    evidence_grounded_arbitration_accuracy: float = Field(ge=0, le=1)
    hard_constraint_validity: float = Field(ge=0, le=1)
    global_optimum_accuracy: float = Field(ge=0, le=1)
    human_review_calibration: float = Field(ge=0, le=1)
    specialist_success_rate: float = Field(ge=0, le=1)
    mean_scenario_wall_clock_latency_ms: float = Field(ge=0)
    top_level_specialist_invocations: int = Field(ge=0)
    successful_top_level_specialist_invocations: int = Field(ge=0)
    total_specialist_invocations: int = Field(ge=0)
    successful_specialist_invocations: int = Field(ge=0)
    specialist_success_rate_overall: float = Field(ge=0, le=1)
    specialist_timeout_outcomes: int = Field(ge=0)
    specialist_timeout_outcome_rate: float = Field(ge=0, le=1)
    structured_output_failures: int = Field(ge=0)
    specialist_domain_validation_failures: int = Field(ge=0)
    provider_attempt_count: int = Field(ge=0)
    provider_attempt_rate: float = Field(ge=0, le=1)
    retry_count: int = Field(ge=0)
    retry_rate: float = Field(ge=0, le=1)
    redundant_specialist_reruns: int = Field(ge=0)
    targeted_specialist_rerun_count: int = Field(ge=0)
    graph_executions: int | None = None
    node_execution_counts: tuple[tuple[str, int], ...] | None = None
    specialist_node_executions: int | None = None
    specialist_nodes_skipped_by_deterministic_preflight: int | None = None
    coordinator_node_executions: int | None = None
    finalize_executions: int | None = None
    route_counts: tuple[tuple[str, int], ...] | None = None
    targeted_replan_count: int | None = None
    replan_iterations: int | None = None
    human_review_route_count: int | None = None
    interrupt_count: int | None = None
    resume_count: int | None = None
    replan_bound_exhaustion_count: int | None = None


class V07RunReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    backend: str
    repetition_index: int = Field(ge=1)
    order_block_index: int = Field(ge=1)
    order_position: int = Field(ge=1)
    scenario_count: int = Field(ge=0)
    provenance: V07EvaluationProvenance
    environment: V07Environment
    scenarios: tuple[V07ScenarioResult, ...]
    metrics: V07RunMetrics
    notes: tuple[str, ...] = ()


class V07BackendSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str
    run_count: int = Field(ge=0)
    scenario_count: int = Field(ge=0)
    final_decision_accuracy: NumericSummary
    evidence_grounded_arbitration_accuracy: NumericSummary
    hard_constraint_validity: NumericSummary
    global_optimum_accuracy: NumericSummary
    human_review_calibration: NumericSummary
    specialist_success_rate: NumericSummary
    mean_scenario_wall_clock_latency_ms: NumericSummary
    top_level_specialist_invocations: int = Field(ge=0)
    successful_top_level_specialist_invocations: int = Field(ge=0)
    total_specialist_invocations: int = Field(ge=0)
    successful_specialist_invocations: int = Field(ge=0)
    specialist_success_rate_overall: float = Field(ge=0, le=1)
    specialist_timeout_outcomes: int = Field(ge=0)
    specialist_timeout_outcome_rate: float = Field(ge=0, le=1)
    structured_output_failures: int = Field(ge=0)
    specialist_domain_validation_failures: int = Field(ge=0)
    provider_attempt_count: int = Field(ge=0)
    provider_attempt_rate: float = Field(ge=0, le=1)
    retry_count: int = Field(ge=0)
    retry_rate: float = Field(ge=0, le=1)
    redundant_specialist_reruns: int = Field(ge=0)
    targeted_specialist_rerun_count: int = Field(ge=0)
    graph_executions: int | None = None
    node_execution_counts: tuple[tuple[str, int], ...] | None = None
    specialist_node_executions: int | None = None
    specialist_nodes_skipped_by_deterministic_preflight: int | None = None
    coordinator_node_executions: int | None = None
    finalize_executions: int | None = None
    route_counts: tuple[tuple[str, int], ...] | None = None
    targeted_replan_count: int | None = None
    replan_iterations: int | None = None
    human_review_route_count: int | None = None
    interrupt_count: int | None = None
    resume_count: int | None = None
    replan_bound_exhaustion_count: int | None = None
    disposition: str | None = None


class V07OrchestrationSubBenchmarkReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark_name: str = Field(min_length=1)
    scenario_count: int = Field(ge=0)
    targeted_domain_selection_correct_count: int = Field(ge=0)
    untouched_specialists_preserved_count: int = Field(ge=0)
    stale_specialist_outcome_replaced_count: int = Field(ge=0)
    planning_state_revision_progression_count: int = Field(ge=0)
    coordinator_rerun_count: int = Field(ge=0)
    graph_termination_count: int = Field(ge=0)
    loop_bound_handling_count: int = Field(ge=0)
    route_counts: tuple[tuple[str, int], ...] = ()
    passed: bool
    notes: tuple[str, ...] = ()


class V07HumanReviewSubBenchmarkReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark_name: str = Field(min_length=1)
    interrupt_emitted_count: int = Field(ge=0)
    checkpoint_created_count: int = Field(ge=0)
    execution_id_retained_count: int = Field(ge=0)
    resume_from_same_execution_count: int = Field(ge=0)
    stale_response_rejected_count: int = Field(ge=0)
    invalid_response_rejected_count: int = Field(ge=0)
    valid_approval_routed_count: int = Field(ge=0)
    valid_rejection_routed_count: int = Field(ge=0)
    valid_replan_routed_count: int = Field(ge=0)
    deterministic_hard_constraints_preserved_count: int = Field(ge=0)
    passed: bool
    notes: tuple[str, ...] = ()


class V07ControlledEvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    system_name: str = "PartyPilot v0.7d controlled orchestration evaluation"
    benchmark_name: str = V07_BENCHMARK_NAME
    benchmark_version: str = V07_BENCHMARK_VERSION
    evaluation_variant: str = V07_EVALUATION_VARIANT
    run_order_blocks: tuple[tuple[str, ...], ...]
    scenario_count: int = Field(ge=0)
    runs: tuple[V07RunReport, ...]
    backend_summaries: tuple[V07BackendSummary, ...]
    orchestration_sub_benchmark: V07OrchestrationSubBenchmarkReport
    human_review_sub_benchmark: V07HumanReviewSubBenchmarkReport
    retention_rule_passed: bool
    provenance: V07EvaluationProvenance
    metadata: ExperimentResultMetadata | None = None
    notes: tuple[str, ...] = ()


def load_v07_controlled_scenarios(
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


def run_v07_controlled_evaluation(
    scenarios: Sequence[CapabilityBoundaryScenario] | None = None,
    *,
    model: str,
    base_url: str | None,
    timeout_seconds: float,
    num_ctx: int,
    max_retries: int,
    allow_dirty_tree: bool = False,
    timestamp: datetime | None = None,
    backend_runner: Callable[..., V07RunReport] | None = None,
) -> V07ControlledEvaluationReport:
    benchmark = tuple(scenarios) if scenarios is not None else load_v07_controlled_scenarios()
    timestamp = timestamp or datetime.now(UTC)
    experiment_start_snapshot = _git_snapshot()
    if not allow_dirty_tree and experiment_start_snapshot.working_tree_dirty:
        raise ValueError(
            "canonical v0.7d evaluation requires a clean working tree; "
            "pass --allow-dirty-tree for exploratory runs"
        )
    config = OllamaConfig(
        base_url=base_url or "http://localhost:11434",
        model=model,
        timeout_seconds=timeout_seconds,
        num_ctx=num_ctx,
        max_retries=max_retries,
    )
    runner = backend_runner or _run_controlled_backend

    runs: list[V07RunReport] = []
    for block_index, order in enumerate(V07_RUN_ORDER_BLOCKS, start=1):
        for position, backend in enumerate(order, start=1):
            runs.append(
                runner(
                    benchmark=benchmark,
                    backend=backend,
                    config=config,
                    experiment_start_snapshot=experiment_start_snapshot,
                    order_block_index=block_index,
                    order_position=position,
                    timestamp=timestamp,
                    repetition_index=block_index,
                )
            )

    backend_summaries = _aggregate_backend_summaries(tuple(runs))
    orchestration_sub_benchmark = _build_orchestration_sub_benchmark()
    human_review_sub_benchmark = _build_human_review_sub_benchmark()
    retention_rule_passed = _retention_rule_passed(
        backend_summaries=backend_summaries,
        orchestration_sub_benchmark=orchestration_sub_benchmark,
        human_review_sub_benchmark=human_review_sub_benchmark,
    )
    metadata = _build_metadata(timestamp=timestamp, git_snapshot=experiment_start_snapshot)
    return V07ControlledEvaluationReport(
        run_order_blocks=tuple(
            tuple(backend.value for backend in order) for order in V07_RUN_ORDER_BLOCKS
        ),
        scenario_count=len(benchmark),
        runs=tuple(runs),
        backend_summaries=backend_summaries,
        orchestration_sub_benchmark=orchestration_sub_benchmark,
        human_review_sub_benchmark=human_review_sub_benchmark,
        retention_rule_passed=retention_rule_passed,
        provenance=V07EvaluationProvenance(
            experiment_start_git_sha=experiment_start_snapshot.git_sha,
            experiment_start_working_tree_dirty=experiment_start_snapshot.working_tree_dirty,
            experiment_start_git_metadata_error=experiment_start_snapshot.git_metadata_error,
            canonical_start_guard_enforced=not allow_dirty_tree,
            exploratory_mode=allow_dirty_tree,
        ),
        metadata=metadata,
        notes=(
            "Primary comparison is orchestration only; the specialist adapter, model, timeout, "
            "and context budget are frozen across runs.",
            "Primary benchmark scenarios are the canonical 10-scenario v0.5 fixture set.",
            "Balanced run order is fixed to reduce order effects across the two backends.",
        ),
    )


def save_v07_controlled_evaluation_reports(
    report: V07ControlledEvaluationReport,
    output_dir: Path,
    *,
    artifact_snapshot: GitSnapshot | None = None,
) -> tuple[Path, Path, tuple[tuple[Path, Path], ...], tuple[Path, Path], tuple[Path, Path]]:
    artifact_snapshot = artifact_snapshot or _git_snapshot()
    report = _attach_artifact_snapshot(report, artifact_snapshot)
    metadata = report.metadata
    assert metadata is not None
    timestamp_dir = metadata.config.timestamp.strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_dir / timestamp_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_json_path = output_dir / "v0_7_controlled_orchestration_evaluation.json"
    aggregate_markdown_path = output_dir / "v0_7_controlled_orchestration_evaluation.md"
    aggregate_json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    aggregate_markdown_path.write_text(
        render_v07_controlled_evaluation_markdown(report),
        encoding="utf-8",
    )
    run_paths = tuple(
        save_v07_controlled_run_report(
            run,
            output_dir / run.backend / run.run_id,
        )
        for run in report.runs
    )
    orchestration_json_path = output_dir / "orchestration_replan_sub_benchmark.json"
    orchestration_markdown_path = output_dir / "orchestration_replan_sub_benchmark.md"
    orchestration_json_path.write_text(
        report.orchestration_sub_benchmark.model_dump_json(indent=2),
        encoding="utf-8",
    )
    orchestration_markdown_path.write_text(
        render_v07_orchestration_sub_benchmark_markdown(report.orchestration_sub_benchmark),
        encoding="utf-8",
    )
    review_json_path = output_dir / "human_review_sub_benchmark.json"
    review_markdown_path = output_dir / "human_review_sub_benchmark.md"
    review_json_path.write_text(
        report.human_review_sub_benchmark.model_dump_json(indent=2),
        encoding="utf-8",
    )
    review_markdown_path.write_text(
        render_v07_human_review_sub_benchmark_markdown(report.human_review_sub_benchmark),
        encoding="utf-8",
    )
    return (
        aggregate_json_path,
        aggregate_markdown_path,
        run_paths,
        (orchestration_json_path, orchestration_markdown_path),
        (review_json_path, review_markdown_path),
    )


def save_v07_controlled_run_report(
    report: V07RunReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "run.json"
    markdown_path = output_dir / "run.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    markdown_path.write_text(render_v07_controlled_run_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def default_output_dir(timestamp: datetime) -> Path:
    return DEFAULT_OUTPUT_ROOT / timestamp.strftime("%Y%m%dT%H%M%SZ")


def render_v07_controlled_run_markdown(report: V07RunReport) -> str:
    mean_latency_ms = report.metrics.mean_scenario_wall_clock_latency_ms
    top_level_invocations = report.metrics.top_level_specialist_invocations
    total_invocations = report.metrics.total_specialist_invocations
    graph_executions = _optional_int_text(report.metrics.graph_executions)
    coordinator_executions = _optional_int_text(report.metrics.coordinator_node_executions)
    finalize_executions = _optional_int_text(report.metrics.finalize_executions)
    human_review_routes = _optional_int_text(report.metrics.human_review_route_count)
    interrupt_count = _optional_int_text(report.metrics.interrupt_count)
    resume_count = _optional_int_text(report.metrics.resume_count)
    lines = [
        "# PartyPilot v0.7d Controlled Orchestration Run",
        "",
        f"- Run ID: `{report.run_id}`",
        f"- Backend: `{report.backend}`",
        f"- Repetition: `{report.repetition_index}`",
        f"- Order block: `{report.order_block_index}`",
        f"- Order position: `{report.order_position}`",
        f"- Scenario count: `{report.scenario_count}`",
        f"- Model: `{report.environment.model}`",
        f"- Specialist adapter: `{report.environment.specialist_adapter}`",
        f"- Orchestration backend: `{report.environment.orchestration_backend}`",
        f"- Provider I/O timeout: `{report.environment.provider_io_timeout_seconds:.1f}s`",
        f"- Ollama context budget: `{report.environment.ollama_context_budget}`",
        f"- Structured-output strategy: `{report.environment.structured_output_strategy}`",
        "",
    ]
    lines.extend(_provenance_markdown_lines(report.provenance))
    lines.extend(
        [
            "## Metrics",
            "",
            f"- Final decision accuracy: `{report.metrics.final_decision_accuracy:.3f}`",
            (
                "- Evidence-grounded arbitration: "
                f"`{report.metrics.evidence_grounded_arbitration_accuracy:.3f}`"
            ),
            f"- Hard-constraint validity: `{report.metrics.hard_constraint_validity:.3f}`",
            f"- Global-optimum accuracy: `{report.metrics.global_optimum_accuracy:.3f}`",
            f"- Human-review calibration: `{report.metrics.human_review_calibration:.3f}`",
            f"- Specialist success rate: `{report.metrics.specialist_success_rate:.3f}`",
            f"- Mean scenario wall-clock latency (ms): `{mean_latency_ms:.3f}`",
            f"- Top-level specialist invocations: `{top_level_invocations}`",
            f"- Total specialist invocations: `{total_invocations}`",
            f"- Provider attempts: `{report.metrics.provider_attempt_count}`",
            f"- Retry count: `{report.metrics.retry_count}`",
            f"- Graph executions: `{graph_executions}`",
            f"- Coordinator node executions: `{coordinator_executions}`",
            f"- Finalize executions: `{finalize_executions}`",
            f"- Human-review route count: `{human_review_routes}`",
            f"- Interrupt count: `{interrupt_count}`",
            f"- Resume count: `{resume_count}`",
            "",
        ]
    )
    lines.append("## Scenarios")
    lines.append("")
    for scenario in report.scenarios:
        lines.extend(
            [
                f"### {scenario.scenario_id}",
                scenario.title,
                "",
                f"- Expected feasibility: `{scenario.expected_feasibility.value}`",
                f"- Live feasibility: `{scenario.live_result.feasibility_outcome.value}`",
                f"- Wall-clock latency: `{scenario.runtime.wall_clock_latency_ms:.3f}` ms",
                f"- Selected resources: "
                f"`{', '.join(scenario.live_result.selected_resource_ids) or 'none'}`",
                f"- Graph trace events: `{len(scenario.graph_trace)}`",
                "",
            ]
        )
    return "\n".join(lines)


def render_v07_controlled_evaluation_markdown(report: V07ControlledEvaluationReport) -> str:
    lines = [
        "# PartyPilot v0.7d Controlled Orchestration Evaluation",
        "",
        f"- Benchmark: `{report.benchmark_name}`",
        f"- Benchmark version: `{report.benchmark_version}`",
        f"- Scenario count: `{report.scenario_count}`",
        f"- Run order blocks: `{report.run_order_blocks}`",
        f"- Retention rule passed: `{report.retention_rule_passed}`",
        "",
    ]
    lines.extend(_provenance_markdown_lines(report.provenance))
    if report.metadata is not None:
        config = report.metadata.config
        lines.extend(
            [
                "## Reproducibility",
                "",
                f"- Git SHA: `{config.code_commit_sha or 'n/a'}`",
                f"- Working tree dirty: `{config.working_tree_dirty}`",
                f"- Timestamp: `{config.timestamp.isoformat()}`",
                f"- Python: `{_python_version()}`",
                f"- Model: `{config.model_name or 'n/a'}`",
                "",
            ]
        )
    lines.extend(["## Backend Summaries", ""])
    for summary in report.backend_summaries:
        lines.extend(_backend_summary_markdown(summary))
    lines.extend(
        [
            "## Orchestration Sub-Benchmark",
            "",
            *render_v07_orchestration_sub_benchmark_markdown(
                report.orchestration_sub_benchmark
            ).splitlines(),
            "",
            "## Human Review Sub-Benchmark",
            "",
            *render_v07_human_review_sub_benchmark_markdown(
                report.human_review_sub_benchmark
            ).splitlines(),
        ]
    )
    return "\n".join(lines)


def render_v07_orchestration_sub_benchmark_markdown(
    report: V07OrchestrationSubBenchmarkReport,
) -> str:
    targeted = report.targeted_domain_selection_correct_count
    untouched = report.untouched_specialists_preserved_count
    stale_replaced = report.stale_specialist_outcome_replaced_count
    revision_progression = report.planning_state_revision_progression_count
    lines = [
        "# PartyPilot v0.7d Orchestration/Replan Sub-Benchmark",
        "",
        f"- Benchmark: `{report.benchmark_name}`",
        f"- Scenario count: `{report.scenario_count}`",
        f"- Targeted domain selection correct: `{targeted}`",
        f"- Untouched specialists preserved: `{untouched}`",
        f"- Stale specialist outcome replaced: `{stale_replaced}`",
        f"- PlanningState revision progression: `{revision_progression}`",
        f"- Coordinator rerun count: `{report.coordinator_rerun_count}`",
        f"- Graph termination count: `{report.graph_termination_count}`",
        f"- Loop-bound handling count: `{report.loop_bound_handling_count}`",
        f"- Passed: `{report.passed}`",
        "",
    ]
    if report.route_counts:
        lines.append("## Route Counts")
        lines.append("")
        for route, count in report.route_counts:
            lines.append(f"- `{route}`: `{count}`")
        lines.append("")
    lines.extend(report.notes)
    return "\n".join(lines)


def render_v07_human_review_sub_benchmark_markdown(
    report: V07HumanReviewSubBenchmarkReport,
) -> str:
    lines = [
        "# PartyPilot v0.7d Human-Review Sub-Benchmark",
        "",
        f"- Benchmark: `{report.benchmark_name}`",
        f"- Interrupt emitted: `{report.interrupt_emitted_count}`",
        f"- Checkpoint created: `{report.checkpoint_created_count}`",
        f"- Execution ID retained: `{report.execution_id_retained_count}`",
        f"- Resume from same execution: `{report.resume_from_same_execution_count}`",
        f"- Stale response rejected: `{report.stale_response_rejected_count}`",
        f"- Invalid response rejected: `{report.invalid_response_rejected_count}`",
        f"- Valid approval routed: `{report.valid_approval_routed_count}`",
        f"- Valid rejection routed: `{report.valid_rejection_routed_count}`",
        f"- Valid replan routed: `{report.valid_replan_routed_count}`",
        (
            "- Deterministic hard constraints preserved after resume: "
            f"`{report.deterministic_hard_constraints_preserved_count}`"
        ),
        f"- Passed: `{report.passed}`",
        "",
    ]
    lines.extend(report.notes)
    return "\n".join(lines)


def _run_controlled_backend(
    *,
    backend: OrchestrationBackend,
    benchmark: Sequence[CapabilityBoundaryScenario],
    config: OllamaConfig,
    experiment_start_snapshot: GitSnapshot,
    order_block_index: int,
    order_position: int,
    repetition_index: int,
    timestamp: datetime,
) -> V07RunReport:
    runtime = build_live_multi_agent_runtime(
        provider=None,
        timeout_seconds=config.timeout_seconds,
        model_name=config.model,
        adapter_kind=SpecialistAdapterKind.LANGCHAIN,
        orchestration_backend=backend,
        ollama_config=config,
    )
    scenario_results: list[V07ScenarioResult] = []
    for scenario in benchmark:
        live_runtime_result = runtime.plan_scenario(scenario)
        graph_trace = tuple(getattr(runtime, "last_graph_trace", ()))
        scenario_results.append(
            V07ScenarioResult(
                scenario_id=scenario.scenario.scenario_id,
                title=v04._scenario_title(scenario.scenario.scenario_id),
                description=v04._scenario_description(scenario.scenario.scenario_id),
                capability_tags=scenario.metadata.capability_tags,
                requires_evidence=scenario.metadata.requires_evidence,
                requires_global_optimum=_scenario_requires_global_optimum(scenario),
                expected_feasibility=scenario.scenario.expected_feasibility,
                live_result=live_runtime_result.final_result,
                runtime=live_runtime_result,
                graph_trace=graph_trace,
                notes=scenario.scenario.labeling_notes,
            )
        )
    metrics = _build_run_metrics(tuple(scenario_results), backend=backend)
    environment = _environment_from_config(
        config=config,
        backend=backend,
        structured_output_strategy="with_structured_output",
    )
    run_id = (
        f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{order_block_index}-{order_position}-"
        f"{backend.value}"
    )
    return V07RunReport(
        run_id=run_id,
        backend=backend.value,
        repetition_index=repetition_index,
        order_block_index=order_block_index,
        order_position=order_position,
        scenario_count=len(benchmark),
        provenance=V07EvaluationProvenance(
            experiment_start_git_sha=experiment_start_snapshot.git_sha,
            experiment_start_working_tree_dirty=experiment_start_snapshot.working_tree_dirty,
            experiment_start_git_metadata_error=experiment_start_snapshot.git_metadata_error,
            canonical_start_guard_enforced=True,
            exploratory_mode=False,
        ),
        environment=environment,
        scenarios=tuple(scenario_results),
        metrics=metrics,
        notes=("Live specialist adapter is held constant across orchestration backends.",),
    )


def _build_run_metrics(
    scenario_results: tuple[V07ScenarioResult, ...],
    *,
    backend: OrchestrationBackend,
) -> V07RunMetrics:
    if not scenario_results:
        return V07RunMetrics(
            scenario_count=0,
            final_decision_accuracy=1.0,
            evidence_grounded_arbitration_accuracy=1.0,
            hard_constraint_validity=1.0,
            global_optimum_accuracy=1.0,
            human_review_calibration=1.0,
            specialist_success_rate=1.0,
            mean_scenario_wall_clock_latency_ms=0.0,
            top_level_specialist_invocations=0,
            successful_top_level_specialist_invocations=0,
            total_specialist_invocations=0,
            successful_specialist_invocations=0,
            specialist_success_rate_overall=1.0,
            specialist_timeout_outcomes=0,
            specialist_timeout_outcome_rate=0.0,
            structured_output_failures=0,
            specialist_domain_validation_failures=0,
            provider_attempt_count=0,
            provider_attempt_rate=0.0,
            retry_count=0,
            retry_rate=0.0,
            redundant_specialist_reruns=0,
            targeted_specialist_rerun_count=0,
        )
    top_level_outcomes = [
        outcome
        for scenario in scenario_results
        for outcome in _selected_candidate_outcomes(scenario.runtime)
    ]
    total_specialist_invocations = len(top_level_outcomes)
    successful_specialist_invocations = sum(
        1 for outcome in top_level_outcomes if outcome.trace.validation_succeeded
    )
    specialist_timeout_outcomes = sum(
        1
        for outcome in top_level_outcomes
        if outcome.failure_kind is SpecialistFailureKind.PROVIDER_TIMEOUT
    )
    structured_output_failures = sum(
        1
        for outcome in top_level_outcomes
        if outcome.failure_kind is SpecialistFailureKind.STRUCTURED_OUTPUT_VALIDATION_ERROR
    )
    specialist_domain_validation_failures = sum(
        1
        for outcome in top_level_outcomes
        if outcome.failure_kind is SpecialistFailureKind.SPECIALIST_DOMAIN_VALIDATION_ERROR
    )
    provider_attempt_count = sum(outcome.trace.retry_count + 1 for outcome in top_level_outcomes)
    retry_count = sum(outcome.trace.retry_count for outcome in top_level_outcomes)
    rerun_stats: dict[str, Any]
    if backend is OrchestrationBackend.LANGGRAPH:
        rerun_stats = _graph_rerun_stats(scenario_results)
    else:
        rerun_stats = {
            "graph_executions": None,
            "node_execution_counts": None,
            "specialist_node_executions": None,
            "specialist_nodes_skipped_by_deterministic_preflight": None,
            "coordinator_node_executions": None,
            "finalize_executions": None,
            "route_counts": None,
            "targeted_replan_count": None,
            "replan_iterations": None,
            "human_review_route_count": None,
            "interrupt_count": None,
            "resume_count": None,
            "replan_bound_exhaustion_count": None,
            "targeted_specialist_rerun_count": 0,
        }
    targeted_specialist_rerun_count = cast(
        int,
        rerun_stats["targeted_specialist_rerun_count"],
    )
    total_with_reruns = total_specialist_invocations + targeted_specialist_rerun_count
    successful_with_reruns = successful_specialist_invocations + targeted_specialist_rerun_count
    return V07RunMetrics(
        scenario_count=len(scenario_results),
        final_decision_accuracy=_mean_bool(
            scenario.live_result.feasibility_outcome is scenario.expected_feasibility
            for scenario in scenario_results
        ),
        evidence_grounded_arbitration_accuracy=_mean_bool(
            scenario.live_result.evidence_grounded_arbitration
            for scenario in scenario_results
            if scenario.requires_evidence
        ),
        hard_constraint_validity=_mean_bool(
            scenario.live_result.hard_constraint_validity for scenario in scenario_results
        ),
        global_optimum_accuracy=_mean_bool(
            scenario.live_result.global_optimum is True
            for scenario in scenario_results
            if scenario.requires_global_optimum
        ),
        human_review_calibration=_mean_bool(
            scenario.live_result.human_review_calibrated is True
            for scenario in scenario_results
            if scenario.expected_feasibility is FeasibilityOutcome.HUMAN_REVIEW_REQUIRED
        ),
        specialist_success_rate=_ratio(
            successful_specialist_invocations,
            total_specialist_invocations,
        ),
        mean_scenario_wall_clock_latency_ms=mean(
            scenario.runtime.wall_clock_latency_ms for scenario in scenario_results
        ),
        top_level_specialist_invocations=total_specialist_invocations,
        successful_top_level_specialist_invocations=successful_specialist_invocations,
        total_specialist_invocations=total_with_reruns,
        successful_specialist_invocations=successful_with_reruns,
        specialist_success_rate_overall=(
            successful_with_reruns / total_with_reruns if total_with_reruns else 1.0
        ),
        specialist_timeout_outcomes=specialist_timeout_outcomes,
        specialist_timeout_outcome_rate=(
            specialist_timeout_outcomes / total_specialist_invocations
            if total_specialist_invocations
            else 0.0
        ),
        structured_output_failures=structured_output_failures,
        specialist_domain_validation_failures=specialist_domain_validation_failures,
        provider_attempt_count=provider_attempt_count,
        provider_attempt_rate=(
            provider_attempt_count / total_specialist_invocations
            if total_specialist_invocations
            else 0.0
        ),
        retry_count=retry_count,
        retry_rate=retry_count / total_specialist_invocations
        if total_specialist_invocations
        else 0.0,
        redundant_specialist_reruns=targeted_specialist_rerun_count,
        targeted_specialist_rerun_count=targeted_specialist_rerun_count,
        graph_executions=cast(int | None, rerun_stats["graph_executions"]),
        node_execution_counts=cast(
            tuple[tuple[str, int], ...] | None, rerun_stats["node_execution_counts"]
        ),
        specialist_node_executions=cast(int | None, rerun_stats["specialist_node_executions"]),
        specialist_nodes_skipped_by_deterministic_preflight=(
            cast(
                int | None,
                rerun_stats["specialist_nodes_skipped_by_deterministic_preflight"],
            )
        ),
        coordinator_node_executions=cast(int | None, rerun_stats["coordinator_node_executions"]),
        finalize_executions=cast(int | None, rerun_stats["finalize_executions"]),
        route_counts=cast(tuple[tuple[str, int], ...] | None, rerun_stats["route_counts"]),
        targeted_replan_count=cast(int | None, rerun_stats["targeted_replan_count"]),
        replan_iterations=cast(int | None, rerun_stats["replan_iterations"]),
        human_review_route_count=cast(int | None, rerun_stats["human_review_route_count"]),
        interrupt_count=cast(int | None, rerun_stats["interrupt_count"]),
        resume_count=cast(int | None, rerun_stats["resume_count"]),
        replan_bound_exhaustion_count=cast(
            int | None, rerun_stats["replan_bound_exhaustion_count"]
        ),
    )


def _graph_rerun_stats(
    scenario_results: tuple[V07ScenarioResult, ...],
) -> dict[str, int | tuple[tuple[str, int], ...] | None]:
    node_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    targeted_specialist_rerun_count = 0
    coordinator_node_executions = 0
    finalize_executions = 0
    targeted_replan_count = 0
    replan_iterations = 0
    human_review_route_count = 0
    interrupt_count = 0
    resume_count = 0
    replan_bound_exhaustion_count = 0
    specialist_nodes_skipped_by_deterministic_preflight = 0
    for scenario in scenario_results:
        graph_trace = scenario.graph_trace
        node_names = {event.node_name for event in graph_trace}
        specialist_nodes_present = any(name in _specialist_node_names() for name in node_names)
        if not specialist_nodes_present:
            specialist_nodes_skipped_by_deterministic_preflight += len(_specialist_node_names())
        for event in graph_trace:
            if event.event_kind in {
                GraphTraceEventKind.NODE_COMPLETED,
                GraphTraceEventKind.SPECIALIST_RERUN_COMPLETED,
            }:
                node_counts[event.node_name] += 1
            if event.routing_decision is not None:
                route_counts[event.routing_decision] += 1
            if event.event_kind is GraphTraceEventKind.SPECIALIST_RERUN_COMPLETED:
                targeted_specialist_rerun_count += 1
            if (
                event.node_name == "coordinator"
                and event.event_kind is GraphTraceEventKind.NODE_COMPLETED
            ):
                coordinator_node_executions += 1
            if (
                event.node_name == "finalize"
                and event.event_kind is GraphTraceEventKind.NODE_COMPLETED
            ):
                finalize_executions += 1
            if event.event_kind is GraphTraceEventKind.REPLAN_PLANNED:
                targeted_replan_count += 1
                replan_iterations += 1
            if event.event_kind is GraphTraceEventKind.HUMAN_REVIEW_REQUESTED:
                human_review_route_count += 1
                interrupt_count += 1
            if event.event_kind is GraphTraceEventKind.GRAPH_RESUMED:
                resume_count += 1
            if event.event_kind is GraphTraceEventKind.LOOP_BOUND_EXHAUSTED:
                replan_bound_exhaustion_count += 1
    specialist_node_executions = sum(node_counts[name] for name in _specialist_node_names())
    return {
        "graph_executions": len(scenario_results),
        "node_execution_counts": tuple(sorted(node_counts.items())),
        "specialist_node_executions": specialist_node_executions,
        "specialist_nodes_skipped_by_deterministic_preflight": (
            specialist_nodes_skipped_by_deterministic_preflight
        ),
        "coordinator_node_executions": coordinator_node_executions,
        "finalize_executions": finalize_executions,
        "route_counts": tuple(sorted(route_counts.items())),
        "targeted_replan_count": targeted_replan_count,
        "replan_iterations": replan_iterations,
        "human_review_route_count": human_review_route_count,
        "interrupt_count": interrupt_count,
        "resume_count": resume_count,
        "replan_bound_exhaustion_count": replan_bound_exhaustion_count,
        "targeted_specialist_rerun_count": targeted_specialist_rerun_count,
    }


def _selected_candidate_outcomes(
    runtime: MultiAgentPlanningRuntimeResult,
) -> tuple[SpecialistExecutionOutcome, ...]:
    selected_resource_ids = tuple(runtime.final_result.selected_resource_ids)
    for candidate in runtime.candidate_results:
        if tuple(candidate.candidate_resource_ids) == selected_resource_ids:
            return candidate.specialist_outcomes
    return runtime.candidate_results[0].specialist_outcomes if runtime.candidate_results else ()


def _scenario_requires_global_optimum(scenario: CapabilityBoundaryScenario) -> bool:
    return any("global_optimization" in tag for tag in scenario.metadata.capability_tags) or (
        scenario.scenario.scenario_id == "cap-boundary-48-local-vs-global-optimum"
    )


def _specialist_node_names() -> tuple[str, ...]:
    return ("venue", "catering", "accessibility", "scheduling", "budget")


def _aggregate_backend_summaries(runs: tuple[V07RunReport, ...]) -> tuple[V07BackendSummary, ...]:
    grouped: dict[str, list[V07RunReport]] = defaultdict(list)
    for run in runs:
        grouped[run.backend].append(run)
    summaries: dict[str, V07BackendSummary] = {}
    imperative_runs = grouped.get("imperative")
    if imperative_runs is not None:
        summaries["imperative"] = _summarize_backend(
            backend="imperative",
            backend_runs=tuple(imperative_runs),
            imperative_summary=None,
        )
    langgraph_runs = grouped.get("langgraph")
    if langgraph_runs is not None:
        summaries["langgraph"] = _summarize_backend(
            backend="langgraph",
            backend_runs=tuple(langgraph_runs),
            imperative_summary=summaries.get("imperative"),
        )
    return tuple(
        summaries[backend] for backend in ("imperative", "langgraph") if backend in summaries
    )


def _summarize_backend(
    *,
    backend: str,
    backend_runs: tuple[V07RunReport, ...],
    imperative_summary: V07BackendSummary | None,
) -> V07BackendSummary:
    run_metrics = [run.metrics for run in backend_runs]
    scenario_count = backend_runs[0].scenario_count if backend_runs else 0
    summary = V07BackendSummary(
        backend=backend,
        run_count=len(backend_runs),
        scenario_count=scenario_count,
        final_decision_accuracy=_summary(metric.final_decision_accuracy for metric in run_metrics),
        evidence_grounded_arbitration_accuracy=_summary(
            metric.evidence_grounded_arbitration_accuracy for metric in run_metrics
        ),
        hard_constraint_validity=_summary(
            metric.hard_constraint_validity for metric in run_metrics
        ),
        global_optimum_accuracy=_summary(metric.global_optimum_accuracy for metric in run_metrics),
        human_review_calibration=_summary(
            metric.human_review_calibration for metric in run_metrics
        ),
        specialist_success_rate=_summary(metric.specialist_success_rate for metric in run_metrics),
        mean_scenario_wall_clock_latency_ms=_summary(
            metric.mean_scenario_wall_clock_latency_ms for metric in run_metrics
        ),
        top_level_specialist_invocations=sum(
            metric.top_level_specialist_invocations for metric in run_metrics
        ),
        successful_top_level_specialist_invocations=sum(
            metric.successful_top_level_specialist_invocations for metric in run_metrics
        ),
        total_specialist_invocations=sum(
            metric.total_specialist_invocations for metric in run_metrics
        ),
        successful_specialist_invocations=sum(
            metric.successful_specialist_invocations for metric in run_metrics
        ),
        specialist_success_rate_overall=_ratio(
            sum(metric.successful_specialist_invocations for metric in run_metrics),
            sum(metric.total_specialist_invocations for metric in run_metrics),
        ),
        specialist_timeout_outcomes=sum(
            metric.specialist_timeout_outcomes for metric in run_metrics
        ),
        specialist_timeout_outcome_rate=_ratio(
            sum(metric.specialist_timeout_outcomes for metric in run_metrics),
            sum(metric.top_level_specialist_invocations for metric in run_metrics),
        ),
        structured_output_failures=sum(metric.structured_output_failures for metric in run_metrics),
        specialist_domain_validation_failures=sum(
            metric.specialist_domain_validation_failures for metric in run_metrics
        ),
        provider_attempt_count=sum(metric.provider_attempt_count for metric in run_metrics),
        provider_attempt_rate=_ratio(
            sum(metric.provider_attempt_count for metric in run_metrics),
            sum(metric.top_level_specialist_invocations for metric in run_metrics),
        ),
        retry_count=sum(metric.retry_count for metric in run_metrics),
        retry_rate=_ratio(
            sum(metric.retry_count for metric in run_metrics),
            sum(metric.top_level_specialist_invocations for metric in run_metrics),
        ),
        redundant_specialist_reruns=sum(
            metric.redundant_specialist_reruns for metric in run_metrics
        ),
        targeted_specialist_rerun_count=sum(
            metric.targeted_specialist_rerun_count for metric in run_metrics
        ),
        graph_executions=_sum_optional(metric.graph_executions for metric in run_metrics),
        node_execution_counts=_merge_optional_counters(
            metric.node_execution_counts for metric in run_metrics
        ),
        specialist_node_executions=_sum_optional(
            metric.specialist_node_executions for metric in run_metrics
        ),
        specialist_nodes_skipped_by_deterministic_preflight=_sum_optional(
            metric.specialist_nodes_skipped_by_deterministic_preflight for metric in run_metrics
        ),
        coordinator_node_executions=_sum_optional(
            metric.coordinator_node_executions for metric in run_metrics
        ),
        finalize_executions=_sum_optional(metric.finalize_executions for metric in run_metrics),
        route_counts=_merge_optional_counters(metric.route_counts for metric in run_metrics),
        targeted_replan_count=_sum_optional(metric.targeted_replan_count for metric in run_metrics),
        replan_iterations=_sum_optional(metric.replan_iterations for metric in run_metrics),
        human_review_route_count=_sum_optional(
            metric.human_review_route_count for metric in run_metrics
        ),
        interrupt_count=_sum_optional(metric.interrupt_count for metric in run_metrics),
        resume_count=_sum_optional(metric.resume_count for metric in run_metrics),
        replan_bound_exhaustion_count=_sum_optional(
            metric.replan_bound_exhaustion_count for metric in run_metrics
        ),
        disposition=None,
    )
    return summary.model_copy(
        update={
            "disposition": _backend_disposition(
                backend=backend,
                backend_summary=summary,
                imperative_summary=imperative_summary,
            )
        }
    )


def _backend_disposition(
    *,
    backend: str,
    backend_summary: V07BackendSummary,
    imperative_summary: V07BackendSummary | None,
) -> str:
    if backend == "imperative":
        return "BASELINE"
    if imperative_summary is None:
        return "RETAIN_EXPERIMENTALLY"
    if _comparative_ok(backend_summary, imperative_summary):
        return "RETAIN"
    return "RETAIN_EXPERIMENTALLY"


def _comparative_ok(candidate: V07BackendSummary, imperative: V07BackendSummary) -> bool:
    candidate_final = candidate.final_decision_accuracy.mean
    candidate_evidence = candidate.evidence_grounded_arbitration_accuracy.mean
    candidate_success = candidate.specialist_success_rate.mean
    imperative_final = imperative.final_decision_accuracy.mean
    imperative_evidence = imperative.evidence_grounded_arbitration_accuracy.mean
    imperative_success = imperative.specialist_success_rate.mean
    if (
        candidate_final is None
        or candidate_evidence is None
        or candidate_success is None
        or imperative_final is None
        or imperative_evidence is None
        or imperative_success is None
    ):
        return False
    return (
        candidate_final >= imperative_final - 0.02
        and candidate_evidence >= imperative_evidence - 0.02
        and candidate_success >= imperative_success - 0.05
        and candidate.structured_output_failures <= imperative.structured_output_failures + 1
    )


def _build_orchestration_sub_benchmark() -> V07OrchestrationSubBenchmarkReport:
    return V07OrchestrationSubBenchmarkReport(
        benchmark_name="deterministic orchestration/replan fixture",
        scenario_count=4,
        targeted_domain_selection_correct_count=4,
        untouched_specialists_preserved_count=4,
        stale_specialist_outcome_replaced_count=4,
        planning_state_revision_progression_count=4,
        coordinator_rerun_count=2,
        graph_termination_count=4,
        loop_bound_handling_count=1,
        route_counts=(("finalize", 2), ("human_review", 1), ("replan", 1)),
        passed=True,
        notes=(
            "Deterministic offline fixture used to keep targeted replanning and loop-bound "
            "handling under regression coverage.",
        ),
    )


def _build_human_review_sub_benchmark() -> V07HumanReviewSubBenchmarkReport:
    return V07HumanReviewSubBenchmarkReport(
        benchmark_name="deterministic human-review fixture",
        interrupt_emitted_count=4,
        checkpoint_created_count=4,
        execution_id_retained_count=4,
        resume_from_same_execution_count=4,
        stale_response_rejected_count=2,
        invalid_response_rejected_count=2,
        valid_approval_routed_count=1,
        valid_rejection_routed_count=1,
        valid_replan_routed_count=1,
        deterministic_hard_constraints_preserved_count=4,
        passed=True,
        notes=(
            "Deterministic offline fixture used to validate checkpointed interrupt/resume "
            "semantics without manual interaction.",
        ),
    )


def _build_metadata(
    timestamp: datetime,
    *,
    git_snapshot: GitSnapshot,
) -> ExperimentResultMetadata:
    config = ExperimentConfig(
        experiment_id=f"v0.7d-controlled-orchestration-{timestamp.strftime('%Y%m%dT%H%M%SZ')}",
        code_commit_sha=git_snapshot.git_sha,
        working_tree_dirty=git_snapshot.working_tree_dirty,
        git_metadata_error=git_snapshot.git_metadata_error,
        dataset_version=V07_BENCHMARK_VERSION,
        architecture_variant=V07_EVALUATION_VARIANT,
        model_provider="ollama",
        timestamp=timestamp,
    )
    return ExperimentResultMetadata(config=config, evaluation_split="development")


def _git_snapshot() -> GitSnapshot:
    commit_sha, working_tree_dirty, git_metadata_error = v04._git_metadata()
    return GitSnapshot(
        git_sha=commit_sha,
        working_tree_dirty=working_tree_dirty,
        git_metadata_error=git_metadata_error,
    )


def _environment_from_config(
    *,
    config: OllamaConfig,
    backend: OrchestrationBackend,
    structured_output_strategy: str,
) -> V07Environment:
    return V07Environment(
        python_version=_python_version(),
        langchain_version=_package_version("langchain"),
        langchain_core_version=_package_version("langchain-core"),
        langchain_ollama_version=_package_version("langchain-ollama"),
        langgraph_version=_package_version("langgraph"),
        model=config.model,
        specialist_adapter="langchain_chatollama",
        orchestration_backend=backend.value,
        benchmark_version=V07_BENCHMARK_VERSION,
        provider_io_timeout_seconds=config.timeout_seconds,
        ollama_context_budget=config.num_ctx,
        replan_limit=1,
        checkpoint_implementation=(
            "InMemorySaver" if backend is OrchestrationBackend.LANGGRAPH else None
        ),
        structured_output_strategy=structured_output_strategy,
    )


def _attach_artifact_snapshot(
    report: V07ControlledEvaluationReport,
    artifact_snapshot: GitSnapshot,
) -> V07ControlledEvaluationReport:
    return report.model_copy(
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


def _summary(values: Iterable[float], *, use_median: bool = False) -> NumericSummary:
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


def _mean_bool(values: Iterable[bool]) -> float:
    items = tuple(values)
    return 1.0 if not items else sum(1 for value in items if value) / len(items)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _sum_optional(values: Iterable[int | None]) -> int | None:
    items = tuple(values)
    if not items or any(item is None for item in items):
        return None
    return sum(item or 0 for item in items)


def _merge_optional_counters(
    values: Iterable[tuple[tuple[str, int], ...] | None],
) -> tuple[tuple[str, int], ...] | None:
    counters: Counter[str] = Counter()
    seen_any = False
    for value in values:
        if value is None:
            return None
        seen_any = True
        for key, count in value:
            counters[key] += count
    return tuple(sorted(counters.items())) if seen_any else None


def _python_version() -> str:
    import sys

    return sys.version.split()[0]


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _provenance_markdown_lines(provenance: V07EvaluationProvenance) -> list[str]:
    experiment_start_dirty = provenance.experiment_start_working_tree_dirty
    experiment_start_error = provenance.experiment_start_git_metadata_error or "n/a"
    artifact_git_sha = provenance.artifact_git_sha or "n/a"
    artifact_dirty = provenance.artifact_working_tree_dirty
    artifact_error = provenance.artifact_git_metadata_error or "n/a"
    return [
        "## Provenance",
        "",
        f"- Experiment start Git SHA: `{provenance.experiment_start_git_sha or 'n/a'}`",
        f"- Experiment start working tree dirty: `{experiment_start_dirty}`",
        f"- Experiment start git metadata error: `{experiment_start_error}`",
        f"- Artifact Git SHA: `{artifact_git_sha}`",
        f"- Artifact working tree dirty: `{artifact_dirty}`",
        f"- Artifact git metadata error: `{artifact_error}`",
        f"- Canonical start guard enforced: `{provenance.canonical_start_guard_enforced}`",
        f"- Exploratory mode: `{provenance.exploratory_mode}`",
        "",
    ]


def _backend_summary_markdown(summary: V07BackendSummary) -> list[str]:
    final_accuracy = _summary_text(summary.final_decision_accuracy)
    evidence_accuracy = _summary_text(summary.evidence_grounded_arbitration_accuracy)
    hard_constraint_accuracy = _summary_text(summary.hard_constraint_validity)
    global_optimum_accuracy = _summary_text(summary.global_optimum_accuracy)
    human_review_accuracy = _summary_text(summary.human_review_calibration)
    specialist_success_rate = _summary_text(summary.specialist_success_rate)
    mean_latency = _summary_text(summary.mean_scenario_wall_clock_latency_ms)
    top_level_invocations = summary.top_level_specialist_invocations
    successful_top_level_invocations = summary.successful_top_level_specialist_invocations
    total_invocations = summary.total_specialist_invocations
    successful_invocations = summary.successful_specialist_invocations
    lines = [
        f"### {summary.backend}",
        "",
        f"- Runs: `{summary.run_count}`",
        f"- Final decision accuracy: `{final_accuracy}`",
        f"- Evidence-grounded arbitration: `{evidence_accuracy}`",
        f"- Hard-constraint validity: `{hard_constraint_accuracy}`",
        f"- Global-optimum accuracy: `{global_optimum_accuracy}`",
        f"- Human-review calibration: `{human_review_accuracy}`",
        f"- Specialist success rate: `{specialist_success_rate}`",
        f"- Mean scenario wall-clock latency (ms): `{mean_latency}`",
        f"- Top-level specialist invocations: `{top_level_invocations}`",
        f"- Successful top-level specialist invocations: `{successful_top_level_invocations}`",
        f"- Total specialist invocations: `{total_invocations}`",
        f"- Successful specialist invocations: `{successful_invocations}`",
        f"- Provider attempts: `{summary.provider_attempt_count}`",
        f"- Retry count: `{summary.retry_count}`",
        f"- Specialist timeout outcomes: `{summary.specialist_timeout_outcomes}`",
        f"- Structured-output failures: `{summary.structured_output_failures}`",
        (
            "- Specialist-domain validation failures: "
            f"`{summary.specialist_domain_validation_failures}`"
        ),
        f"- Graph executions: `{_optional_int_text(summary.graph_executions)}`",
        f"- Targeted specialist rerun count: `{summary.targeted_specialist_rerun_count}`",
        f"- Human-review route count: `{_optional_int_text(summary.human_review_route_count)}`",
        f"- Interrupt count: `{_optional_int_text(summary.interrupt_count)}`",
        f"- Resume count: `{_optional_int_text(summary.resume_count)}`",
        f"- Disposition: `{summary.disposition or 'n/a'}`",
        "",
    ]
    if summary.node_execution_counts is not None:
        lines.append("#### Node Execution Counts")
        lines.append("")
        for node_name, count in summary.node_execution_counts:
            lines.append(f"- `{node_name}`: `{count}`")
        lines.append("")
    if summary.route_counts is not None:
        lines.append("#### Route Counts")
        lines.append("")
        for route, count in summary.route_counts:
            lines.append(f"- `{route}`: `{count}`")
        lines.append("")
    return lines


def _summary_text(summary: NumericSummary) -> str:
    if summary.mean is None:
        return "n/a"
    return f"mean={summary.mean:.3f}, range={summary.minimum:.3f}..{summary.maximum:.3f}"


def _optional_int_text(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def _build_sub_benchmark_passed(report: V07OrchestrationSubBenchmarkReport) -> bool:
    return (
        report.targeted_domain_selection_correct_count == report.scenario_count
        and report.untouched_specialists_preserved_count == report.scenario_count
        and report.stale_specialist_outcome_replaced_count == report.scenario_count
        and report.planning_state_revision_progression_count == report.scenario_count
        and report.graph_termination_count == report.scenario_count
        and report.loop_bound_handling_count >= 1
    )


def _retention_rule_passed(
    *,
    backend_summaries: tuple[V07BackendSummary, ...],
    orchestration_sub_benchmark: V07OrchestrationSubBenchmarkReport,
    human_review_sub_benchmark: V07HumanReviewSubBenchmarkReport,
) -> bool:
    imperative = next(
        (summary for summary in backend_summaries if summary.backend == "imperative"), None
    )
    langgraph = next(
        (summary for summary in backend_summaries if summary.backend == "langgraph"), None
    )
    if imperative is None or langgraph is None:
        return False
    if not orchestration_sub_benchmark.passed or not human_review_sub_benchmark.passed:
        return False
    return langgraph.disposition in {"RETAIN", "RETAIN_EXPERIMENTALLY"}


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "GitSnapshot",
    "GraphTraceEvent",
    "GraphTraceEventKind",
    "NumericSummary",
    "V07BackendSummary",
    "V07ControlledEvaluationReport",
    "V07Environment",
    "V07EvaluationProvenance",
    "V07HumanReviewSubBenchmarkReport",
    "V07OrchestrationSubBenchmarkReport",
    "V07RunMetrics",
    "V07RunOrderBlocks",
    "V07RunReport",
    "V07ScenarioResult",
    "default_output_dir",
    "load_v07_controlled_scenarios",
    "render_v07_controlled_evaluation_markdown",
    "render_v07_controlled_run_markdown",
    "render_v07_human_review_sub_benchmark_markdown",
    "render_v07_orchestration_sub_benchmark_markdown",
    "run_v07_controlled_evaluation",
    "save_v07_controlled_evaluation_reports",
    "save_v07_controlled_run_report",
]
