from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from partypilot.application.v06_langchain_controlled_evaluation import (
    V06_RUN_ORDER_BLOCKS,
)
from partypilot.cli import eval_v06_langchain
from partypilot.cli import v06_langchain_controlled_evaluation_core as core
from partypilot.composition.multi_agent_runtime import SpecialistAdapterKind
from partypilot.domain import SpecialistFailureKind


def test_run_order_blocks_are_balanced() -> None:
    assert V06_RUN_ORDER_BLOCKS == (
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


def _fake_summary(variant: str) -> SimpleNamespace:
    return SimpleNamespace(
        variant=variant,
        run_count=3,
        final_decision_accuracy=SimpleNamespace(mean=1.0),
        evidence_grounded_arbitration_accuracy=SimpleNamespace(mean=1.0),
        top_level_specialist_invocations=150,
        successful_top_level_specialist_invocations=150,
        specialist_success_rate=SimpleNamespace(mean=1.0),
        specialist_timeout_outcomes=0,
        specialist_timeout_outcome_rate=0.0,
        provider_timeout_count=0,
        provider_timeout_rate=0.0,
        provider_connection_failure_count=0,
        provider_response_failure_count=0,
        structured_output_validation_failure_count=0,
        specialist_domain_validation_failure_count=0,
        provider_attempt_count=150,
        provider_attempt_rate=1.0,
        candidate_specialist_invocations=150,
        candidate_provider_timeout_count=0,
        candidate_provider_timeout_rate=0.0,
        candidate_retry_count=0,
        candidate_retry_rate=0.0,
        disposition="BASELINE" if variant == "native_ollama" else "RETAIN_EXPERIMENTALLY",
    )


def _make_outcome(
    *,
    specialist_id: str,
    domain: str = "venue",
    validation_succeeded: bool = True,
    failure_kind: SpecialistFailureKind | None = None,
    retry_count: int = 0,
    latency_ms: float = 10.0,
    tool_call_count: int = 0,
    tool_call_success_count: int = 0,
    tool_call_failure_count: int = 0,
    decision: object | None = object(),
) -> SimpleNamespace:
    return SimpleNamespace(
        trace=SimpleNamespace(
            specialist_id=specialist_id,
            domain=SimpleNamespace(value=domain),
            validation_succeeded=validation_succeeded,
            failure_kind=failure_kind,
            retry_count=retry_count,
            latency_ms=latency_ms,
            tool_call_count=tool_call_count,
            tool_call_success_count=tool_call_success_count,
            tool_call_failure_count=tool_call_failure_count,
            agent_execution_limit_hit=False,
        ),
        decision=decision,
        failure_kind=failure_kind,
    )


def test_cli_orchestrates_three_way_evaluation_and_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_config = SimpleNamespace(
        base_url="http://localhost:11434",
        model="fake-model",
        timeout_seconds=30.0,
        num_ctx=8192,
        max_retries=0,
    )
    fake_report = SimpleNamespace(
        benchmark_name="PartyPilot v0.6d three-way LangChain controlled evaluation",
        benchmark_version="1.0",
        scenario_count=10,
        run_order_blocks=(
            ("native_ollama", "langchain_chatollama", "langchain_agent"),
            ("langchain_chatollama", "langchain_agent", "native_ollama"),
            ("langchain_agent", "native_ollama", "langchain_chatollama"),
        ),
        variant_summaries=(
            _fake_summary("native_ollama"),
            _fake_summary("langchain_chatollama"),
            _fake_summary("langchain_agent"),
        ),
    )
    monkeypatch.setattr(eval_v06_langchain, "_ollama_config", lambda **kwargs: fake_config)
    monkeypatch.setattr(
        eval_v06_langchain,
        "load_v06_controlled_scenarios",
        lambda scenario_ids: tuple(f"scenario-{index}" for index in range(10)),
    )
    monkeypatch.setattr(
        eval_v06_langchain,
        "run_v06_controlled_evaluation",
        lambda *args, **kwargs: fake_report,
    )
    monkeypatch.setattr(
        eval_v06_langchain,
        "save_v06_controlled_run_reports",
        lambda report, output_dir: (
            output_dir / "aggregate.json",
            output_dir / "aggregate.md",
            tuple(
                (
                    output_dir / variant / f"run-{index}.json",
                    output_dir / variant / f"run-{index}.md",
                )
                for index, variant in enumerate(
                    ["native_ollama", "langchain_chatollama", "langchain_agent"] * 3,
                    start=1,
                )
            ),
        ),
    )
    monkeypatch.setattr(
        eval_v06_langchain,
        "default_output_dir",
        lambda timestamp: tmp_path / "v0_6" / "langchain",
    )

    exit_code = eval_v06_langchain.main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PartyPilot v0.6d Three-Way LangChain Controlled Evaluation" in captured.out
    assert "Run artifacts: 9" in captured.out
    assert "Variant: native_ollama" in captured.out
    assert "Disposition: RETAIN_EXPERIMENTALLY" in captured.out


def test_canonical_start_guard_rejects_dirty_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        core,
        "_git_snapshot",
        lambda: core.GitSnapshot(
            git_sha="abc123", working_tree_dirty=True, git_metadata_error=None
        ),
    )

    with pytest.raises(ValueError, match="clean working tree"):
        core.run_v06_controlled_evaluation(
            (),
            model="fake-model",
            base_url="http://localhost:11434",
            timeout_seconds=30.0,
            num_ctx=8192,
            max_retries=0,
        )


def test_summarize_variant_uses_selected_candidate_for_top_level_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_outcomes = tuple(
        _make_outcome(
            specialist_id=f"selected-{index}",
            domain=f"domain-{index % 5}",
            validation_succeeded=True,
            latency_ms=100.0 + index,
            tool_call_count=1,
            tool_call_success_count=1,
        )
        for index in range(150)
    )
    candidate_outcomes = selected_outcomes + tuple(
        _make_outcome(
            specialist_id=f"candidate-{index}",
            domain=f"alt-domain-{index % 5}",
            validation_succeeded=False,
            failure_kind=SpecialistFailureKind.PROVIDER_TIMEOUT,
            latency_ms=200.0 + index,
            tool_call_count=2,
            tool_call_success_count=1,
            tool_call_failure_count=1,
        )
        for index in range(150)
    )
    runtime = SimpleNamespace(
        final_result=SimpleNamespace(selected_resource_ids=("resource-match",)),
        candidate_results=(
            SimpleNamespace(
                candidate_resource_ids=("resource-match",),
                specialist_outcomes=selected_outcomes,
            ),
            SimpleNamespace(
                candidate_resource_ids=("resource-other",),
                specialist_outcomes=tuple(candidate_outcomes[150:]),
            ),
        ),
    )
    live_report = SimpleNamespace(
        metrics=SimpleNamespace(
            live=SimpleNamespace(
                final_decision_accuracy=1.0,
                evidence_grounded_arbitration_accuracy=1.0,
                hard_constraint_validity=1.0,
                cross_domain_compatibility_accuracy=1.0,
                global_optimum_accuracy=1.0,
                human_review_calibration=1.0,
            ),
            runtime=SimpleNamespace(mean_latency_ms=42.0),
        ),
        scenarios=(
            SimpleNamespace(
                runtime=runtime,
                scenario_id="scenario-1",
                live_result=SimpleNamespace(
                    feasibility_outcome=SimpleNamespace(value="NO_FEASIBLE_PLAN")
                ),
            ),
        ),
    )
    variant_runs: tuple[Any, ...] = (
        SimpleNamespace(report=live_report, scenario_count=1),
        SimpleNamespace(report=live_report, scenario_count=1),
        SimpleNamespace(report=live_report, scenario_count=1),
    )

    monkeypatch.setattr(core, "_variant_disposition", lambda **kwargs: "RETAIN")
    monkeypatch.setattr(
        core,
        "_terminal_stability",
        lambda variant_runs: {
            "rate": 1.0,
            "stable": ("scenario-1",),
            "unstable": (),
        },
    )

    summary = core._summarize_variant(
        variant="langchain_agent",
        variant_runs=cast(tuple[core.V06RunReport, ...], variant_runs),
    )

    assert summary.top_level_specialist_invocations == 450
    assert summary.successful_top_level_specialist_invocations == 450
    assert summary.provider_timeout_count == 0
    assert summary.candidate_specialist_invocations == 900
    assert summary.candidate_provider_timeout_count == 450
    assert summary.provider_attempt_count == 450
