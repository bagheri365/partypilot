from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from partypilot.application.v06_langchain_controlled_evaluation import (
    V06_RUN_ORDER_BLOCKS,
)
from partypilot.cli import eval_v06_langchain
from partypilot.composition.multi_agent_runtime import SpecialistAdapterKind


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
        specialist_success_rate=SimpleNamespace(mean=1.0),
        provider_timeout_count=0,
        structured_output_validation_failure_count=0,
        disposition="BASELINE" if variant == "native_ollama" else "RETAIN_EXPERIMENTALLY",
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
