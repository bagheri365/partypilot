from __future__ import annotations

import subprocess
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from partypilot.application.single_pass_llm_planner import (
    SinglePassLLMPlanner,
    SinglePassPlannerProviderError,
)
from partypilot.cli import eval_baseline
from partypilot.domain.evaluation import (
    ComplexityMetadata,
    DatasetSplit,
    EvaluationScenario,
    ScenarioCategory,
)
from partypilot.domain.feasibility import FeasibilityOutcome
from partypilot.domain.party_request import PartyRequest
from partypilot.ports.llm_provider import FakeLLMProvider, GenerationResponse


def _scenario() -> EvaluationScenario:
    return EvaluationScenario(
        scenario_id="scenario-1",
        request=PartyRequest(
            location="Boston",
            event_date=date(2026, 9, 1),
            guest_count=10,
            total_budget=Decimal("500"),
        ),
        expected_feasibility=FeasibilityOutcome.FEASIBLE,
        scenario_category=ScenarioCategory.FEASIBLE,
        complexity=ComplexityMetadata(),
        dataset_split=DatasetSplit.DEVELOPMENT,
    )


def _planner() -> tuple[SinglePassLLMPlanner, str]:
    response = GenerationResponse(
        text="",
        structured_output={
            "resources": [
                {
                    "resource_id": "venue-1",
                    "name": "Venue One",
                    "location": "Boston",
                    "price": "100",
                    "capacity": 20,
                    "availability": [],
                    "age_restrictions": None,
                    "accessibility_attributes": [],
                    "category": "venue",
                }
            ],
            "claimed_total_cost": "100",
            "assumptions": [],
        },
    )
    return SinglePassLLMPlanner(FakeLLMProvider([response])), "fake-model"


def test_cli_runs_and_writes_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(eval_baseline, "load_scenarios", lambda split: (_scenario(),))
    monkeypatch.setattr(
        eval_baseline, "build_live_single_pass_planner", lambda **kwargs: _planner()
    )
    monkeypatch.setattr(
        eval_baseline,
        "_git_metadata",
        lambda: ("abc123", False, None),
    )

    exit_code = eval_baseline.main(["--split", "development", "--output-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PartyPilot v0.1 Baseline Comparison" in captured.out
    assert "Deterministic baseline" in captured.out
    assert "Single-pass LLM baseline" in captured.out
    assert "Median Latency (ms)" in captured.out
    assert "Failure summary:" in captured.out
    assert "Unsupported-claim: N/A" in captured.out
    assert "Commit SHA: abc123" in captured.out
    assert "Prompt version: single-pass-v1" in captured.out

    json_files = list(tmp_path.glob("*.json"))
    md_files = list(tmp_path.glob("*.md"))
    assert len(json_files) == 1
    assert len(md_files) == 1
    assert '"experiment_id"' in json_files[0].read_text(encoding="utf-8")
    assert "# PartyPilot v0.1 Baseline Experiment" in md_files[0].read_text(encoding="utf-8")


def test_cli_fails_cleanly_when_ollama_model_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("PARTYPILOT_OLLAMA_MODEL", raising=False)

    exit_code = eval_baseline.main(["--split", "development", "--output-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "PARTYPILOT_OLLAMA_MODEL is required" in captured.err


def test_cli_reports_provider_failure_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(eval_baseline, "load_scenarios", lambda split: (_scenario(),))

    def raise_provider_error(**kwargs: object) -> tuple[SinglePassLLMPlanner, str]:
        raise SinglePassPlannerProviderError("boom")

    monkeypatch.setattr(eval_baseline, "build_live_single_pass_planner", raise_provider_error)

    exit_code = eval_baseline.main(["--split", "development", "--output-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "live Ollama baseline failed" in captured.err
    assert "boom" in captured.err


def test_ollama_config_honors_environment_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PARTYPILOT_OLLAMA_MODEL", "env-model")
    monkeypatch.setenv("PARTYPILOT_OLLAMA_BASE_URL", "http://env.example:11434")
    monkeypatch.setenv("PARTYPILOT_OLLAMA_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("PARTYPILOT_OLLAMA_NUM_CTX", "4096")
    monkeypatch.setenv("PARTYPILOT_OLLAMA_MAX_RETRIES", "1")

    config = eval_baseline._ollama_config(
        model="override-model",
        base_url="http://override.example:11434/",
        timeout_seconds=12.5,
        num_ctx=2048,
        max_retries=4,
    )

    assert config.model == "override-model"
    assert config.base_url.startswith("http://override.example:11434")
    assert config.timeout_seconds == 12.5
    assert config.num_ctx == 2048
    assert config.max_retries == 4


def test_git_metadata_reads_repo_head_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class CompletedProcess:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def fake_run(*args: object, **kwargs: object) -> CompletedProcess:
        calls.append((args, kwargs))
        command = args[0]
        if command == ["git", "rev-parse", "HEAD"]:
            return CompletedProcess("deadbeef\n")
        if command == ["git", "status", "--porcelain"]:
            return CompletedProcess("")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert eval_baseline._git_metadata() == ("deadbeef", False, None)
    assert len(calls) == 2


def test_git_metadata_returns_explicit_error_when_git_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", fake_run)

    commit_sha, working_tree_dirty, git_metadata_error = eval_baseline._git_metadata()

    assert commit_sha is None
    assert working_tree_dirty is None
    assert git_metadata_error is not None
    assert "Git metadata unavailable" in git_metadata_error
