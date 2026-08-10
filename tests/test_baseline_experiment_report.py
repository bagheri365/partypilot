from pathlib import Path

REPORT_PATH = Path("docs/experiments/001_baselines.md")


def test_baseline_experiment_report_contains_required_sections() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")

    required_sections = (
        "## Question",
        "## Hypothesis",
        "## Compared variants",
        "## Metrics",
        "## Predeclared decision criteria",
        "## Measured results",
        "## Failure analysis",
        "## Conclusion",
        "## Next architectural question",
    )

    for section in required_sections:
        assert section in report


def test_report_does_not_claim_unmeasured_llm_win() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")

    assert "winner **cannot be declared**" in report
    assert "fake-provider outputs are not reported as model benchmark results" in report
    assert "`v0.1-baselines` is **not tagged**" in report
