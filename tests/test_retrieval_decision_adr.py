from pathlib import Path

ADR_PATH = Path("docs/adr/001-retrieval-architecture.md")


def test_retrieval_adr_contains_required_sections_and_decision() -> None:
    adr = ADR_PATH.read_text(encoding="utf-8")

    required_sections = (
        "## Context",
        "## Decision",
        "## Alternatives considered",
        "## Latency trade-offs",
        "## Known weaknesses",
        "## Consequences",
    )
    for section in required_sections:
        assert section in adr

    assert "Retain **BM25 only**" in adr


def test_retrieval_adr_is_grounded_in_measured_benchmark() -> None:
    adr = ADR_PATH.read_text(encoding="utf-8")

    assert "MRR | Wrong-vendor rate | Mean latency" in adr
    assert "BM25 | 0.857 | 0.286 | 0.055 ms" in adr
    assert "Semantic | 0.726 | 0.314 | 0.584 ms" in adr
    assert "BM25 + semantic + RRF | 0.833 | 0.314 | 0.773 ms" in adr
    assert "7 human-authored retrieval labels" in adr


def test_retrieval_adr_does_not_overclaim_semantic_results() -> None:
    adr = ADR_PATH.read_text(encoding="utf-8")

    assert "deterministic hash embedding" in adr
    assert "not evidence about the quality or cost of a production embedding model" in adr
    assert "not a permanent claim" in adr
