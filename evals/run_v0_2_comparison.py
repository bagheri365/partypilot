"""Controlled v0.2 comparison between plain BM25 and conditional rewriting."""

from __future__ import annotations

import json
from pathlib import Path

from partypilot.adapters import (
    BM25EvidenceRetriever,
    InMemoryResourceStore,
    LLMConstraintExtractor,
    OllamaAdapter,
    OllamaConfig,
    UrllibHttpTransport,
)
from partypilot.adapters.query_rewriting_retriever import (
    ConditionalQueryRewritingEvidenceRetriever,
)
from partypilot.application.evidence_grounded_planner import EvidenceGroundedPlanner
from partypilot.application.query_rewriting_experiment import (
    ConditionalQueryRewriter,
    LexicalSignalPreservingRewriter,
)
from partypilot.application.v02_query_rewriting_comparison import (
    run_v02_query_rewriting_comparison,
    save_v02_query_rewriting_comparison_reports,
)
from partypilot.domain.evaluation import EvaluationScenario
from partypilot.domain.evidence_corpus import EvidenceDocument


def _ollama_config() -> OllamaConfig:
    try:
        return OllamaConfig.from_env()
    except ValueError as exc:
        raise ValueError(
            "PARTYPILOT_OLLAMA_MODEL is required for the live v0.2 comparison."
        ) from exc


def _load_documents(path: Path) -> tuple[EvidenceDocument, ...]:
    return tuple(
        EvidenceDocument.model_validate(item)
        for item in json.loads(path.read_text(encoding="utf-8"))
    )


def _load_scenarios(path: Path) -> tuple[EvaluationScenario, ...]:
    return tuple(
        EvaluationScenario.model_validate(item)
        for item in json.loads(path.read_text(encoding="utf-8"))
    )


def _build_live_constraint_extractor() -> LLMConstraintExtractor:
    config = _ollama_config()
    return LLMConstraintExtractor(OllamaAdapter(config, UrllibHttpTransport()))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    corpus = _load_documents(root / "data/evidence/v0_2_documents.json")
    scenarios = _load_scenarios(root / "data/evaluation/core_scenarios.json")
    extractor = _build_live_constraint_extractor()
    base_retriever = BM25EvidenceRetriever(corpus)
    baseline_planner = EvidenceGroundedPlanner(
        resource_store=InMemoryResourceStore(),
        evidence_retriever=base_retriever,
        constraint_extractor=extractor,
    )
    conditional_planner = EvidenceGroundedPlanner(
        resource_store=InMemoryResourceStore(),
        evidence_retriever=ConditionalQueryRewritingEvidenceRetriever(
            base_retriever,
            ConditionalQueryRewriter(LexicalSignalPreservingRewriter()),
        ),
        constraint_extractor=extractor,
    )
    report = run_v02_query_rewriting_comparison(
        baseline_planner=baseline_planner,
        conditional_planner=conditional_planner,
        corpus=corpus,
        scenarios=scenarios,
        top_k=5,
    )
    json_path, md_path = save_v02_query_rewriting_comparison_reports(
        report,
        root / "evals/results/v0_2",
    )
    print(json_path)
    print(md_path)
    print(report.decision)
    print(report.decision_explanation)


if __name__ == "__main__":
    main()
