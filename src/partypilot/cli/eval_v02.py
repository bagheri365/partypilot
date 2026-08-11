"""Canonical PartyPilot v0.2 release evaluation CLI."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from partypilot.adapters import (
    BM25EvidenceRetriever,
    InMemoryResourceStore,
    LLMConstraintExtractor,
    OllamaAdapter,
    OllamaConfig,
    UrllibHttpTransport,
)
from partypilot.application.evidence_grounded_planner import EvidenceGroundedPlanner
from partypilot.application.v02_evaluation import V02EvaluationRunner, save_v02_evaluation_reports
from partypilot.application.v02_release import (
    build_release_metadata,
    build_v02_evaluation_report,
    default_output_dir,
    load_documents,
    load_scenarios,
)
from partypilot.cli.eval_baseline import _ollama_config
from partypilot.domain.evaluation import DatasetSplit
from partypilot.domain.evidence_corpus import EvidenceDocument
from partypilot.ports.llm_provider import LLMProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the canonical PartyPilot v0.2 evidence-grounded evaluation."
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
    parser.add_argument("--base-url", default=None, help="Override PARTYPILOT_OLLAMA_BASE_URL.")
    parser.add_argument("--model", default=None, help="Override PARTYPILOT_OLLAMA_MODEL.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Override PARTYPILOT_OLLAMA_TIMEOUT_SECONDS.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Override PARTYPILOT_OLLAMA_MAX_RETRIES.",
    )
    return parser


def build_live_constraint_extractor(
    *,
    provider: LLMProvider | None = None,
    config: OllamaConfig | None = None,
    transport: UrllibHttpTransport | None = None,
) -> LLMConstraintExtractor:
    if provider is None:
        resolved_config = config or _ollama_config(
            model=None,
            base_url=None,
            timeout_seconds=None,
            max_retries=None,
        )
        resolved_transport = transport or UrllibHttpTransport()
        provider = OllamaAdapter(resolved_config, resolved_transport)
    return LLMConstraintExtractor(provider)


def build_v02_planner(
    *,
    corpus: tuple[EvidenceDocument, ...],
    provider: LLMProvider | None = None,
    config: OllamaConfig | None = None,
    transport: UrllibHttpTransport | None = None,
) -> EvidenceGroundedPlanner:
    return EvidenceGroundedPlanner(
        resource_store=InMemoryResourceStore(),
        evidence_retriever=BM25EvidenceRetriever(corpus),
        constraint_extractor=build_live_constraint_extractor(
            provider=provider,
            config=config,
            transport=transport,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    split = DatasetSplit(args.split)
    root = Path(__file__).resolve().parents[3]
    timestamp = datetime.now(UTC)

    try:
        config = _ollama_config(
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )
        corpus = load_documents(root / "data/evidence/v0_2_documents.json")
        scenarios = load_scenarios(split, root / "data/evaluation/core_scenarios.json")
        planner = build_v02_planner(corpus=corpus, config=config)
        metrics, scenario_results = V02EvaluationRunner(planner, corpus=corpus).run(scenarios)
        report = build_v02_evaluation_report(
            root=root,
            metrics=metrics,
            scenario_results=scenario_results,
            metadata=build_release_metadata(
                split=split,
                model_name=config.model,
                timestamp=timestamp,
            ),
        )
        output_dir = args.output_dir or default_output_dir(split, timestamp)
        json_path, md_path = save_v02_evaluation_reports(report, output_dir)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: live v0.2 evaluation failed. Details: {exc}", file=sys.stderr)
        return 1

    print("# PartyPilot v0.2 Evaluation")
    print(f"Split: {split.value}")
    print(f"Model: {config.model}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print(f"Decision variant: {report.evaluation_variant}")
    print(f"Feasibility accuracy: {report.metrics.feasibility_accuracy:.3f}")
    print(f"Grounded-decision accuracy: {report.metrics.grounded_decision_accuracy:.3f}")
    print(f"Source-attribution accuracy: {report.metrics.source_attribution_accuracy:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
