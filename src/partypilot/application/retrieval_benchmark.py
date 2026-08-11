from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from partypilot.domain.evaluation import EvaluationScenario, RetrievalGroundTruthLabel
from partypilot.ports.evidence_retriever import (
    EvidenceRetrievalQuery,
    EvidenceRetrievalResult,
    EvidenceRetriever,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


class RetrievalBenchmarkMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    recall_at_k: float = Field(ge=0.0, le=1.0)
    precision_at_k: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    correct_policy_retrieval: float = Field(ge=0.0, le=1.0)
    correct_version_retrieval: float = Field(ge=0.0, le=1.0)
    wrong_vendor_retrieval_rate: float = Field(ge=0.0, le=1.0)
    mean_latency_ms: float = Field(ge=0.0)


class RetrievalBenchmarkVariantResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    variant: str
    top_k: int = Field(gt=0)
    query_count: int = Field(ge=0)
    metrics: RetrievalBenchmarkMetrics


class RetrievalBenchmarkReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark_name: str
    embedding_backend: str
    variants: tuple[RetrievalBenchmarkVariantResult, ...]


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkCase:
    scenario_id: str
    query_text: str
    ground_truth: RetrievalGroundTruthLabel


class Clock(Protocol):
    def __call__(self) -> float: ...


class DeterministicHashEmbeddingProvider:
    """Offline deterministic embedding fixture for reproducible retrieval experiments.

    This is intentionally not presented as a production-quality semantic model. It hashes
    lexical tokens into a fixed-dimensional signed vector so the semantic adapter can be
    benchmarked without network access or a model dependency.
    """

    def __init__(self, *, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    def embed(
        self, texts: tuple[str, ...], *, timeout_seconds: float
    ) -> tuple[tuple[float, ...], ...]:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        return tuple(self._embed_one(text) for text in texts)

    def _embed_one(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self._dimensions
        for token in _TOKEN_RE.findall(text.casefold()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return tuple(vector)
        return tuple(value / norm for value in vector)


def build_retrieval_benchmark_cases(
    scenarios: Iterable[EvaluationScenario],
) -> tuple[RetrievalBenchmarkCase, ...]:
    cases: list[RetrievalBenchmarkCase] = []
    for scenario in scenarios:
        for label in scenario.retrieval_ground_truth:
            cases.append(
                RetrievalBenchmarkCase(
                    scenario_id=scenario.scenario_id,
                    query_text=_query_text_for_label(scenario, label),
                    ground_truth=label,
                )
            )
    return tuple(cases)


def _query_text_for_label(scenario: EvaluationScenario, label: RetrievalGroundTruthLabel) -> str:
    request = scenario.request
    terms: list[str] = []
    if request.allergies:
        terms.extend(request.allergies)
    if request.dietary_restrictions:
        terms.extend(request.dietary_restrictions)
    if request.accessibility_needs:
        terms.extend(request.accessibility_needs)
    if request.other_constraints:
        terms.extend(request.other_constraints)

    # Policy-type wording is benchmark metadata authored independently of system output.
    # Resource IDs are included as high-value lexical signals but not used as hard filters,
    # allowing wrong-vendor retrieval to remain measurable.
    terms.extend((label.resource_id, label.policy_type.value, "policy"))
    return " ".join(terms)


def evaluate_retriever(
    *,
    variant: str,
    retriever: EvidenceRetriever,
    cases: Sequence[RetrievalBenchmarkCase],
    top_k: int,
    clock: Clock = perf_counter,
) -> RetrievalBenchmarkVariantResult:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not cases:
        return RetrievalBenchmarkVariantResult(
            variant=variant,
            top_k=top_k,
            query_count=0,
            metrics=RetrievalBenchmarkMetrics(
                recall_at_k=0.0,
                precision_at_k=0.0,
                mrr=0.0,
                correct_policy_retrieval=0.0,
                correct_version_retrieval=0.0,
                wrong_vendor_retrieval_rate=0.0,
                mean_latency_ms=0.0,
            ),
        )

    recall_values: list[float] = []
    precision_values: list[float] = []
    reciprocal_ranks: list[float] = []
    policy_hits = 0
    version_hits = 0
    wrong_vendor_results = 0
    returned_results = 0
    latencies_ms: list[float] = []

    for case in cases:
        start = clock()
        results = retriever.retrieve(EvidenceRetrievalQuery(text=case.query_text, top_k=top_k))
        elapsed = clock() - start
        latencies_ms.append(elapsed * 1000.0)

        expected = set(case.ground_truth.expected_document_ids)
        retrieved_ids = [result.document_id for result in results]
        relevant_count = sum(document_id in expected for document_id in retrieved_ids)
        recall_values.append(relevant_count / len(expected))
        precision_values.append(relevant_count / top_k)

        first_relevant_rank = next(
            (
                index
                for index, document_id in enumerate(retrieved_ids, start=1)
                if document_id in expected
            ),
            None,
        )
        reciprocal_ranks.append(0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank)

        policy_hits += int(any(result.document_id in expected for result in results))
        version_hits += int(_has_correct_version(results, case.ground_truth))
        wrong_vendor_results += sum(
            result.resource_id != case.ground_truth.resource_id for result in results
        )
        returned_results += len(results)

    count = len(cases)
    return RetrievalBenchmarkVariantResult(
        variant=variant,
        top_k=top_k,
        query_count=count,
        metrics=RetrievalBenchmarkMetrics(
            recall_at_k=sum(recall_values) / count,
            precision_at_k=sum(precision_values) / count,
            mrr=sum(reciprocal_ranks) / count,
            correct_policy_retrieval=policy_hits / count,
            correct_version_retrieval=version_hits / count,
            wrong_vendor_retrieval_rate=(
                wrong_vendor_results / returned_results if returned_results else 0.0
            ),
            mean_latency_ms=sum(latencies_ms) / count,
        ),
    )


def _has_correct_version(
    results: Sequence[EvidenceRetrievalResult], label: RetrievalGroundTruthLabel
) -> bool:
    expected_ids = set(label.expected_document_ids)
    return any(
        result.document_id in expected_ids
        and result.resource_id == label.resource_id
        and result.version.version == label.expected_version
        and result.version.status == label.expected_status
        for result in results
    )


def write_retrieval_benchmark_reports(
    report: RetrievalBenchmarkReport,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: RetrievalBenchmarkReport) -> str:
    lines = [
        f"# {report.benchmark_name}",
        "",
        f"Embedding backend: `{report.embedding_backend}`",
        "",
        "This report compares retrieval variants only. It does not select a retained architecture.",
        "",
        (
            "| Variant | Recall@k | Precision@k | MRR | Correct policy | Correct version | "
            "Wrong vendor | Mean latency (ms) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in report.variants:
        metrics = variant.metrics
        lines.append(
            f"| {variant.variant} | {metrics.recall_at_k:.3f} | "
            f"{metrics.precision_at_k:.3f} | {metrics.mrr:.3f} | "
            f"{metrics.correct_policy_retrieval:.3f} | "
            f"{metrics.correct_version_retrieval:.3f} | "
            f"{metrics.wrong_vendor_retrieval_rate:.3f} | "
            f"{metrics.mean_latency_ms:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Correct-policy retrieval means at least one labeled relevant document was returned.",
            "- Correct-version retrieval additionally requires the labeled version and "
            "lifecycle status.",
            "- Wrong-vendor rate is measured over all returned results; benchmark queries do "
            "not hard-filter by vendor.",
            "- Latency is measured in the local execution environment and is not a production "
            "performance claim.",
            "- The deterministic hash embedding backend is an offline reproducibility "
            "fixture, not a claim about a production embedding model.",
            "",
        ]
    )
    return "\n".join(lines)
