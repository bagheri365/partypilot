# ADR 001: Retain BM25 as the v0.2 retrieval architecture

- Status: Accepted for v0.2 evidence-grounded constraint work
- Date: 2026-08-10
- Benchmark: `evals/results/v0_2/retrieval_benchmark.json`

## Context

PartyPilot needs evidence retrieval that can find the correct policy, preserve source/version metadata, and avoid pulling evidence from the wrong vendor. Prompt 35 compared three experimental variants on the current retrieval benchmark: BM25, semantic retrieval, and BM25 + semantic with Reciprocal Rank Fusion (RRF).

The benchmark contains 7 human-authored retrieval labels. At `k=5`, all three variants achieved Recall@5 = 1.000, Precision@5 = 0.200, correct-policy retrieval = 1.000, and correct-version retrieval = 1.000. The variants differed in ranking quality, wrong-vendor retrieval, and latency:

| Variant | MRR | Wrong-vendor rate | Mean latency |
|---|---:|---:|---:|
| BM25 | 0.857 | 0.286 | 0.055 ms |
| Semantic | 0.726 | 0.314 | 0.584 ms |
| BM25 + semantic + RRF | 0.833 | 0.314 | 0.773 ms |

The semantic benchmark used the deterministic hash embedding backend (`deterministic_hash_embedding_256d`) for offline reproducibility. It is not evidence about the quality or cost of a production embedding model.

## Decision

Retain **BM25 only** as the v0.2 retrieval architecture for the next evidence-grounded planning experiments.

BM25 is retained because, on the measured benchmark, it matched the other variants on Recall@5, Precision@5, correct-policy retrieval, and correct-version retrieval while achieving the best MRR, the lowest wrong-vendor retrieval rate, and the lowest measured latency. The hybrid variant added semantic retrieval and fusion complexity without improving any measured retrieval-quality metric over BM25.

This is a milestone-level decision, not a permanent claim that lexical retrieval is universally superior. The semantic and RRF adapters remain experimental implementations and may be reevaluated when the benchmark is broader or when a production-grade embedding backend is measured.

## Alternatives considered

### Semantic retrieval only

Rejected for the retained v0.2 path. On this benchmark it had lower MRR (0.726 vs. 0.857), a higher wrong-vendor rate (0.314 vs. 0.286), and higher latency (0.584 ms vs. 0.055 ms) than BM25, with no gain in Recall@5, Precision@5, correct-policy retrieval, or correct-version retrieval.

The result is limited by the deterministic hash embedding fixture. A future experiment using a production embedding model could justify reconsideration, but that evidence does not exist yet.

### BM25 + semantic + RRF

Rejected for the retained v0.2 path. The hybrid variant reached MRR 0.833, below BM25's 0.857, and had a higher wrong-vendor rate (0.314 vs. 0.286). Its mean latency was also the highest measured at 0.773 ms. The added fusion path therefore has no measured benefit that justifies its extra moving parts for this milestone.

### Another configuration

Not selected. The current benchmark is small, and no additional configuration has measured evidence strong enough to replace the simplest best-performing measured variant.

## Latency trade-offs

The measured local latency values are not production performance claims, but their relative ordering is useful for this experiment. BM25 was roughly 10.6x faster than the semantic fixture and roughly 14.1x faster than the hybrid variant in this run. A real embedding service would also introduce external-call latency and operational cost that are not represented by the deterministic fixture.

Because BM25 already achieved full Recall@5 and correct-version retrieval on the current labels, paying that additional complexity/latency has no demonstrated value yet.

## Known weaknesses

- The retrieval benchmark currently contains only 7 labeled queries, so the decision is based on a small sample.
- Precision@5 is 0.200 for every variant, meaning the top-5 lists still contain many non-ground-truth documents.
- BM25 depends on lexical overlap and may miss paraphrases or conceptually related evidence that shares little terminology with the query.
- Wrong-vendor retrieval is still non-zero (0.286) when vendor filtering is not applied.
- Version correctness was perfect on the current labels, but the corpus is intentionally small and controlled.
- The benchmark does not yet measure downstream grounded-decision accuracy; later prompts should verify that retrieval quality translates into correct constraint extraction and planning decisions.
- The semantic result should not be generalized to production embeddings because the benchmark used a deterministic hash embedding fixture.

## Consequences

- The evidence-grounded planning path should use BM25 as the retained retriever unless a later measured experiment explicitly changes this ADR.
- Resource/vendor metadata filtering should be applied whenever the resource is known because wrong-vendor retrieval remains a measurable failure mode without it.
- Semantic and RRF implementations remain available for experiments but are not default composition dependencies.
- Query rewriting and reranking experiments should compare against retained BM25, not against the hybrid variant by default.
