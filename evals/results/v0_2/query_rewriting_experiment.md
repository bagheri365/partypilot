# v0.2 Query Rewriting Experiment

## Predeclared decision rule

Retain rewriting only if it improves MRR by at least 0.020 or improves correct-policy retrieval, keeps query drift <= 0.050, adds no more than 2.000 ms mean latency, and stays within $0.0000 model cost for this offline experiment.

High-value lexical signals are protected by an append-only rewrite: the original query remains verbatim and expansions are appended.

| Variant | Recall@k | MRR | Correct policy | Query drift | Mean latency (ms) | Rewritten | Model tokens | Model cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct_bm25 | 1.000 | 0.857 | 1.000 | 0.000 | 0.068 | 0/7 | 0 | $0.0000 |
| always_on_rewriting | 1.000 | 0.929 | 1.000 | 0.000 | 0.067 | 7/7 | 0 | $0.0000 |
| conditional_rewriting | 1.000 | 0.929 | 1.000 | 0.000 | 0.057 | 3/7 | 0 | $0.0000 |

## Grounded-decision accuracy

Not yet available: Prompt 43 evidence-grounded planning has not been implemented, so this experiment does not fabricate a downstream decision metric.

## Decision

**retain_conditional_rewriting** — A rewriting variant met the predeclared quality and safety/cost thresholds.

The rewrite variants are deterministic and make no model calls, so measured model tokens and model cost are zero. This does not estimate the cost of an LLM rewriter.
