# v0.2 Reranking Decision Experiment

## Prerequisite diagnostic

Reranking is tested only if correct evidence appears at rank 3 or lower in at least 20% of labeled queries.

Retained retrieval: `bm25_with_conditional_rewriting`
Labeled queries: **7**
Recall@5: **1.000**
MRR: **0.929**
Correct evidence at rank 1: **0.857**
Correct evidence at rank 3+: **0.000**
Missed relevant evidence: **0.000**
Mean retrieval latency: **0.061 ms**

## Decision

**reranking_not_justified** — The retained retriever does not commonly place correct evidence at rank 3 or lower, so Prompt 38 stops before adding a reranker.

## Reranker comparison

Not run. The prerequisite failure pattern was absent, so retrieval quality, downstream decision quality, reranker latency, and model cost cannot be truthfully reported for a reranker that was not justified or invoked.

No reranking model, provider call, or model cost was introduced by this experiment.
