# PartyPilot v0.2 Evidence-Grounded Evaluation

Evaluation variant: `bm25 + live_ollama_constraint_extractor`

## Reproducibility metadata

- Experiment ID: v0.2-evidence-grounded-development-20260811T032747Z
- Evaluation split: development
- Timestamp: 2026-08-11T03:27:47.749074+00:00
- Commit SHA: b10056171fa0120c15892e529a956df46e2f19a8
- Working tree dirty: False
- Git metadata error: none
- Model provider: ollama
- Model name: qwen3:4b-instruct-2507-q4_K_M
- Architecture variant: bm25_plus_live_ollama_constraint_extractor

## Planning and grounding metrics

- Scenarios: 10
- Feasibility accuracy: 0.900
- Hard-constraint validity: 1.000
- Grounded-decision accuracy: 1.000
- Source-attribution accuracy: 1.000
- Derived-constraint accuracy: 1.000
- Unsupported-claim rate: 0.000
- Wrong-source/version rate: 0.000
- No-feasible-plan accuracy: 0.800
- Mean latency: 36464.278 ms
- Tokens: n/a
- Estimated model cost: n/a

## v0.1 measured baseline comparison

### v0.1 deterministic baseline
- Feasibility accuracy: 0.875
- Hard-constraint validity: 1.000
- No-feasible-plan accuracy: 0.923
- Mean latency: 0.059 ms

## Retrieval metrics (separate)

### bm25
- Recall@5: 1.000
- Precision@5: 0.200
- MRR: 0.857
- Correct-policy retrieval: 1.000
- Correct-version retrieval: 1.000
- Wrong-vendor retrieval rate: 0.286
- Mean retrieval latency: 0.055 ms

### semantic
- Recall@5: 1.000
- Precision@5: 0.200
- MRR: 0.726
- Correct-policy retrieval: 1.000
- Correct-version retrieval: 1.000
- Wrong-vendor retrieval rate: 0.314
- Mean retrieval latency: 0.584 ms

### bm25_semantic_rrf
- Recall@5: 1.000
- Precision@5: 0.200
- MRR: 0.833
- Correct-policy retrieval: 1.000
- Correct-version retrieval: 1.000
- Wrong-vendor retrieval rate: 0.314
- Mean retrieval latency: 0.773 ms

## Notes

- The retained v0.2 runtime uses plain BM25, a live Ollama-backed constraint extractor, deterministic request-specific interpretation, and explicit citation validation.
- Token and model-cost metrics are not collected by this evaluation report and are not fabricated.
- The conditional query-rewriting comparison is preserved separately as an experiment artifact and is not part of the retained runtime.
- The controlled live comparison showed identical downstream metrics for conditional rewriting, so the simpler BM25 runtime was retained.
