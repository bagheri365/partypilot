# PartyPilot v0.2 Evidence-Grounded Evaluation

Evaluation variant: `bm25 + live_ollama_constraint_extractor`

## Planning and grounding metrics

- Scenarios: 24
- Feasibility accuracy: 0.875
- Hard-constraint validity: 1.000
- Grounded-decision accuracy: 1.000
- Source-attribution accuracy: 1.000
- Derived-constraint accuracy: 1.000
- Unsupported-claim rate: 0.000
- Wrong-source/version rate: 0.000
- No-feasible-plan accuracy: 0.769
- Mean latency: 43053.930 ms
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

- The v0.2 planning run uses the live Ollama-backed constraint extractor for evidence grounding.
- Token and model-cost metrics are not collected by this evaluation report and are not fabricated.
- The v0.1 single-pass LLM baseline has no comparable measured live-provider benchmark, so only the measured v0.1 deterministic baseline is included.
- Retrieval metrics remain a separate section and are not folded into planning accuracy.
