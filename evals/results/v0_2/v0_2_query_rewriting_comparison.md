# PartyPilot v0.2 Query Rewriting Comparison

Retained retriever: `bm25`
Top-k: **5**

## Decision

**reject_conditional_rewriting** — Conditional rewriting did not improve any meaningful downstream metric, so plain BM25 is preferred.

## Planning and grounding metrics

| Variant | Feasibility | Hard validity | Grounded | Source attribution | Derived | Unsupported claim | Wrong source/version | No-feasible-plan | Mean latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 + live_ollama_constraint_extractor | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.769 | 47413.961 |
| bm25 + conditional_query_rewriting + live_ollama_constraint_extractor | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.769 | 48724.904 |

## Evidence-labeled scenarios

| Scenario | Expected | Expected docs | BM25 predicted | BM25 attributed docs | BM25 grounded | Conditional predicted | Conditional attributed docs | Conditional grounded |
|---|---|---|---|---|---|---|---|---|
| dev-accessible-07 | FEASIBLE | doc-loft-accessibility-current | FEASIBLE | doc-craft-accessibility, doc-craft-supervision-current, doc-family-allergen-current, doc-family-cancellation, doc-family-gluten-current, doc-family-vegan-current, doc-loft-accessibility-conflict, doc-loft-accessibility-current, doc-loft-cancellation-current, doc-loft-outside-food-current, doc-loft-venue-current | yes | FEASIBLE | doc-craft-accessibility, doc-craft-safety-current, doc-craft-supervision-current, doc-family-allergen-current, doc-family-cancellation, doc-family-gluten-current, doc-family-vegan-current, doc-loft-accessibility-conflict, doc-loft-accessibility-current, doc-loft-cancellation-current, doc-loft-outside-food-current, doc-loft-venue-current | yes |
| dev-allergy-evidence-10 | HUMAN_REVIEW_REQUIRED | doc-family-allergen-current | HUMAN_REVIEW_REQUIRED | doc-craft-accessibility, doc-craft-supervision-current, doc-family-allergen-current, doc-family-cancellation, doc-family-gluten-current, doc-family-vegan-current, doc-loft-accessibility-conflict, doc-loft-accessibility-current, doc-loft-cancellation-current, doc-loft-outside-food-current, doc-loft-venue-current | yes | HUMAN_REVIEW_REQUIRED | doc-craft-accessibility, doc-craft-supervision-current, doc-family-allergen-current, doc-family-cancellation, doc-family-gluten-current, doc-family-vegan-current, doc-loft-accessibility-conflict, doc-loft-accessibility-current, doc-loft-cancellation-current, doc-loft-outside-food-current, doc-loft-venue-current | yes |
| frozen-accessibility-15 | FEASIBLE | doc-loft-accessibility-current | FEASIBLE | doc-craft-accessibility, doc-craft-safety-current, doc-craft-supervision-current, doc-family-allergen-current, doc-family-cancellation, doc-family-gluten-current, doc-family-vegan-current, doc-loft-accessibility-conflict, doc-loft-accessibility-current, doc-loft-cancellation-current, doc-loft-outside-food-current, doc-loft-venue-current | yes | FEASIBLE | doc-craft-accessibility, doc-craft-safety-current, doc-craft-supervision-current, doc-family-allergen-current, doc-family-cancellation, doc-family-gluten-current, doc-family-vegan-current, doc-loft-accessibility-conflict, doc-loft-accessibility-current, doc-loft-cancellation-current, doc-loft-outside-food-current, doc-loft-venue-current | yes |
| frozen-dietary-evidence-18 | HUMAN_REVIEW_REQUIRED | doc-family-gluten-current | HUMAN_REVIEW_REQUIRED | doc-craft-accessibility, doc-craft-supervision-current, doc-family-allergen-current, doc-family-cancellation, doc-family-gluten-current, doc-family-vegan-current, doc-loft-accessibility-conflict, doc-loft-accessibility-current, doc-loft-cancellation-current, doc-loft-outside-food-current, doc-loft-venue-current | yes | HUMAN_REVIEW_REQUIRED | doc-craft-accessibility, doc-craft-supervision-current, doc-family-allergen-current, doc-family-cancellation, doc-family-gluten-current, doc-family-vegan-current, doc-loft-accessibility-conflict, doc-loft-accessibility-current, doc-loft-cancellation-current, doc-loft-outside-food-current, doc-loft-venue-current | yes |
| adv-complex-safety-24 | HUMAN_REVIEW_REQUIRED | doc-family-allergen-current, doc-family-vegan-current, doc-loft-accessibility-current | HUMAN_REVIEW_REQUIRED | doc-craft-accessibility, doc-craft-supervision-current, doc-family-allergen-current, doc-family-cancellation, doc-family-gluten-current, doc-family-vegan-current, doc-loft-accessibility-conflict, doc-loft-accessibility-current, doc-loft-cancellation-current, doc-loft-outside-food-current, doc-loft-venue-current | yes | HUMAN_REVIEW_REQUIRED | doc-craft-accessibility, doc-craft-supervision-current, doc-family-allergen-current, doc-family-cancellation, doc-family-gluten-current, doc-family-vegan-current, doc-loft-accessibility-conflict, doc-loft-accessibility-current, doc-loft-cancellation-current, doc-loft-outside-food-current, doc-loft-venue-current | yes |

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

- Both variants use the same corpus, planner semantics, citation validation, and live constraint extractor; only the evidence-retrieval query text differs.
- The conditional variant uses the retained query rewriter only through an EvidenceRetriever decorator.
- Conditional rewriting is retained only if it improves a meaningful downstream metric without degrading hard-constraint validity, unsupported-claim rate, or wrong-source/version rate.
- If downstream metrics are identical, plain BM25 is preferred.
