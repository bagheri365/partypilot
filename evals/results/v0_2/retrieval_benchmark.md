# PartyPilot v0.2 retrieval benchmark

Embedding backend: `deterministic_hash_embedding_256d`

This report compares retrieval variants only. It does not select a retained architecture.

| Variant | Recall@k | Precision@k | MRR | Correct policy | Correct version | Wrong vendor | Mean latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 1.000 | 0.200 | 0.857 | 1.000 | 1.000 | 0.286 | 0.055 |
| semantic | 1.000 | 0.200 | 0.726 | 1.000 | 1.000 | 0.314 | 0.584 |
| bm25_semantic_rrf | 1.000 | 0.200 | 0.833 | 1.000 | 1.000 | 0.314 | 0.773 |

## Notes

- Correct-policy retrieval means at least one labeled relevant document was returned.
- Correct-version retrieval additionally requires the labeled version and lifecycle status.
- Wrong-vendor rate is measured over all returned results; benchmark queries do not hard-filter by vendor.
- Latency is measured in the local execution environment and is not a production performance claim.
- The deterministic hash embedding backend is an offline reproducibility fixture, not a claim about a production embedding model.
