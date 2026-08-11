# PartyPilot v0.3 Replanning Experiment

Benchmark name: `v0.3 replanning benchmark`
Benchmark version: `1.0`
Evaluation variant: `full_replan_vs_dependency_aware_targeted_replan`
Scenario count: **5**

## Reproducibility Metadata

- Git SHA: `0722006540fc4111546a2d111084709cc8e377dd`
- Working tree dirty: `False`
- Dataset version: `v0.3`
- Architecture variant: `stateful_decomposition_and_targeted_replanning`
- Evaluation split: `benchmark`

## Metric Definitions

- `invalidation_accuracy`: Fraction of required recomputations that are reflected in the final invalidated decision set.
- `preserved_decision_accuracy`: Fraction of benchmark-preserved decisions that remain preserved in the resulting state.
- `final_state_correctness`: Exact match between the expected final decision statuses and the strategy's resulting state, independent of recomputation volume.
- `recomputed_decision_count`: Total number of decisions recomputed by the strategy across scenarios.
- `unnecessary_recomputation_count`: Count of recomputed decisions beyond the minimal required set.
- `missed_recomputation_count`: Total count of decisions that should have been recomputed but were not.
- `recomputation_reduction_ratio`: Targeted replanning reduction relative to full replanning, computed from aggregate recomputed decision counts.
- `mean_latency_ms`: Mean wall-clock latency per scenario for the strategy.

## Aggregate Metrics

| Strategy | Invalidation Accuracy | Preserved Accuracy | Final-State Correctness | Recomputed Decisions | Unnecessary Recomputation | Missed Recomputation | Mean Latency (ms) | Cycle Detections |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_replan` | 1.000 | 1.000 | 1.000 | 34 | 16 | 0 | 0.034 | 0 |
| `targeted_replan` | 1.000 | 1.000 | 1.000 | 18 | 0 | 0 | 0.023 | 0 |

- Targeted-vs-full recomputation reduction ratio: 0.471
- Retention rule passed: True

## Per-Scenario Results

### cap-boundary-51-incremental-replanning
Incremental replanning after guest-count increase

- Capability tags: `replanning`, `incremental_update`, `guest_count`, `cross_domain`
- Expected invalidated decisions: `venue_capacity`, `catering_quantity`, `seating`, `parking`, `total_cost`
- Expected preserved decisions: `theme`, `dietary_policies`, `entertainment`, `accessibility_requirements`
- Failure stage: `none`

| Strategy | Invalidation Accuracy | Preserved Accuracy | Final-State Correctness | Recomputed Decisions | Unnecessary Recomputation | Missed Recomputation | Mean Latency (ms) | Cycle Detections |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_replan` | 1.000 | 1.000 | 1.000 | 9 | 4 | 0 | 0.062 | 0 |
| `targeted_replan` | 1.000 | 1.000 | 1.000 | 5 | 0 | 0 | 0.034 | 0 |

### cap-boundary-52-new-safety-constraint-after-planning
New safety constraint after planning

- Capability tags: `replanning`, `safety_update`, `allergy`, `cross_domain`
- Expected invalidated decisions: `catering_safety_conclusion`, `dietary_evidence_review`
- Expected preserved decisions: `venue_choice`, `theme`, `entertainment`, `accessibility`
- Failure stage: `none`

| Strategy | Invalidation Accuracy | Preserved Accuracy | Final-State Correctness | Recomputed Decisions | Unnecessary Recomputation | Missed Recomputation | Mean Latency (ms) | Cycle Detections |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_replan` | 1.000 | 1.000 | 1.000 | 6 | 4 | 0 | 0.028 | 0 |
| `targeted_replan` | 1.000 | 1.000 | 1.000 | 2 | 0 | 0 | 0.021 | 0 |

### cap-boundary-55-cascading-failure
Cascading failure after a rain-triggered schedule change

- Capability tags: `replanning`, `cascading_failure`, `temporal_dependency`, `cross_domain`
- Expected invalidated decisions: `rain_contingency`, `indoor_move`, `indoor_setup_space`, `staffing_adjustment`, `cost_recalculation`, `budget_confirmation`
- Expected preserved decisions: `theme`, `dietary_policy`, `accessibility`
- Failure stage: `none`

| Strategy | Invalidation Accuracy | Preserved Accuracy | Final-State Correctness | Recomputed Decisions | Unnecessary Recomputation | Missed Recomputation | Mean Latency (ms) | Cycle Detections |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_replan` | 1.000 | 1.000 | 1.000 | 9 | 3 | 0 | 0.035 | 0 |
| `targeted_replan` | 1.000 | 1.000 | 1.000 | 6 | 0 | 0 | 0.025 | 0 |

### v0-3-control-no-op-update
No-op update control

- Capability tags: `control`, `no_op`, `replanning`
- Expected invalidated decisions: none
- Expected preserved decisions: `venue_capacity`, `theme`, `dietary`
- Failure stage: `none`

| Strategy | Invalidation Accuracy | Preserved Accuracy | Final-State Correctness | Recomputed Decisions | Unnecessary Recomputation | Missed Recomputation | Mean Latency (ms) | Cycle Detections |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_replan` | 1.000 | 1.000 | 1.000 | 3 | 3 | 0 | 0.020 | 0 |
| `targeted_replan` | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 | 0.012 | 0 |

### v0-3-control-broad-update
Broad schedule update control

- Capability tags: `control`, `broad_update`, `replanning`, `cross_domain`
- Expected invalidated decisions: `venue_availability`, `vendor_availability`, `setup_window`, `parking`, `budget`
- Expected preserved decisions: `accessibility`, `theme`
- Failure stage: `none`

| Strategy | Invalidation Accuracy | Preserved Accuracy | Final-State Correctness | Recomputed Decisions | Unnecessary Recomputation | Missed Recomputation | Mean Latency (ms) | Cycle Detections |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_replan` | 1.000 | 1.000 | 1.000 | 7 | 2 | 0 | 0.025 | 0 |
| `targeted_replan` | 1.000 | 1.000 | 1.000 | 5 | 0 | 0 | 0.021 | 0 |

## Notes

- This experiment is fully offline and deterministic.
- It compares full replanning against dependency-aware targeted replanning.
- Scenario IDs 51, 52, and 55 align with the predeclared capability-boundary research fixtures.
- No agents, orchestration framework, semantic retrieval, RRF, or reranking are used.
