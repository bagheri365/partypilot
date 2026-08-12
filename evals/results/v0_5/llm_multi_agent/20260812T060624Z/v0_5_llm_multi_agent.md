# PartyPilot v0.5 Live Multi-Agent Runtime Experiment

Benchmark name: `v0.5 live multi-agent benchmark`
Benchmark version: `1.0`
Evaluation variant: `deterministic_specialists_vs_live_llm_specialists`
Scenario count: **10**

## Reproducibility Metadata

- Git SHA: `1e22f0f550d51d9cf4d1529865a8083d9b90b966`
- Working tree dirty: `False`
- Dataset version: `1.0`
- Baseline architecture: `v0.4_deterministic_specialist_coordination`
- Live architecture: `v0.5_live_llm_specialist_agents`
- Model name: `n/a`

## Metric Definitions

- `final_decision_accuracy`: Fraction of scenarios whose terminal feasibility outcome matches the benchmark label.
- `hard_constraint_validity`: Fraction of scenarios where the chosen result respects deterministic hard constraints.
- `cross_domain_compatibility_accuracy`: Fraction of scenarios where cross-resource dependencies are handled correctly.
- `evidence_grounded_arbitration_accuracy`: Fraction of evidence-relevant scenarios where arbitration uses authoritative evidence.
- `global_optimum_accuracy`: Fraction of global-optimization scenarios where the lowest-cost viable option is chosen.
- `human_review_calibration`: Fraction of HUMAN_REVIEW_REQUIRED scenarios routed to human review.
- `specialist_call_count`: Total number of specialist recommendations produced by the architecture.
- `coordination_overhead_count`: Total number of explicit coordination/dependency checks performed by the coordinator.
- `mean_latency_ms`: Mean wall-clock latency per scenario for the architecture.

## Aggregate Metrics

| Metric | Baseline | Live |
|---|---:|---:|
| Final decision accuracy | 1.000 | 1.000 |
| Hard-constraint validity | 0.600 | 0.600 |
| Cross-domain compatibility | 0.700 | 0.700 |
| Evidence-grounded arbitration | 1.000 | 1.000 |
| Global optimum accuracy | 1.000 | 1.000 |
| Human review calibration | 1.000 | 1.000 |
| Specialist calls | 50 | 42 |
| Coordination overhead | 80 | 80 |

## Runtime Metrics

- Total specialist calls: `80`
- Mean specialists invoked per scenario: `8.000`
- Specialist success rate: `0.900`
- Structured output validation failure rate: `0.000`
- Retry rate: `0.000`
- Specialist disagreement rate: `0.600`
- Coordinator override count: `21`
- Mean latency (ms): `126854.698`

- Retention rule passed: `True`

## Scenario Diagnostics

- Terminal outcome mismatches: `none`
- Diagnostic failure-stage cases: `cap-boundary-41-venue-caterer-dependency, cap-boundary-42-venue-activity-dependency, cap-boundary-43-setup-scheduling-chain, cap-boundary-44-loading-bay-conflict`

## Per-Scenario Results

### cap-boundary-41-venue-caterer-dependency
Venue and caterer dependency

- Capability tags: `cross_domain, dependency_chain, replanning`
- Expected feasibility: `NO_FEASIBLE_PLAN`
- Requires evidence: `True`
- Requires global optimum: `False`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `minimal_specialist_coordination` | `NO_FEASIBLE_PLAN` | 1.000 | 1.000 | 1.000 | 0.000 | 5 | 5 | none |
| `v0.5_live_llm_specialist_agents` | `NO_FEASIBLE_PLAN` | 1.000 | 1.000 | 1.000 | 0.000 | 3 | 5 | hard_constraints |

#### Runtime Trace

- Wall-clock latency: `90023.899` ms
- Selected resources: `venue-boston-studio, caterer-boston-buffet`
- Execution traces: `5`
- Arbitration outcome: `REJECT`
- Controlling evidence: `doc-cap41-approved-caterer-list, doc-cap41-caterer-vendor-rule`
- Accepted specialists: `accessibility, budget`
- Rejected specialists: `catering`

### cap-boundary-42-venue-activity-dependency
Venue and activity dependency

- Capability tags: `cross_domain, activity_dependency, replanning`
- Expected feasibility: `NO_FEASIBLE_PLAN`
- Requires evidence: `True`
- Requires global optimum: `False`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `minimal_specialist_coordination` | `NO_FEASIBLE_PLAN` | 1.000 | 0.000 | 1.000 | 0.000 | 5 | 5 | none |
| `v0.5_live_llm_specialist_agents` | `NO_FEASIBLE_PLAN` | 1.000 | 0.000 | 1.000 | 0.000 | 3 | 5 | hard_constraints |

#### Runtime Trace

- Wall-clock latency: `90010.988` ms
- Selected resources: `venue-brooklyn-loft, activity-craft-party`
- Execution traces: `5`
- Arbitration outcome: `REJECT`
- Controlling evidence: `doc-cap42-venue-no-prep-room`
- Accepted specialists: `accessibility`
- Rejected specialists: `catering`

### cap-boundary-43-setup-scheduling-chain
Setup scheduling chain

- Capability tags: `temporal_dependency, setup_chain, replanning`
- Expected feasibility: `NO_FEASIBLE_PLAN`
- Requires evidence: `True`
- Requires global optimum: `False`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `minimal_specialist_coordination` | `NO_FEASIBLE_PLAN` | 1.000 | 0.000 | 1.000 | 0.000 | 5 | 5 | none |
| `v0.5_live_llm_specialist_agents` | `NO_FEASIBLE_PLAN` | 1.000 | 0.000 | 1.000 | 0.000 | 3 | 5 | hard_constraints |

#### Runtime Trace

- Wall-clock latency: `90012.550` ms
- Selected resources: `venue-boston-evening-hall, caterer-boston-buffet, activity-craft-party`
- Execution traces: `5`
- Arbitration outcome: `REJECT`
- Controlling evidence: `doc-cap43-venue-access, doc-cap43-setup-window, doc-cap43-caterer-setup, doc-cap43-activity-setup-window`
- Accepted specialists: `accessibility, catering, scheduling`
- Rejected specialists: `none`

### cap-boundary-44-loading-bay-conflict
Loading-bay conflict

- Capability tags: `cross_domain, loading_bay, logistics_conflict`
- Expected feasibility: `NO_FEASIBLE_PLAN`
- Requires evidence: `True`
- Requires global optimum: `False`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `minimal_specialist_coordination` | `NO_FEASIBLE_PLAN` | 1.000 | 0.000 | 1.000 | 0.000 | 5 | 5 | none |
| `v0.5_live_llm_specialist_agents` | `NO_FEASIBLE_PLAN` | 1.000 | 0.000 | 1.000 | 0.000 | 3 | 5 | hard_constraints |

#### Runtime Trace

- Wall-clock latency: `90013.078` ms
- Selected resources: `venue-brooklyn-terrace, caterer-brooklyn-hosted`
- Execution traces: `5`
- Arbitration outcome: `REJECT`
- Controlling evidence: `doc-cap44-loading-bay-window, doc-cap44-caterer-delivery-window`
- Accepted specialists: `budget`
- Rejected specialists: `none`

### cap-boundary-45-outdoor-rain-contingency
Outdoor rain contingency

- Capability tags: `temporal_dependency, contingency, replanning`
- Expected feasibility: `HUMAN_REVIEW_REQUIRED`
- Requires evidence: `True`
- Requires global optimum: `False`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `minimal_specialist_coordination` | `HUMAN_REVIEW_REQUIRED` | 1.000 | 0.000 | 1.000 | 0.000 | 5 | 5 | none |
| `v0.5_live_llm_specialist_agents` | `HUMAN_REVIEW_REQUIRED` | 1.000 | 0.000 | 1.000 | 0.000 | 5 | 5 | none |

#### Runtime Trace

- Wall-clock latency: `70852.165` ms
- Selected resources: `venue-cambridge-garden`
- Execution traces: `5`
- Arbitration outcome: `HUMAN_REVIEW_REQUIRED`
- Controlling evidence: `doc-cap45-outdoor-policy, doc-cap45-rain-contingency`
- Accepted specialists: `budget, catering, scheduling`
- Rejected specialists: `none`

### cap-boundary-47-specialist-disagreement
Specialist disagreement

- Capability tags: `arbitration, cross_domain, conflict`
- Expected feasibility: `HUMAN_REVIEW_REQUIRED`
- Requires evidence: `True`
- Requires global optimum: `False`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `minimal_specialist_coordination` | `HUMAN_REVIEW_REQUIRED` | 1.000 | 1.000 | 1.000 | 1.000 | 5 | 5 | none |
| `v0.5_live_llm_specialist_agents` | `HUMAN_REVIEW_REQUIRED` | 1.000 | 1.000 | 1.000 | 1.000 | 5 | 5 | none |

#### Runtime Trace

- Wall-clock latency: `88942.588` ms
- Selected resources: `venue-boston-garden`
- Execution traces: `5`
- Arbitration outcome: `HUMAN_REVIEW_REQUIRED`
- Controlling evidence: `doc-cap47-specialist-support, doc-cap47-specialist-objection`
- Accepted specialists: `budget, catering, scheduling`
- Rejected specialists: `none`

### cap-boundary-48-local-vs-global-optimum
Local versus global optimum

- Capability tags: `global_optimization, cross_domain, budget_tradeoff`
- Expected feasibility: `FEASIBLE`
- Requires evidence: `True`
- Requires global optimum: `True`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `minimal_specialist_coordination` | `FEASIBLE` | 1.000 | 1.000 | 1.000 | 1.000 | 5 | 10 | none |
| `v0.5_live_llm_specialist_agents` | `FEASIBLE` | 1.000 | 1.000 | 1.000 | 1.000 | 5 | 10 | none |

#### Runtime Trace

- Wall-clock latency: `146937.183` ms
- Selected resources: `venue-brooklyn-loft, caterer-family-table`
- Execution traces: `10`
- Arbitration outcome: `ACCEPT`
- Controlling evidence: `doc-cap48-expensive-venue, doc-cap48-cheap-caterer, doc-cap48-global-cost-note`
- Accepted specialists: `accessibility, budget, catering, scheduling, venue`
- Rejected specialists: `none`

### cap-boundary-59-conflicting-agents-evidence
Conflicting agents and evidence

- Capability tags: `arbitration, cross_domain, conflict`
- Expected feasibility: `HUMAN_REVIEW_REQUIRED`
- Requires evidence: `True`
- Requires global optimum: `False`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `minimal_specialist_coordination` | `HUMAN_REVIEW_REQUIRED` | 1.000 | 1.000 | 1.000 | 1.000 | 5 | 5 | none |
| `v0.5_live_llm_specialist_agents` | `HUMAN_REVIEW_REQUIRED` | 1.000 | 1.000 | 1.000 | 1.000 | 5 | 5 | none |

#### Runtime Trace

- Wall-clock latency: `73450.387` ms
- Selected resources: `venue-boston-garden`
- Execution traces: `5`
- Arbitration outcome: `HUMAN_REVIEW_REQUIRED`
- Controlling evidence: `doc-cap59-recommendation-note, doc-cap59-accessibility-analysis`
- Accepted specialists: `budget, catering, scheduling`
- Rejected specialists: `none`

### cap-boundary-61-large-but-purely-structured
Large but purely structured

- Capability tags: `structured_only, large_candidate_set, optimization`
- Expected feasibility: `FEASIBLE`
- Requires evidence: `False`
- Requires global optimum: `False`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `minimal_specialist_coordination` | `FEASIBLE` | 1.000 | 1.000 | 1.000 | 1.000 | 5 | 30 | none |
| `v0.5_live_llm_specialist_agents` | `FEASIBLE` | 1.000 | 1.000 | 1.000 | 1.000 | 5 | 30 | none |

#### Runtime Trace

- Wall-clock latency: `440864.633` ms
- Selected resources: `venue-boston-hall-c, caterer-boston-a, activity-boston-structured`
- Execution traces: `30`
- Arbitration outcome: `ACCEPT`
- Controlling evidence: `none`
- Accepted specialists: `accessibility, budget, catering, scheduling, venue`
- Rejected specialists: `none`

### cap-boundary-65-ten-structured-constraints
Ten structured constraints

- Capability tags: `structured_constraints, many_constraints, optimization`
- Expected feasibility: `FEASIBLE`
- Requires evidence: `False`
- Requires global optimum: `False`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Evidence-Grounded Arbitration | Global Optimum | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `minimal_specialist_coordination` | `FEASIBLE` | 1.000 | 1.000 | 1.000 | 1.000 | 5 | 5 | none |
| `v0.5_live_llm_specialist_agents` | `FEASIBLE` | 1.000 | 1.000 | 1.000 | 1.000 | 5 | 5 | none |

#### Runtime Trace

- Wall-clock latency: `87439.507` ms
- Selected resources: `venue-boston-hall-d, caterer-boston-c, activity-boston-c`
- Execution traces: `5`
- Arbitration outcome: `ACCEPT`
- Controlling evidence: `none`
- Accepted specialists: `accessibility, budget, catering, scheduling, venue`
- Rejected specialists: `none`

## Notes

- This experiment is fully offline and deterministic except for the live specialist model calls.
- The coordinator remains deterministic; specialists are the only live LLM-backed components.
- The benchmark is intentionally bounded and reused from the v0.4 development subset.