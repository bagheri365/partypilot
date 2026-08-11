# PartyPilot v0.4 Multi-Agent Coordination Experiment

Benchmark name: `v0.4 multi-agent coordination benchmark`
Benchmark version: `1.0`
Evaluation variant: `v0_3_stateful_single_planner_vs_minimal_specialist_coordination`
Scenario count: **10**

## Reproducibility Metadata

- Git SHA: `faf6b1d4a3cc1df761ae748adbfae0c054471021`
- Working tree dirty: `False`
- Dataset version: `1.0`
- Baseline architecture: `v0.3_stateful_single_planner`
- Multi-agent architecture: `minimal_specialist_coordination`

## Metric Definitions

- `final_decision_accuracy`: Fraction of scenarios whose terminal feasibility outcome matches the benchmark label.
- `hard_constraint_validity`: Fraction of scenarios where the chosen result respects deterministic hard constraints.
- `cross_domain_compatibility_accuracy`: Fraction of scenarios where cross-resource dependencies are handled correctly.
- `evidence_grounded_arbitration_accuracy`: Fraction of evidence-relevant scenarios where the controlling evidence and arbitration outcome align with current authoritative evidence.
- `global_optimum_accuracy`: Fraction of globally-optimizable scenarios where the chosen combination is the lowest-total-cost viable option.
- `human_review_calibration`: Fraction of HUMAN_REVIEW_REQUIRED scenarios that the architecture routes to human review.
- `specialist_call_count`: Total number of specialist recommendations produced by the coordinated path.
- `coordination_overhead_count`: Total number of explicit coordination/dependency checks performed by the coordinator.
- `mean_latency_ms`: Mean wall-clock latency per scenario for the architecture.

## Aggregate Metrics

| Architecture | Final Decision Accuracy | Hard Constraint Validity | Cross-Domain Compatibility | Evidence-Grounded Arbitration | Global Optimum Accuracy | Human Review Calibration | Specialist Calls | Coordination Overhead | Mean Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v0.3_stateful_single_planner` | 0.600 | 0.600 | 0.500 | 0.000 | 1.000 | 0.000 | 0 | 0 | 0.009 |
| `minimal_specialist_coordination` | 1.000 | 0.600 | 0.700 | 1.000 | 1.000 | 1.000 | 50 | 80 | 0.093 |

- Coordination overhead ratio: N/A
- Retention rule passed: True

## Per-Scenario Results

### cap-boundary-41-venue-caterer-dependency
Venue and caterer dependency

- Capability tags: `cross_domain, dependency_chain, replanning`
- Expected feasibility: `NO_FEASIBLE_PLAN`
- Agreement control: `True`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Cross-Domain Compatibility | Evidence-Grounded Arbitration | Global Optimum | Human Review Calibrated | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v0.3_stateful_single_planner` | `FEASIBLE` | 0.000 | 1.000 | 0.000 | 0.000 | N/A | N/A | 0 | 0 | cross_domain_compatibility |
| `minimal_specialist_coordination` | `NO_FEASIBLE_PLAN` | 1.000 | 1.000 | 0.000 | 1.000 | N/A | N/A | 5 | 5 | none |

#### Arbitration Trace

- Outcome: `REJECT`
- Feasibility outcome: `NO_FEASIBLE_PLAN`
- Selected resources: `venue-boston-studio, caterer-boston-buffet`
- Accepted specialists: `catering, accessibility, scheduling, budget`
- Rejected specialists: `venue`
- Controlling evidence: `doc-cap41-approved-caterer-list, doc-cap41-caterer-vendor-rule`
- Dependency conflicts: `catering_safety`

### cap-boundary-42-venue-activity-dependency
Venue and activity dependency

- Capability tags: `cross_domain, activity_dependency, replanning`
- Expected feasibility: `NO_FEASIBLE_PLAN`
- Agreement control: `True`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Cross-Domain Compatibility | Evidence-Grounded Arbitration | Global Optimum | Human Review Calibrated | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v0.3_stateful_single_planner` | `NO_FEASIBLE_PLAN` | 1.000 | 0.000 | 0.000 | 0.000 | N/A | N/A | 0 | 0 | none |
| `minimal_specialist_coordination` | `NO_FEASIBLE_PLAN` | 1.000 | 0.000 | 0.000 | 1.000 | N/A | N/A | 5 | 5 | none |

#### Arbitration Trace

- Outcome: `REJECT`
- Feasibility outcome: `NO_FEASIBLE_PLAN`
- Selected resources: `venue-brooklyn-loft, activity-craft-party`
- Accepted specialists: `catering, accessibility, scheduling, budget`
- Rejected specialists: `venue`
- Controlling evidence: `doc-cap42-venue-no-prep-room, doc-cap42-activity-prep-requirement`
- Dependency conflicts: `none`

### cap-boundary-43-setup-scheduling-chain
Setup scheduling chain

- Capability tags: `temporal_dependency, setup_chain, replanning`
- Expected feasibility: `NO_FEASIBLE_PLAN`
- Agreement control: `True`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Cross-Domain Compatibility | Evidence-Grounded Arbitration | Global Optimum | Human Review Calibrated | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v0.3_stateful_single_planner` | `NO_FEASIBLE_PLAN` | 1.000 | 0.000 | 0.000 | 0.000 | N/A | N/A | 0 | 0 | none |
| `minimal_specialist_coordination` | `NO_FEASIBLE_PLAN` | 1.000 | 0.000 | 1.000 | 1.000 | N/A | N/A | 5 | 5 | none |

#### Arbitration Trace

- Outcome: `REJECT`
- Feasibility outcome: `NO_FEASIBLE_PLAN`
- Selected resources: `venue-boston-evening-hall, caterer-boston-buffet, activity-craft-party`
- Accepted specialists: `venue, catering, accessibility, budget`
- Rejected specialists: `scheduling`
- Controlling evidence: `doc-cap43-venue-access, doc-cap43-setup-window, doc-cap43-caterer-setup, doc-cap43-activity-setup-window`
- Dependency conflicts: `venue, catering, activity`

### cap-boundary-44-loading-bay-conflict
Loading-bay conflict

- Capability tags: `cross_domain, loading_bay, logistics_conflict`
- Expected feasibility: `NO_FEASIBLE_PLAN`
- Agreement control: `True`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Cross-Domain Compatibility | Evidence-Grounded Arbitration | Global Optimum | Human Review Calibrated | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v0.3_stateful_single_planner` | `NO_FEASIBLE_PLAN` | 1.000 | 0.000 | 0.000 | 0.000 | N/A | N/A | 0 | 0 | none |
| `minimal_specialist_coordination` | `NO_FEASIBLE_PLAN` | 1.000 | 0.000 | 0.000 | 1.000 | N/A | N/A | 5 | 5 | none |

#### Arbitration Trace

- Outcome: `REJECT`
- Feasibility outcome: `NO_FEASIBLE_PLAN`
- Selected resources: `venue-brooklyn-terrace, caterer-brooklyn-hosted`
- Accepted specialists: `venue, catering, accessibility, budget`
- Rejected specialists: `scheduling`
- Controlling evidence: `doc-cap44-loading-bay-window, doc-cap44-caterer-delivery-window`
- Dependency conflicts: `venue, catering`

### cap-boundary-45-outdoor-rain-contingency
Outdoor rain contingency

- Capability tags: `temporal_dependency, contingency, replanning`
- Expected feasibility: `HUMAN_REVIEW_REQUIRED`
- Agreement control: `True`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Cross-Domain Compatibility | Evidence-Grounded Arbitration | Global Optimum | Human Review Calibrated | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v0.3_stateful_single_planner` | `NO_FEASIBLE_PLAN` | 0.000 | 0.000 | 0.000 | 0.000 | N/A | 0.000 | 0 | 0 | hard_constraints |
| `minimal_specialist_coordination` | `HUMAN_REVIEW_REQUIRED` | 1.000 | 0.000 | 1.000 | 1.000 | N/A | 1.000 | 5 | 5 | none |

#### Arbitration Trace

- Outcome: `HUMAN_REVIEW_REQUIRED`
- Feasibility outcome: `HUMAN_REVIEW_REQUIRED`
- Selected resources: `venue-cambridge-garden`
- Accepted specialists: `catering, accessibility, budget`
- Rejected specialists: `none`
- Controlling evidence: `doc-cap45-outdoor-policy, doc-cap45-rain-contingency`
- Dependency conflicts: `none`

### cap-boundary-47-specialist-disagreement
Specialist disagreement

- Capability tags: `arbitration, cross_domain, conflict`
- Expected feasibility: `HUMAN_REVIEW_REQUIRED`
- Agreement control: `False`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Cross-Domain Compatibility | Evidence-Grounded Arbitration | Global Optimum | Human Review Calibrated | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v0.3_stateful_single_planner` | `FEASIBLE` | 0.000 | 1.000 | 1.000 | 0.000 | N/A | 0.000 | 0 | 0 | evidence_authority |
| `minimal_specialist_coordination` | `HUMAN_REVIEW_REQUIRED` | 1.000 | 1.000 | 1.000 | 1.000 | N/A | 1.000 | 5 | 5 | none |

#### Arbitration Trace

- Outcome: `HUMAN_REVIEW_REQUIRED`
- Feasibility outcome: `HUMAN_REVIEW_REQUIRED`
- Selected resources: `venue-boston-garden`
- Accepted specialists: `venue, catering, scheduling, budget`
- Rejected specialists: `none`
- Controlling evidence: `doc-cap47-specialist-support, doc-cap47-specialist-objection`
- Dependency conflicts: `none`

### cap-boundary-48-local-vs-global-optimum
Local versus global optimum

- Capability tags: `global_optimization, cross_domain, budget_tradeoff`
- Expected feasibility: `FEASIBLE`
- Agreement control: `True`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Cross-Domain Compatibility | Evidence-Grounded Arbitration | Global Optimum | Human Review Calibrated | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v0.3_stateful_single_planner` | `FEASIBLE` | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | N/A | 0 | 0 | none |
| `minimal_specialist_coordination` | `FEASIBLE` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | N/A | 5 | 10 | none |

#### Arbitration Trace

- Outcome: `ACCEPT`
- Feasibility outcome: `FEASIBLE`
- Selected resources: `venue-brooklyn-loft, caterer-family-table`
- Accepted specialists: `venue, catering, accessibility, scheduling, budget`
- Rejected specialists: `none`
- Controlling evidence: `doc-cap48-expensive-venue, doc-cap48-global-cost-note, doc-cap48-cheap-caterer`
- Dependency conflicts: `none`

### cap-boundary-59-conflicting-agents-evidence
Conflicting agents and evidence

- Capability tags: `arbitration, cross_domain, conflict`
- Expected feasibility: `HUMAN_REVIEW_REQUIRED`
- Agreement control: `False`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Cross-Domain Compatibility | Evidence-Grounded Arbitration | Global Optimum | Human Review Calibrated | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v0.3_stateful_single_planner` | `FEASIBLE` | 0.000 | 1.000 | 1.000 | 0.000 | N/A | 0.000 | 0 | 0 | evidence_authority |
| `minimal_specialist_coordination` | `HUMAN_REVIEW_REQUIRED` | 1.000 | 1.000 | 1.000 | 1.000 | N/A | 1.000 | 5 | 5 | none |

#### Arbitration Trace

- Outcome: `HUMAN_REVIEW_REQUIRED`
- Feasibility outcome: `HUMAN_REVIEW_REQUIRED`
- Selected resources: `venue-boston-garden`
- Accepted specialists: `catering, scheduling, budget`
- Rejected specialists: `none`
- Controlling evidence: `doc-cap59-recommendation-note, doc-cap59-accessibility-analysis`
- Dependency conflicts: `none`

### cap-boundary-61-large-but-purely-structured
Large but purely structured

- Capability tags: `structured_only, large_candidate_set, optimization`
- Expected feasibility: `FEASIBLE`
- Agreement control: `True`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Cross-Domain Compatibility | Evidence-Grounded Arbitration | Global Optimum | Human Review Calibrated | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v0.3_stateful_single_planner` | `FEASIBLE` | 1.000 | 1.000 | 1.000 | N/A | N/A | N/A | 0 | 0 | none |
| `minimal_specialist_coordination` | `FEASIBLE` | 1.000 | 1.000 | 1.000 | N/A | N/A | N/A | 5 | 30 | none |

#### Arbitration Trace

- Outcome: `ACCEPT`
- Feasibility outcome: `FEASIBLE`
- Selected resources: `venue-boston-hall-c, caterer-boston-a, activity-boston-structured`
- Accepted specialists: `venue, catering, accessibility, scheduling, budget`
- Rejected specialists: `none`
- Controlling evidence: `none`
- Dependency conflicts: `none`

### cap-boundary-65-ten-structured-constraints
Ten structured constraints

- Capability tags: `structured_constraints, many_constraints, optimization`
- Expected feasibility: `FEASIBLE`
- Agreement control: `True`

| Architecture | Outcome | Final Decision Accuracy | Hard Constraint Validity | Cross-Domain Compatibility | Evidence-Grounded Arbitration | Global Optimum | Human Review Calibrated | Specialist Calls | Coordination Overhead | Failure Stage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v0.3_stateful_single_planner` | `FEASIBLE` | 1.000 | 1.000 | 1.000 | N/A | N/A | N/A | 0 | 0 | none |
| `minimal_specialist_coordination` | `FEASIBLE` | 1.000 | 1.000 | 1.000 | N/A | N/A | N/A | 5 | 5 | none |

#### Arbitration Trace

- Outcome: `ACCEPT`
- Feasibility outcome: `FEASIBLE`
- Selected resources: `venue-boston-hall-d, caterer-boston-c, activity-boston-c`
- Accepted specialists: `venue, catering, accessibility, scheduling, budget`
- Rejected specialists: `none`
- Controlling evidence: `none`
- Dependency conflicts: `none`

## Notes

- This experiment is fully offline and deterministic.
- It compares a stateful single-planner baseline to a minimal specialist/coordinator path.
- The benchmark is intentionally small and research-focused.