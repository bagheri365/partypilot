# PartyPilot v0.7d Post-Hoc Accounting Audit

## Classification

`VALID_BUT_DIFFERENT_WORK_POLICY`

The canonical v0.7d run is internally consistent, but the imperative and
LangGraph backends did not perform identical work.

## Specialist Invocation Reconciliation

Across three LangGraph repetitions:

- 30 graph/scenario executions
- 12 deterministic preflight terminal shortcuts
- 18 executions requiring specialist fan-out
- 5 specialists per full fan-out
- 90 top-level specialist invocations

Equation:

`12 × 0 + 18 × 5 = 90`

Equivalently:

`30 × 5 - 12 × 5 = 90`

The imperative backend executed all five specialists for all 30 scenario-runs:

`30 × 5 = 150`

Therefore the reduction from 150 to 90 provider attempts is caused by
LangGraph's deterministic preflight short-circuit policy, not by lower
framework overhead under identical work.

## Preflight Shortcuts

These scenarios were resolved as `NO_FEASIBLE_PLAN` before specialist fan-out
in all three LangGraph repetitions:

- cap-boundary-41-venue-caterer-dependency
- cap-boundary-42-venue-activity-dependency
- cap-boundary-43-setup-scheduling-chain
- cap-boundary-44-loading-bay-conflict

This accounts for:

`4 scenarios × 3 repetitions × 5 specialists = 60 skipped specialist branches`

## Human Review

The remaining six scenarios per repetition completed all five specialist
branches before coordinator execution and human-review interruption.

Across the three LangGraph repetitions:

- coordinator executions: 18
- human-review interrupts: 18
- resumes in primary unattended benchmark: 0

The interrupts do not account for the 60 skipped specialist invocations.

Resume behavior was evaluated separately in the deterministic human-review
sub-benchmark, which passed.

## Join and Accounting Audit

Canonical traces support correct five-way fan-out and join behavior for every
non-terminal-preflight scenario.

There was no evidence of:

- premature coordinator joins
- dropped timeout outcomes
- evaluator undercounting
- reducer loss affecting invocation accounting

No targeted replans occurred in the primary live benchmark.

## Interpretation

Primary results:

- final decision accuracy:
  - imperative: 0.700
  - LangGraph: 0.700
- evidence-grounded arbitration:
  - imperative: 0.750
  - LangGraph: 0.625
- specialist success:
  - imperative: 0.180
  - LangGraph: 0.200
- deterministic orchestration/replan sub-benchmark: PASS
- checkpointed human-review sub-benchmark: PASS

Provider-attempt counts must not be described as a pure framework-efficiency
comparison because the two backends used different work policies.

## Disposition

`RETAIN_EXPERIMENTALLY`

LangGraph demonstrated real orchestration capabilities while preserving final
decision accuracy, but it is not promoted to the default because:

1. evidence-grounded arbitration declined from 0.750 to 0.625; and
2. the primary live experiment did not isolate framework orchestration from
   work-policy differences.

No implementation changes or evaluation reruns were made as part of this audit.
