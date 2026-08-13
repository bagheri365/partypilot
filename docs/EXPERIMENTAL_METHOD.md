# PartyPilot Experimental Method

PartyPilot is not organized around the assumption that more agents or more orchestration are automatically better.

The project treats architecture complexity as something that must earn its place through evidence.

## Research Question

The central research question is:

> When does multi-agent coordination outperform a simpler planner enough to justify added cost and complexity?

That question is deliberately narrower than "can we build an agentic system?".
The goal is to compare architectures on measurable planning behavior, not to declare that any specific framework is inherently superior.

## Experimental Loop

PartyPilot follows a repeatable experimental loop:

1. establish the simplest credible baseline
2. identify a measurable failure mode
3. introduce one capability
4. hold unrelated variables fixed
5. evaluate
6. retain, retain experimentally, or reject

This loop is meant to prevent architectural drift.
If multiple things change at once, it becomes difficult to say which change caused the result.

The project therefore prefers small, legible steps over sweeping rewrites.

## Controlled Variables

PartyPilot tries to hold the following variables fixed when comparing architectures:

- benchmark cases
- model
- context budget
- provider timeout
- coordinator
- deterministic guardrails
- specialist schemas
- orchestration backend

These variables matter because changing them all at once weakens causal interpretation.
If the benchmark, model, timeout policy, and orchestration backend all change together, a result may still be interesting, but it no longer isolates a single architectural question.

The design intent is to ask whether a new capability improves outcomes under the same benchmark and comparable operating conditions.

## Negative Results

PartyPilot preserves negative results instead of tuning until a new architecture wins.

That choice is part of the method.
If a more complex path does not improve the measured outcome, the project records that result rather than retrofitting prompts or heuristics until the architecture appears successful.

Two examples matter here:

- v0.6 `create_agent` tool use: the canonical evaluation recorded zero tool calls, so the path was not promoted as the default specialist strategy.
- v0.7 work-policy audit: the LangGraph comparison was later classified as `VALID_BUT_DIFFERENT_WORK_POLICY`, which means the two backends did not execute identical work and should not be described as a pure framework-efficiency comparison; the imperative orchestration backend remained the compatibility/default comparison path.

Those results are retained because they constrain future claims.

## Canonical-Run Discipline

Canonical runs are treated as frozen evidence records.

The discipline includes:

- clean-tree guard before the run begins
- frozen Git SHA capture
- balanced repeated-run ordering where applicable
- artifact provenance recorded in the output
- no selective reruns that would bias the accounting

The purpose is to make the run reproducible and to make the provenance legible later.

Exploratory runs may be used during development, but they are not canonical evidence unless they satisfy the canonical-run constraints.

## Post-Hoc Auditing

Post-hoc analysis is allowed to reinterpret the meaning of a canonical result.
It is not allowed to rewrite the canonical evidence.

The v0.7 accounting audit is the clearest example:

- the canonical run stayed frozen
- the audit reconstructed the invocation counts from the stored artifacts
- the audit explained why LangGraph recorded 90 specialist invocations rather than 150
- the audit classified the comparison as `VALID_BUT_DIFFERENT_WORK_POLICY`

That is an interpretation of the evidence, not a change to the evidence itself.

## Limitations

The methodology has constraints that should be kept in view:

- the benchmark is small and development-oriented
- the live model is a local quantized model, so provider behavior is variable
- timeout rates are high enough to affect live specialist success materially
- repeated runs help, but they do not create large statistical power
- conclusions are benchmark-specific rather than universal claims about all multi-agent systems
- human-review capability is primarily validated through deterministic offline fixtures, even when the live graph exercises the path

These limitations do not invalidate the method.
They define the scope of the claims that can be made responsibly.

## What Would Falsify the Thesis?

PartyPilot would simplify rather than add more agentic machinery if the evidence showed that complexity was not paying for itself.

Examples of falsifying evidence would include:

- a simpler planner matching or exceeding the more complex architecture on the same benchmark while using less latency and fewer provider attempts
- targeted replanning or multi-agent coordination failing to improve measurable outcomes after the relevant failure mode is isolated
- orchestration changes improving trace shape without improving decision quality or reliability
- additional agents increasing overhead or failure rate without measurable benefit
- repeated live evaluations showing that a supposed improvement disappears once model, timeout, and benchmark conditions are held constant

In that case, the correct response would be to reduce complexity, not to keep adding machinery.
