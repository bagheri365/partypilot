# PartyPilot Evaluation Dataset Card

## Purpose

The v0.1 benchmark evaluates whether PartyPilot can produce or reject event plans while respecting explicit hard constraints. It is designed to measure deterministic feasibility behavior first and to support later comparisons with LLM-based variants without changing the benchmark definitions.

The current dataset is `data/evaluation/v0_1_scenarios.json` and contains 24 labeled scenarios spanning development, frozen-test, and adversarial splits.

## Non-goals

This dataset is not a measure of subjective event quality, creativity, prose quality, vendor realism, or production readiness. It does not establish real-world safety, allergy suitability, legal compliance, or accessibility compliance without supporting evidence. It must not be used to claim benchmark performance before the evaluation runner produces measured results.

## Scenario construction

Scenarios are hand-authored from PartyPilot's supported domain constraints and deterministic fixture resources. Coverage includes feasible cases, budget and capacity failures, unavailable resources, age restrictions, accessibility requirements, temporal conflicts, exclusive-resource conflicts, multiple-choice cases, impossible combinations, and evidence-dependent allergy/safety cases.

Scenarios vary in complexity using metadata such as hard-constraint count, expected resource count, and notes for intentionally difficult cases.

## Ground-truth definitions

`expected_feasibility` is the labeled terminal outcome for a scenario: feasible, no feasible plan, or human review required. `expected_hard_constraints` records hard constraints that must be honored. `expected_derived_constraints` records any labeled deterministic derivations. `expected_resource_ids` identifies resources expected for scenarios where a specific selection is part of the label. `relevant_evidence_ids` identifies evidence that later evidence-aware systems should use when applicable.

A feasible label means every applicable labeled hard constraint can be satisfied by the benchmark setup. A no-feasible-plan label means at least one required hard constraint makes all allowed combinations invalid. Human review is used when the structured data is insufficient to resolve a safety- or evidence-dependent requirement.

## Labeling rules

- Encode explicit user requirements as hard constraints when violation would make the plan unacceptable.
- Do not infer unsupported facts about resources.
- Do not label evidence-dependent claims as satisfied without evidence.
- Use exact deterministic boundaries for budget, capacity, time, and availability.
- Keep expected resource IDs empty when the scenario is intended to test infeasibility or when multiple selections are intentionally acceptable.
- Record ambiguity or special construction details in `labeling_notes` or complexity notes.

## Split methodology

The dataset contains three explicit splits:

- `development`: may be inspected during implementation and debugging.
- `frozen_test`: reserved for final evaluation and must not guide implementation choices.
- `adversarial`: stresses edge cases, combined constraints, ambiguity, and failure handling.

The current v0.1 allocation is 10 development scenarios, 8 frozen-test scenarios, and 6 adversarial scenarios.

## Frozen-test leakage policy

Do not tune planner logic, thresholds, ranking weights, prompts, retrieval settings, or provider behavior using frozen-test examples or results. Development work should use the development split. Frozen-test evaluation should occur only after an experiment configuration and decision criteria are fixed. If frozen-test content leaks into tuning, record the leak and create a new benchmark version rather than continuing to treat the affected split as frozen.

## Ambiguity handling

Ambiguous requirements should not be silently converted into favorable assumptions. If the benchmark's structured facts cannot establish that a hard requirement is satisfied, the expected outcome should remain unresolved or require human review as appropriate. Labeling notes should state any interpretation necessary to understand the ground truth.

## Limitations

The dataset is small, synthetic, and tied to the current deterministic fixture universe. It does not represent real vendor inventories, live availability, geographic diversity, dynamic pricing, policy text, or verified allergy/safety claims. Many scenarios test one dominant failure mode even though real planning requests may contain several interacting uncertainties.

## Distribution gaps

Known gaps include broader geography, larger party sizes, richer age distributions, multi-day events, tax/tip/service-fee modeling, transportation, weather, cancellation policies, vendor-specific accessibility details, dietary cross-contamination evidence, and more complex task/resource schedules. Future versions should add these deliberately rather than retrofitting frozen-test cases.

## Versioning policy

Treat each published dataset version as immutable. Corrections that change labels, scenario semantics, split membership, or expected resources require a new dataset version and a documented migration note. Additive metadata corrections that do not affect evaluation semantics should still be recorded. Experiment results must record the exact dataset version they used.
