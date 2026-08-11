from __future__ import annotations

import builtins
from datetime import date
from pathlib import Path

import pytest

from partypilot.adapters.in_memory_resource_store import DEFAULT_RESOURCES
from partypilot.application.capability_boundary_benchmark import (
    BENCHMARK_VERSION,
    build_capability_boundary_manifest,
    compute_capability_boundary_checksum,
    load_capability_boundary_manifest,
    load_capability_boundary_scenarios,
)
from partypilot.application.v02_release import load_documents, load_scenarios
from partypilot.domain.evaluation import (
    CapabilityBoundaryScenario,
    CapabilityBoundaryScenarioMetadata,
    DatasetSplit,
    EvaluationScenario,
)
from partypilot.domain.evidence_corpus import EvidenceDocument
from partypilot.domain.resources import Resource


def test_capability_boundary_dataset_is_frozen_and_self_consistent() -> None:
    scenarios = load_capability_boundary_scenarios()
    manifest = load_capability_boundary_manifest()
    built_manifest = build_capability_boundary_manifest(scenarios)
    checksum = compute_capability_boundary_checksum()
    corpus_documents = load_documents(Path("data/evidence/v0_2_documents.json"))
    document_index = {document.metadata.document_id: document for document in corpus_documents}
    for scenario in scenarios:
        for document in scenario.evidence_documents:
            document_index[document.metadata.document_id] = document
    resource_ids = {resource.resource_id for resource in DEFAULT_RESOURCES}
    for scenario in scenarios:
        for resource in scenario.structured_resources:
            resource_ids.add(resource.resource_id)
    document_resource_ids = {document.metadata.resource_id for document in document_index.values()}

    assert len(scenarios) == 50
    assert len({scenario.scenario.scenario_id for scenario in scenarios}) == len(scenarios)
    assert len(
        {
            document.metadata.document_id
            for scenario in scenarios
            for document in scenario.evidence_documents
        }
    ) == sum(len(scenario.evidence_documents) for scenario in scenarios)
    assert all(isinstance(scenario, CapabilityBoundaryScenario) for scenario in scenarios)
    assert all(isinstance(scenario.scenario, EvaluationScenario) for scenario in scenarios)
    assert all(
        isinstance(scenario.metadata, CapabilityBoundaryScenarioMetadata) for scenario in scenarios
    )
    assert all(
        isinstance(document, EvidenceDocument)
        for scenario in scenarios
        for document in scenario.evidence_documents
    )
    assert all(
        isinstance(resource, Resource)
        for scenario in scenarios
        for resource in scenario.structured_resources
    )
    assert all(
        isinstance(document.metadata.effective_date, date)
        and isinstance(document.metadata.version, str)
        for scenario in scenarios
        for document in scenario.evidence_documents
    )
    assert any(scenario.metadata.requires_evidence for scenario in scenarios)
    assert any(scenario.metadata.requires_state_replanning for scenario in scenarios)
    assert any(scenario.metadata.adversarial_flag for scenario in scenarios)
    assert any(scenario.metadata.complexity_trap_flag for scenario in scenarios)
    assert any("derived_arithmetic" in scenario.metadata.capability_tags for scenario in scenarios)
    assert any("conflict" in scenario.metadata.capability_tags for scenario in scenarios)
    assert any("temporal" in scenario.metadata.capability_tags for scenario in scenarios)
    assert "required_architecture" not in CapabilityBoundaryScenarioMetadata.model_fields
    assert "architecture_requirement" not in CapabilityBoundaryScenarioMetadata.model_fields
    assert BENCHMARK_VERSION == "1.0"
    assert manifest == built_manifest
    assert manifest.benchmark_version == BENCHMARK_VERSION
    assert manifest.scenario_count == len(scenarios)
    assert manifest.scenarios_checksum_sha256 == checksum
    assert checksum == compute_capability_boundary_checksum()
    assert {
        "cap-boundary-21-severe-nut-allergy",
        "cap-boundary-22-celiac-vs-gluten-free",
        "cap-boundary-23-hidden-accessibility-issue",
        "cap-boundary-24-ambiguous-quiet-room",
        "cap-boundary-25-outside-catering-restriction",
        "cap-boundary-26-hidden-service-fees",
        "cap-boundary-27-cancellation-policy",
        "cap-boundary-28-multiple-dietary-needs",
        "cap-boundary-29-conflicting-evidence",
        "cap-boundary-30-outdated-evidence",
        "cap-boundary-31-seated-vs-standing-capacity",
        "cap-boundary-32-security-rule-extraction",
        "cap-boundary-33-multi-document-compatibility",
        "cap-boundary-34-evidence-authority",
        "cap-boundary-35-needle-in-haystack",
        "cap-boundary-36-distractor-documents",
        "cap-boundary-37-unsupported-conclusion",
        "cap-boundary-38-conditional-policy",
        "cap-boundary-39-exception-buried-in-policy",
        "cap-boundary-40-prompt-injected-document",
        "cap-boundary-41-venue-caterer-dependency",
        "cap-boundary-42-venue-activity-dependency",
        "cap-boundary-43-setup-scheduling-chain",
        "cap-boundary-44-loading-bay-conflict",
        "cap-boundary-45-outdoor-rain-contingency",
        "cap-boundary-46-end-to-end-accessibility-chain",
        "cap-boundary-47-specialist-disagreement",
        "cap-boundary-48-local-vs-global-optimum",
        "cap-boundary-49-dependency-loop",
        "cap-boundary-50-budget-cascade",
        "cap-boundary-51-incremental-replanning",
        "cap-boundary-52-new-safety-constraint-after-planning",
        "cap-boundary-53-impossible-request-relaxation",
        "cap-boundary-54-counterintuitive-global-solution",
        "cap-boundary-55-cascading-failure",
        "cap-boundary-56-conflicting-user-preferences",
        "cap-boundary-57-ambiguous-hard-vs-soft-constraint",
        "cap-boundary-58-agent-hallucination",
        "cap-boundary-59-conflicting-agents-evidence",
        "cap-boundary-60-full-boss-battle",
        "cap-boundary-61-large-but-purely-structured",
        "cap-boundary-62-long-request-simple-extraction",
        "cap-boundary-63-many-vendors-simple-ranking",
        "cap-boundary-64-scary-wording-explicit-trusted-field",
        "cap-boundary-65-ten-structured-constraints",
    } <= {scenario.scenario.scenario_id for scenario in scenarios}
    assert any(
        scenario.scenario.expected_feasibility.value == "NO_FEASIBLE_PLAN"
        for scenario in scenarios
        if scenario.scenario.scenario_id == "cap-boundary-26-hidden-service-fees"
    )
    assert any(
        label.policy_type.value == "allergen_policy"
        for scenario in scenarios
        if scenario.scenario.scenario_id == "cap-boundary-22-celiac-vs-gluten-free"
        for label in scenario.scenario.retrieval_ground_truth
    )
    assert any(
        label.expected_document_ids == ("doc-cap31-formal-seated-capacity",)
        for scenario in scenarios
        if scenario.scenario.scenario_id == "cap-boundary-31-seated-vs-standing-capacity"
        for label in scenario.scenario.retrieval_ground_truth
    )
    assert any(
        scenario.metadata.adversarial_flag
        for scenario in scenarios
        if scenario.scenario.scenario_id == "cap-boundary-40-prompt-injected-document"
    )
    assert any(
        scenario.scenario.expected_feasibility.value == "FEASIBLE"
        for scenario in scenarios
        if scenario.scenario.scenario_id == "cap-boundary-48-local-vs-global-optimum"
    )
    assert any(
        scenario.metadata.requires_state_replanning
        for scenario in scenarios
        if scenario.scenario.scenario_id == "cap-boundary-51-incremental-replanning"
    )
    assert any(
        "Relaxation alternatives" in " ".join(scenario.scenario.labeling_notes)
        for scenario in scenarios
        if scenario.scenario.scenario_id == "cap-boundary-53-impossible-request-relaxation"
    )
    assert any(
        label.policy_type.value == "allergen_policy"
        for scenario in scenarios
        if scenario.scenario.scenario_id == "cap-boundary-64-scary-wording-explicit-trusted-field"
        for label in scenario.scenario.retrieval_ground_truth
    )
    assert all(
        label.resource_id in document_resource_ids | resource_ids
        for scenario in scenarios
        for label in scenario.scenario.retrieval_ground_truth
    )
    assert all(
        set(label.expected_document_ids) <= set(document_index)
        for scenario in scenarios
        for label in scenario.scenario.retrieval_ground_truth
    )
    assert all(
        set(scenario.scenario.expected_resource_ids) <= (document_resource_ids | resource_ids)
        for scenario in scenarios
        if scenario.scenario.expected_resource_ids
    )
    assert all(
        label.expected_status.value in {"current", "outdated", "superseded", "draft"}
        for scenario in scenarios
        for label in scenario.scenario.retrieval_ground_truth
    )
    assert all(
        document.metadata.status.value in {"current", "outdated", "superseded", "draft"}
        for scenario in scenarios
        for document in scenario.evidence_documents
    )


def test_capability_boundary_manifest_is_deterministic() -> None:
    first = compute_capability_boundary_checksum()
    second = compute_capability_boundary_checksum()

    assert first == second
    assert load_capability_boundary_manifest().scenarios_checksum_sha256 == first


def test_capability_boundary_scenarios_do_not_change_canonical_v0_2_dataset() -> None:
    canonical = Path("data/evaluation/core_scenarios.json")
    assert canonical.exists()
    scenarios = load_scenarios(DatasetSplit.DEVELOPMENT)
    assert len(scenarios) == 10
    assert all(scenario.dataset_split is DatasetSplit.DEVELOPMENT for scenario in scenarios)


def test_capability_boundary_loading_does_not_import_multi_agent_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if "langgraph" in name or "multi_agent" in name or "orchestrator" in name:
            imported.append(name)
            raise AssertionError(f"unexpected multi-agent import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    scenarios = load_capability_boundary_scenarios()

    assert len(scenarios) == 50
    assert imported == []
