from __future__ import annotations

import json
from pathlib import Path

from partypilot.domain.evaluation import EvaluationScenario
from partypilot.domain.evidence_corpus import EvidenceDocument


def test_retrieval_labels_match_evidence_corpus_metadata() -> None:
    documents = {
        item.metadata.document_id: item
        for item in (
            EvidenceDocument.model_validate(raw)
            for raw in json.loads(Path("data/evidence/v0_2_documents.json").read_text())
        )
    }
    scenarios = [
        EvaluationScenario.model_validate(raw)
        for raw in json.loads(Path("data/evaluation/core_scenarios.json").read_text())
    ]
    labeled = [scenario for scenario in scenarios if scenario.retrieval_ground_truth]
    assert labeled

    for scenario in labeled:
        expected_ids: list[str] = []
        for label in scenario.retrieval_ground_truth:
            for document_id in label.expected_document_ids:
                expected_ids.append(document_id)
                document = documents[document_id]
                assert document.metadata.resource_id == label.resource_id
                assert document.metadata.version == label.expected_version
                assert document.metadata.status == label.expected_status
                assert document.metadata.document_type == label.policy_type
        assert tuple(expected_ids) == scenario.relevant_evidence_ids
