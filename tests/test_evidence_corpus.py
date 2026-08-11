from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from partypilot.domain.evidence_corpus import EvidenceDocument, EvidenceDocumentStatus


def test_v02_evidence_corpus_is_valid_and_has_difficult_cases() -> None:
    raw = json.loads(Path("data/evidence/v0_2_documents.json").read_text())
    documents = [EvidenceDocument.model_validate(item) for item in raw]

    assert len(documents) >= 20
    ids = [item.metadata.document_id for item in documents]
    assert len(ids) == len(set(ids))
    statuses = Counter(item.metadata.status for item in documents)
    assert statuses[EvidenceDocumentStatus.CURRENT] > 0
    assert statuses[EvidenceDocumentStatus.OUTDATED] > 0
    assert statuses[EvidenceDocumentStatus.SUPERSEDED] > 0
    assert statuses[EvidenceDocumentStatus.DRAFT] > 0
    assert any("contact" in item.text.casefold() for item in documents)
