from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

NonEmptyString = Annotated[str, Field(min_length=1)]


class EvidenceDocumentType(StrEnum):
    VENUE_POLICY = "venue_policy"
    ALLERGEN_POLICY = "allergen_policy"
    OUTSIDE_FOOD_RULES = "outside_food_rules"
    SUPERVISION_REQUIREMENTS = "supervision_requirements"
    ACTIVITY_SAFETY_GUIDANCE = "activity_safety_guidance"
    ACCESSIBILITY_GUIDANCE = "accessibility_guidance"
    CANCELLATION_TERMS = "cancellation_terms"


class EvidenceDocumentStatus(StrEnum):
    CURRENT = "current"
    OUTDATED = "outdated"
    SUPERSEDED = "superseded"
    DRAFT = "draft"


class EvidenceDocumentMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: NonEmptyString
    resource_id: NonEmptyString
    document_type: EvidenceDocumentType
    version: NonEmptyString
    effective_date: date
    status: EvidenceDocumentStatus


class EvidenceDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: EvidenceDocumentMetadata
    text: NonEmptyString
