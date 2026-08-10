from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

NonEmptyString = Annotated[str, Field(min_length=1)]
ConfigScalar = str | int | float | bool


class ExperimentConfig(BaseModel):
    """Immutable configuration identifying a reproducible evaluation run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: NonEmptyString
    code_commit_sha: NonEmptyString | None = None
    working_tree_dirty: bool | None = None
    git_metadata_error: NonEmptyString | None = None
    dataset_version: NonEmptyString
    architecture_variant: NonEmptyString
    model_provider: NonEmptyString | None = None
    model_name: NonEmptyString | None = None
    model_version: NonEmptyString | None = None
    prompt_version: NonEmptyString | None = None
    retrieval_configuration: dict[NonEmptyString, ConfigScalar] | None = None
    random_seed: int | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("timestamp")
    @classmethod
    def require_timezone_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value


class ExperimentResultMetadata(BaseModel):
    """Traceability metadata embedded alongside experiment results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: ExperimentConfig
