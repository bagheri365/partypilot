"""Deterministic task and resource dependency domain models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from partypilot.domain.temporal import Duration, TimeWindow


class ResourceRequirementMode(StrEnum):
    """Whether a resource may be shared while a task is using it."""

    EXCLUSIVE = "exclusive"
    SHARED = "shared"


class ResourceRequirement(BaseModel):
    """A task's requirement for a specific resource."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resource_id: str
    mode: ResourceRequirementMode

    @field_validator("resource_id")
    @classmethod
    def validate_resource_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("resource_id cannot be empty")
        return normalized


class TaskDependency(BaseModel):
    """Deterministic scheduling requirements for a single task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    prerequisite_task_ids: tuple[str, ...] = Field(default_factory=tuple)
    duration: Duration
    permitted_time_window: TimeWindow
    required_resources: tuple[ResourceRequirement, ...] = Field(default_factory=tuple)

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("task_id cannot be empty")
        return normalized

    @field_validator("prerequisite_task_ids")
    @classmethod
    def validate_prerequisite_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("prerequisite task IDs cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("prerequisite task IDs must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_relationships(self) -> TaskDependency:
        if self.task_id in self.prerequisite_task_ids:
            raise ValueError("a task cannot depend on itself")

        resource_ids = [requirement.resource_id for requirement in self.required_resources]
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError("resource requirements must use unique resource IDs")

        if self.duration.value > self.permitted_time_window.duration.value:
            raise ValueError("task duration cannot exceed its permitted time window")
        return self

    @property
    def required_resource_ids(self) -> frozenset[str]:
        """Return the resource IDs required by the task."""
        return frozenset(requirement.resource_id for requirement in self.required_resources)

    @property
    def exclusive_resource_ids(self) -> frozenset[str]:
        """Return resource IDs that this task requires exclusively."""
        return frozenset(
            requirement.resource_id
            for requirement in self.required_resources
            if requirement.mode is ResourceRequirementMode.EXCLUSIVE
        )

    @property
    def shared_resource_ids(self) -> frozenset[str]:
        """Return resource IDs that this task may share with other tasks."""
        return frozenset(
            requirement.resource_id
            for requirement in self.required_resources
            if requirement.mode is ResourceRequirementMode.SHARED
        )
