"""Deterministic temporal validation for scheduled PartyPilot tasks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from partypilot.domain.dependencies import TaskDependency
from partypilot.domain.temporal import ScheduledInterval, TimeWindow


class TemporalViolationCode(StrEnum):
    """Machine-readable temporal validation failure categories."""

    OUTSIDE_PERMITTED_WINDOW = "outside_permitted_window"
    DEPENDENCY_ORDER = "dependency_order"
    EXCLUSIVE_RESOURCE_OVERLAP = "exclusive_resource_overlap"
    SETUP_TOO_LATE = "setup_too_late"
    BEYOND_EVENT_END = "beyond_event_end"
    MISSING_SCHEDULE = "missing_schedule"
    MISSING_PREREQUISITE_SCHEDULE = "missing_prerequisite_schedule"


class TemporalViolation(BaseModel):
    """A typed deterministic schedule validation violation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: TemporalViolationCode
    task_id: str
    message: str
    related_task_ids: tuple[str, ...] = Field(default_factory=tuple)
    resource_id: str | None = None

    @field_validator("task_id", "message")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value cannot be empty")
        return normalized

    @field_validator("related_task_ids")
    @classmethod
    def validate_related_task_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("related task IDs cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("related task IDs must be unique")
        return normalized

    @field_validator("resource_id")
    @classmethod
    def validate_resource_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("resource_id cannot be empty")
        return normalized


class ScheduledTask(BaseModel):
    """A task dependency paired with its concrete scheduled interval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskDependency
    interval: ScheduledInterval

    @model_validator(mode="after")
    def validate_duration_matches(self) -> ScheduledTask:
        if self.interval.duration != self.task.duration:
            raise ValueError("scheduled interval duration must match task duration")
        return self


class TemporalValidationResult(BaseModel):
    """Result of deterministic temporal validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    violations: tuple[TemporalViolation, ...] = Field(default_factory=tuple)

    @property
    def is_valid(self) -> bool:
        """Return whether the schedule contains no temporal violations."""
        return not self.violations


def validate_temporal_schedule(
    tasks: Iterable[TaskDependency],
    schedule: Mapping[str, ScheduledInterval],
    event_window: TimeWindow,
) -> TemporalValidationResult:
    """Validate deterministic temporal constraints for a set of scheduled tasks."""
    task_by_id = {task.task_id: task for task in tasks}
    violations: list[TemporalViolation] = []

    for task_id, task in task_by_id.items():
        interval = schedule.get(task_id)
        if interval is None:
            violations.append(
                TemporalViolation(
                    code=TemporalViolationCode.MISSING_SCHEDULE,
                    task_id=task_id,
                    message=f"Task {task_id!r} has no scheduled interval.",
                )
            )
            continue

        if interval.duration != task.duration:
            violations.append(
                TemporalViolation(
                    code=TemporalViolationCode.OUTSIDE_PERMITTED_WINDOW,
                    task_id=task_id,
                    message=(
                        f"Task {task_id!r} has a scheduled duration that does not match "
                        "its declared duration."
                    ),
                )
            )

        interval_window = TimeWindow(start=interval.start, end=interval.end)
        if not task.permitted_time_window.contains_window(interval_window):
            violations.append(
                TemporalViolation(
                    code=TemporalViolationCode.OUTSIDE_PERMITTED_WINDOW,
                    task_id=task_id,
                    message=f"Task {task_id!r} is scheduled outside its permitted time window.",
                )
            )

        if (
            interval.setup_time.value.total_seconds() > 0
            and interval.setup_start < task.permitted_time_window.start
        ):
            violations.append(
                TemporalViolation(
                    code=TemporalViolationCode.SETUP_TOO_LATE,
                    task_id=task_id,
                    message=(
                        f"Task {task_id!r} setup would need to begin before its permitted "
                        "time window."
                    ),
                )
            )

        if interval.end > event_window.end:
            violations.append(
                TemporalViolation(
                    code=TemporalViolationCode.BEYOND_EVENT_END,
                    task_id=task_id,
                    message=f"Task {task_id!r} extends beyond the event end time.",
                )
            )

        for prerequisite_id in task.prerequisite_task_ids:
            prerequisite_interval = schedule.get(prerequisite_id)
            if prerequisite_interval is None:
                violations.append(
                    TemporalViolation(
                        code=TemporalViolationCode.MISSING_PREREQUISITE_SCHEDULE,
                        task_id=task_id,
                        related_task_ids=(prerequisite_id,),
                        message=(
                            f"Prerequisite task {prerequisite_id!r} for task {task_id!r} "
                            "has no scheduled interval."
                        ),
                    )
                )
                continue

            required_by = interval.required_by
            if prerequisite_interval.end > required_by:
                violations.append(
                    TemporalViolation(
                        code=TemporalViolationCode.DEPENDENCY_ORDER,
                        task_id=task_id,
                        related_task_ids=(prerequisite_id,),
                        message=(
                            f"Prerequisite task {prerequisite_id!r} finishes after task "
                            f"{task_id!r} requires it."
                        ),
                    )
                )

    task_ids = list(task_by_id)
    for index, left_id in enumerate(task_ids):
        left_interval = schedule.get(left_id)
        if left_interval is None:
            continue
        left_task = task_by_id[left_id]

        for right_id in task_ids[index + 1 :]:
            right_interval = schedule.get(right_id)
            if right_interval is None:
                continue
            right_task = task_by_id[right_id]

            overlapping_resources = (
                left_task.exclusive_resource_ids & right_task.required_resource_ids
            ) | (right_task.exclusive_resource_ids & left_task.required_resource_ids)
            if not overlapping_resources:
                continue
            if not _intervals_overlap(left_interval, right_interval):
                continue

            for resource_id in sorted(overlapping_resources):
                violations.append(
                    TemporalViolation(
                        code=TemporalViolationCode.EXCLUSIVE_RESOURCE_OVERLAP,
                        task_id=left_id,
                        related_task_ids=(right_id,),
                        resource_id=resource_id,
                        message=(
                            f"Tasks {left_id!r} and {right_id!r} overlap while requiring "
                            f"exclusive resource {resource_id!r}."
                        ),
                    )
                )

    return TemporalValidationResult(violations=tuple(violations))


def _intervals_overlap(left: ScheduledInterval, right: ScheduledInterval) -> bool:
    """Return whether two half-open scheduled intervals overlap."""
    return left.start < right.end and right.start < left.end
