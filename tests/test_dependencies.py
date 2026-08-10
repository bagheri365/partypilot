"""Tests for deterministic task and resource dependency models."""

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from partypilot.domain.dependencies import (
    ResourceRequirement,
    ResourceRequirementMode,
    TaskDependency,
)
from partypilot.domain.temporal import Duration, TimeWindow


def window(start_hour: int, end_hour: int) -> TimeWindow:
    return TimeWindow(
        start=datetime(2026, 8, 15, start_hour),
        end=datetime(2026, 8, 15, end_hour),
    )


def test_setup_can_depend_on_venue_before_guest_arrival() -> None:
    setup = TaskDependency(
        task_id="setup",
        prerequisite_task_ids=("venue_access",),
        duration=Duration.minutes(45),
        permitted_time_window=window(14, 16),
        required_resources=(
            ResourceRequirement(
                resource_id="venue-main-room",
                mode=ResourceRequirementMode.EXCLUSIVE,
            ),
        ),
    )

    guest_arrival = TaskDependency(
        task_id="guest_arrival",
        prerequisite_task_ids=(setup.task_id,),
        duration=Duration.minutes(30),
        permitted_time_window=window(16, 17),
    )

    assert setup.prerequisite_task_ids == ("venue_access",)
    assert guest_arrival.prerequisite_task_ids == ("setup",)


def test_food_setup_can_be_required_before_meal() -> None:
    meal = TaskDependency(
        task_id="meal",
        prerequisite_task_ids=("food_setup",),
        duration=Duration.hours(1),
        permitted_time_window=window(17, 19),
        required_resources=(
            ResourceRequirement(resource_id="caterer-1", mode=ResourceRequirementMode.SHARED),
        ),
    )

    assert "food_setup" in meal.prerequisite_task_ids
    assert meal.shared_resource_ids == frozenset({"caterer-1"})


def test_activity_can_require_venue_as_prerequisite() -> None:
    activity = TaskDependency(
        task_id="magic-show",
        prerequisite_task_ids=("venue_access",),
        duration=Duration.minutes(60),
        permitted_time_window=window(15, 18),
    )

    assert activity.prerequisite_task_ids == ("venue_access",)


def test_exclusive_and_shared_resources_are_distinguished() -> None:
    task = TaskDependency(
        task_id="cake-cutting",
        duration=Duration.minutes(20),
        permitted_time_window=window(16, 17),
        required_resources=(
            ResourceRequirement(resource_id="main-stage", mode=ResourceRequirementMode.EXCLUSIVE),
            ResourceRequirement(resource_id="serving-staff", mode=ResourceRequirementMode.SHARED),
        ),
    )

    assert task.required_resource_ids == frozenset({"main-stage", "serving-staff"})
    assert task.exclusive_resource_ids == frozenset({"main-stage"})
    assert task.shared_resource_ids == frozenset({"serving-staff"})


def test_task_cannot_depend_on_itself() -> None:
    with pytest.raises(ValidationError, match="cannot depend on itself"):
        TaskDependency(
            task_id="setup",
            prerequisite_task_ids=("setup",),
            duration=Duration.minutes(30),
            permitted_time_window=window(14, 16),
        )


def test_prerequisite_ids_must_be_unique_and_nonempty() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        TaskDependency(
            task_id="meal",
            prerequisite_task_ids=("food_setup", "food_setup"),
            duration=Duration.minutes(30),
            permitted_time_window=window(17, 19),
        )

    with pytest.raises(ValidationError, match="cannot be empty"):
        TaskDependency(
            task_id="meal",
            prerequisite_task_ids=(" ",),
            duration=Duration.minutes(30),
            permitted_time_window=window(17, 19),
        )


def test_resource_ids_must_be_unique_per_task() -> None:
    with pytest.raises(ValidationError, match="unique resource IDs"):
        TaskDependency(
            task_id="setup",
            duration=Duration.minutes(30),
            permitted_time_window=window(14, 16),
            required_resources=(
                ResourceRequirement(resource_id="room", mode=ResourceRequirementMode.EXCLUSIVE),
                ResourceRequirement(resource_id="room", mode=ResourceRequirementMode.SHARED),
            ),
        )


def test_resource_id_cannot_be_blank() -> None:
    with pytest.raises(ValidationError, match="resource_id cannot be empty"):
        ResourceRequirement(resource_id=" ", mode=ResourceRequirementMode.EXCLUSIVE)


def test_task_duration_must_fit_permitted_window() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        TaskDependency(
            task_id="setup",
            duration=Duration.hours(3),
            permitted_time_window=window(14, 16),
        )


def test_models_are_immutable_and_reject_extra_fields() -> None:
    resource = ResourceRequirement(resource_id="room", mode=ResourceRequirementMode.EXCLUSIVE)

    with pytest.raises(ValidationError):
        resource.resource_id = "other"

    with pytest.raises(ValidationError):
        ResourceRequirement.model_validate(
            {
                "resource_id": "room",
                "mode": ResourceRequirementMode.EXCLUSIVE,
                "unexpected": True,
            }
        )


def test_duration_boundary_equal_to_window_is_allowed() -> None:
    task = TaskDependency(
        task_id="full-window-task",
        duration=Duration(value=timedelta(hours=2)),
        permitted_time_window=window(14, 16),
    )

    assert task.duration.value == task.permitted_time_window.duration.value


def test_task_id_cannot_be_blank() -> None:
    with pytest.raises(ValidationError, match="task_id cannot be empty"):
        TaskDependency(
            task_id=" ",
            duration=Duration.minutes(30),
            permitted_time_window=window(14, 16),
        )
