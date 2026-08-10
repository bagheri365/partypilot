from datetime import datetime

from partypilot.domain import (
    Duration,
    ResourceRequirement,
    ResourceRequirementMode,
    ScheduledInterval,
    TaskDependency,
    TemporalViolationCode,
    TimeWindow,
    validate_temporal_schedule,
)


def dt(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 8, 15, hour, minute)


def task(
    task_id: str,
    *,
    start: int = 9,
    end: int = 18,
    duration_minutes: int = 60,
    prerequisites: tuple[str, ...] = (),
    resources: tuple[ResourceRequirement, ...] = (),
) -> TaskDependency:
    return TaskDependency(
        task_id=task_id,
        prerequisite_task_ids=prerequisites,
        duration=Duration.minutes(duration_minutes),
        permitted_time_window=TimeWindow(start=dt(start), end=dt(end)),
        required_resources=resources,
    )


def test_valid_schedule_has_no_violations() -> None:
    setup = task("setup", duration_minutes=60)
    activity = task("activity", duration_minutes=60, prerequisites=("setup",))
    schedule = {
        "setup": ScheduledInterval(start=dt(10), end=dt(11)),
        "activity": ScheduledInterval(start=dt(11), end=dt(12)),
    }

    result = validate_temporal_schedule(
        (setup, activity), schedule, TimeWindow(start=dt(9), end=dt(18))
    )

    assert result.is_valid
    assert result.violations == ()


def test_detects_task_outside_permitted_window() -> None:
    activity = task("activity", start=10, end=12)
    schedule = {"activity": ScheduledInterval(start=dt(9), end=dt(10))}

    result = validate_temporal_schedule((activity,), schedule, TimeWindow(start=dt(8), end=dt(18)))

    assert {violation.code for violation in result.violations} == {
        TemporalViolationCode.OUTSIDE_PERMITTED_WINDOW
    }


def test_detects_dependency_scheduled_in_wrong_order() -> None:
    food_setup = task("food-setup")
    meal = task("meal", prerequisites=("food-setup",))
    schedule = {
        "food-setup": ScheduledInterval(start=dt(12), end=dt(13)),
        "meal": ScheduledInterval(start=dt(12), end=dt(13)),
    }

    result = validate_temporal_schedule(
        (food_setup, meal), schedule, TimeWindow(start=dt(9), end=dt(18))
    )

    assert TemporalViolationCode.DEPENDENCY_ORDER in {v.code for v in result.violations}


def test_detects_overlapping_exclusive_resource() -> None:
    venue = ResourceRequirement(resource_id="venue-a", mode=ResourceRequirementMode.EXCLUSIVE)
    first = task("first", resources=(venue,))
    second = task("second", resources=(venue,))
    schedule = {
        "first": ScheduledInterval(start=dt(10), end=dt(11)),
        "second": ScheduledInterval(start=dt(10, 30), end=dt(11, 30)),
    }

    result = validate_temporal_schedule(
        (first, second), schedule, TimeWindow(start=dt(9), end=dt(18))
    )

    overlap = next(
        violation
        for violation in result.violations
        if violation.code is TemporalViolationCode.EXCLUSIVE_RESOURCE_OVERLAP
    )
    assert overlap.resource_id == "venue-a"
    assert overlap.related_task_ids == ("second",)


def test_shared_resource_can_overlap() -> None:
    shared = ResourceRequirement(resource_id="table", mode=ResourceRequirementMode.SHARED)
    first = task("first", resources=(shared,))
    second = task("second", resources=(shared,))
    schedule = {
        "first": ScheduledInterval(start=dt(10), end=dt(11)),
        "second": ScheduledInterval(start=dt(10, 30), end=dt(11, 30)),
    }

    result = validate_temporal_schedule(
        (first, second), schedule, TimeWindow(start=dt(9), end=dt(18))
    )

    assert result.is_valid


def test_detects_setup_completing_too_late_for_window() -> None:
    setup = task("setup", start=10, end=13)
    schedule = {
        "setup": ScheduledInterval(
            start=dt(10),
            end=dt(11),
            setup_time=Duration.minutes(30),
        )
    }

    result = validate_temporal_schedule((setup,), schedule, TimeWindow(start=dt(9), end=dt(18)))

    assert TemporalViolationCode.SETUP_TOO_LATE in {v.code for v in result.violations}


def test_detects_plan_extending_beyond_event_end() -> None:
    activity = task("activity", start=16, end=19)
    schedule = {"activity": ScheduledInterval(start=dt(17), end=dt(18))}

    result = validate_temporal_schedule((activity,), schedule, TimeWindow(start=dt(9), end=dt(17)))

    assert TemporalViolationCode.BEYOND_EVENT_END in {v.code for v in result.violations}


def test_detects_missing_schedule_and_missing_prerequisite_schedule() -> None:
    setup = task("setup")
    activity = task("activity", prerequisites=("setup",))
    schedule = {"activity": ScheduledInterval(start=dt(11), end=dt(12))}

    result = validate_temporal_schedule(
        (setup, activity), schedule, TimeWindow(start=dt(9), end=dt(18))
    )

    codes = [violation.code for violation in result.violations]
    assert TemporalViolationCode.MISSING_SCHEDULE in codes
    assert TemporalViolationCode.MISSING_PREREQUISITE_SCHEDULE in codes


def test_touching_exclusive_intervals_do_not_overlap() -> None:
    venue = ResourceRequirement(resource_id="venue-a", mode=ResourceRequirementMode.EXCLUSIVE)
    first = task("first", resources=(venue,))
    second = task("second", resources=(venue,))
    schedule = {
        "first": ScheduledInterval(start=dt(10), end=dt(11)),
        "second": ScheduledInterval(start=dt(11), end=dt(12)),
    }

    result = validate_temporal_schedule(
        (first, second), schedule, TimeWindow(start=dt(9), end=dt(18))
    )

    assert result.is_valid
