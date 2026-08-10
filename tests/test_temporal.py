from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from partypilot.domain.temporal import Duration, ScheduledInterval, TemporalRequirements, TimeWindow


def dt(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, 12, hour, minute)


def test_duration_accepts_zero_and_positive_values() -> None:
    assert Duration().value == timedelta(0)
    assert Duration.minutes(30).value == timedelta(minutes=30)
    assert Duration.hours(2).value == timedelta(hours=2)


def test_duration_rejects_negative_values() -> None:
    with pytest.raises(ValidationError, match="duration cannot be negative"):
        Duration(value=timedelta(seconds=-1))


def test_time_window_accepts_equal_start_and_end() -> None:
    instant = dt(12)
    window = TimeWindow(start=instant, end=instant)
    assert window.duration == Duration()


def test_time_window_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError, match="end cannot precede start"):
        TimeWindow(start=dt(13), end=dt(12))


def test_time_window_contains_instants_inclusively() -> None:
    window = TimeWindow(start=dt(10), end=dt(12))
    assert window.contains(dt(10))
    assert window.contains(dt(11))
    assert window.contains(dt(12))
    assert not window.contains(dt(9, 59))


def test_time_window_contains_another_window() -> None:
    outer = TimeWindow(start=dt(10), end=dt(14))
    inner = TimeWindow(start=dt(11), end=dt(13))
    assert outer.contains_window(inner)
    assert not inner.contains_window(outer)


def test_temporal_requirements_default_to_zero() -> None:
    requirements = TemporalRequirements()
    assert requirements.setup_time == Duration()
    assert requirements.lead_time == Duration()


def test_temporal_requirements_reject_negative_setup_or_lead_time() -> None:
    with pytest.raises(ValidationError):
        TemporalRequirements(setup_time=Duration(value=timedelta(minutes=-1)))
    with pytest.raises(ValidationError):
        TemporalRequirements(lead_time=Duration(value=timedelta(minutes=-1)))


def test_scheduled_interval_exposes_duration_setup_start_and_required_by() -> None:
    interval = ScheduledInterval(
        start=dt(14),
        end=dt(16),
        setup_time=Duration.minutes(45),
        lead_time=Duration.hours(2),
    )
    assert interval.duration == Duration.hours(2)
    assert interval.setup_start == dt(13, 15)
    assert interval.required_by == dt(12)


def test_scheduled_interval_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError, match="end cannot precede start"):
        ScheduledInterval(start=dt(15), end=dt(14))


def test_temporal_models_are_immutable_and_forbid_unknown_fields() -> None:
    duration = Duration.minutes(15)
    with pytest.raises(ValidationError):
        duration.value = timedelta(minutes=20)

    with pytest.raises(ValidationError):
        TimeWindow.model_validate(
            {
                "start": dt(10),
                "end": dt(11),
                "timezone": "UTC",
            }
        )
