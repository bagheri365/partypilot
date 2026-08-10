"""Deterministic temporal domain models."""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Duration(BaseModel):
    """A non-negative duration represented by a standard-library timedelta."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: timedelta = Field(default=timedelta(0))

    @model_validator(mode="after")
    def validate_non_negative(self) -> Duration:
        if self.value < timedelta(0):
            raise ValueError("duration cannot be negative")
        return self

    @classmethod
    def minutes(cls, value: int | float) -> Duration:
        """Construct a duration from a number of minutes."""
        return cls(value=timedelta(minutes=value))

    @classmethod
    def hours(cls, value: int | float) -> Duration:
        """Construct a duration from a number of hours."""
        return cls(value=timedelta(hours=value))


class TimeWindow(BaseModel):
    """An inclusive permitted window for an event or task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_order(self) -> TimeWindow:
        if self.end < self.start:
            raise ValueError("end cannot precede start")
        return self

    @property
    def duration(self) -> Duration:
        """Return the deterministic length of the window."""
        return Duration(value=self.end - self.start)

    def contains(self, instant: datetime) -> bool:
        """Return whether an instant falls inside the inclusive window."""
        return self.start <= instant <= self.end

    def contains_window(self, other: TimeWindow) -> bool:
        """Return whether another window is fully contained in this window."""
        return self.start <= other.start and other.end <= self.end


class TemporalRequirements(BaseModel):
    """Temporal requirements associated with an event or resource."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    setup_time: Duration = Field(default_factory=Duration)
    lead_time: Duration = Field(default_factory=Duration)


class ScheduledInterval(BaseModel):
    """A concrete scheduled interval with optional setup and lead requirements."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: datetime
    end: datetime
    setup_time: Duration = Field(default_factory=Duration)
    lead_time: Duration = Field(default_factory=Duration)

    @model_validator(mode="after")
    def validate_order(self) -> ScheduledInterval:
        if self.end < self.start:
            raise ValueError("end cannot precede start")
        return self

    @property
    def duration(self) -> Duration:
        """Return the duration between start and end."""
        return Duration(value=self.end - self.start)

    @property
    def setup_start(self) -> datetime:
        """Return when setup must begin."""
        return self.start - self.setup_time.value

    @property
    def required_by(self) -> datetime:
        """Return the latest point by which prerequisites must be available."""
        return self.start - self.lead_time.value
