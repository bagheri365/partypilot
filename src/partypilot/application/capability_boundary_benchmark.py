"""Capability-boundary benchmark loading for future PartyPilot experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from partypilot.domain.evaluation import CapabilityBoundaryScenario

DATASET_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "evaluation"
    / "capability_boundary_scenarios.json"
)
MANIFEST_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "evaluation"
    / "capability_boundary_manifest.json"
)
BENCHMARK_VERSION = "1.0"


class CapabilityBoundaryBenchmarkManifest(BaseModel):
    """Frozen metadata for the expanded capability-boundary benchmark."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark_name: str
    benchmark_version: str
    scenario_count: int = Field(ge=0)
    scenarios_checksum_sha256: str
    notes: tuple[str, ...] = ()


def load_capability_boundary_scenarios(
    path: Path = DATASET_PATH,
) -> tuple[CapabilityBoundaryScenario, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TypeAdapter(tuple[CapabilityBoundaryScenario, ...]).validate_python(payload)


def compute_capability_boundary_checksum(path: Path = DATASET_PATH) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical_payload = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


def build_capability_boundary_manifest(
    scenarios: Sequence[CapabilityBoundaryScenario],
    *,
    path: Path = DATASET_PATH,
) -> CapabilityBoundaryBenchmarkManifest:
    return CapabilityBoundaryBenchmarkManifest(
        benchmark_name="capability-boundary",
        benchmark_version=BENCHMARK_VERSION,
        scenario_count=len(scenarios),
        scenarios_checksum_sha256=compute_capability_boundary_checksum(path),
        notes=(
            "This benchmark is frozen for architecture-comparison work.",
            "Any change to the scenario set must be explicit, versioned, and justified.",
            (
                "A simpler architecture matching or outperforming a multi-agent "
                "architecture is a valid result."
            ),
        ),
    )


def load_capability_boundary_manifest(
    path: Path = MANIFEST_PATH,
) -> CapabilityBoundaryBenchmarkManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CapabilityBoundaryBenchmarkManifest.model_validate(payload)
