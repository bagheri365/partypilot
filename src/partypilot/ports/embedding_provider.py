from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Infrastructure-independent interface for embedding text."""

    def embed(
        self,
        texts: Sequence[str],
        *,
        timeout_seconds: float,
    ) -> tuple[tuple[float, ...], ...]: ...
