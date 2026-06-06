"""The normalized diagram: Tetradrome's single internal representation of a knot.

Everything downstream (backends, invariants, validation, reports) consumes this,
not a backend-specific object (SPEC 3, 9).
"""
from __future__ import annotations

from dataclasses import dataclass

# A PD code: one 4-tuple of arc labels per crossing.
PDCode = tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class NormalizedDiagram:
    """A validated PD code plus how it was obtained.

    `identity` is the canonical KnotInfo name when the diagram came from one, else
    None (e.g. a raw PD code with no known name). Frozen: a normalized diagram is a
    value, not mutable state.
    """

    pd_code: PDCode
    source_notation: str  # how it entered: "name", "pd", ... (future: "dt", "braid")
    identity: str | None = None

    @property
    def crossing_number(self) -> int:
        return len(self.pd_code)
