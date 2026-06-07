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
    """A validated diagram presentation -- a PD code and/or a braid word -- plus how
    it was obtained.

    `identity` is the canonical KnotInfo name when the diagram came from one, else
    None (e.g. a raw PD code or braid word with no known name). Frozen: a normalized
    diagram is a value, not mutable state.
    """

    pd_code: PDCode
    source_notation: str  # how it entered: "name", "pd", "braid_word"
    identity: str | None = None
    braid: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.pd_code and self.braid is None:
            raise ValueError("A diagram needs at least a PD code or a braid word.")

    @property
    def crossing_number(self) -> int:
        """Number of crossings in the PD diagram.

        Defined only when a PD code is present. A braid word alone gives the braid
        length (an upper bound on the crossing number, not the crossing number
        itself), so this raises for a braid-only diagram rather than mislead.
        """
        if not self.pd_code:
            raise ValueError(
                "crossing_number needs a PD diagram; this knot is braid-presented."
            )
        return len(self.pd_code)
