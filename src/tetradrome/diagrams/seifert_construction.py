"""Native Seifert's algorithm, stage 1: PD code -> oriented Seifert structure.

Recovers orientation and crossing signs from the PD code, performs the
orientation-preserving (Seifert) smoothing, and reads off the Seifert circles.
This is the foundation the Seifert matrix is built on (stage 2); on its own it
yields the writhe and the Seifert-algorithm genus.

Validated at scale: the Seifert genus computed here equals KnotInfo's three_genus
for every alternating knot in KnotInfo (6729/6729), where Seifert's algorithm is
known to give a minimal-genus surface (Crowell-Murasugi). For non-alternating
knots it is an upper bound on the three-genus.

PD convention (KnotInfo / KnotTheory`): each crossing is [a, b, c, d] with arc
labels in counterclockwise order, the understrand oriented a -> c (so c = a + 1,
mod 2n, in the global edge numbering), and b, d the overstrand. We never guess: a
crossing that violates the convention raises rather than being coerced.
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import PDCode


@dataclass(frozen=True)
class SeifertStructure:
    seifert_circles: int
    crossing_signs: tuple[int, ...]

    @property
    def writhe(self) -> int:
        return sum(self.crossing_signs)

    @property
    def genus(self) -> int:
        # Connected diagram of a knot: 2g = crossings - circles + 1.
        n = len(self.crossing_signs)
        return (n - self.seifert_circles + 1) // 2


def seifert_structure(pd: PDCode) -> SeifertStructure:
    n = len(pd)
    if n == 0:
        return SeifertStructure(seifert_circles=1, crossing_signs=())  # 0-crossing unknot

    edges = 2 * n

    def nxt(e: int) -> int:
        return e % edges + 1

    successor: dict[int, int] = {}  # Seifert smoothing as a permutation on edges
    signs: list[int] = []

    for a, b, c, d in pd:
        if c != nxt(a):
            raise ValueError(
                f"PD crossing {(a, b, c, d)} violates the understrand convention (expected c == a+1)."
            )
        if d == nxt(b):
            over_in, over_out = b, d
        elif b == nxt(d):
            over_in, over_out = d, b
        else:
            raise ValueError(
                f"PD crossing {(a, b, c, d)} has no consistent overstrand orientation."
            )

        # Orientation-preserving smoothing: under_in -> over_out, over_in -> under_out.
        successor[a] = over_out
        successor[over_in] = c

        # Crossing sign from CCW unit positions a@0, b@90, c@180, d@270.
        position = {a: (1, 0), b: (0, 1), c: (-1, 0), d: (0, -1)}
        ux, uy = position[c][0] - position[a][0], position[c][1] - position[a][1]
        ox, oy = (
            position[over_out][0] - position[over_in][0],
            position[over_out][1] - position[over_in][1],
        )
        signs.append(1 if ux * oy - uy * ox > 0 else -1)

    seen: set[int] = set()
    circles = 0
    for start in range(1, edges + 1):
        if start in seen:
            continue
        circles += 1
        e = start
        while e not in seen:
            seen.add(e)
            e = successor[e]

    return SeifertStructure(seifert_circles=circles, crossing_signs=tuple(signs))
