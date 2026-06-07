"""Grid diagrams for combinatorial knot Floer homology (engine Phase 6).

A grid diagram is an n x n toroidal grid with one O and one X marker in every row and
every column; its knot is read off by joining the O and X in each row (a horizontal
segment) and in each column (a vertical segment). KnotInfo tabulates the 2n markers as a
flat list of [row, col] pairs *without* an O/X labelling, so the labelling is recovered
here by tracing the knot through the markers -- alternating horizontal (shared row) and
vertical (shared column) arcs -- and 2-colouring the resulting cycle. For a knot (one
component) the markers form a single 2n-cycle, 2-colourable in exactly two ways related by
swapping O <-> X, i.e. by mirroring; we fix one colouring and reconcile chirality against
HFK later.

A generator of the grid (chain) complex is a permutation sigma of {0, ..., n-1}: it places
one point at (row i, column sigma[i]), one on each horizontal and each vertical grid
circle. There are n! of them -- the generation bottleneck noted in the roadmap.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from functools import cached_property

from ...errors import TetradromeError


class GridDiagram:
    """An n x n grid given by its O and X markers (``O[row]`` / ``X[row]`` = column)."""

    def __init__(self, o_markers, x_markers):
        n = len(o_markers)
        if len(x_markers) != n:
            raise TetradromeError("O and X marker counts differ.")
        if sorted(o_markers) != list(range(n)) or sorted(x_markers) != list(range(n)):
            raise TetradromeError("O and X must each place one marker per column (a permutation).")
        if any(o_markers[i] == x_markers[i] for i in range(n)):
            raise TetradromeError("O and X share a cell in some row.")
        self.n = n
        self.O = list(o_markers)
        self.X = list(x_markers)

    @classmethod
    def from_markers(cls, markers) -> "GridDiagram":
        """Build from KnotInfo-style [row, col] markers (1-based, 2n of them): trace the
        knot cycle through the markers and 2-colour it into O and X."""
        if len(markers) % 2:
            raise TetradromeError("grid has an odd number of markers.")
        n = len(markers) // 2
        pts = [(r - 1, c - 1) for r, c in markers]
        by_row: dict[int, list[int]] = defaultdict(list)
        by_col: dict[int, list[int]] = defaultdict(list)
        for i, (r, c) in enumerate(pts):
            by_row[r].append(i)
            by_col[c].append(i)
        if any(len(v) != 2 for v in by_row.values()) or any(len(v) != 2 for v in by_col.values()):
            raise TetradromeError("each row and column must hold exactly two markers.")

        def partner(groups, i, key):
            a, b = groups[key]
            return b if a == i else a

        color = {0: 0}
        cur, use_row = 0, True
        while True:
            nxt = (partner(by_row, cur, pts[cur][0]) if use_row
                   else partner(by_col, cur, pts[cur][1]))
            if nxt in color:
                break
            color[nxt] = 1 - color[cur]
            cur, use_row = nxt, not use_row
        if len(color) != 2 * n:
            raise TetradromeError(
                "markers do not form a single knot cycle (a link, or bad alternation)."
            )
        o = [0] * n
        x = [0] * n
        for i, (r, c) in enumerate(pts):
            (o if color[i] == 0 else x)[r] = c
        return cls(o, x)

    @classmethod
    def from_knotinfo(cls, name: str) -> "GridDiagram":
        from ...backends import knotinfo_backend as ki

        return cls.from_markers(ki.grid_notation(name))

    @cached_property
    def marker_cells(self) -> frozenset:
        """The 2n cells holding a marker (O or X), as (row, column) pairs."""
        return frozenset(
            [(r, self.O[r]) for r in range(self.n)] + [(r, self.X[r]) for r in range(self.n)]
        )

    def generators(self):
        """Iterator over generators: each is a permutation ``sigma`` placing a point at
        (row i, column sigma[i])."""
        return itertools.permutations(range(self.n))

    def __repr__(self) -> str:
        return f"GridDiagram(n={self.n}, O={self.O}, X={self.X})"
