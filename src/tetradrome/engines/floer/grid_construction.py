# SPDX-License-Identifier: Apache-2.0

"""A grid diagram for any knot, built from its crossing diagram.

The construction is Bae and Park's: a reduced diagram with c crossings has an arc
presentation, and so a grid diagram, of size c + 2 (Bae-Park, Math. Proc. Camb. Phil. Soc. 129
(2000) 491-500), here by No, Oh and Yoo's bisected vertex leveling (arXiv:1806.09719):

1. Order the crossings so that the first and last share a face and every cut between the
   first k and the rest leaves both parts connected. A reduced diagram has no cut vertex, so
   such an order exists, and the greedy choice the proof makes always finds the next crossing.
2. Sweep up through that order drawing the diagram with horizontal and vertical segments.
   Each crossing becomes one horizontal segment across one vertical segment, the first and the
   last two segments each: c + 2 horizontal segments, each crossing at most one vertical one.
3. Shrink each vertical segment to a point on a binding axis and lift each horizontal segment
   to a page of an open book: in front of the axis when it passes over its vertical segment,
   behind it otherwise, lower segments on nearer pages. The pages in their order around the
   axis are the grid's rows and the points on the axis its columns.
"""
from __future__ import annotations

from dataclasses import dataclass

from ...diagrams.leveling import sweep
from ...diagrams.model import PDCode
from ...diagrams.reduction import reduced
from ...errors import TetradromeError
from .grid import GridDiagram


@dataclass(frozen=True)
class Drawing:
    """A knot drawn with horizontal and vertical segments. ``columns`` are the vertical
    segments left to right; ``spans`` gives each one's bottom and top height; ``horizontals``
    are ``(one end's column, the other end's column, the column crossed or None, passes over)``
    with horizontal segment k at height k."""

    columns: tuple
    spans: dict
    horizontals: tuple


def _passes_over(positions) -> bool:
    # Positions 1 and 3 of a PD crossing are its overstrand.
    return {position % 4 for position in positions} == {1, 3}


def drawing(pd: PDCode) -> Drawing:
    """The rectilinear drawing of a reduced diagram (``reduction.reduced``) with three or more
    crossings that the arc presentation is read from, swept up its leveling."""
    columns: list = []
    spans: dict = {}
    horizontals: list = []

    def new_column(bottom):
        column = len(spans)
        spans[column] = [bottom, None]
        return column

    # Crossing 0: its four arcs leave upward, left to right in positions 0, 3, 2, 1. The cup
    # on positions 3 and 1 lies lowest; the cup on 0 and 2 above it crosses the column of 3.
    up = [new_column(1), new_column(0), new_column(1), new_column(0)]
    columns.extend(up)
    horizontals.append((up[1], up[3], None, False))
    horizontals.append((up[0], up[2], up[1], _passes_over((0, 2))))
    line = list(up)                     # the sweep line's columns, left to right

    for _, i, s, p in sweep(pd):
        height = len(horizontals)
        block = line[i:i + p]
        if p == 4:
            # The last crossing: a cap on positions s and s+2 across the column of s+1, and
            # above it a cap on s+1 and s+3.
            spans[block[0]][1] = spans[block[2]][1] = height
            spans[block[1]][1] = spans[block[3]][1] = height + 1
            horizontals.append((block[0], block[2], block[1], _passes_over((s, s + 2))))
            horizontals.append((block[1], block[3], None, False))
            line = []
        elif p == 1:
            # A cup on positions s+3 and s+1 across the strand from s straight up to s+2.
            left, right = new_column(height), new_column(height)
            columns.insert(columns.index(block[0]), left)
            columns.insert(columns.index(block[0]) + 1, right)
            horizontals.append((left, right, block[0], _passes_over((s + 1, s + 3))))
            line[i:i + 1] = [left, block[0], right]
        elif p == 2:
            # The strand from s goes straight up to s+2; the one from s+1 turns left across it
            # to a new column for s+3.
            left = new_column(height)
            columns.insert(columns.index(block[0]), left)
            spans[block[1]][1] = height
            horizontals.append((left, block[1], block[0], _passes_over((s + 1, s + 3))))
            line[i:i + 2] = [left, block[0]]
        else:
            # A cap on positions s and s+2 across the strand from s+1 straight up to s+3.
            spans[block[0]][1] = spans[block[2]][1] = height
            horizontals.append((block[0], block[2], block[1], _passes_over((s, s + 2))))
            line[i:i + 3] = [block[1]]
    return Drawing(tuple(columns), {column: tuple(span) for column, span in spans.items()},
                   tuple(horizontals))


def _grid(rows: list) -> GridDiagram:
    """The grid whose row r joins the two columns ``rows[r]``: trace the knot through the
    markers, the marker each row is entered at an O and the one it is left from an X."""
    size = len(rows)
    rows_of_column: dict = {}
    for row, pair in enumerate(rows):
        for column in pair:
            rows_of_column.setdefault(column, []).append(row)
    o_markers, x_markers = [None] * size, [None] * size
    row, entered = 0, rows[0][0]
    while o_markers[row] is None:
        left = rows[row][1] if rows[row][0] == entered else rows[row][0]
        o_markers[row], x_markers[row] = entered, left
        a, b = rows_of_column[left]
        row, entered = (b if a == row else a), left
    if None in o_markers:
        raise TetradromeError("the grid's markers form more than one closed curve.")
    return GridDiagram(o_markers, x_markers)


def built(diagram: PDCode) -> GridDiagram:
    """The grid of size c + 2 for a reduced diagram with c crossings (``reduction.reduced``);
    the unknot's grid of size 2 for the empty diagram."""
    if not diagram:
        return GridDiagram([0, 1], [1, 0])
    drawn = drawing(diagram)
    rank = {column: x for x, column in enumerate(drawn.columns)}
    behind = [segment for segment in drawn.horizontals if not segment[3]]
    in_front = [segment for segment in reversed(drawn.horizontals) if segment[3]]
    return _grid([(rank[segment[0]], rank[segment[1]]) for segment in behind + in_front])


def grid_from_pd(pd: PDCode) -> GridDiagram:
    """A grid diagram of the knot with PD code ``pd``, of size at most its reduced crossing
    count plus two."""
    return built(reduced(pd))
