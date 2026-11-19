# SPDX-License-Identifier: Apache-2.0

"""A knot diagram cut into elementary pieces between horizontal levels.

Read bottom to top, a Morse presentation is a sequence of pieces acting on the strands that cross
each level, numbered from 0 at the left:

- ``("min", i)``: a local minimum; two new strands at positions i and i + 1.
- ``("max", i)``: a local maximum joining the strands at i and i + 1.
- ``("cross", i, over)``: the strands at i and i + 1 cross; ``over`` is True when the strand from
  the lower left (position i below, i + 1 above) passes over.

It is built from the leveling (``leveling.py``) of the reduced diagram: each crossing becomes one
crossing piece with the minima and maxima around it, so a diagram of c crossings has c crossing
pieces, and as many minima as maxima. ``directions`` orients the knot by walking it.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..errors import TetradromeError
from .leveling import sweep
from .model import PDCode


@dataclass(frozen=True)
class MorseDiagram:
    pieces: tuple

    def widths(self) -> list:
        """The number of strands at each level: below the first piece, and above each piece."""
        widths = [0]
        for piece in self.pieces:
            widths.append(widths[-1] + {"min": 2, "max": -2, "cross": 0}[piece[0]])
        if widths[-1] != 0 or min(widths) < 0:
            raise TetradromeError("the pieces do not close up: strands are left open.")
        return widths

    def directions(self) -> list:
        """For each level (as ``widths`` counts them), each strand's direction along the knot:
        True for upward. The walk starts up the left strand of the first minimum."""
        widths = self.widths()
        upward = [[None] * width for width in widths]

        def above(level, position):
            # Where the strand at (level, position) goes through the piece above it.
            kind, at = self.pieces[level][0], self.pieces[level][1]
            if kind == "min":
                return ("up", level + 1, position + 2 if position >= at else position)
            if kind == "max":
                if position in (at, at + 1):
                    return ("turn", level, at + 1 if position == at else at)
                return ("up", level + 1, position - 2 if position > at + 1 else position)
            if position in (at, at + 1):
                return ("up", level + 1, at + 1 if position == at else at)
            return ("up", level + 1, position)

        def below(level, position):
            # Where the strand at (level, position) goes through the piece below it.
            kind, at = self.pieces[level - 1][0], self.pieces[level - 1][1]
            if kind == "min":
                if position in (at, at + 1):
                    return ("turn", level, at + 1 if position == at else at)
                return ("down", level - 1, position - 2 if position > at + 1 else position)
            if kind == "max":
                return ("down", level - 1, position + 2 if position >= at else position)
            if position in (at, at + 1):
                return ("down", level - 1, at + 1 if position == at else at)
            return ("down", level - 1, position)

        if self.pieces[0][0] != "min":
            raise TetradromeError("a Morse presentation starts with a minimum.")
        level, position, going_up = 1, self.pieces[0][1], True
        while upward[level][position] is None:
            upward[level][position] = going_up
            step, level, position = (above if going_up else below)(level, position)
            if step == "turn":
                going_up = not going_up
        if any(None in row for row in upward):
            raise TetradromeError("the Morse presentation has more than one component.")
        return [tuple(row) for row in upward]


def morse_diagram(diagram: PDCode) -> MorseDiagram:
    """The Morse presentation of a reduced diagram (``reduction.reduced``)."""
    if not diagram:
        return MorseDiagram((("min", 0), ("max", 0)))
    # Crossing 0: two cups side by side and a crossing between them; its lower-left strand is
    # PD positions 0 and 2, the understrand.
    pieces = [("min", 0), ("min", 2), ("cross", 1, False)]
    for _, i, s, p in sweep(diagram):
        # Positions 1 and 3 of a PD crossing are its overstrand.
        if p == 1:
            pieces += [("min", i + 1), ("cross", i, s % 2 == 1)]
        elif p == 2:
            pieces += [("cross", i, s % 2 == 1)]
        elif p == 3:
            pieces += [("cross", i + 1, s % 2 == 0), ("max", i)]
        else:
            pieces += [("cross", 1, s % 2 == 0), ("max", 0), ("max", 0)]
    return MorseDiagram(tuple(pieces))


def pd_from_morse(morse: MorseDiagram) -> PDCode:
    """The PD code the Morse presentation draws, in KnotInfo's labelling."""
    from .reduction import relabellings

    parent: dict = {}

    def find(arc):
        while parent.setdefault(arc, arc) != arc:
            arc = parent[arc]
        return arc

    strands: list = []
    crossings = []
    fresh = 0
    for piece in morse.pieces:
        kind, at = piece[0], piece[1]
        if kind == "min":
            fresh += 1
            strands[at:at] = [fresh, fresh]
        elif kind == "max":
            parent[find(strands[at])] = find(strands[at + 1])
            del strands[at:at + 2]
        else:
            lower_left, lower_right = strands[at], strands[at + 1]
            upper_right, upper_left = fresh + 1, fresh + 2
            fresh += 2
            # Counterclockwise from the lower left; the understrand at positions 0 and 2.
            around = [lower_left, lower_right, upper_right, upper_left]
            crossings.append(around[1:] + around[:1] if piece[2] else around)
            strands[at], strands[at + 1] = upper_left, upper_right
    return relabellings([[find(arc) for arc in crossing] for crossing in crossings])[0]
