# SPDX-License-Identifier: Apache-2.0

"""A bisected vertex leveling of a reduced knot diagram (No, Oh and Yoo, arXiv:1806.09719).

The crossings are put in an order v_1, ..., v_c, read as heights, so that v_1 and v_c lie on one
face and every cut between the first k and the rest leaves both sides connected. A reduced
diagram (``reduction.reduced``) has no cut vertex, so the order exists, and the greedy choice the
proof makes always finds the next crossing. Sweeping up through it, each crossing's arcs to
crossings below it are consecutive among the arcs crossing the sweep line, in the order its PD
entry lists them.

The face in the angle at crossing 0 from PD position 0 counterclockwise to position 1 is taken as
the outer one: the sweep starts at crossing 0, whose four arcs leave upward, left to right, in
positions 0, 3, 2, 1, and ends at another crossing on that face.
"""
from __future__ import annotations

from ..errors import TetradromeError
from .model import PDCode


def slots(pd: PDCode) -> dict:
    """Each arc label -> the two (crossing, position) slots it joins."""
    ends: dict = {}
    for index, crossing in enumerate(pd):
        for position, label in enumerate(crossing):
            ends.setdefault(label, []).append((index, position))
    return ends


def _connected(vertices: set, neighbours) -> bool:
    if not vertices:
        return True
    start = next(iter(vertices))
    seen = {start}
    stack = [start]
    while stack:
        for neighbour in neighbours[stack.pop()] & vertices:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen == vertices


def leveling(pd: PDCode) -> list:
    """The crossings of a reduced diagram with three or more crossings, bottom to top."""
    ends = slots(pd)

    def across(index, position):
        first, second = ends[pd[index][position]]
        return second if first == (index, position) else first

    face = []
    index, angle = 0, 0
    while True:
        face.append(index)
        index, angle = across(index, (angle + 1) % 4)
        if (index, angle) == (0, 0):
            break
    last = next(crossing for crossing in face if crossing != 0)

    neighbours = [set() for _ in pd]
    for (a, _), (b, _) in ends.values():
        neighbours[a].add(b)
        neighbours[b].add(a)
    order = [0]
    rest = set(range(len(pd))) - {0}
    while len(rest) > 1:
        # Of the crossings that may come next, the one with the most arcs reaching below the
        # sweep line narrows the line most: each such arc leaves it, each other arc joins it.
        reaching = {candidate: sum(1 for position in range(4)
                                   if across(candidate, position)[0] not in rest)
                    for candidate in rest - {last}}
        best = next((candidate for candidate in sorted(reaching,
                                                       key=lambda c: (-reaching[c], c))
                     if reaching[candidate] and _connected(rest - {candidate}, neighbours)),
                    None)
        if best is None:
            raise TetradromeError("no crossing continues the leveling; the diagram is not "
                                  "reduced.")
        order.append(best)
        rest.remove(best)
    order.append(last)
    return order


def sweep(pd: PDCode):
    """Walk the leveling upward. The sweep line is the arcs it cuts, left to right; it starts as
    crossing 0's arcs in positions 0, 3, 2, 1. Yields, for each crossing after the first,
    ``(crossing, i, s, p)``: its lower arcs are the line's arcs i .. i + p - 1, which are its PD
    positions s, s + 1, ..., s + p - 1, and its upper arcs replace them, left to right, in
    positions s - 1, s - 2, ..., s - (4 - p). The last crossing has p = 4."""
    ends = slots(pd)

    def across(index, position):
        first, second = ends[pd[index][position]]
        return second if first == (index, position) else first

    order = leveling(pd)
    placed = {order[0]}
    line = [pd[order[0]][position] for position in (0, 3, 2, 1)]
    for crossing in order[1:]:
        labels = pd[crossing]
        down = [position for position in range(4) if across(crossing, position)[0] in placed]
        where = {label: slot for slot, label in enumerate(line)}
        found = sorted(where[labels[position]] for position in down)
        i = found[0]
        s = labels.index(line[i])
        p = len(down)
        if found != list(range(i, i + p)) or any(
                line[i + t] != labels[(s + t) % 4] for t in range(p)):
            raise TetradromeError(f"crossing {crossing}'s lower arcs are not consecutive in the "
                                  f"sweep; the PD code is not a plane diagram.")
        yield crossing, i, s, p
        line[i:i + p] = [labels[(s - 1 - t) % 4] for t in range(4 - p)]
        placed.add(crossing)
