# SPDX-License-Identifier: Apache-2.0

"""The PD code of a braid's closure.

The braid runs upward with its strands numbered left to right; generator ``+j`` crosses the
strand at position j over the one at j + 1, ``-j`` under it (the convention of ``from_braid``
and the Seifert matrix built from a braid). Closing the braid joins each strand's top to the
bottom of the same position. Labels come out in KnotInfo's convention: 1..2n along the knot,
each crossing listed counterclockwise from its incoming understrand.
"""
from __future__ import annotations

from .model import PDCode
from .reduction import relabellings


def pd_from_braid(word) -> PDCode:
    """The closure of the braid ``word`` as a PD code. A word whose closure has several
    components is a link and raises."""
    word = [int(generator) for generator in word]
    if not word or any(generator == 0 for generator in word):
        raise ValueError("a braid word is a nonempty list of nonzero generators.")
    strands = max(abs(generator) for generator in word) + 1
    bottom = list(range(1, strands + 1))
    current = list(bottom)
    fresh = strands
    crossings = []
    for generator in word:
        left = abs(generator) - 1
        left_in, right_in = current[left], current[left + 1]
        left_out, right_out = fresh + 1, fresh + 2
        fresh += 2
        # Around the crossing counterclockwise: lower left, lower right, upper right, upper left.
        # The strand from the lower left ends upper right; the one from the lower right, upper left.
        if generator > 0:
            crossings.append([right_in, right_out, left_out, left_in])
        else:
            crossings.append([left_in, right_in, right_out, left_out])
        current[left], current[left + 1] = left_out, right_out
    closing = dict(zip(current, bottom))
    crossings = [[closing.get(label, label) for label in crossing] for crossing in crossings]
    curves = len({_component(crossings, label) for label in bottom})
    if curves != 1:
        raise ValueError(f"the braid's closure is a {curves}-component link, not a knot.")
    return relabellings(crossings)[0]


def _component(crossings, label) -> int:
    """The least label on the closed curve through arc ``label``: strands run straight through
    each crossing, position p to p + 2."""
    ends: dict = {}
    for index, crossing in enumerate(crossings):
        for position, arc in enumerate(crossing):
            ends.setdefault(arc, []).append((index, position))
    if label not in ends:
        return label                # a strand no generator touches closes on its own
    seen = {label}
    index, position = ends[label][0]
    while True:
        arc = crossings[index][(position + 2) % 4]
        if arc in seen:
            return min(seen)
        seen.add(arc)
        first, second = ends[arc]
        index, position = second if first == (index, (position + 2) % 4) else first
