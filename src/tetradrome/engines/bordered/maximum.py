# SPDX-License-Identifier: Apache-2.0

"""The DA bimodule Omega^c of a local maximum (arXiv:1603.06559, section 8).

Read downwards, the maximum opens two new strands, ``strand`` and ``strand + 1`` of the output
(the paper's c and c + 1), around the position ``strand + 1``. The upward one of the two is
given. A generator Q_y is an allowed output idempotent y: it holds the position inside the cap
and at most one of the two positions beside it. Its input idempotent psi'(y) forgets the inside
position and merges the two beside it.

delta^1_1 sends Q_y along R_{c+1} R_c, L_c L_{c+1} and the C U term on the new strands;
delta^1_2(Q_x, a) is Phi_x(a) (x) Q_y (Lemma 8.1): the input's weights and C's moved to the output
strands, zero weight on the new strands, y the allowed idempotent over the input's right
idempotent whose relative weight to x vanishes on the new strands. The lemma's proof pairs each of
its cases with the types of x it allows; for the other type no such y exists (Y with input
L_{c-1}, for one) and delta^1_2 is zero. Every generator has Maslov and Alexander grading 0. With
no strands above, this is the global maximum's type D structure (section 8.1).

Over a curved algebra (arXiv:1707.00597, section 5) the two new strands are matched to each
other, and the C U loop is gone: with the C's that delta^1_2 carries through it is C^2 (x) Q,
which the curved form drops (section 13.1).
"""
from __future__ import annotations

from ...errors import TetradromeError
from .algebra import FIELD, Algebra, CurvedAlgebra, Element
from .tensor import Generator


class MaximumBimodule:
    def __init__(self, upper: Algebra, strand: int, left_upward: bool):
        if not 0 <= strand <= upper.strands:
            raise TetradromeError(f"no maximum at strand {strand} below {upper}.")
        self.upper = upper
        self.strand = strand

        def inserted(s):
            return s if s < strand else s + 2

        upward = {inserted(s) for s in upper.upward}
        upward.add(strand if left_upward else strand + 1)
        if upper.curved:
            partner = [0] * (upper.strands + 2)
            partner[strand], partner[strand + 1] = strand + 1, strand
            for s, t in enumerate(upper.partner):
                partner[inserted(s)] = inserted(t)
            self.lower = CurvedAlgebra(upper.strands + 2, upper.occupied + 1, frozenset(upward),
                                       tuple(partner))
            self._loop_c = None
        else:
            self.lower = Algebra(upper.strands + 2, upper.occupied + 1, frozenset(upward))
            self._loop_c = 1 << strand if left_upward else 1 << strand + 1

    def _widened(self, state: int) -> int:
        low = state & ((1 << self.strand) - 1)
        high = state >> self.strand + 1 << self.strand + 3
        return low | high | 1 << self.strand + 1

    def _generator(self, state: int, right: int) -> Generator:
        kind = "X" if state >> self.strand & 1 else "Y" if state >> self.strand + 2 & 1 else "Z"
        return Generator(kind, state, right)

    def generators(self, right: int) -> list:
        base = self._widened(right)
        if right >> self.strand & 1:
            return [self._generator(base | 1 << self.strand, right),
                    self._generator(base | 1 << self.strand + 2, right)]
        return [self._generator(base, right)]

    def maslov(self, generator: Generator) -> int:
        return 0

    def alexander(self, generator: Generator) -> int:
        return 0

    def input_limit(self, typed) -> int:
        return 1

    def start(self, generator: Generator) -> tuple:
        return (generator, None)

    def accepts_more(self, state: tuple) -> bool:
        return state[1] is None

    def advance(self, state: tuple, element: Element) -> tuple | None:
        generator, fed = state
        if fed is None:
            return (generator, element)
        return None

    def emit(self, state: tuple) -> set:
        generator, fed = state
        if fed is None:
            return self._delta_one(generator)
        return self._delta_two(generator, fed)

    def _delta_one(self, generator: Generator) -> set:
        lower, c = self.lower, self.strand
        state = generator.left
        across = 1 << FIELD * c | 1 << FIELD * (c + 1)
        candidates = [] if self._loop_c is None else [(state, 2 * across, self._loop_c)]
        if generator.kind != "Z":
            candidates.append((state ^ (1 << c) ^ (1 << c + 2), across, 0))
        terms: set = set()
        for target_state, weights, c_strands in candidates:
            element = lower.element(state, target_state, weights, c_strands)
            if element is not None:
                terms ^= {(element, self._generator(target_state, generator.right))}
        return terms

    def _delta_two(self, generator: Generator, fed: Element) -> set:
        lower, c = self.lower, self.strand
        low = (1 << FIELD * c) - 1
        weights = fed.weights & low | (fed.weights >> FIELD * c) << FIELD * (c + 2)
        c_strands = fed.c_strands & ((1 << c) - 1) | (fed.c_strands >> c) << c + 2
        levels = lower.levels(generator.left)
        matches = [target for target in self.generators(fed.right)
                   if lower.levels(target.left)[c:c + 2] == levels[c:c + 2]]
        if not matches:
            return set()
        if len(matches) > 1:
            raise TetradromeError(f"Lemma 8.1 gives {len(matches)} output idempotents for an input "
                                  f"from {fed.left} to {fed.right}.")
        target = matches[0]
        element = lower.element(generator.left, target.left, weights, c_strands)
        if element is None:
            return set()
        return {(element, target)}
