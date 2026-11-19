# SPDX-License-Identifier: Apache-2.0

"""The DA bimodules P^i and N^i of a crossing (arXiv:1603.06559, sections 5.2 to 5.6).

The crossing is between strands s and s + 1, the paper's i and i + 1, so its middle position is
s + 1. The input algebra is the upper slice's; the output's upward set swaps the two strands.
P^i is the crossing that is positive when both strands point up (the braid orientation).

P^i's operations come from the local model by Definitions 5.7 and 5.12: an input is replaced by
its type ``type_of``, the local terms are looked up, and each local term ``(word, kind)`` gives
the global element b running from the source's left idempotent to the target's, whose weight is
fixed by (5.10); it is kept when b exists and its type is the local word. C's off the two
strands pass through. N^i is the opposite bimodule: delta^1 of N^i on Y with inputs
o(a_l), ..., o(a_1) holds o(b) (x) X exactly when delta^1 of P^i on X with inputs a_1, ..., a_l
holds b (x) Y. Its gradings are the negatives of P^i's.

Generator gradings: the Alexander grading is (5.13) with the output's upward set. The Maslov
grading is -sign on the quadrant between the two outgoing ends and 0 on the other three, sign
being the crossing's sign as a knot crossing; this is (5.2) with the zero chosen as Figure 2
does, so that a state's grading is the sum of the local contributions of its quadrants.

Over a curved algebra (arXiv:1707.00597, sections 3.2 and 13) the output matching is the input's
with the two strands exchanged, and the operations are those above on C-free inputs, which are
the gamma operations of the paper's (13.1), except that when the two strands are not matched
to each other gamma_0 of S gains U_beta R_i (x) W + U_alpha L_{i+1} (x) E: the C-free parts of
delta^1_2(S, C_{i+1, beta}) and delta^1_2(S, C_{i, alpha}) in (3.11). N^i reverses them with
the rest.
"""
from __future__ import annotations

from ...errors import TetradromeError
from .algebra import FIELD, FIELD_MASK, Algebra, CurvedAlgebra, Element
from .local_crossing import Word, local_delta
from .tensor import Generator

# Four times the weight grading (4.5) of each generator of P^i, on strands s and s + 1.
_QUARTER_GRADINGS = {"N": (1, 1), "W": (1, -1), "E": (-1, 1), "S": (-1, -1)}


class CrossingBimodule:
    def __init__(self, upper: Algebra, strand: int, positive: bool):
        if not 0 <= strand < upper.strands - 1:
            raise TetradromeError(f"no crossing at strand {strand} of {upper}.")
        self.upper = upper
        self.strand = strand
        self.positive = positive

        def swap(s):
            return strand + 1 if s == strand else strand if s == strand + 1 else s

        swapped = frozenset(swap(s) for s in upper.upward)
        self._south_moves = None
        if upper.curved:
            partner = [0] * upper.strands
            for s, t in enumerate(upper.partner):
                partner[swap(s)] = swap(t)
            self.lower = CurvedAlgebra(upper.strands, upper.occupied, swapped, tuple(partner))
            alpha, beta = upper.partner[strand], upper.partner[strand + 1]
            if alpha != strand + 1:
                self._south_moves = (
                    ("W", 1 << FIELD * strand | 2 << FIELD * beta),
                    ("E", 1 << FIELD * (strand + 1) | 2 << FIELD * alpha))
        else:
            self.lower = Algebra(upper.strands, upper.occupied, swapped)
        self._types: dict = {}
        self._shift = FIELD * strand
        self._pair_fields = (FIELD_MASK | FIELD_MASK << FIELD) << self._shift
        left_up, right_up = strand in upper.upward, strand + 1 in upper.upward
        outgoing = {(True, True): "N", (True, False): "W",
                    (False, True): "E", (False, False): "S"}[(left_up, right_up)]
        orientation = 1 if positive else -1
        sign = (1 if left_up == right_up else -1) * orientation
        lower_signs = [-1 if s in self.lower.upward else 1 for s in (strand, strand + 1)]
        self._maslov = {kind: -sign if kind == outgoing else 0 for kind in _QUARTER_GRADINGS}
        self._alexander = {
            kind: orientation * (first * lower_signs[0] + second * lower_signs[1]) // 2
            for kind, (first, second) in _QUARTER_GRADINGS.items()}

    def generators(self, right: int) -> list:
        middle = self.strand + 1
        if right >> middle & 1:
            return [Generator("N", right, right)]
        found = [Generator("S", right, right)]
        if right >> middle - 1 & 1:
            found.append(Generator("W", right ^ (3 << middle - 1), right))
        if right >> middle + 1 & 1:
            found.append(Generator("E", right ^ (3 << middle), right))
        return found

    def maslov(self, generator: Generator) -> int:
        return self._maslov[generator.kind]

    def alexander(self, generator: Generator) -> int:
        return self._alexander[generator.kind]

    def input_limit(self, typed) -> int:
        return 2

    def start(self, generator: Generator) -> tuple:
        return (generator, ())

    def accepts_more(self, state: tuple) -> bool:
        generator, inputs = state
        if not inputs:
            return True
        if len(inputs) > 1 or (generator.kind == "S") != self.positive:
            return False
        # Every input pair of the list of section 5.1 moves or carries U's on the two strands in
        # each input, beyond powers of U1 U2 and the C's.
        word = self.type_of(inputs[0])
        return word.letters != "1" or word.u1 != word.u2

    def advance(self, state: tuple, element: Element) -> tuple | None:
        if self.accepts_more(state):
            return (state[0], state[1] + (element,))
        return None

    def emit(self, state: tuple) -> set:
        generator, inputs = state
        if self.positive:
            return self._positive(generator, inputs, None)
        terms: set = set()
        reversed_inputs = tuple(Element(a.right, a.left, a.weights, a.c_strands)
                                for a in reversed(inputs))
        right = inputs[-1].right if inputs else generator.right
        for source in self.generators(right):
            for element, target in self._positive(source, reversed_inputs, generator.kind):
                if target == generator:
                    terms ^= {(Element(element.right, element.left, element.weights,
                                       element.c_strands), source)}
        return terms

    def type_of(self, element: Element) -> Word:
        """The type t(a) of section 5.2 extended to C's (section 5.4): what a looks like near the
        crossing."""
        found = self._types.get(element)
        if found is not None:
            return found
        s = self.strand
        c1, c2 = element.c_strands >> s & 1, element.c_strands >> s + 1 & 1
        first = (element.weights >> self._shift & FIELD_MASK) - 2 * c1
        second = (element.weights >> self._shift + FIELD & FIELD_MASK) - 2 * c2
        before = self.upper.levels(element.left)
        after = self.upper.levels(element.right)
        if first & 1 and second & 1:
            letters = "R2R1" if before[s + 1] < after[s + 1] else "L1L2"
        elif second & 1:
            letters = "R2" if before[s + 1] < after[s + 1] else "L2"
        elif first & 1:
            letters = "R1" if before[s] < after[s] else "L1"
        else:
            letters = "1"
        found = Word(c1, c2, letters, first // 2, second // 2)
        self._types[element] = found
        return found

    def _positive(self, generator: Generator, inputs: tuple, only: str | None) -> set:
        """delta^1 of P^i on ``generator`` with ``inputs``; with ``only``, just the terms ending
        at generators of that kind."""
        if not inputs and generator.kind == "S" and self._south_moves is not None:
            return self._curved_south(generator, only)
        local = local_delta(generator.kind, tuple(self.type_of(a) for a in inputs))
        if only is not None:
            local = [term for term in local if term[1] == only]
        if not local:
            return set()
        s, shift = self.strand, self._shift
        crossing_mask = 3 << s
        outside = 0
        weights = 0
        for a in inputs:
            through = a.c_strands & ~crossing_mask
            if outside & through:
                return set()
            outside |= through
            weights += a.weights
        # The strand at the left above leaves at the right below, and the other way round.
        first = weights >> shift & FIELD_MASK
        second = weights >> shift + FIELD & FIELD_MASK
        rest = weights & ~self._pair_fields
        right = inputs[-1].right if inputs else generator.right
        targets = {target.kind: target for target in self.generators(right)}
        here = _QUARTER_GRADINGS[generator.kind]
        terms: set = set()
        for word, kind in local:
            target = targets.get(kind)
            if target is None:
                continue
            there = _QUARTER_GRADINGS[kind]
            left_weight = second + (here[0] - there[0]) // 2
            right_weight = first + (here[1] - there[1]) // 2
            if left_weight < 0 or right_weight < 0:
                continue
            c_strands = outside | word.c1 << s | word.c2 << s + 1
            element = self.lower.element(
                generator.left, target.left,
                rest | left_weight << shift | right_weight << shift + FIELD, c_strands)
            if element is not None and self.type_of(element) == word:
                terms ^= {(element, target)}
        return terms

    def _curved_south(self, generator: Generator, only: str | None) -> set:
        """gamma_0 of S when the crossing's strands are not matched: U_beta R_i (x) W +
        U_alpha L_{i+1} (x) E, alpha and beta the strands matched to i and i + 1 above."""
        targets = {target.kind: target for target in self.generators(generator.right)}
        terms: set = set()
        for kind, weights in self._south_moves:
            target = targets.get(kind)
            if target is None or only is not None and kind != only:
                continue
            element = self.lower.element(generator.left, target.left, weights, 0)
            if element is not None:
                terms.add((element, target))
        return terms
