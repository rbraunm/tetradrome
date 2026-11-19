# SPDX-License-Identifier: Apache-2.0

"""The DA bimodule of a local minimum (arXiv:1603.06559, section 9).

``MinimumBimodule`` is mho^1: read downwards, the cup closes the input's strands 0 and 1, and
the output's strand s is the input's s + 2. A generator T_x is a preferred input idempotent x;
its output idempotent psi(x) drops the coordinate at positions 0 to 2 and shifts the rest down
by two. Only idempotents that avoid position 0 are built: every type D structure of the knot
lives there (the paper's Remark 11.12 and Lemma 12.4), so an input that reaches position 0
raises.

delta^1_l(T_x, a_1, ..., a_{l-1}) is nonzero only on a preferred sequence (section 9.1): the
inputs trace a path in the graph Gamma, each a_i matching its edge's weights on strands 0 and 1
and its divisibility by the C on the upward strand of the two, the inner vertices off the two
ends, and no other C dividing two inputs. Its value is b (x) T_y, b the element of the output
with the inputs' weights and C's from strand 2 on. Gamma is written for the case where strand 1
points up (Figure 40, with item 14 read as its figure shows it and the figure's edge C2 from L1
to L1C2); when strand 0 points up the paper exchanges U1 and U2, L1 and R2, R1 and L2, C2 and
C1, so the inputs are read with the two strands exchanged. A path from an idempotent off position
0 starts and ends at R2 in the first case and at R2C1, the image of L1C2, in the second.

``minimum_weight_budget`` bounds the total weight of any output on a given type D structure; a
path whose weight passes it is dropped, and it is what makes the tensor product finite.
"""
from __future__ import annotations

from ...errors import TetradromeError
from .algebra import FIELD, FIELD_MASK, Algebra, Element, field_sum
from .tensor import Generator
from .type_d import TypeD

_ENDS = ("R2", "L1C2")


def gamma_step(vertex: tuple, first: int, second: int, divisible: int) -> tuple | None:
    """The vertex after an edge of Gamma from ``vertex`` with doubled weights ``first`` and
    ``second`` on the cup's two strands and C-divisibility ``divisible``, or None."""
    name = vertex[0]
    if name == "R2":
        if (first, second, divisible) == (0, 0, 0):
            return ("R2",)
        if (first, second, divisible) == (0, 1, 0):
            return ("C2",)
        return None
    if name == "L1C2":
        if second or divisible:
            return None
        if first == 0:
            return ("L1C2",)
        return ("L1U", first // 2 - 1) if first % 2 == 0 else ("U", first // 2)
    if name == "C2":
        if second or divisible or first == 0:
            return None
        if first == 1:
            return ("L1C2",)
        return ("U", first // 2 - 1) if first % 2 == 0 else ("L1U", first // 2 - 1)
    power = vertex[1]
    if name == "U":
        if power == 0 and not divisible:
            return {(0, 1): ("R2",), (0, 2): ("C2",)}.get((first, second))
        if power == 0 and (first, second) == (1, 2):
            return ("L1C2",)
        if second != 2 or not divisible or first // 2 + power == 0:
            return None
        return ("U", first // 2 + power - 1) if first % 2 == 0 else ("L1U", first // 2 + power - 1)
    if second != 2 or not divisible:
        return None
    if first % 2:
        return ("U", first // 2 + power)
    if first // 2 + power == 0:
        return ("L1C2",)
    return ("L1U", first // 2 + power - 1)


def minimum_weight_budget(typed: TypeD) -> int:
    """The largest doubled total weight an output of mho^1 on ``typed`` can have.

    An output b (x) T y' of T y has m(b) = M(y) - M(y') - 1 and Alex(b) = A(y) - A(y'), and by
    (3.8) and (3.9) its doubled total weight is 2 Alex(b) - 2 m(b) + 2 #C(b).
    """
    spread = [alexander - 2 * maslov for alexander, maslov in zip(typed.alexander, typed.maslov)]
    upward = len(typed.algebra.upward) - 1
    return max(spread) - min(spread) + 2 + 2 * upward


class MinimumBimodule:
    def __init__(self, upper: Algebra, budget: int | None):
        if upper.strands < 2 or (0 in upper.upward) == (1 in upper.upward):
            raise TetradromeError(f"{upper} has no cup on strands 0 and 1.")
        self.upper = upper
        self.budget = budget
        self.lower = Algebra(upper.strands - 2, upper.occupied - 1,
                             frozenset(s - 2 for s in upper.upward if s >= 2))
        self._exchanged = 0 in upper.upward
        self._loop_bit = 1 if self._exchanged else 2
        self._start = ("L1C2",) if self._exchanged else ("R2",)

    def generators(self, right: int) -> list:
        if right & 1:
            raise TetradromeError("an idempotent at position 0 reached the minimum.")
        if right & 7 != 4:
            return []
        return [Generator("T", right >> 3 << 1, right)]

    def maslov(self, generator: Generator) -> int:
        return 0

    def alexander(self, generator: Generator) -> int:
        return 0

    def input_limit(self, typed: TypeD) -> int:
        if self.budget is None:
            raise TetradromeError("the minimum needs a weight budget to be tensored.")
        return (len(typed) + 1) * (self.budget + 2)

    def start(self, generator: Generator) -> tuple:
        return (self._start, generator, generator.right, 0, 0, False)

    def accepts_more(self, state: tuple) -> bool:
        return not (state[5] and state[0][0] in _ENDS)

    def advance(self, state: tuple, element: Element) -> tuple | None:
        if not self.accepts_more(state):
            return None
        vertex, generator, _, weights, c_strands, _ = state
        first, second = element.weights & FIELD_MASK, element.weights >> FIELD & FIELD_MASK
        if self._exchanged:
            first, second = second, first
        vertex_after = gamma_step(vertex, first, second, element.c_strands & self._loop_bit and 1)
        if vertex_after is None:
            return None
        through = element.c_strands >> 2
        if through & c_strands:
            return None
        weights += element.weights >> 2 * FIELD
        if self.budget is not None and field_sum(weights) > self.budget:
            return None
        return (vertex_after, generator, element.right, weights, c_strands | through, True)

    def emit(self, state: tuple) -> set:
        vertex, generator, right, weights, c_strands, moved = state
        if not moved or vertex[0] not in _ENDS:
            return set()
        targets = self.generators(right)
        if not targets:
            return set()
        if vertex != self._start:
            raise TetradromeError(f"a preferred sequence ends at {vertex[0]} over idempotent "
                                  f"{right}, which belongs to the other end.")
        element = self.lower.element(generator.left, targets[0].left, weights, c_strands)
        if element is None:
            return set()
        return {(element, targets[0])}
