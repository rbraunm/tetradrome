# SPDX-License-Identifier: Apache-2.0

"""The curved DA bimodule mho^1 of a local minimum (arXiv:1707.00597, sections 7 and 13.1).

Read downwards, the cup closes the input's strands 0 and 1, the paper's 1 and 2, which the
input matching pairs with strands alpha + 2 and beta + 2; the output's strand s is the input's
s + 2, and the output matching pairs alpha with beta. A generator T_x is a preferred input
idempotent x off position 0, so x holds position 2 and not 1; its output idempotent psi(x)
drops that coordinate and shifts the rest down by two. These are the generators the paper calls
Y R_2; the others, X L_1, never meet the restricted idempotents (Proposition 8.2).

The operations are the paper's gamma operations, read from the graph (13.2) with vertices
Y R_2, X and Y. Each input follows the edge its weights on the cup's two strands name: from
Y R_2, U_2^m back to Y R_2 (an operation of one input) or U_2^m L_2 to X; from X, U_1^{m+1} to
Y; from Y, U_2^{m+1} to X or U_2^m R_2 back to Y R_2, which ends the operation. Its value is
U_alpha^{v_2} U_beta^{v_1} (x) T_y times the element of the output carrying the inputs' weights
on the strands beyond the cup (Proposition 13.1 and Definition 7.3): v_1 sums the m of the edges
X -> Y and v_2 those of the others, the numbers of C_2's and C_1's that Theorem 7.4's preferred
sequence inserts. There is no operation without inputs.

``curved_minimum_weight_budget`` bounds the total weight of any output on a given type D
structure; a path whose output weight passes it is dropped, which makes the tensor product
finite.
"""
from __future__ import annotations

from ...errors import TetradromeError
from .algebra import FIELD, FIELD_MASK, CurvedAlgebra, Element, field_sum
from .tensor import Generator
from .type_d import TypeD


def curved_minimum_weight_budget(typed: TypeD) -> int:
    """The largest doubled total weight an output of the curved mho^1 on ``typed`` can have.

    The generators T have Delta = 0, so an output b (x) T y' of T y has Delta(b) = Delta(y) -
    Delta(y') - 1, and over a curved algebra Delta(b) is minus the total weight of b.
    """
    doubled_delta = [2 * maslov - alexander
                     for maslov, alexander in zip(typed.maslov, typed.alexander)]
    return max(doubled_delta) - min(doubled_delta) + 2


class CurvedMinimumBimodule:
    def __init__(self, upper: CurvedAlgebra, budget: int | None):
        if not upper.curved or upper.strands < 2 or upper.partner[0] == 1:
            raise TetradromeError(f"{upper} has no cup on two unmatched strands 0 and 1.")
        self.upper = upper
        self.budget = budget
        alpha, beta = upper.partner[0] - 2, upper.partner[1] - 2
        partner = [0] * (upper.strands - 2)
        for s in range(2, upper.strands):
            partner[s - 2] = upper.partner[s] - 2
        partner[alpha], partner[beta] = beta, alpha
        self.lower = CurvedAlgebra(upper.strands - 2, upper.occupied - 1,
                                   frozenset(s - 2 for s in upper.upward if s >= 2),
                                   tuple(partner))
        self._u_alpha = 2 << FIELD * alpha
        self._u_beta = 2 << FIELD * beta

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
        return ("YR2", generator, generator.right, 0, False)

    def accepts_more(self, state: tuple) -> bool:
        return not (state[4] and state[0] == "YR2")

    def advance(self, state: tuple, element: Element) -> tuple | None:
        vertex, generator, _, weights, moved = state
        if moved and vertex == "YR2":
            return None
        first, second = element.weights & FIELD_MASK, element.weights >> FIELD & FIELD_MASK
        if vertex == "X":
            if second or first < 2 or first % 2:
                return None
            vertex, extra = "Y", (first // 2 - 1) * self._u_beta
        elif first:
            return None
        elif vertex == "YR2":
            vertex, extra = "YR2" if second % 2 == 0 else "X", second // 2 * self._u_alpha
        elif second % 2:
            vertex, extra = "YR2", second // 2 * self._u_alpha
        elif second:
            vertex, extra = "X", (second // 2 - 1) * self._u_alpha
        else:
            return None
        weights += (element.weights >> 2 * FIELD) + extra
        if self.budget is not None and field_sum(weights) > self.budget:
            return None
        return (vertex, generator, element.right, weights, True)

    def emit(self, state: tuple) -> set:
        vertex, generator, right, weights, moved = state
        if not moved or vertex != "YR2":
            return set()
        targets = self.generators(right)
        if not targets:
            return set()
        element = self.lower.element(generator.left, targets[0].left, weights, 0)
        if element is None:
            return set()
        return {(element, targets[0])}
