"""Khovanov homology over F2: reduce each quantum-graded complex to its Betti numbers.

This closes the first full faithful path -- diagram -> cube -> enhanced states ->
graded complexes -> homology -- entirely natively, with the shared F2 reference reducer
doing the linear algebra. The result is the unreduced Khovanov homology Kh^{i,j}(L; F2)
as {(i, j): dim}. Validated against KnotInfo (tests).
"""
from __future__ import annotations

from ...algebra import homology
from ...diagrams.model import PDCode
from .differential import khovanov_complexes


def khovanov_homology(pd: PDCode) -> dict[tuple[int, int], int]:
    """Unreduced Khovanov homology over F2, as {(i, j): dim Kh^{i,j}}.

    The crossingless unknot is the cube's representational boundary, handled here the
    same way jones_polynomial handles it: its unreduced Khovanov over F2 is one
    dimension in each of the bidegrees (0, +1) and (0, -1).
    """
    if not pd:
        return {(0, 1): 1, (0, -1): 1}
    result: dict[tuple[int, int], int] = {}
    for j, cx in khovanov_complexes(pd).items():
        for i, dim in homology(cx).items():      # verifies d^2 = 0 by default
            result[(i, j)] = dim
    return result
