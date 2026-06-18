# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Khovanov homology over F2: reduce each quantum-graded complex to its Betti numbers.

This closes the first full faithful path -- diagram -> cube -> enhanced states ->
graded complexes -> homology -- entirely natively, with the shared F2 reference reducer
doing the linear algebra. The result is the unreduced Khovanov homology Kh^{i,j}(L; F2)
as {(i, j): dim}. Validated against KnotInfo (tests).
"""
from __future__ import annotations

from ...algebra import homology, rational_homology
from ...diagrams.model import PDCode
from .differential import khovanov_complexes, khovanov_complexes_q


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
        for i, dim in homology(cx).items():      # verifies d^2 = 0 over F2
            result[(i, j)] = dim
    return result


def khovanov_homology_q(pd: PDCode) -> dict[tuple[int, int], int]:
    """Unreduced rational Khovanov homology, as {(i, j): dim_Q Kh^{i,j}}.

    Same shape as the F2 version over the rational lane; the unknot short-circuit is
    identical (Kh of the unknot is torsion-free, so Q and F2 agree there).
    """
    if not pd:
        return {(0, 1): 1, (0, -1): 1}
    result: dict[tuple[int, int], int] = {}
    for j, cx in khovanov_complexes_q(pd).items():
        for i, dim in rational_homology(cx).items():  # verifies d^2 = 0 over Q
            result[(i, j)] = dim
    return result
