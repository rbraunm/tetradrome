# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Validate the Khovanov gradings (engines/khovanov/gradings.py).

The decisive, coefficient-independent check on the enhanced states and the (i, j)
grading formulas: the chain-level graded Euler characteristic

    chi(q) = sum_{i,j} (-1)^i q^j dim C^{i,j}

must equal the unnormalized Jones polynomial (q + q^-1) V(q^2), where V is the
KnotInfo-validated Jones polynomial from jones.py. Using the same diagram for both
sides makes chirality irrelevant. If a grading formula is off, this fails.
"""
import pytest

from tetradrome import knots
from tetradrome.diagrams import seifert_structure
from tetradrome.engines import khovanov
from tetradrome.invariants import jones

# Small knots only: enhanced-state enumeration is exponential, so keep <= 7 crossings.
KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_4"]


def _euler_characteristic(pd) -> dict[int, int]:
    """chi(q) = sum_{i,j} (-1)^i q^j dim C^{i,j}, as {q_exponent: coefficient}."""
    chi: dict[int, int] = {}
    for (i, j), d in khovanov.chain_dimensions(pd).items():
        chi[j] = chi.get(j, 0) + (-d if i % 2 else d)
    return {e: c for e, c in chi.items() if c}


def _unnormalized_jones(pd) -> dict[int, int]:
    """(q + q^-1) V(q^-2) as {q_exponent: coefficient}, from the validated Jones poly.

    The substitution is q^-2, not q^2: jones.py is pinned to KnotInfo's t-variable,
    which corresponds to q^-2 here. (Amphichiral knots have palindromic V and so cannot
    tell the two apart; chiral ones can, and fix the direction.)
    """
    low, coeffs = jones.jones_polynomial(pd)
    out: dict[int, int] = {}
    for k, c in enumerate(coeffs):
        if not c:
            continue
        base = -2 * (low + k)           # V(q^-2): t^m -> q^(-2m)
        for shift in (+1, -1):          # times (q + q^-1)
            out[base + shift] = out.get(base + shift, 0) + c
    return {e: v for e, v in out.items() if v}


@pytest.mark.parametrize("name", KNOTS)
def test_graded_euler_characteristic_is_jones(name):
    pd = knots.from_name(name).pd_code
    assert _euler_characteristic(pd) == _unnormalized_jones(pd)


def test_unreduced_size_counts_all_generators():
    pd = knots.from_name("3_1").pd_code
    total = sum(khovanov.chain_dimensions(pd).values())
    assert khovanov.unreduced_size(pd) == total


def test_crossing_counts_sum_to_n_and_match_writhe():
    pd = knots.from_name("5_2").pd_code
    n_plus, n_minus = khovanov.crossing_counts(pd)
    assert n_plus + n_minus == len(pd)
    assert n_plus - n_minus == seifert_structure(pd).writhe
