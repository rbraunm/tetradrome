"""Tests for the rational back-end lane (RationalComplex + rational_homology).

Hand-built complexes with homology computed by hand, exercising genuine rational
arithmetic (not just 0/1), the field-dependence that motivates the lane (a map that is
an isomorphism over Q but zero mod 2), and the d^2 = 0 gate.
"""
from fractions import Fraction

import pytest

from tetradrome.algebra import (
    GradedComplex,
    RationalComplex,
    homology,
    rational_homology,
    rational_rank,
)


def test_rank_zero_one_matrix():
    # Two columns onto the same row: rank 1.
    assert rational_rank([{0: 1}, {0: 1}]) == 1
    assert rational_rank([]) == 0
    assert rational_rank([{}]) == 0


def test_rank_full_rational_matrix():
    # [[2, 1], [1, 3]] has determinant 5, so rank 2 -- exercises Fraction division.
    assert rational_rank([{0: 2, 1: 1}, {0: 1, 1: 3}]) == 2


def test_rank_dependent_rational_columns():
    # Third column is 2*(first) - (second): rank 2, not 3.
    c1 = {0: 1, 1: 0, 2: 1}
    c2 = {0: 0, 1: 1, 2: 1}
    c3 = {0: 2, 1: -1, 2: 1}  # = 2*c1 - c2
    assert rational_rank([c1, c2, c3]) == 2


def test_homology_acyclic_iso():
    # 0 -> Q --x2--> Q -> 0. Over Q the map is an isomorphism, so homology vanishes.
    # (Over F2 the same map is zero -- the field matters; see below.)
    cx = RationalComplex({0: 1, 1: 1}, {0: [{0: 2}]})
    assert rational_homology(cx) == {}


def test_field_dependence_against_f2():
    # The x2 map: acyclic over Q, but mod 2 it is the zero map, so F2 homology is
    # Q^0... i.e. F2 in both degrees. Same complex shape, different field, different
    # answer -- the reason the rational lane exists.
    q = RationalComplex({0: 1, 1: 1}, {0: [{0: 2}]})
    assert rational_homology(q) == {}
    f2 = GradedComplex({0: 1, 1: 1}, {0: [set()]})  # 2 mod 2 = 0 -> zero map
    assert homology(f2) == {0: 1, 1: 1}


def test_homology_with_kernel_and_image():
    # 0 -> Q^2 --d--> Q^1 -> 0, d = [1, 1] (both basis vectors -> generator).
    # rank d^0 = 1, so H^0 = 2 - 1 = 1 (the kernel), H^1 = 1 - 1 = 0 (surjective).
    cx = RationalComplex({0: 2, 1: 1}, {0: [{0: 1}, {0: 1}]})
    assert rational_homology(cx) == {0: 1}


def test_signed_complex_d_squared_holds():
    # 0 -> Q --d0--> Q^2 --d1--> Q -> 0 with d0(e)=f0+f1, d1(f0)=g, d1(f1)=-g.
    # d1.d0(e) = g - g = 0, so it is a complex; it is acyclic.
    cx = RationalComplex(
        {0: 1, 1: 2, 2: 1},
        {0: [{0: 1, 1: 1}], 1: [{0: 1}, {0: -1}]},
    )
    cx.verify_d_squared()
    assert rational_homology(cx) == {}


def test_d_squared_violation_raises():
    # d0(e)=f, d1(f)=g, so d1.d0(e)=g != 0.
    cx = RationalComplex({0: 1, 1: 1, 2: 1}, {0: [{0: 1}], 1: [{0: 1}]})
    with pytest.raises(RuntimeError, match=r"d\^2 != 0"):
        cx.verify_d_squared()
    with pytest.raises(RuntimeError, match=r"d\^2 != 0"):
        rational_homology(cx)


def test_construction_validates_column_count():
    with pytest.raises(ValueError, match=r"columns but C"):
        RationalComplex({0: 2, 1: 1}, {0: [{0: 1}]})  # 1 column, dim C^0 = 2


def test_construction_validates_row_index():
    with pytest.raises(ValueError, match=r"references row"):
        RationalComplex({0: 1, 1: 1}, {0: [{5: 1}]})  # row 5, dim C^1 = 1


def test_zero_coefficients_are_dropped():
    cx = RationalComplex({0: 1, 1: 1}, {0: [{0: Fraction(0)}]})
    assert cx.differential(0) == ({},)  # the zero entry was dropped
    assert rational_homology(cx) == {0: 1, 1: 1}  # zero map -> H = Q in both
