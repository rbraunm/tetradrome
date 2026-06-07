"""Tests for the graded chain complex data structure (algebra/complex.py).

These verify real behavior: a genuinely acyclic complex passes the d^2 = 0 check, a
complex with a non-zero composite is caught, and malformed data raises at
construction rather than being silently coerced.
"""
import pytest

from tetradrome.algebra import GradedComplex


def test_dims_degrees_total_dim():
    # C^0 = F2, C^1 = F2^2, C^2 = F2; absent/zero degrees report dimension 0.
    c = GradedComplex({0: 1, 1: 2, 2: 1}, {})
    assert c.dim(0) == 1
    assert c.dim(1) == 2
    assert c.dim(2) == 1
    assert c.dim(5) == 0          # absent
    assert c.degrees() == [0, 1, 2]
    assert c.total_dim() == 4


def test_zero_dimension_degrees_are_dropped():
    c = GradedComplex({0: 0, 1: 3, 2: 0}, {})
    assert c.degrees() == [1]
    assert c.total_dim() == 3


def test_differential_shape_for_zero_map():
    # No map supplied for degree 1: differential is dim(1) empty columns.
    c = GradedComplex({1: 3, 2: 2}, {})
    d1 = c.differential(1)
    assert len(d1) == 3
    assert all(col == frozenset() for col in d1)
    # A degree with no chain group has no columns at all.
    assert c.differential(7) == ()


def test_valid_complex_passes_d_squared():
    # d^0(e0) = f0 + f1 ; d^1(f0) = d^1(f1) = g0.
    # Then d^1(d^0(e0)) = d^1(f0) + d^1(f1) = g0 + g0 = 0 over F2.
    c = GradedComplex(
        {0: 1, 1: 2, 2: 1},
        {0: [{0, 1}], 1: [{0}, {0}]},
    )
    c.verify_d_squared()  # must not raise
    # Columns are stored faithfully.
    assert c.differential(0) == (frozenset({0, 1}),)
    assert c.differential(1) == (frozenset({0}), frozenset({0}))


def test_nonzero_d_squared_is_caught():
    # d^0(e0) = f0 ; d^1(f0) = g0  ->  composite is g0 != 0.
    c = GradedComplex(
        {0: 1, 1: 1, 2: 1},
        {0: [{0}], 1: [{0}]},
    )
    with pytest.raises(RuntimeError, match=r"d\^2 != 0"):
        c.verify_d_squared()


def test_column_count_must_match_source_dimension():
    # C^0 has dimension 1, so d^0 must have exactly 1 column.
    with pytest.raises(ValueError, match=r"d\^0 has 2 columns but C\^0 has dimension 1"):
        GradedComplex({0: 1, 1: 1}, {0: [{0}, {0}]})


def test_row_index_must_lie_in_target():
    # C^1 has dimension 2, so a column of d^0 may only reference rows 0, 1.
    with pytest.raises(ValueError, match=r"references row 2"):
        GradedComplex({0: 1, 1: 2}, {0: [{2}]})


def test_negative_dimension_rejected():
    with pytest.raises(ValueError, match=r"negative dimension"):
        GradedComplex({0: -1}, {})


def test_empty_complex():
    c = GradedComplex({}, {})
    assert c.degrees() == []
    assert c.total_dim() == 0
    c.verify_d_squared()  # vacuously true
