# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

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
    # No map supplied for degree 1: differential is dim(1) empty columns in CSC form.
    c = GradedComplex({1: 3, 2: 2}, {})
    indices, indptr = c.differential(1)
    assert len(indptr) == 3 + 1            # one offset per column, plus the leading 0
    assert list(indptr) == [0, 0, 0, 0]    # every column empty
    assert len(indices) == 0
    # A degree with no chain group has no columns: indptr is just [0].
    idx7, ptr7 = c.differential(7)
    assert list(idx7) == [] and list(ptr7) == [0]


def test_valid_complex_passes_d_squared():
    # d^0(e0) = f0 + f1 ; d^1(f0) = d^1(f1) = g0.
    # Then d^1(d^0(e0)) = d^1(f0) + d^1(f1) = g0 + g0 = 0 over F2.
    c = GradedComplex(
        {0: 1, 1: 2, 2: 1},
        {0: [{0, 1}], 1: [{0}, {0}]},
    )
    c.verify_d_squared()  # must not raise
    # Columns are stored faithfully in CSC: column j is indices[indptr[j]:indptr[j+1]].
    i0, p0 = c.differential(0)
    assert list(p0) == [0, 2] and sorted(i0[p0[0]:p0[1]]) == [0, 1]   # d^0(e0) = {0, 1}
    i1, p1 = c.differential(1)
    assert list(p1) == [0, 1, 2]                                      # two single-entry columns
    assert list(i1) == [0, 0]                                         # d^1(f0) = d^1(f1) = {0}


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


def test_nnz_counts_stored_ones():
    c = GradedComplex({0: 1, 1: 2, 2: 1}, {0: [{0, 1}], 1: [{0}, {0}]})
    assert c.nnz(0) == 2      # the column {0, 1}
    assert c.nnz(1) == 2      # columns {0}, {0}
    assert c.nnz(2) == 0      # no map out of C^2


def test_from_csc_matches_column_constructor():
    # Same complex built two ways agrees on homology and on the stored buffers.
    from array import array

    columns = GradedComplex({0: 1, 1: 2, 2: 1}, {0: [{0, 1}], 1: [{0}, {0}]})
    csc = GradedComplex.from_csc(
        {0: 1, 1: 2, 2: 1},
        {
            0: (array("i", [0, 1]), array("i", [0, 2])),
            1: (array("i", [0, 0]), array("i", [0, 1, 2])),
        },
    )
    csc.verify_d_squared()
    for n in (0, 1, 2):
        assert list(csc.differential(n)[1]) == list(columns.differential(n)[1])
    assert csc.dim(2) == 1 and csc.total_dim() == 4


def test_from_csc_rejects_inconsistent_indptr():
    from array import array

    # indptr must have dim(C^0) + 1 = 2 entries; here it has 3.
    with pytest.raises(ValueError, match=r"indptr length"):
        GradedComplex.from_csc({0: 1, 1: 1}, {0: (array("i", [0]), array("i", [0, 1, 1]))})
    # endpoints must bracket the stored entries.
    with pytest.raises(ValueError, match=r"indptr ends"):
        GradedComplex.from_csc({0: 1, 1: 2}, {0: (array("i", [0, 1]), array("i", [0, 1]))})
