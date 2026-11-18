# SPDX-License-Identifier: Apache-2.0

"""The compiled sparse reducer returns exactly what the reference reducer returns."""
import random
from array import array

import pytest

from tetradrome.algebra.reduce_f2_sparse import (
    HAVE_SPARSE,
    f2_kernel_escape_sparse,
    f2_rank_sparse,
)
from tetradrome.algebra.reduce_reference import f2_kernel_escape, f2_rank

pytestmark = pytest.mark.skipif(not HAVE_SPARSE, reason="the sparse reducer needs numpy and numba")


def _csc(columns):
    indices, indptr = array("i"), array("i", [0])
    for column in columns:
        indices.extend(column)
        indptr.append(len(indices))
    return indices, indptr


def _sets(columns):
    return [frozenset(column) for column in columns]


def test_rank_of_a_matrix_with_a_dependent_column():
    # The third column is the sum of the first two.
    assert f2_rank_sparse(_csc([[0, 1], [1, 2], [0, 2]])) == 2


def test_rank_of_nothing_is_zero():
    assert f2_rank_sparse(_csc([])) == 0
    assert f2_rank_sparse(_csc([[], []])) == 0


def test_a_repeated_row_index_counts_once_as_in_the_reference():
    # The reference reads a column as the set of its indices.
    csc = _csc([[1, 1, 0], [0]])
    assert f2_rank_sparse(csc) == f2_rank(csc) == 2


def test_kernel_escape_finds_the_first_level_whose_relation_the_image_misses():
    # Columns 0 and 1 are equal, so {0, 1} is a relation; it is complete at level 5, where the
    # later of the two enters. Column 2 is zero, a relation at level 9.
    columns = [[0], [0], []]
    levels = [3, 5, 9]
    assert f2_kernel_escape_sparse(_csc(columns), levels, _csc([])) == (5, 0)
    # An image spanning {0, 1} leaves the zero column's relation {2} as the first one missed.
    assert f2_kernel_escape_sparse(_csc(columns), levels, _csc([[0, 1]])) == (9, 1)
    assert f2_kernel_escape_sparse(_csc(columns), levels, _csc([[0, 1], [2]])) == (None, 2)


def test_kernel_escape_takes_columns_in_level_order_with_ties_in_column_order():
    # The relation {0, 1} is complete when the higher-level column of the two has entered,
    # whichever position it holds in the matrix.
    assert f2_kernel_escape_sparse(_csc([[0], [0]]), [7, 2], _csc([])) == (7, 0)
    assert f2_kernel_escape_sparse(_csc([[0], [0]]), [2, 2], _csc([])) == (2, 0)


def _random_case(generator):
    rows, cols = generator.randint(0, 14), generator.randint(0, 18)
    columns = [[generator.randrange(rows) for _ in range(generator.randint(0, min(rows, 5)))]
               if rows else [] for _ in range(cols)]
    levels = [generator.randint(-2, 2) for _ in range(cols)]
    image = [[generator.randrange(cols) for _ in range(generator.randint(0, min(cols, 4)))]
             if cols else [] for _ in range(generator.randint(0, 8))]
    return columns, levels, image


def test_rank_matches_the_reference_on_random_matrices():
    generator = random.Random(20261118)
    for _ in range(300):
        columns, _, _ = _random_case(generator)
        assert f2_rank_sparse(_csc(columns)) == f2_rank(_csc(columns)), columns


def test_kernel_escape_matches_the_reference_on_random_matrices():
    generator = random.Random(20261119)
    outcomes = set()
    for _ in range(300):
        columns, levels, image = _random_case(generator)
        expected = f2_kernel_escape(_sets(columns), levels, _sets(image))
        assert f2_kernel_escape_sparse(_csc(columns), levels, _csc(image)) == expected, (
            columns, levels, image)
        outcomes.add(expected[0] is None)
    assert outcomes == {True, False}        # both "escapes" and "never escapes" were exercised
