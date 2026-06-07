"""Tests for the F2 reference reducer (algebra/reduce_reference.py).

The reducer is the gold-master every faster reducer must later match, so these check
real homology dimensions against hand-computed answers, not just that it runs.
"""
import pytest

from tetradrome.algebra import GradedComplex, f2_rank, homology


# ---- f2_rank ------------------------------------------------------------

def test_f2_rank_independent_columns():
    # Two columns with distinct leading rows are independent.
    assert f2_rank([frozenset({0, 1}), frozenset({1})]) == 2


def test_f2_rank_dependent_column():
    # Third column is the F2 sum of the first two, so rank stays 2.
    cols = [frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})]
    assert f2_rank(cols) == 2


def test_f2_rank_zero_and_empty():
    assert f2_rank([]) == 0
    assert f2_rank([frozenset(), frozenset()]) == 0
    assert f2_rank([frozenset(), frozenset({3})]) == 1


# ---- homology -----------------------------------------------------------

def test_homology_zero_differentials_is_the_chain_groups():
    # No maps: every cycle is a class, so H^n = C^n.
    cx = GradedComplex({0: 2, 1: 3}, {})
    assert homology(cx) == {0: 2, 1: 3}


def test_homology_rank_one_map_kills_one_dimension():
    # C^0 = F2^2, C^1 = F2; d^0(e0) = f0, d^0(e1) = 0.
    # H^0 = 2 - rank(d^0) = 2 - 1 = 1 ; H^1 = 1 - rank(d^0) = 1 - 1 = 0.
    cx = GradedComplex({0: 2, 1: 1}, {0: [{0}, set()]})
    assert homology(cx) == {0: 1}


def test_homology_acyclic_complex_is_empty():
    # The exact complex F2 -> F2^2 -> F2 from the data-structure tests.
    cx = GradedComplex({0: 1, 1: 2, 2: 1}, {0: [{0, 1}], 1: [{0}, {0}]})
    assert homology(cx) == {}


def test_homology_circle_over_f2():
    # Cellular S^1: one 0-cell, one 1-cell, zero differential -> H^0 = H^1 = F2.
    cx = GradedComplex({0: 1, 1: 1}, {})
    assert homology(cx) == {0: 1, 1: 1}


def test_verify_flag_does_not_change_valid_results():
    cx = GradedComplex({0: 2, 1: 1}, {0: [{0}, set()]})
    assert homology(cx, verify=True) == homology(cx, verify=False)


def test_homology_rejects_non_complex_by_default():
    # d^1 . d^0 != 0 : the default d^2 check must catch it.
    bad = GradedComplex({0: 1, 1: 1, 2: 1}, {0: [{0}], 1: [{0}]})
    with pytest.raises(RuntimeError, match=r"d\^2 != 0"):
        homology(bad)


def test_homology_negative_dimension_backstop():
    # Same bad complex with the upfront check skipped: the impossible-result backstop
    # (a negative Betti number) must still fire.
    bad = GradedComplex({0: 1, 1: 1, 2: 1}, {0: [{0}], 1: [{0}]})
    with pytest.raises(RuntimeError, match=r"negative homology dimension"):
        homology(bad, verify=False)
