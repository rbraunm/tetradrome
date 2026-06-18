# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tests for the multimodular rational path (engine Phase 5).

Correctness is equivalence to the exact rational reducer: multimodular ranks and homology
must match `rational_reduce` across the catalog (Khovanov over ℚ and Lee). The max-over-
primes design is also pinned directly -- a prime that divides an entry under-ranks, and the
maximum across primes recovers the true rank -- since that is the property the whole method
rests on.
"""
from fractions import Fraction

import pytest

from tetradrome import knots
from tetradrome.algebra.multimodular import (
    rank_mod_p,
    rational_homology_multimodular,
    rational_rank_multimodular,
)
from tetradrome.algebra.rational_reduce import rational_homology, rational_rank
from tetradrome.engines import khovanov
from tetradrome.engines.khovanov.lee import lee_complex

KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_4"]


def test_rank_mod_p_handbuilt():
    cols = [{0: Fraction(1), 1: Fraction(2)}, {0: Fraction(2), 1: Fraction(4)}]  # dependent
    assert rank_mod_p(cols, 1_000_000_007) == 1
    assert rank_mod_p([{0: Fraction(3)}, {1: Fraction(5)}], 1_000_000_007) == 2


def test_to_fp_bad_prime_raises():
    # 1/7 mod 7 is undefined; must fail loudly, not return a wrong residue.
    with pytest.raises(ValueError):
        rank_mod_p([{0: Fraction(1, 7)}], 7)


def test_max_over_primes_recovers_from_bad_prime():
    # entry 2 vanishes mod 2 (rank 0) but is a unit mod a large prime (rank 1).
    col = [{0: Fraction(2)}]
    assert rank_mod_p(col, 2) == 0
    assert rational_rank_multimodular(col, primes=(2, 1_000_000_007)) == 1


@pytest.mark.parametrize("name", KNOTS)
def test_rational_khovanov_equivalence(name):
    pd = knots.from_name(name).pd_code
    for cx in khovanov.khovanov_complexes_q(pd).values():
        for n in cx.degrees():
            assert rational_rank_multimodular(cx.differential(n)) == rational_rank(cx.differential(n))
        assert rational_homology_multimodular(cx) == rational_homology(cx)


@pytest.mark.parametrize("name", KNOTS)
def test_lee_equivalence(name):
    cx = lee_complex(knots.from_name(name).pd_code)
    assert rational_homology_multimodular(cx) == rational_homology(cx) == {0: 2}
