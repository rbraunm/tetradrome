"""Validate the Rasmussen s-invariant against KnotInfo (Phase 3, step 3c).

s(mirror K) = -s(K), and our Khovanov/Lee gradings are the mirror of KnotInfo's table
convention (pinned in Phase 2c), so our s of KnotInfo's PD is minus KnotInfo's stored
value. The spread below covers s = 0 (amphichiral / slice), +/-2, +/-4, and +/-6
(8_19 = T(3,4)). Computing s runs the d^2 = 0 check and asserts the filtration gap is 2,
so a wrong deformation, sign, or filtration reading would raise rather than pass.
"""
import pytest

from tetradrome import knots
from tetradrome.backends import knotinfo_backend as ki
from tetradrome.engines.khovanov import rasmussen_s

KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_4", "8_19"]


@pytest.mark.parametrize("name", KNOTS)
def test_s_matches_knotinfo_up_to_mirror(name):
    pd = knots.from_name(name).pd_code
    expected = -int(ki.lookup(name)["rasmussen_invariant"])  # mirror -> sign flip
    assert rasmussen_s(pd) == expected


def test_unknot_s_is_zero():
    assert rasmussen_s(()) == 0


def test_left_trefoil_explicit():
    # KnotInfo's 3_1 PD is the left-handed trefoil: s = -2.
    assert rasmussen_s(knots.from_name("3_1").pd_code) == -2


def test_torus_knot_s_equals_twice_genus():
    # 8_19 = T(3,4): |s| = 2*g_4 = (3-1)(4-1) = 6 for the positive torus knot.
    assert rasmussen_s(knots.from_name("8_19").pd_code) == -6
