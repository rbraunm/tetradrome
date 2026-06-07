"""Validate the signed rational Khovanov complex (Phase 3, step 3a-2).

Three independent checks on the cube edge signs and the rational lane:

1. d^2 = 0 over Q (enforced inside rational_homology). Over F2 signs are invisible, so
   this is what actually certifies the sign assignment.
2. The rational homology equals KnotInfo's free ranks -- the torsion-free part of the
   unreduced integral vector -- up to the global mirror pinned in Phase 2c. Rational
   Khovanov has no torsion, so it sees exactly the free summands.
3. Reducing the signed complex mod 2 reproduces the Phase 2 F2 homology exactly (signs
   vanish mod 2). This ties the two lanes together: same construction, two coefficients.
"""
import ast

import pytest

from tetradrome import knots
from tetradrome.algebra import GradedComplex, homology
from tetradrome.backends import knotinfo_backend as ki
from tetradrome.engines import khovanov

KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_4"]


def _knotinfo_free_ranks(name: str) -> dict[tuple[int, int], int]:
    """KnotInfo's rational (torsion-free) unreduced Khovanov, from the integral vector
    [torsion, multiplicity, i, j]."""
    vec = ast.literal_eval(ki.lookup(name)["khovanov_unreduced_integral_vector"])
    out: dict[tuple[int, int], int] = {}
    for torsion, mult, i, j in vec:
        if torsion == 0:
            out[(i, j)] = out.get((i, j), 0) + mult
    return out


def _mirror(table: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    return {(-i, -j): d for (i, j), d in table.items()}


def _signed_reduced_mod2(pd) -> dict[tuple[int, int], int]:
    """Reduce the signed rational complex mod 2 (odd coefficients become 1)."""
    out: dict[tuple[int, int], int] = {}
    for j, cx in khovanov.khovanov_complexes_q(pd).items():
        dims = {n: cx.dim(n) for n in cx.degrees()}
        maps = {
            n: [frozenset(r for r, v in col.items() if v % 2) for col in cx.differential(n)]
            for n in cx.degrees()
        }
        for i, d in homology(GradedComplex(dims, maps)).items():
            out[(i, j)] = d
    return out


@pytest.mark.parametrize("name", KNOTS)
def test_rational_khovanov_matches_knotinfo_free_ranks(name):
    pd = knots.from_name(name).pd_code
    # khovanov_homology_q runs verify_d_squared over Q internally -> certifies signs.
    assert khovanov.khovanov_homology_q(pd) == _mirror(_knotinfo_free_ranks(name))


@pytest.mark.parametrize("name", KNOTS)
def test_signed_complex_reduced_mod2_reproduces_f2(name):
    pd = knots.from_name(name).pd_code
    assert _signed_reduced_mod2(pd) == khovanov.khovanov_homology(pd)


@pytest.mark.parametrize("name", KNOTS)
def test_rational_chain_dimensions_unchanged_by_signs(name):
    # Signs do not change the chain groups, only the maps: total Q dim == unreduced size.
    pd = knots.from_name(name).pd_code
    total = sum(cx.total_dim() for cx in khovanov.khovanov_complexes_q(pd).values())
    assert total == khovanov.unreduced_size(pd)


def test_unknot_rational_khovanov():
    assert khovanov.khovanov_homology_q(()) == {(0, 1): 1, (0, -1): 1}


def test_trefoil_explicit_rational():
    # Left trefoil (KnotInfo's 3_1 PD), rational: the four free generators, no torsion.
    pd = knots.from_name("3_1").pd_code
    assert khovanov.khovanov_homology_q(pd) == {
        (0, -1): 1,
        (0, -3): 1,
        (-2, -5): 1,
        (-3, -9): 1,
    }


def test_empty_diagram_rejected_in_cube():
    with pytest.raises(ValueError, match=r"empty diagram"):
        khovanov.khovanov_complexes_q(())
