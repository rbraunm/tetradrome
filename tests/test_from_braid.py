# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""from_braid: braid-word input, including off-table knots.

The off-table checks below use torus knots, whose invariants have documented closed
forms, at crossing numbers (>=14) beyond KnotInfo's 13-crossing tables -- so they
exercise computation on knots the oracle does not contain.
"""
import pytest

from tetradrome import invariants, knots
from tetradrome.errors import UnknownKnot, UnvalidatedResult


def torus(p, q):
    """Braid word for the (p, q) torus knot: (s1 s2 ... s_{p-1})^q."""
    return list(range(1, p)) * q


# --- input handling ---------------------------------------------------------

def test_from_braid_rejects_empty():
    with pytest.raises(ValueError):
        knots.from_braid([])


def test_from_braid_rejects_zero_entry():
    with pytest.raises(ValueError):
        knots.from_braid([1, 0, 2])


def test_crossing_number_unavailable_for_braid_only():
    k = knots.from_braid([1, 1, 1])
    assert k.braid == (1, 1, 1)
    with pytest.raises(ValueError):
        _ = k.crossing_number


# --- tabulated braid: validates against the oracle --------------------------

def test_from_braid_with_identity_validates():
    k = knots.from_braid([1, 1, 1], identity="3_1")
    det = invariants.compute(k, "determinant")  # validate=True
    sig = invariants.compute(k, "signature")
    assert det.value == 3 and det.validation.verdict("knotinfo") == "pass"
    assert sig.value == -2 and sig.validation.verdict("knotinfo") == "pass"
    assert det.provenance.inputs == "braid_word"


# --- off-table: no oracle, so validate=True must refuse ---------------------

def test_offtable_requires_opt_in():
    k = knots.from_braid(torus(2, 15))  # 15-crossing torus knot, not in KnotInfo
    with pytest.raises(UnvalidatedResult):
        invariants.compute(k, "determinant")  # validate=True
    out = invariants.compute(k, "determinant", validate=False)
    assert out.validation.verdict("knotinfo") == "not_run"
    assert out.knot == "(braid word)"


def test_compute_needs_braid_or_identity():
    # A PD-only diagram with no identity has no braid word and no oracle -> compute
    # cannot reach the Seifert matrix and refuses. (Use a validated PD from KnotInfo,
    # then drop the identity by re-wrapping it as raw PD.)
    pd_code = knots.from_name("3_1").pd_code
    k = knots.from_pd(pd_code)  # identity=None, braid=None
    with pytest.raises(UnknownKnot):
        invariants.compute(k, "determinant", validate=False)


# --- off-table correctness against documented closed forms ------------------

@pytest.mark.parametrize("n", [15, 17, 21, 25])
def test_offtable_torus_2n_closed_form(n):
    """T(2,n), n odd: determinant = n, signature = -(n-1)."""
    k = knots.from_braid(torus(2, n))
    det = invariants.compute(k, "determinant", validate=False)
    sig = invariants.compute(k, "signature", validate=False)
    assert det.value == n
    assert sig.value == -(n - 1)


def test_offtable_torus_symmetry():
    """T(p,q) == T(q,p): two different off-table braids of the same knot must give
    the same invariants. This check needs no formula and no oracle."""
    a = knots.from_braid(torus(4, 5))  # 15 crossings
    b = knots.from_braid(torus(5, 4))  # 16 crossings
    da = invariants.compute(a, "determinant", validate=False).value
    db = invariants.compute(b, "determinant", validate=False).value
    sa = invariants.compute(a, "signature", validate=False).value
    sb = invariants.compute(b, "signature", validate=False).value
    assert da == db
    assert sa == sb
