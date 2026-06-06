import pytest

from tetradrome import invariants, knots
from tetradrome.errors import UnknownKnot

# Known answers from KnotInfo; the point is that we *compute* these natively and
# they must match the oracle.
CASES = [
    ("3_1", 3, -2),
    ("4_1", 5, 0),
    ("5_1", 5, -4),
    ("5_2", 7, -2),
    ("6_1", 9, 0),
    ("6_2", 11, -2),
    ("6_3", 13, 0),
]


@pytest.mark.parametrize("name,det,sig", CASES)
def test_determinant_and_signature_match_knotinfo(name, det, sig):
    k = knots.from_name(name)
    d = invariants.compute(k, "determinant")
    s = invariants.compute(k, "signature")
    assert d.value == det
    assert s.value == sig
    assert d.validation.known_answer_match == "pass"
    assert s.validation.known_answer_match == "pass"
    assert d.validation.is_validated and s.validation.is_validated
    assert d.provenance.backend == "tetradrome-native"
    assert d.provenance.method == "seifert_form_from_braid"


def test_unsupported_invariant_raises():
    k = knots.from_name("3_1")
    with pytest.raises(ValueError):
        invariants.compute(k, "khovanov_homology")


def test_offtable_knot_without_identity_raises():
    k = knots.from_name("4_1")
    raw = knots.from_pd(k.pd_code)  # identity is None
    with pytest.raises(UnknownKnot):
        invariants.compute(raw, "determinant")
