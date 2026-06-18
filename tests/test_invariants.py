# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

import pytest

from tetradrome import invariants, knots
from tetradrome.errors import UnknownKnot, UnvalidatedResult

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

HOMOLOGICAL = ["khovanov_homology", "rational_khovanov_homology", "rasmussen_s"]


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


@pytest.mark.parametrize("name", ["3_1", "4_1", "5_2", "6_2", "7_4"])
@pytest.mark.parametrize("invariant", HOMOLOGICAL)
def test_homological_invariants_validate_against_knotinfo(invariant, name):
    # The native cube/Lee computation must match KnotInfo's oracle (mirrored to our
    # chirality in the backend), and the d^2 = 0 check must have run.
    r = invariants.compute(knots.from_name(name), invariant)
    assert r.validation.known_answer_match == "pass"
    assert r.validation.is_validated
    assert r.validation.d_squared_check == "pass"
    assert r.provenance.backend == "tetradrome-native"


def test_rasmussen_s_values():
    # s of KnotInfo's PD (the mirror of its tabulated value): left trefoil -2, T(3,4) -6.
    assert invariants.compute(knots.from_name("3_1"), "rasmussen_s").value == -2
    assert invariants.compute(knots.from_name("8_19"), "rasmussen_s").value == -6


def test_offtable_pd_homological_is_unvalidated():
    raw = knots.from_pd(knots.from_name("5_2").pd_code)  # identity is None -> no oracle
    with pytest.raises(UnvalidatedResult):
        invariants.compute(raw, "khovanov_homology")
    # validate=False returns the computed value, marked as having no oracle.
    r = invariants.compute(raw, "khovanov_homology", validate=False)
    assert r.validation.known_answer_match == "not_available"


def test_diagrammatic_invariant_needs_a_pd():
    k = knots.from_braid([1, 1, 1])  # trefoil as a braid: no PD diagram
    with pytest.raises(UnknownKnot):
        invariants.compute(k, "rasmussen_s")


def test_unsupported_invariant_raises():
    k = knots.from_name("3_1")
    with pytest.raises(ValueError):
        invariants.compute(k, "knot_floer_homology")  # genuinely not supported


def test_offtable_knot_without_identity_raises():
    k = knots.from_name("4_1")
    raw = knots.from_pd(k.pd_code)  # identity is None
    with pytest.raises(UnknownKnot):
        invariants.compute(raw, "determinant")
