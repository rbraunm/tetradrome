# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tests for the comparison adapters' pure normalization logic.

These exercise the Khovanov parser, the UCT F2 derivation, the mirror, and the verdict against
real oracle output captured from CT 250, with the native reference values (from
``invariants.compute``, itself validated against KnotInfo) hard-coded. They run anywhere -- no
tetradrome, no oracle binaries -- because the functions under test are pure. The ``_agree*``
wrappers that call ``invariants.compute`` are covered by a full artifact run on CT 250.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapters  # noqa: E402


# Native (reference) Khovanov / s for the left-handed KnotInfo knots, from invariants.compute.
NATIVE_RATIONAL = {
    "3_1": {(-3, -9): 1, (-2, -5): 1, (0, -3): 1, (0, -1): 1},
    "4_1": {(-2, -5): 1, (-1, -1): 1, (0, -1): 1, (0, 1): 1, (1, 1): 1, (2, 5): 1},
    "5_1": {(-5, -15): 1, (-4, -11): 1, (-3, -11): 1, (-2, -7): 1, (0, -5): 1, (0, -3): 1},
}
NATIVE_F2 = {
    "3_1": {(-3, -9): 1, (-3, -7): 1, (-2, -7): 1, (-2, -5): 1, (0, -3): 1, (0, -1): 1},
    "4_1": {(-2, -5): 1, (-2, -3): 1, (-1, -3): 1, (-1, -1): 1, (0, -1): 1,
            (0, 1): 1, (1, 1): 1, (1, 3): 1, (2, 3): 1, (2, 5): 1},
    "5_1": {(-5, -15): 1, (-5, -13): 1, (-4, -13): 1, (-4, -11): 1, (-3, -11): 1,
            (-3, -9): 1, (-2, -9): 1, (-2, -7): 1, (0, -5): 1, (0, -3): 1},
}
NATIVE_S = {"3_1": -2, "4_1": 0, "5_1": -4}

# KnotJob -kb0 -s0 output fields on Tetradrome's (mirror-convention) PD, captured on CT 250.
KNOTJOB = {
    "3_1": {"free": "q + q^3 + t^2 q^5 + t^3 q^9", "torsion": "t^3 q^7", "s": 2},
    "4_1": {"free": "t^-2 q^-5 + t^-1 q^-1 + q^-1 + q + t q + t^2 q^5",
            "torsion": "t^-1 q^-3 + t^2 q^3", "s": 0},
    "5_1": {"free": "q^3 + q^5 + t^2 q^7 + t^3 q^11 + t^4 q^11 + t^5 q^15",
            "torsion": "t^3 q^9 + t^5 q^13", "s": 4},
}

CHIRAL = ("3_1", "5_1")
AMPHICHIRAL = "4_1"


def test_monomial_parses_coefficient_and_signed_exponents():
    assert adapters._monomial("q") == ((0, 1), 1)
    assert adapters._monomial("q^3") == ((0, 3), 1)
    assert adapters._monomial("t^2 q^5") == ((2, 5), 1)
    assert adapters._monomial("q^5*t^2") == ((2, 5), 1)          # JavaKh / KhoHo star form
    assert adapters._monomial("q^1*t^0") == ((0, 1), 1)          # explicit t^0
    assert adapters._monomial("t^-2 q^-5") == ((-2, -5), 1)
    assert adapters._monomial("t q") == ((1, 1), 1)              # juxtaposed, implicit exponent 1
    assert adapters._monomial("2 t^3 q^9") == ((3, 9), 2)


def test_parse_khovanov_poly_free_part():
    assert adapters._parseKhovanovPoly(KNOTJOB["3_1"]["free"]) == {
        (0, 1): 1, (0, 3): 1, (2, 5): 1, (3, 9): 1}


def test_parse_khovanov_poly_strips_khoho_parentheses():
    assert adapters._parseKhovanovPoly("q^9*t^3 + q^5*t^2 + (q^3 + q)") == {
        (3, 9): 1, (2, 5): 1, (0, 3): 1, (0, 1): 1}


def test_rational_khovanov_matches_native_up_to_mirror():
    for name, native in NATIVE_RATIONAL.items():
        free = adapters._parseKhovanovPoly(KNOTJOB[name]["free"])
        assert adapters._mirrorKhovanov(free) == native, name


def test_chiral_knots_need_the_mirror_amphichiral_does_not():
    for name in CHIRAL:
        free = adapters._parseKhovanovPoly(KNOTJOB[name]["free"])
        assert free != NATIVE_RATIONAL[name], name
    free = adapters._parseKhovanovPoly(KNOTJOB[AMPHICHIRAL]["free"])
    assert free == NATIVE_RATIONAL[AMPHICHIRAL]


def test_f2_via_uct_matches_native_up_to_mirror():
    for name, native in NATIVE_F2.items():
        free = adapters._parseKhovanovPoly(KNOTJOB[name]["free"])
        torsion = adapters._parseKhovanovPoly(KNOTJOB[name]["torsion"])
        f2 = adapters._f2FromIntegral(free, torsion)
        assert adapters._mirrorKhovanov(f2) == native, name


def test_verdict_pass_mirror_mismatch():
    native = NATIVE_RATIONAL["3_1"]
    free = adapters._parseKhovanovPoly(KNOTJOB["3_1"]["free"])
    assert adapters._verdict(free, native, adapters._mirrorKhovanov) == "mirror"
    assert adapters._verdict(native, native, adapters._mirrorKhovanov) == "pass"
    assert adapters._verdict({(9, 9): 1}, native, adapters._mirrorKhovanov) == "mismatch"
    negate = lambda v: -v
    assert adapters._verdict(2, -2, negate) == "mirror"          # s of a chiral knot
    assert adapters._verdict(0, 0, negate) == "pass"             # s of an amphichiral knot
    assert adapters._verdict(3, -2, negate) == "mismatch"


def test_field_after_colon():
    text = ("Knot 1\n"
            "S-Invariant mod 0 : 2\n"
            "Integral unreduced Khovanov Homology : q + q^3 + t^2 q^5 + t^3 q^9\n"
            "Torsion of order 2 : t^3 q^7\n")
    assert adapters._fieldAfterColon(text, "S-Invariant mod 0") == "2"
    assert adapters._fieldAfterColon(text, "Torsion of order 2") == "t^3 q^7"
    assert adapters._fieldAfterColon(text, "not present") is None


# JavaKh -Q output (quoted q^a*t^b string) on Tetradrome PD, captured on CT 250.
JAVAKH = {
    "3_1": '"q^1*t^0 + q^3*t^0 + q^5*t^2 + q^9*t^3 "',
    "4_1": '"q^-5*t^-2 + q^-1*t^-1 + q^-1*t^0 + q^1*t^0 + q^1*t^1 + q^5*t^2 "',
    "5_1": '"q^3*t^0 + q^5*t^0 + q^7*t^2 + q^11*t^3 + q^11*t^4 + q^15*t^5 "',
}


def test_javakh_rational_khovanov_matches_native_up_to_mirror():
    for name, native in NATIVE_RATIONAL.items():
        groups = adapters._parseKhovanovPoly(JAVAKH[name].replace('"', ""))
        assert adapters._mirrorKhovanov(groups) == native, name


if __name__ == "__main__":
    import traceback
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        try:
            test()
            print("PASS", test.__name__)
        except Exception:
            failures += 1
            print("FAIL", test.__name__)
            traceback.print_exc()
    print("\n%d/%d passed" % (len(tests) - failures, len(tests)))
    sys.exit(1 if failures else 0)
