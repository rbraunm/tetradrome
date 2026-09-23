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

import pytest

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


# KhoHo KhPol_Q(torus(2,n)) final polynomial line, captured on CT 250.
KHOHO = {
    "3_1": "q^9*t^3 + q^5*t^2 + (q^3 + q)",
    "5_1": "q^15*t^5 + q^11*t^4 + q^11*t^3 + q^7*t^2 + (q^5 + q^3)",
}


def test_torus_params_only_odd_n_1():
    assert adapters._torusParams("3_1") == (2, 3)
    assert adapters._torusParams("5_1") == (2, 5)
    assert adapters._torusParams("7_1") == (2, 7)
    assert adapters._torusParams("4_1") is None      # amphichiral, not a torus knot
    assert adapters._torusParams("6_1") is None      # even, a twist knot
    assert adapters._torusParams("K11n34") is None
    assert adapters._torusParams(None) is None


def test_khoho_poly_extracts_final_polynomial_line():
    sample = ("  ***   Warning: new stack size = 512000000 (488.281 Mbytes).\n"
              "Computing Betti numbers ...\n"
              "Secondary grading: 9. Reducing the chain complex ... done.\n"
              "   ... done with computing Betti numbers.\n"
              "q^9*t^3 + q^5*t^2 + (q^3 + q)\n")
    assert adapters._khohoPoly(sample) == "q^9*t^3 + q^5*t^2 + (q^3 + q)"


def test_khoho_rational_khovanov_matches_native_up_to_mirror():
    for name in ("3_1", "5_1"):
        groups = adapters._parseKhovanovPoly(KHOHO[name])
        assert adapters._mirrorKhovanov(groups) == NATIVE_RATIONAL[name], name


# regina jones() output (Laurent in x = t^1/2) on Tetradrome PD, captured on CT 250.
REGINA_JONES = {
    "3_1": "-x^8 + x^6 + x^2",
    "4_1": "x^4 - x^2 + 1 - x^-2 + x^-4",
}
# Native Jones as {t-exponent: coeff}: 3_1 from invariants.compute (1,(1,0,1,-1)); 4_1 the
# symmetric figure-eight value (amphichiral, convention-independent).
NATIVE_JONES = {
    "3_1": {1: 1, 3: 1, 4: -1},
    "4_1": {-2: 1, -1: -1, 0: 1, 1: -1, 2: 1},
}


def test_parse_laurent_single_variable_signs_and_negative_exponents():
    assert adapters._parseLaurent("-x^8 + x^6 + x^2", "x") == {8: -1, 6: 1, 2: 1}
    assert adapters._parseLaurent("x^4 - x^2 + 1 - x^-2 + x^-4", "x") == {
        4: 1, 2: -1, 0: 1, -2: -1, -4: 1}
    assert adapters._parseLaurent("2 x^3 - x", "x") == {3: 2, 1: -1}


def test_regina_jones_matches_native_after_halving():
    for name, native in NATIVE_JONES.items():
        xPoly = adapters._parseLaurent(REGINA_JONES[name], "x")
        assert all(e % 2 == 0 for e in xPoly), name
        jones = {e // 2: c for e, c in xPoly.items()}
        assert jones == native, name


# Sage structured output (what sageRun's script prints) for 3_1 on Tetradrome PD, built from the
# captured sage values: Jones in the t <-> t^-1 (negative-power) convention, Khovanov as
# invariant-factor tuples (0 = a Z summand, 2 = a Z/2 summand).
SAGE_3_1_OUTPUT = (
    "JONES {-1: 1, -3: 1, -4: -1}\n"
    "ALEXANDER {-1: 1, 0: -1, 1: 1}\n"
    "SIGNATURE 2\n"
    "DETERMINANT 3\n"
    "KHOVANOV {(-3, -9): (0,), (-2, -5): (0,), (0, -3): (0,), (0, -1): (0,), (-2, -7): (2,)}\n"
)
# Native Alexander as {exponent: coeff}: canonical form (lowest term at t^0, positive constant).
NATIVE_ALEXANDER = {"3_1": {0: 1, 1: -1, 2: 1}}
NATIVE_SIGNATURE = {"3_1": -2}


def test_parse_sage_fields():
    fields = adapters._parseSageFields(SAGE_3_1_OUTPUT)
    assert fields["JONES"] == {-1: 1, -3: 1, -4: -1}
    assert fields["SIGNATURE"] == 2
    assert fields["DETERMINANT"] == 3
    assert fields["KHOVANOV"][(-2, -7)] == (2,)


def test_sage_khovanov_splits_free_and_torsion():
    fields = adapters._parseSageFields(SAGE_3_1_OUTPUT)
    free, torsion = adapters._sageKhovanov(fields["KHOVANOV"])
    assert free == {(-3, -9): 1, (-2, -5): 1, (0, -3): 1, (0, -1): 1}
    assert torsion == {(-2, -7): 1}


def test_sage_khovanov_invariant_factors_free_rank_and_even_torsion():
    # (0,0) is Z^2 (free rank 2); (2,0) is Z + Z/2; (3,) is Z/3 (odd -> no F2 torsion).
    free, torsion = adapters._sageKhovanov({(0, 0): (0, 0), (1, 2): (2, 0), (2, 4): (3,)})
    assert free == {(0, 0): 2, (1, 2): 1}
    assert torsion == {(1, 2): 1}


def test_sage_rational_and_f2_khovanov_match_native_directly():
    fields = adapters._parseSageFields(SAGE_3_1_OUTPUT)
    free, torsion = adapters._sageKhovanov(fields["KHOVANOV"])
    assert free == NATIVE_RATIONAL["3_1"]                     # identity: sage shares the convention
    assert adapters._f2FromIntegral(free, torsion) == NATIVE_F2["3_1"]


def test_sage_jones_matches_native_after_negation():
    fields = adapters._parseSageFields(SAGE_3_1_OUTPUT)
    assert adapters._negateExponents(fields["JONES"]) == NATIVE_JONES["3_1"]


def test_canonical_alexander_matches_native_up_to_unit():
    fields = adapters._parseSageFields(SAGE_3_1_OUTPUT)
    assert adapters._canonicalAlexander(fields["ALEXANDER"]) == NATIVE_ALEXANDER["3_1"]
    # a shifted, sign-flipped copy (same polynomial up to +/- t^k) canonicalizes identically
    assert adapters._canonicalAlexander({-3: -1, -2: 1, -1: -1}) == NATIVE_ALEXANDER["3_1"]


def test_sage_signature_matches_native_after_negation():
    fields = adapters._parseSageFields(SAGE_3_1_OUTPUT)
    assert -fields["SIGNATURE"] == NATIVE_SIGNATURE["3_1"]


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


# ---- knotkit (kk kh LaTeX grid, kk s line) ----------------------------------------------------
# Real kk output captured in the sandbox from knots.from_name(...).pd_code: the trefoil over Q
# draws only filled circles; 7_4 over Z2 exercises the rank-N ``node {$N$}`` cells.

KK_KH_3_1_Q = r"""\documentclass{article}
\usepackage{amsmath, tikz, hyperref}
\DeclareMathOperator{\rank}{rank}
\setlength{\parindent}{0pt}

\begin{document}
\pagestyle{empty}
\sloppy
Kh = $Kh(\verb~PD[X[1,5,2,4],X[3,1,4,6],X[5,3,6,2]]~; \verb~Q~)$:\\
$\rank Kh = 4$\\
\begin{tikzpicture}[scale=.66]
  \draw[->] (0,0) -- (4.5,0) node[right] {$t$};
  \draw[->] (0,0) -- (0,5.5) node[above] {$q$};
  \draw[step=1] (0,0) grid (4,5);
  \draw (2.250,-0.8) node[below] {$Kh$};
  \draw (0.5,-.2) node[below] {$0$};
  \draw (1.5,-.2) node[below] {$1$};
  \draw (2.5,-.2) node[below] {$2$};
  \draw (3.5,-.2) node[below] {$3$};
  \draw (-.2,0.5) node[left] {$1$};
  \draw (-.2,1.5) node[left] {$3$};
  \draw (-.2,2.5) node[left] {$5$};
  \draw (-.2,3.5) node[left] {$7$};
  \draw (-.2,4.5) node[left] {$9$};
  \fill (0.5, 0.5) circle (.15);
  \fill (0.5, 1.5) circle (.15);
  \fill (2.5, 2.5) circle (.15);
  \fill (3.5, 4.5) circle (.15);
\end{tikzpicture}
\end{document}
"""

KK_KH_7_4_Z2 = r"""\documentclass{article}
\usepackage{amsmath, tikz, hyperref}
\DeclareMathOperator{\rank}{rank}
\setlength{\parindent}{0pt}

\begin{document}
\pagestyle{empty}
\sloppy
Kh = $Kh(\verb~PD[X[2,10,3,9],X[4,12,5,11],X[6,14,7,13],X[8,4,9,3],X[10,2,11,1],X[12,8,13,7],X[14,6,1,5]]~; \verb~Z2~)$:\\
$\rank Kh = 30$\\
\begin{tikzpicture}[scale=.66]
  \draw[->] (0,0) -- (8.5,0) node[right] {$t$};
  \draw[->] (0,0) -- (0,9.5) node[above] {$q$};
  \draw[step=1] (0,0) grid (8,9);
  \draw (4.250,-0.8) node[below] {$Kh$};
  \draw (0.5,-.2) node[below] {$0$};
  \draw (1.5,-.2) node[below] {$1$};
  \draw (2.5,-.2) node[below] {$2$};
  \draw (3.5,-.2) node[below] {$3$};
  \draw (4.5,-.2) node[below] {$4$};
  \draw (5.5,-.2) node[below] {$5$};
  \draw (6.5,-.2) node[below] {$6$};
  \draw (7.5,-.2) node[below] {$7$};
  \draw (-.2,0.5) node[left] {$1$};
  \draw (-.2,1.5) node[left] {$3$};
  \draw (-.2,2.5) node[left] {$5$};
  \draw (-.2,3.5) node[left] {$7$};
  \draw (-.2,4.5) node[left] {$9$};
  \draw (-.2,5.5) node[left] {$11$};
  \draw (-.2,6.5) node[left] {$13$};
  \draw (-.2,7.5) node[left] {$15$};
  \draw (-.2,8.5) node[left] {$17$};
  \fill (0.5, 0.5) circle (.15);
  \fill (0.5, 1.5) circle (.15);
  \draw (1.5, 1.5) node {$2$};
  \draw (1.5, 2.5) node {$2$};
  \draw (2.5, 2.5) node {$3$};
  \draw (2.5, 3.5) node {$3$};
  \draw (3.5, 3.5) node {$2$};
  \draw (3.5, 4.5) node {$2$};
  \draw (4.5, 4.5) node {$3$};
  \draw (4.5, 5.5) node {$3$};
  \draw (5.5, 5.5) node {$2$};
  \draw (5.5, 6.5) node {$2$};
  \fill (6.5, 6.5) circle (.15);
  \fill (6.5, 7.5) circle (.15);
  \fill (7.5, 7.5) circle (.15);
  \fill (7.5, 8.5) circle (.15);
\end{tikzpicture}
\end{document}
"""

NATIVE_F2_7_4 = {
    (-7, -17): 1, (-7, -15): 1, (-6, -15): 1, (-6, -13): 1, (-5, -13): 2, (-5, -11): 2,
    (-4, -11): 3, (-4, -9): 3, (-3, -9): 2, (-3, -7): 2, (-2, -7): 3, (-2, -5): 3,
    (-1, -5): 2, (-1, -3): 2, (0, -3): 1, (0, -1): 1,
}


def test_knotkit_grid_decodes_filled_circles_to_the_mirror_of_native():
    assert adapters._mirrorKhovanov(adapters._parseKnotkitGrid(KK_KH_3_1_Q)) == NATIVE_RATIONAL["3_1"]


def test_knotkit_grid_decodes_rank_nodes_to_the_mirror_of_native():
    groups = adapters._parseKnotkitGrid(KK_KH_7_4_Z2)
    assert max(groups.values()) == 3
    assert adapters._mirrorKhovanov(groups) == NATIVE_F2_7_4


def test_knotkit_grid_rejects_an_unrecognised_element():
    tampered = KK_KH_3_1_Q.replace(
        r"\end{tikzpicture}", "  \\draw (1,1) -- (2,2);\n\\end{tikzpicture}")
    with pytest.raises(ValueError, match="unrecognised element"):
        adapters._parseKnotkitGrid(tampered)


def test_knotkit_grid_rejects_a_rank_total_that_does_not_reconcile():
    tampered = KK_KH_3_1_Q.replace(r"\rank Kh = 4", r"\rank Kh = 5")
    with pytest.raises(ValueError, match="grid sums to 4"):
        adapters._parseKnotkitGrid(tampered)


def test_knotkit_grid_rejects_a_generator_off_the_labelled_axes():
    tampered = KK_KH_3_1_Q.replace(r"\fill (0.5, 0.5)", r"\fill (9.5, 0.5)")
    with pytest.raises(ValueError, match="no axis label"):
        adapters._parseKnotkitGrid(tampered)


def test_knotkit_s_reads_exactly_one_line():
    assert adapters._parseKnotkitS("s(PD[X[1,5,2,4]]; Q) = 2\n") == 2
    assert adapters._parseKnotkitS("s(10_124; Q) = -8\n") == -8
    for text in ("", "s(3_1; Q) = 2\ns(3_1; Q) = 2\n", "nothing here\n"):
        with pytest.raises(ValueError, match="expected one s line"):
            adapters._parseKnotkitS(text)
