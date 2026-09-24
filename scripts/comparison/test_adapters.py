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


def test_parse_khovanov_poly_rejects_parenthesized_groups():
    """``(q^6 + q^4)*t^2`` split on ``+`` keeps the total rank but misplaces a grading."""
    with pytest.raises(ValueError, match="parenthesized"):
        adapters._parseKhovanovPoly("q^10*t^4 + (q^6 + q^4)*t^2 + 1")


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


# A KhoHo run's KHOHO_ROWS line for 5_2, captured in the sandbox from knots.from_name(...).pd_code
# (gp expands the Laurent polynomial itself, so there is no polynomial text to misparse).
KHOHO_ROWS_5_2 = 'KHOHO_ROWS [[0, 1, 1], [0, 3, 1], [1, 3, 1], [2, 5, 1], [2, 7, 1], [3, 9, 1], [4, 9, 1], [5, 13, 1]]'
NATIVE_RATIONAL_5_2 = {(-5, -13): 1, (-4, -9): 1, (-3, -9): 1, (-2, -7): 1, (-2, -5): 1, (-1, -3): 1, (0, -3): 1, (0, -1): 1}


def test_khoho_rows_mirror_to_native():
    groups = adapters._parseKhohoRows("Computing Betti numbers ...\n" + KHOHO_ROWS_5_2 + "\n")
    assert adapters._mirrorKhovanov(groups) == NATIVE_RATIONAL_5_2
    assert groups != NATIVE_RATIONAL_5_2          # chiral: the raw value is not canonical


def test_khoho_rows_reject_missing_duplicate_or_nonpositive():
    with pytest.raises(ValueError, match="expected one KHOHO_ROWS line"):
        adapters._parseKhohoRows("no rows here\n")
    with pytest.raises(ValueError, match="expected one KHOHO_ROWS line"):
        adapters._parseKhohoRows(KHOHO_ROWS_5_2 + "\n" + KHOHO_ROWS_5_2 + "\n")
    with pytest.raises(ValueError, match="non-positive KhoHo rank"):
        adapters._parseKhohoRows("KHOHO_ROWS [[0, 1, 0]]\n")


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


# ---- KnotJob sectioned output (integral, reduced, width, odd, sl(3)) ------------------------
# Real knotjob output captured in the sandbox from knots.from_name(...).pd_code.

KJ_KB_5_2 = """Knot 1
S-Invariant mod 0 : 2
Integral unreduced Khovanov Homology : q + q^3 + t q^3 + t^2 q^5 + t^2 q^7 + t^3 q^9 + t^4 q^9 + t^5 q^13
Torsion of order 2 : t^2 q^5 + t^3 q^7 + t^5 q^11
Integral reduced Khovanov Homology : q^2 + t q^4 + 2 t^2 q^6 + t^3 q^8 + t^4 q^10 + t^5 q^12
"""

KJ_KB_8_19 = """Knot 1
S-Invariant mod 0 : 6
Integral unreduced Khovanov Homology : q^5 + q^7 + t^2 q^9 + t^3 q^13 + t^4 q^11 + t^4 q^13 + t^5 q^15 + t^5 q^17
Torsion of order 2 : t^3 q^11
Integral reduced Khovanov Homology : q^6 + t^2 q^10 + t^3 q^12 + t^4 q^12 + t^5 q^16
"""

KJ_KO_5_2 = """Knot 1
Odd integral Khovanov Homology : q^2 + t q^4 + 2 t^2 q^6 + t^3 q^8 + t^4 q^10 + t^5 q^12
"""

KJ_KS_3_1 = """Knot 1
Unreduced integral sl_3 Homology : t^-3 q^12 + t^-3 q^14 + t^-2 q^6 + t^-2 q^8 + q^2 + q^4 + q^6
Torsion of order 3 : t^-2 q^10
Reduced integral sl_3 Homology : t^-3 q^12 + t^-2 q^8 + q^4
"""

UNREDUCED = "Integral unreduced Khovanov Homology"
REDUCED = "Integral reduced Khovanov Homology"


def test_knotjob_sections_keep_torsion_with_the_homology_it_follows():
    sections, scalars = adapters._knotjobSections(KJ_KB_8_19)
    assert set(sections) == {UNREDUCED, REDUCED}
    assert scalars == {"S-Invariant mod 0": "6"}
    assert sections[UNREDUCED]["torsion"] == {2: {(3, 11): 1}}
    assert sections[REDUCED]["torsion"] == {}
    assert sum(sections[REDUCED]["free"].values()) == 5


def test_knotjob_torsion_order_is_read_not_assumed():
    sections, _ = adapters._knotjobSections(KJ_KS_3_1)
    assert sections["Unreduced integral sl_3 Homology"]["torsion"] == {3: {(-2, 10): 1}}
    assert sections["Reduced integral sl_3 Homology"]["torsion"] == {}


def test_f2_counts_every_even_order_torsion_and_no_odd():
    """Z/2 and Z/4 each contribute to F2 by UCT; Z/3 contributes nothing."""
    section = {"free": {(0, 1): 1},
               "torsion": {2: {(2, 5): 1}, 4: {(3, 7): 1}, 3: {(4, 9): 1}}}
    assert adapters._evenTorsion(section) == {(2, 5): 1, (3, 7): 1}
    f2 = adapters._f2FromIntegral(section["free"], adapters._evenTorsion(section))
    assert f2 == {(0, 1): 1, (2, 5): 1, (1, 5): 1, (3, 7): 1, (2, 7): 1}


def test_khovanov_width_thin_and_thick():
    """5_2 is alternating, so thin (two diagonals); 8_19 = T(3,4) is Khovanov-thick."""
    thin, _ = adapters._knotjobSections(KJ_KB_5_2)
    thick, _ = adapters._knotjobSections(KJ_KB_8_19)
    assert adapters._khovanovWidth(thin[UNREDUCED]) == 2
    assert adapters._khovanovWidth(thick[UNREDUCED]) == 3


def test_khovanov_width_rejects_mixed_parity_support():
    with pytest.raises(ValueError, match="mixed-parity"):
        adapters._khovanovWidth({"free": {(0, 1): 1, (0, 2): 1}, "torsion": {}})


def test_integral_summary_counts_free_rank_and_each_torsion_order():
    sections, _ = adapters._knotjobSections(KJ_KB_5_2)
    assert adapters._integralSummary(sections[UNREDUCED]) == "free=8 Z/2x3"
    assert adapters._integralSummary(sections[REDUCED]) == "free=7"
    odd, _ = adapters._knotjobSections(KJ_KO_5_2)
    assert adapters._integralSummary(odd["Odd integral Khovanov Homology"]) == "free=7"


def test_knotjob_torsion_before_any_homology_raises():
    with pytest.raises(ValueError, match="before any homology"):
        adapters._knotjobSections("Knot 1\nTorsion of order 2 : t q^3\n")


def test_knotjob_missing_section_raises():
    sections, _ = adapters._knotjobSections(KJ_KO_5_2)
    with pytest.raises(ValueError, match="no 'Integral reduced"):
        adapters._knotjobSection(sections, REDUCED)


# ---- khoca integral (ring 0) --------------------------------------------------------------
# The unreduced half of KH('braidaBaB') -- the figure-eight -- verbatim from khoca's own
# InteractiveCalculator docstring. Real output carrying zero and negative multiplicities that
# must cancel per key before anything is read off it.
KHOCA_Z_4_1_UNREDUCED = [
    [-2, 3, 0, 1], [-2, 5, 0, 1], [-1, 1, 0, 1], [-1, 3, 0, 1], [0, -1, 0, 1], [0, 1, 0, 1],
    [1, -3, 0, 1], [1, -1, 0, 1], [2, -5, 0, 1], [2, -3, 0, 1], [-1, 3, 2, 1], [-2, 3, 0, -1],
    [-1, 3, 0, -1], [-2, 5, 0, 0], [-1, 5, 0, 0], [-1, 1, 0, 0], [0, 1, 0, 0], [-1, 3, 0, 0],
    [0, 3, 0, 0], [0, -1, 0, 0], [1, -1, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [2, -3, 2, 1],
    [1, -3, 0, -1], [2, -3, 0, -1], [1, -1, 0, 0], [2, -1, 0, 0],
]


def test_khoca_integral_section_cancels_multiplicities_per_key():
    section = adapters._khocaIntegralSection(KHOCA_Z_4_1_UNREDUCED)
    assert adapters._integralSummary(section) == "free=6 Z/2x2"
    assert section["torsion"] == {2: {(-1, -3): 1, (2, 3): 1}}


def test_khoca_integral_torsion_gives_native_f2_without_a_degree_shift():
    """The torsion placement claim: UCT on khoca's q-negated groups, unshifted, is native F2.
    Moving the torsion one homological degree -- the homology/cohomology confusion -- is not."""
    section = adapters._khocaIntegralSection(KHOCA_Z_4_1_UNREDUCED)
    torsion = adapters._evenTorsion(section)
    assert adapters._f2FromIntegral(section["free"], torsion) == NATIVE_F2["4_1"]
    shifted = {(h - 1, q): count for (h, q), count in torsion.items()}
    assert adapters._f2FromIntegral(section["free"], shifted) != NATIVE_F2["4_1"]


def test_khoca_integral_width_of_the_figure_eight_is_two():
    assert adapters._khovanovWidth(adapters._khocaIntegralSection(KHOCA_Z_4_1_UNREDUCED)) == 2


def test_khoca_integral_negative_aggregate_raises():
    with pytest.raises(ValueError, match="negative aggregate multiplicity"):
        adapters._khocaIntegralSection([[0, 1, 0, 1], [0, 1, 0, -2]])


# ---- JavaKh integral (-Z) -----------------------------------------------------------------
# Real javakh -Z output captured in the sandbox from knots.from_name(...).pd_code.
JAVAKH_Z_3_1 = '"q^1*t^0*Z[0] + q^3*t^0*Z[0] + q^5*t^2*Z[0] + q^7*t^3*Z[2] + q^9*t^3*Z[0]"'
JAVAKH_Z_5_2 = ('"q^1*t^0*Z[0] + q^3*t^0*Z[0] + q^3*t^1*Z[0] + q^5*t^2*Z[0,2] + q^7*t^2*Z[0] + '
                'q^7*t^3*Z[2] + q^9*t^3*Z[0] + q^9*t^4*Z[0] + q^11*t^5*Z[2] + q^13*t^5*Z[0]"')


def test_javakh_integral_parses_free_and_torsion_per_bidegree():
    integral = adapters._parseJavakhIntegral(JAVAKH_Z_3_1)
    assert integral["free"] == {(0, 1): 1, (0, 3): 1, (2, 5): 1, (3, 9): 1}
    assert integral["torsion"] == {2: {(3, 7): 1}}
    assert adapters._integralSummary(adapters._parseJavakhIntegral(JAVAKH_Z_5_2)) == "free=8 Z/2x3"


def test_javakh_integral_counts_repeated_entries_and_coefficients():
    integral = adapters._parseJavakhIntegral('"q^5*t^2*Z[0,2,2] + 2*q^1*t^0*Z[0]"')
    assert integral["free"] == {(2, 5): 1, (0, 1): 2}
    assert integral["torsion"] == {2: {(2, 5): 2}}


def test_javakh_f2_is_uct_on_raw_groups_then_mirror():
    """The ordering claim: UCT first, then mirror, is native F2. Mirroring first is not,
    because the mirror of integral homology also moves torsion one degree."""
    integral = adapters._parseJavakhIntegral(JAVAKH_Z_3_1)
    torsion = adapters._evenTorsion(integral)
    uct_then_mirror = adapters._mirrorKhovanov(adapters._f2FromIntegral(integral["free"], torsion))
    mirror_then_uct = adapters._f2FromIntegral(adapters._mirrorKhovanov(integral["free"]),
                                               adapters._mirrorKhovanov(torsion))
    assert uct_then_mirror == NATIVE_F2["3_1"]
    assert mirror_then_uct != NATIVE_F2["3_1"]


def test_javakh_integral_rejects_unparseable_or_empty_output():
    with pytest.raises(ValueError, match="unparseable javakh -Z term"):
        adapters._parseJavakhIntegral('"q^1*t^0*Z[0] + garbage"')
    with pytest.raises(ValueError, match="empty javakh -Z output"):
        adapters._parseJavakhIntegral('""')


# ---- Regina Alexander -> determinant ------------------------------------------------------
# Real regina alexander() strings captured in the sandbox from knots.from_name(...).pd_code.

def test_determinant_is_alexander_at_minus_one_in_absolute_value():
    cases = {"x^2 - x + 1": 3, "x^2 - 3 x + 1": 5, "2 x^2 - 3 x + 2": 7,
             "x^6 - x^5 + x^3 - x + 1": 3, "x^8 - x^7 + x^5 - x^4 + x^3 - x + 1": 1}
    for text, determinant in cases.items():
        assert adapters._determinantFromAlexander(adapters._parseLaurent(text, "x")) == determinant


# ---- SageMath target rows -----------------------------------------------------------------
# Values from scripts/probe_sage_targets.py on CT 250 (SageMath 9.5). The probe sampled
# omega_signature at k = 1..5; k = 6 (omega = -1) is the guard sample added in the adapter.

def test_signature_function_is_negated_after_the_omega_minus_one_guard():
    samples = [0, 1, 2, 2, 2, 2]          # 3_1: probe's k=1..5, then omega = -1
    assert adapters._canonicalSignatureFunction(samples, 2) == [0, -1, -2, -2, -2, -2]


def test_signature_function_guard_rejects_a_sample_that_is_not_the_signature():
    with pytest.raises(ValueError, match="omega_signature"):
        adapters._canonicalSignatureFunction([0, 1, 2, 2, 2, 2], -2)
    with pytest.raises(ValueError, match="6 Levine-Tristram samples"):
        adapters._canonicalSignatureFunction([0, 1, 2, 2, 2], 2)


def test_sage_target_fields_parse_from_tagged_output():
    out = ("HOMFLY '-L^4 + L^2*M^2 - 2*L^2'\nHOMFLY_SECONDS 0.1345\n"
           "OMEGA_SIGNATURE [0, 1, 2, 2, 2, 2]\nOMEGA_SECONDS 1.1867\nSIGNATURE 2\n"
           "ARF 1\nARF_SECONDS 0.0164\nnoise line\n")
    fields = adapters._parseSageFields(out, adapters._SAGE_TARGET_TAGS)
    assert set(fields) == set(adapters._SAGE_TARGET_TAGS)
    assert fields["HOMFLY"] == "-L^4 + L^2*M^2 - 2*L^2"
    assert fields["OMEGA_SIGNATURE"] == [0, 1, 2, 2, 2, 2]
    assert fields["ARF"] == 1


def test_sage_target_script_compiles_as_python():
    """Sage's preparser accepts a superset of Python, so a syntax error here would also
    break on CT 250 -- the one check possible without sage."""
    script = adapters._SAGE_TARGET_SCRIPT % {"pd": "[[1, 5, 2, 4], [3, 1, 4, 6], [5, 3, 6, 2]]",
                                             "reps": 1}
    compile(script, "targets.sage", "exec")


# ---- kht++ (braid-word Morse encoding, Bar-Natan complex at H = 0) ------------------------
# Real cxCKh-c2 data files captured in the sandbox, from each knot's KnotInfo braid word
# (the generated-by header comment line dropped). 8_19 carries a C_2 summand (H^2).
KHTPP_3_1 = '1) h^ 0 q^-2 δ^-1 ⬮\n2) h^-3 q^-8 δ^-1 ⬮——H—>⬮\n'
KHTPP_8_19 = '1) h^ 0 q^ -6 δ^-3 ⬮\n2) h^-3 q^-12 δ^-3 ⬮——H—>⬮\n3) h^-5 q^-16 δ^-3 ⬮——H^2—>⬮\n'
NATIVE_F2_8_19 = {(-5, -17): 1, (-5, -15): 1, (-4, -13): 1, (-4, -11): 1, (-3, -13): 1, (-3, -11): 1, (-2, -11): 1, (-2, -9): 1, (0, -7): 1, (0, -5): 1}


def test_khtpp_morse_word_closes_a_braid_as_a_one_one_tangle():
    assert adapters._khtppMorseWord([1, -2, 1, -2]) == "% braid closure\nl1.l2.x0.y1.x0.y1.u2.u1\n,0\n"
    assert adapters._khtppMorseWord([1, 1, 1]) == "% braid closure\nl1.x0.x0.x0.u1\n,0\n"


def test_khtpp_reduced_f2_tensored_is_native_f2_for_a_chiral_knot():
    reduced = adapters._parseKhtppComplex(KHTPP_3_1)
    assert sum(reduced.values()) == 3
    assert adapters._shumakovitchF2(reduced) == NATIVE_F2["3_1"]


def test_khtpp_c_n_target_is_one_degree_up_not_n():
    """A C_n summand's target sits at (h + 1, q + 2n): the differential raises h by one.
    8_19 has a C_2, so placing the target at (h + n, q + 2n) instead must fail."""
    assert adapters._shumakovitchF2(adapters._parseKhtppComplex(KHTPP_8_19)) == NATIVE_F2_8_19
    wrong = {}
    for line in KHTPP_8_19.splitlines():
        h, q, n = adapters._KHTPP_LINE.match(line).group(1, 2, 4)
        h, q = int(h), int(q)
        power = int(n.split("^")[1].split("—")[0]) if "H^" in n else 1
        keys = [(h, q)] + ([(h + power, q + 2 * power)] if "H" in n else [])
        for key in keys:
            wrong[key] = wrong.get(key, 0) + 1
    assert adapters._shumakovitchF2(wrong) != NATIVE_F2_8_19


def test_khtpp_parser_rejects_unknown_lines_and_bad_gradings():
    with pytest.raises(ValueError, match="unrecognised kht\\+\\+ summand"):
        adapters._parseKhtppComplex("1) h^ 0 q^ 0 δ^0 something else\n")
    with pytest.raises(ValueError, match="violates q/2 = h \\+ delta"):
        adapters._parseKhtppComplex("1) h^ 0 q^ 2 δ^0 ⬮\n")
    with pytest.raises(ValueError, match="empty kht"):
        adapters._parseKhtppComplex("% only a comment\n")


def test_braid_notation_flat_nested_and_malformed():
    assert adapters._parseBraidNotation("[1,-2,1,-2]") == [1, -2, 1, -2]
    assert adapters._parseBraidNotation(
        "[[-1,-1,-2,3,-2,1,-2,-2,3,2,2],[-1,2,-1,2,3,-2,-2,-4,3,-4]]") == [
        -1, -1, -2, 3, -2, 1, -2, -2, 3, 2, 2]
    for text in ("[]", "[1,0,2]", "[[1,2],3]", "not a braid"):
        with pytest.raises((ValueError, SyntaxError)):
            adapters._parseBraidNotation(text)
