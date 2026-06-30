# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Declarative catalog for the comparison artifact (BENCHMARKS.md).

This is the *prose* half of the chart: every invariant Tetradrome computes or aims to compute,
grouped by the computational gold-master that already exists for it, with the math, the I/O, a
status flag, and whether KnotInfo tabulates it. The *measured* half (timings, value agreement)
is filled in at generation time by ``generate.py`` against ``adapters.py``; nothing here is a
number. Keep this hand-authored and reviewed -- it is the spec, the generator only measures.

Status flags (honest about what is real):
  done      -- implemented natively and validated against KnotInfo
  landing   -- engine in progress (the grid Floer Phase 6 work)
  near      -- cleanly authorable on machinery we already have (coverage-map Tier 1)
  build     -- a substantial native engine, well-defined (Tier 2)
  bound     -- no program computes it for general knots; we author the bound our invariants
               imply and surface the tabulated value, never resolving open cases (Tier 3)
  research  -- a major separate engine, decision deferred (Tier 4 / 4D apex)
  out       -- explicitly out of combinatorial scope (gauge theory)

Group keys are the computational gold-master. KnotInfo is NOT a group here except for the band
where it is the *sole* source (the Tier-3 bounds): everywhere else it is an orthogonal
"tabulated?" marker on each row, because its values were themselves produced by these programs.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Group:
    key: str
    title: str
    oracle: str                 # the external tool(s) this group is checked against
    blurb: str                  # the blanket value-add that holds for every row in the group


@dataclasses.dataclass(frozen=True)
class Invariant:
    name: str                   # join key: tetradrome compute-name where one exists, else a slug
    label: str                  # human label for the row
    group: str                  # Group.key
    math: str                   # one line: what is computed
    inputs: str
    outputs: str
    status: str                 # one of the flags above
    knotinfo: str               # yes | partial | no
    tetra: tuple | None         # ("compute", name) | ("floer", None) | None (not implemented)


# Ordered so the chart reads from the most-grounded groups to the most-aspirational.
GROUPS = [
    Group(
        "kfh",
        "knot Floer homology -- vs Szabo's HFK Calculator (`knot_floer_homology`)",
        "knot_floer_homology (Szabo HFKcalc, GPLv2+, binary wheel)",
        "Tetradrome computes these natively from a grid diagram (Apache-2.0, pure Python, "
        "auditable, multi-core/optional-GPU accelerated with every tier checked to reproduce the "
        "reference). `knot_floer_homology` wraps Szabo's compiled-C HFKcalc and is a validator "
        "only, never a compute backend (decision 0006). Expect the compiled cube to be far faster "
        "than the grid engine on a real knot -- that gap is the documented case for the Tier-2 "
        "Szabo-cube engine.",
    ),
    Group(
        "knotjob",
        "Khovanov-family homology -- vs KnotJob / JavaKh / KnotTheory`",
        "KnotJob (Schutz, Java); JavaKh/knotkit; KnotTheory` (Bar-Natan, Mathematica)",
        "Native cube-of-resolutions over F2 and Q (and, planned, Z/reduced/odd). KnotInfo's "
        "Khovanov columns are themselves KnotJob output. Pure Python and one schema vs Java / "
        "Mathematica; reproducible across the acceleration tiers.",
    ),
    Group(
        "sage",
        "Classical & polynomial invariants -- vs SageMath / KnotTheory`",
        "SageMath knot tools; KnotTheory`; (Spherogram's are Sage-gated)",
        "Computed natively from a Seifert matrix or a skein recursion. KnotInfo tabulates the "
        "values; Sage is the live computational peer (Spherogram under plain pip is diagram-only). "
        "Pure Python, no Sage runtime required.",
    ),
    Group(
        "khoca",
        "Khovanov-Rozansky / sl(N) -- vs Khoca",
        "Khoca (C++/Python research program)",
        "Aspirational higher-homology direction. Khoca demonstrates the computation exists and is "
        "the reference target; no native engine yet.",
    ),
    Group(
        "snappy",
        "Hyperbolic & geometric -- vs SnapPy / SnapPea",
        "SnapPy / SnapPea (the geometric oracle)",
        "A SnapPea-class numerical-geometry engine; build-vs-oracle-only is deliberately deferred. "
        "Diagram I/O (PD/DT/Gauss/braid) is native; the geometry is not yet.",
    ),
    Group(
        "knotinfo_bounds",
        "Concordance & 4-genus bounds -- KnotInfo tabulated, no computing program",
        "KnotInfo (tabulated values / ranges / unknown)",
        "No program computes these for general knots. Tetradrome authors the BOUND its invariants "
        "imply (|s|/2 <= g4, signature bounds, ...) and surfaces KnotInfo's tabulated value -- and "
        "never presents a bound as a resolved value.",
    ),
    Group(
        "apex",
        "Four-manifold apex -- no external oracle",
        "literature / by-hand / internal consistency (no KnotInfo-scale oracle exists)",
        "The north star: knot invariants reframed as 4D obstructions, knot->4-manifold bridges, "
        "Floer up the dimensions, genuinely-4D structures, and the Piccirillo trace flagship. "
        "Validated by reduction to invariants we already check, structural laws, and literature "
        "examples -- not a timing race.",
    ),
]


INVARIANTS = [
    # ---- knot Floer (kfh group) -------------------------------------------------------------
    Invariant("hfk", "HFK-hat (ranks / polynomial)", "kfh",
              "Grid (MOS) knot Floer homology; bigraded ranks, Euler char = Alexander.",
              "PD / grid", "bigraded ranks, HFK polynomial", "landing", "yes", ("floer", None)),
    Invariant("tau", "Ozsvath-Szabo tau", "kfh",
              "tau from the Alexander filtration on grid homology (concordance, |tau| <= g4).",
              "PD / grid", "integer", "landing", "yes", ("floer", None)),
    Invariant("seifert_genus", "Seifert genus (via HFK)", "kfh",
              "Top Alexander grading with nonzero HFK (HFK detects genus).",
              "PD / grid", "integer", "landing", "yes", None),
    Invariant("fibered", "Fibered-ness (via HFK)", "kfh",
              "HFK detects fibredness (Ni): top Alexander grading has rank 1.",
              "PD / grid", "bool", "near", "yes", None),
    Invariant("epsilon", "Hom epsilon", "kfh",
              "epsilon from the CFK^infinity structure (concordance).",
              "PD / grid", "integer in {-1,0,1}", "near", "partial", None),
    Invariant("nu", "Ozsvath-Szabo nu", "kfh",
              "nu from (tau, epsilon).",
              "PD / grid", "integer", "near", "partial", None),
    Invariant("l_space", "L-space knot predicate", "kfh",
              "Knot is an L-space knot iff HFK is a thin staircase (all ranks 1).",
              "PD / grid", "bool", "near", "yes", None),

    # ---- Khovanov family (knotjob group) ----------------------------------------------------
    Invariant("khovanov_homology", "Khovanov homology (F2)", "knotjob",
              "Unreduced Khovanov over F2 from the cube of resolutions; d^2 = 0 checked.",
              "PD", "bigraded Betti numbers", "done", "yes", ("compute", "khovanov_homology")),
    Invariant("rational_khovanov_homology", "Khovanov homology (Q)", "knotjob",
              "Unreduced Khovanov over Q (exact rational reduction).",
              "PD", "bigraded Betti numbers", "done", "yes",
              ("compute", "rational_khovanov_homology")),
    Invariant("rasmussen_s", "Rasmussen s", "knotjob",
              "s read off the quantum filtration on Lee homology over Q (|s|/2 <= g4).",
              "PD", "even integer", "done", "yes", ("compute", "rasmussen_s")),
    Invariant("lee_homology", "Lee homology (Q)", "knotjob",
              "Lee deformation of Khovanov over Q (2-dim for a knot); the source of s.",
              "PD", "filtered homology", "done", "partial", None),
    Invariant("khovanov_integral", "Khovanov homology (Z, torsion)", "knotjob",
              "Integral Khovanov including torsion; extends the F2/Q engine.",
              "PD", "bigraded groups + torsion", "build", "yes", None),
    Invariant("khovanov_odd", "Odd Khovanov homology", "knotjob",
              "Odd Khovanov (Ozsvath-Rasmussen-Szabo), a sibling engine.",
              "PD", "bigraded groups", "build", "yes", None),
    Invariant("khovanov_width", "Homological width", "knotjob",
              "Width of (integral) Khovanov homology.",
              "PD", "integer", "build", "yes", None),

    # ---- classical / polynomial (sage group) ------------------------------------------------
    Invariant("jones_polynomial", "Jones polynomial", "sage",
              "Kauffman bracket over the resolution cube.",
              "PD", "Laurent polynomial", "done", "yes", ("compute", "jones_polynomial")),
    Invariant("determinant", "Determinant", "sage",
              "|Delta(-1)| from the Seifert form.",
              "braid / PD", "integer", "done", "yes", ("compute", "determinant")),
    Invariant("signature", "Signature", "sage",
              "Signature of V + V^T (Seifert form).",
              "braid / PD", "integer", "done", "yes", ("compute", "signature")),
    Invariant("alexander_polynomial", "Alexander polynomial", "sage",
              "Canonical Alexander from the Seifert form.",
              "braid / PD", "Laurent polynomial", "done", "yes",
              ("compute", "alexander_polynomial")),
    Invariant("conway_polynomial", "Conway polynomial", "sage",
              "Alexander in the Conway variable z = t^.5 - t^-.5.",
              "braid / PD", "polynomial", "near", "yes", None),
    Invariant("homfly_polynomial", "HOMFLY-PT polynomial", "sage",
              "HOMFLY-PT skein recursion (crossing resolution / Hecke).",
              "PD", "two-variable polynomial", "near", "yes", None),
    Invariant("kauffman_polynomial", "Kauffman polynomial", "sage",
              "Dubrovnik / F two-variable skein.",
              "PD", "two-variable polynomial", "near", "yes", None),
    Invariant("signature_function", "Levine-Tristram signature function", "sage",
              "Signature of (1-w)V + (1-wbar)V^T for w on the unit circle.",
              "braid / PD", "step function on S^1", "near", "yes", None),
    Invariant("arf_invariant", "Arf invariant", "sage",
              "Delta(t) mod 8 / Seifert-form Arf.",
              "braid / PD", "bit", "near", "yes", None),
    Invariant("algebraic_concordance_order", "Algebraic concordance order", "sage",
              "Witt class of the Seifert form (Levine's group).",
              "braid / PD", "order (int or inf)", "near", "yes", None),
    Invariant("braid_index", "Braid index", "sage",
              "MFW (HOMFLY) lower bound + Seifert-circle upper bound.",
              "PD / braid", "integer", "build", "yes", None),

    # ---- Khovanov-Rozansky (khoca group) ----------------------------------------------------
    Invariant("sl_n_homology", "sl(N) / Khovanov-Rozansky homology", "khoca",
              "Categorified sl(N) invariant (HOMFLY-homology family).",
              "PD", "triply-graded groups", "research", "no", None),

    # ---- hyperbolic / geometric (snappy group) ----------------------------------------------
    Invariant("hyperbolic_volume", "Hyperbolic volume", "snappy",
              "Volume of the complement via an ideal triangulation solved to high precision.",
              "PD / triangulation", "real", "research", "yes", None),
    Invariant("chern_simons", "Chern-Simons invariant", "snappy",
              "Chern-Simons of the hyperbolic structure.",
              "PD / triangulation", "real mod 1", "research", "yes", None),
    Invariant("a_polynomial", "A-polynomial", "snappy",
              "SL2(C) character-variety A-polynomial.",
              "PD / triangulation", "two-variable polynomial", "research", "partial", None),

    # ---- bounds, KnotInfo-only (knotinfo_bounds group) --------------------------------------
    Invariant("smooth_four_genus", "Smooth 4-genus (bound)", "knotinfo_bounds",
              "Bounds from |s|/2 and |tau|; exact value open in general.",
              "PD", "integer range", "bound", "yes", None),
    Invariant("topological_four_genus", "Topological 4-genus (bound)", "knotinfo_bounds",
              "Bounds from the Levine-Tristram signature; exact value open in general.",
              "PD / braid", "integer range", "bound", "yes", None),
    Invariant("unknotting_number", "Unknotting number (bound + u=1 cert)", "knotinfo_bounds",
              "Exact value open; but certifying u=1 / finding an unknotting crossing is buildable.",
              "PD", "integer range / certificate", "bound", "yes", None),
    Invariant("concordance_order", "Smooth concordance order (bound)", "knotinfo_bounds",
              "Obstructions from concordance invariants; exact order open in general.",
              "PD", "order (int or inf)", "bound", "yes", None),

    # ---- four-manifold apex (apex group) ----------------------------------------------------
    Invariant("sliceness_obstruction", "Sliceness obstruction (smooth vs topological)", "apex",
              "Slice-Bennequin / Floer-Khovanov obstructions; the smooth-vs-topological gap.",
              "PD / braid", "obstruction verdict", "research", "partial", None),
    Invariant("knot_trace", "Knot trace X_n(K)", "apex",
              "The 4-manifold from an n-framed 2-handle on B^4 along K; intersection form + boundary.",
              "PD + framing", "handle description", "research", "no", None),
    Invariant("d_invariant", "d-invariants (correction terms)", "apex",
              "d-invariants of surgeries / branched covers (Manolescu-Owens delta obstruction).",
              "knot Floer complex", "rationals", "research", "no", None),
    Invariant("upsilon", "Upsilon / concordance homomorphisms", "apex",
              "Upsilon(t), nu+, Hom phi_j read off CFK^infinity -- maps to the concordance group.",
              "knot Floer complex", "piecewise-linear function / homs", "research", "partial", None),
    Invariant("skein_lasagna", "Khovanov skein-lasagna module", "apex",
              "The genuinely-4D refinement of Khovanov (Morrison-Walker-Wedrich); detects exotica.",
              "4-manifold + link", "graded module", "research", "no", None),
    Invariant("piccirillo_trace", "Piccirillo trace method (flagship)", "apex",
              "Sliceness via trace siblings: trace-embedding lemma + d-invariant obstruction.",
              "PD", "slice verdict via friend", "research", "no", None),
]


def groups_in_order():
    return list(GROUPS)


def invariants_for(group_key):
    return [inv for inv in INVARIANTS if inv.group == group_key]
