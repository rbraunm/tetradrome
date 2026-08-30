# Coverage map: authoring KnotInfo from first principles

The long-horizon goal is to **author**, from first principles, native engines for as much of
KnotInfo as is honestly computable — and to **validate each against the oracle**, never to wrap
it. KnotInfo (and the programs it aggregates: KnotTheory`, Szabó's HFK Calculator, JavaKh/knotkit,
SnapPy/SnapPea, …) is the ground truth we check *against*; the value we add is the auditable
derivation, not the number.

This is the *feeder*, not the goal: the project is a 4D-topology workbench, and knot/concordance
invariants matter here because they obstruct surfaces and slice disks in 4-manifolds. The
4-dimensional objects those invariants feed into — sliceness, knot traces, Kirby calculus,
d-invariants, concordance structure, trisections, skein lasagna — are mapped in
`four-manifold-objectives.md`. Knots are dense in checkable invariants, so they get implemented
first and right; they are not the subject.

This document is the map: every substantive KnotInfo column, the method that would compute it,
what we already have, and — critically — an honest cost/feasibility tier. Column names are the
machine keys from `database_knotinfo` (122 non-identifier columns out of 244 fields), so each row
is directly checkable against the offline database.

**Validation discipline (every engine, no exceptions):** reproduce the oracle's value exactly on
the tabulated knots (small-knot suite), recorded in the claim ledger. An engine is not "done"
until it agrees; an optimization is not admitted until it agrees with the reference it accelerates.

**One honest caveat up front.** "Author all of KnotInfo" has a hard ceiling in two places: the
hyperbolic/geometric block (Tier 4) is a SnapPea-class numerical-geometry project, an order of
magnitude more engine than everything above it combined; and a band of invariants (Tier 3) are
*not computed by any program* for general knots — KnotInfo stores known values, ranges, or
"unknown." For those we can author the **bounds** our invariants imply and surface the tabulated
value, but we do not claim to resolve open cases. Saying otherwise would be the oversell §6 of the
homology-engine design exists to prevent.

---

## Tier 0 — authored and validated (have)

| Column(s) | Engine / method |
|---|---|
| `alexander_polynomial` (+`_vector`), `determinant` | Seifert form → `canonical_alexander`; det = \|Δ(−1)\| |
| `jones_polynomial` (+`_vector`) | Kauffman bracket (`jones.py`) |
| `khovanov_unreduced_*` (mod2, rational) | native Khovanov F2 + ℚ |
| `rasmussen_invariant` | Lee deformation → s |
| `hfk_polynomial` (+`_vector`), `three_genus` | grid (MOS) knot Floer; genus = top Alexander grading |
| `ozsvath_szabo_tau_invariant` | τ from the Alexander filtration |
| `seifert_matrix` | `seifert_matrix_from_braid` |

`grid_notation`, `braid_notation`, `pd_notation`, etc. are *ingested* (diagram readers), not
computed — see Tier I.

---

## Tier 1 — cleanly authorable, near-term (near)

Each is a small, well-understood computation on machinery we already have (a Seifert matrix, an
Alexander polynomial, an HFK group, a skein recursion).

| Column(s) | Method |
|---|---|
| `conway_polynomial` (+`_vector`) | Alexander in the Conway variable (z = t^½ − t^−½) |
| `homfly_polynomial` (+`_vector`) | HOMFLY-PT skein recursion (crossing resolution / Hecke) |
| `kauffman_polynomial` (+`_vector`) | Dubrovnik/F two-variable skein |
| `q_polynomial` | Brandt–Lickorish–Millett–Ho (specialization of Kauffman) |
| `signature`, `signature_function` | Levine–Tristram from V + V^T (ω on the unit circle) |
| `arf_invariant` | Δ(t) mod 8 / Seifert form Arf |
| `algebraic_concordance_order` | Witt class of the Seifert form (Levine's algebraic concordance group) |
| `epsilon`, `nu` | knot Floer ε from the CFK structure; ν from (τ, ε) — **Phase 7** |
| `fibered` | HFK detects fibredness (Ni): top Alexander grading rank 1 |
| `l_space` | knot is L-space ⇔ HFK is a staircase (thin, ranks 1) |
| `braid_length` | from `braid_notation` (word length); ingestion-adjacent |

---

## Tier 2 — authorable, substantial (build)

Real engines or real algebra, but bounded and well-defined.

| Column(s) | Method / engine |
|---|---|
| `khovanov_unreduced_integral_*` | integral Khovanov (Z coefficients incl. torsion) — extends the F2/ℚ engine |
| `khovanov_reduced_*` (Z, Q, mod2) | reduced Khovanov complex — basepoint-constrained front end, back end untouched; path in `homology-engine.md` §7 Phase 9. Cheapest Tier 2 entry: three computed oracles already provisioned |
| `khovanov_odd_*` (Z, Q, mod2) | odd Khovanov homology (Ozsváth–Rasmussen–Szabó) — a sibling engine |
| `width` | homological width of (integral) Khovanov |
| `turaev_genus` | Turaev surface from the diagram (combinatorial) |
| `hfk_polynomial` at scale | **Szabó cube-of-resolutions** Floer engine — the non-n! algorithm; the *only* path past the grid wall, and the one that makes "show the work" *and* "reach larger knots" coexist (see homology-engine §7 Phase 8 is a different lever — representation, not algorithm) |
| `monodromy` | open-book monodromy of fibered knots (from the fibration) |
| `nakanishi_index`, `torsion_numbers` | Alexander module structure (minimal generators / torsion) |
| `braid_index` | MFW (HOMFLY) lower bound + Seifert-circle upper bound; sharp for most |
| `thurston_bennequin_number` | Kauffman/HOMFLY bound on max tb (often sharp) |
| `arc_index` | = grid number; for alternating knots crossing + 2 (Bae–Park) |
| `crosscap_number` | nonorientable genus (bounds + small-case enumeration) |
| `quasi_alternating`, `adequate`, `almost_alternating` | diagrammatic / homological-thinness predicates |
| `positive`, `positive_braid`, `quasipositive`, `strongly_quasipositive`, `almost_strongly_qp` | positivity certificates (HOMFLY/τ/s obstructions + braid search) |

---

## Tier 3 — bounds, not exact values (bound)

KnotInfo records values, ranges, or "unknown" here; **no program computes these for general
knots**. Honest scope: author the bound our invariants give (e.g. \|s\|/2 ≤ g₄, σ bounds), surface
the bound and the oracle's tabulated value, and never present a bound as a resolved value.

`smooth_four_genus`, `topological_four_genus`, `smooth_4d_crosscap_number`,
`topological_4d_crosscap_number`, `smooth_concordance_genus`, `topological_concordance_genus`,
`smooth_concordance_crosscap_number`, `topological_concordance_crosscap_number`,
`smooth_concordance_order`, `topological_concordance_order`, `double_slice_genus`,
`fd_clasp_number`, `td_clasp_number`, `ribbon`, `ribbon_number`, `unknotting_number`,
`unknotting_number_algebraic`, `bridge_index`, `super_bridge_index`, `tunnel_number`,
`morse_novikov_number`, `crossing_number` (minimal, in general), `cosmetic_crossing`.

Note on `unknotting_number`: the *exact* value is the open/bounded problem here, but its
*operational* siblings — certifying u(K) = 1 and finding an unknotting crossing in a diagram — are a
distinct, buildable engine (Reidemeister-graph search for existence; Montesinos-trick / branched-cover
d-invariant + Nakanishi for the u ≥ 2 obstruction). They are an explicit prerequisite of the
Piccirillo flagship in `four-manifold-objectives.md`, not a Tier 3 bound.

---

## Tier 4 — hyperbolic & geometric (research)

A SnapPea-class engine: ideal triangulation, gluing-equation variety solved to high precision
(Newton), then derived geometry. This is a separate major project — by far the largest "no-wrap"
frontier — and the **decision to build vs. leave oracle-only is deliberately deferred**. Listed so
the boundary is explicit, not so it's promised.

`volume`, `volume_imaginary_part`, `chern_simons_invariant`, `maximum_cusp_volume`,
`longitude_translation`, `meridian_translation`, `longitude_length`, `meridian_length`,
`other_short_geodesics`, `full_symmetry_group`, `symmetry_type`, `geometric_type`,
`boundary_slopes`, `a_polynomial` (SL₂(ℂ) character variety — algebraic, equally heavy).

Also geometric/numerical and effectively their own studies: `polygon_index` (stick number),
`mosaic_tile_number`, `ropelength`.

---

## Tier I — ingestion & canonicalization (not invariant engines)

Diagram encodings and names: converting between them and canonicalizing is plumbing the schema
needs, but it computes identifiers, not invariants. Most we already read.

`name`, `category`, `dt_notation`, `gauss_notation`, `enhanced_gauss_notation`, `pd_notation`,
`braid_notation`, `conway_notation`, `two_bridge_notation`, `montesinos_notation`,
`pretzel_notation`, `grid_notation`, `*_braid_notation`, `positive_pd_notation`, `dt_name`,
`classical_conway_name`, `tetrahedral_census_name`, `alternating`, `small_large`.

---

## Sequencing note

Tier 1 is the natural next band after Phase 6/7: it reuses the Seifert form, Alexander, and HFK we
already author, and closes most of the *classical* and *concordance-from-Seifert* columns at low
cost. Tier 2's Khovanov extensions and the Szabó-cube Floer engine are the substantial homology
builds. Tier 3 is a posture (bounds + honesty) more than a build. Tier 4 is its own decision, to be
taken on its own merits — not folded silently into "finish KnotInfo."
