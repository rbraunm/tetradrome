# Coverage map: KnotInfo columns

Which KnotInfo columns Tetradrome computes natively, which it plans to, and by what method.
Each engine is validated against KnotInfo and the computed oracles, never wrapped around them
(decision 0006). The 4-dimensional objects these invariants feed are mapped in
`four-manifold-objectives.md`.

Column names are the machine keys from `database_knotinfo` (248 fields: 124 columns, each with
an `_anon` twin).

An engine is done when it reproduces the oracle's value exactly on the tabulated knots and the
claim ledger records it. An optimization is admitted when it agrees with the reference it
accelerates.

Two limits:

- **Tier 3** holds invariants whose exact value no program computes for a general knot.
  Tetradrome computes the bounds its invariants imply and shows the tabulated value; it does not
  resolve open cases.
- **Tier 4**, the hyperbolic and geometric block, is a separate numerical-geometry engine larger
  than everything above it combined. Whether to build it is undecided.

---

## Tier 0 — computed and validated (have)

| Column(s) | Engine / method |
|---|---|
| `alexander_polynomial` (+`_vector`), `determinant`, `signature` | Seifert form → `canonical_alexander`; det = \|Δ(−1)\|; signature of V + Vᵀ |
| `jones_polynomial` (+`_vector`) | Kauffman bracket (`jones.py`) |
| `khovanov_unreduced_*` (mod2, rational) | native Khovanov F2 + ℚ |
| `rasmussen_invariant` | Lee deformation → s |
| `hfk_polynomial` (+`_vector`), `three_genus` | grid knot Floer, grid number ≤ 10; genus = top Alexander grading |
| `ozsvath_szabo_tau_invariant` | τ from the Alexander filtration, grid number ≤ 10 |
| `seifert_matrix` | `seifert_matrix_from_braid` |

`grid_notation`, `braid_notation`, `pd_notation`, etc. are read, not computed — see Tier I.

---

## Tier 1 — small computations on existing machinery (near)

| Column(s) | Method |
|---|---|
| `conway_polynomial` (+`_vector`) | Alexander in the Conway variable (z = t^½ − t^−½) |
| `homfly_polynomial` (+`_vector`) | HOMFLY-PT skein recursion (crossing resolution / Hecke) |
| `kauffman_polynomial` (+`_vector`) | Dubrovnik/F two-variable skein |
| `q_polynomial` | Brandt–Lickorish–Millett–Ho (specialization of Kauffman) |
| `signature_function` | Levine–Tristram from V + V^T (ω on the unit circle) |
| `arf_invariant` | Δ(t) mod 8 / Seifert form Arf |
| `algebraic_concordance_order` | Witt class of the Seifert form (Levine's algebraic concordance group) |
| `epsilon`, `nu` | knot Floer ε from the CFK structure; ν from (τ, ε) — `homology-engine.md` §7 Phase 7 |
| `fibered` | HFK detects fibredness (Ni): top Alexander grading rank 1 |
| `l_space` | L-space knot ⇔ CFK is a staircase (HFK rank ≤ 1 in each Alexander grading; not thin in general) |
| `braid_length` | from `braid_notation` (word length) |

---

## Tier 2 — engines or substantial algebra (build)

| Column(s) | Method / engine |
|---|---|
| `khovanov_unreduced_integral_*` | integral Khovanov (Z coefficients incl. torsion) — extends the F2/ℚ engine |
| `khovanov_reduced_*` (Z, Q, mod2) | reduced Khovanov complex — basepoint-constrained front end, back end untouched; `homology-engine.md` §7 Phase 9. Cheapest Tier 2 entry: three computed oracles already provisioned |
| `khovanov_odd_*` (Z, Q, mod2) | odd Khovanov homology (Ozsváth–Rasmussen–Szabó) — a sibling engine |
| `width` | homological width of (integral) Khovanov |
| `turaev_genus` | Turaev surface from the diagram (combinatorial) |
| `hfk_polynomial` at scale | Ozsváth–Szabó bordered-algebra Floer engine (arXiv:1707.00597, arXiv:1912.01657; the algorithm of Szabó's HFK calculator) — does not pay n!, so it reaches past the grid wall. A class (A) route (decision 0011): it must agree with the grid reference. Not the same lever as `homology-engine.md` §7 Phase 8, which changes the grid complex's representation, not the algorithm |
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

KnotInfo records values, ranges, or "unknown" here. No program computes the exact value for a
general knot. Programs do produce certified upper bounds and witnesses at scale: SnapPy 3.3
searches for ribbon disks by band moves, and Dunfield–Gong (arXiv:2512.21825) report 1,633,786
ribbon knots among 352.2 million knots to 19 crossings. Scope here: compute the bound our
invariants give
(e.g. \|s\|/2 ≤ g₄, σ bounds), show it beside the tabulated value, and never present a bound as
a resolved value.

`smooth_four_genus`, `topological_four_genus`, `smooth_4d_crosscap_number`,
`topological_4d_crosscap_number`, `smooth_concordance_genus`, `topological_concordance_genus`,
`smooth_concordance_crosscap_number`, `topological_concordance_crosscap_number`,
`smooth_concordance_order`, `topological_concordance_order`, `double_slice_genus`,
`fd_clasp_number`, `td_clasp_number`, `ribbon`, `ribbon_number`, `unknotting_number`,
`unknotting_number_algebraic`, `bridge_index`, `super_bridge_index`, `tunnel_number`,
`morse_novikov_number`, `crossing_number` (minimal, in general), `cosmetic_crossing`.

`unknotting_number`: the exact value is the bounded problem. Certifying u(K) = 1 and finding an
unknotting crossing in a diagram is a separate, buildable engine (Reidemeister-graph search for
existence; Montesinos trick / branched-cover d-invariant + Nakanishi index for the u ≥ 2
obstruction). It is a prerequisite of the Piccirillo program in `four-manifold-objectives.md`.

---

## Tier 4 — hyperbolic and geometric (research)

A SnapPea-class engine: ideal triangulation, gluing equations solved to high precision (Newton),
then derived geometry. Whether to build it or leave these columns oracle-only is undecided.

`volume`, `volume_imaginary_part`, `chern_simons_invariant`, `maximum_cusp_volume`,
`longitude_translation`, `meridian_translation`, `longitude_length`, `meridian_length`,
`other_short_geodesics`, `full_symmetry_group`, `symmetry_type`, `geometric_type`,
`boundary_slopes`, `a_polynomial` (SL₂(ℂ) character variety — algebraic, equally heavy).

Also geometric or numerical: `polygon_index` (stick number), `mosaic_tile_number`, `ropelength`.

---

## Tier I — input and naming (not invariant engines)

Diagram encodings and names. Converting between them computes identifiers, not invariants.

`name`, `category`, `dt_notation`, `gauss_notation`, `enhanced_gauss_notation`, `pd_notation`,
`braid_notation`, `conway_notation`, `two_bridge_notation`, `montesinos_notation`,
`pretzel_notation`, `grid_notation`, `*_braid_notation`, `positive_pd_notation`, `dt_name`,
`classical_conway_name`, `tetrahedral_census_name`, `alternating`, `small_large`.

---

## Order

Tier 1 follows engine Phases 6 and 7: it reuses the Seifert form, Alexander and HFK, and closes
most of the classical and Seifert-form concordance columns. Tier 2's Khovanov extensions and the
bordered-algebra Floer engine are the large homology builds. Tier 3 is bounds only. Tier 4 is its
own decision.
