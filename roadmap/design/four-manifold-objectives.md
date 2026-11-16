# Four-manifold objectives

The 4-dimensional constructions and invariants the knot engines feed. `coverage-map.md` lists
the knot-side KnotInfo columns; this document lists what is 4-dimensional and has no column in a
knot table. Tiers as in `coverage-map.md`: have / near / build / research.

## Validation

There is no KnotInfo-scale oracle for 4-manifolds. An engine here is validated by three checks
together:

1. **Reduction** — when the 4D construction specializes to a knot, it reproduces the knot
   invariant already validated (a sliceness obstruction vanishes on known slice knots and
   matches the s/τ/σ bounds on known 4-genus).
2. **Internal consistency** — the structural laws hold: adjunction inequalities, surgery exact
   triangles, blow-up and handle-slide invariance, functoriality of cobordism maps.
3. **Literature examples** — the cases computed by hand or by other groups (exotic pairs,
   specific d-invariants, the E₈ manifold, computed skein-lasagna examples).

---

## Layer A — existing invariants presented as 4D obstructions (near; mostly presentation)

s, τ, σ, the Seifert form and HFK are computed. This layer states what they obstruct.

- **Slice genus and sliceness obstructions.** The bounds \|s\|/2 ≤ g₄, \|τ\| ≤ g₄ and the
  signature bounds, reported as obstructions to a knot bounding a smooth disk or low-genus
  surface in B⁴, with the bound and the obstruction stated. Validates by reduction (vanishes on
  known slice knots; matches tabulated g₄).
- **The smooth–topological gap.** Smooth obstructions (s, τ, Υ) and topological ones
  (Levine–Tristram signature, algebraic concordance, Casson–Gordon) reported side by side, with
  the places they diverge marked: a smooth bound exceeding what the topological data forbids is
  the sign of exotic surface behavior.
- **Adjunction framing.** The genus bounds presented as the adjunction inequality specialized to
  the 4-ball and knot traces.

---

## Layer B — knot → 4-manifold bridges (build)

- **Knot traces X_n(K).** The 4-manifold from one n-framed 2-handle on B⁴ along K. Build the
  handle description; read off the intersection form (⟨n⟩) and the boundary (n-surgery S³_n(K)).
- **Kirby calculus.** Framed links as 4-manifold handle decompositions; blow-up/down and handle
  slides as moves; the linking matrix as the intersection form. Validated by move-invariance of
  everything computed from it.
- **Intersection-form algebra.** Unimodular symmetric bilinear forms: signature, type
  (even/odd), definiteness, diagonalization over ℤ, the E₈ form. Donaldson's theorem (a smooth
  closed definite form is diagonalizable) and Rokhlin's theorem enter as obstruction inputs.
- **Double branched covers Σ(K)** of S³ along K (a rational homology sphere): the route from a
  knot to a 3-manifold whose Floer invariants (Layer C) obstruct sliceness.

---

## Layer C — Floer up the dimensions (build → research)

- **3-manifold Heegaard Floer via surgery.** HF-hat / HF⁺ of surgeries on K from the knot Floer
  complex (the integer / rational surgery mapping-cone formula). Validates by surgery exact
  triangles and known surgery computations.
- **d-invariants (correction terms)** of surgeries and branched covers — computable
  combinatorially for plumbed / negative-definite cases (Ozsváth–Szabó plumbing formula). The
  d-invariant of Σ(K) is a sliceness obstruction (Manolescu–Owens δ).
- **Concordance homomorphisms from CFK^∞.** Once engine Phase 7 exposes the reduced filtered
  complex, Υ(t), ε, ν⁺ and the Hom ϕ_j homomorphisms are read off it.
- **Involutive Heegaard Floer and homology cobordism.** The involution on CFK, involutive
  d-invariants, and the homology cobordism group Θ³_ℤ. Research.

---

## Layer D — 4-dimensional combinatorial structures (research)

- **Trisections (Gay–Kirby).** The 4-manifold analog of Heegaard splittings: trisection
  diagrams, trisection genus, the moves. A combinatorial handle on closed 4-manifolds, with a
  small census to check against.
- **Khovanov skein lasagna modules (Morrison–Walker–Wedrich).** An invariant of a
  4-manifold-with-link that refines Khovanov homology and has been used to detect exotic
  phenomena. It extends the Khovanov engine into dimension 4.

---

## Layer E — gauge theory (out of scope)

Seiberg–Witten and Donaldson invariants are not combinatorially computable in this framework.
They enter only as theorems applied (the adjunction inequality, the diagonalization theorem,
basic-class constraints) and as literature oracles.

---

## The Piccirillo trace method (slice detection via friends)

Piccirillo's proof that the Conway knot is not slice recasts a knot question (sliceness) as a
4-manifold question (does the 0-trace embed in S⁴) and answers it by swapping the knot for a
trace sibling that Rasmussen s detects. It is assembled from pieces on this map.

- **Trace embedding lemma.** K is smoothly slice ⇔ its 0-trace X₀(K) embeds smoothly in S⁴
  (Layer B builds X_n(K)).
- **Trace siblings / Piccirillo friends.** Knots with diffeomorphic 0-traces share smooth slice
  status. They also share ν and \|ε\| (invariants of the n-trace for every n; arXiv:2608.09734)
  but not Rasmussen s. If s(K) is zero, construct a friend K′ with s(K′) ≠ 0 and transfer the
  conclusion. Given an unknotting-number-1 diagram with a marked unknotting crossing, the friend
  is algorithmic [Pic20]. A shared 0-surgery alone is weaker than a shared trace and does not
  transfer slice status: a pair with the same 0-surgery, one slice and one not, would give an
  exotic 4-sphere (Dunfield–Gong, arXiv:2512.21825, Theorem 1.12).
- **Prerequisite — unknotting machinery.** The friend construction starts from an
  unknotting-number-1 diagram with a chosen unknotting crossing. Producing that is its own
  engine with two halves:
  (i) **finding** an unknotting crossing — a search over the Reidemeister graph for a crossing
  whose change yields the unknot;
  (ii) **certifying** u(K) = 1 — existence is a witnessed search; the obstruction side (u ≥ 2)
  uses the Montesinos trick (u = 1 ⇒ the double branched cover is a half-integer surgery)
  feeding Heegaard Floer d-invariant / Donaldson obstructions on Σ(K), plus Alexander-module
  bounds (Nakanishi index).
  The exact unknotting number is open in general (`coverage-map.md` Tier 3); this needs only
  u = 1 detection and crossing-finding, reusing Layer B branched covers and Layer C
  d-invariants. Validates by reproducing the known unknotting-number-1 census and the unknotting
  crossings used in the Conway-knot and Manolescu–Piccirillo constructions.
- **RBG links [Manolescu–Piccirillo 2023].** The general construction of same-0-surgery pairs:
  a 3-component framed link encoding a 0-surgery homeomorphism, with special / n-RBG variants,
  dualizable patterns and annulus twisting as special cases. Turns "find a friend" into
  enumeration.
- **Detection (native).** Rasmussen s is the first detector; the Steenrod-refined s^{Sq}
  invariants (Lipshitz–Sarkar) and skein-lasagna-refined obstructions catch cases where plain s
  vanishes. These are the Khovanov / Lee engines (Tier 0 s; Tier 2 refinements).
- **Exotic-4-manifold candidates.** 0-surgery homeomorphisms glue into candidate exotic definite
  4-manifolds (#nℂP²); a friend H-slice in one copy but s-obstructed in another would be an
  exotic pair, bearing on the smooth 4-dimensional Poincaré conjecture. Manolescu–Piccirillo's
  five candidate knots are not slice (Nakamura, arXiv:2203.14270). Later candidates: Nakamura,
  arXiv:2503.20684; Dunfield–Gong, Theorem 1.12 (16n68278, 17nh0010647, 18nh00098198).

**What exists and what Tetradrome adds.** The friend construction and the sliceness search have
been run at scale: Dunfield–Gong (arXiv:2512.21825) surveyed 352.2 million knots to 19
crossings, leaving 11,383 smoothly unknown, found 500,000 0-friend pairs, showed 25 knots
non-slice by the Piccirillo method, and merged the key code into SnapPy 3.3. That work is built
on SnapPy, KnotJob, Szabó's HFK calculator and SageMath. Tetradrome adds a native,
single-schema, audited implementation, not the search: the trace / friend / RBG constructions
written natively, fed by its own s detectors, with SnapPy as an optional oracle (volume /
HOMFLY-PT to certify that a constructed friend is a distinct knot). Validation: reproduce the
Conway-knot result (s of its friend ≠ 0) and the Manolescu–Piccirillo example pairs; check
against the published Dunfield–Gong friend pairs; verify trace diffeomorphism from the RBG
certificate; agree with SnapPy on distinguishing volumes where used.

---

## Order

Layer A rides with engine Phase 7, which produces the CFK the concordance invariants are read
from. Layer B is the first 4D build and is self-contained (Kirby calculus and intersection forms
validate by internal consistency). Layers C and D follow. Layer E is a boundary. The Piccirillo
program consumes Layer A's s and its refinements and Layer B's traces and RBG / Kirby machinery.
Nothing here is done until it reduces correctly to the knot invariants already validated and
obeys the structural laws it must.
