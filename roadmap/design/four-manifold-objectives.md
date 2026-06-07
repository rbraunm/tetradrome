# Four-manifold objectives: the actual north star

Tetradrome is a workbench for **smooth 4-dimensional topology**. Knots are not the subject — they
are the richest *computable, checkable* feeder into it: knot and concordance invariants are
obstructions to embeddings, slice disks, and surfaces in 4-manifolds, and they are the part of the
4D story that has a dense oracle (KnotInfo) to validate against. `coverage-map.md` enumerates that
feeder. **This document is the thing the feeder feeds.** Its job is to make sure the genuinely
4-dimensional tools never get skipped just because they don't appear as a column in a knot table.

**Validation is different up here.** There is no KnotInfo-scale oracle for 4-manifolds. So an
engine in this document is validated by three weaker-but-real checks, in combination:
1. **Reduction** — when the 4D construction specializes to a knot, it must reproduce the knot
   invariant we already validate (e.g. a sliceness obstruction must vanish on known slice knots,
   match s/τ/σ bounds on known 4-genus).
2. **Internal consistency** — the structural laws must hold: adjunction inequalities, surgery exact
   triangles, blow-up/handle-slide invariance, functoriality of cobordism maps.
3. **Literature examples** — the handful of cases computed by hand or by other groups (exotic pairs,
   specific d-invariants, the E₈ manifold, computed skein-lasagna examples).

Honest tiering as in `coverage-map.md`: have / near / build / research. Most of this is long-horizon;
the point of writing it now is the *map*, and the recognition that **Layer A is nearly free and is
the whole reason the project exists**.

---

## Layer A — reframe what we already compute as 4D obstructions (near; mostly presentation)

We already author s, τ, σ, the Seifert form, HFK; what we *don't* yet do is present them as what
they are to a 4D topologist. This layer is cheap and high-value.

- **Slice genus / sliceness obstructions, first-class.** The slice–Bennequin and Floer/Khovanov
  bounds — \|s\|/2 ≤ g₄, \|τ\| ≤ g₄, signature bounds — surfaced as obstructions to a knot bounding a
  smooth disk or low-genus surface in B⁴, with the bound and the obstruction stated, not just a
  number. Validates by reduction (vanishes on known slice knots; matches tabulated g₄).
- **The smooth–topological gap as the headline, not two columns.** Report smooth obstructions
  (s, τ, Υ) *and* topological ones (Levine–Tristram signature, algebraic concordance, Casson–Gordon)
  side by side and surface where they diverge — a smooth bound exceeding what the topological data
  forbids is the fingerprint of exotic surface behavior. This is the project's distinctive lens.
- **Adjunction framing.** Present the genus bounds as the adjunction inequality specialized to the
  4-ball / knot traces, so the same machinery reads as "genus of an embedded surface in this class."

---

## Layer B — knot → 4-manifold bridges (build)

The combinatorial constructions that turn knot data into honest 4-manifolds.

- **Knot traces X_n(K).** The 4-manifold from a single n-framed 2-handle on B⁴ along K. Build the
  handle description, read off the intersection form (⟨n⟩) and boundary (n-surgery S³_n(K)). Trace
  embeddings are a live route to exotica; the trace is the cleanest knot→4-manifold bridge.
- **Kirby calculus.** Framed links as 4-manifold handle decompositions; blow-up/down and handle
  slides as moves; the linking matrix as the intersection form. The native combinatorial language of
  smooth 4-manifolds — and validated by move-invariance of everything computed from it.
- **Intersection-form algebra.** Unimodular symmetric bilinear forms: signature, type (even/odd),
  definiteness, diagonalization over ℤ, the E₈ form. Donaldson's theorem (a smooth closed definite
  form is diagonalizable) and Rokhlin become *obstruction inputs* the tool can apply.
- **Double branched covers Σ(K)** of S³ along K (a rational homology sphere), as the gateway from a
  knot to a 3-manifold whose Floer-theoretic invariants (below) obstruct sliceness.

---

## Layer C — Floer up the dimensions (build → research)

Our HFK engine is a 3-/4-manifold engine waiting to be unfolded.

- **3-manifold Heegaard Floer via surgery.** HF-hat/HF⁺ of surgeries on K from the knot Floer
  complex (the integer/rational surgery / mapping-cone formula). Connects our CFK to 3-manifold
  invariants, and through cobordism maps, to 4D. Validates by surgery exact triangles + known
  surgery computations.
- **d-invariants (correction terms).** Of surgeries and branched covers — computable combinatorially
  for plumbed / negative-definite cases (Ozsváth–Szabó plumbing formula). The d-invariant of Σ(K) is
  a sharp sliceness obstruction (Manolescu–Owens δ).
- **Concordance homomorphisms from CFK^∞.** Once Phase 7 exposes the reduced filtered complex,
  Υ(t), ε, ν⁺, and the Hom ϕ_j homomorphisms are read off it — actual maps to the smooth concordance
  group, not just numbers. This is where "knot invariant" becomes "4D concordance structure."
- **Involutive Heegaard Floer & homology cobordism.** The involution on CFK, involutive d-invariants,
  and the homology cobordism group Θ³_ℤ. Research-leaning, but the natural endpoint of the Floer line.

---

## Layer D — genuinely 4D combinatorial structures (research / exploratory)

Objects that are 4-dimensional from the start, with no knot in sight.

- **Trisections (Gay–Kirby).** The 4-manifold analog of Heegaard splittings: trisection diagrams,
  trisection genus, the moves. A combinatorial handle on *closed* 4-manifolds; a clean exploratory
  target with its own small census to check against.
- **Khovanov skein lasagna modules.** The genuinely 4-dimensional refinement of Khovanov homology
  (Morrison–Walker–Wedrich) — an invariant of a 4-manifold-with-link, recently used to *detect exotic
  phenomena*. It is the 4D extension of the Khovanov engine we already author, and the most direct
  way our existing homology machinery reaches into dimension 4. Hot, hard, and squarely on-mission.

---

## Layer E — gauge theory (out of combinatorial scope)

Seiberg–Witten and Donaldson invariants, the analytic backbone of smooth 4-manifold topology, are
not combinatorially computable in this framework. They enter only as *theorems we apply* (the
adjunction inequality, the diagonalization theorem, basic-class constraints) and as literature
oracles, not as engines we author. Stated so the boundary is explicit.

---

## Flagship program — the Piccirillo trace method (slice detection via friends)

Piccirillo's proof that the Conway knot is not slice is the whole Tetradrome thesis in one
result: a knot question (sliceness) recast as a 4-manifold question (does the 0-trace embed in
S⁴), solved by swapping the knot for a *trace sibling* that the same smooth invariant — Rasmussen s
— actually detects. It belongs here as a flagship, not a footnote, and it is mostly built from
pieces already on this map.

- **Trace embedding lemma.** K is smoothly slice ⇔ its 0-trace X₀(K) embeds smoothly in S⁴ — the
  bridge that turns every slice question into a 4-manifold-embedding question (Layer B builds X_n(K)).
- **Trace siblings / Piccirillo friends.** Knots with diffeomorphic traces (for the 0-trace, equal
  0-surgeries) share smooth slice status but *not* the values of the slice invariants. If s(K) is
  uselessly zero, construct a friend K′ with s(K′) ≠ 0 and transfer the conclusion. Given an
  unknotting-number-1 diagram with a marked unknotting crossing, the friend is algorithmic [Pic20].
- **Prerequisite — unknotting machinery (explicit, not assumed).** The friend construction does not
  start from a knot; it starts from *an unknotting-number-1 diagram with a chosen unknotting crossing*,
  and producing that is its own engine with two distinct halves:
  (i) **finding** an unknotting crossing — a diagram-combinatorics search (walk the Reidemeister graph
  for a crossing whose change yields the unknot), the operational input the friend needs;
  (ii) **certifying** u(K) = 1 — the existence side is a witnessed search, but the *obstruction* side
  (proving u ≥ 2) is real theory: the Montesinos trick (u = 1 ⇒ the double branched cover is a
  half-integer surgery) feeding Heegaard Floer d-invariant / Donaldson obstructions on Σ(K), plus
  Alexander-module bounds (Nakanishi index). This is distinct from the *exact* unknotting number,
  which is open in general (it lives in `coverage-map.md` Tier 3 as a bound). The flagship needs only
  u = 1 detection + crossing-finding, not the general value — but it needs them as a built, validated
  engine, reusing Layer B branched covers and Layer C d-invariants for the obstruction half. Validates
  by reproducing the known unknotting-number-1 census and the unknotting crossings used in the
  Conway-knot and Manolescu–Piccirillo constructions.
- **RBG links [Manolescu–Piccirillo 2023].** The fully general machine for same-0-surgery pairs: a
  3-component framed link encoding a 0-surgery homeomorphism, with special / n-RBG variants,
  dualizable patterns, and annulus twisting as special cases. Turns "find a friend" into enumeration.
- **Detection layer (ours, natively).** The friends are inert without an invariant that sees them.
  Rasmussen s is the workhorse; the Steenrod-refined s^{Sq} invariants (Lipshitz–Sarkar) and
  skein-lasagna-refined obstructions catch cases where plain s vanishes. This is exactly the
  Khovanov/Lee machinery we author (Tier 0 s; Tier 2 refinements) — the Piccirillo pipeline is a
  *consumer* of our detection engines, which is why it sits so naturally here.
- **Exotic-4-manifold candidates (the apex).** 0-surgery homeomorphisms glue into candidate exotic
  definite 4-manifolds (#nℂP²); a friend H-slice in one copy but s-obstructed in another would be an
  exotic pair, bearing on the smooth 4-dimensional Poincaré conjecture. The frontier the bridge aims at.

**Why this is on the roadmap, not "done."** The construction *is* implemented in research code — the
friend algorithm has been coded and run over census knots, and recent exotic-trace searches automate
it — but those implementations are paper-specific, unmaintained, and built **on top of SnapPy**
(triangulating the RBG-link exterior, filling surgery slopes, distinguishing knots by volume), with
s pulled from separate tools. There is no from-first-principles, single-schema implementation. That
is Tetradrome's opening: author the trace / friend / RBG constructions natively, feed them our own s
and refined-s detectors, and keep SnapPy as an *optional oracle* (volume / HOMFLYPT to certify a
constructed friend is genuinely a distinct knot) — never the engine. Validation: reproduce the
Conway-knot result (s of its friend ≠ 0) and the Manolescu–Piccirillo example pairs; verify trace
diffeomorphism from the RBG certificate; agree with SnapPy on distinguishing volumes where used.

---

## Sequencing

Layer A is next-to-free and is the project's whole thesis made visible — it should ride along with
Phase 7 (which produces the CFK the concordance invariants read off). Layer B is the first
substantial 4D build and is self-contained (Kirby calculus + intersection forms validate purely by
internal consistency, no oracle needed). Layers C–D are the long horizon. Layer E is a boundary, not
a backlog. The discipline from the homology engine carries over unchanged: nothing is "done" until
it reduces correctly to the knot invariants we already validate and obeys the structural laws it must. The **Piccirillo program** is the natural integrating target: it consumes Layer A's s (and
its refinements), Layer B's traces and RBG/Kirby machinery, and points straight at the exotica
frontier — so as those layers land, the flagship is what assembles them into a result no single
existing tool produces from first principles.
