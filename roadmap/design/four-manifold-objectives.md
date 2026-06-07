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

## Sequencing

Layer A is next-to-free and is the project's whole thesis made visible — it should ride along with
Phase 7 (which produces the CFK the concordance invariants read off). Layer B is the first
substantial 4D build and is self-contained (Kirby calculus + intersection forms validate purely by
internal consistency, no oracle needed). Layers C–D are the long horizon. Layer E is a boundary, not
a backlog. The discipline from the homology engine carries over unchanged: nothing is "done" until
it reduces correctly to the knot invariants we already validate and obeys the structural laws it must.
