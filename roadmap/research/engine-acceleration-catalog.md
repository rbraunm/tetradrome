# Research: engine acceleration catalog

Candidate accelerations for the brute reference paths, gathered so the research survives
between sessions. Governed by [ADR 0011](../decisions/0011-harden-the-brute-reference.md):
every entry is tagged with its **(B)/(A)/(C)** class, the core may use only the exact
**(B)** ones, and the realistic ceiling is constant-factor on an irreducible floor. This
note is a menu of *candidates*, not committed work; it feeds milestone **M8** (native
exact algebra / performance) and engine **Phases 5–6**, and is cross-linked from
[`../design/homology-engine.md`](../design/homology-engine.md) §6.

These are not assertions that any technique is implemented or measured here. Where a
claim about cost or behaviour came from reading our own code it is marked *(ours)*;
everything else is literature, cited by author/year so the source can be pulled.

## The shape every engine shares

Each engine is a **brute reference** = a combinatorial explosion, split into two phases:

1. **Generation** — build the chain complex (enumerate generators, compute gradings,
   build the differential).
2. **Reduction** — take homology (rank of the boundary maps over the field, per grading).

A technique attaches to one or both phases. The explosion is the engine's floor:
exact work pushes the reachable size out a few units and lowers the constant, never the
growth (0011). The three classes, recapped:

- **(B)** answer-preserving transform of the *same* complex — permitted in the core,
  toggleable, verified `== reference`.
- **(A)** alternative algorithm that changes the route — separate validated fast path,
  must agree with its own brute reference, never the core.
- **(C)** input reduction that changes *which* complex is built — permitted, often a big
  win, but never the floor (the reference must still run on the input as given).

## Engine × explosion × technique

| Engine | Explosion | Dominant phase | Leading (B) candidates | (A) | (C) |
|---|---|---|---|---|---|
| Khovanov / Lee | `2^c` (c = crossings) | reduction | delooping + local Gaussian elimination (Bar-Natan scanning); discrete-Morse cancellation; clearing/twist; implicit reduction; block-Lanczos rank | — | diagram simplification |
| Grid Floer | `n!` | generation *(ours: ~75% now)* | W-precompute; differential `O(n³)→O(n²)`; Gray-code incremental generation; bigrading split *(in code)*; stream/spill by Alexander grading; discrete-Morse cancel-as-you-build | bordered / tangle Floer | grid simplification (Cromwell / destabilization) |
| Shared F2 reducer | matrix size + fill-in | reduction | clearing; implicit reduction; M4RI (four Russians); block Lanczos / Wiedemann (iterative, fill-in-free) | — | — |

## Per-engine notes

### Khovanov / Lee — `2^c`
Done and validated (M7/M9), so these are headroom for high-crossing inputs, not blockers.
The historically decisive lever here is **delooping + local Gaussian elimination**
(Bar-Natan's divide-and-conquer / "Gaussian elimination made abstract"): it shrinks the
cube so much that the residual field linear algebra is small. ADR 0007 already names it;
it is exact (a chain-homotopy equivalence), so it leads. **Discrete-Morse cancellation**
is the same idea framed via an acyclic matching, and is an active area for knot homology
right now. **Clearing/twist** and **implicit reduction** then cut the residual reduction;
**block-Lanczos** is the GPU-friendly way to take the rank of whatever dense block remains.

### Grid Floer — `n!`
The current bottleneck is **generation** *(ours: ~75% of gen+reduce at the sizes swept;
the ratio is size-dependent and likely inverts toward reduction as n grows)*. Within
generation, the differential dominates: the empty-rectangle scan is `O(n³)` per state
*(ours)* because of an `O(n)` emptiness check inside the `O(n²)` row-pair loop. The
levers, in rough leverage order:

- **(C) Grid simplification first.** Reducing n by even one or two via Cromwell
  moves / destabilization is a *factorial* win — larger than any (B) gain — because the
  complex is `n!`. It is what the field does (Droz; Culler's gridlink "simplify"). It
  depends on a smaller diagram existing, so per 0011 it is never the floor; the brute
  path must still run on the diagram as given.
- **(B) Differential `O(n³)→O(n²)`.** Replace the rebuilt-set emptiness scan with an
  order-structure / prefix-count test (the state is a permutation — one point per
  row/column), hitting the dominant generation cost.
- **(B) W-precompute.** The Alexander grading is an additive assignment sum
  `A(σ) = affine(Σ_i W[i][σ(i)])` *(ours; the `_sw(gen,gen)` self-terms cancel in
  `M_O − M_X`)*, so precomputing the winding matrix `W` once makes each grading `O(n)`
  instead of the current `O(n²)`. Maslov keeps its inversion self-term (`~n²`, or
  incremental via a Fenwick tree).
- **(B) Gray-code incremental generation.** Enumerate permutations so consecutive states
  differ by one transposition (Steinhaus-Johnson-Trotter / plain changes; Knuth TAOCP
  4A); grading and differential then update locally instead of from scratch. Pairs with
  the W-precompute incremental sum.
- **(B) Stream / spill by Alexander grading** for memory. The complex is block-diagonal
  by Alexander grading *(in code: `grid_complexes` already returns `{A: complex}`)*;
  generating/reducing/freeing one block at a time, or external-bucket-sorting blocks to
  disk, turns the `n!` memory ceiling into "largest single block." Converts memory-bound
  to time-bound — the trade the owner wants for reaching larger n.
- **(B) Discrete-Morse cancel-as-you-build** bridges both phases: cancel matched
  generators while generating so the full `n!` complex is never materialized — attacks
  the memory ceiling and the reduction cost together.
- **(A) Bordered / tangle Floer.** The modern fast route — cut into tangles, tensor
  small invariants, scales with crossing number / braid width, never pays `n!`. But it
  is far harder to implement and audit than the transparent grid complex, so it is an
  adjunct that must agree with the grid reference, not a replacement for it.

Caveats worth keeping: a **GPU generation kernel** is a poor first move — the differential
is branchy and early-exiting (warp divergence) with variable-length per-state output, and
generation may be output-bound rather than compute-bound; if a GPU generation kernel is
ever built it should be *fused* (unrank + grade + differentiate + feed the on-device
reducer) so the block never leaves the card. The **GPU reduction** target is block-Lanczos
over F2 (fill-in-free, SpMM-shaped), consistent with the deferral in
[0010](../decisions/0010-defer-gpu-kernel.md).

### Shared F2 reducer
The reducer every engine calls. **Clearing** (use `∂²=0` to skip vanishing columns) and
**implicit reduction** (recompute columns instead of storing the matrix) are exact and
cut both time and memory. For the rank step at scale, **M4RI** (method of the four
Russians) and the **block Lanczos / Wiedemann** iterative methods avoid fill-in entirely
by working through sparse matrix×block products. ADR 0010 defers the on-device GPU kernel;
when it is built, block-Lanczos-over-F2 SpMM is the structural target it should aim at.

## Technique reference (provenance)

| Technique | Class | Phase | Source |
|---|---|---|---|
| Delooping + local Gaussian elimination | B | reduction | Bar-Natan, *Fast Khovanov homology computations* (2007) — "Gaussian elimination made abstract" |
| Algebraic discrete Morse theory | B | both | Sköldberg (2006), Kozlov (2005); for knot homology: Kelomäki (2023, arXiv:2306.11186), torus-link DMT (2025, arXiv:2507.15060), Khovanov complexity (2026, arXiv:2601.02119) |
| Connection matrices / Conley complex | B | reduction | algebraic Morse minimal reduction respecting a poset grading (2025, arXiv:2503.09301) |
| Clearing / twist | B | reduction | Chen & Kerber (2011); as used in Ripser |
| Implicit matrix reduction | B | reduction | Bauer, *Ripser* (J. Appl. Comput. Topol., 2021) |
| Persistent cohomology (skip more columns) | B | reduction | de Silva, Morozov, Vejdemo-Johansson (2011); over a field, ranks coincide with homology |
| Apparent / emergent pairs | B | reduction | Bauer, *Ripser* (2021) — cheap discrete-Morse matchings |
| Block Lanczos | B | reduction | Montgomery (1995) — dependencies over GF(2) |
| Block Wiedemann | B | reduction | Coppersmith (1994), *Math. Comp.* 62:333–350; rank variant: Dumas & Giorgi |
| Block Lanczos/Wiedemann on GPU | B | reduction | Schmidt et al. (2013) CUDA SpMV for NFS over GF(2), ~4–8× over multicore; 4-Russians GF(2) matmul on GPU |
| M4RI (four Russians) | B | reduction | dense GF(2) elimination; noted as a target in ADR 0010 |
| W-precompute (additive Alexander sum) | B | generation | *(ours)* — derived from `gradings.py`; the winding-number form of the grid Alexander grading |
| Differential `O(n³)→O(n²)` | B | generation | *(ours)* — order-structure emptiness test over a permutation, vs the current rebuilt-set scan in `differential.py` |
| Gray-code permutation enumeration | B | generation | Steinhaus-Johnson-Trotter / plain changes; Knuth, TAOCP Vol. 4A |
| Bigrading (Alexander) decomposition | B | both | grid homology splits by Alexander grading; *(in code: `homology.py`)* |
| Stream / external-bucket-sort by grading | B | both | standard external sort applied per Alexander block; *(synthesis)* |
| Grid simplification | C | input | Droz, *Effective computation of knot Floer homology* (arXiv:0803.2379, 2008); Cromwell moves; Culler's gridlink |
| Bordered / tangle Floer | A | route | Lipshitz-Ozsváth-Thurston; flagged as the faster modern route in GridPyM (2024, arXiv:2210.07399) |

## Pointers

- **Governing decision:** [0011](../decisions/0011-harden-the-brute-reference.md)
  (Locked) — investment in the brute path, the irreducible floor, the (B)/(A)/(C) rule,
  the binding-engine frontier.
- **Related:** [0007](../decisions/0007-faithful-raw-path-no-heuristics.md) (exact-only,
  no heuristics), [0008](../decisions/0008-memory-prediction-gate.md) (memory gate),
  [0010](../decisions/0010-defer-gpu-kernel.md) (GPU deferral),
  [0009](../decisions/0009-scope-smooth-4d-toolset.md) (4D scope).
- **Design:** [`../design/homology-engine.md`](../design/homology-engine.md) §3
  (tier model), §6 (honest calibration), §7 (implementation path).
- **Milestone:** M8 (native exact algebra / performance) in
  [`../milestones.md`](../milestones.md).
