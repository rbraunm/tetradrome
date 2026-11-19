# Homology engines

The engines behind the homological invariants: Khovanov, Lee, Rasmussen *s*, and knot Floer.

How calculations run is in [scheduler.md](scheduler.md). The project's rules are in
`SPEC.md` section 2.

Code comments cite this document by section and phase number, so those numbers stay fixed.

## 1. Scope

| Invariant | Engine | Back end | State |
|---|---|---|---|
| determinant, signature, Alexander | Seifert form (`invariants/seifert.py`) | its own dense integer algebra | done |
| Jones polynomial | Kauffman bracket over the resolution cube | none | done |
| Khovanov homology over F2 and ℚ | Khovanov complex on the cube | shared reducer: F2 lane, exact ℚ lane | done |
| Rasmussen *s* | Lee deformation of the signed cube | exact ℚ lane | done |
| HFK-hat, three-genus, τ | grid homology | shared F2 reducer | done to grid number 10 |
| ε, ν, Υ | reduced CFK from the grid complex | shared F2 reducer | planned (Phase 7) |

Every invariant marked done is in `compute()` and runs as scheduler jobs.

## 2. Layering

```text
FRONT ENDS (one per theory)
  Seifert form        Khovanov / Lee             knot Floer
  (own algebra)       (cube of resolutions)      (grid rectangles)
                             |                         |
                             v                         v
                   graded chain complex over F2 or Q
                             |
                             v
                SHARED BACK END (knows no theory)
          graded sparse linear algebra -> homology ranks
```

- **Front ends** build a graded chain complex from a diagram. They are different mathematics
  and share only the cube skeleton. Khovanov has 2^c cube vertices for c crossings. Grid Floer
  has n! generators, the permutations on an n×n grid, and its differential counts empty
  rectangles (Manolescu–Ozsváth–Sarkar, arXiv:math/0607691;
  Manolescu–Ozsváth–Szabó–Thurston, arXiv:math/0610559).
- **The back end** (`algebra/`) takes a graded complex and returns homology dimensions from
  ranks. Every reduction tier lives here, so every front end gets it.
- **The Seifert-form engine** is outside this: a 2g×2g dense integer matrix.
- **Not built:** the Ozsváth–Szabó bordered-algebra algorithm (arXiv:1707.00597,
  arXiv:1912.01657), which Szabó's HFK calculator implements. It does not pay n!. It is a
  class (A) route (section 6): a separate fast path that must agree with the grid reference.

```text
src/tetradrome/
  invariants/
    seifert.py            Seifert-form engine: determinant, signature, Alexander
    jones.py              Kauffman bracket -> Jones
    compute.py            dispatch, scheduling, validation, provenance
  engines/
    cube.py               resolution cube: states, smoothings, circles
    khovanov/             gradings, differential (F2 and signed Q), homology, lee.py, rasmussen.py
    floer/                grid.py, gradings.py, differential.py, generation.py, homology.py
                          (HFK-hat, genus), tau.py, scheduling.py (job graphs), roster.py,
                          the compiled block kernels (*_jit.py) and prewarm.py
  algebra/
    complex.py            F2 graded complex (CSC columns)
    reduce_reference.py   set-based F2 reducer: rank, kernel, homology -- the reference
    rational_complex.py   Q graded complex
    rational_reduce.py    exact Q reducer (Fraction)
    reduce_gaussian.py    Gaussian cancellation over F2 and Q
    reduce_f2_packed.py   bit-packed F2 reducers
    reduce_f2_jit.py      Numba packed reducer
    multimodular.py       rank over Q from ranks mod primes
    memory.py             size and cost prediction
    gpu.py                GPU detection and VRAM budget
    tiers.py              reducer registry
  scheduler/              job graph, placement, executor
```

Lee and *s* live under `engines/khovanov/` because they reuse the signed cube and its
generators and change only the Frobenius maps.

The scheduled Floer path grades states and builds differentials with block kernels
(`gradings_jit.py`, `differential_jit.py`). They need numpy, are compiled by Numba when it is
installed, and are tested against the pure-Python definitions in `gradings.py` and
`differential.py`. Without numpy the scheduled path raises; the serial oracles of section 4
need neither library.

## 3. Reduction tiers

Every tier returns the identical answer as the reference. A tier that is asked for and is not
available raises.

| Tier (`algebra/tiers.py`) | What it is | Needs |
|---|---|---|
| `reference` | set-based column reduction | nothing |
| `bitint` | each column a Python int used as a bit-vector; the default | nothing |
| `jit` | packed uint64 reducer, compiled by Numba | numpy, numba |
| `packed-cpu` | uint64 word arrays | numpy |
| `packed-gpu` | dense vectorized kernel on a CUDA device | cupy, a GPU |

`reduce_f2_sparse.py` is the reference's reduction on sorted index arrays, compiled by Numba:
a rank and the filtered kernel search τ needs, for the two maps of τ's complex, which span every
generator of their Maslov degrees and are too sparse and too wide for the packed tiers. τ's jobs
use it where Numba is installed and the reference otherwise.

- **F2.** `GradedComplex` stores columns in CSC form; the reducers pack them to bits.
  `packed-cpu` and `packed-gpu` run the same code with a different array module.
- **ℚ.** `compute()` uses exact `Fraction` reduction (`rational_reduce.py`).
- **`multimodular.py`** takes the rank over ℚ as the largest rank mod three large primes. A
  bad prime can only lower a rank, so the result is the rank unless every prime is bad. That
  is a probabilistic rank, which section 4 bans from the core, so no calculation uses it.
  It is tested equal to the exact reducer and kept as the starting point for an exact
  multimodular reducer.
- **GPU.** `gpu.py` detects the device and derives the VRAM budget from the card. The
  threshold below which the CPU wins is an explicit knob, never a hidden default
  (`memory.route_backend`, used by `scripts/bench_reducers.py`). The dense kernel is correct
  and slower than `bitint` at every size measured (section 6). A kernel that stays on the
  device (bit-packed columns, no host sync per column) is built only when there is a
  calculation the CPU tiers cannot finish, and must equal the reference like every tier.
- **Choosing a tier in a run.** A reduction job declares a CPU path for the named tier, and
  under `backend="auto"` a GPU path as well when a CUDA reducer is present. The scheduler
  chooses between them ([scheduler.md](scheduler.md)).
- **Parallelism.** A graded complex splits into independent pieces by grading, and a grading's
  homology needs only the rank of each boundary map, each computable alone. The scheduler runs
  each rank as its own job.

## 4. Faithfulness

No lossy shortcut, heuristic or probabilistic rank in the core.

| Allowed: exact reductions | Banned: heuristics |
|---|---|
| Delooping and local Gaussian elimination. They are chain homotopy equivalences: same homology, gradings and derived invariants, with no assumption about the knot. | Truncating gradings; assuming thinness or alternation; probabilistic rank; "mod 2 and assume no torsion"; early-termination guesses. |

How it is enforced:

1. The raw, unreduced path is always runnable and is the reference.
2. An exact reduction is optional and verified `raw == reduced`.
3. Every reduction tier equals the reference.
4. Every scheduled form equals its serial pure-Python oracle: `grid_complexes`,
   `grading_histogram_reference`, `tau_reference`.
5. `d² = 0` is checked on every native complex.

## 5. Memory gate

A calculation that will not fit is refused with the number before it starts.

- **Exact, from the diagram, with nothing built.** Khovanov: `unreduced_size(pd)` sums
  2^(circles) over the 2^c states. Floer: the grading histogram counts generators per
  (Alexander, Maslov) over the n! states in slices, with no differential; its dimensions are
  the reducer's matrix sizes.
- **Cost model** (`algebra/memory.py`). A boundary block of `cols` columns
  into `rows` rows costs `(cols + min(cols, rows)) * ceil(rows / 64) * 8` bytes: the packed
  matrix plus its pivots. A grading's peak is the largest block over its degrees.
- **The gate.** Every job declares the RAM and VRAM each of its paths needs. The scheduler
  admits a job only when that memory is free, and before the run starts it reports any job
  that no path on this machine can serve; `run_to_result` then raises. `mem_cap_bytes` and
  `vram_cap_bytes` lower the ceilings below the detected hardware.
- **No shrinking the mathematics to fit.** An exact reduction may be used to make a
  calculation fit only as the caller's explicit choice.

## 6. Performance

Times are data, never a gate.

| Measured | Result |
|---|---|
| CPU reducers | `bitint` is the fastest in every regime tested; small dense blocks favour the CPU tiers |
| Dense GPU kernel | correct; about 14 times slower than `bitint` at 2048 × 2048, because it syncs with the host once per column |
| Grid Floer, grid number 10 (3.6 million states), 32-CPU two-socket container | HFK-hat 15 s, τ 24 s ([scheduler.md](scheduler.md)) |
| Grid Floer, grid number 11 (39.9 million states) | does not fit: over 200 GiB |

Szabó's calculator runs a different algorithm (section 2), so its times are not a
constant-factor target for the grid engine. `BENCHMARKS.md` records both.

**Speedups.** Making a reference faster by exact means is in scope: its speed sets how large
a case can be reached and how much cross-checking can be afforded. The gain is a constant
factor. A reference that builds its whole complex before reducing it still grows as n! (grid
Floer) or 2^c (Khovanov); an algorithm that never builds the whole complex can escape that
(arXiv:2601.02119, closed 3-braids in polynomial time). Every proposed speedup is classed
before it is built:

| Class | What it is | Rule |
|---|---|---|
| (B) | an answer-preserving transform of the same complex: delooping, local elimination, clearing, implicit reduction, incremental generation, iterative sparse rank, discrete-Morse cancellation | allowed in the core, in generation as well as reduction; optional, and verified equal to the reference |
| (A) | a different algorithm, such as bordered or tangle Floer | a separate fast path that must agree with the brute reference; never the reference itself |
| (C) | a smaller input: diagram simplification | allowed; the reference must still run on the input as given |

Effort goes to the engine that limits the target at hand: what can be reached in 4-dimensional
topology is the intersection of what each required invariant can reach.

Candidates, each with its class, are in
[`../research/engine-acceleration-catalog.md`](../research/engine-acceleration-catalog.md).

## 7. Phases

A phase is validated before the next builds on it. A reduction or a tier is added only after
the reference it must agree with.

| Phase | Delivers | State | Open |
|---|---|---|---|
| 0. Jones | Kauffman bracket over the cube skeleton (states, smoothings, circles) | done; validated against KnotInfo through 11 crossings | |
| 1. Back end | `GradedComplex` over F2; the set-based reference reducer (rank, kernel, homology) with the `d² = 0` check; the exact ℚ lane | done | |
| 2. Khovanov over F2 | unreduced cube complex: enhanced states, gradings, differential; size from the diagram | done; validated against KnotInfo's mod-2 ranks | |
| 3. Lee and *s* | signed Khovanov over ℚ; the Lee deformation; *s* from the quantum filtration; exact `Fraction` arithmetic | done; validated against KnotInfo for s = 0, ±2, ±4, ±6 | KnotInfo has s = 2τ on every row, so a check that tells s from 2τ needs an off-table knot such as 14n19265 (`docs/validation.md`). A calculation is one job; its build and per-grading reductions are not split |
| 4. Exact reductions | Gaussian cancellation over F2 and ℚ (`reduce_gaussian.py`), verified `raw == reduced` | done, as a reference: it rescans for a pivot at each step | It cancels after the full cube is built. Cancelling while building (Bar-Natan scanning) is not implemented |
| 5. Reduction tiers | `bitint`, numpy words, Numba, the dense GPU kernel; the tier registry; GPU detection; size and cost prediction | done; every tier equals the reference, GPU and Numba confirmed on hardware | On-device GPU kernel and `multimodular.py` (section 3) |
| 6. Grid Floer | grid model, gradings, bigraded and filtered differentials, HFK-hat, three-genus, τ; in `compute()`; validated by `knot_floer_homology` and KnotInfo | done to grid number 10. Default tests cover the 13 roster knots of grid number ≤ 8; `pytest --heavy` covers the 185 of grid number ≤ 10 | Off-table knots: the grid comes from KnotInfo's `grid_notation`. Grid number 11. τ's rank and level search are two long single jobs on the set-based reducer. No test sends an oversized grid through the gate (the refusal is tested with synthetic jobs) |
| 7. Filtered complex as an artifact | (a) the bigraded grid complex with its bifiltered differential as a serializable object: generators with (Maslov, Alexander), arrows with their O- and X-crossing counts. (b) Reduction over F2, by filtered change of basis, to the simplified CFK (staircase ⊕ boxes). (c) ε, ν and Υ(t) read from the reduced model | planned | Oracles: ε and ν against KnotInfo and `knot_floer_homology`; the reduced complex against `knot_floer_homology` (`complex=True`); Υ against torus-knot closed forms. KnotInfo has no Υ column and Szabó's calculator does not output Υ; `yui-hfk` is a candidate oracle. Same grid-number bound as Phase 6 |
| 8. Compact grid representation | generators named by lexicographic rank; slice results as columnar arrays; complexes stored CSC | built in the scheduled path. `grid_complexes` keeps permutation tuples as the readable oracle | Grid number 11 still does not fit. An on-disk form of the complex, decided with Phase 7 (a) |
| 9. Reduced Khovanov | the basepoint-reduced complex: the subcomplex of enhanced states that label the basepoint circle *x*. A front-end change in `engines/khovanov/gradings.py::enhanced_generators`; the back end is untouched. F2 first | planned | Pin the *q*-shift against KnotInfo's `khovanov_reduced_mod2_vector`. Self-check: the result is the same at two basepoints. Wire a computed oracle: Khoca already returns the reduced theory, knotkit has `kk kh -r`, kht++ covers knots with a braid word (`../research/khtpp.md`, `../research/knotkit.md`). The canonical name follows `SPEC.md` section 7 |

The Floer roster is the KnotInfo knots with HFK, τ and three-genus all tabulated (2,977),
filtered by grid number: 13 at ≤ 8, 41 at ≤ 9, 185 at ≤ 10. `floer_roster` derives it from the
installed table.

## 8. Conventions

| Convention | Value | How it is pinned |
|---|---|---|
| Jones variable | `jones.py` uses KnotInfo's *t*; relative to the Khovanov *q*, t = q⁻² | reproduces KnotInfo's Jones polynomial, chirality included |
| Khovanov gradings | i = \|s\| − n₋, j = (#₊ − #₋) + \|s\| + n₊ − 2n₋, where \|s\| counts 1-smoothings | graded Euler characteristic of the cube = (q + q⁻¹) · V(q⁻²) |
| Khovanov chirality | the native value is the Khovanov homology of the given PD. KnotInfo tabulates the opposite chirality from its own PD, so the KnotInfo oracle is mirrored, (i, j) → (−i, −j), in `backends/knotinfo_backend.py` | F2 and ℚ tables against KnotInfo |
| Khovanov over F2 from KnotInfo | derived from the stored integral vector by universal coefficients | |
| Lee complex | one complex graded by i and filtered by q; Lee homology has dimension 2 for a knot | checked on every calculation |
| Rasmussen *s* | s of the given diagram: s_min + 1, with s_max = s_min + 2 checked. Compared against −(KnotInfo's `rasmussen_invariant`) | s(mirror) = −s |
| Grid markers | KnotInfo's `grid_notation` does not label O and X. They are recovered by tracing the knot through the markers and 2-colouring the cycle, then the rows are reversed to reach KnotInfo's chirality | HFK-hat and τ equal KnotInfo's values with no mirror and no sign change |
| Floer gradings | (Maslov, Alexander); A = (M_O − M_X − (n − 1)) / 2; back-end degree = −Maslov | graded Euler characteristic = (1 − t)^(n−1) · Δ_K(t) |
| HFK-hat from grid homology | grid homology = HFK-hat ⊗ V^(n−1), V = F2 in (0, 0) ⊕ F2 in (−1, −1) | the division is checked by rebuilding the product |
| Three-genus | the top Alexander grading where HFK-hat is nonzero | |
| τ | the least Alexander level i at which H₀(F_i) → H₀(total) is nonzero, in the complex whose differential counts rectangles avoiding only O | dim H₀ = 1 is checked; the mirror negates τ |
| Floer oracles | compared raw: KnotInfo's HFK, τ and genus columns share its PD chirality, and so does `knot_floer_homology` fed that PD | |

Code comments call the KnotInfo comparisons of Phases 2 and 3 "Phase 2c" and "Phase 3c".

"Reduced" has three unrelated senses here. In Phase 9 and `coverage-map.md` it is the
basepoint-reduced theory. In section 4 and Phase 4, `raw == reduced` is the
Gaussian-cancelled complex. In Phase 7 and `four-manifold-objectives.md`, the reduced
filtered complex is the simplified CFK.
