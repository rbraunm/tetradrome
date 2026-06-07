# Homology Computation Engine — Design & Implementation Path

**Status:** Phases 0–4 implemented and validated — Jones (warm-up), the shared
back end, native Khovanov over F2 and ℚ, the Lee deformation, the Rasmussen
*s*-invariant (all wired into `compute()` and checked against KnotInfo), and exact
Gaussian-cancellation reduction (cross-checked `raw == reduced`). Acceleration
(Phase 5) is code-complete — every tier (bit-packed CPU, numba JIT, dense GPU kernel,
multi-core + NUMA pinning, size/VRAM routing, multimodular ℚ) is written and validated
`== reference` here through shared code, with a CPU/GPU benchmark harness; what remains
is validating the GPU/numba/multi-socket *speed* on real hardware and calibrating the
routing threshold. Floer (Phase 6) follows. The §7 phase plan carries per-phase
status inline.

**Scope:** The computational substrate for the homological invariants — Khovanov
(and Lee / Rasmussen *s*) and, later, knot Floer (τ, ε, ν, HFK). This document
elaborates `SPEC.md` §13.4–13.7 and §19, is governed by decisions
[0003](../decisions/0003-native-coefficient-field.md) (F2 first, packed-bit GF(2))
and [0004](../decisions/0004-validate-by-default-error-policy.md) (validate by
default), and follows the licensing posture of `SPEC.md` §20 (Apache-2.0; GPL tools
as external validators only, never incorporated).

It exists because the design conversation produced more detail than the SPEC sketch
holds — specifically the engine-vs-acceleration layering, the faithfulness rule, the
memory predictor, the multimodular path to ℚ, and a general-to-reduced implementation
order. Capture is the point; nothing here should be lost when code starts.

---

## 1. Background: why native, not an external backend

The obvious shortcut for Floer is the `knot_floer_homology` PyPI package — a Python
wrapper around Zoltán Szabó's HFK Calculator (a C++ program), maintained by the
SnapPy team. It is rejected as a *core* dependency for four independent reasons, any
one of which is sufficient:

1. **License.** It is GPLv2+. Tetradrome is Apache-2.0. A GPL runtime dependency is
   a copyleft entanglement: shipped as a combined work, GPL terms could reach our
   code. Acceptable only as an opt-in extra the *user* installs into their own
   environment — never bundled, never core.
2. **It is not our math.** Using it means reporting Szabó's answer, not computing
   one. That defeats the purpose of the workbench.
3. **Portability.** PyPI shows *no source distribution* — binary wheels only, and
   each is a glibc/arch-pinned C++ extension (manylinux_2_17, etc.). On any
   platform/Python without a prebuilt wheel it is not slow to install, it is
   *uninstallable* — there is nothing to build from. That fails "pure Python, runs
   anywhere." (The bare package is fairly leaf-like — PD in, dict out — but it is
   normally consumed via the heavy spherogram/SnapPy stack.)
4. **The kernel is the opportunity, not the liability.** The reason Floer/Khovanov
   are hard is the inner loop: sparse linear algebra over a cube of resolutions.
   That is exactly the kernel we want to *own* and accelerate (JIT / NUMA / GPU),
   not rent as an opaque binary.

Maintenance status is secondary to all of the above, and ambiguous anyway: last
*version* was 1.2.2 (Jan 2025), with a sporadic history (1.2 in 2022, then 1.2.1 in
late 2024), but Python 3.14 wheels were rebuilt Dec 2025 — "stable, low-activity,
kept building," which is normal for a thin wrapper over a frozen 2017 calculator. We
do not depend on it regardless.

**A sharper mathematical point.** Even Floer's τ does not directly resolve the
motivating example: τ, *s*, ε, and ν all vanish on the Conway knot itself — that is
why it resisted for decades. Piccirillo's result came from *s(K′)* on a
trace-sibling K′, not from any invariant computed on the Conway knot. So the
invariant with real concordance teeth is **Rasmussen *s* (from Khovanov/Lee)**, and
Khovanov is therefore the higher-priority native engine. (See `SPEC.md` §18 and
`../../docs/conway_notes.md`.)

External tools may still appear later strictly as **opt-in cross-checkers**, exactly
where KnotInfo sits (a validation oracle), never as a compute step. That is the
`SPEC.md` §13.8 migration-layer / adapter contract, not this engine.

---

## 2. Architecture: two layers that must not be conflated

There are two separate layers. The acceleration discussion (pure → JIT → NUMA → GPU)
is *one* of them; the invariant theories are the *other*. Floer is not a tier in the
acceleration stack — it is a peer of Khovanov in the theory layer.

```text
FRONT ENDS (theories — different math per engine)
  Seifert-form      Khovanov complex      Floer complex
  (exception)       (cube of res.)        (grid rectangles  OR  Szabo HFK cube)
        |                  |                          |
        |                  v                          v
        |            graded chain complex over a ring (F2 / F_p / Z / Q)
        |                  |                          |
        |                  +------------+-------------+
        |                               v
        |               SHARED BACK END (invariant-agnostic)
        |        graded sparse linear algebra  ->  homology (ranks)
        |          [ acceleration tiers live HERE, see §3 ]
        v
  tiny dense integer linear algebra (bypasses the F2 back end entirely)
```

**Front ends (theory layer).** Each emits a graded chain complex from a diagram and
is *different math*; they are not interchangeable:

- **Khovanov** — cube of resolutions over a Frobenius algebra; 2ⁿ vertices.
- **Floer** — entirely separate machinery. Two combinatorial routes:
  *grid homology* (Manolescu–Ozsváth–Sarkar–Szabó: generators are permutations on an
  n×n grid, differential counts empty rectangles) or *Szabó's HFK cube*. Its
  bottleneck is **n! generation** — a different scaling beast than Khovanov's 2ⁿ —
  so even though the back-end reducer is shared, the front ends are not.
- **Seifert-form** — the exception. It is a 2g×2g *dense integer* matrix
  (`src/tetradrome/invariants/seifert.py`, already built and validated). It does not
  touch the F2 back end at all; different regime.

**Back end (algebra layer, `SPEC.md` §13.6).** Takes a graded sparse complex over a
ring and computes homology by reduction (rank / kernel / image → homology
dimension). It "knows nothing about the Conway knot" — invariant-agnostic. **This is
where all acceleration lives**, which is the whole payoff: written once, every theory
that emits a complex feeds it. Khovanov and Floer both reduce here; only Seifert-form
sits outside.

### Invariant → engine map

| Invariant | Engine | Back end | Status |
|---|---|---|---|
| determinant, signature, Alexander | Seifert-form | dense integer (own) | **done, validated** |
| Jones polynomial | Kauffman bracket + cube skeleton | (combinatorial, light) | **done, validated** |
| Khovanov homology | Khovanov complex | shared reducer (F2 fast lane / exact ℚ lane) | **done, validated (F2 and ℚ)** |
| Rasmussen *s* | Khovanov + Lee deformation | shared reducer over ℚ (exact `Fraction`; multimodular is a Phase 5 optimization) | **done, validated** |
| τ, ε, ν, HFK ranks | Floer (grid or Szabó cube) | shared reducer | later |

### Module layout (as built; Phase-5/6 entries still indicative)

```text
src/tetradrome/
  invariants/
    seifert.py            # done: dense-integer engine (det/sig/Alexander)
    jones.py              # done: Kauffman bracket -> Jones
    compute.py            # done: dispatch + validate-by-default + provenance
  engines/                # FRONT ENDS (one subpackage per theory)
    cube.py               # done: resolution cube (states, smoothings, circles)
    khovanov/             # done: gradings, differential (F2 + signed ℚ), homology,
                          #       lee.py (Lee deformation), rasmussen.py (s-invariant)
    floer/                # grid homology and/or Szabo cube (Phase 6)
  algebra/                # SHARED BACK END (invariant-agnostic)
    complex.py            # done: F2 graded complex (the fast-lane representation)
    reduce_reference.py   # done: pure-Python F2 reference reducer
    rational_complex.py   # done: ℚ graded complex (the rational lane)
    rational_reduce.py    # done: exact-ℚ reference reducer (Fraction)
    reduce_f2_packed.py   # bit-packed F2 reducer (Phase 5)
    multimodular.py       # primes + CRT + rational reconstruction (Phase 5 optimization)
    memory.py             # complex-size predictor + fill-in estimate + routing (Phase 5)
    tiers.py              # runtime tier detection + selection + fallback (Phase 5)
```

Lee and the s-invariant live under `engines/khovanov/` rather than a separate
`lee/` subpackage: they reuse the same signed cube and generators, deforming only
the Frobenius maps. The back end ended up two reference lanes behind one interface
(an F2 fast lane and an exact-ℚ lane) rather than one ring-parameterized reducer —
the field-tested split (Ripser/PHAT keep F2 special; Sage parameterizes one
complex), chosen so the validated F2 path stayed untouched when ℚ was added.

---

## 3. Acceleration tier model (runtime, hardware-adaptive)

Distinct from the *development* ladder in `SPEC.md` §4.2/§13.7
(reference → optimized CPU → GPU → agreement tests), which is how each tier is
*built*. This section is the *runtime* model: how a built system *chooses* a tier on
the host it finds itself on.

**Tiers**, floor to ceiling, each optional and detected at runtime:

1. **Pure-Python / stdlib** — the floor. Always present, source of truth, the
   reference every other tier is checked against. Slowest; never absent.
2. **JIT (Numba)** — bit-packed kernels compiled to machine code. Present when Numba
   is installed.
3. **Multi-core / NUMA** — parallel across cores and sockets.
4. **GPU (CuPy / Numba-CUDA)** — present only if a usable GPU and the optional deps
   are there.

**Selection is for the *user's* hardware, not the author's.** The same code runs on a
Raspberry Pi and a 4-GPU box; it simply lights up more of itself on the bigger
machine. GPU is never required. Tier selection is runtime-detected with graceful
fallback, and — non-negotiable — **every tier returns the identical answer** (§4).

**Representation.**

- **F2:** bit-packed — F2 vectors as arrays of uint64 words; reduction is XOR /
  popcount over word arrays. Exact (it is bits; no precision story) and a near-ideal
  JIT/GPU target. Matches decision 0003's packed-bit GF(2) matrix type.
- **ℚ (Lee / *s*):** big rationals are JIT-hostile, so do **not** carry ℚ through the
  hot loop. Use **multimodular**: reduce over a batch of primes (each a fixed-width
  F_p reduction, JIT/GPU-friendly), then recombine by CRT + rational reconstruction.
  Exact result, fixed-width inner loop. (`algebra/multimodular.py`.)

**Grading parallelism.** A graded complex splits into independent summands by
(homological, quantum) grading — embarrassingly parallel: many independent matrices
to reduce at once. This is the main lever for multi-core/NUMA/GPU and is theory-
agnostic (lives in the back end).

---

## 4. Faithfulness principle (non-negotiable)

**Hard rule: no lossy shortcuts, no heuristics, no probabilistic rank anywhere in the
core.** This matters *more* here than in a typical library, because the fast
knot-homology tools are precisely the ones that bake in assumptions — thin /
alternating / "mod 2 and assume no torsion" — and those assumptions break on exactly
the outlier knots this project exists for.

### The distinction that the rule hinges on

Not all "smart" algebra is a shortcut. There are two categorically different things:

- **Exact reductions — ALLOWED.** Delooping and local Gaussian elimination
  (Bar-Natan's divide-and-conquer / "local" Khovanov algorithm) are **chain homotopy
  equivalences**: provably identical homology, gradings, and every derived invariant.
  They assume nothing about the knot and close off no use-case. They are "Bareiss
  instead of cofactor expansion" — a faithful *algorithm choice*, not a mathematical
  compromise.
- **Heuristics — BANNED in the core.** Truncating gradings; assuming thinness or
  alternation; Monte-Carlo / probabilistic rank; "mod 2 and assume no torsion";
  early-termination guesses. These bake in assumptions that can be wrong on the
  inputs that matter.

### How the architecture enforces faithfulness

1. **The raw, unreduced path is first-class and always runnable.** It is the source
   of truth, the most general (any coefficient ring, any derived quantity), and the
   reference everything else is checked against. It is allowed to be slower and more
   memory-hungry; correctness and generality outrank speed.
2. **Exact reductions are an optional, toggleable pre-pass**, and we verify
   `raw == reduced` across the catalog. This is decision 0004 (validate by default)
   pointed *inward*.
3. **The GPU kernel is just exact rank/elimination over the ring** — faithful by
   nature, and agnostic to whether it is fed the raw complex or a reduced one. So
   leading with reductions (§6) never taints the GPU path: the kernel does the
   identical math either way, and the raw complex can always be run on it directly if
   one wants to spend the memory.

The same agreement discipline covers the tiers: **fast path must equal the reference
path** — a free internal cross-check, again 0004 inward.

---

## 5. Memory prediction & gating (requirement)

Predict the memory a specific calculation needs, and **never blindly throw a ton of
memory at a system with a limited GPU**. This is the "fail loud and early" rule
applied to memory: tell the user "this needs ~X GB, you have Y" *before* starting,
never an OOM forty minutes in or a silent swap-death.

**What is predictable, honestly:**

- **Initial complex size — exact and cheap.** The dimension of every (homological,
  quantum) graded piece is computable directly from the diagram *without building
  anything*: it is combinatorial — 2ⁿ vertices, 2^(circles at that vertex)
  generators each, O(n·2ⁿ) to profile the whole thing. This gives the exact storage
  for the *unreduced* complex.
- **Elimination peak (fill-in) — boundable and estimable, not exact.** Fill-in
  during reduction is the classic sparse-direct unknown. We have a hard upper bound
  (dense worst case, which is predictable) and a heuristic/symbolic estimate, but not
  an exact figure.

**Gate and route.** Compute the estimate; compare against **available VRAM
specifically** (the tight constraint — far smaller than system RAM) as well as system
RAM; then route:

- fits in VRAM → GPU tier;
- fits in RAM but not VRAM → CPU/RAM tier;
- fits in neither directly → tile/stream through VRAM, or **refuse loudly with the
  number**.

Never silently degrade to swap, and never silently shrink the math to fit a small
GPU.

**Where exact reductions legitimately earn a default-off opt-in:** as a *size* tool,
when the faithful raw path will not fit the box. Because the reduction is exact
(§4), shrinking-to-fit costs no fidelity — but it is the user's explicit choice, not
a silent default to cram onto constrained VRAM.

---

## 6. Honest performance calibration (so we do not oversell)

- **CPU JIT vs pure-Python baseline:** a massive win, unambiguous. It is the right
  default.
- **CPU JIT vs Szabó's hand-written C++:** same order of magnitude. LLVM gets Numba
  close to C on the bit-packed F2 kernels; on many-core/NUMA we can plausibly pull
  ahead; on a single laptop "comparable" is the honest word.
- **The decisive advantage is not raw speed — it is that ours *runs at all*.** For a
  user whose platform has no prebuilt wheel, "comparable C-speed that installs" beats
  "faster C++ that will not install" by an infinite margin.
- **GPU:** real upside on the wide/dense reductions and many-grading batches, but
  *not* a clean win — F2 elimination with pivoting is partly serial — and never
  required. The frontier *hard* cases are usually **memory-bound, not flop-bound**
  (fill-in vs small VRAM), so GPU is a big win in the medium-large regime (where a
  workbench is most useful) and not a bleeding-edge magic bullet.
- **Algorithm beats kernel.** The biggest historical wins came from smarter algebra
  *before* linear algebra — Bar-Natan's local/divide-and-conquer reduction shrinks
  the problem so much that the residual field linear algebra is often small. That is
  both the largest multiplier *and* what makes large cases fit in memory. It is also
  exact, so it leads; GPU is a multiplier on whatever dense work remains.
- **NUMA-awareness is the genuinely fiddly part:** thread pinning plus memory
  locality — likely `numactl` and explicit work partitioning, and eventually perhaps
  a small Cython kernel *we* write and can read. The point of owning it is that it is
  auditable and tunable to the host, not a black box.

Why is GPU underexploited in this space today? Mostly path dependence: the mainstream
tools (KhoHo, KnotKit, regina, Szabó's calculator) are decades of accreted C/C++ by a
handful of people, and porting irregular sparse C++ to CUDA is a large effort with no
paper at the end. That is an incentive problem, not a verdict that GPU cannot help.

---

## 7. Implementation path (general → reduced/accelerated)

Each phase is validated before the next begins. Reductions and acceleration are added
*after* a faithful reference exists and must never change an answer.

- **Phase 0 — Jones warm-up. [done, validated]** Kauffman bracket plus the cube
  *skeleton* (state enumeration, crossing resolution, circle detection) at trivial
  compute cost. Exercises and validates the cube machinery and the bracket; it is the
  scaffold Khovanov bolts directly onto. Jones validated against KnotInfo through 11
  crossings. (Pinned here: our gradings put `t = q⁻²` relative to the Khovanov `q`.)
- **Phase 1 — Back-end interface + reference reducer. [done]** The graded chain
  complex (chain groups by grading; sparse boundary maps) and a pure-Python F2 reducer
  (rank → homology dimension) with the `d² = 0` check — the general, faithful core and
  the reference for everything later. `total_dim` lives here; the cheap-from-diagram
  size predictor is cube-specific and lands with Khovanov in Phase 2. (The exact-ℚ
  lane — `rational_complex` / `rational_reduce` — was added alongside in Phase 3, when
  Lee first needed it.)
- **Phase 2 — Khovanov front end (raw/faithful). [done, validated]** The unreduced
  cube complex over F2 (enhanced states, gradings, differential), fed to the reference
  reducer; Khovanov homology validated against KnotInfo's mod-2 data (derived from the
  stored integral vector by universal coefficients). First full faithful path end to
  end. Includes the cheap-from-diagram size predictor. **Chirality pinned here:**
  KnotInfo tabulates Khovanov in the opposite chirality from its own stored PD, so our
  value (the correct Khovanov of the given diagram) matches up to the global mirror
  `(i, j) → (−i, −j)`; the oracle mirrors to compensate.
- **Phase 3 — Lee / Rasmussen *s*. [done, validated]** Signed Khovanov over ℚ (cube
  edge signs, validated by `d² = 0` over ℚ and against KnotInfo's free ranks); the Lee
  deformation (a single complex graded by `i`, filtered by `q`; Lee homology is
  2-dimensional for a knot); and *s* read off the quantum filtration on Lee homology,
  validated against KnotInfo across `s = 0, ±2, ±4, ±6`. **Deviation from the original
  plan:** the reference works over ℚ with **exact `Fraction` arithmetic, not
  multimodular**. Multimodular (primes + CRT + rational reconstruction) is a
  fixed-width *optimization* and belongs in Phase 5 with the other acceleration tiers
  (decision 0007: faithful reference first), not in the reference path. All three
  invariants, plus Jones, are wired into `compute()` with validate-by-default.
- **Phase 4 — Exact reductions. [done]** Gaussian cancellation of the complex (the
  elimination lemma; delooping is implicit in the generators), field-agnostic over F2
  and ℚ. An independent homology algorithm, validated `raw == reduced` across the
  catalog on both lanes, that collapses the full cube to its homology dimension (the
  engine's memory tool in embryo — e.g. 7₄'s Lee cube, 1182 → 2). Unoptimized
  reference; partial-reduction-as-pre-pass and the fill-in/VRAM routing land in Phase 5
  where the tiers exist.
- **Phase 5 — Acceleration tiers. [in progress]** Bit-packed F2 reducer → Numba JIT →
  multi-core/NUMA → GPU (CuPy / Numba-CUDA), each validated against the pure
  reference via agreement tests (`SPEC.md` §4.2/§13.7). Wire the runtime tier
  selector and the memory predictor/router (the fill-in estimate + VRAM-aware
  routing land here, where GPU/tiers actually exist).
  *Done:* the bit-packed F2 tier (`reduce_f2_packed.py` — pure-Python int bit-vectors,
  plus uint64 word arrays parameterized by the array module so one body of code runs on
  numpy/CPU and cupy/GPU), the runtime registry with detection and loud fallback
  (`tiers.py`), the per-backend agreement tests, the CPU-vs-GPU accuracy+speed harness
  (`scripts/bench_reducers.py`), and runtime GPU detection / auto-configuration /
  enablement guidance (`gpu.py` — driver + cupy + card specs probed at runtime, VRAM
  budget derived from the card, install steps when the hardware is present but the stack
  is not; no card model assumed), multi-core parallel reduction across the independent
  complexes (`parallel.py` — process pool, parallel == serial), and size prediction with
  size/VRAM-aware routing (`memory.py` — packed reduction peak per complex, GPU only when
  it fits the measured budget and clears a calibratable threshold, loud failure over a RAM
  budget), and the multimodular ℚ path (`multimodular.py` — rank over ℚ as the max of ranks
  mod several large primes, dodging `Fraction` coefficient explosion; validated identical to
  the exact reducer), the Numba JIT tier (`reduce_f2_jit.py` — njit-compatible packed
  reducer, compiled when numba is present, plain numpy otherwise), the batched dense GPU
  kernel (`f2_rank_dense` — vectorized row reduction, one host sync per column instead of
  per step), and NUMA-aware core pinning for the parallel pool (`parallel.py`, Linux).
  Every tier is validated `== reference` here via shared code (the GPU and numba paths
  through their un-accelerated equivalents). *Remaining is hardware validation, not new
  code:* on a CUDA box, a multi-socket box, and with numba installed, run the suite
  (agreement) and `scripts/bench_reducers.py` (`--sizes`, `--workers`, `--pin`) to confirm
  each accelerator is correct and faster on real silicon, then calibrate the GPU routing
  threshold (`memory.py`) from the measured crossover.
- **Phase 6 — Floer front end (peer engine).** Grid homology (MOS rectangles)
  and/or the Szabó HFK cube, feeding the *same* back end; τ, ε, ν, HFK ranks;
  validate against KnotInfo. Note the n! generation bottleneck → generation-side
  parallelism is its own concern, separate from the shared reducer.

Ordering rationale: general and faithful first (Phases 0–3 produce correct answers
with the reference reducer), exact reductions second (Phase 4, still answer-identical
and individually validated), acceleration last (Phase 5, validated against the
reference) — so at no point does an optimization precede the thing it must agree with.

---

## 8. Relationship to existing SPEC & decisions

- **Elaborates** `SPEC.md` §13.4 (Khovanov), §13.5 (Lee/Rasmussen), §13.6 (algebra
  layer), §13.7 (GPU/CUDA), §19 (performance strategy).
- **Consistent with** §4.2 (GPU real-but-narrow) and §20 (Apache-2.0; GPL tools
  external-only).
- **Governed by** decision 0003 (F2 first; packed-bit GF(2)) and 0004 (validate by
  default — here also pointed inward: `raw == reduced`, and tier `== reference`).

### Decision records this design rests on

The following were split out of this design into their own ADRs (the process itself is
recorded in [0005](../decisions/0005-decision-record-process.md), which also defines
the soft-lock semantics — ADRs stay reviewable even when "locked"):

1. [0006](../decisions/0006-no-external-compute-backends.md) — **No external compute
   backends in the core.** Portability (no-sdist binary wheels), license (GPLv2+ vs
   Apache-2.0), and "not our math" disqualify `knot_floer_homology` and similar as
   anything but opt-in external validators.
2. [0007](../decisions/0007-faithful-raw-path-no-heuristics.md) — **Faithful raw path
   first-class; only exact (homotopy-equivalence) reductions, toggleable and verified
   `raw == reduced`; no heuristics in the core.**
3. [0008](../decisions/0008-memory-prediction-gate.md) — **Memory-prediction gate with
   VRAM-aware routing; fail loud and early; exact reduction is an opt-in size tool,
   never a silent shrink-to-fit.**
