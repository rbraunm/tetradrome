# Research: kht++ -- Bar-Natan tangle invariants, read through braid words

Empirical discovery for the `khtpp` binary, provisioned by `scripts/install_oracles.sh`.

The short version: kht++ computes the **reduced** Bar-Natan / Khovanov theory of tangles, and
reads only its own Morse-word tangle format. Braid words transcribe mechanically into that
format, so it is in the comparison roster: it measures the `khovanov_reduced` target row over
F2, and -- by the Shumakovitch tensor -- the native `khovanov_homology` (F2) row, verified
against native. It is not a validator: F2 Khovanov already has three, and the reduced theory
has no canonical name until `roadmap/design/homology-engine.md` section 7, Phase 9 builds a
native engine (ADR 0006 makes an oracle a checker of a native computation, never a producer).

Claims are marked **verified** (observed against native or another oracle) or **derived**.

## Provisioning (verified)

- Binary `/opt/oracles/khtpp/kht++`, wrapper `/usr/local/bin/khtpp`, version reported as
  a git sha (`a8be0af-dirty`).
- Heavy C++ build requiring Eigen. On a 1 CPU / 4 GB sandbox the compile peaks around
  1 GB resident on `Cob.cpp`, well inside the ceiling, but it is slow and single-core.
- A backgrounded `setsid nohup` install was observed **reaped mid-build with no error
  written to the log**, leaving 6 object files and no binary. Not OOM. Rerunning the
  installer converged because `make` is incremental. Poll and rerun rather than trusting
  a single launch.

## CLI and input (verified)

- `khtpp --help` and `khtpp -h` print only a version banner. There is no usage text.
- `docs/Input.md` documents exactly **three** input paths: an interactive dialogue, a
  `.kht` file, and derivation from an existing `.kht`. A grep of the whole docs tree for
  PD, braid, DT, or Rolfsen-name input returns nothing.
- The `.kht` format is a **Morse tangle word**. The shipped `examples/tests/3_1.kht` is:

  ```
  % 3_1
  r1.y0.y0.y0.u1
  ,1
  ```

  A comment line, the Morse word (cup / crossing / cap generators with positions), and
  an orientation line.

**The gate opens from braid words (verified).** A braid on n strands, closed as a 1-1
tangle, transcribes mechanically: nested caps `l1 ... l(n-1)` open the return arcs to the
right of the braid, each generator becomes a crossing slice at index i-1 (`x` or `y` by
sign), cups `u(n-1) ... u1` close the arcs, and the single top end points down (`,0`).
KnotInfo carries braid words for the table knots, so no PD-to-Morse topology code is
needed. Run on the trefoil (`l1.x0.x0.x0.u1`) and the figure-eight
(`l1.l2.x0.y1.x0.y1.u2.u1`), kht++ computed both with no orientation complaints and the
right reduced ranks: C_0 + C_1 = 3 generators, and C_0 + C_1 + C_1 = 5. The C_0 summand
sat at q = -2 for that (left-handed) trefoil and q = 0 for 4_1, consistent with the
s-from-C_0 hypothesis below.

**Wiring conventions (verified).** A positive generator is `x` and a negative one `y`. Fed
KnotInfo's braid word for each knot, that mapping lands on our knot's own chirality: unreduced
F2 derived from the output equals native F2 on 3_1, 4_1, 5_2, 6_1, 6_2, 7_4, 7_7, 8_19, 9_42
and 10_124 (10/10, covering the benchmark ladder), and the swapped mapping gives the mirror --
which also shows KnotInfo's braid words share our PD's chirality there. For a few knots
KnotInfo lists alternative braid words as a list of lists (10_136 is the only one through 10
crossings); the first is used, and on 10_136 it too gives native F2.

A braid-presented knot is not fed its own word: native Khovanov needs a PD, so that route
could never be checked against native. The adapter reads the `cxCKh-c2` data file, which
carries the same summand lines as the terminal output without ANSI codes, after a `%` header.

**Path quirk (verified):** kht++ refuses a `.kht` file in the working directory itself
("Please put your file in a subdirectory") and strips a leading slash from absolute paths,
so a caller must place the file in a subdirectory and pass a relative path.

## Output (verified)

Running the shipped trefoil prints a loop-type complex, plus an HTML file, plus
machine-readable data files **`cxCKh`, `cxBNr`, `cxKhr`** which `docs/Output.md`
describes as intended for automatic post-processing. Any adapter should read those
files, not the terminal output.

The trefoil decomposes into two summands, reported as roughly:

```
1) h^0 q^2 d^1   (a single object, no differential)
2) h^2 q^6 d^1   (two objects joined by multiplication by H)
```

`docs/Output.md` states the framework directly: the Khovanov invariant of a link is a
complex over `k[H]`, and **what is usually known as reduced Khovanov homology is the
complex obtained by setting H = 0**. Irreducible complexes are classified: `C_0` is a
single object with no differential; `C_n` is two objects joined by multiplication by
`H^n`. For `n > 0` only the **source** bigrading is printed. The gradings satisfy
`q/2 = h + d`.

Default coefficient field is F2 (`-c2`).

## The H = 0 expansion (verified)

- **Setting H = 0 recovers reduced Khovanov.** A `C_0` summand gives one generator at its
  printed (h, q); a `C_n` summand gives its printed source (h, q) and its target
  **(h + 1, q + 2n)** -- the differential raises h by exactly one, and H has q-degree -2.
  Verified by the F2 results above, which include `C_2` summands on 8_19 and 10_124: placing
  the target at (h + n, q + 2n) instead breaks 8_19 (pinned by a test). Up to 10 crossings,
  `C_2` appears in 8_19, 10_124, 10_128, 10_139, 10_152, 10_154 and 10_161; no higher n.
- **Every summand line satisfies kht++'s identity q/2 = h + d**, which the parser enforces.

## Why kht++ has no rasmussen_s cell

The `s` read off the unique `C_0` summand looked plausible on the trefoil and figure-eight,
but kht++ is trustworthy only over F2 -- its documentation calls the rational arithmetic
experimental and unchecked for integer overflow -- and s over F2 is a different invariant
from Rasmussen's s over Q; the two are known to differ on some knots. Measuring it in the
canonical `rasmussen_s` row would label one invariant with another's name.

## Cross-oracle corroboration (verified)

khoca's `InteractiveCalculator` returns `[reduced, unreduced]`. The field-ring validator
path reads only the unreduced half; the benchmark's integral call reports the reduced half
in the `khovanov_reduced` row. The reduced half for the trefoil, both field rings, in
KnotInfo's q-convention: (0,2), (2,6), (3,8) -- identical to what kht++ prints.

The same probe also settles the reduced-to-unreduced question empirically:

- **Over F2**, khoca's unreduced output is exactly the reduced groups tensored with
  `(q + q^-1)`: reduced `(0,-2),(2,-6),(3,-8)` expands to
  `(0,-1),(0,-3),(2,-5),(2,-7),(3,-7),(3,-9)`, which is `out[1]` term for term. The
  Shumakovitch relation holds.
- **Over Q** it does not: unreduced is four generators, not six, with no tensor
  structure.

So a reduced oracle maps onto the existing `khovanov_homology` (F2) row by a
theorem-backed transform, and onto nothing for the Q row.

## Cost ranking for Phase 9

When a native reduced engine exists and needs a computed oracle (ADR 0006):

1. **khoca** -- already computes the reduced half, and the benchmark measures it. No new
   provisioning at all.
2. **knotkit** -- `kk kh -r -f {Q,Z2}` works; costs LaTeX parsing.
3. **kht++** -- the most natural fit mathematically, and already measured on the reduced
   row, but reachable only for knots with a KnotInfo braid word.

## Vocabulary warning

"Reduced" means three unrelated things in this repo, and conflating them is the most
likely way someone wires the wrong thing:

- here and in `coverage-map.md`: the **basepoint-reduced theory**;
- in `homology-engine.md` section 4 / Phase 4 and ADR 0007, `raw == reduced`: the
  **Gaussian-cancelled** complex, an optimization;
- in `four-manifold-objectives.md`: the **reduced filtered complex** for CFK-infinity.
