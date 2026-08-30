# Research: kht++ -- Bar-Natan tangle invariants, and why it is sequenced behind Phase 9

Empirical discovery for the `khtpp` binary, provisioned by `scripts/install_oracles.sh`
but deliberately **not** in the comparison roster or the validator registry.

The short version: kht++ is a coherent, published tool computing the **reduced** theory,
and our schema has no reduced Khovanov row until `roadmap/design/homology-engine.md`
section 7, Phase 9 builds one. It is not a dead end -- it is correctly ordered behind a
native engine that does not exist yet, since ADR 0006 makes an oracle a checker of a
native computation and never a producer. When Phase 9 lands, kht++ is the third-cheapest
of three available reduced oracles.

Claims are marked **verified** (observed) or **derived** (reasoned from the docs and one
data point, needing a probe).

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

**This is the gate.** No output convention can be probed on the chiral sweep until the
sweep can be expressed as Morse words, and we ship exactly two `.kht` files, both
examples. Whether a braid word we already carry transcribes mechanically into a Morse
word is the first question any future kht++ work must answer.

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

## Derived, not verified

- **Setting H = 0 recovers reduced Khovanov.** For the trefoil, `C_0` at (0,2) plus
  `C_1` with source (2,6) gives generators at (0,2), (2,6), (3,8) -- the reduced
  Khovanov homology of the right-handed trefoil. **Independently corroborated**: khoca's
  reduced output (see below) is identical.
- **Target bigrading rule.** Since only the source is printed and `H` preserves `d`
  while shifting `(h,q)`, a `C_n` summand's target sits at `(h+n, q+2n)`. Consistent
  with the trefoil's (3,8), but derived from one example.
- **`s` may be the q-grading of the unique `C_0` summand.** The trefoil's `C_0` is at
  `q^2` and `s` of the right-handed trefoil is 2. **One data point.** If it holds, kht++
  is a third `s` oracle.

## Cross-oracle corroboration (verified)

khoca's `InteractiveCalculator` returns `[reduced, unreduced]`, and
`scripts/comparison/adapters.py::_khocaGroups` takes `out[1]` and **discards `out[0]`**.
The discarded half for the trefoil, both coefficient rings, in KnotInfo's q-convention:
(0,2), (2,6), (3,8) -- identical to what kht++ prints.

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

1. **khoca** -- already computes the reduced half and throws it away. No new
   provisioning at all.
2. **knotkit** -- `kk kh -r -f {Q,Z2}` works; costs LaTeX parsing.
3. **kht++** -- the most natural fit mathematically, and the most expensive, because it
   needs a PD-to-Morse-word encoder first.

## Vocabulary warning

"Reduced" means three unrelated things in this repo, and conflating them is the most
likely way someone wires the wrong thing:

- here and in `coverage-map.md`: the **basepoint-reduced theory**;
- in `homology-engine.md` section 4 / Phase 4 and ADR 0007, `raw == reduced`: the
  **Gaussian-cancelled** complex, an optimization;
- in `four-manifold-objectives.md`: the **reduced filtered complex** for CFK-infinity.
