# Research: knotkit (`kk`) as a measurement oracle and validator candidate

Empirical CLI discovery for the `knotkit` binary, provisioned by
`scripts/install_oracles.sh` but not yet in the comparison roster or the validator
registry. Feeds checkpoints 7g (measurement adapter) and 7h (validator).

Everything below was run against the provisioned binary. Claims are marked
**verified** (observed output) or **hypothesis** (needs the full chiral sweep before
anything is wired). Per the standing doctrine, nothing here is wired until a convention
probe verifies it.

**Probe hygiene.** Every probe must feed `kk` the PD from `knots.from_name(name).pd_code`
-- the exact diagram native computes on. PD literals copied from elsewhere (the install
script's smoke test, KnotTheory examples) can be the mirror of ours, and a probe run on
one silently inverts every sign it reports.

## Provisioning

- Wrapper `/usr/local/bin/kk`; built from source by `install_oracles.sh`, version
  derived as a git sha (`git:4fdd08f`).
- The install script's smoke gate is `kk s "T(2,3)"` asserting the trefoil `s` is
  plus or minus 2, so the s-invariant path is exercised at provisioning time.

## CLI surface (verified)

```
kk <invariant> [options] <link>
```

- **Invariants**: `kh`, `gss`, `lsss`, `sq2`, `leess`, `s`.
  Only `kh` and `s` map onto canonical invariant names (ADR 0001); `gss`, `lsss`,
  `sq2` and `leess` are spectral-sequence and Steenrod refinements with no canonical
  row and are out of scope.
- **`-f <field>`**: `Z2` (default), `Z3`, `Q`. `Z3` has no canonical row.
- **`-r`**: compute the reduced theory (no canonical row until
  `homology-engine.md` section 7 Phase 9).
- `kk` with no arguments errors with `too few arguments, <invariant> or <knot> missing`;
  `kk -h` prints full usage.

### Input formats (verified)

Accepts **our PD shape verbatim** -- the bracket string
`PD[X[a,b,c,d],...]` built from `knot.pd_code` parses with no translation -- plus
Rolfsen names (`3_1`, `10_124`), DT codes, braid words, and torus specs (`T(2,3)`).
The name path and the PD path give identical `s` values across the whole chiral sweep,
so knotkit's Rolfsen table and our PD agree on chirality.

### Output formats (verified)

- **`kk s`** prints one line to stdout: `s(<input>; Z2) = 2`. Trivially parseable.
- **`kk kh`** prints a **standalone LaTeX document to stdout** (no file is written):
  a `\rank Kh = N` line followed by a TikZ grid. The grid's axis labels map cell
  centres to gradings -- x centre `i + 0.5` carries a `t` label, y centre `j + 0.5` a
  `q` label -- and each generator is a `\fill (x, y) circle (.15);` at its cell centre.
  Rational trefoil example: fills at (0.5,0.5), (0.5,1.5), (2.5,2.5), (3.5,4.5) decode
  through the labels to `(t,q)` = (0,1), (0,3), (2,5), (3,9), with `\rank Kh = 4`.
  A cell of **rank N > 1** is drawn as `\draw (x, y) node {$N$};` at its centre instead
  of a circle (seen on 6_1, 7_4, 8_20, 9_42 and 10_132; 7_4 over Z2 has rank-3 cells).
  The measurement parser (`_parseKnotkitGrid` in `scripts/comparison/adapters.py`)
  accounts for every line of the picture, raises on any element it does not recognise,
  and requires the decoded ranks to sum to kk's own `\rank Kh` total.

## Conventions

**`s` -- verified on the full chiral sweep, both fields.** Fed our PD, knotkit's `s` is
the negation of native's: `s -> -s`, the same convention as knotjob.

| knot | native | `kk s -f Z2` | `kk s -f Q` |
|---|---|---|---|
| `3_1` | -2 | 2 | 2 |
| `4_1` | 0 | 0 | 0 |
| `5_2` | -2 | 2 | 2 |
| `8_19` | -6 | 6 | 6 |
| `10_124` | -8 | 8 | 8 |

Negation matched 10/10 cells; direct matched 2/10 (only the amphichiral `4_1`, which
cannot discriminate). Wired as a validator over Q (`backends/knotkit_adapter.py`),
since native `s` is Rasmussen's original, read off Lee homology over Q.

**`kh` -- verified on the chiral sweep plus 6_1 and 7_4, both fields.** Knotkit's
Khovanov is the full mirror of native's, `(h,q) -> (-h,-q)`, consistent with the `s`
result: full mirror matched 14/14 cells, direct 2/14 (only amphichiral `4_1`), and
q-negation 0/14. 6_1 and 7_4 were added to the sweep so the rank > 1 cells are
exercised in both fields. Measured in the benchmark; not wired as a validator, because
Khovanov over F2 and Q already has three independent validators and each added one runs
inside every strict compute (the wiring standard in `backends/registry.py`).

## Coefficient field

`s` over Z2 and over Q agreed on every knot in the sweep. They are not equal in general,
so the wired value must name the field it comes from rather than treating them as
interchangeable. Z2 is knotkit's default.

## Why this oracle is worth the work

`rasmussen_s` currently has exactly one wired validator (knotjob). A verified knotkit
`s` is the second, and it needs no PD translation. Its `kh` additionally covers
`khovanov_homology` (Z2) and `rational_khovanov_homology` (Q), so knotkit is a
three-invariant oracle.
