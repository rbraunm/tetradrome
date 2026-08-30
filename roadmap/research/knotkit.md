# Research: knotkit (`kk`) as a measurement oracle and validator candidate

Empirical CLI discovery for the `knotkit` binary, provisioned by
`scripts/install_oracles.sh` but not yet in the comparison roster or the validator
registry. Feeds checkpoints 7g (measurement adapter) and 7h (validator).

Everything below was run in the sandbox container against the provisioned binary.
Claims are marked **verified** (observed output) or **hypothesis** (derived, needs the
full chiral sweep before anything is wired). Per the standing doctrine, nothing here is
wired until a convention probe verifies it.

## Provisioning

- Wrapper `/usr/local/bin/kk`; built from source by `install_oracles.sh`, version
  derived as a git sha (`git:4fdd08f` on this run).
- The install script's own smoke gate is `kk s "T(2,3)"` asserting the trefoil `s` is
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
- **`-r`**: compute the reduced theory.
- `kk` with no arguments errors with `too few arguments, <invariant> or <knot> missing`;
  `kk -h` prints full usage.

### Input formats (verified)

Accepts, among others, **our PD shape verbatim** -- `PD[X[1,4,2,5],X[3,6,4,1],X[5,2,6,3]]`
parses with no translation -- plus Rolfsen names (`3_1`, `10_124`), DT codes, braid
words, and torus specs (`T(2,3)`). The PD path is the one a validator would use, since
our knots carry PD.

### Output formats (verified)

- `kk s` writes plain text to stdout: `s(3_1; Z2) = 2`. Trivially parseable.
- `kk kh` writes a **`.tex` file**, not stdout. `kk kh -r -f Q 3_1` produces LaTeX
  containing a `\widetilde{Kh}(3_1; Q)` heading and a `\rank Kh = 3` line. Parsing this
  is the substantive work in 7g, and is why 7g/7h are split `s` first, `kh` second.

## The chirality finding (verified, and the reason to probe on the PD path)

The same knot gives opposite signs depending on which input path is used:

| invocation | result |
|---|---|
| `kk s 'PD[X[1,4,2,5],X[3,6,4,1],X[5,2,6,3]]'` | **-2** |
| `kk s 3_1` | **+2** |

That PD is the right-handed trefoil. Our own `compute(knot('3_1'), 'rasmussen_s')`
returns **-2**.

So on this single data point, **knotkit's PD path agrees with native directly** (no
transform) while its **name path is mirrored** relative to it. This is the opposite
shape from knotjob, which needs the full mirror.

**Hypothesis, not a result**: the PD path transform is DIRECT. One knot is not a probe.
7h must run the standard chiral sweep (3_1 / 4_1 / 5_2 / 8_19 / 10_124) **through the PD
path**, because a probe run through the name path would verify a convention no validator
will ever use and would invert the sign of every wired result.

## Sweep values (verified, name path, `-f Z2`)

| knot | `kk s` (name path) |
|---|---|
| `3_1` | 2 |
| `4_1` | 0 |
| `5_2` | 2 |
| `8_19` | 6 |
| `10_124` | 8 |

Recorded for reference only. These are name-path values; see the chirality finding above.

## Coefficient field

`s` over `Q` and over `Z2` agreed on both knots tested (`3_1` -> 2, `10_124` -> 8).
They are not equal in general, so 7h must state which field the wired value comes from
rather than treating them as interchangeable. `Z2` is knotkit's default.

## Why this oracle is worth the work

`rasmussen_s` currently has exactly one wired validator (knotjob). A verified knotkit
`s` is the second, and it arrives through an input path that needs no PD translation.
Its `kh` additionally covers `khovanov_homology` (F2) and `rational_khovanov_homology`
(Q), so knotkit is a three-invariant candidate rather than the single-invariant one the
earlier planning assumed.
