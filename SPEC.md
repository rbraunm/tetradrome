# Tetradrome Specification

Tetradrome is a Python workbench that computes the invariants of smooth 4-dimensional
topology natively, validates every result against independent tools, and records how each
result was produced.

Engine design is in `roadmap/design/homology-engine.md` and `roadmap/design/scheduler.md`.

## 1. Scope

```text
knot / link / braid input
  -> normalized diagram
  -> native invariant engine, run as scheduler jobs
  -> validation against computed oracles and KnotInfo
  -> result with provenance
  -> report with a stated claim strength
```

| | Implemented | Planned |
|---|---|---|
| Inputs | KnotInfo name, PD code, braid word | DT code, Gauss code, links, surgery / Kirby descriptions |
| Invariants | determinant, signature, Alexander, Jones, Khovanov over F2 and ℚ, Rasmussen *s*, HFK-hat, three-genus, τ | the rest of section 7; `roadmap/design/coverage-map.md` |
| Outputs | `InvariantResult` (section 5) | reports, the content-hashed roster export, concordance and trace surfaces (section 8) |

Where the plans are:

| Plan | Document |
|---|---|
| Engine phases, their state and open work | `roadmap/design/homology-engine.md` section 7 |
| KnotInfo columns: computed, planned, out of reach | `roadmap/design/coverage-map.md` |
| 4-dimensional constructions | `roadmap/design/four-manifold-objectives.md` |
| Status of every correctness claim | `roadmap/claim-ledger.md` |

## 2. Principles

1. **Native compute.** Every returned value is computed by Tetradrome's own code. No external
   topology tool computes a returned value or is needed to obtain one; they only check answers
   (section 6). There is no `backend=` argument: the invariant name selects the engine.
2. **Faithful.** The raw computation is the reference and is always runnable. Only exact,
   answer-preserving reductions may optimize it. No heuristics: no truncated gradings, no
   assumption of thinness or alternation, no probabilistic rank, no early-termination guess
   (`roadmap/design/homology-engine.md` section 4).
3. **Agreement.** Every reduction tier and every scheduled form returns the identical answer
   as its reference (`roadmap/design/homology-engine.md` section 4).
4. **One way to run.** Every calculation is a scheduler job (`roadmap/design/scheduler.md`).
5. **Validated by default.** `strict` requires a computed oracle wherever one exists, and any
   oracle that runs and disagrees raises (section 6).
6. **Provenance.** No bare values. Every result records its method, the versions that can
   move it, and each validator's identity, version and verdict (section 5). Provenance is
   never dropped for size. Dependencies are updated, not pinned: the run records the version
   it used.
7. **Fail loud.** No silent fallback, default or coercion. A calculation that will not fit is
   refused before it starts (`roadmap/design/homology-engine.md` section 5).
8. **General objects.** The scope is the toolset of smooth 4-dimensional topology, not a knot
   calculator. Knots are validated first, but no representation may assume a single
   component. Links and manifold inputs are built when an invariant needs them.
9. **Specialize the dataset, not the mathematics.** No engine contains anything specific to
   one knot.
10. **Claim strength.** A reported claim is a computation, an obstruction or a theorem
    reference, never a proof. A nonzero obstruction can show a knot lacks a property; a
    vanishing one shows nothing (`docs/conventions.md`).
11. **Speed is data, never a gate.** Correctness, auditability and portability outrank it.

## 3. Non-goals

- Proving theorems, or claiming new mathematics without expert review.
- Replacing the existing tools. They are the validation reference.
- Calling a knot slice because an obstruction vanishes.
- Requiring a GPU or an external tool to compute.
- A graphical interface or a diagram editor.

## 4. Architecture

```text
src/tetradrome/
  knots.py        public input surface
  errors.py       public error types
  diagrams/       NormalizedDiagram; PD parsing; construction from name, PD and braid;
                  Seifert structure of a PD
  invariants/     compute() (dispatch, scheduling, validation, provenance); the result schema;
                  the Seifert-form engine; Jones
  engines/        front ends, one per theory: cube.py, khovanov/, floer/
  algebra/        shared graded-complex back end and its reduction tiers
  scheduler/      runs every calculation as jobs
  backends/       validators only: knotinfo_backend.py, registry.py, one adapter per oracle

scripts/          install_oracles.sh, provision_runner.py, comparison/ (generates BENCHMARKS.md),
                  benchmarks
tools/            ct_exec.py, generate_benchmarks.py
tests/            pytest; --heavy widens the validation rosters
roadmap/          design/, research/, claim-ledger.md
docs/             conventions, validation, backend_matrix, existing_tools, conway_notes, outreach
```

- Everything downstream of `diagrams/` consumes a `NormalizedDiagram`, never a tool's object.
- Engines are theories and differ in their mathematics. `algebra/` knows no theory.
  `backends/` never produces a returned value.
- What a calculation needs installed: the Seifert-form and cube engines use the standard
  library only. The scheduled Floer path needs numpy for its block kernels, uses Numba when
  it is installed, and raises `BackendUnavailable` without numpy; its serial pure-Python
  oracles need neither. `database_knotinfo` supplies the tabulated diagrams and answers.
- Planned packages: `concordance`, `traces`, `catalog`, `export`, `reports` (section 8).

## 5. Result schema

`src/tetradrome/invariants/schema.py`. All frozen dataclasses.

```python
class Provenance:
    backend: str                  # "tetradrome-native"
    backend_version: str
    method: str                   # "seifert_form_from_braid" | "kauffman_bracket" | "khovanov_cube_f2"
                                  # | "khovanov_cube_q" | "lee_quantum_filtration" | "grid_hfk_hat"
                                  # | "grid_filtered_tau" | "grid_hfk_genus" | "bordered_hfk_hat"
                                  # | "bordered_hfk_genus"
    inputs: str                   # "braid_word" | "knotinfo:braid_notation" | "pd_code"
    library_versions: tuple[tuple[str, str], ...]   # (name, version); the interpreter, then each
                                                    # computational library the caller declares
    presentation: GridSource | MorseSource | None   # Floer results: what the engine ran on. The
                                  # grid, the reduced diagram it was built from, "built" or
                                  # "stored", when, by which code, and its file
                                  # (engines/floer/grid_store.py); or the reduced diagram and
                                  # its Morse presentation (engines/bordered)
    engine_choice: str | None     # Floer results: "automatic" or "named"
    computation: ComputationRecord  # every scheduler job the calculation ran: kind (for a
                                    # reduction, the reducer), placement, CPUs, NUMA node, GPU,
                                    # seconds; the machine; the package source digest
                                    # (scheduler/session.py)

class ValidatorRecord:
    oracle: str                   # "knotinfo", "regina", "knot_floer_homology", ...
    version: str
    verdict: str                  # "pass" | "fail" | "not_run"

class ValidationStatus:
    validators: tuple[ValidatorRecord, ...]         # KnotInfo is one of them
    d_squared_check: str          # "pass" | "fail" | "not_applicable"; the native complex's own check
    is_validated: bool            # property: some validator passed
    has_disagreement: bool        # property: some validator failed
    def verdict(self, oracle: str) -> str: ...

class InvariantResult:
    knot: str | None              # canonical id (docs/conventions.md), or a label such as "(braid word)"
    invariant: str                # canonical name (section 7)
    value: Any
    provenance: Provenance
    validation: ValidationStatus
    raw: Any = None
```

| Invariant | `value` |
|---|---|
| `determinant`, `signature`, `rasmussen_s`, `ozsvath_szabo_tau`, `three_genus` | `int` |
| `alexander_polynomial` | tuple of ascending coefficients, shifted so the constant term is nonzero and positive |
| `jones_polynomial` | `(low_exponent, coefficients)` in *t*, ascending |
| `khovanov_homology`, `rational_khovanov_homology` | `{(i, j): dimension}` |
| `knot_floer_homology` | `{(Maslov, Alexander): rank}` |

Planned: a reported result also carries `claim`, `strength`
(`"computation" | "obstruction" | "theorem_reference"`) and `limitations`.

## 6. Validators

`src/tetradrome/backends/registry.py`.

```python
class Validator(Protocol):
    name: str
    covered_invariants: set[str]

    def is_available(self) -> bool: ...
    def version_info(self) -> dict: ...
    def known_value(self, knot, invariant: str) -> Any | None: ...
```

- A validator confirms or contradicts a native value. It never produces the value a caller
  receives.
- `known_value` returns `None` when the validator cannot check this input; the record is
  `not_run`. A blank or sentinel cell in KnotInfo is "cannot check", never a default value.
- The registry holds the wired validators in consultation order and, in `_UNWIRED`, the
  computed oracles that exist but are not wired. Wiring standard: at least two independent
  computed validators per invariant wherever two exist.
- KnotInfo is recorded as one more validator. It validates alone only where no computed
  oracle exists for the invariant.

`compute(..., validate=...)` takes a mode. There is no boolean form.

| Mode | Behaviour |
|---|---|
| `strict` (default) | A computed oracle is required wherever one exists for the invariant, installed on this host or not. A missing one raises `UnvalidatedResult`, which says whether it is not installed (`scripts/install_oracles.sh`) or not wired. KnotInfo is consulted as well. |
| `soft` | Uses the computed oracle when it is installed. Otherwise falls back to KnotInfo and logs that `strict` would have raised. |
| `off` | No validation. The result still carries its provenance and an empty validation record. |

In `strict` and `soft` an oracle that runs and disagrees raises, and a result with no oracle
agreeing raises `UnvalidatedResult` instead of being returned.

| Topic | Where |
|---|---|
| What each oracle computes and how it is provisioned | `docs/backend_matrix.md`, `scripts/install_oracles.sh` |
| Tool licences | `docs/existing_tools.md` |
| The benchmark's oracle roster (measurement, not validation) | `scripts/comparison/adapters.py` |

## 7. Canonical names

This table is the list of canonical names, beside the KnotInfo column and the
`knot_floer_homology` key for each. A name is canonical when `compute()` accepts it or the
table reserves it for an engine not yet built; `tests/test_canonical_names.py` pins the Native
column against `compute()`.

- Names are `lower_snake_case`, use the term the literature uses, and take the attributed form
  where the invariant is standardly attributed. `rasmussen_s` keeps the letter the literature
  writes.
- F2 is the unmarked coefficient field and every other ring is named: `khovanov_homology` is
  over F2, `rational_khovanov_homology` over ℚ.
- Every other validator's spelling lives in its adapter. Renaming a canonical name is a change
  across `src/`, `scripts/` and `tests/`.
- Knot identity and diagram notation are in `docs/conventions.md`.

| Tetradrome (canonical) | Native | KnotInfo column | `knot_floer_homology` key |
|---|---|---|---|
| `alexander_polynomial` | yes | `alexander_polynomial` | — |
| `jones_polynomial` | yes | `jones_polynomial` | — |
| `signature` | yes | `signature` | — |
| `determinant` | yes | `determinant` | — |
| `arf_invariant` | — | `arf_invariant` | — |
| `three_genus` | yes | `three_genus` | `seifert_genus` |
| `smooth_four_genus` | — | `smooth_four_genus` | — |
| `topological_four_genus` | — | `topological_four_genus` | — |
| `rasmussen_s` | yes | `rasmussen_invariant` | — |
| `ozsvath_szabo_tau` | yes | `ozsvath_szabo_tau_invariant` | `tau` |
| `epsilon` | — | `epsilon` | `epsilon` |
| `nu` | — | `nu` | `nu` |
| `fibered` | — | `fibered` | `fibered` |
| `l_space_knot` | — | `l_space` | `L_space_knot` |
| `khovanov_homology` (F2) | yes | `khovanov_unreduced_integral_vector` (mod 2 by universal coefficients) | — |
| `rational_khovanov_homology` | yes | `khovanov_unreduced_integral_vector` (free part) | — |
| `knot_floer_homology` | yes | `hfk_polynomial_vector` | `ranks` / `total_rank` |
| `smoothly_slice` | — | none; slice exactly when `smooth_four_genus` is 0 | — |
| `topologically_slice` | — | none; slice exactly when `topological_four_genus` is 0 | — |

## 8. Public API

Implemented:

```python
import tetradrome as td

td.knots.from_name(name)                      # "K11n34", "4_1"
td.knots.from_pd(pd_code, identity=None)
td.knots.from_braid(word, identity=None)      # off-table input
td.invariants.compute(knot, invariant, validate="strict", engine="auto") -> InvariantResult
```

`engine` applies to the Floer invariants. `"auto"` takes the bordered engine wherever it computes
the invariant (HFK-hat, three-genus) and the grid engine otherwise (τ). `"grid"` runs the grid
engine on any knot. `"bordered"` raises for an invariant it does not compute, and any engine but
`"auto"` raises for a non-Floer invariant.

Errors, all subclasses of `TetradromeError`: `UnknownKnot`, `BackendUnavailable`,
`UnvalidatedResult`. Planned: `ConventionMismatch`, `ExportHashMismatch`.

Planned, not implemented:

```python
knots.from_dt(dt); knots.from_gauss(code); knots.normalize(knot); knots.mirror(knot)

invariants.list()
invariants.compute_all(knot, *, validate="strict") -> dict[str, InvariantResult]

concordance.obstruction_profile(knot) -> ObstructionProfile
concordance.slice_status(knot) -> SlicenessVerdict

traces.trace(knot, framing=0) -> KnotTrace
traces.same_trace(a, b, framing=0) -> bool
traces.siblings(knot) -> list[TraceSibling]
traces.slice_certificate(knot) -> Optional[SliceCertificate]

catalog.names(); catalog.get(name)

export.build(names=None, *, validate="strict") -> RosterExport
export.save(roster, path); export.load(path) -> RosterExport   # verifies content_hash

validators.available(); validators.capabilities(); validators.require(name)
```

```python
class ObstructionOutcome:
    name: str                     # "fox_milnor" | "signature" | "arf_invariant"
                                  #   | "rasmussen_s" | "ozsvath_szabo_tau"
    obstructs: bool               # False: it vanishes
    value: Any
    provenance: Provenance

class ObstructionProfile:
    knot: str
    outcomes: dict[str, ObstructionOutcome]
    all_vanish: bool

class SlicenessVerdict:
    knot: str
    smoothly_slice: Optional[bool]
    topologically_slice: Optional[bool]
    smooth_four_genus: Optional[int]
    topological_four_genus: Optional[int]
    obstruction_profile: ObstructionProfile
    certificate: Optional[SliceCertificate]

class KnotTrace:
    knot: str
    framing: int                  # 0: the 0-trace X_0(K)
    boundary: str                 # the n-surgery S^3_n(K)
    description: str              # Kirby / RBG handle description

class TraceSibling:
    a: str
    b: str                        # diffeomorphic traces: X(a) = X(b)
    framing: int
    source: str                   # construction or reference

class SliceCertificate:
    knot: str
    via: str                      # "direct" | "trace_sibling" | "topological"
    sibling: Optional[str]        # K' with the shared trace, when via == "trace_sibling"
    witness: dict                 # e.g. {"rasmussen_s(K')": 2}
    references: list[str]

class RosterEntry:
    knot: NormalizedDiagram
    invariants: dict[str, InvariantResult]    # keyed by canonical name
    sliceness: SlicenessVerdict
    trace_siblings: list[TraceSibling]

class RosterExport:
    version: str
    content_hash: str
    built_at: str
    backend_versions: dict
    validation_summary: dict
    entries: dict[str, RosterEntry]           # keyed by canonical id
```

**The consumption contract.** A downstream consumer depends on one artifact: a `RosterExport`
built offline by `export.build(..., validate="strict")`, saved and content-hashed. Reading it
touches no tool. Every value in it carries its `Provenance` and `ValidationStatus`, so a
consumer can require validation and refuse to proceed without it.

**Reports.** Planned. A report states, for each result, the schema of section 5 plus the
claim, its strength and its limitations. No number appears without its validation context.

## 9. Licensing

Apache 2.0.

- All first-party code is original. No GPL or other third-party source is copied or adapted
  into the tree.
- No third-party data is in the tree. `database_knotinfo` (GPL) is the one declared runtime
  dependency; it is read as data and cited.
- GPL tools are optional validators: installed by the user, imported in the validation path
  or run as separate programs, never bundled.
- Because no GPL source or data is distributed inside Tetradrome, no copyleft obligation
  attaches.
- Tool licences are recorded in `docs/existing_tools.md`.

## References

Lisa Piccirillo, "The Conway knot is not slice," *Annals of Mathematics* 191(2), 2020;
arXiv:1808.02923. The motivating example: `docs/conway_notes.md`.
