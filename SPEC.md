# Tetradrome Project Specification

**Working title:** Tetradrome  
**Project type:** Python-first computational workbench for the invariants of smooth 4-dimensional topology — knots, links, and braids, not knots alone  
**Strategic focus:** Native, faithful computation of the invariants on one validated, auditable surface, with existing tools used only as opt-in validators (decisions 0006, 0007, 0009)  
**Initial technical focus:** Conway-adjacent Khovanov / Lee / Rasmussen invariants, computed natively, validated against KnotInfo and cross-checked against existing tools  
**Long-term direction:** Extensible library that consolidates the smooth-4D-topology tool sprawl — diagrams compiled into graded chain complexes, processed by exact-algebra engines, validated against known results  
**Status:** Architecture specification. The strategic framing was revised from the earlier orchestrate-first survey to native-first; where older orchestration language survives below it is being reconciled, and `roadmap/decisions/` (esp. 0006, 0007, 0009) and `roadmap/design/homology-engine.md` govern on any conflict.  

---

## 1. Executive Summary

Tetradrome is a Python-first computational workbench for building, validating, reproducing, and reporting the invariants of smooth 4-dimensional topology — the Rasmussen \(s\)-invariant, \(\tau\), \(\epsilon\), \(\nu\), the classical and concordance invariants, and the Khovanov / Lee / knot Floer homologies they are read from — for knots, links, and braids on one normalized, fully-provenanced surface.

Serious tooling for these invariants already exists (Khovanov via KnotJob / Khoca, knot Floer via Szabó's HFK calculator, diagrams via Spherogram / SnapPy, the KnotInfo database). Tetradrome does not pretend otherwise, and it credits and cross-checks against those tools. But it does not orchestrate them to produce answers: **the compute path is native** (decision 0006). Each invariant is computed by Tetradrome's own code, and the existing tools are used only as opt-in validators — the role KnotInfo already plays — never as a runtime dependency of a computation.

That choice is deliberate and is the project's value proposition. Much of the existing software is research-tool shaped: mathematically serious and often fast, but bound to older ecosystems, Java / Mathematica / C++ packaging, Sage-specific environments, GPL licensing, or binary-wheel install paths that fail wherever no prebuilt wheel exists. A native, permissively-licensed, pure-Python core that runs anywhere — and that owns its kernel well enough to accelerate it (JIT, multi-core / NUMA, optional GPU) — consolidates that sprawl into one auditable place. The raw, faithful computation is first-class and always runnable; only exact, answer-preserving reductions optimize it; no heuristics enter the core (decision 0007).

The immediate goal is a **Conway-adjacent reproducibility and validation pipeline**:

```text
knot / link / braid input
  -> normalized diagram representation
  -> native invariant computation
  -> validation (KnotInfo oracle, cross-checks, d^2 = 0)
  -> reproducible report
  -> carefully limited 4D-topology interpretation
```

The longer-term goal is broader: Tetradrome should become a modular library where diagrams are compiled into graded chain complexes, processed by exact-algebra engines, validated against known results, and extended across the invariants that bear on smooth 4D structure — consolidating tools that today live in separate, differently-packaged ecosystems.

The guiding phrase is unchanged in spirit, sharpened in target:

> Use existing instruments honestly — as the gold-master check — and own the mathematics we compute. Faithful and portable beats fast-but-won't-install, and a validated number is the only kind we report.

---

## 2. Background and Motivation

A knot is drawn in 3-dimensional space, but some of the most interesting questions about that knot concern what it can bound in 4 dimensions. For example, a knot is slice if it bounds a smoothly embedded disk in the 4-ball \(B^4\). The Conway knot was historically significant because its smooth sliceness resisted classification until Lisa Piccirillo proved that the Conway knot is not slice. Her Annals of Mathematics paper states that this completed the classification of slice knots under 13 crossings and produced the first example of a non-slice knot that is both topologically slice and a positive mutant of a slice knot.[^piccirillo-annals]

Tetradrome is motivated by the idea that these 4-dimensional questions require a kind of mathematical measuring apparatus. The apparatus is not a physical device. It is an algebraic and computational system: knot diagrams, chain complexes, differentials, gradings, homology calculations, known-answer tables, and carefully limited interpretations of the resulting invariants.

The immediate need is not to pretend no apparatus exists; it is to build that apparatus as one disciplined, native engineering layer:

- compute the invariants natively, in portable Python;
- normalize inputs and outputs into one schema;
- record conventions, assumptions, and versions;
- validate against KnotInfo and cross-check against independent tools;
- produce reproducible reports;
- keep the raw, faithful computation first-class and accelerate it (JIT / multi-core / GPU) without ever changing its answers.

---

## 3. Project Identity: What Is a Tetradrome?

The word **Tetradrome** is used here as a project name for a computational instrument aimed at smooth 4-dimensional topology.

Conceptually:

- An **orrery** models celestial motion.
- An **armillary sphere** models astronomical reference frames.
- A **tetradrome** models the algebraic constraints on how knots and links may behave when considered through 4-dimensional topology.

In software terms, Tetradrome is a workbench-and-instrument architecture — and the instruments are native:

```text
knot / link / braid input
  -> normalized representation
  -> native invariant engine
  -> validated invariant result
  -> reproducibility metadata
  -> report and claim ledger
```

For the homological invariants the engine is a chain-complex compiler:

```text
diagram
  -> combinatorial state space
  -> graded chain complex
  -> exact sparse algebra
  -> homology / invariant
  -> carefully limited 4D-topology interpretation
```

The priority is to build the **workbench** — the normalized, validated surface and the native engines behind it — and to grow the set of instruments incrementally, validating each against known results before it is trusted.

---

## 4. Existing Tooling Landscape

The field already has meaningful computational tooling. Tetradrome acknowledges it explicitly and uses it as the validation reference — the gold master a native result is checked against — not as a compute backend (decision 0006).

| Tool / ecosystem | Area | What it provides | Relevance to Tetradrome |
|---|---|---|---|
| **KnotTheory` / Knot Atlas** | Mathematica / knot tables / Khovanov | KnotTheory` was a main tool used to produce the Knot Atlas. Knot Atlas documents Khovanov computations and mentions FastKh and JavaKh backends, with JavaKh supporting different coefficient choices. | Historical reference and Khovanov cross-check; Tetradrome claims no novelty for basic Khovanov computation, only a native, portable implementation of it. |
| **FastKh / JavaKh** | Khovanov computation | Khovanov-focused computational engines associated with KnotTheory` / Knot Atlas. Knot Atlas notes JavaKh programs are much faster than the Mathematica implementation. | Potential external backend or validation reference. |
| **KnotJob** | Java knot homology software | Computes several knot invariants, including Khovanov-related data and Rasmussen-style invariants. KnotInfo notes that its displayed Khovanov homology invariants were calculated using Dirk Schütz's KnotJob and reformatted by Jason Garcia. | Primary validator for Khovanov data and the Rasmussen-style \(s\)-invariant. |
| **SageMath knot tools** | Python-based computer algebra ecosystem | Sage provides link/knot objects, knot diagrams, invariant functionality, and optional access to KnotInfo data. Its documentation notes support for planar diagrams, braids, Gauss codes, and a connection to KnotInfo / LinkInfo. | Useful Python-adjacent ecosystem. Tetradrome should interoperate where possible rather than duplicate all infrastructure. |
| **SageMath chain-complex tools** | Algebra / homology | Sage includes chain-complex and homology functionality over appropriate rings/fields. | Useful reference for algebra validation and exact homology work. |
| **Khoca** | Khovanov-Rozansky homology | C++/Python research program for computing certain Khovanov-Rozansky homologies of knots. | Demonstrates that advanced homology computation already exists; potential reference for higher homology directions. |
| **HFKcalc** | Knot Floer homology | C++11 program by Ozsváth/Szabó ecosystem using planar diagrams for knots and computing knot Floer data modulo a prime. | Important reference for Floer-side computation; possible backend or validation reference. |
| **knot-floer-homology PyPI package** | Python wrapper for HFKcalc | Python package wrapping Zoltán Szabó's HFK Calculator. Accepts PD input and Spherogram links; returns knot Floer data such as ranks, \(\tau\), \(\epsilon\), \(\nu\), fibered status, genus, and related data. | Floer-side validator. Rejected as a compute dependency — GPLv2+, binary-wheels-only with no sdist, and not our math (decision 0006); used only to cross-check the native Floer engine. |
| **Spherogram / SnapPy** | Planar diagrams and 3-manifold topology | Python module in the SnapPy ecosystem for planar diagrams arising in 3-dimensional topology, including links and Heegaard diagrams. It can create links programmatically and return PD codes and DT codes. | Diagram-format reference and optional interop bridge; diagram handling itself is native. |
| **KnotInfo** | Knot invariant database | Database of knot invariants, downloadable data, polynomial invariants, Khovanov variants, Heegaard Floer polynomial data, and references. Its about page notes HFK invariants computed using HFKcalc and Khovanov data calculated using KnotJob. | Validation oracle and known-answer source. Tetradrome should integrate it as a reference dataset. |
| **pyknotid** | Knots as 3D curves | Python package for knots and links represented as three-dimensional space curves, including diagram generation and topological analysis. | Adjacent utility for geometric/visual input, not likely central to the first Conway/Khovanov pipeline. |

### Strategic conclusion

Tetradrome computes its invariants itself and validates them against these tools:

```text
Tetradrome = native computation + normalization + validation + reporting + hardware-adaptive acceleration
```

not:

```text
Tetradrome = orchestrate existing tools to produce the answer
```

and not:

```text
Tetradrome = pretend existing knot software does not exist
```

### 4.1 Why native, not orchestration

The existing tools encode years of mathematical work and are treated with respect — as validation references, not discarded competitors. But the engineering case for owning the compute path is decisive (decisions 0006, 0007):

- **Portability.** A pure-Python core runs anywhere Python runs — no compiler, no JVM, no Sage environment, no GPL, no binary-wheel platform lottery. "Runs at all" beats "faster but won't install."
- **License.** Tetradrome is Apache-2.0; several of the strongest tools are GPL. As validators — separate programs or optional installs — that is fine; as bundled compute dependencies it would not be.
- **Our math.** A native result is one we computed and can audit end to end, not an opaque number from someone else's binary.
- **The kernel is the opportunity.** The compute kernel — exact sparse linear algebra over a cube of resolutions — is exactly what we can own and accelerate (JIT, multi-core / NUMA, optional GPU). It is the value, not a cost to outsource.

Owning the kernel still demands the engineering discipline the modernization framing called for — installability, Python ergonomics, reproducibility, raw-output capture, cross-tool comparison, test coverage, exact-algebra performance, documentation for non-specialist engineers — now applied to our own code rather than to wrappers around others'.

The rule, restated for native-first:

> Existing tools are the gold masters. A native computation must match them on known examples before it is trusted — and the raw, faithful computation is the reference every optimization is checked against.

### 4.2 Why GPU Support Is a Real, but Narrow, Opportunity

GPU acceleration should not be marketed as magic topology acceleration. The likely force multiplier is much more specific: exact algebra over finite fields, packed-bit matrix operations, batched rank computations, and repeated validation runs. CuPy provides a NumPy/SciPy-like GPU array ecosystem, and Numba-CUDA provides a Python path for writing CUDA kernels; both are plausible implementation tools for the eventual acceleration layer.[^cupy][^numba-cuda]

The first GPU target should not be the symbolic topology logic. It should be the heavy algebra underneath native computations:

```text
chain complex / grading block
  -> sparse or packed GF(2) matrix
  -> rank / kernel / image computation
  -> homology dimension
```

The engineering principle remains, pointed at our own code (decision 0007):

```text
raw, faithful CPU computation (the reference)
  -> optimized CPU implementation
  -> GPU implementation
  -> raw / optimized / GPU agreement tests, cross-checked against the external gold masters
```

---

## 5. Revised End Goal

The end goal is an extensible, Python-first workbench for computational experiments in smooth 4-dimensional topology, spanning the invariants that constrain it — for knots, links, and braids.

A mature version of Tetradrome should support:

1. **Input and normalization**
   - knots, links, and braids
   - PD codes, DT codes, Gauss codes, braid words, and fixed catalogs (KnotInfo by name)
   - eventually grid and surgery descriptions

2. **Native invariant engines**
   - a Seifert-form engine (determinant, signature, Alexander) — done
   - a resolution-cube engine feeding Khovanov / Lee / Rasmussen \(s\)
   - a knot Floer engine (grid and/or HFK-cube)
   - a shared graded-complex back end with hardware-adaptive acceleration

3. **Validators (opt-in, never compute)**
   - KnotInfo as the known-answer oracle
   - KnotJob, Szabó's HFK calculator, SageMath, Khoca as cross-checks behind the §13.8 adapter contract

4. **Exact algebra back end**
   - finite-field and \(\mathbb{Q}\) (multimodular) graded chain complexes
   - sparse / packed-bit matrices over \(\mathbb{F}_2\)
   - a pure-Python reference reducer, then JIT / multi-core / optional GPU tiers, each validated against the reference

5. **Validation tooling**
   - \(d^2 = 0\) checks for native complexes
   - known-answer tests against KnotInfo
   - cross-checks against independent tools
   - reproducibility logs and a claim-status ledger

6. **Research-grade humility**
   - a clear distinction between computing an invariant, obstructing a property, and proving a property
   - explicit documentation of conventions and of what is not yet validated
   - no claims of new mathematics without expert review

The mature project should make it possible to say:

> For this diagram, under these coefficient and grading conventions, Tetradrome computed the relevant invariant natively, recorded its version and conventions, checked the raw computation against any exact reduction and against known examples or independent tools, and reported only the conclusions that invariant justifies.

---

## 6. Immediate Need

The immediate need is to build the **Tetradrome workbench**, not the entire universe of homological knot theory.

The first successful version should be a **Conway-adjacent workflow**, not a general-purpose theorem machine.

The first release should focus on:

- a small fixed catalog of knots, plus off-table input by PD / braid;
- normalized native diagram input;
- native computation of the classical invariants (determinant, signature, Alexander, Jones) — done — then the Khovanov / Lee / \(s\) path;
- known-answer validation against KnotInfo, with cross-checks where a tool is installed;
- reproducible report generation;
- a clear claim ledger;
- a generic core that the later homology engines extend without rework.

The immediate build should not attempt to do everything. It should build the extensible workbench correctly.

---

## 7. Revised Initial Mathematical Scope

### 7.1 Primary v1 target

The first mathematical workflow is:

```text
normalized diagram
  -> native Khovanov / Lee / Rasmussen s computation
  -> validation against KnotInfo (and cross-checks where available)
  -> reproducible Conway-adjacent report
```

The conceptual pipeline:

```text
Khovanov homology
  -> Lee deformation
  -> Rasmussen s-invariant
```

This is the most relevant first pipeline because Piccirillo's Conway-knot result uses a related knot with the same 4-dimensional trace and applies Rasmussen's \(s\)-invariant to obstruct sliceness.[^piccirillo-arxiv] \(s\) — read from Khovanov/Lee — is therefore higher priority than the Floer side: \(\tau\), \(s\), \(\epsilon\), and \(\nu\) all vanish on the Conway knot itself, and the obstruction came from \(s\) applied to its trace-sibling.

### 7.2 Floer-side target

The Floer side is a **native engine** (grid homology and/or the Szabó HFK cube), not a wrapper — `knot_floer_homology` is a validator only, never a backend (decision 0006). It is sequenced after the Khovanov / \(s\) path, since \(s\) is what bears on the Conway-adjacent question:

```text
diagram
  -> native knot Floer engine (grid / HFK-cube)
  -> tau, epsilon, nu, HFK ranks
  -> validation against KnotInfo, cross-checked against Szabó's HFK calculator where installed
```

### 7.3 Initial knot catalog

The initial catalog should include small and well-understood examples before Conway-adjacent examples:

```text
Validation knots:
  - unknot
  - trefoil
  - figure-eight knot
  - selected small knots with known Khovanov and Floer data

Conway-adjacent knots:
  - Conway knot, 11n34
  - Kinoshita-Terasaka knot
  - Piccirillo's related trace-equivalent knot K' if a reliable diagram/input encoding is included
```

The Conway knot should not be the first test case. It should be a later integration target after the machinery has passed simpler cases.

### 7.4 Deferred scope

Native Khovanov, Lee, and knot Floer are *sequenced* (see the roadmap phases), not deferred. What is intentionally deferred:

- general 3-manifold surgery calculations;
- arbitrary knot-database support beyond selected importers;
- polished diagram editing;
- a graphical user interface;
- automated theorem proving;
- claims of new mathematical results.

---

## 8. Non-Goals

Tetradrome v1 is **not** intended to:

1. prove new theorems;
2. deprecate or replace the existing tools — they remain Tetradrome's validation references;
3. pretend Khovanov / Floer tooling does not already exist;
4. claim that a knot is slice merely because an obstruction vanishes;
5. provide a complete implementation of every Khovanov variant;
6. provide a complete implementation of Heegaard-Floer theory;
7. hide conventions behind a black-box interface;
8. rely on GPU acceleration before the pure-Python reference is correct;
9. confuse a computed invariant with a proof, or a validator's number with our own computation.

The project should be built around the principle:

> Fast wrong math is worse than slow honest math; black-box correct math is still not enough unless the box, version, conventions, and validation path are recorded.

---

## 9. Architecture Overview

Repository layout (current and planned; `src/`-layout package):

```text
src/tetradrome/
  diagrams/                 # native diagram handling
    model.py                #   NormalizedDiagram, PDCode
    pd.py                   #   PD parse / normalize
    build.py                #   from_name / from_pd / from_braid construction
    seifert_construction.py #   PD -> oriented Seifert structure (signs, circles, writhe)
    braid.py                #   future: braid -> diagram
    grid.py                 #   future: grid diagrams

  engines/                  # front-end engines (per-theory machinery)
    cube.py                 #   resolution-cube skeleton (shared scaffold)
    khovanov/               #   future: Khovanov / Lee
    floer/                  #   future: grid / HFK-cube

  algebra/                  # future: shared graded-complex back end (acceleration lives here)
    complex.py              #   graded chain complex
    reduce_reference.py     #   pure-Python reference reducer
    reduce_f2_packed.py     #   packed-bit F2 (later)
    multimodular.py         #   Q via primes + CRT (later)
    memory.py / tiers.py    #   memory predictor + tier selector (later)

  invariants/
    schema.py               #   Provenance / ValidationStatus / InvariantResult
    seifert.py              #   determinant, signature, Alexander (Seifert form)
    jones.py                #   Kauffman bracket -> Jones
    compute.py              #   dispatch + validate-by-default
    khovanov.py / rasmussen.py / knot_floer.py   # future
    concordance.py / traces.py                   # future: slice status, trace machinery

  backends/                 # VALIDATORS only (never compute) -- §13.8 adapter contract
    knotinfo_backend.py     #   the known-answer oracle + name resolution
    knotjob_adapter.py      #   future: Khovanov / s cross-check
    hfk_adapter.py          #   future: Szabo HFK cross-check
    sage_adapter.py         #   future: optional interop

  export/                   # future: validated, content-hashed roster
  reports/                  # future: report templates

roadmap/                    # decisions (ADRs), design docs, milestones, research
docs/                       # conventions, validation, backend_matrix, conway_notes, outreach
tests/                      # known-answer + structural tests (pytest)
```

The split that matters: **engines** are theories (different mathematics), the **algebra** back end is invariant-agnostic and is where acceleration lives, and **backends/** holds validators, not compute.

---

## 10. Core Design Principle: Narrow Mission, Extensible Core

The v1 mission is narrow, but the core should be generic.

Good:

```python
k = td.knots.from_name("K11n34")
result = td.invariants.compute(k, "rasmussen_invariant")   # native, validated
report = td.report(result)
```

Also good — off-table, by braid:

```python
k = td.knots.from_braid([1] * 15)                          # T(2,15)
result = td.invariants.compute(k, "determinant", validate=False)
```

Bad:

```python
result = compute_conway_answer()
```

There is no `backend=` argument: the computation is native, and the invariant name selects the engine (decision 0006). Where a validator is installed it is consulted automatically as a cross-check, never to produce the value.

The code should be Conway-focused at the experiment level, not Conway-specific at the invariant level.

The key architectural rule:

> Specialize the dataset and experiment script, not the mathematics.

---

## 11. Result Schema

Tetradrome should normalize all backend outputs into a shared schema.

Example:

```yaml
knot:
  id: conway_11n34
  name: Conway knot
  input_format: name
  input_value: K11n34
  pd_code_hash: ...

computation:
  invariant: rasmussen_invariant
  backend: tetradrome-native
  backend_version: ...
  coefficient_field: ...
  grading_convention: ...
  timestamp_utc: ...

result:
  value: ...
  raw_output_ref: ...
  normalized_output: ...

validation:
  known_answer_match: pass/fail/not_available
  independent_backend_match: pass/fail/not_run
  d_squared_check: pass/fail/not_applicable
  notes: ...

interpretation:
  claim: ...
  strength: computation / obstruction / theorem_reference
  limitations: ...
```

The report should make it obvious whether a result came from:

- a wrapped external backend;
- a Tetradrome native implementation;
- a database lookup;
- a cross-check between multiple sources.

### 11.1 Typed in-memory view

The YAML above is the on-disk / report form. In memory the same schema is a small set of immutable dataclasses — the public types every computation returns. Field names mirror the YAML exactly, so it is one schema in two encodings.

```python
@dataclass(frozen=True)
class Provenance:                 # mirrors `computation:`
    backend: str                  # "tetradrome-native" for computed values; a cross-check validator is recorded in `validation`, not here
    backend_version: str
    method: str                   # how it was computed, e.g. "seifert_form_from_braid" | "kauffman_bracket"
    inputs: str                   # source of the diagram, e.g. "braid_word" | "knotinfo:braid_notation" | "pd_code"
    input_format: str             # "pd" | "dt" | "gauss" | "braid" | "name"
    input_value: str
    pd_code_hash: str
    coefficient_field: Optional[str]     # "F2" | "Q" | ...; None where N/A
    grading_convention: Optional[str]
    timestamp_utc: str
    raw_output_ref: Optional[str]

@dataclass(frozen=True)
class ValidationStatus:           # mirrors `validation:`
    known_answer_match: str              # "pass" | "fail" | "not_available"
    independent_backend_match: str       # "pass" | "fail" | "not_run"
    d_squared_check: str                 # "pass" | "fail" | "not_applicable"
    notes: str
    @property
    def is_validated(self) -> bool: ...  # True iff a known-answer or cross-backend check passed

Strength = Literal["computation", "obstruction", "theorem_reference"]   # mirrors `interpretation.strength`

@dataclass(frozen=True)
class InvariantResult:
    knot: str                     # canonical KnotInfo name, e.g. "K11n34"
    invariant: str                # canonical name (see §12.4)
    value: Any                    # typed per invariant
    provenance: Provenance
    validation: ValidationStatus
    claim: Optional[str] = None
    strength: Strength = "computation"
    limitations: Optional[str] = None
```

No result is returned as a bare value: each carries its provenance, its validation status, and the computation / obstruction / theorem-reference strength from §11's `interpretation` block.

---

## 12. Compute, Validators, and Vocabulary

### 12.1 Validator adapters

Computation is native; external tools are validators behind one adapter contract (§13.8). Each validator exposes a minimal, read-only interface:

```python
class Validator:
    name: str
    covered_invariants: set[str]

    def is_available(self) -> bool: ...
    def version_info(self) -> dict: ...
    def known_value(self, knot, invariant) -> Any | None: ...   # to cross-check a native result
```

A validator never produces the value a user receives; it only confirms or contradicts the native one (decision 0006).

### 12.2 Build and validator priorities

Native build order:

1. **Seifert-form engine** — determinant, signature, Alexander. Done.
2. **Resolution cube + Kauffman bracket / Jones.** Done.
3. **Shared graded-complex back end** + pure-Python reference reducer.
4. **Khovanov / Lee / Rasmussen \(s\)** on the cube.
5. **Acceleration tiers** (packed-bit F2 → JIT → NUMA → GPU), each validated against the reference.
6. **Knot Floer engine** (grid / HFK-cube).

Validator priority (opt-in, never compute):

1. **KnotInfo** — the known-answer oracle (already integrated).
2. **KnotJob** — Khovanov / \(s\) cross-check.
3. **Szabó's HFK calculator** — Floer cross-check.
4. **SageMath** — optional interop / cross-check.

### 12.3 Coverage matrix

The native column is the producer; the rest are validators that can cross-check it. Maintained in `docs/backend_matrix.md`.

| Invariant | Tetradrome native | KnotInfo (oracle) | KnotJob | HFK calc | Sage |
|---|---|---|---|---|---|
| determinant / signature / Alexander | done | yes | — | — | yes |
| Jones polynomial | done | yes | — | — | yes |
| Khovanov ranks | planned (M4) | yes | yes | — | — |
| Rasmussen \(s\) | planned (M5) | yes | yes | — | — |
| knot Floer ranks | planned (M8) | yes | — | yes | — |
| \(\tau\), \(\epsilon\), \(\nu\) | planned (M8) | partial | — | yes | — |

### 12.4 Vocabulary alignment with the referenced tools

The §12.3 matrix says *which* native engine produces an invariant and which tools can validate it. This table fixes the *names*. The left column is Tetradrome's canonical name — the standard term for the object in the knot-theory literature, chosen on the mathematics and independent of any tool. The remaining columns record how each tool happens to spell that same object, so one normalizer can read a validator's output for comparison. Listing a tool's spelling here is interop, not adoption: the right columns tell the normalizer how to read each validator; the canonical column stands on the mathematics. Where these coincide with KnotInfo's column names, it is because KnotInfo also uses the standard literature names — not because Tetradrome takes them from it.

The exact canonical spelling is a deliberate open design decision, localized to the normalizer (§13.3), not a commitment baked across the code. Where the literature gives more than one proper name for the same object — the symbol `tau` versus the attributed `ozsvath_szabo_tau`, or `three_genus` versus `seifert_genus` — either is mathematically legitimate; pick one in the normalizer and every result and export follows, with no effect on the mathematics or on any consumer.

| Tetradrome (canonical) | KnotInfo column | Sage `Knot`/`Link` | `knot_floer_homology` key | Spherogram / KnotJob |
|---|---|---|---|---|
| `alexander_polynomial` | `alexander_polynomial` | `.alexander_polynomial()` | — | Spherogram `.alexander_polynomial()` |
| `jones_polynomial` | `jones_polynomial` | `.jones_polynomial()` | — | — |
| `signature` | `signature` | `.signature()` | — | Spherogram `.signature()` |
| `determinant` | `determinant` | `.determinant()` | — | — |
| `arf_invariant` | `arf_invariant` | (Seifert matrix) | — | — |
| `three_genus` | `three_genus` | `.genus()` | `seifert_genus` | — |
| `smooth_four_genus` | `smooth_four_genus` | — | — | — |
| `topological_four_genus` | `topological_four_genus` | — | — | — |
| `rasmussen_invariant` | `rasmussen_invariant` | — | — | KnotJob (Khovanov / Lee → s) |
| `ozsvath_szabo_tau` | `ozsvath_szabo_tau` | — | `tau` | — |
| `epsilon` | (HFK data) | — | `epsilon` | — |
| `nu` | (HFK data) | — | `nu` | — |
| `fibered` | `fibered` | — | `fibered` | — |
| `l_space_knot` | (HFK data) | — | `L_space_knot` | — |
| `khovanov_homology` | `khovanov_*` | — | — | KnotJob (reduced / unreduced) |
| `knot_floer_homology` | `hfk_*` | — | `ranks` / `total_rank` | — |
| `smoothly_slice` | `smoothly_slice` | — | — | — |
| `topologically_slice` | `topologically_slice` | — | — | — |

**Identity.** A knot's canonical id is its KnotInfo name: Hoste–Thistlethwaite `K11n34` for ≥ 11 crossings — the form both `spherogram.Link('K11n34')` and Sage's `KnotInfo` enum accept — and Rolfsen `3_1` / `4_1` / `10_124` for ≤ 10.

**Diagram notation.** All four Spherogram / KnotInfo notations are first-class inputs and map name-for-name: PD code (list of 4-tuples) ↔ `pd_notation`; DT code (`DT[...]`) ↔ `dt_notation`; Gauss code ↔ `gauss_notation`; braid word (`braid_closure`) ↔ `braid_notation`.

**Bridges (opt-in interop).** `Knot.to_spherogram()` → `spherogram.Link`; `Knot.sage_link()` → Sage `Knot`; `Knot.exterior()` → SnapPy `Manifold`. These are convenience exports for users who want to hand a diagram to another tool; the native engines take the normalized diagram directly, and any Floer cross-check feeds the validator from the same PD.

---

## 13. Core Components

### 13.1 Diagram Layer

Responsible for:

- parsing knot, link, and braid inputs (name, PD, DT, Gauss, braid);
- parsing and normalizing PD codes;
- storing crossings and arcs;
- tracking orientation where needed;
- tracking crossing signs where needed;
- exposing diagram metadata;
- eventually supporting additional notations.

Initial deliverable:

```text
KnotDiagram
  crossings
  arcs
  orientation
  crossing_signs
  metadata
  source_notation
  canonical_hash
```

The diagram layer is native: it parses and normalizes the standard notations itself. Spherogram remains available as an optional interop target (§12.4), not as the parser.

### 13.2 Validator Layer

Responsible for (validators only — never the source of a returned value):

- consulting opt-in validators to cross-check a native result;
- parsing their raw outputs;
- recording versions;
- treating an absent validator as a skipped cross-check, not an error;
- normalizing results for comparison;
- storing raw outputs for auditability.

### 13.3 Invariant Schema Layer

Responsible for:

- defining canonical result formats;
- preventing backend-specific leakage into user reports;
- documenting conventions;
- making cross-backend comparison possible.

### 13.4 Native Khovanov Construction Layer

Deferred until the workbench exists.

Responsible for:

- enumerating cube-of-resolution states;
- resolving crossings;
- detecting circles in each resolution;
- assigning enhanced states;
- computing gradings;
- building differentials;
- verifying \(d^2 = 0\).

Initial coefficient target should be \(\mathbb{F}_2\), to avoid sign and torsion complexity in the first native version.

### 13.5 Native Lee / Rasmussen Layer

Deferred until native Khovanov basics are correct.

Responsible for:

- implementing Lee deformation after Khovanov basics are correct;
- handling filtered complexes;
- extracting Rasmussen's \(s\)-invariant under documented conventions;
- reporting when the computation is not sufficiently validated.

### 13.6 Algebra Layer

Responsible for:

- chain complexes;
- graded chain complexes;
- sparse matrices over \(\mathbb{F}_2\);
- rank computations;
- homology dimension calculations;
- CPU reference implementation;
- optional CUDA backend.

The algebra layer should not know anything about the Conway knot.

### 13.7 GPU / CUDA Acceleration

CUDA should be treated as an acceleration tier, not as the source of mathematical truth.

Potential CUDA/JIT acceleration points:

- packed-bit matrix operations;
- sparse \(\mathbb{F}_2\) rank calculations;
- batch grading-block calculations;
- validation runs across many knots;
- possible future state enumeration for native engines.

Development order:

```text
clear CPU reference
  -> tested CPU implementation
  -> optimized CPU implementation
  -> CUDA backend
  -> CPU/GPU agreement tests
```

### 13.8 Modernization / Migration Layer

Responsible for turning existing brittle or research-shaped code into stable, testable, Python-first infrastructure without losing mathematical fidelity.

This layer should include:

- command-line adapters for tools that cannot be imported directly;
- Python package wrappers where library APIs exist;
- containerized runners for hard-to-install dependencies;
- version detection and environment reporting;
- raw-output archiving;
- parser tests for every supported backend output format;
- golden-master fixtures from known knots;
- compatibility shims for different diagram conventions;
- migration notes for each external tool;
- a deprecation plan when native Tetradrome code can fully replace a backend for a specific invariant.

The migration layer should not rewrite math casually. It should first capture existing behavior, make it reproducible, and only then replace components one at a time.

A good adapter contract:

```python
class ExternalToolAdapter:
    name: str

    def is_available(self) -> bool: ...
    def version_info(self) -> dict: ...
    def supported_inputs(self) -> set[str]: ...
    def supported_invariants(self) -> set[str]: ...
    def run_raw(self, request) -> RawBackendOutput: ...
    def normalize(self, raw) -> InvariantResult: ...
```

### 13.9 Validation Layer

Responsible for:

- checking known knots;
- comparing against published or independent data;
- comparing multiple backends;
- checking \(d^2 = 0\) for native complexes;
- logging conventions;
- identifying unvalidated claims.

Every serious computation should produce a report.

### 13.10 Public API surface

§13.1–§13.9 describe the components; this is the stable public surface over them — the functions and types a downstream consumer depends on. It carries no application or domain concepts, and the library never learns anything about what consumes it. Internals (`native/*`, `algebra/*`, individual backend adapters) may change freely as long as this surface and the §11 schema hold.

**Public types** (in addition to §11.1's `InvariantResult` / `Provenance` / `ValidationStatus`):

```python
@dataclass(frozen=True)
class Knot:                       # the public handle; wraps a normalized KnotDiagram (§13.1)
    id: str                       # canonical KnotInfo name, e.g. "K11n34" (§12.4)
    diagram: "KnotDiagram"        # crossings / arcs / orientation / crossing_signs / canonical_hash
    aliases: dict                 # pd / dt / gauss / braid notations (§12.4)

@dataclass(frozen=True)
class ObstructionOutcome:
    name: str                     # "fox_milnor" | "signature" | "arf_invariant"
                                  #   | "rasmussen_invariant" | "ozsvath_szabo_tau"
    obstructs: bool               # does it obstruct sliceness? (False == vanishes / "looks ordinary")
    value: Any
    provenance: Provenance

@dataclass(frozen=True)
class ObstructionProfile:
    knot: str
    outcomes: dict[str, ObstructionOutcome]
    all_vanish: bool              # ordinary by every cheap measure

@dataclass(frozen=True)
class SlicenessVerdict:
    knot: str
    smoothly_slice: Optional[bool]
    topologically_slice: Optional[bool]
    smooth_four_genus: Optional[int]
    topological_four_genus: Optional[int]
    obstruction_profile: ObstructionProfile
    certificate: Optional["SliceCertificate"]

@dataclass(frozen=True)
class KnotTrace:                  # the trace embedding lemma object
    knot: str
    framing: int                  # 0 == the 0-trace X_0(K)
    boundary: str                 # the n-surgery S^3_n(K)
    description: str              # Kirby / RGB handle description

@dataclass(frozen=True)
class TraceSibling:
    a: str
    b: str                        # diffeomorphic traces: X(a) ≅ X(b)
    framing: int
    source: str                   # construction / reference (RGB link, Akbulut, ...)

@dataclass(frozen=True)
class SliceCertificate:
    knot: str
    via: str                      # "direct" | "trace_sibling" | "topological"
    sibling: Optional[str]        # K' with shared trace, when via == "trace_sibling"
    witness: dict                 # e.g. {"rasmussen_invariant(K')": 2}
    references: list[str]

@dataclass(frozen=True)
class RosterEntry:
    knot: Knot
    invariants: dict[str, InvariantResult]    # keyed by canonical name (§12.4)
    sliceness: SlicenessVerdict
    trace_siblings: list[TraceSibling]

@dataclass(frozen=True)
class RosterExport:               # the consumption contract
    version: str
    content_hash: str
    built_at: str
    backend_versions: dict
    validation_summary: dict
    entries: dict[str, RosterEntry]           # keyed by canonical KnotInfo name
```

**Public functions:**

```python
# tetradrome.knots — construction & normalization (native, §12.4)
knots.from_name(name)         # "K11n34", "4_1", ...
knots.from_pd(pd); knots.from_dt(dt); knots.from_gauss(code); knots.from_braid(word)
knots.normalize(knot); knots.mirror(knot)

# tetradrome.invariants — native compute; always returns a typed InvariantResult (§11.1)
invariants.list()
invariants.compute(knot, name, *, validate=True) -> InvariantResult
invariants.compute_all(knot, *, validate=True) -> dict[str, InvariantResult]

# tetradrome.concordance — sliceness / obstructions
concordance.obstruction_profile(knot) -> ObstructionProfile
concordance.slice_status(knot) -> SlicenessVerdict

# tetradrome.traces — trace embedding lemma surface (promotes experiments/piccirillo_trace_notes.md)
traces.trace(knot, framing=0) -> KnotTrace
traces.same_trace(a, b, framing=0) -> bool
traces.siblings(knot) -> list[TraceSibling]
traces.slice_certificate(knot) -> Optional[SliceCertificate]

# tetradrome.catalog — the curated knot set (§7.3)
catalog.names(); catalog.get(name) -> Knot

# tetradrome.export — build / load the validated consumption contract (promotes catalog/ + validation/)
export.build(names=None, *, validate=True) -> RosterExport
export.save(roster, path); export.load(path) -> RosterExport   # verifies content_hash

# tetradrome.validators — opt-in cross-checks (authoring-time; §12)
validators.available(); validators.capabilities(); validators.require(name)
```

**The consumption contract.** A downstream consumer depends on exactly one artifact: a `RosterExport`, produced offline by `export.build(..., validate=True)`, saved, and content-hashed. At read time nothing external is touched; any validator (KnotJob, HFK, Sage) is consulted only at authoring time, and only as a cross-check — never to produce a value. Every value reachable through the export carries its `Provenance` and `ValidationStatus`, so a consumer can assert validation and refuse to proceed otherwise. This boundary is what keeps consumers thin and the mathematics pure: the consumer reads knot-math facts and never reaches into computation.

```python
roster = export.load("roster-vN.json")            # verifies hash; raises on mismatch
e = roster.entries["K11n34"]
e.invariants["rasmussen_invariant"].value          # verified, with provenance
e.sliceness.smoothly_slice                         # False
e.sliceness.topologically_slice                    # True
e.sliceness.obstruction_profile.all_vanish         # True — ordinary by every cheap measure
e.sliceness.certificate.via                        # "trace_sibling"
e.sliceness.certificate.sibling                    # K' with the shared 0-trace
e.sliceness.certificate.witness                    # {"rasmussen_invariant(K')": ...}
```

**Errors (all loud):** `UnknownKnot`, `BackendUnavailable`, `UnvalidatedResult` (raised when `validate=True` but no oracle or cross-backend agreement exists), `ConventionMismatch`, `ExportHashMismatch`.

**Stability.** Public / stable: the functions and types above, plus the `RosterExport` schema. Internal / unstable: `native/*`, `algebra/*`, individual backend adapters. New public modules vs. §9: `traces` and `export`, which promote `experiments/piccirillo_trace_notes.md`, `catalog/`, and `validation/claim_ledger.py` into stable, queryable surfaces; `concordance` is promoted from `invariants/concordance.py` to a documented public module.

---

## 14. Validation Philosophy

Tetradrome should be built for skepticism.

Every output should answer:

1. What definition or invariant was requested?
2. What code path or backend was used?
3. What backend version was used?
4. What conventions were assumed?
5. Was the output checked against known examples?
6. Was the output checked against an independent implementation or published table?
7. If native, was \(d^2 = 0\) verified?
8. What does this result prove?
9. What does this result not prove?

The validation mantra:

> No mystical topology output counts unless it passes boring, independent, repeatable tests.

---

## 15. Claim Ledger

Tetradrome should maintain a claim ledger similar to this:

| Claim | Status | Evidence | Notes |
|---|---:|---|---|
| Classical invariants (det, signature, Alexander) match KnotInfo | Green | Validated across the tables | Seifert-form engine |
| Native Jones matches KnotInfo | Green | Validated through ~11 crossings | Kauffman bracket |
| KnotInfo oracle retrieves known-answer data | Green | Implemented | Validation-first workflow |
| Shared graded-complex back end + reference reducer | Red | Not implemented | M3 |
| Native Khovanov ranks match KnotInfo (mod 2) | Red | Not implemented | M4 |
| Native differential satisfies \(d^2 = 0\) | Red | Not implemented | Required for every native complex |
| Native \(s\) matches KnotInfo | Red | Not implemented | M5 |
| raw == reduced after exact reductions | Red | Not implemented | M6 (decision 0007) |
| GPU tier agrees with the pure-Python reference | Red | Not implemented | M7; the reference is the gold master |

Claim statuses should be updated as the project matures:

```text
Red     = not implemented or not validated
Yellow  = partially validated / limited evidence
Green   = validated against multiple known cases
Blue    = independently reproduced or externally reviewed
```

---

## 16. Milestones

The canonical, maintained roadmap lives in `roadmap/milestones.md` and `roadmap/design/homology-engine.md`; this is a summary. Sequencing is native-first (decisions 0006, 0007): each invariant is computed by Tetradrome and validated against KnotInfo before any optimization or cross-tool comparison.

- **M0 — Scaffold.** Repository, package metadata, tests, conventions, claim ledger, decision records. *Done.*
- **M1 — Classical invariants (Seifert form).** Native determinant, signature, Alexander from a braid / PD; validated across the KnotInfo tables. *Done.*
- **M2 — Resolution cube + Jones.** Native Kauffman bracket → Jones; validated against KnotInfo. *Done.*
- **M3 — Shared graded-complex back end.** Graded chain complex, pure-Python reference reducer, F2 rank / kernel / image homology, \(d^2 = 0\) check, exact complex-size predictor.
- **M4 — Native Khovanov over F2.** Khovanov ranks on the cube; validated against KnotInfo's mod-2 data.
- **M5 — Native Lee / Rasmussen \(s\).** Lee deformation, filtered complex over \(\mathbb{Q}\) (multimodular), \(s\) extraction; validated on known knots.
- **M6 — Exact reductions.** Delooping + Bar-Natan local elimination; verify raw == reduced (decision 0007).
- **M7 — Acceleration tiers.** Packed-bit F2 → JIT → multi-core / NUMA → optional GPU, behind a memory-prediction gate (decision 0008); each tier validated against the reference.
- **M8 — Native knot Floer.** Grid / HFK-cube engine; \(\tau, \epsilon, \nu\); validated against KnotInfo, cross-checked against Szabó's HFK calculator where installed.
- **M9 — Validators, report, export.** Opt-in cross-check adapters (KnotJob, HFK, Sage) behind the §13.8 contract; reproducible reports; content-hashed roster export.
- **M10 — External review package.** Technical summary, reproducibility instructions, validation report, limitations, outreach.

---

## 17. Example Computation Report Format

Each serious run should output a report like:

```yaml
experiment: conway_workflow_reproducer
input:
  knot: conway_11n34
  input_format: name
  canonical_pd_hash: ...

computation:
  invariant_pipeline:
    - khovanov
    - lee
    - rasmussen_s
  backend: tetradrome-native
  method: cube_kauffman / lee_rasmussen
  tetradrome_version: ...
  coefficient_field: ...
  grading_convention: ...

result:
  raw_output_file: reports/raw/...
  normalized_value: ...

validation:
  known_answer_source: knotinfo
  known_answer_match: PASS/FAIL/NOT_AVAILABLE
  cross_check_validator: knotjob / hfk / sage / not_run
  cross_check_match: PASS/FAIL/NOT_RUN
  d_squared_check: PASS/FAIL/NOT_APPLICABLE

interpretation:
  computed_invariant: ...
  mathematical_claim: ...
  claim_strength: native_computation / known_table_match / theorem_reference
  limitations: ...

reproducibility:
  tetradrome_version: ...
  environment_hash: ...
  input_hash: ...
  report_hash: ...
```

No final number should appear without its validation context.

---

## 18. Mathematical Caution: Obstruction Is Not Classification

Tetradrome must document the distinction between obstruction and proof.

A nonzero obstruction may prove that a knot lacks a property, such as smooth sliceness. A zero obstruction usually does not prove the property exists.

For example:

```text
nonzero sliceness obstruction
  -> can imply not smoothly slice

zero sliceness obstruction
  -> does not imply smoothly slice
```

This distinction should be built into both the documentation and the reporting language.

---

## 19. Performance Strategy

Performance matters, but correctness comes first.

The performance strategy is:

```text
raw, faithful pure-Python reference (correct first)
  -> validation harness + known-answer + d^2 = 0
  -> exact, answer-preserving reductions
  -> packed-bit / JIT / multi-core optimization
  -> optional GPU acceleration
  -> cross-check against trusted tools throughout
```

Likely performance bottlenecks for native components:

- cube-of-resolutions explosion;
- grading-block matrix construction;
- sparse rank calculations over \(\mathbb{F}_2\);
- Lee deformation filtration handling;
- repeated validation across knot catalogs.

CUDA is most appropriate for:

- integer-array state enumeration;
- packed-bit XOR row operations;
- large batch rank calculations;
- repeated grading-block workloads.

CUDA is not appropriate for:

- replacing mathematical validation;
- symbolic correctness;
- untested grading logic;
- early proof-of-concept code;
- wrapped-backend results where the bottleneck is external tool invocation.

---

## 20. Licensing and Distribution Notes

**License: Apache 2.0.** All Tetradrome code is original; nothing inherits a license from another project.

Several useful tools are GPL-licensed or live in ecosystems with their own licensing and installation constraints. Tetradrome avoids distribution problems by treating them strictly as external validators and historical references, never as incorporated code.

Rules:

- All first-party code is original; no GPL (or other) source is copied or adapted into the tree.
- GPL tools are optional external validators only: invoked as separate programs, or installed by the user as optional dependencies. Tetradrome calls them; it does not vendor or statically combine them into its distribution.
- No backend data is vendored either; KnotInfo and similar are queried/validated against and cited, not copied into the repo.
- Record backend license and installation requirements in `docs/existing_tools.md`.
- Keep raw-output parsers modular; make every validator optional; degrade gracefully when one is unavailable — a missing validator skips a cross-check, it never blocks a result.

Because no GPL source or data is distributed inside Tetradrome, no copyleft obligation attaches and the permissive Apache 2.0 license fits. Existing tools also serve as a parity reference — a maintenance signal for new maths or features worth matching, not a dependency.

---

## 21. Outreach Framing

If shared externally with a mathematician such as Lisa Piccirillo, the project should be framed humbly and concretely.

Suggested framing:

> I am building a Python-first project called Tetradrome for reproducible computation of the invariants of smooth 4-dimensional topology. It computes them natively — Khovanov, Lee, the Rasmussen \(s\)-invariant, knot Floer, and the classical and concordance invariants — and validates each result against KnotInfo and, where installed, against existing tools, with full provenance and an explicit validation status. The first target is a Conway-adjacent workflow. It is not a claim of a new theorem, and the existing tools are treated as the validation reference, not replaced or disparaged.

Avoid framing it as:

> I solved 4D knot theory in Python.

Better framing:

> I built a careful computational workbench for exploring a narrow, validated slice of the machinery.

---

## 22. Proposed Short Email for External Sharing

Subject options:

```text
Tetradrome: a Python project for reproducible Conway-adjacent knot invariant workflows
```

or

```text
A small Python computational workbench inspired by the Conway knot
```

Draft:

```text
Professor Piccirillo,

I am working on a Python project called Tetradrome, intended as a readable, reproducible workbench that computes the invariants of smooth 4-dimensional topology natively and validates them against known data.

The initial target is intentionally narrow: a Conway-adjacent workflow that normalizes inputs, computes the invariants natively, validates the results against KnotInfo (and cross-checks against existing tools where installed), records conventions and versions, and produces transparent reports. The goal is not to claim a new theorem or to replace established tools — those remain my validation reference — but to build a disciplined, auditable apparatus for reproducing and exploring small examples.

Longer-term, I am building out the native Khovanov / Lee / Floer engines and accelerating the exact-algebra core (multi-core and optional GPU) behind the same validation discipline.

I have attached the project specification in case it is of interest. I would be grateful for any high-level warning signs, references, or suggestions about where such a project would be most likely to go wrong.

Best,
Randy
```

---

## 23. Success Criteria

Tetradrome v1 succeeds if it can honestly say:

1. The project has a clean architecture that separates diagrams, native engines, the algebra back end, invariants, validation, reports, and experiments.
2. Existing tooling is acknowledged and used as the validation reference, never as a compute dependency.
3. Knot, link, and braid inputs (name, PD, DT, Gauss, braid) are normalized natively into a common representation.
4. KnotInfo or equivalent known-answer data is used for validation.
5. At least one homological invariant is computed natively and agrees with KnotInfo (and with an external tool where installed).
6. Every result records the native method, version, input, convention notes, raw output, normalized output, and validation status.
7. Conway-adjacent reports are reproducible.
8. The documentation clearly distinguishes computation, obstruction, theorem reference, and proof.
9. The code can grow toward more native engines without rewriting the workbench.
10. Optional validators run behind one stable adapter contract with version capture, raw-output archiving, and golden-master regression tests.

Tetradrome v1 fails if it produces impressive-looking results without validation, if it pretends existing computational topology tools do not exist, or if it ships a native computation that was never checked against known behavior.

---

## 24. Summary

Tetradrome should begin as a narrow but honest workbench:

```text
Conway-adjacent mission
  + native invariant engines
  + generic diagram model (knots, links, braids)
  + shared exact-algebra back end
  + normalized invariant schema
  + rigorous validation against existing tools
  + reproducible reports
  + opt-in validators (never compute)
  + acceleration tiers (multi-core, optional GPU)
  + clear limitations
```

The goal is not to build the whole mathematical universe immediately. The goal is to build the calibrated bench on which these instruments are built, validated, audited, and extended.

---

## References

[^piccirillo-annals]: Lisa Piccirillo, **"The Conway knot is not slice,"** *Annals of Mathematics*, 191(2), 2020. https://annals.math.princeton.edu/2020/191-2/p05

[^piccirillo-arxiv]: Lisa Piccirillo, **"The Conway knot is not slice,"** arXiv:1808.02923. https://arxiv.org/abs/1808.02923

[^knot-atlas-khovanov]: **Knot Atlas, "Khovanov Homology."** Documents FastKh, JavaKh, coefficient options, and the relation between Khovanov homology and the Jones polynomial. https://katlas.org/wiki/Khovanov_Homology

[^knottheory]: **Knot Atlas, "The Mathematica Package KnotTheory`."** Describes KnotTheory` as the main tool used to produce the Knot Atlas. https://katlas.org/wiki/The_Mathematica_Package_KnotTheory%60

[^sage-links]: **SageMath Documentation, "Links - Knot Theory."** Documents Sage link objects, diagram conventions, and references to KnotInfo / LinkInfo. https://doc.sagemath.org/html/en/reference/knots/sage/knots/link.html

[^sage-chain-complex]: **SageMath Documentation, "Chain complexes and homology."** Documents Sage chain-complex and homology tools. https://doc.sagemath.org/html/en/reference/homology/sage/homology/chain_complex.html

[^khoca]: **Khoca GitHub repository.** Describes Khoca as a C++/Python program computing certain Khovanov-Rozansky homologies of knots. https://github.com/LLewark/khoca

[^hfkcalc]: **Ozsváth/Szabó HFK Calculator.** C++11 program for computing knot Floer homology from planar diagrams. https://web.math.princeton.edu/~szabo/HFKcalc.html

[^knot-floer-pypi]: **knot-floer-homology on PyPI.** Python wrapper for Zoltán Szabó's HFK Calculator; accepts PD and Spherogram input. https://pypi.org/project/knot-floer-homology/

[^spherogram-docs]: **SnapPy/Spherogram documentation.** Describes Spherogram as a module for creating links programmatically and obtaining PD/DT codes. https://snappy.computop.org/spherogram.html

[^spherogram-github]: **Spherogram GitHub repository.** Describes Spherogram as a Python module for planar diagrams in 3-dimensional topology, including links and Heegaard diagrams. https://github.com/3-manifolds/Spherogram

[^knotinfo-about]: **KnotInfo About page.** Notes HFK invariants computed using HFKcalc and Khovanov data calculated using KnotJob. https://knotinfo.org/homelinks/about.html

[^knotinfo-downloads]: **KnotInfo Download Files.** Provides database and polynomial invariant downloads, including multiple Khovanov variants and Heegaard Floer polynomial data. https://knotinfo.org/homelinks/database_download.php

[^cupy]: **CuPy documentation, Overview.** Describes CuPy as a NumPy/SciPy-compatible array library for GPU-accelerated computing on CUDA or ROCm devices. https://docs.cupy.dev/en/stable/overview.html

[^numba-cuda]: **Numba-CUDA documentation.** Describes Numba-CUDA as a CUDA target for the Numba Python JIT compiler for writing CUDA kernels in Python. https://nvidia.github.io/numba-cuda/
