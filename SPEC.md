# Tetradrome Project Specification

**Working title:** Tetradrome  
**Project type:** Python-first computational topology workbench for knot-invariant experiments  
**Revised strategic focus:** Orchestration, validation, reproducibility, and selective native acceleration rather than immediate from-scratch reinvention  
**Initial technical focus:** Conway-adjacent Khovanov / Lee / Rasmussen workflows using existing tools as reference backends, then selectively replacing components with native Python/CUDA implementations  
**Long-term direction:** Extensible computational library for exploring 3D knot diagrams through invariants that constrain smooth 4D behavior  
**Status:** Concept / architecture specification, revised after survey of existing tooling  

---

## 1. Executive Summary

Tetradrome is a proposed Python-first computational workbench for building, validating, reproducing, and eventually extending knot-invariant calculations relevant to smooth 4-dimensional topology.

The project should not begin as a claim that no tooling exists. Serious tooling already exists for Khovanov homology, Rasmussen's \(s\)-invariant, knot Floer homology, knot databases, and planar diagram manipulation. The useful version of Tetradrome is therefore not “the first program that computes these invariants.” It is a modern, Python-first, audit-friendly, reproducible, extensible workbench that can orchestrate existing tools, compare results, document conventions, produce readable reports, and eventually add native Python/CUDA components where they provide clear value.

The stronger modernization thesis is that much of the existing software is research-tool shaped: valuable, mathematically serious, and sometimes very fast, but often bound to older ecosystems, command-line workflows, Java/Mathematica/C++ packaging, Sage-specific environments, or fragile installation paths. Tetradrome should treat those tools as reference instruments first, then gradually migrate selected capabilities into a modern Python package with explicit backend contracts, regression tests, reproducible reports, and optional GPU acceleration for exact algebra workloads.

The immediate goal is a **Conway-focused reproducibility and validation pipeline**:

```text
known knot input
  -> normalized diagram representation
  -> backend selection
  -> invariant computation
  -> independent cross-checks
  -> reproducible report
  -> carefully limited 4D-topology interpretation
```

The long-term goal is broader: Tetradrome should become a modular Python library where knot diagrams can be compiled into algebraic chain complexes, processed by exact algebra engines, validated against known results, and extended toward additional invariants such as grid-homology versions of knot Floer homology.

The guiding phrase is:

> Build the bench before building every instrument. Use existing instruments honestly, validate them against one another, and only machine new parts where Tetradrome adds real value.

---

## 2. Background and Motivation

A knot is drawn in 3-dimensional space, but some of the most interesting questions about that knot concern what it can bound in 4 dimensions. For example, a knot is slice if it bounds a smoothly embedded disk in the 4-ball \(B^4\). The Conway knot was historically significant because its smooth sliceness resisted classification until Lisa Piccirillo proved that the Conway knot is not slice. Her Annals of Mathematics paper states that this completed the classification of slice knots under 13 crossings and produced the first example of a non-slice knot that is both topologically slice and a positive mutant of a slice knot.[^piccirillo-annals]

Tetradrome is motivated by the idea that these 4-dimensional questions require a kind of mathematical measuring apparatus. The apparatus is not a physical device. It is an algebraic and computational system: knot diagrams, chain complexes, differentials, gradings, homology calculations, known-answer tables, and carefully limited interpretations of the resulting invariants.

The immediate need is not to pretend no apparatus exists. The immediate need is to build a **disciplined engineering layer** around that apparatus:

- make existing computations easier to run from Python;
- normalize inputs and outputs;
- record conventions and assumptions;
- cross-check results across independent sources;
- produce reproducible reports;
- isolate places where native implementation would genuinely add value;
- eventually accelerate exact algebra workloads where GPU computation is appropriate.

---

## 3. Project Identity: What Is a Tetradrome?

The word **Tetradrome** is used here as a project name for a computational instrument aimed at 4-dimensional knot mathematics.

Conceptually:

- An **orrery** models celestial motion.
- An **armillary sphere** models astronomical reference frames.
- A **tetradrome** models algebraic constraints on how 3D knots may behave when considered through 4D topology.

In software terms, Tetradrome is a workbench-and-instrument architecture:

```text
knot diagram / knot database entry
  -> normalized representation
  -> one or more computational backends
  -> validated invariant result
  -> reproducibility metadata
  -> report and claim ledger
```

Later, Tetradrome may also include native chain-complex compilers:

```text
knot diagram
  -> combinatorial state space
  -> graded chain complex
  -> exact sparse algebra
  -> homology / invariant
  -> carefully limited 4D-topology interpretation
```

The revised priority is to build the **workbench** first, then build selected instruments.

---

## 4. Existing Tooling Landscape

The field already has meaningful computational tooling. Tetradrome should acknowledge this explicitly and use it strategically.

| Tool / ecosystem | Area | What it provides | Relevance to Tetradrome |
|---|---|---|---|
| **KnotTheory` / Knot Atlas** | Mathematica / knot tables / Khovanov | KnotTheory` was a main tool used to produce the Knot Atlas. Knot Atlas documents Khovanov computations and mentions FastKh and JavaKh backends, with JavaKh supporting different coefficient choices. | Important reference backend and historical ecosystem. Tetradrome should not claim novelty for basic Khovanov computation. |
| **FastKh / JavaKh** | Khovanov computation | Khovanov-focused computational engines associated with KnotTheory` / Knot Atlas. Knot Atlas notes JavaKh programs are much faster than the Mathematica implementation. | Potential external backend or validation reference. |
| **KnotJob** | Java knot homology software | Computes several knot invariants, including Khovanov-related data and Rasmussen-style invariants. KnotInfo notes that its displayed Khovanov homology invariants were calculated using Dirk Schütz's KnotJob and reformatted by Jason Garcia. | Strong candidate for independent validation and perhaps backend integration. |
| **SageMath knot tools** | Python-based computer algebra ecosystem | Sage provides link/knot objects, knot diagrams, invariant functionality, and optional access to KnotInfo data. Its documentation notes support for planar diagrams, braids, Gauss codes, and a connection to KnotInfo / LinkInfo. | Useful Python-adjacent ecosystem. Tetradrome should interoperate where possible rather than duplicate all infrastructure. |
| **SageMath chain-complex tools** | Algebra / homology | Sage includes chain-complex and homology functionality over appropriate rings/fields. | Useful reference for algebra validation and exact homology work. |
| **Khoca** | Khovanov-Rozansky homology | C++/Python research program for computing certain Khovanov-Rozansky homologies of knots. | Demonstrates that advanced homology computation already exists; potential reference for higher homology directions. |
| **HFKcalc** | Knot Floer homology | C++11 program by Ozsváth/Szabó ecosystem using planar diagrams for knots and computing knot Floer data modulo a prime. | Important reference for Floer-side computation; possible backend or validation reference. |
| **knot-floer-homology PyPI package** | Python wrapper for HFKcalc | Python package wrapping Zoltán Szabó's HFK Calculator. Accepts PD input and Spherogram links; returns knot Floer data such as ranks, \(\tau\), \(\epsilon\), \(\nu\), fibered status, genus, and related data. | Very strong reason not to build Floer from scratch first. Use as a backend. |
| **Spherogram / SnapPy** | Planar diagrams and 3-manifold topology | Python module in the SnapPy ecosystem for planar diagrams arising in 3-dimensional topology, including links and Heegaard diagrams. It can create links programmatically and return PD codes and DT codes. | Strong candidate for Tetradrome's first diagram layer instead of writing diagram mechanics from zero. |
| **KnotInfo** | Knot invariant database | Database of knot invariants, downloadable data, polynomial invariants, Khovanov variants, Heegaard Floer polynomial data, and references. Its about page notes HFK invariants computed using HFKcalc and Khovanov data calculated using KnotJob. | Validation oracle and known-answer source. Tetradrome should integrate it as a reference dataset. |
| **pyknotid** | Knots as 3D curves | Python package for knots and links represented as three-dimensional space curves, including diagram generation and topological analysis. | Adjacent utility for geometric/visual input, not likely central to the first Conway/Khovanov pipeline. |

### Strategic conclusion

Tetradrome should not start as a from-scratch replacement for these tools.

The project should start as:

```text
Tetradrome = orchestration + normalization + validation + reporting + selective native acceleration
```

not:

```text
Tetradrome = pretend existing knot software does not exist
```

### 4.1 Modernization Opportunity

The most useful version of Tetradrome is a modernization and force-multiplier project. Existing computational topology tools should be treated with respect: they encode years of mathematical work and should become validation references, not discarded competitors. But the engineering surface around them can be improved dramatically.

The modernization target is:

```text
legacy / research-shaped tool
  -> stable adapter
  -> normalized schema
  -> golden-master tests
  -> reproducible reports
  -> selected native Python/CUDA replacement only where justified
```

This is closer to porting, packaging, validating, and accelerating a scientific software ecosystem than to inventing a new invariant. The project should aim to preserve mathematical behavior while improving:

- installability;
- Python API ergonomics;
- backend interchangeability;
- reproducibility;
- raw-output capture;
- cross-tool comparison;
- test coverage;
- exact algebra performance;
- documentation for non-specialist software engineers.

A good migration rule is:

> Existing tools are the gold masters until Tetradrome proves otherwise. Native code must match them on known examples before it is trusted.

### 4.2 Why GPU Support Is a Real, but Narrow, Opportunity

GPU acceleration should not be marketed as magic topology acceleration. The likely force multiplier is much more specific: exact algebra over finite fields, packed-bit matrix operations, batched rank computations, and repeated validation runs. CuPy provides a NumPy/SciPy-like GPU array ecosystem, and Numba-CUDA provides a Python path for writing CUDA kernels; both are plausible implementation tools for the eventual acceleration layer.[^cupy][^numba-cuda]

The first GPU target should not be the symbolic topology logic. It should be the heavy algebra underneath native computations:

```text
chain complex / grading block
  -> sparse or packed GF(2) matrix
  -> rank / kernel / image computation
  -> homology dimension
```

The engineering principle remains:

```text
external backend reference
  -> clear CPU reference implementation
  -> optimized CPU implementation
  -> GPU implementation
  -> CPU/GPU/reference agreement tests
```

---

## 5. Revised End Goal

The end goal is to build an extensible Python-first workbench for computational experiments in low-dimensional topology, beginning with knot invariants relevant to smooth 4-dimensional questions.

A mature version of Tetradrome should support:

1. **Formal knot input and normalization**
   - Spherogram links
   - PD codes
   - braid words
   - fixed knot catalogs
   - eventually grid diagrams

2. **Backend orchestration**
   - SageMath where useful
   - KnotJob / JavaKh where callable
   - KnotInfo as validation data
   - `knot-floer-homology` as a Floer backend
   - native Tetradrome implementations when justified

3. **Invariant construction and retrieval**
   - Khovanov homology through external backends first
   - Lee deformation / Rasmussen \(s\)-invariant where supported
   - knot Floer invariants through existing HFK tooling first
   - native mod-2 Khovanov implementation later
   - native exact algebra backend later

4. **Exact algebra backend**
   - finite-field chain complexes
   - sparse matrices over \(\mathbb{F}_2\)
   - packed-bit rank algorithms
   - CPU reference implementation
   - optional CUDA/JIT acceleration for heavy exact algebra

5. **Validation tooling**
   - \(d^2 = 0\) checks for native complexes
   - known-answer tests
   - independent implementation comparisons
   - backend agreement reports
   - reproducibility logs
   - claim-status ledger

6. **Research-grade humility**
   - clear distinction between computing an invariant, obstructing a property, and proving a property
   - explicit documentation of conventions
   - explicit documentation of what is not yet validated
   - no claims of new mathematics without expert review

The mature project should make it possible to say:

> For this knot diagram, under these coefficient and grading conventions, Tetradrome requested or constructed the relevant invariant, recorded the backend and version, compared the output against known examples or independent data when possible, and reported only the conclusions justified by that invariant.

---

## 6. Immediate Need

The immediate need is to build the **Tetradrome workbench**, not the entire universe of homological knot theory.

The first successful version should be a **Conway Workflow Reproducer**, not a general-purpose theorem machine.

The first release should focus on:

- a small fixed catalog of knots;
- normalized PD/Spherogram input;
- wrappers around existing invariant tools;
- known-answer validation using KnotInfo and small examples;
- reproducible report generation;
- a clear claim ledger;
- an architecture that leaves room for native computation later.

The immediate build should not attempt to do everything. It should build the extensible workbench correctly.

---

## 7. Revised Initial Mathematical Scope

### 7.1 Primary v1 target

The first mathematical workflow is:

```text
normalized knot diagram
  -> external/backend Khovanov or Rasmussen computation
  -> validation against known data
  -> reproducible Conway-adjacent report
```

The relevant conceptual pipeline remains:

```text
Khovanov homology
  -> Lee deformation
  -> Rasmussen s-invariant
```

This is the most relevant first pipeline because Piccirillo's Conway-knot result uses a related knot with the same 4-dimensional trace and applies Rasmussen's \(s\)-invariant to obstruct sliceness.[^piccirillo-arxiv]

### 7.2 Floer-side v1 target

Floer functionality should not be implemented from scratch in v1.

The v1 Floer goal should be:

```text
Spherogram/PD input
  -> knot-floer-homology backend
  -> Tetradrome normalized output schema
  -> comparison against KnotInfo where available
```

Native grid homology can be a later module.

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

### 7.4 Deferred mathematical scope

The following should be intentionally deferred:

- full native Heegaard-Floer machinery;
- full native Khovanov implementation beyond exploratory mod-2 work;
- general 3-manifold surgery calculations;
- arbitrary knot database support beyond selected importers;
- full integer torsion computations;
- polished diagram editing;
- a graphical user interface;
- automated theorem proving;
- claims of new mathematical results.

---

## 8. Non-Goals

Tetradrome v1 is **not** intended to:

1. prove new theorems;
2. replace existing mathematical software;
3. pretend Khovanov/Floer tooling does not already exist;
4. claim that a knot is slice merely because an obstruction vanishes;
5. provide a complete implementation of all Khovanov variants;
6. provide a complete implementation of Heegaard-Floer theory;
7. hide conventions behind a black-box interface;
8. rely on GPU acceleration before CPU correctness is established;
9. confuse a wrapper result with an independently implemented theorem.

The project should be built around the principle:

> Fast wrong math is worse than slow honest math; black-box correct math is still not enough unless the box, version, conventions, and validation path are recorded.

---

## 9. Architecture Overview

Recommended repository layout:

```text
tetradrome/
  diagrams/
    model.py
    pd.py
    spherogram_adapter.py
    braid.py                  # future
    grid.py                   # future

  backends/
    base.py
    sage_backend.py
    knotjob_backend.py
    knotinfo_backend.py
    knot_floer_backend.py
    java_kh_backend.py        # possible future
    native_backend.py         # future

  invariants/
    schema.py
    khovanov.py
    rasmussen.py
    knot_floer.py
    concordance.py
    traces.py             # public: trace / same-trace / slice certificate

  native/
    khovanov/
      cube.py
      resolutions.py
      frobenius.py
      differential.py
      gradings.py
      complex.py

    lee/
      deformation.py
      filtered_complex.py

    grid_floer/               # future placeholder
      README.md

  algebra/
    chain_complex.py
    graded_complex.py
    gf2_matrix.py
    homology.py
    backends/
      cpu.py
      cuda.py

  catalog/
    knots.yaml
    known_answers.yaml
    sources.yaml

  experiments/
    conway_workflow_reproducer.py
    piccirillo_trace_notes.md

  validation/
    known_answer.py
    backend_agreement.py
    d_squared.py
    compare_known.py
    claim_ledger.py
    reports.py
    roster_export.py      # public: build / load the validated RosterExport

  reports/
    templates/
      computation_report.md.j2
      backend_comparison.md.j2
      claim_ledger.md.j2

  tests/
    test_pd_parser.py
    test_spherogram_adapter.py
    test_backend_schema.py
    test_knotinfo_import.py
    test_unknot.py
    test_trefoil.py
    test_figure_eight.py
    test_conway_pipeline.py

  docs/
    conventions.md
    validation.md
    backend_matrix.md
    existing_tools.md
    conway_notes.md
    outreach.md
```

---

## 10. Core Design Principle: Narrow Mission, Extensible Core

The v1 mission is narrow, but the core should be generic.

Good:

```python
diagram = tetradrome.diagrams.load("conway_11n34")
result = tetradrome.compute(
    diagram,
    invariant="rasmussen_s",
    backend="knotjob",
)
report = tetradrome.report(result)
```

Also good:

```python
diagram = tetradrome.diagrams.from_spherogram("K11n34")
result = tetradrome.compute(
    diagram,
    invariant="tau",
    backend="knot_floer_homology",
)
```

Bad:

```python
result = compute_conway_answer()
```

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
  input_format: spherogram
  input_value: K11n34
  pd_code_hash: ...

computation:
  invariant: rasmussen_invariant
  backend: knotjob
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
    backend: str                  # "spherogram" | "knot_floer_homology" | "knotjob" | "knotinfo" | "sage" | "native"
    backend_version: str
    input_format: str             # "spherogram" | "pd" | "dt" | "gauss" | "braid" | "name"
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

## 12. Backend Strategy

### 12.1 Backend classes

Each backend should expose a minimal interface:

```python
class InvariantBackend:
    name: str
    supported_invariants: set[str]

    def is_available(self) -> bool: ...
    def version_info(self) -> dict: ...
    def compute(self, diagram, invariant, options) -> InvariantResult: ...
```

### 12.2 Backend priorities

Initial integration priority:

1. **Spherogram adapter** for diagram handling.
2. **KnotInfo importer** for validation data.
3. **knot-floer-homology backend** for Floer invariants such as \(\tau\), \(\epsilon\), \(\nu\), and rank data.
4. **SageMath adapter** where feasible.
5. **KnotJob / JavaKh adapter** if command-line invocation can be made reliable.
6. **Native Tetradrome mod-2 Khovanov engine** after the validation harness exists.
7. **Native CUDA exact algebra backend** after CPU reference correctness exists.

### 12.3 Backend matrix

Tetradrome should maintain a table like:

| Invariant | KnotInfo lookup | Sage | KnotJob / JavaKh | knot-floer-homology | Tetradrome native CPU | Tetradrome CUDA |
|---|---:|---:|---:|---:|---:|---:|
| Jones polynomial | planned | possible | possible | n/a | future | n/a |
| Khovanov ranks | planned | possible | planned | n/a | future | future |
| Rasmussen \(s\) | planned | possible | planned | n/a | future | future? |
| Knot Floer ranks | planned | n/a/possible | n/a | planned | future | future? |
| \(\tau\) | planned | n/a/possible | n/a | planned | future | future? |
| \(\epsilon\), \(\nu\) | planned | n/a/possible | n/a | planned | future | future? |

This matrix should be maintained in `docs/backend_matrix.md`.

### 12.4 Vocabulary alignment with the referenced tools

The §12.3 matrix says *which* backend can produce an invariant. This table fixes the *names*. The left column is Tetradrome's canonical name — the standard term for the object in the knot-theory literature, chosen on the mathematics and independent of any tool. The remaining columns record how each backend happens to spell that same object, so one normalizer can translate a backend's output into the schema. Listing a tool's spelling here is interop, not adoption: the right columns tell the normalizer how to read each backend; the canonical column stands on the mathematics. Where these coincide with KnotInfo's column names, it is because KnotInfo also uses the standard literature names — not because Tetradrome takes them from it.

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

**Bridges.** `Knot.to_spherogram()` → `spherogram.Link`; `Knot.sage_link()` → Sage `Knot`; `Knot.exterior()` → SnapPy `Manifold`. The Floer backend is fed a Spherogram link or PD directly (`knot_floer_homology.pd_to_hfk(...)`), so there is no lossy round-trip.

---

## 13. Core Components

### 13.1 Diagram Layer

Responsible for:

- accepting Spherogram links;
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
  source_backend
  canonical_hash
```

The diagram layer should prefer interoperability over purity. If Spherogram already handles a representation well, use it.

### 13.2 Backend Layer

Responsible for:

- calling existing tools;
- parsing raw outputs;
- recording versions;
- handling unavailable backends gracefully;
- normalizing results;
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

### 13.7 GPU / CUDA Backend

CUDA should be treated as an acceleration backend, not as the source of mathematical truth.

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
# tetradrome.knots — construction & normalization (Spherogram-backed, §12.4)
knots.from_name(name)         # "K11n34", "4_1", ...
knots.from_pd(pd); knots.from_dt(dt); knots.from_gauss(code); knots.from_braid(word)
knots.from_spherogram(link); knots.normalize(knot); knots.mirror(knot)

# tetradrome.invariants — compute; always returns a typed InvariantResult (§11.1)
invariants.list()
invariants.compute(knot, name, *, backend=None, validate=True) -> InvariantResult
invariants.compute_all(knot, *, backend=None, validate=True) -> dict[str, InvariantResult]

# tetradrome.concordance — sliceness / obstructions
concordance.obstruction_profile(knot, *, backend=None) -> ObstructionProfile
concordance.slice_status(knot, *, backend=None) -> SlicenessVerdict

# tetradrome.traces — trace embedding lemma surface (promotes experiments/piccirillo_trace_notes.md)
traces.trace(knot, framing=0) -> KnotTrace
traces.same_trace(a, b, framing=0) -> bool
traces.siblings(knot) -> list[TraceSibling]
traces.slice_certificate(knot) -> Optional[SliceCertificate]

# tetradrome.catalog — the curated knot set (§7.3)
catalog.names(); catalog.get(name) -> Knot

# tetradrome.export — build / load the validated consumption contract (promotes catalog/ + validation/)
export.build(names=None, *, backend=None, validate=True) -> RosterExport
export.save(roster, path); export.load(path) -> RosterExport   # verifies content_hash

# tetradrome.backends — selection (authoring-time; §12)
backends.available(); backends.capabilities(); backends.require(name)
```

**The consumption contract.** A downstream consumer depends on exactly one artifact: a `RosterExport`, produced offline by `export.build(..., validate=True)`, saved, and content-hashed. At read time no backend is touched — the Java / C++ / Sage tools are an authoring-only dependency. Every value reachable through the export carries its `Provenance` and `ValidationStatus`, so a consumer can assert validation and refuse to proceed otherwise. This boundary is what keeps consumers thin and the mathematics pure: the consumer reads knot-math facts and never reaches into computation.

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
| Existing tooling survey completed | Yellow | Initial list assembled | Needs maintenance and expert review |
| Spherogram adapter handles initial knot catalog | Red | Not implemented | First engineering milestone |
| KnotInfo importer retrieves known-answer data | Red | Not implemented | Required for validation-first workflow |
| knot-floer-homology backend runs on small knots | Red | Not implemented | Floer v1 target |
| Khovanov/Rasmussen backend selected and callable | Red | Not implemented | Requires Sage/KnotJob/JavaKh evaluation |
| Backend outputs normalize into shared schema | Red | Not implemented | Required before reports are meaningful |
| Conway workflow report is reproducible | Red | Not implemented | Requires input, backend, validation, and report trail |
| Native mod-2 Khovanov complex builds for unknot/trefoil | Red | Deferred | Native v2/v3 feature |
| Native differential satisfies \(d^2 = 0\) | Red | Deferred | Required for every native complex |
| GPU backend agrees with CPU backend | Red | Deferred | GPU is optional until CPU is trusted |

Claim statuses should be updated as the project matures:

```text
Red     = not implemented or not validated
Yellow  = partially validated / limited evidence
Green   = validated against multiple known cases
Blue    = independently reproduced or externally reviewed
```

---

## 16. Milestones

### Milestone 0: Project scaffold

Deliverables:

- repository structure;
- package metadata;
- test framework;
- documentation skeleton;
- coding conventions;
- claim ledger.

### Milestone 1: Existing tooling integration map

Deliverables:

- `docs/existing_tools.md`;
- backend matrix;
- install notes;
- license notes;
- assessment of which tools are reference-only, callable backends, validation data, or future inspiration.

### Milestone 2: Diagram input and catalog

Deliverables:

- Spherogram adapter;
- PD-code normalizer;
- hardcoded knot catalog;
- unknot / trefoil / figure-eight examples;
- Conway / Kinoshita-Terasaka entries if reliable identifiers and encodings are available.

### Milestone 3: Validation data import

Deliverables:

- KnotInfo data importer or manual known-answer loader;
- known-answer schema;
- comparison tool;
- reproducible source metadata.

### Milestone 4: First backend adapters

Deliverables:

- `knot-floer-homology` adapter;
- SageMath adapter if feasible;
- KnotJob/JavaKh feasibility spike;
- backend availability checks;
- normalized output schema.

### Milestone 4A: Modernization harness

Deliverables:

- adapter contract for external tools;
- backend installation probes;
- raw-output capture format;
- parser fixtures for each integrated tool;
- golden-master outputs for unknot, trefoil, figure-eight, and selected small knots;
- container or environment notes for brittle dependencies;
- a migration matrix identifying which backend capabilities are wrapper-only, reproducible, native-candidate, or not worth replacing.

### Milestone 5: Report generator

Deliverables:

- computation report template;
- backend comparison report;
- claim-ledger report;
- reproducibility hash;
- raw-output capture.

### Milestone 6: Conway workflow reproducer

Deliverables:

- Conway-knot input;
- Kinoshita-Terasaka input;
- Piccirillo-related \(K'\) input if reliably encoded;
- Conway-adjacent computation report;
- clear statement of what the reproducer does and does not establish.

### Milestone 7: Native mod-2 Khovanov engine

Deliverables:

- cube-of-resolutions enumeration;
- resolution-circle detection;
- chain group construction;
- differential construction over \(\mathbb{F}_2\);
- \(d^2 = 0\) verification;
- known-answer validation against external backends.

### Milestone 8: Native exact algebra / performance backend

Deliverables:

- packed-bit \(\mathbb{F}_2\) matrix representation;
- CPU optimized rank computation;
- optional Numba/CUDA backend;
- CPU/GPU agreement tests;
- benchmarks against native CPU and external tools where appropriate.

### Milestone 9: Native Lee / Rasmussen experiments

Deliverables:

- Lee deformation implementation;
- filtered-complex handling;
- Rasmussen \(s\)-invariant extraction;
- validation on known knots;
- comparison with external backends.

### Milestone 10: External review package

Deliverables:

- compact technical summary;
- reproducibility instructions;
- validation report;
- limitations page;
- polite outreach note for researchers.

---

## 17. Example Computation Report Format

Each serious run should output a report like:

```yaml
experiment: conway_workflow_reproducer
input:
  knot: conway_11n34
  input_format: spherogram
  canonical_pd_hash: ...

computation:
  invariant_pipeline:
    - khovanov
    - lee
    - rasmussen_s
  requested_backend: knotjob
  actual_backend: knotjob
  backend_version: ...
  coefficient_field: ...
  grading_convention: ...

backend_result:
  raw_output_file: reports/raw/...
  normalized_value: ...

validation:
  known_answer_source: knotinfo
  known_answer_match: PASS/FAIL/NOT_AVAILABLE
  independent_backend: sage/java_kh/native/not_run
  independent_backend_match: PASS/FAIL/NOT_RUN
  d_squared_check: NOT_APPLICABLE_FOR_WRAPPED_BACKEND

interpretation:
  computed_invariant: ...
  mathematical_claim: ...
  claim_strength: backend_computation / known_table_match / theorem_reference
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

The revised performance strategy is:

```text
use existing trusted tools first
  -> build validation harness
  -> implement clear CPU reference where useful
  -> compare against trusted tools
  -> optimize exact algebra
  -> optionally add CUDA acceleration
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
- Keep raw-output parsers modular; make every backend optional; degrade gracefully when one is unavailable.

Because no GPL source or data is distributed inside Tetradrome, no copyleft obligation attaches and the permissive Apache 2.0 license fits. Existing tools also serve as a parity reference — a maintenance signal for new maths or features worth matching, not a dependency.

---

## 21. Outreach Framing

If shared externally with a mathematician such as Lisa Piccirillo, the project should be framed humbly and concretely.

Suggested framing:

> I am building a Python-first project called Tetradrome for reproducible computational experiments around knot invariants relevant to smooth 4-dimensional topology. The first target is a Conway-adjacent workflow that orchestrates existing tools where appropriate, validates outputs against known data, records conventions and versions, and produces transparent reports. The longer-term goal is to add native components and exact algebra acceleration only where they provide a clear benefit. It is not a claim of a new theorem and not an attempt to replace established tools.

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

I am working on a Python project called Tetradrome, intended as a readable and reproducible computational workbench for knot-invariant experiments related to smooth 4-dimensional topology.

The initial target is intentionally narrow: a Conway-adjacent workflow that normalizes knot inputs, orchestrates existing tools where appropriate, validates outputs against known data, records conventions and backend versions, and produces transparent reports. The goal is not to claim a new theorem or replace established tools, but to build a disciplined computational apparatus that can reproduce and explore small examples in a way that is easier to audit.

Longer-term, I am interested in adding native Python/CUDA components for exact algebra workloads only where that adds clear value.

I have attached the project specification in case it is of interest. I would be grateful for any high-level warning signs, references, or suggestions about where such a project would be most likely to go wrong.

Best,
Randy
```

---

## 23. Success Criteria

Tetradrome v1 succeeds if it can honestly say:

1. The project has a clean architecture that separates diagrams, backends, invariants, validation, reports, and experiments.
2. Existing tooling is acknowledged and integrated where useful.
3. Spherogram/PD inputs can be normalized into a common representation.
4. KnotInfo or equivalent known-answer data can be used for validation.
5. At least one Floer backend and one Khovanov/Rasmussen backend can be called or compared.
6. Every result records backend, version, input, convention notes, raw output, normalized output, and validation status.
7. Conway-adjacent reports are reproducible.
8. The documentation clearly distinguishes computation, obstruction, theorem reference, and proof.
9. The code can grow toward native implementations without rewriting the workbench.
10. Existing brittle or research-shaped tools can be run through stable adapters with version capture, raw-output archiving, normalized schemas, and golden-master regression tests.

Tetradrome v1 fails if it produces impressive-looking results without validation, if it pretends existing computational topology tools do not exist, or if it rewrites trusted mathematical software without first capturing known behavior.

---

## 24. Summary

Tetradrome should begin as a narrow but honest workbench:

```text
Conway-focused mission
  + existing tooling integration
  + generic diagram model
  + backend abstraction
  + normalized invariant schema
  + rigorous validation
  + reproducible reports
  + legacy-tool modernization layer
  + optional native engines later
  + optional CUDA acceleration later
  + clear limitations
```

The goal is not to build the whole mathematical universe immediately. The goal is to build the calibrated bench on which such instruments can be run, compared, audited, and eventually extended.

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
