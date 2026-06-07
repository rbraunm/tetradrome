# Milestones

SPEC Section 16, as a tracked checklist. `[ ]` not started, `[~]` in progress,
`[x]` done, `[-]` deferred (post-foundation). "research" tags an item that depends
on exercising an external tool; "data" tags an item that depends on imported
known-answer values.

**Status note (native-first pivot, ADRs 0006/0007/0009).** This checklist predates
the pivot from an orchestration foundation (adapt external tools) to native
computation (own the engines; external tools are validation oracles only). The live
tracker for the homology engines is now `roadmap/design/homology-engine.md` §7:
**Phases 0–3 are done and validated** — Jones, the shared back end, native Khovanov
over F2 and ℚ, Lee, and the Rasmussen *s*-invariant, all wired into `compute()` and
checked against KnotInfo. Below, the native milestones (M7, M9) are marked done; the
external-backend milestones (M4 adapters, M4A modernization harness, the Sage/KnotJob
spikes) are **superseded** — Tetradrome computes these natively and only validates
against KnotInfo. The remaining open work is acceleration (M8 = engine Phase 5), the
report generator (M5), the Conway reproducer (M6), and native Floer (engine Phase 6).

---

## Milestone 0 -- Project scaffold
- [~] repository structure (roadmap skeleton landed; source layout pending)
- [ ] package metadata (`pyproject.toml`)
- [ ] test framework chosen and wired
- [ ] documentation skeleton (`docs/`)
- [ ] coding conventions (`docs/conventions.md`)
- [ ] claim ledger (`roadmap/claim-ledger.md` -- landed; keep current)

## Milestone 1 -- Existing tooling integration map
- [~] `docs/existing_tools.md`  (research)
- [ ] `docs/backend_matrix.md`  (research)
- [ ] install notes per tool  (research)
- [ ] license notes per tool  (research)
- [ ] classification: reference-only / callable backend / validation data / future inspiration

## Milestone 2 -- Diagram input and catalog
- [ ] Spherogram adapter  (research)
- [ ] PD-code normalizer
- [ ] hardcoded knot catalog (`catalog/knots.yaml`)
- [ ] unknot / trefoil / figure-eight examples
- [ ] Conway / Kinoshita-Terasaka entries if reliable encodings exist  (research)

## Milestone 3 -- Validation data import
- [ ] KnotInfo importer or manual known-answer loader  (research)
- [ ] known-answer schema (`catalog/known_answers.yaml`)  (data)
- [ ] comparison tool
- [ ] reproducible source metadata (`catalog/sources.yaml`)

## Milestone 4 -- First backend adapters
- [ ] `knot_floer_homology` adapter  (research)
- [-] SageMath adapter if feasible  (deferred -- see Deferred section)
- [-] KnotJob / JavaKh feasibility spike  (deferred -- see Deferred section)
- [ ] backend availability checks
- [ ] normalized output schema

## Milestone 4A -- Modernization harness
- [ ] adapter contract for external tools
- [ ] backend installation probes
- [ ] raw-output capture format
- [ ] parser fixtures per integrated tool
- [ ] golden-master outputs (unknot, trefoil, figure-eight, selected small knots)
- [ ] container / environment notes for brittle dependencies
- [ ] migration matrix (wrapper-only / reproducible / native-candidate / not worth replacing)

## Milestone 5 -- Report generator
- [ ] computation report template
- [ ] backend comparison report
- [ ] claim-ledger report
- [ ] reproducibility hash
- [ ] raw-output capture

## Milestone 6 -- Conway workflow reproducer
- [ ] Conway-knot input
- [ ] Kinoshita-Terasaka input
- [ ] Piccirillo-related K' input if reliably encoded
- [ ] Conway-adjacent computation report
- [ ] explicit statement of what it does and does not establish

## Milestone 7 -- Native mod-2 Khovanov engine
- [x] cube-of-resolutions enumeration
- [x] resolution-circle detection
- [x] chain group construction
- [x] differential over F2
- [x] d^2 = 0 verification
- [x] known-answer validation against external backends
  *(done & validated; also extended to a signed ℚ lane -- see engine Phase 2/3a)*

## Milestone 8 -- Native exact algebra / performance backend
*Candidate accelerations (per engine, per phase, with provenance) are catalogued in
`roadmap/research/engine-acceleration-catalog.md`, governed by ADR 0011.*
- [ ] packed-bit F2 matrix representation
- [ ] CPU optimized rank computation
- [ ] optional Numba / CUDA backend
- [ ] CPU / GPU agreement tests
- [ ] benchmarks

## Milestone 9 -- Native Lee / Rasmussen experiments
- [x] Lee deformation
- [x] filtered-complex handling
- [x] Rasmussen s-invariant extraction
- [x] validation on known knots  *(vs KnotInfo, s = 0, ±2, ±4, ±6)*
- [x] comparison with external backends  *(KnotInfo oracle, up to the documented mirror)*

## Milestone 10 -- External review package
- [ ] compact technical summary
- [ ] reproducibility instructions
- [ ] validation report
- [ ] limitations page
- [ ] outreach note

---

## Deferred (post-foundation)

These are real work items, intentionally parked until the foundation (diagram
layer + Floer backend + KnotInfo oracle + validation harness + reports) exists.
They are not part of the roadmap flesh-out.

- [-] Sage feasibility spike -- can Sage stand up outside a heavy environment, and
  is `Spherogram-in-Sage` a viable path to the polynomial/Khovanov invariants its
  pip build gates behind Sage? Marker tag `[deferred]`.
- [-] KnotJob / JavaKh CLI spike -- reliable command-line invocation for computing
  Khovanov homology and the Rasmussen `s` invariant fresh.

Why safe to defer: KnotInfo already supplies Khovanov (incl. mod-2) and Rasmussen
`s` as known answers for tabulated knots, so the foundation and its validation can
be built and trusted without either spike. They become necessary only when
computing those invariants for knots not in the tables (the eventual
Conway-adjacent / native-engine work).
