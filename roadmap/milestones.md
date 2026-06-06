# Milestones

SPEC Section 16, as a tracked checklist. `[ ]` not started, `[~]` in progress,
`[x]` done, `[-]` deferred (post-foundation). "research" tags an item that depends
on exercising an external tool; "data" tags an item that depends on imported
known-answer values.

Current focus: Milestone 0 and Milestone 1, plus the catalog/known-answer parts
of Milestones 2 and 3. These are the pre-code deliverables.

Research status: the three pip backends -- Spherogram (diagrams),
`knot_floer_homology` (Floer), and KnotInfo via `database_knotinfo` (oracle) --
are empirically verified; see `roadmap/research/`. All three are pip-only and
standalone. The remaining "research"-tagged items below are syntheses of those
notes, not new investigation.

Deferred to post-foundation (see "Deferred" section): the Sage-feasibility and
KnotJob spikes. They gate *computing* Khovanov/Rasmussen fresh, but KnotInfo
already supplies those as known answers (including mod-2 Khovanov), so they do
not block the foundation.

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
- [ ] cube-of-resolutions enumeration
- [ ] resolution-circle detection
- [ ] chain group construction
- [ ] differential over F2
- [ ] d^2 = 0 verification
- [ ] known-answer validation against external backends

## Milestone 8 -- Native exact algebra / performance backend
- [ ] packed-bit F2 matrix representation
- [ ] CPU optimized rank computation
- [ ] optional Numba / CUDA backend
- [ ] CPU / GPU agreement tests
- [ ] benchmarks

## Milestone 9 -- Native Lee / Rasmussen experiments
- [ ] Lee deformation
- [ ] filtered-complex handling
- [ ] Rasmussen s-invariant extraction
- [ ] validation on known knots
- [ ] comparison with external backends

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
