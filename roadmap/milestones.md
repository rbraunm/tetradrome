# Milestones

SPEC Section 16, as a tracked checklist. `[ ]` not started, `[~]` in progress,
`[x]` done. "research" tags an item that depends on exercising an external tool;
"data" tags an item that depends on imported known-answer values.

Current focus: Milestone 0 and Milestone 1, plus the catalog/known-answer parts
of Milestones 2 and 3. These are the pre-code deliverables.

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
- [ ] SageMath adapter if feasible  (research)
- [ ] KnotJob / JavaKh feasibility spike  (research)
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
