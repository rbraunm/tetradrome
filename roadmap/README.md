# Tetradrome Roadmap

Planning workspace for taking `SPEC.md` from specification to code. This folder
sequences the pre-code work, holds research gathered from the existing tools, and
records design decisions. Nothing in `roadmap/` is load-bearing at runtime.

## How this folder is used

- `milestones.md` -- the SPEC Section 16 milestones as a tracked checklist.
- `claim-ledger.md` -- the living copy of SPEC Section 15. Statuses move
  Red -> Yellow -> Green -> Blue as evidence accrues.
- `research/` -- per-tool findings (real APIs, versions, install paths,
  licenses), gathered by exercising the tools, not by assumption. Feeds
  `docs/existing_tools.md` and `docs/backend_matrix.md`.
- `decisions/` -- short decision records. `0005` defines the ADR process itself,
  including the soft "locked" gate (locked means an extra are-you-sure step, never
  immutability -- every ADR stays reviewable). Topics so far: canonical invariant
  names (SPEC 12.4), Python target, native coefficient field, validate-by-default,
  no external compute backends, the faithful-raw-path rule, and the memory-prediction
  gate.
- `design/` -- design specs that deepen a SPEC section into an implementable plan.
  `homology-engine.md` covers the Khovanov/Lee/Floer computation substrate: the
  engine-vs-acceleration layering, the faithful-raw-path rule, the memory-prediction
  gate, and a general-to-reduced implementation path.

## Graduation

Artifacts mature here, then move to their permanent home in the repo layout
(SPEC Section 9): research -> `docs/`, curated knot data -> `catalog/`, schema
fixtures -> alongside the code that consumes them. The roadmap keeps the planning
trail; the repo keeps the result.

## Build order

```
research -> decisions -> docs + catalog -> schema fixtures -> scaffold -> code
```

Each step validates the one before it. No invariant code is written until the
tools it must agree with are pinned and the known-answer data it validates
against exists. Existing tools are the gold masters until Tetradrome proves
otherwise (SPEC 4.1).

## A topologist would help, but nothing waits

The math-adjacent work (the SPEC 12.4 canonical names, the trace / concordance
wording, `experiments/piccirillo_trace_notes.md`) proceeds on the same track as
everything else. We draft it from the literature and validate against the
backends as oracles. Anything not yet confirmed is marked unvalidated /
needs-review inline rather than withheld. Outside expert review accelerates and
de-risks this work; it does not gate it.
