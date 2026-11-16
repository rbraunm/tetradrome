# Roadmap

Decisions, designs, research notes and the claim ledger. Nothing here is read at run time.

| Path | Holds |
|---|---|
| `decisions/` | decision records. `0005` defines how they work: a locked record takes an explicit confirmation to change and stays reviewable |
| `design/homology-engine.md` | the engines: layering, reduction tiers, faithfulness, the memory gate, the phases with their state and open work, the pinned conventions |
| `design/scheduler.md` | how every calculation runs |
| `design/coverage-map.md` | KnotInfo columns: computed, planned, out of reach |
| `design/four-manifold-objectives.md` | the 4-dimensional constructions the invariants feed |
| `claim-ledger.md` | every correctness claim, its status and its evidence |
| `research/` | per-tool findings gathered by running the tools, and the acceleration catalog |

## Order of work

An engine is built reference first: a pure, faithful computation; then validation against
its oracles; then exact reductions and faster paths, each equal to the reference. An
invariant is not done until a computed oracle for it is wired (decision 0006).

## Expert review

The mathematical wording (canonical names, trace and concordance statements,
`docs/conway_notes.md`) is drafted from the literature and checked against the oracles.
What is not confirmed is marked as needing review. Expert review speeds the work up; it does
not gate it.
