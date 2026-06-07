# 0011 - Harden the brute reference; exact-only acceleration is in scope; the 4D reach frontier

**Status:** Accepted

## Context

Decision 0007 settled *which* computation is the source of truth — the raw,
unreduced path — and that only exact, answer-preserving reductions may accelerate it.
What it did not settle is whether investing effort in the **speed** of that brute path
is worthwhile. Read alongside 0007's "correctness and generality outrank speed" and
0010's "do not out-engineer an already-excellent reducer until a workload demands it,"
the standing ADRs lean *against* optimizing the reference. That reading recurs as a
debate — most concretely, "why pour effort into the n! grid-Floer path when the field
has faster routes?" — and it deserves a settled answer rather than relitigation each
time a new engine lands.

The answer turns on what the brute reference *is* for this project. It is not only a
correctness backstop whose slowness is tolerated. On the inputs the workbench exists to
study — the Conway-class outliers where the standard invariants go silent (0009) — the
brute path is the one you fall to when every shortcut declines to answer, and its speed
also gates how much cross-checking the validation discipline can afford. Its
performance is therefore a first-class concern, not an accepted tax.

This must be reconciled with two facts. First, the brute reference is bounded by its
own combinatorics — grid Floer is n!, Khovanov is the 2^c cube of resolutions — and no
exact transformation escapes that growth; the boundedness is precisely what makes the
path trustworthy (it does the obvious thing, with no cleverness that could be subtly
wrong). Second, 0010 already deferred a GPU *reduction* kernel on the measured finding
that `bitint` wins at current sizes. The present decision is broader than 0010: it
governs the whole reference path — generation included, which is currently the dominant
cost — and the standing question of whether to harden it at all.

## Decision

- **Hardening the brute reference's performance, by exact answer-preserving means, is
  explicitly in scope and valued.** This does not contradict 0007: correctness still
  outranks speed, and the reference stays the most general path. It refines 0007 by
  stating that making that path *faster* — without changing what it computes — is
  legitimate, ongoing work, because the reference's speed sets the reachable envelope
  on the cases that matter and bounds validation throughput. It refines 0010 by scoping
  that deferral to the GPU reduction kernel at present sizes, not to reference
  performance in general.

- **The floor is irreducible, and we do not pretend otherwise.** Each engine's brute
  reference is bounded by its combinatorial explosion. Exact acceleration is
  constant-factor: it lowers the constant and pushes the reachable size out by a few
  units (turning, say, feasible-at-n into feasible-at-n+2), and never bends the growth
  rate. We therefore do not chase "speedups" that are really heuristics or route
  changes dressed as optimizations (0007); a guaranteed-correct wall is still a wall,
  and that is the point of having one.

- **Every proposed speedup is classified before it is built**, generalizing 0007's
  delooping / local-elimination allowance into a rule:
  - **(B) Answer-preserving transforms of the same complex** — permitted core
    optimizations, optional and toggleable, verified `== reference` (0004, 0007). This
    class is larger than 0007 spelled out and spans both phases: delooping and local
    (Gaussian) elimination; clearing / the twist (use d² = 0 to skip vanishing
    columns); implicit reduction (recompute columns instead of storing the matrix);
    incremental / Gray-code generation (consecutive states differ by a transposition,
    so grading and differential update locally); iterative sparse rank over the field
    (block Lanczos / Wiedemann, from number-field-sieve linear algebra); and
    discrete-Morse cancellation (Bar-Natan-style "Gaussian elimination made abstract,"
    from computational topology). Provenance outside knot theory does not disqualify a
    technique; being answer-preserving qualifies it. **Generation accelerations count,
    not only reduction** — 0007 reads reduction-flavored, but the generation phase is in
    scope for this class on the same terms.
  - **(A) Alternative algorithms that change the route** — bordered / tangle Floer and
    the like — carry applicability conditions and a harder trust surface. They are
    permitted only as **separate, validated fast paths that must agree with their own
    brute reference**, never as the core and never as the reference itself.
  - **(C) Input reduction that changes which complex is built** — diagram
    simplification (Cromwell moves, destabilization) that presents a smaller-n diagram
    of the same object — is permitted and can be a factorial win, but it depends on a
    smaller representation existing, so it is **never the floor**: the brute reference
    must remain runnable on the input as given.

- **The 4D reach frontier is the binding-engine rule.** The workbench's reachable
  region in smooth 4D topology is the *intersection* of each required invariant's exact
  reach. Hardening an engine moves the frontier only when that engine is the binding
  constraint for the target at hand; effort ordering follows the binding wall, not
  whichever engine is most familiar. **Every new engine inherits this**: it owns a
  brute reference plus a (B)-class acceleration toolkit, is judged on its exact reach,
  and treats any (A) route or (C) input reduction as an adjunct that must agree with
  that reference.

## Consequences

- The "should we invest in the brute path / can we beat n!" debate is settled and
  citable; new-engine authors get one rule set instead of re-deriving it. The honest
  expectation — constant-factor gains on an irreducible floor — is on record, so the
  effort is neither dismissed as pointless nor oversold as breaking the wall.
- The classification rule makes "is this allowed in the core?" mechanical: answer-
  preserving and same-complex is in; route-changing or input-changing is an adjunct
  behind its own agreement check. Cross-domain techniques (crypto-NFS sparse rank, TDA
  reduction tricks) are admissible on their exactness, which widens the toolbox without
  weakening the discipline.
- Effort is directed by the binding-engine rule, so reference hardening is spent where
  it actually moves the 4D frontier rather than on the most legible engine.
- This is consistent with the surrounding decisions and does not disturb them: 0007
  (faithful path, exact-only) and 0009 (4D scope) are the foundations this builds on;
  0008 (memory gate) and 0010 (deferred GPU kernel) continue to govern their areas, and
  the deferred kernel is simply a (B)-class accelerator that still earns its place only
  by `== reference`, never by being fast.
- Natural candidate to be **Locked** once stable (0005): this is the kind of
  load-bearing principle whose reversal should be deliberate and recorded. It starts
  Accepted.
