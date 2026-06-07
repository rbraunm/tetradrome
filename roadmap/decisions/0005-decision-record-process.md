# 0005 - How decision records work (the ADR about ADRs)

**Status:** Accepted

## Context

The project records choices as numbered ADRs in `roadmap/decisions/` (0001 onward),
but the process itself was never written down. As ADRs accumulate and some become
load-bearing, two things need stating: what an ADR's statuses mean, and — the point
that prompted this record — that a "locked" ADR is never immutable.

The failure mode being guarded against is treating a settled decision as
unchallengeable. This is a research-shaped project; a call that is right today can be
wrong once evidence accrues. Decisions must stay reversible, while still being stable
enough that they are not casually or accidentally undone.

## Decision

An ADR is a short record of a choice with lasting consequences: **Context, Decision,
Consequences**, one decision per file, prose, numbered, kept in
`roadmap/decisions/`.

**Status lifecycle:**

- **Proposed** — under discussion; not yet in force.
- **Accepted** — in force; amend freely by editing the file (the change is recorded
  in git history).
- **Locked** — in force and load-bearing enough that reversing or materially changing
  it requires an explicit "are you sure?" confirmation step and a recorded rationale
  in the ADR. **Locked is not immutable.** A locked ADR remains fully reviewable and
  reversible at any time; the lock is friction against casual or accidental reversal,
  nothing more.
- **Superseded by NNNN** — replaced; the body stays for the historical trail and
  points to its successor.

**Locking is a deliberate act, not a default.** New ADRs start Proposed or Accepted;
an ADR becomes Locked only by explicit decision.

This ADR is itself subject to this policy — it can be amended or superseded under its
own rules. (It is a natural candidate to be Locked once stable, which would mean only
that changing *how ADRs work* gets the same are-you-sure gate as any other locked
decision.)

Existing ADRs 0001–0004 are Accepted under this scheme; no status change is needed.

## Consequences

- "Locked" buys stability without ossification: the gate prevents drift while keeping
  every decision open to new evidence.
- Git history is the audit trail; an ADR body always reflects the current decision,
  plus — where useful — why it changed.
- No ADR is ever a closed door. The project can revisit any call when the math or the
  engineering warrants it; the only difference a lock makes is that doing so is
  deliberate and recorded.
