# 0013 - Stamp every artifact with reproducible provenance

**Status:** Accepted

## Context

The project exists to be reproducible and audit-friendly: any result should be
re-derivable and checkable after the fact. That requirement was only partly honored in
what actually gets recorded. Results carried the computing backend and its version, but a
validated result did not record which oracle validated it or at what version, the
comparison artifact captured versions for only two of the tools it ran, and third-party
library versions were not captured at all, even though a change in a dependency can change
an answer. Provenance was treated as metadata. It is the deliverable.

This also reinforces the native-first stance (ADR 0006): keeping computation in our own
code, with external tools confined to validation, keeps the provenance chain as short as
it can honestly be. Every backend relied on to produce an answer is another version whose
drift could move the result and another link the audit must carry; native-first collapses
that chain, and this decision stamps whatever remains.

## Decision

Anything the project produces is stamped with the provenance required to reproduce it.
Provenance is complete only when it names every version that materially shaped the result:

- **The mathematics**: the backend that computed the answer and its version, the method,
  and the inputs.
- **The validators**: for every oracle consulted, its identity, version, and verdict.
  KnotInfo is one validator among the computed oracles, not a special case.
- **The libraries**: the versions of the third-party libraries and tools pulled in that
  can affect the result. If we depend on it and it can change an answer, its version is
  logged.

Provenance is never omitted for size. "The data is large" or "this bloats the record" is
not a reason to drop it: the reproducibility is the data. Missing provenance is a defect,
not an acceptable trade-off; where volume is a genuine concern the answer is compression or
storage layout, never dropping versions. Updating a dependency or an oracle therefore takes
no ceremony: the next run records the new version automatically, which is why we apply
updates rather than pin.

## Consequences

- Result and artifact schemas carry versioned provenance for computation, validation, and
  the relevant library and tool set. Concretely this session: the oracle adapter contract
  gains a `version()`, `ValidationStatus` records versioned validator entries rather than
  bare pass/fail, the comparison artifact iterates every oracle for its version, the
  compute path records the versions of the computational libraries it imports, and
  `install_oracles.sh` reports the version set it converged to.
- Reinforces ADR 0006: the shorter the compute chain, the fewer versions can silently move
  a result, so native-first is a reproducibility argument and not only a license or
  portability one.
- A result can always be traced to the exact backend, oracle, and dependency versions
  behind it, which is the property the project exists to provide.
- Every result-producing path populates this provenance; existing paths are migrated as the
  schema changes rather than left partial.
