# Contributing to Tetradrome

The authoritative design and engineering rules live in `SPEC.md`, `roadmap/decisions/`, and
`CLAUDE.md`. Read those for the native compute model, the validation discipline, the
no-silent-fallback rule, and the acceleration-agreement requirements. This file covers one norm
that is easy to forget: keeping the benchmark artifact honest.

## The benchmark artifact tracks what is measurable

`BENCHMARKS.md` is generated, committed, and is the project's "why use this" page: each invariant,
Tetradrome vs the computational gold-master that already exists for it, with the math, the inputs
and outputs, the validation status, and live compute timings measured at generation time. The
generator lives in `scripts/comparison/`: `spec.py` is the hand-authored catalog, `adapters.py`
holds the oracle and Tetradrome callers, `generate.py` measures and emits.

**The rule: as soon as something becomes testable, wire it into the generator.** Do not leave a
measurable thing sitting as a static status row.

- **Implemented an invariant natively?** Point its `spec.py` entry's `tetra` field at the real
  caller and set its status to `done`. The next run times it and validates it against KnotInfo.
- **Made an oracle available** (installed it on the host, or it became pip-installable)? Give it a
  real `run` in `adapters.py` in place of the probe-only stub, so its column fills in with real
  numbers instead of "absent".
- **Adding an invariant the project aims at but does not yet compute?** Add a status-only row so
  the chart shows the ambition and the oracle's time stands as the target to aim at.

The artifact should always reflect what is actually measurable on the host that generated it, not
a hand-maintained wishlist. If you can measure it, the generator should be the thing that does.

## Regenerating it

- On any host: `python scripts/comparison/generate.py`. Only the oracles installed there light up;
  the rest read "absent (this run)". Add `--with-floer-grid` on a many-core host to also time the
  grid Floer engine.
- The comprehensive artifact is produced on a fully provisioned host (every oracle installed) via
  the throughline `tools/generate_benchmarks.py`, which runs the generator there, pulls the
  artifact back, writes it locally, and commits it.

## Speed is data, not a gate

The artifact reports times and the measured gap as data. Do not encode a speed expectation
anywhere in the code: no benchmark assertion, no "must beat the oracle" check, no CI speed gate.
Correctness, auditability, and portability outrank speed (`SPEC.md`, decisions 0006 and 0007);
trailing a compiled oracle while staying native and portable is the accepted trade. Interpreting a
gap as a target versus a problem is a human's call, not the code's.
