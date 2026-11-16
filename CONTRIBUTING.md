# Contributing to Tetradrome

The design and engineering rules are in `SPEC.md` and `CLAUDE.md`:
native compute, validation, no silent fallbacks, and identical answers from every faster
path. This file adds three rules.

## Oracles come from one script

`scripts/install_oracles.sh` is the single source of every validator on a host. Run it at
the start of a session and do not hand-install an oracle: a stray `pip install` or manual
build leaves a version the run never recorded, and the recorded version is the
reproducibility.

- `scripts/install_oracles.sh` converges the host: installs what is missing, applies
  upstream updates, rebuilds a source oracle only when its upstream moved, re-smokes only
  what changed, and prints the version of every oracle it converged to.
- `scripts/install_oracles.sh --check` is a dry run that reports each oracle's version and
  whether an update is available; a nonzero exit means something is missing.
- `VERIFY=1 scripts/install_oracles.sh` re-runs every smoke even when converged.
- SageMath is the one oracle not provisioned in the sandbox; its apt tree is CT 250 work
  (`INSTALL_SAGE=1` there).

Adding an oracle is an edit to this script: a version-probed install plus a smoke.

## The benchmark artifact tracks what is measurable

`BENCHMARKS.md` is generated and committed: each invariant, Tetradrome against the
calculator that already exists for it, with validation status and times measured at
generation. The generator is in `scripts/comparison/`: `spec.py` is the hand-written
catalog, `adapters.py` holds the oracle and Tetradrome callers, `generate.py` measures and
writes.

**As soon as something becomes testable, wire it into the generator.**

- **Implemented an invariant natively?** Point its `spec.py` entry's `tetra` field at the
  real caller and set its status to `done`.
- **Made an oracle available?** Give it a real `run` in `adapters.py` in place of the
  probe-only stub.
- **Adding an invariant the project aims at but does not compute?** Add a status-only row,
  so the oracle's time stands as the target.

To regenerate: `python scripts/comparison/generate.py` on any host measures the oracles
installed there and reports the rest as "absent (this run)". The full artifact comes from a
fully provisioned host through `tools/generate_benchmarks.py`, which runs the generator
there, pulls the artifact back and commits it.

## Speed is data, not a gate

The artifact reports times and the measured gap as data. Do not encode a speed expectation
anywhere in the code: no benchmark assertion, no "must beat the oracle" check, no CI speed
gate. Trailing a compiled oracle while staying native and auditable is the accepted trade
(`SPEC.md` section 2).
