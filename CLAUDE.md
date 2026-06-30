# CLAUDE.md -- Tetradrome

A reproducible, audit-friendly, Python-first workbench for the invariants that constrain
smooth 4-dimensional topology -- the Khovanov, Lee, and knot Floer homologies, the
Rasmussen *s*-invariant, tau, epsilon, nu, and the classical and concordance invariants --
for knots, links, and braids, on one normalized, fully-provenanced surface.

Authoritative design lives in `SPEC.md`, `roadmap/decisions/` (esp. 0006, 0007, 0009),
and `roadmap/design/homology-engine.md`. Those govern on any conflict with this file or
with older "orchestration" language surviving elsewhere in the tree.

## Core principles (these are the project, not preferences)
- **Native compute path (decision 0006).** Tetradrome computes each invariant with its own code. Existing tools (Spherogram/SnapPy, `knot_floer_homology`, KnotJob/Khoca, KnotInfo, SageMath) are opt-in validators and gold-master checks ONLY -- never a runtime dependency of a computation. Do not add an orchestration path that calls them to produce an answer.
- **No heuristics in the core (decision 0007).** Only exact, answer-preserving reductions may optimize a computation. The raw, faithful path is first-class and always runnable.
- **Never present an unvalidated number as a fact.** Every result carries method, version, conventions, raw output, and an explicit validation status. The reporter distinguishes computation, obstruction, and theorem reference.

## Acceleration: agreement discipline (non-negotiable)
The core is built to be accelerated (JIT, multi-core/NUMA, optional GPU) WITHOUT changing
its answers.
- Every acceleration tier MUST reproduce the reference result exactly -- bit-for-bit, validated independently per tier. A faster tier that disagrees is broken, not "approximate". `d^2 = 0` checks on native complexes hold regardless of tier.
- The GPU dispatch threshold is a calibratable knob, never a hidden default. VRAM-aware routing reads available VRAM rather than hardcoding a cutoff; the right threshold differs by card.

Per-tier agreement traps to guard against:
- JIT (Numba) reducer: today's only Numba tier is the F2 packed reducer, which is mod 2 (XOR over uint64 words) and so immune to integer overflow -- this trap does not apply to it. It applies to a FUTURE multimodular Q reducer: Numba compiles to fixed-width ints and wraps silently on overflow, where the pure-Python reference uses arbitrary precision. When that Q reducer is added, ensure intermediate products (a*b before reduction) cannot overflow int64 under the chosen moduli, and that its agreement tests include inputs large enough to expose overflow -- not just small cases.
- GPU dense kernel: arithmetic is exact (GF(2)/integer, no float), so any disagreement is from reduction order or races -- not rounding. Today the packed-gpu tier runs on the device whenever it is selected; there is no size-based router yet, so GPU agreement is validated on large inputs with a positive device-execution check (the kernel must allocate the device matrix). The VRAM dispatch threshold and CPU fallback are FUTURE/intended design: when that router is added, agreement tests must also exercise inputs on both sides of the threshold, so the GPU path and the CPU fallback are both covered and shown to agree.

## No silent fallbacks
If a requested GPU/JIT/NUMA path is unavailable, fail loudly with a clear message. Never
silently fall back to the reference path and report success -- that hides a broken
environment and produces misleading benchmarks.

## Speed is subordinate to correctness (warn, never gate)
Tetradrome's thesis is auditability and correctness over speed (SPEC, decisions 0006/0007):
a validated number beats a fast one, and faithful-but-slow beats fast-but-wrong or
fast-but-won't-install. The comparison artifact (tetradrome vs the gold-master oracles)
reports times and the measured gap as DATA -- never a pass/fail, threshold, or gate. Do NOT
encode a speed expectation anywhere in the application: no benchmark assertion, no "must beat
the oracle" check, no CI speed gate. The generator stays neutral and just measures.

The warning is Claude's job, in conversation, not the code's. When tetradrome trails an oracle
on math we compute, SAY SO -- frame it as a target to aim for, with the likely reason (compiled
C/Java vs portable Python, or the n! grid vs a polynomial cube) and the roadmap lever that would
close it (e.g. the Szabo-cube Floer engine). It is a target, NEVER a critical failure: trailing a
compiled oracle while staying native, auditable, and portable is the expected, accepted trade. An
oracle's speed on math we have NOT implemented is also a target -- record it as the bar a future
engine aims at.

## Hardware-dependent tiers
Exercise each only where its hardware exists; the agreement check must pass before any
benchmark number means anything.
- Numba JIT reducer -- any CPU.
- Batched dense GPU kernel -- a CUDA-capable GPU. Calibrate the VRAM threshold to the actual card; small-VRAM results are a checkpoint, not the final calibration for a larger card.
- NUMA-aware core pinning -- a multi-socket host; a no-op on single-socket machines.

Local validation order: baseline pytest; GPU detection + cupy install; JIT agreement
tests; NUMA pinning benchmarks.

## Running commands on a provisioned box (tools/ct_exec.py)
`scripts/provision_runner.py` stands up a compute container; `tools/ct_exec.py` runs commands
on it over SSH using the key and login the provisioner wrote beside itself
(`scripts/ctNNN-ssh-key`, `scripts/ctNNN-ssh-credentials.txt`). It defaults to `--ctid 250`,
needs no password, streams output live, and exits with the remote command's status.
- Put the command after `--` or quote it:

      python tools/ct_exec.py -- nproc
      python tools/ct_exec.py -- "cd /opt/tetradrome/src && venv/bin/python -m pytest -q"

- Invoke it as a SINGLE standalone command so it matches the `Bash(python tools/ct_exec.py:*)`
  allow rule. Do not glue it to other local commands with `;`/`&&`/`echo`/`cd` -- those extra
  segments are not in the allow list and force a permission prompt. Keep any sequencing INSIDE
  the remote argument (it runs in the container's shell). Chaining two `ct_exec` calls is fine
  (each segment matches), but the single remote-arg form is preferred.
- Long jobs: launch detached and poll the log as two separate ct_exec calls, so the run is not
  tied to the SSH session:

      python tools/ct_exec.py -- 'cd /opt/tetradrome/src && setsid venv/bin/python <cmd> > /tmp/job.log 2>&1 < /dev/null & echo PID=$!'
      python tools/ct_exec.py -- 'cat /tmp/job.log'

## Testing
Tests make real assertions about computed invariants and complexes. No monkeypatching the
logic under test. Tests verify behavior; they never drive design.

## Python
Invoke as `python`, not `python3`.

---
Global standards (git workflow, fail-loud philosophy, no dead code, PowerShell rules) live
in the user-level `~/.claude/CLAUDE.md`. Transient session state and host-specific notes
live in `CLAUDE.local.md` (gitignored), not here.
