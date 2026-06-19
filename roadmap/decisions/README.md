# Decision Records

Short records of choices that must be settled before code, because the code keys
off them. Format: Status / Context / Decision / Consequences. These are
deliberately reversible where the SPEC says so; reversibility is noted per record.

- `0001-canonical-invariant-names.md` -- the canonical names every result and
  export uses (SPEC 12.4). Gates the normalizer.
- `0002-python-target.md` -- supported Python versions.
- `0003-native-coefficient-field.md` -- the field the first native engine works over.
- `0004-validate-by-default-error-policy.md` -- validate-by-default and the loud
  error set.
- `0005-decision-record-process.md` -- how these records work.
- `0006-no-external-compute-backends.md` -- external tools validate, they don't compute.
- `0007-faithful-raw-path-no-heuristics.md` -- faithful raw path; exact reductions only.
- `0008-memory-prediction-gate.md` -- predict size, fail loud; no silent shrink-to-fit.
- `0009-scope-smooth-4d-toolset.md` -- scope is the smooth-4D toolset, not a knot calculator.
- `0010-defer-gpu-kernel.md` -- the on-device GPU rank kernel is a deferred late-project
  goal; `bitint` is the measured workhorse, so it waits for a CPU-infeasible workload.
- `0011-harden-the-brute-reference.md` -- hardening the brute reference's speed (by
  exact means) is in scope; the floor is irreducible; (B)/(A)/(C) classification; the
  4D reach frontier is the binding engine.
- `0012-performance-and-cost-architecture.md` -- names the four performance axes (backend
  tier / generation parallelism / reduction parallelism + memory budget / one cost model),
  unifies the duplicate reduction cost model onto the pivot-inclusive formula, and
  dissolves the grab-bag `scaling.py` into honest homes. Reversible.
