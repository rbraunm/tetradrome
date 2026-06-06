# 0002 - Python target

**Status:** Accepted.

## Context

The SPEC and README state Python 3.11+ with 3.13 targeted. The verified pip
backends (`spherogram`, `knot_floer_homology`, `database_knotinfo`) all install and
run cleanly on Python 3.12 in the research sandbox -- a useful data point that the
toolchain is not pinned to a single minor version.

## Decision

- **Minimum supported: 3.11.** Use it as the floor in `pyproject.toml`.
- **Primary development / CI: 3.13.**
- Do not adopt syntax or stdlib features newer than 3.11 in library code without a
  documented reason.

## Consequences

- `requires-python = ">=3.11"`.
- CI should run at least 3.11 (floor) and 3.13 (primary); 3.12 is known-good.
- Native-engine and algebra code stays portable across the supported range.
