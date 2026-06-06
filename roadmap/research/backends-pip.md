# Research: pip backends (Spherogram, knot_floer_homology)

Empirical verification, not assumption. Findings from installing and exercising
the two SPEC 12.2 priority-1/priority-3 backends in a clean pip environment
(Python 3.12, no Sage). Feeds `docs/existing_tools.md` and `docs/backend_matrix.md`.

## Spherogram

- Version exercised: **2.4.1**; install: `pip install spherogram` (clean).
- License (pip metadata): **GPLv2+**.
- Requires: `decorator, knot_floer_homology, networkx, packaging, snappy_manifolds`.
  Note it pulls in `knot_floer_homology` transitively.
- `spherogram.Link('K11n34')`, `('4_1')`, `('3_1')` all construct correctly.
  Confirms the SPEC 12.4 identity claim that the Hoste-Thistlethwaite /
  Rolfsen names are accepted directly.

### What works standalone (no Sage) vs Sage-gated

| Method | Standalone (pip) | Notes |
|---|---|---|
| `PD_code()`  | yes | list of 4-tuples |
| `DT_code()`  | yes | |
| `seifert_matrix()` | yes | integer matrix |
| `alexander_polynomial()` | **no** | raises `SageNotAvailable` |
| `jones_polynomial()`     | **no** | raises `SageNotAvailable` |
| `signature()`            | **no** | raises `SageNotAvailable` |
| `determinant()`          | **no** | raises `SageNotAvailable` |
| `exterior()`             | **no** | needs SnapPy installed (`snappy` not pip-pulled here) |

The polynomial/invariant methods exist on the object (they appear in `dir(L)`)
but are decorated `@_sage_method` and raise unless running inside Sage. So
**under plain pip, Spherogram is a diagram layer, not an invariant source.**

## knot_floer_homology

- Version exercised: **1.2.2**; install: `pip install knot_floer_homology`
  (compiled clean under the allowed network).
- Fully **standalone** -- no Sage required.
- License: pip metadata blank; upstream (Szabo's HFKcalc) is GPL-2.0-or-later
  per the README's "Built on" -- confirm and record explicitly. [needs-confirm]
- Public callables: `pd_to_hfk`, `pd_to_morse`, `hfk`.
- `pd_to_hfk(L.PD_code())` returns a dict with keys:
  `L_space_knot, epsilon, fibered, modulus, nu, ranks, seifert_genus, tau, total_rank`.
  This matches the SPEC 12.4 `knot_floer_homology` column exactly
  (`tau, epsilon, nu, fibered, L_space_knot, ranks/total_rank, seifert_genus`).

### Known-answer spot check (sanity, not validation)

- `4_1` (figure-eight): `tau=0, epsilon=0, fibered=True, seifert_genus=1,
  L_space_knot=False` -- consistent with the literature.

## Corrections / refinements to SPEC 12.4

1. **Spherogram column is too generous.** The table implies Spherogram supplies
   `alexander_polynomial` and `signature`. It does, but **only inside Sage**.
   For the pip environment these belong to the Sage column, not Spherogram. The
   matrix should distinguish "Spherogram-in-Sage" from "Spherogram standalone."
2. **Add `seifert_matrix` as a standalone Spherogram capability.** It is a
   native-candidate route to `signature` and `determinant` without Sage
   (signature of S + S^T; determinant from the same), worth noting for the
   eventual native layer.
3. **Method names:** `PD_code()` / `DT_code()` (capitalized), not `pd_code`.
4. **knot_floer_homology column is accurate** -- no changes.

## Implications for the build

- **Diagram layer (Milestone 2)** is achievable pip-only via Spherogram
  (PD/DT codes, Seifert matrix). No Sage needed for input handling.
- **Floer v1 (Milestone 4, priority 3)** is achievable pip-only and standalone
  via `knot_floer_homology` -- the strongest "do not build from scratch" case.
- **All polynomial / Khovanov / Rasmussen invariants need Sage or KnotJob.**
  This sharpens Milestone 4: the Sage-feasibility and KnotJob spikes are on the
  critical path for anything Khovanov-side, because Spherogram alone will not
  deliver them.

## License note (for the licensing decision record)

Both backends are **GPLv2+** and are consumed by `import`, not subprocess. The
SPEC 20 / README posture (Apache 2.0, original code, call-not-vendor, optional
user-installed deps) still holds: Tetradrome vendors none of their source and
the user assembles the combination. The import-vs-subprocess nuance should be
written up in the licensing decision record rather than left implicit.

## Open items

- Confirm and record `knot_floer_homology` license explicitly. [needs-confirm]
- SnapPy install path for the `exterior()` bridge (SPEC 12.4 "Bridges").
- Sage environment feasibility (heavy; may not fit a pip-only sandbox) -- the
  gate for Khovanov-side invariants. [research]
- KnotJob CLI feasibility for Khovanov / Rasmussen s. [research]
