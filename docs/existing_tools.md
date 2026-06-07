# Existing Tools

The computational landscape Tetradrome builds on, and how each tool is used.
Tetradrome does not pretend this tooling does not exist; existing tools are the
gold masters until Tetradrome proves otherwise (SPEC 4.1). Roles:

- **validation data** -- a known-answer oracle.
- **callable backend** -- invoked to compute an invariant.
- **diagram layer** -- knot input / normalization.
- **reference-only** -- consulted for behavior or parity, not called.
- **future inspiration** -- relevant later, not now.

Empirically verified entries cite a research note under `roadmap/research/`;
versions there are the ones actually exercised, in a clean pip environment with no
Sage.

## Verified (pip, standalone)

| Tool | Version seen | Install | License | Role | Notes |
|---|---|---|---|---|---|
| Spherogram / SnapPy | spherogram 2.4.1 | `pip install spherogram` | GPLv2+ | diagram layer | PD/DT codes and Seifert matrix work standalone; polynomial invariants are Sage-gated. `research/backends-pip.md` |
| knot_floer_homology | 1.2.2 | `pip install knot_floer_homology` | GPL-2.0-or-later (confirm) | callable backend (Floer) | Standalone, no Sage. `pd_to_hfk` returns tau/epsilon/nu/fibered/L_space_knot/ranks/total_rank/seifert_genus. `research/backends-pip.md` |
| KnotInfo (`database_knotinfo`) | 2026.6.1 | `pip install database_knotinfo` | GPL | validation data | Offline KnotInfo table, 12,967 entries x 244 cols; ships mod-2 Khovanov data. `research/knotinfo.md` |

## Surveyed (SPEC 4), not yet exercised

| Tool | Area | Role | Notes |
|---|---|---|---|
| KnotTheory` / Knot Atlas | Mathematica, Khovanov, tables | reference-only / future inspiration | Historical ecosystem; do not claim novelty for basic Khovanov computation. |
| FastKh / JavaKh | Khovanov computation | future backend / reference | Associated with Knot Atlas; faster than the Mathematica path. |
| KnotJob | Java knot homology | callable backend (deferred spike) | Computes Khovanov-related data and Rasmussen-style invariants; KnotInfo's Khovanov data was produced with it. |
| SageMath (knots) | Python CAS | callable backend (deferred spike) | Link/knot objects, invariants, KnotInfo access; the path that unlocks Spherogram's Sage-gated invariants. |
| SageMath (chain complexes) | algebra / homology | reference-only | Reference for algebra/homology validation. |
| Khoca | Khovanov-Rozansky | future inspiration | C++/Python; higher-homology directions. |
| HFKcalc | knot Floer | reference | The C++ engine `knot_floer_homology` wraps. |
| pyknotid | 3D space-curve knots | reference-only | Geometric/visual input; not central to the first pipeline. |

## License posture

All verified backends are GPL-family and are consumed by `import` (or, for
KnotInfo, as data), never vendored. This is consistent with Tetradrome's Apache 2.0
/ call-not-vendor decision (SPEC 20): no third-party source or data lives in the
tree; the user assembles the combination by installing optional dependencies. The
import-vs-subprocess nuance is noted in `research/backends-pip.md`.

## Deferred spikes

KnotJob (Java CLI) and Sage feasibility are parked until the foundation exists; see
`roadmap/milestones.md` "Deferred." KnotInfo already supplies the Khovanov and
Rasmussen known answers the foundation needs, so neither blocks it.
