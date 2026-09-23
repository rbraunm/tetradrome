# Backend Matrix

What each external tool can compute, how it gets onto a host, and where its details
are recorded. Tetradrome computes every invariant natively; these tools only ever
**validate** a native result or get **measured** against it in the benchmark -- never
produce a returned value (decision 0006). For canonical invariant names, see decision
0001.

## Where the live state is -- not in this file

This page records capabilities, which change rarely. What is actually wired or
measured changes at every checkpoint and is owned by code, so it is not repeated here:

- **Wired validators, and computed oracles known but not yet wired:**
  `src/tetradrome/backends/registry.py` (`_WIRED`, `_UNWIRED`). The unwired map is
  pinned exactly by `tests/test_validate_modes.py`.
- **Benchmark measurement roster:** `ORACLES` in `scripts/comparison/adapters.py`;
  which rows each oracle fills is in its run function.
- **Provisioning, versions, and build recipes:** `scripts/install_oracles.sh`.

## Capabilities

Canonical invariants only. A capability marked *verified* has been observed against
native on the chiral sweep; *unprobed* means the tool is documented or believed to
compute it but no convention probe has run, so nothing may be wired on it yet.

| Oracle | Provisioned by | Input | Canonical invariants it computes | Notes |
|---|---|---|---|---|
| `knot_floer_homology` (kfh, Szabó HFKcalc) | pip (binary wheel) | PD | `knot_floer_homology`, `ozsvath_szabo_tau`, `three_genus` -- verified | Also returns epsilon, nu, fibered and the L-space predicate from the same call. GPLv2+. |
| Regina | pip | PD only | `jones_polynomial`, `alexander_polynomial`, `determinant` (as \|Δ(-1)\|) -- verified | Also HOMFLY-PT. No signature. Cannot check a braid-word knot, which has no PD. |
| SageMath | `INSTALL_SAGE=1`, CT 250 only | PD, braid | `alexander_polynomial`, `determinant`, `signature`, `jones_polynomial`, `khovanov_homology`, `rational_khovanov_homology` -- verified | The only computed signature oracle provisioned. Re-verification tool: `scripts/verify_sage_conventions.py`. |
| KnotJob | install script (jar) | PD | `khovanov_homology` (via UCT from integral), `rational_khovanov_homology`, `rasmussen_s` -- verified | Its output also carries integral Khovanov with torsion. |
| Khoca | pip | PD | `khovanov_homology`, `rational_khovanov_homology` -- verified | Each field computed natively. Also returns the reduced theory (unused; see homology-engine.md Phase 9) and integral via ring 0. |
| knotkit (`kk`) | install script (source) | PD, name, DT, braid | `rasmussen_s`, `khovanov_homology`, `rational_khovanov_homology` -- verified | Details: `roadmap/research/knotkit.md`. |
| JavaKh | install script (source) | PD | `rational_khovanov_homology` -- unprobed as a validator | Integral / mod-2 mode not yet probed. |
| KhoHo | install script (source, PARI/gp) | torus(2,n) only today | `rational_khovanov_homology` for T(2,n) -- unprobed as a validator | A general input path is unprobed. |
| kht++ (`khtpp`) | install script (source) | `.kht` Morse word only | none today | Computes the reduced theory, which has no canonical name until homology-engine.md Phase 9. Details: `roadmap/research/khtpp.md`. |
| SnapPy / Spherogram | pip | PD, DT, braid, name | none without Sage | Classical invariants are Sage-gated under plain pip. Hyperbolic volume, which is not a canonical invariant. |
| KnotInfo (`database_knotinfo`) | pip | name | tabulated values for most canonical invariants | Not a computed oracle: rides along as a cross-check, and is the sole validator only where nothing computes the invariant (decision 0006). Details: `roadmap/research/knotinfo.md`. |

## What a plain pip environment gives you

kfh, Regina, Khoca, SnapPy and KnotInfo all install from pip with no compiler. That
covers strict validation of the Floer invariants, Jones, Alexander, determinant, and
Khovanov over F2 and Q. It does not cover signature (Sage only) or Rasmussen `s`
(KnotJob and knotkit both come from `install_oracles.sh`), so strict raises for those
on a pip-only host, naming the missing oracle, while `soft` and `off` work
(decision 0004).
