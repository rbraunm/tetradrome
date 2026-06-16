# Tetradrome

*A reproducible, audit-friendly Python workbench for the invariants that constrain smooth 4-dimensional topology — for knots, links, and braids alike, on one validated surface.*

Tetradrome computes the invariants of smooth 4-dimensional topology natively — the Khovanov, Lee, and knot Floer homologies, the Rasmussen *s*-invariant, and the classical and concordance invariants — for knots, links, and braids, and validates every result against KnotInfo and (where installed) the established tools, with full provenance and an explicit validation status on each one. The serious tools that already exist (Spherogram/SnapPy, `knot_floer_homology`, KnotJob, KnotInfo, SageMath) are used as the gold-master check, never as the thing that produces the answer.

The project is built around a single idea:

> Own the mathematics you compute, and check it against the instruments that already exist. Faithful and portable beats fast-but-won't-install; a validated number is the only kind worth reporting.

The motivating example is the Conway knot — historically the smallest knot whose smooth sliceness resisted classification until Lisa Piccirillo settled it in 2020 — and the trace-based machinery around it. Tetradrome is not a claim to redo that mathematics; it is a disciplined, native, auditable engine for computing, cross-checking, and reproducing invariants of this kind.

## What it is — and isn't

**It is:**
- a Python-first, reproducible workbench that computes the invariants of smooth 4-dimensional topology natively, for knots, links, and braids;
- a normalizer that attaches the method, version, conventions, raw output, and validation status to every result, in one schema;
- a validation harness — known-answer checks against KnotInfo, cross-checks against independent tools, and `d² = 0` checks for native complexes;
- a reporter that distinguishes *computation*, *obstruction*, and *theorem reference*, and never presents an unvalidated number as a fact;
- a permissively-licensed, pure-Python core with a validated multi-core / optional-GPU acceleration layer that never changes its answers.

**It isn't:**
- a claim that no tooling exists — it leans on, credits, and validates against the tools listed below;
- an orchestration layer that calls those tools to produce its answers — the compute path is native (see `roadmap/decisions/0006`);
- a generator of new theorems, or a proof assistant;
- a thing that calls a knot "slice" just because an obstruction happens to vanish.

## Status

Active, with several engines built and validated. The specification (`SPEC.md`) is complete, and the native compute path is well underway: the Jones polynomial, the shared exact-algebra back end, Khovanov homology (mod-2 and rational), Lee homology, and the Rasmussen *s*-invariant are implemented and validated against KnotInfo, behind a multi-core / optional-GPU acceleration backend whose every tier is checked to reproduce the reference exactly. The native knot Floer (grid homology) engine is in progress; the reporter, the Conway-knot reproducer, and broader link / 3–4-manifold input paths are the work ahead. Interfaces may still change.

## Design at a glance

```
input: knot / link / braid (name / PD / DT / Gauss / braid)
  → normalized diagram (native)
  → native invariant engine
  → normalized, validated result (with provenance)
  → reproducible report / claim ledger
```

- **Canonical names are the mathematics, not any tool.** Invariants are referred to by their standard names from the literature; an alignment table maps each backend's spelling onto them, so the normalizer's only job is a rename (`SPEC.md` §12.4).
- **Every result carries provenance and validation status.** Nothing is returned as a bare value (`SPEC.md` §11).
- **Downstream consumers depend on a frozen, content-hashed export.** A validated roster of knots can be built once and read forever, with no backend touched at read time.

## Using it (shape of the public API)

```python
import tetradrome as td

k = td.knots.from_name("K11n34")                      # the Conway knot
s = td.invariants.compute(k, "rasmussen_invariant")   # typed result, with provenance
verdict = td.concordance.slice_status(k)
print(verdict.smoothly_slice, verdict.certificate.via)

# off-table: present a knot by braid word (here T(2,15), beyond the 13-crossing tables)
t = td.knots.from_braid([1] * 15)
det = td.invariants.compute(t, "determinant", validate=False)   # -> 15 (no oracle off-table)

# build a validated, content-hashed roster others can depend on
roster = td.export.build(["K11n34", "4_1", "3_1"], validate=True)
td.export.save(roster, "roster-v1.json")
```

Full public surface: `SPEC.md` §13.10.

## Requirements

- Python 3.11+ (3.13 targeted). The compute path is pure Python.
- Runtime data: `database_knotinfo` (the KnotInfo tables, used as the offline validation oracle and for name resolution).
- Optional validators (authoring-time, never required to compute): KnotJob (Java) for Khovanov homology and the Rasmussen *s*-invariant, `knot_floer_homology` / HFKcalc for Floer, SageMath, Khoca. Spherogram/SnapPy is an optional interop target, not the diagram parser.

Packaging is in progress; until then, treat the list above as the environment Tetradrome expects.

## Provisioning a compute environment

The full grid scaling sweeps want more cores than a workstation. `scripts/provision_runner.py` is a small provisioning runner that stands up a clean compute environment, installs Tetradrome into it, smoke-tests it, and prints the command to run the sweep at the box's full core count — so getting from "I have a node" to "the sweep is running" is one command, not an afternoon of setup.

It is written as a general runner with a single path today: an unprivileged Debian LXC on a Proxmox node, driven over SSH. The project is baked into the script; only the *host* side is parameterized — node, storage, size, network — so it assumes nothing about a particular cluster. Other target environments (a cloud VM, bare metal, another hypervisor) are meant to be added as further paths behind the same runner.

```bash
python scripts/provision_runner.py --host root@<node> --rootfs-storage <pool> \
    --cores 16 --memory 16384
```

It creates the container, installs Tetradrome with the `accel` extra, runs a tiny smoke sweep, then prints the command to run the full sweep at the container's core count with NUMA pinning. Re-running on an existing container is refused unless `--recreate` is given — no silent clobber.

Common flags (`--help` for the full set):
- `--host root@<node>` — Proxmox node to drive over SSH (required).
- `--rootfs-storage` — storage pool for the rootfs (default `local-lvm`).
- `--cores` / `--memory` — vCPUs and RAM in MiB (default `4` / `4096`; size up for real sweeps).
- `--ctid` — container ID (default `250`).
- `--ip` / `--gateway` / `--vlan` — static networking; defaults to DHCP, untagged.
- `--branch` — repo branch to install (default `main`).
- `--recreate` — destroy and rebuild an existing CTID.

Standing up the environment is deliberately engineer-side scaffolding — one command, kept out of the way — so the work that needs a mathematician stays the focus (see *Contributing*).

## Contributing

**Tetradrome actively wants contributors — and especially wants mathematicians.**

The project is maintained by a software engineer, not a topologist. The engineering side — architecture, the validation harness, reproducibility, packaging, performance — is well-tended, and the maintainer is glad to do that work and to help solve and expand problems across the codebase. The *mathematical* depth is the real gap, and it is exactly where outside expertise is most valuable: choosing and naming invariants correctly, getting conventions and orientations right, judging when an obstruction does or does not support what a report claims, and steering the native-implementation work. On questions of mathematical correctness and interpretation, the maintainer defers to people who know the field.

Ways to help:
- **Math review** — conventions, invariant definitions, the wording of obstruction / claim / strength in reports, and the trace and concordance machinery.
- **Backend adapters** — wrapping or hardening KnotJob, SageMath, HFKcalc, Khoca, or others behind the common contract.
- **Validation data** — expanding known-answer coverage against KnotInfo and cross-backend checks.
- **Native engines** — the Khovanov / Lee–Rasmussen and knot Floer work and the exact-algebra back end, behind the validation harness.
- **Docs and examples** — making the conventions and workflows legible to newcomers.

To get started, open an issue or a discussion describing what you'd like to work on, or send a pull request. The best first contributions are small, well-scoped, and come with the known-answer or cross-backend check that demonstrates them.

### Maintainership

This is meant to be a shared project, not a solo one. Sustained, high-quality contribution earns **maintainer status, including review and merge rights** — the maintainer would rather share authority with people who understand the mathematics than gatekeep it. If you have been contributing and want a larger role, say so; that is the intended path, not the exception.

## Validation and trust

Tetradrome's whole value is that its results are checkable. Every computation records the backend, its version, the coefficient field and grading conventions, the raw output, and a validation status (known-answer match, independent-backend agreement, `d² = 0`). Reports separate what was *computed*, what is an *obstruction*, and what rests on a *theorem reference*. An impressive-looking result with no validation path is treated as a bug, not a feature.

## Built on

Tetradrome computes its own invariants, but it stands on the work of others as its validation references and historical sources, and aims to credit them clearly:

- **Spherogram / SnapPy** — Marc Culler, Nathan Dunfield, and collaborators (planar diagrams, PD/DT codes).
- **knot_floer_homology / HFKcalc** — Zoltán Szabó's HFK calculator, with the Python wrapper by Marc Culler, Nathan Dunfield, and Matthias Goerner (knot Floer cross-checks).
- **KnotJob** — Dirk Schütz (Khovanov homology, Rasmussen *s*-invariant).
- **KnotInfo** — the knot invariant database, used here as a validation oracle.
- **SageMath**, **Khoca** (Lukas Lewark), and the wider low-dimensional-topology software community.

The motivating mathematics is Lisa Piccirillo, *The Conway knot is not slice*, Annals of Mathematics **191** (2020).

## License

Apache License 2.0.

All Tetradrome code is original. The tools under [Built on](#built-on) are used only as external validators and historical references — invoked as separate programs or optional, user-installed dependencies, and consulted for the shape of their data and the mathematics they implement. None of their source or data is vendored into this repository, so no copyleft obligation attaches and a permissive license is the right fit.

---

Part of the [OneSourceIT open-source projects](https://onesourceit.us/open-source.html).
