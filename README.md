# Tetradrome

*A reproducible, audit-friendly Python workbench for the invariants that constrain smooth 4-dimensional topology — for knots, links, and braids, on one validated surface.*

Tetradrome computes these invariants natively — Khovanov, Lee and knot Floer homology, the Rasmussen *s*-invariant, τ, and the classical invariants — and validates every result against independent tools and KnotInfo, with full provenance on each one. The tools that already exist (`knot_floer_homology`, KnotJob, Regina, Khoca, knotkit, SageMath, KnotInfo) are the check, never the thing that produces the answer.

> Own the mathematics you compute, and check it against the instruments that already exist. A validated number is the only kind worth reporting.

The motivating example is the Conway knot, whose smooth sliceness Lisa Piccirillo settled in 2020, and the trace-based machinery around it (`docs/conway_notes.md`).

## What it is — and isn't

**It is:**
- a workbench that computes its invariants with its own code and runs every calculation through one scheduler;
- a schema that attaches the method, the versions, and each validator's verdict to every result;
- a validation harness: computed oracles, KnotInfo, and `d² = 0` checks on native complexes;
- permissively licensed, with a pure-Python reference path for every engine and faster paths that must return the identical answer.

**It isn't:**
- a claim that no tooling exists — it credits and validates against the tools listed below;
- a layer that calls those tools to produce its answers (`roadmap/decisions/0006`);
- a generator of new theorems, or a proof assistant;
- a thing that calls a knot "slice" because an obstruction happens to vanish.

## Status

| | |
|---|---|
| Computed and validated | determinant, signature, Alexander polynomial, Jones polynomial, Khovanov homology over F2 and ℚ, Rasmussen *s*, knot Floer homology (HFK-hat), three-genus, τ |
| Limits | the Floer invariants need a tabulated knot; grid number 10 is the largest that has fit in memory |
| Planned | ε, ν, Υ; reduced Khovanov; reports; the Conway reproducer; links and 4-manifold constructions |

Detail: `roadmap/claim-ledger.md` (every claim and its evidence) and `roadmap/design/homology-engine.md` section 7 (phases and open work). Interfaces may still change.

[BENCHMARKS.md](BENCHMARKS.md) is generated: per invariant, Tetradrome against the existing calculators and KnotInfo, with validation status and measured times.

## Design

```
input: name / PD code / braid word
  → normalized diagram
  → native invariant engine, run as scheduler jobs
  → validated result with provenance
```

| Topic | Where |
|---|---|
| Scope, principles, result schema, validator contract, canonical names, public API | `SPEC.md` |
| Decisions | `roadmap/decisions/` |
| Engines, and how calculations run | `roadmap/design/homology-engine.md`, `roadmap/design/scheduler.md` |
| Conventions and validation | `docs/conventions.md`, `docs/validation.md` |
| What each oracle computes | `docs/backend_matrix.md` |

## Using it

```python
import tetradrome as td

k = td.knots.from_name("K11n34")                 # the Conway knot
s = td.invariants.compute(k, "rasmussen_s")     # strict by default: typed result, with provenance
print(s.value)                                  # 0
print([(v.oracle, v.verdict) for v in s.validation.validators])

# off-table: present a knot by braid word (here T(2,15), beyond the 13-crossing tables)
t = td.knots.from_braid([1] * 15)
det = td.invariants.compute(t, "determinant", validate="off")   # -> 15, empty validation record
```

Validation is a mode (decision 0004). `"strict"`, the default, requires a computed oracle
wherever one exists and raises rather than return an unchecked value. `"soft"` falls back to
KnotInfo when the computed oracle is absent. `"off"` skips validation and still records
provenance. The off-table call above needs `"off"` unless SageMath is installed: a braid
word has no PD, Regina reads only PDs, and KnotInfo has no row for T(2,15), so under strict
or soft it raises `UnvalidatedResult` naming the missing oracle. Knots can also be given as
PD codes with `td.knots.from_pd`.

## Requirements

- Python 3.11+ (3.13 targeted).
- `database_knotinfo`: the KnotInfo tables, for name resolution, tabulated diagrams and tabulated answers.
- numpy for the Floer invariants; Numba and a CUDA GPU with cupy are optional and change no answer. The other invariants need the standard library only. Extras are defined in `pyproject.toml`.
- Validators are optional and never compute a returned value; strict mode requires the ones that exist for an invariant. `scripts/install_oracles.sh` provisions them (`docs/backend_matrix.md`).

## Provisioning a compute environment

`scripts/provision_runner.py` stands up a clean compute environment, installs Tetradrome with every CPU tier and the comparison oracles, smoke-tests it, and prints the commands for the full runs. It has one path today: an unprivileged Debian LXC on a Proxmox node, driven over SSH. Only the host side is parameterized.

```bash
python scripts/provision_runner.py --host root@<node> --rootfs-storage <pool> \
    --cores 16 --memory 32768
```

The runner reads the node's NUMA and hyperthread layout and pins the container to whole physical cores, the same number from every NUMA node, because the scheduler places work per NUMA node and treats every CPU it is allowed as a core. Re-running on an existing container is refused unless `--recreate` is given.

Common flags (`--help` for the full set):
- `--host <user>@<node>` — login on the Proxmox node, prompted for its password (required). `root`, or any account with passwordless sudo.
- `--cores` — logical CPUs (required): a multiple of NUMA nodes × threads per core.
- `--memory` — RAM in MiB (required); the heavy test tier wants 32768 or more. Refused if the node does not have that much available.
- `--rootfs-storage` — storage pool for the rootfs (default `local-lvm`).
- `--ctid` — container ID (default `250`).
- `--ip` / `--gateway` / `--vlan` — static networking; defaults to DHCP, untagged.
- `--branch` — repo branch to install (default `main`).
- `--recreate` — destroy and rebuild an existing CTID.

## Contributing

**Tetradrome wants contributors, and especially mathematicians.**

The project is maintained by a software engineer, not a topologist. The mathematical depth is the gap: choosing and naming invariants correctly, getting conventions and orientations right, judging when an obstruction supports what a report claims, and steering the native engines. On questions of mathematical correctness and interpretation, the maintainer defers to people who know the field.

Ways to help:
- **Math review** — conventions, invariant definitions, the wording of obstruction / claim / strength, and the trace and concordance machinery.
- **Validators** — wiring or hardening an oracle behind the validator contract (`SPEC.md` section 6).
- **Validation data** — wider known-answer and cross-oracle coverage.
- **Native engines** — the Khovanov / Lee and knot Floer work and the exact-algebra back end.
- **Docs and examples.**

Open an issue or a discussion describing what you'd like to work on, or send a pull request. The best first contributions are small and come with the check that demonstrates them. Engineering rules are in `CONTRIBUTING.md`.

Sustained, high-quality contribution earns maintainer status, including review and merge rights.

## Validation and trust

Tetradrome's value is that its results are checkable. Every result records its method, the versions that can move it, each validator's identity, version and verdict, and the `d² = 0` check on a native complex. A result with no validation path is treated as a bug.

## Built on

Tetradrome computes its own invariants and stands on the work of others as its validation references:

- **Spherogram / SnapPy** — Marc Culler, Nathan Dunfield, and collaborators.
- **knot_floer_homology / HFKcalc** — Zoltán Szabó's HFK calculator, with the Python wrapper by Marc Culler, Nathan Dunfield, and Matthias Goerner.
- **KnotJob** — Dirk Schütz (Khovanov homology, Rasmussen *s*-invariant).
- **KnotInfo** — the knot invariant database.
- **SageMath**, **Khoca** (Lukas Lewark), and the other oracles listed in `docs/backend_matrix.md`.

The motivating mathematics is Lisa Piccirillo, *The Conway knot is not slice*, Annals of Mathematics **191** (2020).

## License

Apache License 2.0. All Tetradrome code is original. The tools under [Built on](#built-on) are external validators: separate programs or optional, user-installed dependencies. None of their source or data is in this repository (`SPEC.md` section 9).

---

Supported in part by the work of [OneSourceIT](https://onesourceit.us/open-source.html).
