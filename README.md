# Tetradrome

*A reproducible, audit-friendly Python workbench for the invariants that constrain smooth 4-dimensional topology — for knots, links, and braids alike, on one validated surface.*

Tetradrome builds, validates, reproduces, and reports knot-invariant computations — Khovanov homology, the Rasmussen *s*-invariant, knot Floer homology, and the classical and concordance invariants — by orchestrating the serious tools that already exist (Spherogram/SnapPy, `knot_floer_homology`, KnotJob, KnotInfo, SageMath) behind one normalized schema with full provenance and an explicit validation status on every result.

The project is built around a single idea:

> Build the bench before building every instrument. Use existing instruments honestly, validate them against one another, and only machine new parts where Tetradrome adds real value.

The motivating example is the Conway knot — historically the smallest knot whose smooth sliceness resisted classification until Lisa Piccirillo settled it in 2020 — and the trace-based machinery around it. Tetradrome is not an attempt to redo that mathematics; it is a disciplined engineering layer for running, cross-checking, and reproducing computations of this kind.

## What it is — and isn't

**It is:**
- a Python-first, reproducible workbench around existing knot-homology tooling;
- a normalizer that turns each backend's output into one schema, with backend, version, conventions, raw output, and validation status attached to every result;
- a validation harness — known-answer checks against KnotInfo, cross-backend agreement, and `d² = 0` checks for native complexes;
- a reporter that distinguishes *computation*, *obstruction*, and *theorem reference*, and never presents an unvalidated number as a fact;
- extensible toward native Python (and, later, CUDA) implementations of exact-algebra workloads, added only behind the validation harness.

**It isn't:**
- a claim that no tooling exists — it leans on, and credits, the tools listed below;
- a from-scratch reimplementation of Khovanov or Heegaard-Floer theory;
- a generator of new theorems, or a proof assistant;
- a thing that calls a knot "slice" just because an obstruction happens to vanish.

## Status

Early. The specification (`SPEC.md`) is complete — architecture, result schema, public API surface, backend strategy, and validation philosophy are all laid out — and the implementation is being built behind it, simplest cases first. Interfaces may still change.

## Design at a glance

```
knot input (name / PD / DT / Gauss / braid)
  → normalized diagram (Spherogram-backed)
  → one or more backends
  → normalized, validated invariant result (with provenance)
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

- Python 3.11+ (3.13 targeted).
- Pip-installable backends: `spherogram` / `snappy` (diagrams), `knot_floer_homology` (Floer invariants).
- Optional / authoring-time backends: KnotJob (Java) for Khovanov homology and the Rasmussen *s*-invariant, SageMath, KnotInfo data, HFKcalc, Khoca.

Packaging is in progress; until then, treat the list above as the environment Tetradrome expects.

## Contributing

**Tetradrome actively wants contributors — and especially wants mathematicians.**

The project is maintained by a software engineer, not a topologist. The engineering side — architecture, the validation harness, reproducibility, packaging, performance — is well-tended, and the maintainer is glad to do that work and to help solve and expand problems across the codebase. The *mathematical* depth is the real gap, and it is exactly where outside expertise is most valuable: choosing and naming invariants correctly, getting conventions and orientations right, judging when an obstruction does or does not support what a report claims, and steering the native-implementation work. On questions of mathematical correctness and interpretation, the maintainer defers to people who know the field.

Ways to help:
- **Math review** — conventions, invariant definitions, the wording of obstruction / claim / strength in reports, and the trace and concordance machinery.
- **Backend adapters** — wrapping or hardening KnotJob, SageMath, HFKcalc, Khoca, or others behind the common contract.
- **Validation data** — expanding known-answer coverage against KnotInfo and cross-backend checks.
- **Native engines** — the mod-2 Khovanov / Lee–Rasmussen work and exact-algebra backends, behind the validation harness.
- **Docs and examples** — making the conventions and workflows legible to newcomers.

To get started, open an issue or a discussion describing what you'd like to work on, or send a pull request. The best first contributions are small, well-scoped, and come with the known-answer or cross-backend check that demonstrates them.

### Maintainership

This is meant to be a shared project, not a solo one. Sustained, high-quality contribution earns **maintainer status, including review and merge rights** — the maintainer would rather share authority with people who understand the mathematics than gatekeep it. If you have been contributing and want a larger role, say so; that is the intended path, not the exception.

## Validation and trust

Tetradrome's whole value is that its results are checkable. Every computation records the backend, its version, the coefficient field and grading conventions, the raw output, and a validation status (known-answer match, independent-backend agreement, `d² = 0`). Reports separate what was *computed*, what is an *obstruction*, and what rests on a *theorem reference*. An impressive-looking result with no validation path is treated as a bug, not a feature.

## Built on

Tetradrome stands on the work of others and aims to credit it clearly:

- **Spherogram / SnapPy** — Marc Culler, Nathan Dunfield, and collaborators (planar diagrams, PD/DT codes).
- **knot_floer_homology / HFKcalc** — Zoltán Szabó's HFK calculator, with the Python wrapper by Marc Culler, Nathan Dunfield, and Matthias Goerner (knot Floer invariants).
- **KnotJob** — Dirk Schütz (Khovanov homology, Rasmussen *s*-invariant).
- **KnotInfo** — the knot invariant database, used here as a validation oracle.
- **SageMath**, **Khoca** (Lukas Lewark), and the wider low-dimensional-topology software community.

The motivating mathematics is Lisa Piccirillo, *The Conway knot is not slice*, Annals of Mathematics **191** (2020).

## License

Apache License 2.0.

All Tetradrome code is original. The tools under [Built on](#built-on) are used only as external validators and historical references — invoked as separate programs or optional, user-installed dependencies, and consulted for the shape of their data and the mathematics they implement. None of their source or data is vendored into this repository, so no copyleft obligation attaches and a permissive license is the right fit.
