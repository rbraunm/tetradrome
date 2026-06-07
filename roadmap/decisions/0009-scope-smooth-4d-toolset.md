# 0009 - Scope: the smooth-4D-topology toolset, not a knot calculator

**Status:** Accepted

## Context

The repo's framing — the README one-liner, the SPEC header — leans on the phrase
"knot invariants," and that phrasing has repeatedly caused the project to be read,
including by a recent working session, as *knot-centric*: links and other inputs
treated as out of scope, knots as the whole point. That narrowing was never intended.

The motivating problem is a smooth-4-dimensional-topology problem: smooth sliceness,
the Conway knot, the 4-genus, exotic phenomena. The invariants that bear on it — the
Rasmussen *s*-invariant, tau, epsilon, nu, and the Khovanov / Lee / knot Floer
homologies underneath them — take knots, links, and braids as inputs, and today they
are spread across many separate, differently-packaged tools (Spherogram/SnapPy,
Regina, KnotJob, knot_floer_homology, SageMath, the Khovanov tools). A mathematician
working in this area pays a tool-sprawl tax to move between them.

## Decision

Tetradrome's scope is the **computational toolset of smooth 4-dimensional topology** —
one validated, auditable surface that consolidates that sprawl — not a knot-invariant
calculator.

- **Inputs** are the descriptions those tools accept: knots, links, braids, and (as
  they enter scope) surgery / Kirby descriptions. **Outputs** are the invariants that
  constrain smooth 4D structure and the homologies they are read from.
- **Knots are a first-class, robustly-validated citizen** — the densest validation
  surface (KnotInfo) and the natural proving ground — but the floor, not the ceiling.
  Code must not bake in knot-only (single-component) assumptions that a link or other
  input would later have to tear out. The standard diagram model is the general one
  (crossings + strands + a count of crossingless unlinked components, as Regina and
  Spherogram use), even while knots are what gets validated first.
- **Breadth is built incrementally, not speculatively.** Multicomponent/link and
  3/4-manifold support land when an invariant or input path actually reaches for them,
  behind the same validation discipline — never as up-front scaffolding without
  consumers (the project's no-legacy-without-users principle).

## Consequences

- "Is this in scope?" for a link, a braid input, or a 3/4-manifold invariant is
  answered yes-in-principle; the only gate is incremental need plus validation, not a
  knot boundary.
- Representations and engine interfaces stay general even while knots are the first
  thing validated, so breadth is an extension rather than a rewrite.
- Front-facing docs must not describe the project as knot-only; the README and SPEC
  scope language is corrected to match, and this ADR is the reason on record so the
  narrowing does not creep back.
- This decision is about the **breadth of objects**, deliberately separate from *how*
  invariants are computed (natively) and *how* external tools are used (validators
  only) — those are decided in 0006 and 0007.
