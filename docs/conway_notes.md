# Conway Notes

The motivating mathematics behind the Conway Workflow Reproducer (SPEC 6, 7), in
plain terms, with the parts that need expert review flagged. This is drafted from
the literature; it is not a new result and makes no new claim. Where a precise
statement matters, the citation is to Piccirillo's paper.

## The knot

The Conway knot is the 11-crossing knot `K11n34`. It and the Kinoshita-Terasaka
knot (`K11n42`) form a mutant pair. The Conway knot has trivial Alexander
polynomial, so by Freedman's work it is **topologically slice**; and essentially
all the cheap smooth slice obstructions -- the tau and nu invariants from knot
Floer homology, and Rasmussen's s-invariant -- **vanish** for it. That vanishing is
exactly why its smooth sliceness resisted classification for so long.

## The question

A knot is **smoothly slice** if it bounds a smoothly embedded disk in the 4-ball.
The Conway knot's smooth sliceness was the last open case among knots with 12 or
fewer crossings until 2020.

## Piccirillo's strategy (the part the schema encodes)

The key ingredient is the **trace embedding lemma** (folklore): a knot `K` is
smoothly slice if and only if its 4-dimensional **trace** `X(K)` smoothly embeds in
the 4-sphere. Crucially, a knot is *not* determined by its trace -- there exist
distinct knots `K` and `K'` with diffeomorphic traces (Akbulut), and such siblings
need not even be concordant (Miller-Piccirillo). So two knots can share a trace
while one has a vanishing s-invariant and the other does not.

Piccirillo's argument:

1. Construct a knot `K'` that shares the 0-trace with the Conway knot.
2. Since they share a trace, `K'` is smoothly slice if and only if the Conway knot
   is.
3. Compute Rasmussen's s-invariant of `K'` and find it nonzero. A nonzero
   s-invariant obstructs smooth sliceness, so `K'` is not smoothly slice.
4. By step 2, the Conway knot is not smoothly slice either.

Reference: Lisa Piccirillo, "The Conway knot is not slice," Annals of Mathematics
191(2), 2020 (arXiv:1808.02923).

This maps directly onto the public schema (SPEC 13.10): the Conway entry has an
`obstruction_profile` with `all_vanish = True`, and its `SliceCertificate` has
`via = "trace_sibling"`, `sibling = K'`, and a `witness` recording
`rasmussen_invariant(K')`.

## What the reproducer does and does not establish

**Does:** normalize the catalog inputs; compute / look up the relevant invariants
(`rasmussen_invariant`, `ozsvath_szabo_tau`, ...) for the Conway knot, the
Kinoshita-Terasaka knot, and -- given a reliably encoded `K'` -- for `K'`;
cross-check them against KnotInfo and an independent backend; and, given `K'` as
input, re-derive the nonzero-`s` witness and present it with full provenance.

**Does not:** re-prove that `K'` shares a trace with the Conway knot. That
trace-equivalence is geometric input (Piccirillo's RGB-link construction), not
something the invariant pipeline establishes. The reproducer reports it as a
`theorem_reference`, not as something it computed. It makes no new mathematical
claim.

## Needs review

- The precise statement and hypotheses of the trace embedding lemma (0-trace vs
  general framing).
- A reliable, checked encoding of `K'` (PD / DT) -- the SPEC lists this as
  conditional on a trustworthy diagram (SPEC 7.3).
- The RGB-link construction used to build trace siblings.
- Confirmation that the obstruction wording above matches how a topologist would
  state it.
