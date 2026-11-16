# Conway Notes

The mathematics behind the Conway workflow reproducer, drafted from the literature, with
the parts that need expert review listed. It makes no new claim.

## The knot

The Conway knot is the 11-crossing knot `K11n34`. It and the Kinoshita-Terasaka
knot (`K11n42`) form a mutant pair. The Conway knot has trivial Alexander
polynomial, so by Freedman's work it is **topologically slice**; and the cheap smooth
slice obstructions -- the tau and nu invariants from knot Floer homology, and
Rasmussen's s-invariant -- **vanish** for it.

## The question

A knot is **smoothly slice** if it bounds a smoothly embedded disk in the 4-ball.
The Conway knot's smooth sliceness was the last open case among knots with 12 or
fewer crossings until 2020.

## Piccirillo's strategy

The key ingredient is the **trace embedding lemma** (folklore): a knot `K` is
smoothly slice if and only if its 4-dimensional **trace** `X(K)` smoothly embeds in
the 4-sphere. A knot is not determined by its trace: there exist distinct knots `K`
and `K'` with diffeomorphic traces (Akbulut), and such siblings need not be
concordant (Miller-Piccirillo). So two knots can share a trace while one has a
vanishing s-invariant and the other does not.

Piccirillo's argument:

1. Construct a knot `K'` that shares the 0-trace with the Conway knot.
2. Since they share a trace, `K'` is smoothly slice if and only if the Conway knot
   is.
3. Compute Rasmussen's s-invariant of `K'` and find it nonzero. A nonzero
   s-invariant obstructs smooth sliceness, so `K'` is not smoothly slice.
4. By step 2, the Conway knot is not smoothly slice either.

Reference: Lisa Piccirillo, "The Conway knot is not slice," Annals of Mathematics
191(2), 2020 (arXiv:1808.02923).

In the planned public schema (SPEC 8) the Conway entry has an `obstruction_profile`
with `all_vanish = True`, and its `SliceCertificate` has `via = "trace_sibling"`,
`sibling = K'`, and a `witness` recording `rasmussen_s(K')`.

## Later results

The Conway knot has infinite order in the smooth concordance group. Two proofs:

- Donatone-Kegel-Lewark-Truol, arXiv:2606.30422: unknotting number 1 together with
  s != 0 on the Piccirillo pattern knot gives infinite order. For the Conway knot
  that pattern knot is Piccirillo's `K'`, so the witness is the same `rasmussen_s(K')`.
- Golla-Pinzon-Caicedo, arXiv:2609.22378: an independent proof from the d-invariant of
  the 4-fold branched cover. Its one computation is tau(J_0) = -1, by Szabo's
  algorithm in SnapPy.

KnotInfo records `smooth_concordance_order` = `infty` for `11n_34`.

## What the reproducer does and does not establish

**Does:** normalize the inputs; compute the relevant invariants (`rasmussen_s`,
`ozsvath_szabo_tau`, ...) for the Conway knot, the Kinoshita-Terasaka knot, and --
given a checked encoding of `K'` -- for `K'`; validate them against the computed
oracles and KnotInfo; and re-derive the nonzero-`s` witness with full provenance.

**Does not:** re-prove that `K'` shares a trace with the Conway knot. That
trace-equivalence is geometric input (Piccirillo's RBG-link construction), not
something the invariant pipeline establishes. The reproducer reports it as a
`theorem_reference`.

## Needs review

- The precise statement and hypotheses of the trace embedding lemma (0-trace vs
  general framing).
- A checked encoding of `K'` (PD / DT).
- The RBG-link construction used to build trace siblings.
- Confirmation that the obstruction wording above matches how a topologist would
  state it.
