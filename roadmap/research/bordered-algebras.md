# Bordered algebras: how the paper was read

`src/tetradrome/engines/bordered/` implements Ozsváth and Szabó, "Kauffman states, bordered
algebras, and a bigraded knot invariant" (arXiv:1603.06559), for HFK-hat over F2. Some of what
the engine needs is in figures, is left implicit, or is misprinted. This table records each such
point, how it was read, and the evidence. Its tests are `tests/test_bordered_algebra.py`,
`tests/test_bordered_bimodules.py` and `tests/test_bordered_knots.py`.

Evidence names:

- **Relation tests:** the DA structure relation (section 2.6), checked on input sequences. These
  are random sequences, and for the minimum every path through Γ of up to five or six inputs.
- **The oracle:** agreement with KnotInfo's HFK-hat, chirality included, on all 2,977 knots
  through 12 crossings. Also agreement with the grid engine on raw codes with kinks and a nugatory
  crossing.

## Conventions

| Point | Reading | Evidence |
|---|---|---|
| Which crossing is P^i | The over strand runs from the lower left (lower position i) to the upper right. A Morse piece `("cross", i, True)` is P^{i+1} | The 4-plat P^2 P^2 P^2 with both strands up gives the right-handed trefoil; the oracle; the mirror test |
| Local Maslov contributions (a figure) | −sign on the quadrant between the two outgoing ends, 0 on the other three. N^i negates P^i | Homogeneity of every operation fixes the relative gradings; the oracle fixes the absolute one; Proposition 12.1 |
| S in (5.13) | The output (lower) slice's S | Alexander homogeneity of every crossing operation |
| Marked point | The edge through the global minimum | The oracle |
| Idempotent convention | The explicit generator formulas; Definition 5.1 conflicts with them and is not used | Relation tests; the oracle |
| Strand order | Left to right | The oracle |

## Algebra

| Point | Reading | Evidence |
|---|---|---|
| The monomial g of a product | Products add weights; g is never formed | Associativity, Leibniz and the paper's worked products |
| Element from idempotents and weight | Stored by both idempotents, weight and C's | Left idempotent, weight and C's never name two elements (Proposition 3.9) |
| U1 U2 at I_{1} in B(2,1) | Zero, as Proposition 3.7 gives | Algebra test |
| Third line of the DA relation | δ^1_{i−1}, not δ^1_i | The relation the bimodule tests check holds |

## Crossing and maximum

| Point | Reading | Evidence |
|---|---|---|
| (8.1) indices | The output idempotent y has zero relative weight to x on the new strands c, c+1 | Relation tests of Ω; the oracle |
| Lemma 8.1 existence | The proof pairs each case with types of x; for the other type no y exists (Y with input L_{c−1}), and δ^1_2 is zero there. Two candidates raise | Relation tests of Ω; the oracle |
| Maximum's summary diagram | The loop is U_c C_{c+1} when c+1 ∈ S_2; δ^1_2 labels read output ⊗ input | Relation tests; the global-maximum test |
| Maslov count in (5.3) | Inputs minus one, as in section 2 | Homogeneity of every operation |
| I_{1,2,3} | I_{0,1,2} | Relation tests over B(2,3) |
| The characterization of δ^1_3 | The explicit list of section 5.1 | Relation tests over every B(2, k, S) |
| "Uniquely specify" | δ^1_2 is the closure of the sixteen terms of (5.4) under (5.5) and the three rules; all else is zero | Relation tests over all 32 local cases |
| N^i | δ^N(Y, o(a_l), …, o(a_1)) holds o(b) ⊗ X exactly when δ^P(X, a_1, …, a_l) holds b ⊗ Y; gradings negated | Relation tests of N^i; the paper's three N^i examples; the oracle |
| Definitions 5.7 and 5.12 | The local model is evaluated on types; the global element is fixed by (5.10) and kept when it exists and its type is the local word | Relation tests between non-adjacent strands; the paper's examples |
| Lemma 5.8's third check | A misprint; (5.10) governs, and the printed term vanishes in B | — |
| Actions from S with C's | The section 5.3 table literally, extended by U1 U2, added to the C-equivariant image of the S-to-S action | Relation tests over every S containing strand 1 or 2 |
| The C example of δ^1_3 | Read with S: δ^1_3(S, C1 R1, C2 R2) = δ^1_3(S, C1 C2 R1, R2) = R1 C1 C2 ⊗ E | Both groupings tested |

## Minimum

| Point | Reading | Evidence |
|---|---|---|
| Γ item 14 | Target U1^{n−1} L1, as the figure has it | Every Γ path of up to six inputs satisfies the relation; the oracle |
| Γ edge L1 → L1C2 labelled C2 | Included (it is in the figure, not the text) | Same; the sequence (U2, C1) gives the identity |
| Path length | One edge per input | Same |
| Start vertex when strand 0 points up | The text's rule: an idempotent off position 0 starts and ends at R2C1 | Relation tests on every exchanged path; the oracle |
| End idempotent | b runs to ψ(x_l); a sequence whose last idempotent is not preferred gives nothing | Relation tests; the oracle |
| Maslov identity | l is the number of inputs | Homogeneity of every minimum operation |
| Minimum away from the left end | mho^1 after the two positive crossings of (9.4) per step left; nearer the right end, the rotation of mho^1 (Lemma 10.1) after the rotated crossings per step right. Generators are no longer Kauffman states; the homology is unchanged | Relation tests of the rotated minimum; the oracle |

## Ends and answer

| Point | Reading | Evidence |
|---|---|---|
| A(1,{1}) | B(2,1,{1}) | The oracle |
| Terminal module | m_1 = 0 and m_{≥3} = 0; in the hat version only idempotents act | The oracle; the reversed-orientation test |
| Grading of the answer | The internal (M, A) is the standard (Maslov, Alexander); σ(right-handed trefoil) = −2 | Proposition 12.1; the oracle |
| Global maximum | Ω^1 on B(0,0,∅) applied to the one-generator structure | Its one generator and loop; the oracle |

## Algorithm

| Point | Reading | Evidence |
|---|---|---|
| Finiteness of mho^1 ⊠ Y | A path is dropped once the doubled total weight fed beyond the cup exceeds max(2A − 2M) − min(2A − 2M) + 2 + 2·#S over Y's generators. An output b ⊗ T y′ of T y has doubled total weight (2A − 2M)(y) − (2A − 2M)(y′) + 2 + 2·#C(b), and the fed weight only grows, so nothing dropped could be an output. A path longer than (#Y + 1)(budget + 2) inputs raises | A budget twelve half-units larger gives identical operations; the guard never fired |
| Cancellation | An arrow whose coefficient is exactly an idempotent is cancelled by the perturbation lemma, which stops after one step. `cancel=False` keeps every generator | Unreduced equals cancelled on eight presentations; every carried structure satisfies the type D relation |
| Idempotents | None off positions 0 and m is built; the minimum and the terminal module raise if one appears | Never raised |
| Strict unitality | Holds for every bimodule | Unitality test |
