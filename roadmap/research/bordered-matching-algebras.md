# Bordered matching algebras: how the paper was read

The curved part of `src/tetradrome/engines/bordered/` implements Ozsváth and Szabó, "Bordered
knot algebras with matchings" (arXiv:1707.00597), in the curved form of its section 13 and of
arXiv:1912.01657. It builds the knot Floer complex over F2[U, V]/(UV) and reads τ, ν and ε from
it. Signs are dropped: everything is over F2. This page records each point the paper leaves to a
figure, leaves implicit or misprints, how it was read, and the evidence. The readings of the
earlier paper are in [bordered-algebras.md](bordered-algebras.md). Tests:
`tests/test_bordered_curved.py` (algebra, curvature, bimodules) and
`tests/test_bordered_complex.py` (knots).

Evidence names:

- **Curved relation tests.** The curved DA structure relations of 1912.01657, checked on random
  input sequences for every bimodule. Crossings are tested at both ends and in the middle of 4
  and 6 strands, matched and unmatched. The minimum and its rotation are tested on every path
  through the graph (13.2) of up to 5 or 6 inputs.
- **Homogeneity.** Every operation keeps the Alexander grading and shifts Maslov by (inputs) − 1.
- **The oracles.** On every KnotInfo knot through 12 crossings: τ equals KnotInfo's and Szabó's
  program's (`knot_floer_homology`); ν and ε equal the program's; ε equals KnotInfo's; HFK-hat
  from the complex equals KnotInfo's and the 1603 engine's; the generator count equals the
  program's total rank. Through 9 crossings, the homology of every Alexander part of C, C/U and
  C/V also equals that of the program's complex.

| Point | Reading | Evidence |
|---|---|---|
| Δ constant in Definition 2.6 (also in 1912.01657) | (number of inputs) − 1, as in 1603; the printed "−ℓ + 2" is a misprint | Homogeneity; a weight budget derived from it prunes nothing (a larger budget gives identical products) |
| N^i as an "opposite" module | Arrow reversal over the same algebras; the curved γ_0 reverses with the rest, giving γ_0(W) = U_β L_i ⊗ S and γ_0(E) = U_α R_{i+1} ⊗ S, with α, β the input partners of i and i + 1 | Curved relation tests in all eight crossing cases; with α and β exchanged, 32 generators of a 6-strand case fail (measured); mirror test |
| Proposition 10.2's (P^i)^op | A misprint for N^i; Proposition 10.3 (mirror = dual) is what is tested | The mirror's complex equals Hom(C, R); τ, ν, ε of the mirror behave as they must |
| Local gradings (Figure 1, missing) | The 1603 engine's validated (M, A), with Δ = M − A; maxima, minima, top and terminal generators at (0, 0) | Homogeneity, including the new γ_0 terms; every final complex is homogeneous over F2[U, V]/(UV); the oracles |
| Which crossing is P^i (Figures 2 to 10, missing) | As in 1603: `("cross", i, True)` is P^{i+1} | The 4-plat right-handed trefoil gives the paper's complex exactly (d b = U a + V c) |
| τ, ν, ε (bibliography missing) | The standard definitions: τ and ν as the least s at which H_0 of the vertical filtration level, respectively of Â_s, maps onto HF-hat; ε by (11.2). The paper's own Definitions 1.5 and 1.6 (horizontal complex, largest s) are computed as well | The two must agree or the code raises; they agreed everywhere. τ(dual) = −τ, ν ∈ {τ, τ + 1} and ε ∈ {−1, 0, 1} are enforced; the oracles |
| ψ in section 5.2 | As in 1603 and `maximum.py` | Curved relation tests of the maximum; the oracles |
| 1603 pieces the paper relies on without restating | The 1603 engine's readings, unchanged, on C-free inputs | Curved relation tests of the crossings |
| Range of c (Theorem 7.10 against section 7.1) | 1 ≤ c ≤ 2n + 1 | Every minimum position of the swept diagrams |
| Definition 7.2's paths | One edge per input; the U_2^m loop at Y R_2 is a whole operation of one input; b ends at ψ(x_{m+1}); a non-preferred end gives nothing; inputs match edges by their weights on the cup's two strands | Every path of up to 6 inputs satisfies the curved relation; the operations the paper writes out |
| Lemma 8.1's matching (pairs c and c + 1) | The reflected minimum uses the minimum's own algebras | Curved relation tests of the rotated minimum; 1912.01657's lower-diagram example reproduced exactly |
| Section 13's C | The sum of the C_p over the matching; the curved γ_0 drops C ⊗ C, so the maximum loses its C loop and the minimum has no γ_0 | Curved relation tests; the oracles |
| Terminal module orientation | 1707's rule: U_s becomes U on the strand pointing up just above the global minimum, V on the other (1912.01657 always sends U_1 to U) | Every final complex is homogeneous; reversing the orientation gives (M − 2A, −A) with U and V exchanged (Proposition 10.1) and the same τ, ν, ε |
| Cancellation (Definition 13.2, Lemma 13.3) | Only arrows whose coefficient is exactly an idempotent, by the 1603 engine's formula | Uncancelled and cancelled complexes have the same invariants and HFK-hat on seven presentations |
| Proposition 13.1's coverage | Paths ending at X L_1 never occur (no idempotent at position 0 reaches the minimum, which raises if one does) | Never raised |
| "1 is oriented upwards" | Strand 1 of the slice just above the global minimum | As the terminal orientation |
| ν_0's "U^d W_s ≠ 0 for all d" | H(Â_s) has a class that survives setting U = 1 | The paper's ν so read equals the standard ν everywhere |
| α and β | Crossing: the input partners of i and i + 1. Minimum: the output strands matched to the cup's two strands, with v_2 on U_α and v_1 on U_β | Exchanging them fails the relation (measured: 32 generators at a crossing, 3 minimum paths) |
| Matchings | Every matched pair has one upward strand; `CurvedAlgebra` raises otherwise or on anything not a perfect matching | Never raised in the sweeps |
| Curvature in the type D check | `TypeD.check` requires d + δδ = μ_0 ⊗ Id; μ_0 is computed per idempotent with the algebra's relations | Curvature is central with Maslov −2 and Alexander 0 on B(4, 2) for three matchings |
| Horizontal and vertical | U arrows are horizontal (A rises by u, M by 2u − 1); V arrows are vertical (A falls by v, M by 1); the vertical complex's homology is HF-hat | The program's own complex converts with these conventions on every knot through 9 crossings |

## KnotInfo's ν column

KnotInfo lists ν as a pair `[a, b]`. Through 12 crossings its first entry equals Szabó's
program's ν except on exactly the 467 alternating knots with ε = −1, where it lists ν = τ. That
contradicts Hom's relation (ν = τ + 1 when ε = −1) and the program. KnotInfo's τ and ε agree with
the program on all 2,977 knots. So `compute()` does not use KnotInfo's ν column as an oracle;
Szabó's program is ν's computed oracle.
