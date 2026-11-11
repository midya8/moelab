# Worst regret on one gate table

This is an exact audit of a supplied probability table, not a trained uncertainty
model. Fix capacities and eligible experts. Pick a maximum-coverage dispatch
that has the most retained gate mass at the nominal probabilities. Freeze that
reference. How much mass can it lose to another maximum-coverage dispatch on
one admissible probability table?

The native producer uses bounded-simplex linear support and a capacity-state
dynamic program. The independent verifier enumerates row vertices and complete
assignments. The browser has its own BigInt vertex engine and capacity search.
No training, original data, checkpoints or previous numerical results change.

## Run it

```bash
python -m moelab.gate_cli audit examples/gate-input.json > /tmp/gate-report.json
python -m moelab.gate_cli verify examples/gate-input.json /tmp/gate-report.json
python -m moelab.gate_cli demo
bash examples/verify_gate_wheel.sh
node --test tests/test_gate.cjs tests/test_gate_browser.cjs
```

Open `docs/gate.html` directly in a modern browser, or use the
[hosted inspector](https://midya8.github.io/moelab/gate.html). All computation is
local. Input and full report downloads replay through the native verifier.
A successful verification establishes consistency with the given input, not
authorship, calibration or provenance.

## Input contract

Exactly these five fields are required:

```json
{
  "probabilities": [["2/5", "3/10", "3/10"]],
  "capacities": [1, 1, 0],
  "eligible": [[0, 1]],
  "lower": [["3/10", "1/5", "3/10"]],
  "upper": [["3/5", "1/2", "3/10"]]
}
```

There are 1..7 tokens and 2..4 experts. All three matrices have the same shape.
Integer JSON values or integer/fraction strings represent exact rationals.
Decimals, exponents, booleans and nonfinite values are rejected. Input rational
strings have at most 160 characters; reduced numerator/denominator have at
most 256 bits. Nominal rows are nonnegative and sum to one exactly.
`0 <= lower <= nominal <= upper <= 1` coordinatewise, so each declared row set
is nonempty. Capacities are integers from zero through the token count.
Eligibility is a distinct list of expert indices for each token, possibly empty.
The CLI rejects duplicate decoded JSON keys, files larger than eight million
bytes, and nesting deeper than 48. Computed report fractions may be larger
than input fractions and are checked exactly, without floating tolerances.

The uncertainty set is the product of row polytopes:

```
U_i = {p_i : lower_ij <= p_ij <= upper_ij, sum_j p_ij = 1}
U = U_0 x ... x U_(n-1)
```

These are supplied bounds, not confidence bands, learned perturbation radii,
router-logit jitter, an input-space intervention or a probability distribution
over scenarios. Ineligible and zero-capacity coordinates still take part in
normalization. They cannot be dispatched. An eligible zero-probability route
can dispatch an update with zero retained mass. Skip `-1` contributes zero
mass and consumes no slot; it does not delete the token.

## Reference, challenger and order of quantifiers

Let `F` be all capacity/eligibility-feasible assignments, including skips, and
`K` their maximum dispatch count. Let `F_K` contain assignments dispatching
exactly `K` tokens. For any route `a`, define retained mass:

```
f(a,p) = sum_i p_i,a_i    (only terms with a_i >= 0)
```

The reference `A` maximizes `f(a, nominal)` over `F_K`. Ties use expert indices
in ascending order, followed by skip, in token order. This canonical choice is
part of the input semantics. A permutation of expert labels can change which
nominal tie is selected and therefore change its regret; invariance is only
asserted for unique references in the permutation test.

The audited value is:

```
R(A) = max_(B in F_K) max_(p in U) [f(B,p) - f(A,p)]
```

`A` is feasible, so regret is nonnegative. Both quantifiers are maxima, so their
order can be exchanged. This evaluates one reference. It does **not** minimize
`R(A)` over possible references, choose a minimax-regret policy or retrain a
router. Maximizing coverage remains first even if a lower-coverage route would
retain more mass. Counts refer to tied assignment choices, never the number or
volume of continuous gate-table witnesses.

For fixed `B`, rows separate. For token `i`, the coefficient is
`c_j = 1[j=B_i] - 1[j=A_i]`. Shared route entries cancel before maximization.
The exact row support is `s_i(B_i) = max_(p_i in U_i) c.p_i`.
A capacity DP then maximizes coverage followed by `sum_i s_i(B_i)`.

## A row optimum with an upper bound

The producer starts at the lower bounds and fills remaining unit mass in
nonincreasing coefficient order, breaking ties by ascending expert index.
Feasibility requires `sum(lower) <= 1 <= sum(upper)` and holds by containment
of the nominal row. An exchange from a smaller coefficient to a larger one
cannot reduce the objective; filling the larger one first is optimal.

The report also makes this check independent of the fill procedure. For any
scalar `lambda`, the equality `sum p = 1` gives:

```
c.p = lambda + sum_j (c_j - lambda) p_j
    <= lambda + sum_j max((c_j-lambda) lower_j,
                         (c_j-lambda) upper_j)
```

The supplied point attains this upper bound. `lambda` is the coefficient of
the last coordinate to receive positive fill, or the largest coefficient when
no fill is needed. Exact equality proves the row support. The verifier instead
enumerates all vertices: all but at most one coordinate must be at a bound.
For each free coordinate, enumerate endpoint choices for the others, solve the
sum-one equality, and keep feasible rows. It checks the produced point, its
vertex-optimal value and its dual equality. Alternative valid row witnesses
are allowed; report ordering and dispatch ties remain canonical.

For each capacity state `(token index, used slots)`, the DP retains every
feasible outgoing action, its best suffix coverage/score, whether it is optimal,
and the exact number of completing optimal assignments. All reachable states
are emitted, including those reached by losing prefixes. The independent
verifier enumerates complete assignments and suffix assignments to check all
state values, counts and edges; it imports neither producer nor producer parser.

At the chosen worst challenger, collect its row support points into one table.
They jointly attain `R(A)`. This challenger is also maximum-mass among `F_K`
at that table. Otherwise a better dispatch there would produce a larger regret,
contradicting the outer maximum. Random assignment tests check this directly.

## Why separate extrema can overstate regret

A tempting upper bound is:

```
S(A) = max_(B in F_K, p in U) f(B,p) - min_(q in U) f(A,q)
```

The tables `p` and `q` are independent here. `S(A) >= R(A)`, but there may be
no single table attaining that difference. Reports retain both actual tables,
the best-table challenger search, each mass and the exact excess `S(A)-R(A)`.

In `simplex_gap`, nominal `[2/5,3/10,3/10]` chooses expert 0. Expert 2 is fixed
at `3/10`; expert 0 cannot fall below `3/10`. On one normalized row, expert 1
can therefore reach only `2/5` when expert 0 is at its minimum. Regret is
`2/5 - 3/10 = 1/10`. The independent box contrast would put expert 1 at `1/2`
and expert 0 at `3/10`, but those coordinates plus expert 2 sum to `11/10`.
That is not an admissible table. The separate bound is `1/2 - 3/10 = 1/5`,
using different valid rows for its two extrema.

In `shared_route`, eligibility forces the same expert for reference and
challenger. Every coefficient cancels, so regret is zero. Separate extrema
still give `3/5 - 2/5 = 1/5`. A large bound is not evidence of an actual loss.

## Report contract

`moelab-gate-regret-v1` has exactly these top-level fields:

- `schema` and canonicalized `input`.
- `nominal`: maximum coverage, nominal mass `score`, optimal assignment `count`,
  canonical reference and complete `states`.
- `row_supports`: token-major records for ascending eligible actions, then skip.
  Each has `expert`, `coefficient`, `point`, `value`, `lambda`, `dual_terms`,
  and `dual_bound`.
- `worst`: maximum coverage, support-score `score`, worst-challenger count,
  canonical challenger and complete states.
- `adversarial_probabilities`, `reference_mass`, `challenger_mass`, `regret`.
- `separate_extrema_bound`, `separate_extrema_gap`.
- `separate_extrema`: its `challenger` search, `best_probabilities`,
  `reference_worst_probabilities`, `best_mass`, `reference_worst_mass`.

Each search state has `id`, `token`, `used`, suffix `objective` as
`[coverage, exact score]`, string `count`, and `edges`. Each edge records
`expert`, `next`, suffix objective including the current action, and boolean
`optimal`. Terminal states have objective `[0,"0"]`, count `"1"`, no edges.
All rational report values are reduced fraction strings; assignment counts are
integer strings. Counts are deterministic exact enumeration, not stochastic
routing probabilities. A certificate is a computational witness, not a
machine-checked formal proof.

## Controls and tests

Ten complete declared controls cover simplex coupling, shared-route
cancellation, nominal ties, zero-width bounds, zero capacity, missing eligibility
and zero gates, capacity-induced skips, stable references under nonzero bounds,
small non-decimal regret and four-expert asymmetric bounds. No model-derived
uncertainty is implied. Generated controls and report bindings replay exactly.

The 19 added Python tests retain the previous 53. Independent oracles check 500
row-support problems and 90 random small dispatch tables, every emitted state,
attaining-table challenger optimality, wider-set monotonicity, unique-reference
expert permutations, all 7-token/4-expert actions, report isolation, every
primitive report leaf on a control, strict JSON and installed CLI replay.
Four new Node tests cover ten exact report matches, 125 generated browser
reports checked by the independent native verifier, 256-bit inputs producing
larger internal fractions and invalid inputs. One actual-Chrome test checks
30 tables, 58 row proofs, 234 states across all three searches, 30 responsive
layouts, trusted keyboard invalidation, edited input, actual Blob downloads
and independent native verification. Whole suite: 72 Python and 14 Node/Chrome
tests. All previous tests and tolerances remain in place.

The standalone verifier is deliberately exponential and bounded to seven rows
and four experts. At most `5^7 = 78,125` complete action sequences exist before
capacity filtering. This is educational exact enumeration, not a throughput
benchmark or production-scale solver. A single admissible set cannot establish
model confidence, task error, balanced load, generalization or input/weight
robustness. There are no targets or expert predictions in this input.

## Sources and attribution

These are established optimization and routing ideas. This project contributes
the implementation, complete evidence format and finite controls, not a new
theorem, solver family or routing method.

- [Aissi, Bazgan and Vanderpooten, min-max and min-max regret survey](https://www.lamsade.dauphine.fr/~bazgan/Papers/EJOR09a.pdf): regret and the distinction between evaluating a fixed choice and optimizing that choice. Our objective maximizes mass, so the regret sign follows that convention.
- [Kasperski and Zielinski, robust discrete optimization survey](https://cs.pwr.edu.pl/zielinski/publications/papers/survey16.pdf): assumed uncertainty sets with fixed feasible decisions. Their independent interval cases are not our coupled row simplex.
- [YALMIP robust optimization tutorial](https://yalmip.github.io/tutorial/robustoptimization/): quantified worst-case models and polytope enumeration. No YALMIP runtime is used.
- [McKenna, bounded probability-simplex projection](http://www.ryanhmckenna.com/2019/10/projecting-onto-probability-simplex.html): feasibility `sum(lower)<=1<=sum(upper)`. That article solves a quadratic projection; our linear support procedure is different.
- [Box/equality constrained optimization note, arXiv:2004.08666](https://ar5iv.labs.arxiv.org/html/2004.08666): distinction between box/equality linear and quadratic problems. Its projection iteration is not used.
- [Switch Transformers](https://arxiv.org/html/2101.03961v2) and [current Hugging Face implementation documentation](https://huggingface.co/docs/transformers/model_doc/switch_transformers): retained gate weighting, expert capacity and skipped expert updates. This audit does not implement Switch training or model its router jitter.
- [Glover, surrogate and continuous-knapsack teaching notes](http://leeds-faculty.colorado.edu/glover/Surrogate_Notes_1.pdf): related single-constraint bounded linear problems. The direct row proof above supplies this implementation's justification; integer-knapsack heuristics are not imported.
