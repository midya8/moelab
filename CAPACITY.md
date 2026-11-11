# Capacity is a batch decision

MoeLab v0.3 adds exact finite capacity dispatch without changing the original trainer, checkpoints, data, route-seam engine or evaluation results. The [interactive inspector](https://midya8.github.io/moelab/capacity.html) has its own BigInt engine. A native dynamic program constructs certificates; an independent exhaustive Cartesian-product implementation checks every leaf, including every capacity-state edge, all optimum counts, the full optimal-family marginals and attained scalar-error extrema.

This is an educational optimization audit of supplied rational gate tables, not a new MoE router, an implementation of Switch/BASE/Expert Choice, a production allocator, a training result or a throughput claim.

## Two counterexamples worth inspecting

### Confidence priority can still spend the wrong slot

Two tokens, three experts, capacity `[1,1,0]`, all experts eligible:

| Token | Expert 0 | Expert 1 | Expert 2 |
|---|---:|---:|---:|
| 0 | 1/2 | 49/100 | 1/100 |
| 1 | 49/100 | 1/100 | 1/2 |

The confidence-best-available control visits token 0 first, chooses expert 0, then sends token 1 to expert 1: mass `51/100`. The global optimum sends token 0 to expert 1 and token 1 to expert 0: mass `49/50`. Both dispatch two tokens under precisely the same eligibility and capacity constraints. The expert with zero capacity remains in the probability row, but cannot receive an assignment. Confidence is the highest **eligible** probability, not the highest currently available probability. That distinction and its tie rule are fixed before evaluation.

A separate two-expert `order_loss` control has the opposite useful result: confidence-best-available exactly matches the optimum. Controls are not forced to remain weak. All four explicit greedy rules are evaluated on every input.

### The entire routing optimum can disagree about task error

Two tokens, two experts, every probability `1/2`, capacity `[1,1]`. Scalar expert values are `[[0,10],[10,0]]`; scalar update targets are `[0,0]`. Both full assignments have mass `1`. Assignment `[0,1]` gives weighted updates `[0,0]` and MSE `0`; assignment `[1,0]` gives updates `[5,5]` and MSE `25`. Both are global routing optima.

The report exposes the **attainable minimum and maximum** MSE over all routing optima and a complete assignment witness for each, rather than measuring just whichever tie happened to be selected. The routing optimizer never reads target values or expert values when selecting its optimal family. The secondary error analysis is explicitly conditional on that already-fixed family. It cannot turn this routing criterion into a task-quality guarantee.

## Exact contract

Input has exactly these five fields:

```json
{
  "probabilities": [["3/5", "2/5"], ["99/100", "1/100"]],
  "capacities": [1, 1],
  "eligible": [[0, 1], [0, 1]],
  "expert_values": [[1, 1], [1, 1]],
  "targets": [0, 0]
}
```

- `1..7` tokens and `2..4` experts. Each probability row is nonnegative and sums to one **exactly**.
- Rationals are integer JSON values or integer/fraction strings, for example `"-3/7"`. Strings use integer syntax and a positive denominator. No decimal/exponent JSON numbers, approximate float probabilities, booleans, zero denominators or nonfinite values. Each input rational numerator and denominator has at most 256 bits; source string length is at most 160 characters. Output fractions are reduced and may be larger.
- Capacities are integer slots `0..n` per expert. No capacity factor, rounding rule, padding cost, device grouping or timing model is inferred.
- Eligibility is an explicit set of distinct expert indices per token. Empty sets and eligible zero-probability routes are supported. The inspector does not silently substitute top-k for this supplied relation.
- Expert values and targets are rational scalars. They audit the **expert update** `p[i,j] * value[i,j]`, using the original gate without renormalizing after assignment. A skipped update is zero. Targets are not residual-stream values or end-to-end model labels.
- Every token chooses one eligible expert or action `-1`, which skips the expert update. Each expert load is at most its capacity. A token is never deleted from a sequence. No residual network is executed.
- First maximize the number of dispatched tokens, then maximize `sum_i p[i,assignment_i]` over dispatched tokens. This is a lexicographic objective, not a scalar penalty with an undisclosed tradeoff.

Coverage first is a choice, not a theorem about model quality. `coverage_before_mass` has a two-token maximum-coverage allocation of mass `1/5`, while a greedy one-token allocation retains `9/10`. The reported pair is ordered by count first; probabilities alone must not be compared when counts differ. Eligible zero gates can be dispatched to gain coverage while producing no update. Unequal capacities and sparse eligibility can produce very unequal loads. There is no equal-load constraint.

## Producer: all states, all optima

`moelab/capacity.py` uses `Fraction`, never floating comparisons. State `(i,u)` means the first `i` tokens have been considered and expert-use vector is `u`. Available actions are every eligible expert with a remaining slot plus skip. The terminal suffix objective is `(0,0)`, with one empty completion.

For each action `j`, add `(1,p[i,j])` to the successor's suffix pair, or `(0,0)` for skip. Take the lexicographic maximum. Retain **every** action attaining it. The completion count is the sum of the successor counts over these distinct tied actions. This Bellman recurrence is exact because a suffix's feasibility depends only on `(i,u)`; prefix probabilities already earned are additive constants and do not affect its maximizing suffix.

The certificate contains every state reachable by **any** feasible prefix, not merely the chosen route. Each state records its remaining objective, optimum completion count, every feasible next edge and whether that edge is locally optimal. A locally optimal edge at a state reached through a suboptimal prefix need not lie in a global optimum. The whole global family is the root's optimal-edge DAG.

A forward path count through that DAG, multiplied by the successor completion count, yields each token/action's complete optimum count. Each marginal row sums to the root count. These counts describe uniform counting over a finite family, **not** probabilities used by a stochastic router. Distinct assignments are counted once; unused identical capacity slots are not distinguishable and do not create artificial multiplicities.

Canonical tie selection visits experts by ascending index, then skip, and follows the first optimal edge at each token. This chooses an allocation for display, not a special best-quality model. A second additive dynamic program over only optimal edges computes the minimum and maximum sum of squared update errors. Dividing by token count gives the exact attainable MSE interval endpoints and reconstructs a witness for each. Internal error ties retain the first action in the same order. The interval notation means attained extrema, not that every real number between them is an attainable discrete error.

This is a small-state dynamic program, not the auction algorithm used by BASE, not Sinkhorn transport and not a scalable min-cost flow implementation. Capacities are bounded by token count; only states with at most `n` used slots can be reached. No runtime or memory efficiency benchmark is claimed.

## Capacity bottlenecks: a separate cardinality certificate

For every nonempty subset `S` of tokens, the report lists its union of eligible experts `N(S)`, available neighbor slots `sum_{j in N(S)} c[j]`, and positive deficiency

`max(0, |S| - sum_{j in N(S)} c[j])`.

Every assignment must skip at least that many members of `S`. The maximum deficiency over all subsets equals the minimum number of skips in this finite bipartite capacity-assignment problem. One way to see the converse is to replace each expert by `c[j]` identical adjacent slots and use a maximum matching. If unmatched tokens cannot reach an unused slot by an alternating path, the alternating-reachable tokens and slots give a set with exactly the matching deficit. Otherwise augment the matching. This supplies the matching-size/minimum-skip equality; the score optimizer separately chooses the best mass among maximum-size assignments.

The producer asserts that its optimal dispatched count equals `n - maximum deficiency`. The independent verifier reconstructs both sides by direct enumeration. The `unused_capacity` fixture has two tokens, total capacity three, and one unavoidable skip because both tokens are eligible only for the single slot at expert 0. The `empty_eligibility` fixture includes a token with no neighbor at all. The browser displays every maximum-deficiency subset and exports **all** subsets, up to `2^7-1 = 127`.

## Four baselines with no hidden restrictions

All rules share the same gates, capacities and eligibility.

| Rule | Token order | Expert choice / overflow |
|---|---|---|
| Input-order top choice | Input row order | Best eligible expert only; skip if full |
| Confidence-order top choice | Descending maximum eligible gate | Best eligible expert only; skip if full |
| Input-order best available | Input row order | Try all eligible experts by descending gate |
| Confidence-order best available | Descending maximum eligible gate | Try all eligible experts by descending gate |

Input order breaks confidence ties. Lower expert index breaks gate ties. Empty eligibility has confidence zero. Confidence is computed before dispatch and not recomputed when capacity is consumed. These are precisely specified controls, not an assertion that a particular library version uses all four. The installed and browser reports include each assignment, load, retained mass, weighted update and MSE. Neither training nor CI gates require a greedy method to lose.

## Independent evidence

`moelab/capverify.py` never imports or calls the producer, its validator or its solver. It independently parses the input and enumerates Cartesian products of eligible experts plus skip. Capacity-infeasible assignments are removed. This reconstructs the complete global optimum family directly. It independently enumerates suffix completions at every reachable state to check every objective, edge, optimum flag and count. Exhaustive optimum assignments give the complete marginal counts and exact minimum/maximum update-MSE witnesses. Bitmask subsets independently reconstruct all Hall rows. A strict recursive comparison checks exact keys, list order, values **and types**, so a substituted boolean is not accepted as the integer one.

`docs/capacity-engine.js` is independently written BigInt rational arithmetic and capacity-DP code. It reconstructs the whole report, including all error extrema and Hall subsets. It does not fetch or call Python. `capacity-data.js` is checked-in input/evidence, not a substitute for recomputation.

Fourteen original controls include two order/confidence loss mechanisms, eligibility-induced avoidable drops, coverage/mass priority, mass/task-error disagreement, task-error ambiguity within ties, six tied allocation optima, all-zero buffers, excess but unusable capacity, eligible zero gates, empty eligibility and three frozen-model bridges. The browser recomputes them, never trains a model.

The original frozen aux, no-aux and initialization models remain byte-for-byte untouched. Each bridge uses the **first six original training rows**, their original scalar expert outputs and original targets. Original binary64 gate values converted to exact rational numbers may not sum to exactly one. Therefore the bridge divides each rational gate by its **exact rational row sum**, and labels that change explicitly. It does not claim to reproduce original binary64 arithmetic exactly after that normalization. Tests reconstruct every bridge field from the original frozen source. This is a dispatch illustration on a small selected batch, not a new test-set comparison.

## Run and independently replay

```bash
python -m moelab.capacity_cli audit examples/capacity-input.json > /tmp/capacity-report.json
python -m moelab.capacity_cli verify examples/capacity-input.json /tmp/capacity-report.json
python -m moelab.capacity_cli demo
python scripts/generate_capacity_fixtures.py
python -m unittest discover -s tests -v
node --test tests/test_capacity.cjs tests/test_capacity_browser.cjs
bash examples/verify_capacity_wheel.sh
```

The CLI rejects duplicate keys, nonfinite values, decimal/exponent JSON numbers, depth beyond 48 and input files above eight million bytes. Parse/validation/certificate errors exit 2 without a success report. The browser checks decoded duplicate keys too, invalidates edited results before replay, and disables downloads while stale or invalid. Exported input and certificate use real Blob downloads; an actual Chrome test independently replays those downloaded bytes with the native checker.

New verification includes every complete fixture; all 225 pairs of ternary quarter-grid distributions; all 1,024 three-token binary eligibility/capacity controls; the seven-token/four-expert bound with `2^-128` gates; token/expert permutation-equivariant objective and family marginals; strict invalid contracts; every-leaf tampering; the original frozen bridge; and 125 additional full generated native/BigInt comparisons. Actual Chrome checks 98 displayed dispatches, 1,085 exact rendered gate/support cells, all 813 retained capacity states across the fixtures, trusted keyboard invalidation/restore, two actual JSON downloads with native replay and 42 responsive layouts. These are finite implementation checks, not a proof of every runtime or browser behavior.

## Sources and distinct scope

1. Fedus, Zoph and Shazeer, [Switch Transformers](https://arxiv.org/html/2101.03961v2), §§2.1-2.2 and appendix on No-Token-Left-Behind. Top-1 gate weighting, expert capacity, overflow as skipped expert computation through a residual path, auxiliary balancing and the warning that rerouting need not improve performance. This project supplies no residual transformer or new training experiment.
2. Lewis et al., [BASE Layers](https://proceedings.mlr.press/v139/lewis21a/lewis21a.pdf), §3.2. Training solves equal-count affinity assignment, with a different inference rule. Our upper-capacity, explicit-eligibility, coverage-first **probability** objective is different from their equal-load dot-product objective and auction solver.
3. Zhou et al., [Mixture-of-Experts with Expert Choice Routing](https://proceedings.neurips.cc/paper_files/paper/2022/file/2f00ecd787b432c1d36f3de9800728eb-Paper-Conference.pdf), §3.2. Experts choose top tokens, allowing a variable number of experts per token. Our at-most-one-expert assignment is **not** Expert Choice, even when each expert fills its slots.
4. Hugging Face, [Switch Transformers documentation](https://huggingface.co/docs/transformers/model_doc/switch_transformers), router and capacity descriptions, read November 11, 2026. Documents overflow and capacity without guaranteeing every token is processed. We do not claim byte-equivalence to that implementation.
5. Liu et al., [Routers in Vision Mixture of Experts: An Empirical Study](https://arxiv.org/html/2401.15969v2), §§2.1-2.4. Distinguishes token/expert choice, probability/transport scores and dispatch/combining tensors. Our finite hard assignment is not their entropy-regularized transport or empirical vision comparison.
6. Google Research, [Mixture-of-Experts with Expert Choice Routing](https://research.google/blog/mixture-of-experts-with-expert-choice-routing/), November 16, 2022. Official explanatory comparison of token choice and expert choice. Its reported training gains belong to that work, not this project.

## Limits

No new algorithm, transformer, gradients through the discrete allocator, optimization of original training, production traffic, GPU communication, throughput, energy, memory savings, robustness or improved model quality is claimed. Batch-coupled allocation may itself be unsuitable for autoregressive training across future positions; this finite inspector supplies no causal-masking or serving guarantee. Gates and experts are supplied data, not ground-truth suitability. Capacity slots have no wall-clock meaning. Changing eligibility changes the feasible problem. Enlarging capacity cannot reduce maximum dispatch count, but if it increases that first objective the secondary retained mass need not rise. Dropping zero-gate expert computation and failing to dispatch a token are distinct in the declared objective. Original training and all prior results are retained without reinterpretation.
