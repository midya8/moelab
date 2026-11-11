# Exact route boundaries, approximate predictions

The v0.2 inspector freezes the original four-expert models, the original 240 training and 100 test examples, the dense baseline and the entire report. It changes inputs along a path, not training. The trained aux model, trained no-aux model and untrained initialization are separate controls, not an ablation of only one learned mechanism.

## Reproduce

```bash
python -m unittest discover -s tests -v
node --test tests/test_routes.cjs tests/test_routes_browser.cjs
bash examples/demo.sh
bash examples/verify_wheel.sh
python -m moelab route-demo /tmp/moe-demo
# Save a copied input from the demo as input.json:
python -m moelab route-audit input.json --probe .5
python -m http.server 8000 --directory docs
```

Python 3.10+ runs without third-party runtime dependencies. Node 22+ and Chrome are required for the browser suite. `CHROME_BIN` selects Chrome. The wheel test creates an isolated environment and runs all four commands outside the checkout, imports all seven implementation modules, retrains all models, checks all examples and the full report, and regenerates every witness.

The checked-in fixture was generated with CPython 3.10 on one Linux runtime. Floating-point `sum` changed in Python 3.12, and libm/runtime differences can also perturb full training on 3.10/3.11. Every version therefore compares all regenerated numeric values at absolute/relative tolerance 1e-11, while preserving exact keys, lengths and discrete fields. Within the same runtime, the original trainer and independent reconstruction must agree exactly. The wheel check separately audits all 32 **frozen** witnesses, requiring exact rational/discrete values and 1e-12 numeric tolerance; newly retrained models may have slightly different rational boundary fractions and must not be relabeled as the frozen model. It verifies newly generated witnesses against their own regenerated model. Exact rational topology is always computed from the supplied model values, never rounded back to a different fixture. Original training, dense baseline, data generator, original tests and gate conditions are unchanged.

## Affine routing geometry

For path x(t) = origin + t (destination - origin), 0 <= t <= 1, router scores are z_i(t) = a_i + b_i t and expert outputs are h_i(t) = u_i + v_i t. All these coefficients are built with exact rational arithmetic from the supplied finite binary64 values, including endpoint subtraction and dot products. Python uses `Fraction`; the independent browser evaluator decodes the IEEE binary64 sign, exponent and mantissa to `BigInt` fractions.

For each pair with unequal slopes, the exact candidate crossing is (a_j - a_i)/(b_i - b_j). The evaluator sorts every in-domain crossing and both endpoints, then evaluates a rational midpoint in each open cell. It retains crossings below the upper envelope rather than calling them transitions. Identical lines remain co-maximal in a whole cell. At each point and in each cell, the lowest expert index breaks exact ties. A transition is an interior point whose left and right owners differ. Endpoint ties have only one one-sided limit.

Fractions are exported as numerator and denominator strings. Decimal positions are approximations. The low-level `envelope()` helper also supports 2..16 lines, coefficients and domain bounded by one million, parallel lines and out-of-domain crossings; a decimal approximation outside the finite range is JSON `null`, never Infinity. The model audit is more tightly bounded below.

## The selected probability is retained

Let g_i = exp(z_i)/sum_j exp(z_j). The original sparse output is g_e h_e, not h_e: it does not renormalize a selected top-1 gate to one. On a fixed branch e,

```
g_i' = g_i (b_i - sum_j g_j b_j)
y_e' = g_e [v_e + h_e (b_e - sum_j g_j b_j)]
```

This is the ordinary derivative inside a routing cell and a branch derivative at a tie. It is not differentiation through argmax or a soft backward surrogate. Both neighboring limits are evaluated at the exact rational boundary with the relevant branch held fixed; no arbitrary epsilon enters a limit. Jump = right limit - left limit. A route can change with zero jump. At a triple tie the chosen point expert can differ from both neighboring experts, so its point output can differ from both limits.

Softmax and output values are floating-point approximations, even at exact rational boundaries. The 65-point display curve does not find boundaries. It breaks sparse polylines when adjacent sampled owners differ; an interval thinner than that grid may not be visible in the curve or a one-pixel route strip. The exact cell table remains authoritative for mathematical geometry.

## Mathematical geometry is not a rounded-device certificate

The original inference separately rounds path arithmetic, dot products, exponentials, their sum and the probability argmax. Near a tie its selected expert may differ from the exact affine extension, including when unequal logits round to equal softmax probabilities. Both states and a disagreement flag are exported; they are never silently snapped together. A dedicated control uses logits [0, smallest positive binary64] to show exact owner 1 but rounded gate owner 0 at all 65 probes. Such disagreement does not invalidate the rational geometry; it limits the claim to that extension. No guarantee is made for a GPU, fused operations, alternate summation or another floating-point library.

## Three different balance statistics

For N rows and E experts, let P_i = mean_n g_ni and f_i = count(owner_n = i)/N. The original implementation's per-example penalty averages E sum_i g_ni^2. An earlier module docstring described E sum_i P_i^2 instead. That docstring is now corrected; the training arithmetic is unchanged. These are not equal:

```
mean_n [E sum_i g_ni^2] - E sum_i P_i^2
    = E sum_i mean_n [(g_ni - P_i)^2]
```

Switch's batch-frequency/probability statistic is E sum_i f_i P_i. It is yet a third expression. The UI reports each one, each row's probabilities, counts, shares and original targets. The training loop is not changed to match the earlier docstring or the paper. Uniform ties give all three values equal to one while every row dispatches to expert 0. A low penalty is not a traffic guarantee. Sharp balanced gates can give a per-example penalty near E while the other two equal one.

Targets affect only descriptive evaluation errors on the original rows. There is no target for an arbitrary interpolation. An input path is not a robustness radius, causal experiment, generalization test or inference-cost benchmark. No token capacity, dropping, transformer, serving load or measured speed is implemented.

## Input and output contract

`route-audit` accepts JSON with exactly these top-level keys:

```json
{"model":{"W":[[0],[0]],"b":[-1,1],"R":[[-1],[1]],"c":[0.5,-0.5]},"origin":[0],"destination":[1],"evaluation":{"example":[{"x":[0],"target":0}]}}
```

- `W`, `R`: 2..16 expert rows, each 1..32 inputs. `b`, `c`: one value per expert. All model values are finite numbers bounded by 100 in magnitude, never booleans.
- `origin`, `destination` and evaluation `x`: correct input dimension, finite values bounded by 20.
- Optional `evaluation`: an object, not null. Up to four nonempty named splits, name length 1..40 characters, at most 512 total rows. Each row has `x`, finite `target` bounded by 1000, and optional integer-valued `regime` 0..15. Unknown keys fail.
- Probe is finite 0..1. The library supports 2..129 display samples, default 65. The CLI uses that default.
- Strict JSON parsing rejects duplicate keys, nonfinite numbers, invalid grammar, inputs above 1,000,000 UTF-8 bytes and nesting beyond 64 containers. Errors disable browser exports and cannot revive a stale valid report.

`moelab-route-v1` returns dimensions, exact router/expert coefficients, every pair crossing, parallel pairs, open cell and point, original/extension probe states, all display states, nearest interior transition in lambda units, disagreement count and every evaluation position. Download wraps the input and report; it does not upload the input anywhere. Empty evaluation objects are allowed.

## Validation, not a new routing method

39 Python tests include the original nine, 11 exact-envelope tests, full original training/model/data/report reconstruction, finite differences on 15 trained-model branch states, 160 independent 70-digit Decimal prediction/derivative checks, boundary controls, parser/CLI failures and installed data. Five substantive Node/Chrome tests compare 32 complete Python witnesses with exact rational/discrete values and 1e-12 absolute/relative numeric tolerance, 80 independent integer geometries, strict parsing and the actual UI. The full witnesses cover 8,176 evaluation positions and 2,080 path display positions. Chrome checks all 240 destination selectors, every one of the 340 evaluation row positions, ties, exports, disabled states, clipboard failure/success, invalid-input non-revival, keyboard changes and 320/390/1280 layouts. This is a deterministic audit fixture, not evidence of general model quality.

## Sources and conventions

- [Switch Transformers, equations 1-6](https://arxiv.org/html/2101.03961v2): selected gate probability and frequency/probability balancing. This is attribution, not a claim that the original per-example penalty implements equation 4.
- [Hugging Face Switch implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/switch_transformers/modeling_switch_transformers.py): reference top-1 masking, gate values and batch balancing convention.
- [Blanchard, Higham and Higham: numerical analysis of log-sum-exp and softmax](https://academic.oup.com/imajna/article/41/4/2311/5893596): shifted evaluation and finite-precision limits.
- [ReMoE ICLR 2025 slides](https://iclr.cc/media/iclr-2025/Slides/31028.pdf): retained-probability top-k discontinuity context. This project does not implement ReMoE.
- [Routing-cell geometry](https://arxiv.org/html/2606.19036): context for cells and tie geometry. Its selected-top-k renormalization differs from this project's retained probability; its output formula is not substituted here.
- [Python fractions](https://docs.python.org/3/library/fractions.html): a float becomes its exact binary rational value, not its human decimal spelling.
- [Elementwise softmax derivative](https://indii.org/blog/gradients-of-softmax-and-logsumexp/): the elementwise equation was checked. The page's displayed matrix `I - ss^T` omits `diag(s)` and is not used. The correct Jacobian is `diag(s) - ss^T`, independently verified here by Decimal and finite differences.

The new contribution is the original implementation and its inspectable controls, exact/rounded separation, reproducible data and interface. Softmax, sparse routing and affine-envelope geometry are standard methods.
