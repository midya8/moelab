# MoeLab

A small mixture-of-experts model written in pure Python. A linear router chooses one linear expert and keeps its gate probability in the prediction. Training uses hand-derived gradients, checked by finite differences, and a trained dense baseline on synthetic regime-switching data. No ML frameworks.

## v0.3: the batch decides

[Open the exact capacity inspector](https://midya8.github.io/moelab/capacity.html) · [Proof, complete schema, sources and limits](CAPACITY.md)

A separate exact-rational capacity allocator retains every optimal dispatch, complete capacity-state witnesses, assignment counts and all eligibility bottlenecks. The objective is explicit: first dispatch as many tokens as possible, then retain the greatest sum of original gate probabilities. Four full-strength greedy controls share the same capacities and eligibility.

Confidence-priority best-available routing retains `51/100` gate mass on one control; global dispatch retains `49/50`. But optimizing routing is not optimizing predictions: another control has two equally optimal dispatches with scalar-update MSE **0 to 25**, both attained and independently checked. Excess total capacity can still force drops when eligible slots are concentrated. A skipped update never deletes the token.

The native producer uses a capacity-state dynamic program; the independent verifier enumerates every feasible assignment and never imports it. A separate BigInt inspector matches every report leaf. Fourteen controls include three explicitly normalized bridges from the original frozen models. New tests check 225 probability pairs, 1,024 eligibility/capacity cases, tiny gates, every-leaf tampering, permutation invariants, exact task-error extrema, original-data parity and actual-browser downloads. Original training, data, checkpoints and earlier results remain unchanged.

```bash
python -m moelab.capacity_cli audit examples/capacity-input.json > /tmp/capacity.json
python -m moelab.capacity_cli verify examples/capacity-input.json /tmp/capacity.json
bash examples/verify_capacity_wheel.sh
```

Finite supplied tables only, at most seven tokens/four experts. Not a new routing algorithm, production router, gradient method, training gain, GPU speed claim or implementation of Switch, BASE or Expert Choice. The original 39 Python/five Node tests are preserved; the complete suite now has 53 Python/nine Node tests.

## Training and routing

Four input regimes use different linear target maps. A single linear map cannot
represent this construction; a nonlinear dense model is compared empirically.
The sparse model predicts `g_top * expert_top(x)`. Training holds the selected
route fixed and differentiates its gate probability and expert output. This is
not a straight-through estimator for argmax. [DESIGN.md](DESIGN.md) gives the
actual task and per-example auxiliary-loss equations.

## Measured results (checked-in fixture, pinned in tests)

```
MoE test MSE (top-1 routing): 0.0303   (136 params)
Dense MLP baseline test MSE:  0.1853   (145 params)
MoE test MSE (soft routing):  0.6291
Routing purity (worst regime): 1.0000
Expert usage shares: [0.25, 0.25, 0.25, 0.25]
```

The mixture has lower test error than this one-hidden-layer tanh baseline under
the recorded optimizer budget, despite having fewer parameters. A wider or
better-tuned dense model may close the gap. CI pins the mixture's own MSE,
purity and usage as regression checks; it does not require the baseline to lose.

Every test regime routes to one expert. Soft evaluation has higher error than
the trained sparse objective, but that comparison does not isolate a cause.
The symmetric fixture also self-balances without the auxiliary penalty:
maximum expert share is 0.25 with and without it, and no-aux MSE is 0.0015,
better than 0.0303 with the penalty. This does not test the penalty's value
under imbalanced traffic.

## CLI

```bash
python -m moelab evaluate
python -m moelab gates --max-mse 0.06 --min-purity 0.95 --max-expert-share 0.26
```

Gates exit 1 with named failures; the demo asserts both sides (an impossible purity bar fails).

## Development

```bash
python -m unittest discover -s tests
bash examples/demo.sh
```

53 Python tests, plus nine substantive Node/Chrome tests for the route and capacity inspectors. No Python runtime dependencies beyond Python 3.10+.

## Limitations

Experts and the router are linear; the fixture is small and synthetic. This model routes whole examples, not transformer tokens. Its measured fit, routing purity and usage do not establish production quality, balanced traffic or a general advantage over dense models.

## License

MIT


## v0.2: follow the routing seam

[Open the interactive route inspector](https://midya8.github.io/moelab/) · [Equations, schema, sources and limits](ROUTES.md)

Choose any pair of the original 240 training examples and follow the frozen model through input space. Exact rational affine geometry finds every cell and point tie, including nonwinning crossings, persistent identical winners, endpoint ties and intervals too thin for the display grid. Inspect retained-probability sparse outputs, both one-sided limits, point values, signed jumps and analytic branch derivatives. Original binary64 inference and mathematical geometry remain separate; disagreement is exposed rather than hidden.

The original aux, no-aux and initialization models, dense model, 340 data rows and complete original report are frozen and reproducible. Eight arithmetic controls show visible and zero jumps, a triple tie whose point is neither limit, persistent ties, hidden crossings, a sub-grid cell, an endpoint tie and sharp balanced gates. Every original evaluation position remains inspectable. Original per-example, batch-mean-square and Switch frequency/probability penalties are reported separately. None guarantees balanced dispatch under ties.

```bash
python -m moelab route-demo /tmp/moe-demo
python -m moelab route-audit input.json --probe .5
node --test tests/test_routes.cjs tests/test_routes_browser.cjs
bash examples/verify_wheel.sh
```

The frozen no-aux report is **0.0014970307791088836**, rounding to **0.0015**. An earlier README said 0.0010; training and results were not changed to correct that text. The module and test docstrings now describe the actual fixed-branch objective and per-example auxiliary penalty. [ROUTES.md](ROUTES.md) keeps the distinction from squared batch-mean gates and the Switch statistic explicit. No production-traffic, robustness or causal-mechanism claim follows from this fixture.
