# MoeLab

Mixture-of-experts from scratch: a sparse router and per-expert models trained with hand-derived, finite-difference-verified gradients, measured against a trained dense baseline on a regime-switching fixture. No ML frameworks.

## What it does

- Builds a regime-switching fixture: four input regimes, each with its own linear input-target map, so the sign and scale of the relationship flips per regime. A single *linear* map cannot represent it by construction (the sign and scale of the relationship flip per regime); whether a nonlinear dense model closes the gap is an empirical question, answered by the measured baseline below.
- Trains a Switch-style sparse mixture: the router softmax scores each expert, the top expert answers, and the prediction is the gate probability times that expert's output. The selected route is held fixed during differentiation; gradients flow through the gate probability and chosen expert on that branch, derived by hand and finite-difference verified away from routing boundaries.
- Measures routing, not just fit: per-regime routing purity, expert usage shares, a same-budget dense MLP baseline, and the soft-vs-sparse evaluation gap.

## Measured results (checked-in fixture, pinned in tests)

```
MoE test MSE (top-1 routing): 0.0303   (136 params)
Dense MLP baseline test MSE:  0.1853   (145 params)
MoE test MSE (soft routing):  0.6291
Routing purity (worst regime): 1.0000
Expert usage shares: [0.25, 0.25, 0.25, 0.25]
```

What the fixture shows, stated plainly:

- On this configuration the mixture has lower measured error: 0.0303 vs 0.1853 test MSE against a one-hidden-layer tanh MLP with slightly *more* parameters (136 vs 145) trained with the same optimizer budget. That is a measured trained-baseline comparison, not a proof that no dense model can fit this data - a wider or better-tuned MLP may close the gap, and if it does the honest outcome is to report the new numbers. The CI gates pin the mixture's own fit (MSE, routing purity, usage shares) as deterministic regression checks; they do not gate on the baseline staying weak.
- The router specializes: worst-regime routing purity is 1.0000 - every regime routes to exactly one expert on the test set.
- Soft evaluation is worse than sparse (0.6291 vs 0.0303): the training objective is sparse, and the soft evaluation has higher measured error on this fixture. This comparison does not isolate the cause. The suite pins the inversion so the number stays measured.
- Honest aux-loss note: on this symmetric fixture the load-balancing term is not load-bearing - regimes self-balance (max share 0.25 with and without it) and dropping it fits slightly better (0.0015 vs 0.0303). That is the opposite of the usual sales pitch for auxiliary losses, and it is what this fixture measured; the standard motivation is imbalanced expert traffic in production serving; that motivation is untested here, since this fixture is symmetric by construction.

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

39 Python tests, plus five substantive Node/Chrome tests for the route inspector. No Python runtime dependencies beyond Python 3.10+.

## Limitations

Experts and the router are linear, the fixture is synthetic and small, and routing is top-1 with fixed-branch derivatives - real MoE layers sit inside transformers and route tokens, not whole examples. The mechanics that transfer are the ones measured here: sparse gating with gate-probability weighting, specialization as a measurable property, and load balance as a measured distribution rather than an assumption.

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

The no-aux report regenerates to **0.0014970307791088836**, rounding to **0.0015**. The previous README's 0.0010 was stale text, corrected here without changing training or results. The old model docstring's squared batch-mean auxiliary formula also differs from its actual per-example implementation; both and the distinct Switch statistic are explained in ROUTES.md. No production traffic, robustness or causal mechanism claim follows from this synthetic fixture.
