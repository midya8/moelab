# MoeLab

Mixture-of-experts from scratch: a sparse router and per-expert models trained with hand-derived, finite-difference-verified gradients, on a fixture built so a single dense model cannot win. No ML frameworks.

## What it does

- Builds a regime-switching fixture: four input regimes, each with its own linear input-target map, so the sign and scale of the relationship flips per regime. One shared linear-ish map cannot represent it at any width.
- Trains a Switch-style sparse mixture: the router softmax scores each expert, the top expert answers, and the prediction is the gate probability times that expert's output. The argmax choice is straight-through; gradients flow through the gate probability and the chosen expert, all derived by hand and finite-difference verified in the suite.
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

- The mixture wins on structure, not size: 0.0303 vs 0.1853 test MSE against a dense MLP with slightly *more* parameters (136 vs 145), and the baseline gate (`--min-mlp-mse`) fails CI if the baseline ever catches up, so the comparison cannot quietly stop separating the architectures.
- The router specializes: worst-regime routing purity is 1.0000 - every regime routes to exactly one expert on the test set.
- Soft evaluation is worse than sparse (0.6291 vs 0.0303): the training objective is sparse, so averaging experts that specialized on different regimes actively hurts. The suite pins the inversion so the number stays measured.
- Honest aux-loss note: on this symmetric fixture the load-balancing term is not load-bearing - regimes self-balance (max share 0.25 with and without it) and dropping it fits slightly better (0.0010 vs 0.0303). That is the opposite of the usual sales pitch for auxiliary losses, and it is what this fixture measured; imbalanced real traffic is where the term earns its keep.

## CLI

```bash
python -m moelab evaluate
python -m moelab gates --max-mse 0.06 --min-mlp-mse 0.15 --min-purity 0.95 --max-expert-share 0.26
```

Gates exit 1 with named failures; the demo asserts both sides (an impossible purity bar fails).

## Development

```bash
python -m unittest discover -s tests
bash examples/demo.sh
```

9 tests. No dependencies beyond Python 3.10+.

## Limitations

Experts and the router are linear, the fixture is synthetic and small, and routing is top-1 with straight-through gradients - real MoE layers sit inside transformers and route tokens, not whole examples. The mechanics that transfer are the ones measured here: sparse gating with gate-probability weighting, specialization as a measurable property, and load balance as a measured distribution rather than an assumption.

## License

MIT
