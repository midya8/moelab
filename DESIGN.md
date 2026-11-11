# MoeLab design

The original model uses a linear router and linear experts. Its selected gate
probability stays in the prediction. It has no capacity limit or token dropping;
the later capacity allocator is a separate experiment.

## Training

`data.py` generates four noisy input regimes with different linear target maps.
A single linear map cannot represent the construction. The trained dense tanh
baseline in `eval.py` tests a different question: how one nonlinear model fits
this particular fixture under the recorded training budget.

`moe.py` computes gates `g = softmax(R x + c)` and expert outputs
`h_e = W_e . x + b_e`. Let `e` be the lowest-index maximum gate. Training uses
`y = g_e h_e`; `predict_soft()` is a separate evaluation path.

With the selected branch fixed and `s = 2(y - target)`, the task derivatives are:

```
dL/dW_ej = s g_e x_j
dL/db_e  = s g_e
dL/dz_k  = s h_e g_e (delta_ek - g_k)
```

Router derivatives follow from `z = R x + c`. Other experts receive no task-loss
update. This is an ordinary branch derivative, not a straight-through estimator
for argmax. Finite-difference tests stay away from routing boundaries.

The per-example auxiliary penalty is `A = E sum_k g_k^2`. With
`S2 = sum_k g_k^2`, its logit derivative is `2 E g_k (g_k - S2)`.
Averaging this penalty across rows differs from penalizing squared batch-mean
gates and from Switch's frequency/probability statistic. [ROUTES.md](ROUTES.md)
compares all three, including cases where a low penalty still sends every row
to one expert.

## Inspectors

- `routes.py` and `envelope.py` inspect exact rational routing boundaries and
  separate them from approximate predictions. [ROUTES.md](ROUTES.md) defines
  the input schema, tie rules, derivatives and floating-point limits.
- `capacity.py` optimizes a supplied finite rational dispatch table. It first
  maximizes dispatch count, then retained gate mass. `capverify.py` independently
  enumerates assignments. [CAPACITY.md](CAPACITY.md) defines the objective,
  bottleneck witnesses and task-error limits. This allocator does not train or
  alter the original model.
- `gate_regret.py` evaluates worst retained-mass regret for one fixed nominal
  dispatch under supplied normalized probability bounds. `gate_verify.py`
  independently enumerates vertices and assignments. [GATE-REGRET.md](GATE-REGRET.md)
  defines the assumptions, one-table quantifiers, dual witnesses and loose
  separate-table bound. It neither trains nor calibrates a model.
- `eval.py` reports the trained sparse/soft errors, dense baseline, per-regime
  routing purity, usage shares and aux/no-aux results. `cli.py` exposes the
  evaluation and regression gates. Gates pin the mixture's behavior, not a
  requirement that a baseline lose.
