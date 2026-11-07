# MoeLab design

Sparse (Switch-style) mixture-of-experts, dependency-free.

- data.py: four regimes; input = regime center + Gaussian noise; target =
  regime-specific linear map. A single shared linear map cannot fit it (width does not extend a linear model); dense nonlinear baselines are compared empirically in eval.py.
- moe.py: router softmax + linear experts. Trained prediction is
  g_top * expert_top(x) (argmax straight-through). Gradients derived by hand:
  dL/dw_e = s g_e x, dL/dz_k = s y_e g_e (delta_ek - g_k), load-balance aux
  E * sum g_k^2 with d/d R_ej = 2 E x_j g_e (g_e - S2). All FD-verified.
- eval.py: MoE vs same-budget dense MLP; routing purity per regime; expert
  usage shares; soft-vs-sparse gap; aux-on vs aux-off.
- cli.py: evaluate + gates both sides.
