"""Synthetic regime-switching fixture: K regimes, each with its own linear map.

Inputs are cluster centers (one per regime) plus seeded Gaussian noise; the
target is that regime's own linear function of the input. A single linear map
cannot represent the data, because the sign and scale of the
input-target relationship flips per regime; a mixture with a router can.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

N_REGIMES = 4
DIM = 16

# Regime centers on well-separated axis-aligned directions.
CENTERS: List[List[float]] = [
    [2.0 if i == r * 4 else 0.0 for i in range(DIM)] for r in range(N_REGIMES)
]

# Each regime maps input to target with its own weight pattern.
REGIME_WEIGHTS: List[List[float]] = [
    [math.sin(1.7 * (r + 1) * (i + 1)) for i in range(DIM)] for r in range(N_REGIMES)
]
REGIME_BIAS = [0.5 * (r + 1) for r in range(N_REGIMES)]


def make_split(n_per_regime: int, seed: int, noise: float = 0.3) -> List[Tuple[List[float], float, int]]:
    rng = random.Random(seed)
    out = []
    for regime in range(N_REGIMES):
        for _ in range(n_per_regime):
            x = [c + rng.gauss(0.0, noise) for c in CENTERS[regime]]
            y = sum(w * xi for w, xi in zip(REGIME_WEIGHTS[regime], x)) + REGIME_BIAS[regime]
            out.append((x, y, regime))
    rng.shuffle(out)
    return out
