"""Generowanie losowych zbiorów X zgodnie z instrukcją lab 2."""

from __future__ import annotations

import numpy as np

from owd.types import DistributionName, Points


def generate_points(
    n: int,
    k: int,
    distribution: DistributionName = "uniform",
    *,
    seed: int | None = None,
    low: float = 0.0,
    high: float = 2.0,
    loc: float = 1.0,
    scale: float = 1.0,
    lam: float = 2.0,
) -> Points:
    """
    Generuje macierz (n, k) punktów.

    Domyślne parametry z instrukcji:
    - jednostajny na [0, 2]
    - Gauss(1, 1) → loc=1, scale=1
    - ekspencjalny(1) → scale=1
    - Poisson(2) → lam=2
    """
    if n < 1 or k < 1:
        raise ValueError("n and k must be >= 1")

    rng = np.random.default_rng(seed)

    if distribution == "uniform":
        return rng.uniform(low, high, size=(n, k))
    if distribution == "normal":
        return rng.normal(loc, scale, size=(n, k))
    if distribution == "exponential":
        return rng.exponential(scale, size=(n, k))
    if distribution == "poisson":
        return rng.poisson(lam, size=(n, k)).astype(float)

    raise ValueError(f"unknown distribution: {distribution}")


# Przykład z instrukcji (str. 5) — do weryfikacji ręcznej.
LAB_EXAMPLE_POINTS: Points = np.array(
    [
        [5, 5],
        [3, 6],
        [4, 4],
        [5, 3],
        [3, 3],
        [1, 8],
        [3, 4],
        [4, 5],
        [3, 10],
        [6, 6],
        [4, 1],
        [3, 5],
    ],
    dtype=float,
)
