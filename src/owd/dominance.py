"""Porównania dominacji Pareta (po normalizacji kierunku min/max)."""

from __future__ import annotations

import numpy as np

from owd.metrics import ComparisonCounter
from owd.types import Criterion, Points, Sense


def to_minimization(points: Points, criteria: list[Criterion]) -> Points:
    """Zwraca kopię punktów w przestrzeni, gdzie każde kryterium jest minimalizowane."""
    out = np.asarray(points, dtype=float).copy()
    if out.ndim != 2:
        raise ValueError("points must be a 2D array of shape (n, k)")
    if out.shape[1] != len(criteria):
        raise ValueError("number of columns must match number of criteria")
    for j, crit in enumerate(criteria):
        if crit.sense is Sense.MAX:
            out[:, j] = -out[:, j]
    return out


def dominates(
    a: np.ndarray,
    b: np.ndarray,
    *,
    counter: ComparisonCounter | None = None,
) -> bool:
    """
    True iff a ⪯ b i a ≠ b (a nie jest gorsze w żadnym kryterium i lepsze w ≥1).

    Zakłada, że obie współrzędne są już w sensie minimalizacji.
    """
    if counter is not None:
        counter.point_comparisons += 1

    better_or_equal = True
    strictly_better = False
    for ai, bi in zip(a, b, strict=True):
        if counter is not None:
            counter.coordinate_comparisons += 1
        if ai > bi:
            better_or_equal = False
            break
        if ai < bi:
            strictly_better = True

    return better_or_equal and strictly_better
