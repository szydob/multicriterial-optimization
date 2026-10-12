"""Algorytm oparty o punkt idealny (sortowanie odległości + filtracja)."""
# Skład grupy:
# Konrad Biel 416189
# Szymon Dobrowolski 415835

from __future__ import annotations

import numpy as np

from owd.dominance import dominates, to_minimization
from owd.metrics import ComparisonCounter, timed_run
from owd.types import Criterion, ParetoResult, Points


def ideal_point(points: Points, *, criteria: list[Criterion]) -> ParetoResult:
    """
    Metoda punktu idealnego (instrukcja, str. 4).

    1. xmin ← minima współrzędnych (po normalizacji do minimalizacji)
    2. sortuj rosnąco d(xmin, X(j))^2
    3. bierz kolejnego kandydata i filtruj punkty przez niego zdominowane
    """
    counter = ComparisonCounter()
    with timed_run() as timer:
        original = np.asarray(points, dtype=float)
        normalized = to_minimization(original, criteria)
        if not np.isfinite(normalized).all():
            raise ValueError("points must contain only finite values")

        n = len(normalized)
        if n == 0:
            indices = np.array([], dtype=np.int_)
        else:
            xmin = normalized.min(axis=0)
            dist2 = np.sum((normalized - xmin) ** 2, axis=1)
            # Stabilne sortowanie — przy równej odległości zachowujemy kolejność wejścia.
            remaining = list(np.argsort(dist2, kind="mergesort"))
            selected: list[int] = []
            while remaining:
                candidate = remaining[0]
                selected.append(candidate)
                remaining = [
                    index
                    for index in remaining[1:]
                    if not dominates(normalized[candidate], normalized[index], counter=counter)
                ]
            indices = np.array(sorted(selected), dtype=np.int_)

    return ParetoResult(
        nondominated=original[indices].copy(),
        nondominated_indices=indices,
        stats=counter.as_stats(timer.elapsed_seconds),
        algorithm="ideal_point",
    )
