"""Algorytm Kung–Luccio–Preparata (divide & conquer)."""
# Skład grupy:
# Konrad Biel 416189
# Szymon Dobrowolski 415835

from __future__ import annotations

import numpy as np

from owd.dominance import dominates, to_minimization
from owd.metrics import ComparisonCounter, timed_run
from owd.types import Criterion, ParetoResult, Points


def _filter_not_dominated_by(
    candidates: list[int],
    front: list[int],
    points: np.ndarray,
    counter: ComparisonCounter,
) -> list[int]:
    """Zachowaj kandydatów, których żaden punkt z frontu nie dominuje."""
    kept: list[int] = []
    for candidate in candidates:
        dominated = False
        for other in front:
            if dominates(points[other], points[candidate], counter=counter):
                dominated = True
                break
        if not dominated:
            kept.append(candidate)
    return kept


def _klp_rec(indices: list[int], points: np.ndarray, counter: ComparisonCounter) -> list[int]:
    """Rekurencyjne dzielenie po pierwszej współrzędnej + filtracja prawej połowy."""
    n = len(indices)
    if n <= 1:
        return list(indices)
    if n == 2:
        i, j = indices
        if dominates(points[i], points[j], counter=counter):
            return [i]
        if dominates(points[j], points[i], counter=counter):
            return [j]
        return [i, j]

    ordered = sorted(indices, key=lambda idx: (points[idx, 0], idx))
    mid = n // 2
    left_half = ordered[:mid]
    right_half = ordered[mid:]

    left_nd = _klp_rec(left_half, points, counter)
    right_nd = _klp_rec(right_half, points, counter)
    right_kept = _filter_not_dominated_by(right_nd, left_nd, points, counter)

    # Gdy połowy nachodzą na pierwszej osi, lewa też może być zdominowana przez prawą.
    left_max = max(points[i, 0] for i in left_half)
    right_min = min(points[i, 0] for i in right_half)
    if left_max < right_min:
        return left_nd + right_kept

    left_kept = _filter_not_dominated_by(left_nd, right_kept, points, counter)
    return left_kept + right_kept


def kung_luccio_preparata(points: Points, *, criteria: list[Criterion]) -> ParetoResult:
    """
    Klasyczny algorytm Kunga, Luccio i Preparaty (divide & conquer).

    Po normalizacji do minimalizacji: sortowanie / podział po 1. kryterium,
    rekurencja na połowach i filtracja zbioru niezdominowanego prawej strony.
    """
    counter = ComparisonCounter()
    with timed_run() as timer:
        original = np.asarray(points, dtype=float)
        normalized = to_minimization(original, criteria)
        if not np.isfinite(normalized).all():
            raise ValueError("points must contain only finite values")

        selected = _klp_rec(list(range(len(normalized))), normalized, counter)
        indices = np.array(sorted(selected), dtype=np.int_)

    return ParetoResult(
        nondominated=original[indices].copy(),
        nondominated_indices=indices,
        stats=counter.as_stats(timer.elapsed_seconds),
        algorithm="kung_luccio_preparata",
    )
