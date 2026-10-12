"""Generowanie klatek animacji budowy zbioru P(X)."""

from __future__ import annotations

import numpy as np

from owd.dominance import dominates, to_minimization
from owd.types import Criterion, Points


def generate_filter_frames(
    points: Points,
    criteria: list[Criterion],
    *,
    max_frames: int = 80,
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Klatki w stylu metody punktu idealnego:
    sortowanie odległością od xmin, potem filtracja zdominowanych.

    Zwraca (frames_nondom, frames_compared) — listy indeksów na każdej klatce.
    """
    original = np.asarray(points, dtype=float)
    normalized = to_minimization(original, criteria)
    n = len(normalized)
    if n == 0:
        return [[]], [[]]

    xmin = normalized.min(axis=0)
    dist2 = np.sum((normalized - xmin) ** 2, axis=1)
    remaining = list(np.argsort(dist2, kind="mergesort"))
    selected: list[int] = []

    frames_nondom: list[list[int]] = [[]]
    frames_compared: list[list[int]] = [[]]

    while remaining:
        candidate = remaining[0]
        selected = sorted(selected + [candidate])
        frames_nondom.append(list(selected))
        frames_compared.append([candidate])

        new_remaining: list[int] = []
        for index in remaining[1:]:
            frames_nondom.append(list(selected))
            frames_compared.append([candidate, index])
            if not dominates(normalized[candidate], normalized[index]):
                new_remaining.append(index)
        remaining = new_remaining

        if len(frames_nondom) >= max_frames:
            break

    # Domknięcie: końcowy front (gdy przerwano wcześniej i tak dociągamy).
    if remaining:
        # dokończ bez dodatkowych klatek porównań
        while remaining:
            candidate = remaining[0]
            selected = sorted(selected + [candidate])
            remaining = [
                index
                for index in remaining[1:]
                if not dominates(normalized[candidate], normalized[index])
            ]
        frames_nondom.append(list(selected))
        frames_compared.append([])

    return frames_nondom, frames_compared
