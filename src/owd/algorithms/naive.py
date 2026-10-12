"""Algorytmy naiwne: bez filtracji oraz z filtracją punktów zdominowanych."""
# Skład grupy:
# Konrad Biel 416189
# Szymon Dobrowolski 415835

from __future__ import annotations

import numpy as np

from owd.dominance import to_minimization
from owd.metrics import ComparisonCounter, timed_run
from owd.types import Criterion, ParetoResult, Points


def _compare(a: np.ndarray, b: np.ndarray, counter: ComparisonCounter) -> int:
    """-1: a dominuje b, 1: b dominuje a, 0: równość lub nieporównywalność."""
    counter.point_comparisons += 1
    better = worse = False
    for ai, bi in zip(a, b, strict=True):
        counter.coordinate_comparisons += 1
        better |= ai < bi
        worse |= ai > bi
        # Przeciwne kierunki na dwóch osiach wystarczą do nieporównywalności.
        if better and worse:
            return 0
    # Jak w kontrakcie projektu, równe punkty nie dominują się ściśle.
    return -1 if better else (1 if worse else 0)


def _naive(points: Points, criteria: list[Criterion], *, filtering: bool) -> ParetoResult:
    counter = ComparisonCounter()
    with timed_run() as timer:
        normalized = to_minimization(points, criteria)
        if not np.isfinite(normalized).all():
            raise ValueError("points must contain only finite values")
        original = np.asarray(points)
        # Indeksy pozwalają usuwać punkty z listy roboczej bez zmiany wejścia.
        remaining = list(range(len(normalized)))
        selected: set[int] = set()
        while remaining:
            candidate = remaining[0]
            changed = False  # fl z pseudokodu: czy zastąpiono początkowe Y?
            position = 1
            while position < len(remaining):
                other = remaining[position]
                relation = _compare(normalized[candidate], normalized[other], counter)
                if relation == -1:
                    remaining.pop(position)
                    # Następny element zajmuje pozycję usuniętego punktu.
                elif relation == 1:
                    remaining.remove(candidate)
                    candidate = other
                    changed = True
                    # Usunięto wcześniejszy element, więc nie zwiększamy position.
                else:
                    position += 1

            # Bez filtra to samo Y może być znalezione w kilku przebiegach.
            selected.add(candidate)
            if filtering:
                # Nowe Y może dominować punkty sprawdzone przed jego podmianą.
                # Dopiero po filtracji całej listy można bezpiecznie usunąć Y.
                remaining = [
                    index
                    for index in remaining
                    if index != candidate
                    and _compare(normalized[candidate], normalized[index], counter) != -1
                ]
                if len(remaining) == 1:
                    selected.add(remaining.pop())
            elif not changed:
                remaining.remove(candidate)
            # Przy fl=1 bez filtra Y zostaje do ponownego przeglądu (slajd 9).
            # Inaczej np. (3,6) byłby błędnie uznany za niezdominowany.

        # Wynik ma kolejność wejściową; zachowujemy też indeksy równych punktów.
        indices = np.array(sorted(selected), dtype=np.int_)
    return ParetoResult(
        nondominated=original[indices].copy(),
        nondominated_indices=indices,
        stats=counter.as_stats(timer.elapsed_seconds),
        algorithm="naive_with_filter" if filtering else "naive_no_filter",
    )


def naive_no_filter(points: Points, *, criteria: list[Criterion]) -> ParetoResult:
    """Wariant ze slajdu 2: zachowuje zmienione Y do kolejnego przebiegu (fl=1)."""
    return _naive(points, criteria, filtering=False)


def naive_with_filter(points: Points, *, criteria: list[Criterion]) -> ParetoResult:
    """Wariant ze slajdu 3: po znalezieniu Y filtruje całą pozostałą listę."""
    return _naive(points, criteria, filtering=True)
