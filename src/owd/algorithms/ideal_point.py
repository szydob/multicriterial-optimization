"""Algorytm oparty o punkt idealny (sortowanie odległości + filtracja)."""

from __future__ import annotations

from owd.types import Criterion, ParetoResult, Points


def ideal_point(points: Points, *, criteria: list[Criterion]) -> ParetoResult:
    """
    Metoda punktu idealnego (instrukcja, str. 4).

    1. xmin ← minima współrzędnych
    2. sortuj rosnąco d(xmin, X(j))^2
    3. filtruj zdominowane względem kolejnych kandydatów

    TODO: implementacja wg pseudokodu.
    """
    raise NotImplementedError("ideal_point — do zaimplementowania")
