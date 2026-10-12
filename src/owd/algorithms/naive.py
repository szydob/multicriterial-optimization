"""Algorytmy naiwne: bez filtracji oraz z filtracją punktów zdominowanych."""

from __future__ import annotations

from owd.types import Criterion, ParetoResult, Points


def naive_no_filter(points: Points, *, criteria: list[Criterion]) -> ParetoResult:
    """
    Algorytm naiwny bez filtracji (instrukcja, str. 2).

    TODO: implementacja wg pseudokodu z ćwiczenia 2.
    """
    raise NotImplementedError("naive_no_filter — do zaimplementowania")


def naive_with_filter(points: Points, *, criteria: list[Criterion]) -> ParetoResult:
    """
    Algorytm naiwny z filtracją zdominowanych po znalezieniu niezdominowanego (str. 3).

    TODO: implementacja wg pseudokodu z ćwiczenia 2.
    """
    raise NotImplementedError("naive_with_filter — do zaimplementowania")
