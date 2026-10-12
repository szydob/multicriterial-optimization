"""Wspólny kontrakt algorytmów P(X)."""

from __future__ import annotations

from typing import Literal

from owd.types import Criterion, ParetoResult, Points, Sense

AlgorithmName = Literal[
    "naive_no_filter",
    "naive_with_filter",
    "ideal_point",
    "kung_luccio_preparata",
]


def default_criteria(k: int, sense: Sense = Sense.MIN) -> list[Criterion]:
    return [Criterion(name=f"kryterium{i + 1}", sense=sense) for i in range(k)]


def find_nondominated(
    points: Points,
    algorithm: AlgorithmName,
    criteria: list[Criterion] | None = None,
) -> ParetoResult:
    """Dispatcher — wybór zaimplementowanego algorytmu po nazwie."""
    from owd.algorithms import ALGORITHMS

    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm: {algorithm}")
    if criteria is None:
        criteria = default_criteria(points.shape[1])
    return ALGORITHMS[algorithm](points, criteria=criteria)
