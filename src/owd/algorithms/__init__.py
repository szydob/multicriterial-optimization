"""Algorytmy wyznaczania zbioru punktów niezdominowanych P(X)."""

from __future__ import annotations

from collections.abc import Callable

from owd.algorithms.base import AlgorithmName, find_nondominated
from owd.algorithms.ideal_point import ideal_point
from owd.algorithms.kung import kung_luccio_preparata
from owd.algorithms.naive import naive_no_filter, naive_with_filter
from owd.types import ParetoResult

ALGORITHMS: dict[AlgorithmName, Callable[..., ParetoResult]] = {
    "naive_no_filter": naive_no_filter,
    "naive_with_filter": naive_with_filter,
    "ideal_point": ideal_point,
    "kung_luccio_preparata": kung_luccio_preparata,
}

__all__ = [
    "ALGORITHMS",
    "AlgorithmName",
    "find_nondominated",
    "ideal_point",
    "kung_luccio_preparata",
    "naive_no_filter",
    "naive_with_filter",
]
