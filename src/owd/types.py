"""Wspólne typy i kontrakty danych."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Points = NDArray[np.floating]


class Sense(StrEnum):
    """Kierunek optymalizacji pojedynczego kryterium."""

    MIN = "min"
    MAX = "max"


@dataclass(frozen=True)
class Criterion:
    name: str
    sense: Sense = Sense.MIN


@dataclass
class RunStats:
    """Metryki wydajności jednego przebiegu algorytmu."""

    point_comparisons: int = 0
    coordinate_comparisons: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class ParetoResult:
    """Wynik wyznaczania P(X)."""

    nondominated: Points
    nondominated_indices: NDArray[np.int_]
    stats: RunStats = field(default_factory=RunStats)
    algorithm: str = ""


DistributionName = Literal["uniform", "normal", "exponential", "poisson"]
