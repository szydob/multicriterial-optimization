"""Liczniki porównań i pomiar czasu."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from owd.types import RunStats


@dataclass
class ComparisonCounter:
    point_comparisons: int = 0
    coordinate_comparisons: int = 0

    def as_stats(self, elapsed_seconds: float = 0.0) -> RunStats:
        return RunStats(
            point_comparisons=self.point_comparisons,
            coordinate_comparisons=self.coordinate_comparisons,
            elapsed_seconds=elapsed_seconds,
        )


@dataclass
class Timer:
    elapsed_seconds: float = field(default=0.0, init=False)
    _start: float | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer was not started")
        self.elapsed_seconds = time.perf_counter() - self._start
        self._start = None
        return self.elapsed_seconds


@contextmanager
def timed_run() -> Iterator[Timer]:
    timer = Timer()
    timer.start()
    try:
        yield timer
    finally:
        if timer._start is not None:
            timer.stop()
