from __future__ import annotations

import numpy as np

from owd.dominance import dominates, to_minimization
from owd.metrics import ComparisonCounter
from owd.types import Criterion, Sense


def test_dominates_basic() -> None:
    assert dominates(np.array([1.0, 1.0]), np.array([2.0, 2.0]))
    assert not dominates(np.array([1.0, 3.0]), np.array([2.0, 2.0]))
    assert not dominates(np.array([1.0, 1.0]), np.array([1.0, 1.0]))


def test_dominates_counts() -> None:
    counter = ComparisonCounter()
    dominates(np.array([1.0, 2.0]), np.array([3.0, 4.0]), counter=counter)
    assert counter.point_comparisons == 1
    assert counter.coordinate_comparisons == 2


def test_to_minimization_flips_max() -> None:
    pts = np.array([[1.0, 5.0], [2.0, 3.0]])
    criteria = [
        Criterion("a", Sense.MIN),
        Criterion("b", Sense.MAX),
    ]
    out = to_minimization(pts, criteria)
    np.testing.assert_allclose(out[:, 0], [1.0, 2.0])
    np.testing.assert_allclose(out[:, 1], [-5.0, -3.0])
