import numpy as np
import pytest

from owd.algorithms.base import default_criteria
from owd.algorithms.ideal_point import ideal_point
from owd.algorithms.kung import kung_luccio_preparata
from owd.generators import LAB_EXAMPLE_POINTS
from owd.types import Criterion, Sense


@pytest.mark.parametrize("algorithm", [ideal_point, kung_luccio_preparata])
def test_slide_example(algorithm):
    points = LAB_EXAMPLE_POINTS.copy()
    result = algorithm(points, criteria=default_criteria(2))
    np.testing.assert_array_equal(result.nondominated_indices, [4, 5, 10])
    np.testing.assert_array_equal(result.nondominated, [[3, 3], [1, 8], [4, 1]])
    np.testing.assert_array_equal(points, LAB_EXAMPLE_POINTS)
    assert result.stats.elapsed_seconds >= 0
    assert result.algorithm == algorithm.__name__


@pytest.mark.parametrize("algorithm", [ideal_point, kung_luccio_preparata])
@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_against_pairwise_reference(algorithm, k):
    rng = np.random.default_rng(42)
    criteria = [Criterion(str(j), Sense.MAX if j % 2 else Sense.MIN) for j in range(k)]
    for n in [0, 1, 2, 15, 40]:
        points = rng.integers(-3, 4, size=(n, k)).astype(float)
        normalized = points * np.array([1 if c.sense == Sense.MIN else -1 for c in criteria])
        expected = [
            i
            for i in range(n)
            if not any(
                np.all(other <= normalized[i]) and np.any(other < normalized[i])
                for other in normalized
            )
        ]
        result = algorithm(points, criteria=criteria)
        np.testing.assert_array_equal(result.nondominated_indices, expected)
        np.testing.assert_array_equal(result.nondominated, points[expected])


@pytest.mark.parametrize("algorithm", [ideal_point, kung_luccio_preparata])
def test_duplicates_and_validation(algorithm):
    result = algorithm(np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]]), criteria=default_criteria(2))
    np.testing.assert_array_equal(result.nondominated_indices, [0, 1])
    with pytest.raises(ValueError, match="number of columns"):
        algorithm(np.zeros((2, 3)), criteria=default_criteria(2))
    with pytest.raises(ValueError, match="2D"):
        algorithm(np.zeros(2), criteria=default_criteria(2))
    with pytest.raises(ValueError, match="finite"):
        algorithm(np.array([[np.nan, 1.0]]), criteria=default_criteria(2))
