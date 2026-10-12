from __future__ import annotations

import numpy as np

from owd.generators import LAB_EXAMPLE_POINTS, generate_points


def test_lab_example_shape() -> None:
    assert LAB_EXAMPLE_POINTS.shape == (12, 2)


def test_generate_uniform_reproducible() -> None:
    a = generate_points(10, 3, "uniform", seed=1)
    b = generate_points(10, 3, "uniform", seed=1)
    assert a.shape == (10, 3)
    np.testing.assert_allclose(a, b)


def test_generate_all_distributions() -> None:
    for dist in ("uniform", "normal", "exponential", "poisson"):
        pts = generate_points(8, 2, dist, seed=0)  # type: ignore[arg-type]
        assert pts.shape == (8, 2)
