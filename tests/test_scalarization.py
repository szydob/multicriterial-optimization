"""Tests for Cw3 scalarization methods and Pareto sweeps."""

from __future__ import annotations

import numpy as np
import pytest

from owd.scalarization import (
    DEFAULT_PROBLEM,
    estimate_objective_bounds,
    solve_distance,
    solve_epsilon_constraint,
    solve_weighted_sum,
    sort_front_by_f1,
    sweep_distance,
    sweep_epsilon_constraint,
    sweep_weighted_sum,
    weighted_norm,
)
from owd.scalarization.problem import LB, UB


def test_problem_bounds_and_eval():
    u = np.array([2.0, 3.0])
    f = DEFAULT_PROBLEM.evaluate(u)
    assert f.shape == (2,)
    assert np.all(np.isfinite(f))
    assert np.all(LB == 1.0) and np.all(UB == 4.0)


def test_weighted_norm_chebyshev_and_p2():
    y = np.array([2.0, -4.0])
    lam = np.array([1.0, 0.5])
    assert weighted_norm(y, lam, 2.0) == pytest.approx(np.sqrt(1 * 4 + 0.5 * 16))
    assert weighted_norm(y, lam, float("inf")) == pytest.approx(max(1 * 2, 0.5 * 4))


def test_solve_weighted_sum_feasible():
    sol = solve_weighted_sum(np.array([0.5, 0.5]))
    assert sol.success
    assert np.all(sol.u >= LB - 1e-8) and np.all(sol.u <= UB + 1e-8)
    assert sol.f.shape == (2,)


def test_solve_epsilon_constraint_feasible():
    f_min, f_max = estimate_objective_bounds()
    a = 0.5 * (f_min[1] + f_max[1])
    sol = solve_epsilon_constraint(float(a), primary=0)
    assert sol.success
    assert sol.f[1] <= a + 1e-4


def test_solve_distance_to_utopia():
    f_min, _ = estimate_objective_bounds()
    ref = f_min - 0.1
    sol = solve_distance(ref, np.array([0.5, 0.5]), p=2.0)
    assert sol.success
    assert sol.scalar_value >= 0


def test_sweep_weighted_sum_reconstructs_front():
    front = sort_front_by_f1(sweep_weighted_sum(12))
    assert front.decisions.shape[0] >= 8
    assert front.objectives.shape == front.decisions.shape
    # F1 should be nondecreasing after sort
    assert np.all(np.diff(front.objectives[:, 0]) >= -1e-8)


def test_sweep_epsilon_reconstructs_front():
    front = sort_front_by_f1(sweep_epsilon_constraint(12, primary=0))
    assert np.any(front.successes)
    assert front.objectives.shape[1] == 2


def test_sweep_distance_reconstructs_front():
    front = sort_front_by_f1(sweep_distance(10, p=float("inf")))
    assert np.any(front.successes)
    assert front.method == "distance"
