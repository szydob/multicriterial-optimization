"""Tests for Cw3 scalarization on U=[1,4]²."""

from __future__ import annotations

import numpy as np
import pytest

from owd.scalarization import (
    DEFAULT_PROBLEM,
    INSTANCES,
    benchmark_scalarization,
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


def test_only_square_14_instance():
    assert list(INSTANCES.keys()) == ["square_1_4"]
    assert np.allclose(DEFAULT_PROBLEM.lb, [1.0, 1.0])
    assert np.allclose(DEFAULT_PROBLEM.ub, [4.0, 4.0])


def test_problem_eval_corners():
    f11 = DEFAULT_PROBLEM.evaluate(np.array([1.0, 1.0]))
    f44 = DEFAULT_PROBLEM.evaluate(np.array([4.0, 4.0]))
    assert f11[0] == pytest.approx(0.0)
    assert f44[1] == pytest.approx(0.0)
    assert f11[1] > f44[1]
    assert f44[0] > f11[0]


def test_weighted_norm():
    y = np.array([2.0, -4.0])
    lam = np.array([1.0, 0.5])
    assert weighted_norm(y, lam, 2.0) == pytest.approx(np.sqrt(1 * 4 + 0.5 * 16))
    assert weighted_norm(y, lam, float("inf")) == pytest.approx(max(1 * 2, 0.5 * 4))


def test_solves_succeed():
    sol = solve_weighted_sum(np.array([0.5, 0.5]))
    assert sol.success
    assert DEFAULT_PROBLEM.is_feasible(sol.u, tol=1e-5)
    f_min, f_max = estimate_objective_bounds()
    a = 0.5 * (f_min[1] + f_max[1])
    sol2 = solve_epsilon_constraint(float(a), primary=0)
    assert sol2.success
    sol3 = solve_distance(f_min - 0.1, np.array([0.5, 0.5]), 2.0)
    assert sol3.success


@pytest.mark.parametrize("n", [5, 10, 30])
@pytest.mark.parametrize(
    "sweep",
    [sweep_weighted_sum, sweep_epsilon_constraint, sweep_distance],
)
def test_sweep_returns_n_distinct_points(n, sweep):
    front = sort_front_by_f1(sweep(n, problem=DEFAULT_PROBLEM))
    assert len(front.decisions) == n
    uniq = np.unique(np.round(front.objectives, 5), axis=0)
    assert len(uniq) == n
    # Front should trade off: F1 increases roughly as F2 decreases
    assert front.objectives[0, 0] <= front.objectives[-1, 0] + 1e-8
    assert front.objectives[0, 1] >= front.objectives[-1, 1] - 1e-6


def test_benchmark_table():
    df = benchmark_scalarization(n_sweep=8)
    assert len(df) == 3
    assert set(df["method"]) == {"weighted_sum", "epsilon_constraint", "distance"}
    assert (df["n_success"] == 8).all()
    assert (df["dominated_share"] == 0.0).all()
