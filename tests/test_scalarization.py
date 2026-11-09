"""Tests for Cw3 scalarization methods, sweeps and timing benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from owd.scalarization import (
    DEFAULT_PROBLEM,
    benchmark_scalarization,
    estimate_objective_bounds,
    get_problem,
    solve_distance,
    solve_epsilon_constraint,
    solve_weighted_sum,
    sort_front_by_f1,
    sweep_distance,
    sweep_epsilon_constraint,
    sweep_weighted_sum,
    weighted_norm,
)


def test_problem_bounds_and_eval():
    u = np.array([2.0, 3.0])
    f = DEFAULT_PROBLEM.evaluate(u)
    assert f.shape == (2,)
    assert np.all(np.isfinite(f))


def test_lab_pdf_instances_feasible():
    sq = get_problem("square_m1_1")
    disk = get_problem("unit_disk")
    assert sq.is_feasible(np.array([0.0, 0.0]))
    assert disk.is_feasible(np.array([0.0, 0.0]))
    assert not disk.is_feasible(np.array([1.0, 1.0]))


def test_weighted_norm_chebyshev_and_p2():
    y = np.array([2.0, -4.0])
    lam = np.array([1.0, 0.5])
    assert weighted_norm(y, lam, 2.0) == pytest.approx(np.sqrt(1 * 4 + 0.5 * 16))
    assert weighted_norm(y, lam, float("inf")) == pytest.approx(max(1 * 2, 0.5 * 4))


def test_solve_weighted_sum_feasible_and_timed():
    sol = solve_weighted_sum(np.array([0.5, 0.5]))
    assert sol.success
    assert sol.elapsed_seconds >= 0.0
    assert sol.f.shape == (2,)


def test_solve_epsilon_on_pdf_square():
    problem = get_problem("square_m1_1")
    f_min, f_max = estimate_objective_bounds(problem)
    a = 0.5 * (f_min[1] + f_max[1])
    sol = solve_epsilon_constraint(float(a), primary=0, problem=problem)
    assert sol.success
    assert sol.f[1] <= a + 1e-3


def test_solve_distance_on_unit_disk():
    problem = get_problem("unit_disk")
    f_min, _ = estimate_objective_bounds(problem)
    ref = f_min - 0.1
    sol = solve_distance(ref, np.array([0.5, 0.5]), p=2.0, problem=problem)
    assert sol.success
    assert problem.is_feasible(sol.u, tol=1e-5)


def test_sweep_weighted_sum_reconstructs_front():
    front = sort_front_by_f1(sweep_weighted_sum(12))
    assert front.decisions.shape[0] >= 8
    assert front.elapsed_seconds > 0
    assert len(front.per_solve_seconds) == 12 or len(front.per_solve_seconds) == len(
        front.decisions
    )
    assert np.all(np.diff(front.objectives[:, 0]) >= -1e-8)


def test_sweep_epsilon_and_distance():
    front_e = sort_front_by_f1(sweep_epsilon_constraint(10, primary=0))
    front_d = sort_front_by_f1(sweep_distance(10, p=float("inf")))
    assert np.any(front_e.successes) or len(front_e.decisions) > 0
    assert front_d.method == "distance"
    assert front_e.elapsed_seconds > 0


def test_benchmark_scalarization_table():
    df = benchmark_scalarization(instance_ids=["square_m1_1"], n_sweep=8)
    assert len(df) == 3
    assert set(df["method"]) == {"weighted_sum", "epsilon_constraint", "distance"}
    assert (df["sweep_time_ms"] >= 0).all()
