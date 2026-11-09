"""Compare S1/S2/S3 on U=[1,4]²: wall time and front quality."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from owd.scalarization.methods import (
    FrontApproximation,
    sort_front_by_f1,
    sweep_distance,
    sweep_epsilon_constraint,
    sweep_weighted_sum,
)
from owd.scalarization.problem import DEFAULT_PROBLEM, BiObjectiveProblem, estimate_objective_bounds


@dataclass
class FrontQuality:
    n_points: int
    n_success: int
    success_rate: float
    f1_span: float
    f2_span: float
    mean_spacing: float
    dominated_share: float


def _pairwise_dominated_share(obj: np.ndarray) -> float:
    if len(obj) == 0:
        return 0.0
    dominated = 0
    for i, a in enumerate(obj):
        for j, b in enumerate(obj):
            if i == j:
                continue
            if np.all(b <= a) and np.any(b < a):
                dominated += 1
                break
    return dominated / len(obj)


def front_quality(front: FrontApproximation) -> FrontQuality:
    sorted_f = sort_front_by_f1(front)
    obj = sorted_f.objectives
    n_succ = len(obj)
    if n_succ == 0:
        return FrontQuality(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    diffs = np.linalg.norm(np.diff(obj, axis=0), axis=1) if n_succ > 1 else np.array([0.0])
    return FrontQuality(
        n_points=n_succ,
        n_success=n_succ,
        success_rate=1.0,
        f1_span=float(obj[:, 0].max() - obj[:, 0].min()),
        f2_span=float(obj[:, 1].max() - obj[:, 1].min()),
        mean_spacing=float(np.mean(diffs)) if len(diffs) else 0.0,
        dominated_share=_pairwise_dominated_share(obj),
    )


def run_method_sweep(
    method: str,
    n: int,
    problem: BiObjectiveProblem,
    *,
    p: float = 2.0,
) -> FrontApproximation:
    if method == "weighted_sum":
        return sweep_weighted_sum(n, problem=problem)
    if method == "epsilon_constraint":
        return sweep_epsilon_constraint(n, primary=0, problem=problem)
    if method == "distance":
        return sweep_distance(n, p=p, problem=problem)
    raise ValueError(f"unknown method: {method}")


METHOD_LABELS = {
    "weighted_sum": "S1 suma ważona",
    "epsilon_constraint": "S2 ε-ograniczenia",
    "distance": "S3 odległość ℓ_p",
}


def benchmark_scalarization(
    *,
    n_sweep: int = 30,
    p_distance: float = 2.0,
) -> pd.DataFrame:
    """Compare S1/S2/S3 sweeps on the sole instance U=[1,4]²."""
    problem = DEFAULT_PROBLEM
    rows: list[dict] = []
    for method in ("weighted_sum", "epsilon_constraint", "distance"):
        front = run_method_sweep(method, n_sweep, problem, p=p_distance)
        q = front_quality(front)
        per = front.per_solve_seconds
        rows.append(
            {
                "instance_id": problem.id,
                "instance": problem.name,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_sweep": n_sweep,
                "n_success": q.n_success,
                "success_rate": q.success_rate,
                "sweep_time_s": front.elapsed_seconds,
                "sweep_time_ms": front.elapsed_seconds * 1000.0,
                "mean_solve_ms": float(np.mean(per) * 1000.0) if len(per) else float("nan"),
                "median_solve_ms": float(np.median(per) * 1000.0) if len(per) else float("nan"),
                "f1_span": q.f1_span,
                "f2_span": q.f2_span,
                "mean_spacing": q.mean_spacing,
                "dominated_share": q.dominated_share,
            }
        )
    return pd.DataFrame(rows)


def benchmark_single_solves(
    *,
    instance_id: str = "square_1_4",
    n_params: int = 15,
    p_distance: float = 2.0,
) -> pd.DataFrame:
    from owd.scalarization.methods import (
        solve_distance,
        solve_epsilon_constraint,
        solve_weighted_sum,
    )
    from owd.scalarization.problem import get_problem

    problem = get_problem(instance_id)
    f_min, f_max = estimate_objective_bounds(problem)
    ts = np.linspace(1e-3, 1.0 - 1e-3, n_params)
    rows: list[dict] = []

    for t in ts:
        lam = np.array([t, 1.0 - t], dtype=float)
        sol = solve_weighted_sum(lam, problem=problem)
        rows.append(
            {
                "instance": problem.name,
                "method": "weighted_sum",
                "method_label": METHOD_LABELS["weighted_sum"],
                "param": float(t),
                "success": sol.success,
                "solve_ms": sol.elapsed_seconds * 1000.0,
                "F1": float(sol.f[0]),
                "F2": float(sol.f[1]),
            }
        )

    # ε along Pareto-relevant span of F2
    sol_lo = solve_weighted_sum(np.array([1.0 - 1e-3, 1e-3]), problem=problem)
    sol_hi = solve_weighted_sum(np.array([1e-3, 1.0 - 1e-3]), problem=problem)
    a_lo = min(sol_lo.f[1], sol_hi.f[1])
    a_hi = max(sol_lo.f[1], sol_hi.f[1])
    for a in np.linspace(a_lo, a_hi, n_params):
        sol = solve_epsilon_constraint(float(a), primary=0, problem=problem)
        rows.append(
            {
                "instance": problem.name,
                "method": "epsilon_constraint",
                "method_label": METHOD_LABELS["epsilon_constraint"],
                "param": float(a),
                "success": sol.success,
                "solve_ms": sol.elapsed_seconds * 1000.0,
                "F1": float(sol.f[0]),
                "F2": float(sol.f[1]),
            }
        )

    ref = np.minimum(sol_lo.f, sol_hi.f) - 0.05
    for t in ts:
        lam = np.array([t, 1.0 - t], dtype=float)
        sol = solve_distance(ref, lam, p_distance, problem=problem)
        rows.append(
            {
                "instance": problem.name,
                "method": "distance",
                "method_label": METHOD_LABELS["distance"],
                "param": float(t),
                "success": sol.success,
                "solve_ms": sol.elapsed_seconds * 1000.0,
                "F1": float(sol.f[0]),
                "F2": float(sol.f[1]),
            }
        )

    return pd.DataFrame(rows)
