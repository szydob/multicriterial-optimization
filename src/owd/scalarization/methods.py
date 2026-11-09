"""Three Cw3 scalarization methods + parameter sweeps with timing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import minimize

from owd.scalarization.problem import DEFAULT_PROBLEM, BiObjectiveProblem

MethodName = Literal["weighted_sum", "epsilon_constraint", "distance"]


@dataclass
class ScalarizationSolution:
    """Single scalarized solve result."""

    method: MethodName
    u: np.ndarray
    f: np.ndarray
    scalar_value: float
    success: bool
    message: str
    params: dict
    elapsed_seconds: float = 0.0


@dataclass
class FrontApproximation:
    """Discrete approximation of P(U,F) / FP(U) from a parameter sweep."""

    method: MethodName
    decisions: np.ndarray
    objectives: np.ndarray
    params: list[dict]
    successes: np.ndarray
    elapsed_seconds: float = 0.0
    per_solve_seconds: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))


def _constraints_for(problem: BiObjectiveProblem) -> list[dict]:
    return [{"type": "ineq", "fun": g} for g in problem.nonlinear_constraints]


def _minimize(
    fun,
    x0: np.ndarray,
    problem: BiObjectiveProblem,
    *,
    extra_constraints: list | None = None,
) -> tuple[np.ndarray, float, bool, str, float]:
    cons = list(_constraints_for(problem))
    if extra_constraints:
        cons.extend(extra_constraints)
    t0 = time.perf_counter()
    res = minimize(
        fun,
        np.asarray(x0, dtype=float),
        method="SLSQP",
        bounds=problem.bounds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 400, "disp": False},
    )
    elapsed = time.perf_counter() - t0
    u = np.asarray(res.x, dtype=float)
    return u, float(res.fun), bool(res.success), str(res.message), elapsed


def weighted_norm(y: np.ndarray, lam: np.ndarray, p: float) -> float:
    """Weighted ℓ_p norm ‖y‖_{λ,p}; p≥1 or p=∞ (Chebyshev)."""
    y = np.asarray(y, dtype=float).reshape(-1)
    lam = np.asarray(lam, dtype=float).reshape(-1)
    if y.shape != lam.shape:
        raise ValueError("y and λ must have the same shape")
    if np.any(lam <= 0):
        raise ValueError("λ must be componentwise positive")
    abs_y = np.abs(y)
    if np.isinf(p):
        return float(np.max(lam * abs_y))
    if p < 1:
        raise ValueError("p must be ≥ 1 or ∞")
    return float(np.sum(lam * abs_y**p) ** (1.0 / p))


def solve_weighted_sum(
    lam: np.ndarray,
    *,
    problem: BiObjectiveProblem | None = None,
    x0: np.ndarray | None = None,
) -> ScalarizationSolution:
    """S1: minimize Σ λᵢ Fᵢ(u), λ>0."""
    problem = problem or DEFAULT_PROBLEM
    lam = np.asarray(lam, dtype=float).reshape(-1)
    if lam.size != 2 or np.any(lam <= 0):
        raise ValueError("λ must be a positive 2-vector")

    def objective(u: np.ndarray) -> float:
        return float(np.dot(lam, problem.evaluate(u)))

    u, sval, ok, msg, elapsed = _minimize(
        objective, x0 if x0 is not None else problem.default_x0(), problem
    )
    return ScalarizationSolution(
        method="weighted_sum",
        u=u,
        f=problem.evaluate(u),
        scalar_value=sval,
        success=ok,
        message=msg,
        params={"lambda": lam.copy()},
        elapsed_seconds=elapsed,
    )


def solve_epsilon_constraint(
    a: float,
    *,
    primary: int = 0,
    problem: BiObjectiveProblem | None = None,
    x0: np.ndarray | None = None,
) -> ScalarizationSolution:
    """S2: minimize Fⱼ s.t. Fᵢ(u) ≤ a for i≠j."""
    problem = problem or DEFAULT_PROBLEM
    if primary not in (0, 1):
        raise ValueError("primary must be 0 (F1) or 1 (F2)")
    constrained = 1 - primary

    def objective(u: np.ndarray) -> float:
        return float(problem.evaluate(u)[primary])

    def cons_ineq(u: np.ndarray) -> float:
        return float(a - problem.evaluate(u)[constrained])

    u, sval, ok, msg, elapsed = _minimize(
        objective,
        x0 if x0 is not None else problem.default_x0(),
        problem,
        extra_constraints=[{"type": "ineq", "fun": cons_ineq}],
    )
    return ScalarizationSolution(
        method="epsilon_constraint",
        u=u,
        f=problem.evaluate(u),
        scalar_value=sval,
        success=ok,
        message=msg,
        params={"a": float(a), "primary": primary, "constrained": constrained},
        elapsed_seconds=elapsed,
    )


def solve_distance(
    ref: np.ndarray,
    lam: np.ndarray,
    p: float,
    *,
    problem: BiObjectiveProblem | None = None,
    x0: np.ndarray | None = None,
) -> ScalarizationSolution:
    """S3: minimize ‖F(u)−x‖_{λ,p} to dominating reference x."""
    problem = problem or DEFAULT_PROBLEM
    ref = np.asarray(ref, dtype=float).reshape(-1)
    lam = np.asarray(lam, dtype=float).reshape(-1)
    if ref.size != 2 or lam.size != 2:
        raise ValueError("ref and λ must be 2-vectors")
    if np.any(lam <= 0):
        raise ValueError("λ must be componentwise positive")

    def objective(u: np.ndarray) -> float:
        return weighted_norm(problem.evaluate(u) - ref, lam, p)

    u, sval, ok, msg, elapsed = _minimize(
        objective, x0 if x0 is not None else problem.default_x0(), problem
    )
    return ScalarizationSolution(
        method="distance",
        u=u,
        f=problem.evaluate(u),
        scalar_value=sval,
        success=ok,
        message=msg,
        params={"ref": ref.copy(), "lambda": lam.copy(), "p": float(p)},
        elapsed_seconds=elapsed,
    )


def _normalize_weights(t: float) -> np.ndarray:
    eps = 1e-6
    t = float(np.clip(t, eps, 1.0 - eps))
    return np.array([t, 1.0 - t], dtype=float)


def sweep_weighted_sum(
    n: int = 25,
    *,
    problem: BiObjectiveProblem | None = None,
) -> FrontApproximation:
    problem = problem or DEFAULT_PROBLEM
    if n < 2:
        raise ValueError("n must be ≥ 2")
    ts = np.linspace(0.0, 1.0, n)
    decisions, objectives, params, oks, times = [], [], [], [], []
    x0 = problem.default_x0()
    t_all = time.perf_counter()
    for t in ts:
        lam = _normalize_weights(float(t))
        sol = solve_weighted_sum(lam, problem=problem, x0=x0)
        if sol.success:
            x0 = sol.u.copy()
        decisions.append(sol.u)
        objectives.append(sol.f)
        params.append(sol.params)
        oks.append(sol.success)
        times.append(sol.elapsed_seconds)
    return FrontApproximation(
        method="weighted_sum",
        decisions=np.asarray(decisions, dtype=float),
        objectives=np.asarray(objectives, dtype=float),
        params=params,
        successes=np.asarray(oks, dtype=bool),
        elapsed_seconds=time.perf_counter() - t_all,
        per_solve_seconds=np.asarray(times, dtype=float),
    )


def sweep_epsilon_constraint(
    n: int = 25,
    *,
    primary: int = 0,
    a_min: float | None = None,
    a_max: float | None = None,
    problem: BiObjectiveProblem | None = None,
) -> FrontApproximation:
    from owd.scalarization.problem import estimate_objective_bounds

    problem = problem or DEFAULT_PROBLEM
    if n < 2:
        raise ValueError("n must be ≥ 2")
    f_min, f_max = estimate_objective_bounds(problem)
    constrained = 1 - primary
    lo = f_min[constrained] if a_min is None else float(a_min)
    hi = f_max[constrained] if a_max is None else float(a_max)
    if hi <= lo:
        raise ValueError("a_max must be > a_min")
    alphas = np.linspace(lo, hi, n)
    decisions, objectives, params, oks, times = [], [], [], [], []
    x0 = problem.default_x0()
    t_all = time.perf_counter()
    for a in alphas:
        sol = solve_epsilon_constraint(float(a), primary=primary, problem=problem, x0=x0)
        if sol.success:
            x0 = sol.u.copy()
        decisions.append(sol.u)
        objectives.append(sol.f)
        params.append(sol.params)
        oks.append(sol.success)
        times.append(sol.elapsed_seconds)
    return FrontApproximation(
        method="epsilon_constraint",
        decisions=np.asarray(decisions, dtype=float),
        objectives=np.asarray(objectives, dtype=float),
        params=params,
        successes=np.asarray(oks, dtype=bool),
        elapsed_seconds=time.perf_counter() - t_all,
        per_solve_seconds=np.asarray(times, dtype=float),
    )


def sweep_distance(
    n: int = 25,
    *,
    ref: np.ndarray | None = None,
    p: float = 2.0,
    problem: BiObjectiveProblem | None = None,
) -> FrontApproximation:
    from owd.scalarization.problem import estimate_objective_bounds

    problem = problem or DEFAULT_PROBLEM
    if n < 2:
        raise ValueError("n must be ≥ 2")
    if ref is None:
        f_min, _ = estimate_objective_bounds(problem)
        ref = f_min - 0.05 * np.abs(f_min) - 0.05
    ref = np.asarray(ref, dtype=float)
    ts = np.linspace(0.0, 1.0, n)
    decisions, objectives, params, oks, times = [], [], [], [], []
    x0 = problem.default_x0()
    t_all = time.perf_counter()
    for t in ts:
        lam = _normalize_weights(float(t))
        sol = solve_distance(ref, lam, p, problem=problem, x0=x0)
        if sol.success:
            x0 = sol.u.copy()
        decisions.append(sol.u)
        objectives.append(sol.f)
        params.append(sol.params)
        oks.append(sol.success)
        times.append(sol.elapsed_seconds)
    return FrontApproximation(
        method="distance",
        decisions=np.asarray(decisions, dtype=float),
        objectives=np.asarray(objectives, dtype=float),
        params=params,
        successes=np.asarray(oks, dtype=bool),
        elapsed_seconds=time.perf_counter() - t_all,
        per_solve_seconds=np.asarray(times, dtype=float),
    )


def sort_front_by_f1(front: FrontApproximation) -> FrontApproximation:
    """Sort successful points by F1 for spline / polyline plotting."""
    mask = front.successes
    if not np.any(mask):
        return front
    obj = front.objectives[mask]
    dec = front.decisions[mask]
    times = (
        front.per_solve_seconds[mask]
        if len(front.per_solve_seconds) == len(front.successes)
        else front.per_solve_seconds
    )
    order = np.argsort(obj[:, 0], kind="mergesort")
    params = [front.params[i] for i, ok in enumerate(front.successes) if ok]
    params = [params[i] for i in order]
    return FrontApproximation(
        method=front.method,
        decisions=dec[order],
        objectives=obj[order],
        params=params,
        successes=np.ones(len(order), dtype=bool),
        elapsed_seconds=front.elapsed_seconds,
        per_solve_seconds=np.asarray(times, dtype=float)[order] if len(times) else times,
    )
