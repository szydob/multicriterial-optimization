from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize

from owd.scalarization.problem import BOUNDS, DEFAULT_PROBLEM, LB, UB, BiObjectiveProblem

MethodName = Literal["weighted_sum", "epsilon_constraint", "distance"]


@dataclass
class ScalarizationSolution:
    """Single scalarized solve result."""

    method: MethodName
    u: np.ndarray  # decision (x, y)
    f: np.ndarray  # F(u)
    scalar_value: float
    success: bool
    message: str
    params: dict


@dataclass
class FrontApproximation:
    """Discrete approximation of P(U,F) / FP(U) from a parameter sweep."""

    method: MethodName
    decisions: np.ndarray  # (n, 2)
    objectives: np.ndarray  # (n, 2)
    params: list[dict]
    successes: np.ndarray


def _default_x0() -> np.ndarray:
    return 0.5 * (LB + UB)


def _minimize(
    fun,
    x0: np.ndarray,
    *,
    constraints: list | None = None,
) -> tuple[np.ndarray, float, bool, str]:
    res = minimize(
        fun,
        np.asarray(x0, dtype=float),
        method="SLSQP",
        bounds=BOUNDS,
        constraints=constraints or (),
        options={"ftol": 1e-10, "maxiter": 400, "disp": False},
    )
    u = np.asarray(res.x, dtype=float)
    return u, float(res.fun), bool(res.success), str(res.message)


def weighted_norm(y: np.ndarray, lam: np.ndarray, p: float) -> float:
    """
    Weighted ℓ_p norm ‖y‖_{λ,p} = d_{p,λ}(0,y) from the lab PDF.

    p ≥ 1, or p = ∞ (Chebyshev): max_i λ_i |y_i|.
    """
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
    """
    S1(v, λ) = Σ λ_i v_i  — linear (weighted-sum) scalarization.

    Minimize Σ λ_i F_i(u) over u ∈ U, λ > 0.
    """
    problem = problem or DEFAULT_PROBLEM
    lam = np.asarray(lam, dtype=float).reshape(-1)
    if lam.size != 2 or np.any(lam <= 0):
        raise ValueError("λ must be a positive 2-vector")

    def objective(u: np.ndarray) -> float:
        return float(np.dot(lam, problem.evaluate(u)))

    u, sval, ok, msg = _minimize(objective, x0 if x0 is not None else _default_x0())
    return ScalarizationSolution(
        method="weighted_sum",
        u=u,
        f=problem.evaluate(u),
        scalar_value=sval,
        success=ok,
        message=msg,
        params={"lambda": lam.copy()},
    )


def solve_epsilon_constraint(
    a: float,
    *,
    primary: int = 0,
    problem: BiObjectiveProblem | None = None,
    x0: np.ndarray | None = None,
) -> ScalarizationSolution:
    """
    S2(u, a, j) = F_j(u) s.t. F_i(u) ≤ a_i for i ≠ j  (ε-constraint).

    For N=2: minimize F_j subject to F_{1-j}(u) ≤ a.
    """
    problem = problem or DEFAULT_PROBLEM
    if primary not in (0, 1):
        raise ValueError("primary must be 0 (F1) or 1 (F2)")
    constrained = 1 - primary

    def objective(u: np.ndarray) -> float:
        return float(problem.evaluate(u)[primary])

    def cons_ineq(u: np.ndarray) -> float:
        # scipy expects cons ≥ 0  →  a - F_i(u) ≥ 0
        return float(a - problem.evaluate(u)[constrained])

    constraints = [{"type": "ineq", "fun": cons_ineq}]
    u, sval, ok, msg = _minimize(
        objective,
        x0 if x0 is not None else _default_x0(),
        constraints=constraints,
    )
    return ScalarizationSolution(
        method="epsilon_constraint",
        u=u,
        f=problem.evaluate(u),
        scalar_value=sval,
        success=ok,
        message=msg,
        params={"a": float(a), "primary": primary, "constrained": constrained},
    )


def solve_distance(
    ref: np.ndarray,
    lam: np.ndarray,
    p: float,
    *,
    problem: BiObjectiveProblem | None = None,
    x0: np.ndarray | None = None,
) -> ScalarizationSolution:
    """
    S3(u, x, λ, p) = ‖F(u) − x‖_{λ,p} — distance to a dominating reference point.

    Minimize the weighted ℓ_p distance from F(u) to reference x (often utopia).
    """
    problem = problem or DEFAULT_PROBLEM
    ref = np.asarray(ref, dtype=float).reshape(-1)
    lam = np.asarray(lam, dtype=float).reshape(-1)
    if ref.size != 2 or lam.size != 2:
        raise ValueError("ref and λ must be 2-vectors")
    if np.any(lam <= 0):
        raise ValueError("λ must be componentwise positive")

    def objective(u: np.ndarray) -> float:
        return weighted_norm(problem.evaluate(u) - ref, lam, p)

    u, sval, ok, msg = _minimize(objective, x0 if x0 is not None else _default_x0())
    return ScalarizationSolution(
        method="distance",
        u=u,
        f=problem.evaluate(u),
        scalar_value=sval,
        success=ok,
        message=msg,
        params={"ref": ref.copy(), "lambda": lam.copy(), "p": float(p)},
    )


def _normalize_weights(t: float) -> np.ndarray:
    """Map t∈[0,1] to positive λ with λ1+λ2=1 (endpoints nudged away from 0)."""
    eps = 1e-6
    t = float(np.clip(t, eps, 1.0 - eps))
    return np.array([t, 1.0 - t], dtype=float)


def sweep_weighted_sum(
    n: int = 25,
    *,
    problem: BiObjectiveProblem | None = None,
) -> FrontApproximation:
    """Sweep λ with constant step ρ = 1/(n−1) over the simplex λ1+λ2=1, λ>0."""
    problem = problem or DEFAULT_PROBLEM
    if n < 2:
        raise ValueError("n must be ≥ 2")
    ts = np.linspace(0.0, 1.0, n)
    decisions, objectives, params, oks = [], [], [], []
    x0 = _default_x0()
    for t in ts:
        lam = _normalize_weights(float(t))
        sol = solve_weighted_sum(lam, problem=problem, x0=x0)
        if sol.success:
            x0 = sol.u.copy()
        decisions.append(sol.u)
        objectives.append(sol.f)
        params.append(sol.params)
        oks.append(sol.success)
    return FrontApproximation(
        method="weighted_sum",
        decisions=np.asarray(decisions, dtype=float),
        objectives=np.asarray(objectives, dtype=float),
        params=params,
        successes=np.asarray(oks, dtype=bool),
    )


def sweep_epsilon_constraint(
    n: int = 25,
    *,
    primary: int = 0,
    a_min: float | None = None,
    a_max: float | None = None,
    problem: BiObjectiveProblem | None = None,
) -> FrontApproximation:
    """Sweep ε (=a) with constant step ρ over [a_min, a_max] for the constrained criterion."""
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
    decisions, objectives, params, oks = [], [], [], []
    x0 = _default_x0()
    for a in alphas:
        sol = solve_epsilon_constraint(float(a), primary=primary, problem=problem, x0=x0)
        if sol.success:
            x0 = sol.u.copy()
        decisions.append(sol.u)
        objectives.append(sol.f)
        params.append(sol.params)
        oks.append(sol.success)
    return FrontApproximation(
        method="epsilon_constraint",
        decisions=np.asarray(decisions, dtype=float),
        objectives=np.asarray(objectives, dtype=float),
        params=params,
        successes=np.asarray(oks, dtype=bool),
    )


def sweep_distance(
    n: int = 25,
    *,
    ref: np.ndarray | None = None,
    p: float = 2.0,
    problem: BiObjectiveProblem | None = None,
) -> FrontApproximation:
    """Sweep λ on the simplex for fixed reference point and p (incl. ∞)."""
    from owd.scalarization.problem import estimate_objective_bounds

    problem = problem or DEFAULT_PROBLEM
    if n < 2:
        raise ValueError("n must be ≥ 2")
    if ref is None:
        f_min, _ = estimate_objective_bounds(problem)
        # Dominating / utopia-like point slightly better than grid ideal.
        ref = f_min - 0.05 * np.abs(f_min) - 0.05
    ref = np.asarray(ref, dtype=float)
    ts = np.linspace(0.0, 1.0, n)
    decisions, objectives, params, oks = [], [], [], []
    x0 = _default_x0()
    for t in ts:
        lam = _normalize_weights(float(t))
        sol = solve_distance(ref, lam, p, problem=problem, x0=x0)
        if sol.success:
            x0 = sol.u.copy()
        decisions.append(sol.u)
        objectives.append(sol.f)
        params.append(sol.params)
        oks.append(sol.success)
    return FrontApproximation(
        method="distance",
        decisions=np.asarray(decisions, dtype=float),
        objectives=np.asarray(objectives, dtype=float),
        params=params,
        successes=np.asarray(oks, dtype=bool),
    )


def sort_front_by_f1(front: FrontApproximation) -> FrontApproximation:
    """Sort successful points by F1 for spline / polyline plotting."""
    mask = front.successes
    if not np.any(mask):
        return front
    obj = front.objectives[mask]
    dec = front.decisions[mask]
    order = np.argsort(obj[:, 0], kind="mergesort")
    params = [front.params[i] for i, ok in enumerate(front.successes) if ok]
    params = [params[i] for i in order]
    return FrontApproximation(
        method=front.method,
        decisions=dec[order],
        objectives=obj[order],
        params=params,
        successes=np.ones(len(order), dtype=bool),
    )
