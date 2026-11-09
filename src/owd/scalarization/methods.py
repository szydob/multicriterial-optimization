"""Three Cw3 scalarization methods + Pareto sweeps that return n distinct points."""

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
    method: MethodName
    decisions: np.ndarray
    objectives: np.ndarray
    params: list[dict]
    successes: np.ndarray
    elapsed_seconds: float = 0.0
    per_solve_seconds: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))


def _constraints_for(problem: BiObjectiveProblem) -> list[dict]:
    return [{"type": "ineq", "fun": g} for g in problem.nonlinear_constraints]


def _minimize_once(
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
        options={"ftol": 1e-12, "maxiter": 500, "disp": False},
    )
    elapsed = time.perf_counter() - t0
    u = np.asarray(res.x, dtype=float)
    # Prefer feasible evaluation even if solver flagged failure but stayed in bounds.
    ok = bool(res.success) or problem.is_feasible(u, tol=1e-6)
    return u, float(res.fun), ok, str(res.message), elapsed


def _minimize_multistart(
    fun,
    problem: BiObjectiveProblem,
    *,
    extra_constraints: list | None = None,
    prefer: np.ndarray | None = None,
) -> tuple[np.ndarray, float, bool, str, float]:
    starts = problem.start_points()
    if prefer is not None:
        starts = [np.asarray(prefer, dtype=float), *starts]
    best: tuple[np.ndarray, float, bool, str, float] | None = None
    total_t = 0.0
    last_msg = ""
    for x0 in starts:
        u, sval, ok, msg, elapsed = _minimize_once(
            fun, x0, problem, extra_constraints=extra_constraints
        )
        total_t += elapsed
        last_msg = msg
        if not ok:
            continue
        if best is None or sval < best[1]:
            best = (u, sval, True, msg, total_t)
    if best is None:
        # Fall back to default start even if marked unsuccessful.
        u, sval, ok, msg, elapsed = _minimize_once(
            fun, problem.default_x0(), problem, extra_constraints=extra_constraints
        )
        return u, sval, ok, msg, total_t + elapsed
    return best[0], best[1], best[2], best[3], total_t


def weighted_norm(y: np.ndarray, lam: np.ndarray, p: float) -> float:
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
    problem = problem or DEFAULT_PROBLEM
    lam = np.asarray(lam, dtype=float).reshape(-1)
    if lam.size != 2 or np.any(lam <= 0):
        raise ValueError("λ must be a positive 2-vector")

    def objective(u: np.ndarray) -> float:
        return float(np.dot(lam, problem.evaluate(u)))

    u, sval, ok, msg, elapsed = _minimize_multistart(
        objective, problem, prefer=x0
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
    problem = problem or DEFAULT_PROBLEM
    if primary not in (0, 1):
        raise ValueError("primary must be 0 (F1) or 1 (F2)")
    constrained = 1 - primary

    def objective(u: np.ndarray) -> float:
        return float(problem.evaluate(u)[primary])

    def cons_ineq(u: np.ndarray) -> float:
        return float(a - problem.evaluate(u)[constrained])

    u, sval, ok, msg, elapsed = _minimize_multistart(
        objective,
        problem,
        prefer=x0,
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
    problem = problem or DEFAULT_PROBLEM
    ref = np.asarray(ref, dtype=float).reshape(-1)
    lam = np.asarray(lam, dtype=float).reshape(-1)
    if ref.size != 2 or lam.size != 2:
        raise ValueError("ref and λ must be 2-vectors")
    if np.any(lam <= 0):
        raise ValueError("λ must be componentwise positive")

    def objective(u: np.ndarray) -> float:
        return weighted_norm(problem.evaluate(u) - ref, lam, p)

    u, sval, ok, msg, elapsed = _minimize_multistart(
        objective, problem, prefer=x0
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
    # Keep endpoints slightly inside (0,1) so both weights stay strictly positive.
    eps = 1e-3
    t = float(np.clip(t, eps, 1.0 - eps))
    return np.array([t, 1.0 - t], dtype=float)


def _is_dominated(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(b <= a) and np.any(b < a))


def _nondominated_mask(obj: np.ndarray) -> np.ndarray:
    n = len(obj)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if _is_dominated(obj[i], obj[j]):
                keep[i] = False
                break
    return keep


def _dedupe_front(
    decisions: np.ndarray,
    objectives: np.ndarray,
    params: list[dict],
    times: np.ndarray,
    *,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray]:
    """Drop near-duplicate objective points (keep first along F1 order)."""
    if len(objectives) == 0:
        return decisions, objectives, params, times
    order = np.argsort(objectives[:, 0], kind="mergesort")
    decisions = decisions[order]
    objectives = objectives[order]
    times = times[order]
    params = [params[i] for i in order]
    keep: list[int] = []
    for i, f in enumerate(objectives):
        if not keep:
            keep.append(i)
            continue
        prev = objectives[keep[-1]]
        if np.linalg.norm(f - prev) > tol:
            keep.append(i)
    idx = np.asarray(keep, dtype=int)
    return decisions[idx], objectives[idx], [params[i] for i in idx], times[idx]


def _subsample_evenly(
    decisions: np.ndarray,
    objectives: np.ndarray,
    params: list[dict],
    times: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray]:
    if len(objectives) <= n:
        return decisions, objectives, params, times
    # Even indices along sorted F1.
    idx = np.linspace(0, len(objectives) - 1, n)
    idx = np.unique(np.round(idx).astype(int))
    # If unique rounding collapsed, fall back to slice
    if len(idx) < n:
        idx = np.linspace(0, len(objectives) - 1, n).astype(int)
    return (
        decisions[idx],
        objectives[idx],
        [params[i] for i in idx],
        times[idx],
    )


def _finalize_front(
    method: MethodName,
    decisions: list[np.ndarray],
    objectives: list[np.ndarray],
    params: list[dict],
    oks: list[bool],
    times: list[float],
    elapsed: float,
    n_target: int,
) -> FrontApproximation:
    dec = np.asarray(decisions, dtype=float)
    obj = np.asarray(objectives, dtype=float)
    tim = np.asarray(times, dtype=float)
    ok = np.asarray(oks, dtype=bool)
    if np.any(ok):
        dec, obj, params, tim = dec[ok], obj[ok], [params[i] for i, k in enumerate(ok) if k], tim[ok]
    nd = _nondominated_mask(obj) if len(obj) else np.array([], dtype=bool)
    if len(nd):
        dec, obj, params, tim = dec[nd], obj[nd], [params[i] for i, k in enumerate(nd) if k], tim[nd]
    dec, obj, params, tim = _dedupe_front(dec, obj, params, tim)
    dec, obj, params, tim = _subsample_evenly(dec, obj, params, tim, n_target)
    return FrontApproximation(
        method=method,
        decisions=dec,
        objectives=obj,
        params=params,
        successes=np.ones(len(dec), dtype=bool),
        elapsed_seconds=elapsed,
        per_solve_seconds=tim,
    )


def _anchor_points(problem: BiObjectiveProblem) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Individual minima of F1 and F2 (anchors for ε range / utopia)."""
    sol1 = solve_weighted_sum(np.array([1.0 - 1e-3, 1e-3]), problem=problem)
    sol2 = solve_weighted_sum(np.array([1e-3, 1.0 - 1e-3]), problem=problem)
    return sol1.u, sol1.f, sol2.u, sol2.f


def sweep_weighted_sum(
    n: int = 25,
    *,
    problem: BiObjectiveProblem | None = None,
) -> FrontApproximation:
    problem = problem or DEFAULT_PROBLEM
    if n < 2:
        raise ValueError("n must be ≥ 2")
    # Oversample then subsample to n distinct nondominated points.
    m = max(n, min(5 * n, 120))
    ts = np.linspace(0.0, 1.0, m)
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
    return _finalize_front(
        "weighted_sum",
        decisions,
        objectives,
        params,
        oks,
        times,
        time.perf_counter() - t_all,
        n,
    )


def sweep_epsilon_constraint(
    n: int = 25,
    *,
    primary: int = 0,
    a_min: float | None = None,
    a_max: float | None = None,
    problem: BiObjectiveProblem | None = None,
) -> FrontApproximation:
    problem = problem or DEFAULT_PROBLEM
    if n < 2:
        raise ValueError("n must be ≥ 2")
    # Pareto-relevant ε range: between F_constrained at the two anchors.
    _, f_at_min_primary, _, f_at_min_other = _anchor_points(problem)
    constrained = 1 - primary
    # When minimizing F_primary, ε on the other objective runs from its value at
    # the primary-optimum (loose) down/up to its individual minimum (tight).
    lo_auto = float(min(f_at_min_primary[constrained], f_at_min_other[constrained]))
    hi_auto = float(max(f_at_min_primary[constrained], f_at_min_other[constrained]))
    # Pad slightly so endpoints are attainable.
    pad = 0.02 * max(hi_auto - lo_auto, 1.0)
    lo = lo_auto + pad if a_min is None else float(a_min)
    hi = hi_auto - pad if a_max is None else float(a_max)
    if hi <= lo:
        lo, hi = lo_auto, hi_auto
    m = max(n, min(5 * n, 120))
    alphas = np.linspace(lo, hi, m)
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
    return _finalize_front(
        "epsilon_constraint",
        decisions,
        objectives,
        params,
        oks,
        times,
        time.perf_counter() - t_all,
        n,
    )


def sweep_distance(
    n: int = 25,
    *,
    ref: np.ndarray | None = None,
    p: float = 2.0,
    problem: BiObjectiveProblem | None = None,
) -> FrontApproximation:
    problem = problem or DEFAULT_PROBLEM
    if n < 2:
        raise ValueError("n must be ≥ 2")
    _, f1, _, f2 = _anchor_points(problem)
    if ref is None:
        # Utopia: componentwise better than both anchors.
        ref = np.minimum(f1, f2) - 0.05
    ref = np.asarray(ref, dtype=float)
    m = max(n, min(5 * n, 120))
    ts = np.linspace(0.0, 1.0, m)
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
    return _finalize_front(
        "distance",
        decisions,
        objectives,
        params,
        oks,
        times,
        time.perf_counter() - t_all,
        n,
    )


def sort_front_by_f1(front: FrontApproximation) -> FrontApproximation:
    """Sort points by F1 (already mostly sorted; keep API stable)."""
    if len(front.objectives) == 0:
        return front
    order = np.argsort(front.objectives[:, 0], kind="mergesort")
    return FrontApproximation(
        method=front.method,
        decisions=front.decisions[order],
        objectives=front.objectives[order],
        params=[front.params[i] for i in order],
        successes=front.successes[order] if len(front.successes) == len(order) else np.ones(len(order), dtype=bool),
        elapsed_seconds=front.elapsed_seconds,
        per_solve_seconds=(
            front.per_solve_seconds[order]
            if len(front.per_solve_seconds) == len(front.decisions)
            else front.per_solve_seconds
        ),
    )
