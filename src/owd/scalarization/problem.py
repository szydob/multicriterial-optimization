"""Cw3 problem: bi-objective polynomials on the square U=[1,4]²."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BiObjectiveProblem:
    """Vector criterion F:U→R² to be minimized (natural order)."""

    id: str
    name: str
    f1_latex: str
    f2_latex: str
    lb: np.ndarray
    ub: np.ndarray
    nonlinear_constraints: tuple[Callable[[np.ndarray], float], ...] = ()

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(float(self.lb[0]), float(self.ub[0])), (float(self.lb[1]), float(self.ub[1]))]

    def f1(self, x: float, y: float) -> float:
        raise NotImplementedError

    def f2(self, x: float, y: float) -> float:
        raise NotImplementedError

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float).reshape(-1)
        if u.size != 2:
            raise ValueError("decision vector must be 2-dimensional")
        return np.array([self.f1(float(u[0]), float(u[1])), self.f2(float(u[0]), float(u[1]))])

    def evaluate_many(self, U: np.ndarray) -> np.ndarray:
        U = np.asarray(U, dtype=float)
        if U.ndim != 2 or U.shape[1] != 2:
            raise ValueError("U must have shape (n, 2)")
        out = np.empty((U.shape[0], 2), dtype=float)
        for i, row in enumerate(U):
            out[i] = self.evaluate(row)
        return out

    def is_feasible(self, u: np.ndarray, tol: float = 1e-8) -> bool:
        u = np.asarray(u, dtype=float).reshape(-1)
        if np.any(u < self.lb - tol) or np.any(u > self.ub + tol):
            return False
        for g in self.nonlinear_constraints:
            if float(g(u)) < -tol:
                return False
        return True

    def default_x0(self) -> np.ndarray:
        return 0.5 * (self.lb + self.ub)

    def start_points(self) -> list[np.ndarray]:
        """Diverse SLSQP starts (corners + center + mid-edges)."""
        lo, hi = self.lb, self.ub
        mid = 0.5 * (lo + hi)
        return [
            mid.copy(),
            np.array([lo[0], lo[1]]),
            np.array([lo[0], hi[1]]),
            np.array([hi[0], lo[1]]),
            np.array([hi[0], hi[1]]),
            np.array([mid[0], lo[1]]),
            np.array([mid[0], hi[1]]),
            np.array([lo[0], mid[1]]),
            np.array([hi[0], mid[1]]),
        ]


class Square14Problem(BiObjectiveProblem):
    """
    Competing quadratic objectives on U=[1,4]² (Ćw. 3).

    Ideal corners pull to opposite ends of the square, so the Pareto set is
    a clear curve from near (1,1) toward (4,4):

        F1(x,y) = (x − 1)² + (y − 1)²
        F2(x,y) = (x − 4)² + (y − 4)²
    """

    def __init__(self) -> None:
        super().__init__(
            id="square_1_4",
            name="Kwadrat U=[1,4]²",
            f1_latex=r"F_1(x,y)=(x-1)^2+(y-1)^2",
            f2_latex=r"F_2(x,y)=(x-4)^2+(y-4)^2",
            lb=np.array([1.0, 1.0]),
            ub=np.array([4.0, 4.0]),
        )

    def f1(self, x: float, y: float) -> float:
        return (x - 1.0) ** 2 + (y - 1.0) ** 2

    def f2(self, x: float, y: float) -> float:
        return (x - 4.0) ** 2 + (y - 4.0) ** 2


DEFAULT_PROBLEM = Square14Problem()
INSTANCES = {"square_1_4": DEFAULT_PROBLEM}
LB = DEFAULT_PROBLEM.lb
UB = DEFAULT_PROBLEM.ub
BOUNDS = DEFAULT_PROBLEM.bounds
Cw3PolynomialProblem = Square14Problem
InstanceId = str


def get_problem(instance_id: str = "square_1_4") -> BiObjectiveProblem:
    if instance_id not in INSTANCES:
        # Only one supported instance; map legacy IDs to it.
        return DEFAULT_PROBLEM
    return INSTANCES[instance_id]


def sample_feasible_grid(
    problem: BiObjectiveProblem | None = None,
    n_per_axis: int = 40,
) -> np.ndarray:
    problem = problem or DEFAULT_PROBLEM
    xs = np.linspace(problem.lb[0], problem.ub[0], n_per_axis)
    ys = np.linspace(problem.lb[1], problem.ub[1], n_per_axis)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    mask = np.array([problem.is_feasible(p) for p in pts], dtype=bool)
    return pts[mask]


def estimate_objective_bounds(
    problem: BiObjectiveProblem | None = None,
    n_per_axis: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    problem = problem or DEFAULT_PROBLEM
    grid = sample_feasible_grid(problem, n_per_axis)
    if len(grid) == 0:
        raise RuntimeError("empty feasible grid")
    vals = problem.evaluate_many(grid)
    return vals.min(axis=0), vals.max(axis=0)
