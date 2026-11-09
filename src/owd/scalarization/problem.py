"""Cw3 problem instances: bi-objective polynomials on lab-style feasible sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

InstanceId = Literal["square_1_4", "square_m1_1", "unit_disk"]


@dataclass(frozen=True)
class BiObjectiveProblem:
    """Vector criterion F:U→R² to be minimized (natural order)."""

    id: InstanceId
    name: str
    f1_latex: str
    f2_latex: str
    lb: np.ndarray
    ub: np.ndarray
    # Optional nonlinear inequality constraints g(u) >= 0 (SciPy style).
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


class Square14Problem(BiObjectiveProblem):
    """User square U=[1,4]² with competing nonlinear polynomials."""

    def __init__(self) -> None:
        super().__init__(
            id="square_1_4",
            name="Kwadrat U=[1,4]²",
            f1_latex=r"F_1=(x-1)^2+(y-2)^2+0.25\,xy",
            f2_latex=r"F_2=(x-4)^2+(y-3)^2+0.15\,x^2",
            lb=np.array([1.0, 1.0]),
            ub=np.array([4.0, 4.0]),
        )

    def f1(self, x: float, y: float) -> float:
        return (x - 1.0) ** 2 + (y - 2.0) ** 2 + 0.25 * x * y

    def f2(self, x: float, y: float) -> float:
        return (x - 4.0) ** 2 + (y - 3.0) ** 2 + 0.15 * x**2


class SquareM11Problem(BiObjectiveProblem):
    """Lab-PDF style square U=[-1,1]² (instrukcja Cw3)."""

    def __init__(self) -> None:
        super().__init__(
            id="square_m1_1",
            name="Kwadrat U=[-1,1]² (PDF Cw3)",
            f1_latex=r"F_1=(x+0.5)^2+(y-0.4)^2+0.2\,xy",
            f2_latex=r"F_2=(x-0.6)^2+(y+0.5)^2+0.15\,y^2",
            lb=np.array([-1.0, -1.0]),
            ub=np.array([1.0, 1.0]),
        )

    def f1(self, x: float, y: float) -> float:
        return (x + 0.5) ** 2 + (y - 0.4) ** 2 + 0.2 * x * y

    def f2(self, x: float, y: float) -> float:
        return (x - 0.6) ** 2 + (y + 0.5) ** 2 + 0.15 * y**2


class UnitDiskProblem(BiObjectiveProblem):
    """Lab-PDF style disk U={(x,y): x²+y²≤1}."""

    def __init__(self) -> None:
        def g_disk(u: np.ndarray) -> float:
            return float(1.0 - u[0] ** 2 - u[1] ** 2)

        super().__init__(
            id="unit_disk",
            name="Koło jednostkowe x²+y²≤1 (PDF Cw3)",
            f1_latex=r"F_1=(x+0.4)^2+(y-0.3)^2",
            f2_latex=r"F_2=(x-0.5)^2+(y+0.4)^2+0.1\,x^2",
            lb=np.array([-1.0, -1.0]),
            ub=np.array([1.0, 1.0]),
            nonlinear_constraints=(g_disk,),
        )

    def f1(self, x: float, y: float) -> float:
        return (x + 0.4) ** 2 + (y - 0.3) ** 2

    def f2(self, x: float, y: float) -> float:
        return (x - 0.5) ** 2 + (y + 0.4) ** 2 + 0.1 * x**2


INSTANCES: dict[InstanceId, BiObjectiveProblem] = {
    "square_1_4": Square14Problem(),
    "square_m1_1": SquareM11Problem(),
    "unit_disk": UnitDiskProblem(),
}

# Back-compat aliases used by earlier Cw3 code.
DEFAULT_PROBLEM = INSTANCES["square_1_4"]
LB = DEFAULT_PROBLEM.lb
UB = DEFAULT_PROBLEM.ub
BOUNDS = DEFAULT_PROBLEM.bounds
Cw3PolynomialProblem = Square14Problem


def get_problem(instance_id: InstanceId | str = "square_1_4") -> BiObjectiveProblem:
    if instance_id not in INSTANCES:
        raise KeyError(f"unknown instance: {instance_id}")
    return INSTANCES[instance_id]  # type: ignore[index]


def sample_feasible_grid(
    problem: BiObjectiveProblem | None = None,
    n_per_axis: int = 40,
) -> np.ndarray:
    """Regular grid over the bounding box, filtered to feasible U."""
    problem = problem or DEFAULT_PROBLEM
    xs = np.linspace(problem.lb[0], problem.ub[0], n_per_axis)
    ys = np.linspace(problem.lb[1], problem.ub[1], n_per_axis)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    mask = np.array([problem.is_feasible(p) for p in pts], dtype=bool)
    return pts[mask]


def estimate_objective_bounds(
    problem: BiObjectiveProblem | None = None,
    n_per_axis: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    problem = problem or DEFAULT_PROBLEM
    grid = sample_feasible_grid(problem, n_per_axis)
    if len(grid) == 0:
        raise RuntimeError("empty feasible grid")
    vals = problem.evaluate_many(grid)
    return vals.min(axis=0), vals.max(axis=0)
