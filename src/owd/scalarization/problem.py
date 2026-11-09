from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Feasible set U = [1, 4] × [1, 4] (user / lab instance).
LB = np.array([1.0, 1.0], dtype=float)
UB = np.array([4.0, 4.0], dtype=float)
BOUNDS = [(1.0, 4.0), (1.0, 4.0)]


@dataclass(frozen=True)
class BiObjectiveProblem:
    """Vector criterion F:U→R² to be minimized (natural order)."""

    name: str
    f1_latex: str
    f2_latex: str

    def f1(self, x: float, y: float) -> float:
        raise NotImplementedError

    def f2(self, x: float, y: float) -> float:
        raise NotImplementedError

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        """Return F(u) = (F1, F2) for decision vector u=(x,y)."""
        u = np.asarray(u, dtype=float).reshape(-1)
        if u.size != 2:
            raise ValueError("decision vector must be 2-dimensional")
        return np.array([self.f1(float(u[0]), float(u[1])), self.f2(float(u[0]), float(u[1]))])

    def evaluate_many(self, U: np.ndarray) -> np.ndarray:
        """Evaluate F on an (n, 2) array of decision points → (n, 2) objectives."""
        U = np.asarray(U, dtype=float)
        if U.ndim != 2 or U.shape[1] != 2:
            raise ValueError("U must have shape (n, 2)")
        out = np.empty((U.shape[0], 2), dtype=float)
        for i, row in enumerate(U):
            out[i] = self.evaluate(row)
        return out


class Cw3PolynomialProblem(BiObjectiveProblem):
    """
    Nonlinear polynomial bi-objective problem for Ćw. 3 (part a).

    The lab PDF does not fix a single F for every group (functions differ per
    group; polynomials are allowed). This instance uses clear competing
    polynomials on the user-specified square U=[1,4]²:

        F1(x,y) = (x − 1)² + (y − 2)² + 0.25·x·y
        F2(x,y) = (x − 4)² + (y − 3)² + 0.15·x²

    Both are minimized. Ideal corners pull toward opposite sides of the square,
    so the Pareto set is a nontrivial curve inside U.
    """

    def __init__(self) -> None:
        super().__init__(
            name="Cw3 polynomials on [1,4]²",
            f1_latex=r"F_1(x,y)=(x-1)^2+(y-2)^2+0.25\,xy",
            f2_latex=r"F_2(x,y)=(x-4)^2+(y-3)^2+0.15\,x^2",
        )

    def f1(self, x: float, y: float) -> float:
        return (x - 1.0) ** 2 + (y - 2.0) ** 2 + 0.25 * x * y

    def f2(self, x: float, y: float) -> float:
        return (x - 4.0) ** 2 + (y - 3.0) ** 2 + 0.15 * x**2


DEFAULT_PROBLEM = Cw3PolynomialProblem()


def sample_feasible_grid(n_per_axis: int = 40) -> np.ndarray:
    """Regular grid over U for background / attainable-set visualization."""
    xs = np.linspace(LB[0], UB[0], n_per_axis)
    ys = np.linspace(LB[1], UB[1], n_per_axis)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])


def estimate_objective_bounds(
    problem: BiObjectiveProblem | None = None,
    n_per_axis: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rough min/max of each objective over a dense grid of U.

    Useful for ε ranges and a default utopia / nadir estimate.
    """
    problem = problem or DEFAULT_PROBLEM
    vals = problem.evaluate_many(sample_feasible_grid(n_per_axis))
    return vals.min(axis=0), vals.max(axis=0)
