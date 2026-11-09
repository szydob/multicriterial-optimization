"""Cw3 — metody skalaryzacji i aproksymacja zbioru Pareta."""

from owd.scalarization.methods import (
    FrontApproximation,
    MethodName,
    ScalarizationSolution,
    solve_distance,
    solve_epsilon_constraint,
    solve_weighted_sum,
    sort_front_by_f1,
    sweep_distance,
    sweep_epsilon_constraint,
    sweep_weighted_sum,
    weighted_norm,
)
from owd.scalarization.problem import (
    BOUNDS,
    DEFAULT_PROBLEM,
    LB,
    UB,
    BiObjectiveProblem,
    Cw3PolynomialProblem,
    estimate_objective_bounds,
    sample_feasible_grid,
)
from owd.scalarization.visualization import build_cw3_figures

__all__ = [
    "BOUNDS",
    "DEFAULT_PROBLEM",
    "LB",
    "UB",
    "BiObjectiveProblem",
    "Cw3PolynomialProblem",
    "FrontApproximation",
    "MethodName",
    "ScalarizationSolution",
    "build_cw3_figures",
    "estimate_objective_bounds",
    "sample_feasible_grid",
    "solve_distance",
    "solve_epsilon_constraint",
    "solve_weighted_sum",
    "sort_front_by_f1",
    "sweep_distance",
    "sweep_epsilon_constraint",
    "sweep_weighted_sum",
    "weighted_norm",
]
