from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from owd.scalarization.methods import FrontApproximation, ScalarizationSolution, sort_front_by_f1
from owd.scalarization.problem import (
    DEFAULT_PROBLEM,
    LB,
    UB,
    BiObjectiveProblem,
    sample_feasible_grid,
)

_BG = "#0f172a"
_PAPER = "#111827"
_GRID = "#334155"
_FONT = "#e2e8f0"
_FEASIBLE = "#64748b"
_FRONT = "#22d3ee"
_SINGLE = "#f59e0b"
_SPLINE = "#a78bfa"


def _layout_style(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=_FONT)),
        paper_bgcolor=_PAPER,
        plot_bgcolor=_BG,
        font=dict(color=_FONT),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=20, t=70, b=50),
    )
    fig.update_xaxes(gridcolor=_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=_GRID, zeroline=False)
    return fig


def _attainable_cloud(
    problem: BiObjectiveProblem,
    n_per_axis: int = 35,
) -> tuple[np.ndarray, np.ndarray]:
    U = sample_feasible_grid(n_per_axis)
    F = problem.evaluate_many(U)
    return U, F


def build_decision_figure(
    *,
    front: FrontApproximation | None = None,
    single: ScalarizationSolution | None = None,
    problem: BiObjectiveProblem | None = None,
    title: str = "Przestrzeń decyzji U = [1,4]²",
) -> go.Figure:
    """Feasible square + Pareto decisions P(U,F) (+ optional single solve)."""
    problem = problem or DEFAULT_PROBLEM
    U, _ = _attainable_cloud(problem, n_per_axis=25)
    fig = go.Figure()

    # Feasible square outline
    box_x = [LB[0], UB[0], UB[0], LB[0], LB[0]]
    box_y = [LB[1], LB[1], UB[1], UB[1], LB[1]]
    fig.add_trace(
        go.Scatter(
            x=box_x,
            y=box_y,
            mode="lines",
            name="U (kwadrat)",
            line=dict(color="#94a3b8", width=2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=U[:, 0],
            y=U[:, 1],
            mode="markers",
            name="siatka U",
            marker=dict(size=3, color=_FEASIBLE, opacity=0.25),
            hoverinfo="skip",
        )
    )

    if front is not None:
        sorted_front = sort_front_by_f1(front)
        if len(sorted_front.decisions):
            fig.add_trace(
                go.Scatter(
                    x=sorted_front.decisions[:, 0],
                    y=sorted_front.decisions[:, 1],
                    mode="markers+lines",
                    name="P(U,F) (sweep)",
                    marker=dict(size=8, color=_FRONT, symbol="diamond"),
                    line=dict(color=_SPLINE, width=2, shape="spline"),
                )
            )

    if single is not None:
        fig.add_trace(
            go.Scatter(
                x=[single.u[0]],
                y=[single.u[1]],
                mode="markers",
                name="rozwiązanie skalaryzacji",
                marker=dict(size=14, color=_SINGLE, symbol="star", line=dict(width=1, color="white")),
            )
        )

    fig.update_xaxes(title_text="x", range=[0.8, 4.2])
    fig.update_yaxes(title_text="y", range=[0.8, 4.2], scaleanchor="x", scaleratio=1)
    return _layout_style(fig, title)


def build_objective_figure(
    *,
    front: FrontApproximation | None = None,
    single: ScalarizationSolution | None = None,
    problem: BiObjectiveProblem | None = None,
    title: str = "Przestrzeń celów F(U) / FP(U)",
) -> go.Figure:
    """Attainable set cloud + approximated nondominated front FP(U)."""
    problem = problem or DEFAULT_PROBLEM
    _, F = _attainable_cloud(problem, n_per_axis=35)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=F[:, 0],
            y=F[:, 1],
            mode="markers",
            name="F(U) (siatka)",
            marker=dict(size=4, color=_FEASIBLE, opacity=0.35),
            hoverinfo="skip",
        )
    )

    if front is not None:
        sorted_front = sort_front_by_f1(front)
        if len(sorted_front.objectives):
            fig.add_trace(
                go.Scatter(
                    x=sorted_front.objectives[:, 0],
                    y=sorted_front.objectives[:, 1],
                    mode="markers+lines",
                    name="FP(U) (sweep)",
                    marker=dict(size=8, color=_FRONT, symbol="diamond"),
                    line=dict(color=_SPLINE, width=2, shape="spline"),
                )
            )

    if single is not None:
        fig.add_trace(
            go.Scatter(
                x=[single.f[0]],
                y=[single.f[1]],
                mode="markers",
                name="F(u*)",
                marker=dict(size=14, color=_SINGLE, symbol="star", line=dict(width=1, color="white")),
            )
        )

    fig.update_xaxes(title_text="F₁")
    fig.update_yaxes(title_text="F₂")
    return _layout_style(fig, title)


def build_cw3_figures(
    *,
    front: FrontApproximation | None = None,
    single: ScalarizationSolution | None = None,
    problem: BiObjectiveProblem | None = None,
    method_label: str = "",
) -> tuple[go.Figure, go.Figure]:
    """Return (decision_fig, objective_fig)."""
    suffix = f" · {method_label}" if method_label else ""
    return (
        build_decision_figure(
            front=front,
            single=single,
            problem=problem,
            title=f"Przestrzeń decyzji P(U,F){suffix}",
        ),
        build_objective_figure(
            front=front,
            single=single,
            problem=problem,
            title=f"Przestrzeń celów FP(U){suffix}",
        ),
    )


def build_combined_figure(
    *,
    front: FrontApproximation | None = None,
    single: ScalarizationSolution | None = None,
    problem: BiObjectiveProblem | None = None,
    method_label: str = "",
) -> go.Figure:
    """Side-by-side decision / objective view (optional compact layout)."""
    dfig, ofig = build_cw3_figures(
        front=front, single=single, problem=problem, method_label=method_label
    )
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Decyzje U", "Cele F"))
    for tr in dfig.data:
        fig.add_trace(tr, row=1, col=1)
    for tr in ofig.data:
        fig.add_trace(tr, row=1, col=2)
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=1, scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text="F₁", row=1, col=2)
    fig.update_yaxes(title_text="F₂", row=1, col=2)
    return _layout_style(fig, f"Cw3 skalaryzacja{(' · ' + method_label) if method_label else ''}")
