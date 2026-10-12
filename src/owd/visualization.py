"""Wykresy Plotly frontu Pareto (2D / 3D + kolor 4. kryterium)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from owd.types import Criterion, Points

# Paleta zbliżona do GUI z labów, ale czytelniejsza.
_COLOR_SET = "#94a3b8"
_COLOR_DOMINATED = "#3b82f6"
_COLOR_NONDOM = "#06b6d4"
_COLOR_COMPARED = "#ef4444"
_BG = "#0f172a"
_PAPER = "#111827"
_GRID = "#334155"
_FONT = "#e2e8f0"


def _axis_title(crit: Criterion) -> str:
    direction = "Min" if crit.sense.value == "min" else "Max"
    return f"{crit.name} ({direction})"


def _split_indices(
    n: int,
    nondominated_indices: Sequence[int],
    compared_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nd = set(int(i) for i in nondominated_indices)
    compared = set(int(i) for i in (compared_indices or []))
    all_idx = np.arange(n)
    dominated = np.array([i for i in all_idx if i not in nd], dtype=int)
    nondom = np.array(sorted(nd), dtype=int)
    compared_arr = np.array(sorted(compared), dtype=int)
    return nondom, dominated, compared_arr


def build_pareto_figure(
    points: Points,
    criteria: list[Criterion],
    nondominated_indices: Sequence[int],
    *,
    compared_indices: Sequence[int] | None = None,
    title: str | None = None,
) -> go.Figure:
    """
    Buduje wykres:
    - 2 kryteria → 2D
    - 3 → 3D
    - ≥4 → 3D + kolor = 4. kryterium
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 1:
        raise ValueError("points must be a 2D array with ≥1 column")
    if len(criteria) != pts.shape[1]:
        raise ValueError("criteria length must match point dimensions")

    n, k = pts.shape
    nondom, dominated, compared = _split_indices(n, nondominated_indices, compared_indices)
    fig_title = title or "Zbiór punktów niezdominowanych P(X)"

    if k == 1:
        fig = go.Figure()
        _add_1d(fig, pts, criteria, nondom, dominated, compared)
    elif k == 2:
        fig = go.Figure()
        _add_2d(fig, pts, criteria, nondom, dominated, compared)
    else:
        fig = go.Figure()
        color_values = pts[:, 3] if k >= 4 else None
        color_title = _axis_title(criteria[3]) if k >= 4 else None
        _add_3d(fig, pts, criteria, nondom, dominated, compared, color_values, color_title)

    fig.update_layout(
        title=dict(text=fig_title, font=dict(size=18, color=_FONT)),
        paper_bgcolor=_PAPER,
        plot_bgcolor=_BG,
        font=dict(color=_FONT, family="Inter, Segoe UI, sans-serif"),
        legend=dict(
            bgcolor="rgba(17,24,39,0.85)",
            bordercolor=_GRID,
            borderwidth=1,
            orientation="v",
            x=1.02,
            y=1,
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=640,
    )
    return fig


def _scatter_kwargs(color_values: np.ndarray | None, color_title: str | None) -> dict:
    if color_values is None:
        return {}
    return {
        "marker_color": color_values,
        "marker_colorscale": "Turbo",
        "marker_showscale": True,
        "marker_colorbar": dict(title=color_title or "", thickness=14),
    }


def _add_1d(
    fig: go.Figure,
    pts: np.ndarray,
    criteria: list[Criterion],
    nondom: np.ndarray,
    dominated: np.ndarray,
    compared: np.ndarray,
) -> None:
    y_all = np.zeros(len(pts))
    fig.add_trace(
        go.Scatter(
            x=pts[:, 0],
            y=y_all,
            mode="markers",
            name="Zbiór punktów",
            marker=dict(size=7, color=_COLOR_SET, opacity=0.45, symbol="circle"),
        )
    )
    if len(dominated):
        fig.add_trace(
            go.Scatter(
                x=pts[dominated, 0],
                y=np.zeros(len(dominated)),
                mode="markers",
                name="Punkty zdominowane",
                marker=dict(size=9, color=_COLOR_DOMINATED, symbol="circle"),
            )
        )
    if len(nondom):
        fig.add_trace(
            go.Scatter(
                x=pts[nondom, 0],
                y=np.zeros(len(nondom)),
                mode="markers",
                name="Zbiór niezdominowany",
                marker=dict(
                    size=12,
                    color=_COLOR_NONDOM,
                    symbol="diamond",
                    line=dict(width=1, color="#fff"),
                ),
            )
        )
    if len(compared):
        fig.add_trace(
            go.Scatter(
                x=pts[compared, 0],
                y=np.zeros(len(compared)),
                mode="markers",
                name="Punkty porównywane",
                marker=dict(size=14, color=_COLOR_COMPARED, symbol="square", opacity=0.9),
            )
        )
    fig.update_xaxes(title=_axis_title(criteria[0]), gridcolor=_GRID, zeroline=False)
    fig.update_yaxes(visible=False, range=[-1, 1])


def _add_2d(
    fig: go.Figure,
    pts: np.ndarray,
    criteria: list[Criterion],
    nondom: np.ndarray,
    dominated: np.ndarray,
    compared: np.ndarray,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=pts[:, 0],
            y=pts[:, 1],
            mode="markers",
            name="Zbiór punktów",
            marker=dict(size=7, color=_COLOR_SET, opacity=0.4, symbol="circle"),
        )
    )
    if len(dominated):
        fig.add_trace(
            go.Scatter(
                x=pts[dominated, 0],
                y=pts[dominated, 1],
                mode="markers",
                name="Punkty zdominowane",
                marker=dict(
                    size=9,
                    color=_COLOR_DOMINATED,
                    symbol="circle",
                    line=dict(width=0.5, color="#1e3a8a"),
                ),
            )
        )
    if len(nondom):
        fig.add_trace(
            go.Scatter(
                x=pts[nondom, 0],
                y=pts[nondom, 1],
                mode="markers",
                name="Zbiór niezdominowany",
                marker=dict(
                    size=13,
                    color=_COLOR_NONDOM,
                    symbol="diamond",
                    line=dict(width=1.5, color="#ecfeff"),
                ),
            )
        )
    if len(compared):
        fig.add_trace(
            go.Scatter(
                x=pts[compared, 0],
                y=pts[compared, 1],
                mode="markers",
                name="Punkty porównywane",
                marker=dict(size=15, color=_COLOR_COMPARED, symbol="square", opacity=0.95),
            )
        )
    fig.update_xaxes(title=_axis_title(criteria[0]), gridcolor=_GRID, zeroline=False, showline=True)
    fig.update_yaxes(title=_axis_title(criteria[1]), gridcolor=_GRID, zeroline=False, showline=True)


def _add_3d(
    fig: go.Figure,
    pts: np.ndarray,
    criteria: list[Criterion],
    nondom: np.ndarray,
    dominated: np.ndarray,
    compared: np.ndarray,
    color_values: np.ndarray | None,
    color_title: str | None,
) -> None:
    base_kwargs = _scatter_kwargs(color_values, color_title)

    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            name="Zbiór punktów",
            marker=dict(size=3, color=_COLOR_SET, opacity=0.35, symbol="circle"),
            showlegend=True,
        )
    )
    if len(dominated):
        marker = dict(size=4, symbol="circle", opacity=0.85)
        if color_values is None:
            marker["color"] = _COLOR_DOMINATED
        else:
            marker["color"] = color_values[dominated]
            marker["colorscale"] = "Turbo"
            marker["showscale"] = True
            marker["colorbar"] = dict(title=color_title or "", thickness=14)
        fig.add_trace(
            go.Scatter3d(
                x=pts[dominated, 0],
                y=pts[dominated, 1],
                z=pts[dominated, 2],
                mode="markers",
                name="Punkty zdominowane",
                marker=marker,
            )
        )
    if len(nondom):
        fig.add_trace(
            go.Scatter3d(
                x=pts[nondom, 0],
                y=pts[nondom, 1],
                z=pts[nondom, 2],
                mode="markers",
                name="Zbiór niezdominowany",
                marker=dict(
                    size=7,
                    color=_COLOR_NONDOM,
                    symbol="diamond",
                    line=dict(width=1, color="#ecfeff"),
                ),
            )
        )
    if len(compared):
        fig.add_trace(
            go.Scatter3d(
                x=pts[compared, 0],
                y=pts[compared, 1],
                z=pts[compared, 2],
                mode="markers",
                name="Punkty porównywane",
                marker=dict(size=8, color=_COLOR_COMPARED, symbol="square"),
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis_title=_axis_title(criteria[0]),
            yaxis_title=_axis_title(criteria[1]),
            zaxis_title=_axis_title(criteria[2]),
            bgcolor=_BG,
            xaxis=dict(backgroundcolor=_BG, gridcolor=_GRID, showbackground=True),
            yaxis=dict(backgroundcolor=_BG, gridcolor=_GRID, showbackground=True),
            zaxis=dict(backgroundcolor=_BG, gridcolor=_GRID, showbackground=True),
        )
    )
    _ = base_kwargs  # color scale applied on dominated when k>=4


def build_animation_figure(
    points: Points,
    criteria: list[Criterion],
    frames_nondom: list[Sequence[int]],
    frames_compared: list[Sequence[int]],
    *,
    title: str = "Animacja budowy P(X)",
) -> go.Figure:
    """Animacja: kolejne klatki z rosnącym zbiorem niezdominowanym + podświetlone porównania."""
    pts = np.asarray(points, dtype=float)
    if not frames_nondom:
        return build_pareto_figure(pts, criteria, [], title=title)

    final_nd = frames_nondom[-1]
    base = build_pareto_figure(
        pts,
        criteria,
        final_nd,
        compared_indices=frames_compared[0] if frames_compared else None,
        title=title,
    )

    plotly_frames = []
    for step, (nd, compared) in enumerate(zip(frames_nondom, frames_compared, strict=False)):
        frame_fig = build_pareto_figure(
            pts,
            criteria,
            nd,
            compared_indices=compared,
            title=f"{title} — krok {step + 1}/{len(frames_nondom)}",
        )
        plotly_frames.append(go.Frame(data=frame_fig.data, name=str(step), layout=frame_fig.layout))

    base.frames = plotly_frames
    base.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.0,
                xanchor="left",
                yanchor="top",
                pad=dict(t=60, r=10),
                buttons=[
                    dict(
                        label="▶ Odtwórz",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=450, redraw=True),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ Pauza",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(prefix="Krok: ", visible=True, xanchor="right"),
                pad=dict(t=30, b=10),
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(i)],
                            dict(mode="immediate", frame=dict(duration=0, redraw=True)),
                        ],
                        label=str(i + 1),
                    )
                    for i in range(len(plotly_frames))
                ],
            )
        ],
    )
    return base


def build_benchmark_figure(df) -> go.Figure:
    """Prosty wykres porównawczy czasu i liczby porównań."""
    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
    if ok.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Brak danych benchmarku",
            paper_bgcolor=_PAPER,
            font=dict(color=_FONT),
        )
        return fig

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Czas [ms]", "Porównania punktów"))
    fig.add_trace(
        go.Bar(
            x=ok["algorithm"],
            y=ok["elapsed_seconds"] * 1000.0,
            name="czas [ms]",
            marker_color="#06b6d4",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=ok["algorithm"],
            y=ok["point_comparisons"],
            name="porównania",
            marker_color="#3b82f6",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        paper_bgcolor=_PAPER,
        plot_bgcolor=_BG,
        font=dict(color=_FONT),
        height=420,
        title="Benchmark algorytmów",
    )
    fig.update_xaxes(gridcolor=_GRID)
    fig.update_yaxes(gridcolor=_GRID)
    return fig
