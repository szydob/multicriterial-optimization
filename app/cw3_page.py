"""Cw3 Streamlit panel: scalarization, Pareto sweeps, visible S1/S2/S3 benchmark mode."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from owd.scalarization import (
    INSTANCES,
    benchmark_scalarization,
    estimate_objective_bounds,
    get_problem,
    solve_distance,
    solve_epsilon_constraint,
    solve_weighted_sum,
    sort_front_by_f1,
    sweep_distance,
    sweep_epsilon_constraint,
    sweep_weighted_sum,
)
from owd.scalarization.benchmark import METHOD_LABELS as BENCH_METHOD_LABELS
from owd.scalarization.benchmark import run_method_sweep
from owd.scalarization.visualization import build_cw3_figures

METHOD_LABELS = {
    "weighted_sum": "S1 — suma ważona (funkcja liniowa)",
    "epsilon_constraint": "S2 — metoda ε-ograniczeń",
    "distance": "S3 — odległość od punktu dominującego",
}

P_CHOICES = {
    "1 (Manhattan)": 1.0,
    "2 (Euklides)": 2.0,
    "4": 4.0,
    "∞ (Czebyszew)": float("inf"),
}

CW3_MODES = {
    "benchmark": "⚡ Benchmark — porównaj S1 / S2 / S3",
    "sweep": "Sweep Pareta (jedna metoda)",
    "single": "Pojedyncze rozwiązanie",
}


def _init_cw3_state() -> None:
    defaults = {
        "cw3_front": None,
        "cw3_single": None,
        "cw3_method": "weighted_sum",
        "cw3_bench": None,
        "cw3_bench_fronts": None,
        "cw3_instance": "square_m1_1",
        "cw3_ui_mode": "benchmark",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _bench_show_table(bench_df: pd.DataFrame) -> pd.DataFrame:
    show = bench_df[
        [
            "instance",
            "method_label",
            "n_success",
            "success_rate",
            "sweep_time_ms",
            "mean_solve_ms",
            "median_solve_ms",
            "f1_span",
            "f2_span",
            "mean_spacing",
            "dominated_share",
        ]
    ].copy()
    return show.rename(
        columns={
            "instance": "U",
            "method_label": "Metoda",
            "n_success": "|P|",
            "success_rate": "sukces",
            "sweep_time_ms": "sweep [ms]",
            "mean_solve_ms": "śr. solve [ms]",
            "median_solve_ms": "mediana solve [ms]",
            "f1_span": "rozpiętość F1",
            "f2_span": "rozpiętość F2",
            "mean_spacing": "śr. odstęp",
            "dominated_share": "udział zdominowanych",
        }
    )


def _render_benchmark_mode(problem, n_points: int, scope_all: bool) -> None:
    """Dedicated full-page mode: run S1+S2+S3 and compare times / fronts."""
    st.subheader("Porównanie S1 / S2 / S3")
    st.markdown(
        f"**U:** {problem.name}  ·  sweep **n = {n_points}** punktów na metodę"
        + ("  ·  także pozostałe zbiory U" if scope_all else "")
    )

    c_run, c_hint = st.columns([1, 2])
    run_bench = c_run.button(
        "▶ Uruchom porównanie S1 + S2 + S3",
        type="primary",
        use_container_width=True,
        key="cw3_bench_run_main",
    )
    c_hint.caption(
        "Liczy niezależny sweep dla każdej metody, mierzy czas i zbiera metryki frontu. "
        "Wynik pojawia się poniżej w tabeli i na wykresach."
    )

    if run_bench:
        with st.spinner("Benchmark S1 / S2 / S3…"):
            ids = list(INSTANCES.keys()) if scope_all else [problem.id]
            st.session_state.cw3_bench = benchmark_scalarization(
                instance_ids=ids,  # type: ignore[arg-type]
                n_sweep=int(n_points),
            )
            # Fronts for the currently selected U (for side-by-side objective plots).
            fronts = {}
            for method in ("weighted_sum", "epsilon_constraint", "distance"):
                fronts[method] = sort_front_by_f1(
                    run_method_sweep(method, int(n_points), problem)
                )
            st.session_state.cw3_bench_fronts = fronts

    bench_df: pd.DataFrame | None = st.session_state.cw3_bench
    fronts = st.session_state.cw3_bench_fronts

    if bench_df is None:
        st.info(
            "To jest tryb **Benchmark**. Naciśnij zielony przycisk powyżej, "
            "żeby porównać czasy i fronty S1, S2 i S3 obok siebie."
        )
        return

    # Filter table to current U unless comparing all
    view = bench_df if scope_all else bench_df[bench_df["instance_id"] == problem.id]
    if view.empty:
        view = bench_df

    st.markdown("#### Tabela porównawcza")
    st.dataframe(_bench_show_table(view), use_container_width=True)

    # Bar charts
    st.markdown("#### Czas: sweep i średni solve")
    fig_time = px.bar(
        view,
        x="method_label",
        y=["sweep_time_ms", "mean_solve_ms"],
        barmode="group",
        facet_col="instance" if scope_all and view["instance"].nunique() > 1 else None,
        labels={
            "method_label": "Metoda",
            "value": "czas [ms]",
            "variable": "metryka",
        },
        title="Czas odtworzenia frontu (sweep) vs średni czas jednej skalaryzacji",
    )
    fig_time.update_layout(legend_title_text="")
    st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("#### Pokrycie frontu (rozpiętość F₁ / F₂)")
    fig_span = px.bar(
        view,
        x="method_label",
        y=["f1_span", "f2_span"],
        barmode="group",
        facet_col="instance" if scope_all and view["instance"].nunique() > 1 else None,
        labels={"method_label": "Metoda", "value": "rozpiętość", "variable": "oś"},
    )
    st.plotly_chart(fig_span, use_container_width=True)

    # Side-by-side fronts for selected U
    if fronts:
        st.markdown(f"#### Fronty FP(U) na wybranym U — {problem.name}")
        cols = st.columns(3)
        for col, method in zip(
            cols, ("weighted_sum", "epsilon_constraint", "distance"), strict=True
        ):
            front = fronts.get(method)
            with col:
                st.caption(METHOD_LABELS[method])
                if front is not None:
                    _, ofig = build_cw3_figures(
                        front=front,
                        problem=problem,
                        method_label=BENCH_METHOD_LABELS[method],
                    )
                    ofig.update_layout(height=360, margin=dict(t=40, b=30, l=40, r=10))
                    st.plotly_chart(ofig, use_container_width=True)
                    st.metric("sweep [ms]", f"{front.elapsed_seconds * 1000:.1f}")
                    mean_ms = (
                        float(np.mean(front.per_solve_seconds) * 1000.0)
                        if len(front.per_solve_seconds)
                        else float("nan")
                    )
                    st.metric("śr. solve [ms]", f"{mean_ms:.2f}")

    st.download_button(
        "Pobierz benchmark (CSV)",
        data=bench_df.to_csv(index=False).encode("utf-8"),
        file_name="cw3_benchmark.csv",
        mime="text/csv",
    )


def _render_single_or_sweep(
    *,
    problem,
    method: str,
    mode: str,
    n_points: int,
    lam1: float,
    primary: int,
    a_eps: float,
    p_val: float,
    ref1: float,
    ref2: float,
    run: bool,
) -> None:
    if run:
        with st.spinner("Skalaryzacja / sweep Pareta…"):
            if mode == "sweep":
                if method == "weighted_sum":
                    front = sweep_weighted_sum(n_points, problem=problem)
                elif method == "epsilon_constraint":
                    front = sweep_epsilon_constraint(
                        n_points, primary=primary, problem=problem
                    )
                else:
                    front = sweep_distance(
                        n_points,
                        ref=np.array([ref1, ref2], dtype=float),
                        p=p_val,
                        problem=problem,
                    )
                st.session_state.cw3_front = sort_front_by_f1(front)
                st.session_state.cw3_single = None
            else:
                if method == "weighted_sum":
                    lam = np.array([lam1, 1.0 - lam1], dtype=float)
                    single = solve_weighted_sum(lam, problem=problem)
                elif method == "epsilon_constraint":
                    single = solve_epsilon_constraint(
                        a_eps, primary=primary, problem=problem
                    )
                else:
                    lam = np.array([lam1, 1.0 - lam1], dtype=float)
                    single = solve_distance(
                        np.array([ref1, ref2], dtype=float),
                        lam,
                        p_val,
                        problem=problem,
                    )
                st.session_state.cw3_single = single

    front = st.session_state.cw3_front
    single = st.session_state.cw3_single

    if front is None and single is None:
        st.info("Ustaw parametry w sidebarze i naciśnij **Uruchom**.")
        dfig, ofig = build_cw3_figures(problem=problem)
        c1, c2 = st.columns(2)
        c1.plotly_chart(dfig, use_container_width=True)
        c2.plotly_chart(ofig, use_container_width=True)
        return

    n_ok = int(front.successes.sum()) if front is not None else 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Metoda", METHOD_LABELS[method].split("—")[0].strip())
    m2.metric("|P(U,F)|", n_ok if front is not None else "—")
    if single is not None:
        m3.metric("F₁(u*)", f"{single.f[0]:.4f}")
        m4.metric("Czas solve", f"{single.elapsed_seconds * 1000:.2f} ms")
    else:
        assert front is not None
        m3.metric("Czas sweep", f"{front.elapsed_seconds * 1000:.1f} ms")
        mean_ms = (
            float(np.mean(front.per_solve_seconds) * 1000.0)
            if len(front.per_solve_seconds)
            else float("nan")
        )
        m4.metric("Śr. solve", f"{mean_ms:.2f} ms")

    dfig, ofig = build_cw3_figures(
        front=front,
        single=single,
        problem=problem,
        method_label=METHOD_LABELS[method],
    )
    c1, c2 = st.columns(2)
    c1.plotly_chart(dfig, use_container_width=True)
    c2.plotly_chart(ofig, use_container_width=True)

    if front is not None and len(front.decisions):
        with st.expander("Tabela punktów frontu"):
            df = pd.DataFrame(
                {
                    "x": front.decisions[:, 0],
                    "y": front.decisions[:, 1],
                    "F1": front.objectives[:, 0],
                    "F2": front.objectives[:, 1],
                    "solve_ms": (
                        front.per_solve_seconds * 1000.0
                        if len(front.per_solve_seconds) == len(front.decisions)
                        else np.nan
                    ),
                }
            )
            st.dataframe(df, use_container_width=True)


def render_cw3() -> None:
    """Render the Ćwiczenie 3 UI with a first-class Benchmark mode."""
    _init_cw3_state()

    st.title("Ćw. 3 — metody skalaryzacji")
    st.caption(
        "Przełącz **tryb** poniżej: Benchmark porównuje S1/S2/S3 obok siebie "
        "(czas + front). Sweep / pojedyncze — jedna metoda."
    )

    # ── Main-panel mode switch (hard to miss) ────────────────────────────────
    ui_mode = st.radio(
        "Tryb Ćw. 3",
        options=list(CW3_MODES.keys()),
        format_func=lambda k: CW3_MODES[k],
        horizontal=True,
        key="cw3_ui_mode",
    )

    with st.sidebar:
        st.markdown("## Ćw. 3 — ustawienia")
        st.success("Jesteś w panelu Ćw. 3 (nie Ćw. 2).")
        instance_id = st.selectbox(
            "Zbiór dopuszczalny U",
            options=list(INSTANCES.keys()),
            format_func=lambda i: INSTANCES[i].name,
            index=list(INSTANCES.keys()).index(st.session_state.cw3_instance)
            if st.session_state.cw3_instance in INSTANCES
            else 0,
            key="cw3_instance_select",
        )
        st.session_state.cw3_instance = instance_id
        problem = get_problem(instance_id)
        f_min, f_max = estimate_objective_bounds(problem, n_per_axis=50)

        n_points = st.slider("Liczba punktów aproksymacji n", 5, 80, 30, 1)

        scope_all = False
        method = st.session_state.cw3_method
        lam1 = 0.5
        primary = 0
        a_eps = float(0.5 * (f_min[1] + f_max[1]))
        p_val = 2.0
        ref1, ref2 = float(f_min[0] - 0.1), float(f_min[1] - 0.1)
        run = False

        if ui_mode == "benchmark":
            st.markdown("### Benchmark")
            scope_all = st.checkbox(
                "Porównaj też na wszystkich U (nie tylko wybrane)",
                value=False,
                help="Domyślnie: S1/S2/S3 tylko na wybranym U. Zaznacz, by dodać pozostałe zbiory.",
            )
            st.caption("Uruchomienie jest na stronie głównej (duży zielony przycisk).")
        else:
            method = st.selectbox(
                "Metoda",
                options=list(METHOD_LABELS.keys()),
                format_func=lambda m: METHOD_LABELS[m],
                key="cw3_method_select",
            )
            st.session_state.cw3_method = method

            if method == "weighted_sum":
                st.markdown("### Parametry S1")
                if ui_mode == "single":
                    lam1 = st.slider("λ₁ (λ₂ = 1−λ₁)", 0.01, 0.99, 0.5, 0.01)
            elif method == "epsilon_constraint":
                st.markdown("### Parametry S2")
                primary = st.radio(
                    "Kryterium optymalizowane Fⱼ",
                    options=[0, 1],
                    format_func=lambda i: f"F{i + 1}",
                    horizontal=True,
                )
                constrained = 1 - primary
                a_lo = float(f_min[constrained])
                a_hi = float(f_max[constrained])
                if ui_mode == "single":
                    a_eps = st.slider(
                        f"a = ε (górne ograniczenie F{constrained + 1})",
                        min_value=a_lo,
                        max_value=a_hi,
                        value=float(0.5 * (a_lo + a_hi)),
                        step=max((a_hi - a_lo) / 100, 1e-3),
                    )
            else:
                st.markdown("### Parametry S3")
                p_label = st.selectbox("p (metryka)", options=list(P_CHOICES.keys()), index=1)
                p_val = P_CHOICES[p_label]
                c1, c2 = st.columns(2)
                ref1 = c1.number_input("x₁ (ref F₁)", value=ref1, format="%.4f")
                ref2 = c2.number_input("x₂ (ref F₂)", value=ref2, format="%.4f")
                if ui_mode == "single":
                    lam1 = st.slider("λ₁ (λ₂ = 1−λ₁)", 0.01, 0.99, 0.5, 0.01)

            run = st.button("Uruchom", type="primary", use_container_width=True)

        if st.button("Wyczyść wyniki Cw3", use_container_width=True):
            st.session_state.cw3_front = None
            st.session_state.cw3_single = None
            st.session_state.cw3_bench = None
            st.session_state.cw3_bench_fronts = None

    st.markdown(
        f"**Instancja:** {problem.name}  \n"
        f"`{problem.f1_latex}` · `{problem.f2_latex}`  \n"
        f"Zakresy (siatka): F₁ ∈ [{f_min[0]:.3f}, {f_max[0]:.3f}], "
        f"F₂ ∈ [{f_min[1]:.3f}, {f_max[1]:.3f}]"
    )

    if ui_mode == "benchmark":
        _render_benchmark_mode(problem, n_points, scope_all)
    else:
        _render_single_or_sweep(
            problem=problem,
            method=method,
            mode=ui_mode,
            n_points=n_points,
            lam1=lam1,
            primary=primary,
            a_eps=a_eps,
            p_val=p_val,
            ref1=ref1,
            ref2=ref2,
            run=run,
        )
