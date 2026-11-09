"""Cw3 Streamlit panel: scalarization methods + Pareto-set reconstruction (sweeps)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from owd.scalarization import (
    DEFAULT_PROBLEM,
    estimate_objective_bounds,
    solve_distance,
    solve_epsilon_constraint,
    solve_weighted_sum,
    sort_front_by_f1,
    sweep_distance,
    sweep_epsilon_constraint,
    sweep_weighted_sum,
)
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


def _init_cw3_state() -> None:
    defaults = {
        "cw3_front": None,
        "cw3_single": None,
        "cw3_method": "weighted_sum",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_cw3() -> None:
    """Render the Ćwiczenie 3 scalarization + Pareto reconstruction UI."""
    _init_cw3_state()
    problem = DEFAULT_PROBLEM
    f_min, f_max = estimate_objective_bounds(problem, n_per_axis=50)

    st.title("Ćw. 3 — metody skalaryzacji")
    st.caption(
        "Rekonstrukcja zbioru Pareta P(U,F) / FP(U) przez sweep parametrów "
        "(wagi λ, ε, metryka ℓ_p) — jak w instrukcji Cw3 i demo MATLAB (suma ważona / ε)."
    )

    st.markdown(
        f"""
**Instancja:** U = [1, 4] × [1, 4], minimalizacja F = (F₁, F₂)

- `{problem.f1_latex}`
- `{problem.f2_latex}`

Orientacyjne zakresy celów (siatka):  
F₁ ∈ [{f_min[0]:.3f}, {f_max[0]:.3f}], F₂ ∈ [{f_min[1]:.3f}, {f_max[1]:.3f}]
"""
    )

    with st.sidebar:
        st.markdown("## Cw3 — skalaryzacja")
        method = st.selectbox(
            "Metoda",
            options=list(METHOD_LABELS.keys()),
            format_func=lambda m: METHOD_LABELS[m],
            key="cw3_method_select",
        )
        st.session_state.cw3_method = method

        mode = st.radio(
            "Tryb",
            options=["sweep", "single"],
            format_func=lambda m: (
                "Odtwórz zbiór Pareta (sweep)" if m == "sweep" else "Pojedyncze rozwiązanie"
            ),
            horizontal=False,
        )
        n_points = st.slider("Liczba punktów aproksymacji n", 5, 80, 30, 1)

        # Method-specific parameters
        lam1 = 0.5
        primary = 0
        a_eps = float(0.5 * (f_min[1] + f_max[1]))
        p_val = 2.0
        ref1, ref2 = float(f_min[0] - 0.1), float(f_min[1] - 0.1)

        if method == "weighted_sum":
            st.markdown("### Parametry S1")
            st.caption("λ > 0, znormalizowane λ₁+λ₂=1. Sweep: λ₁ = ε … 1−ε krokiem ρ.")
            if mode == "single":
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
            st.caption(f"Ograniczenie: F{constrained + 1}(u) ≤ a (=ε)")
            a_lo = float(f_min[constrained])
            a_hi = float(f_max[constrained])
            if mode == "single":
                a_eps = st.slider(
                    f"a = ε (górne ograniczenie F{constrained + 1})",
                    min_value=a_lo,
                    max_value=a_hi,
                    value=float(0.5 * (a_lo + a_hi)),
                    step=max((a_hi - a_lo) / 100, 1e-3),
                )
            else:
                st.caption(f"Sweep a ∈ [{a_lo:.3f}, {a_hi:.3f}] stałym krokiem ρ.")
        else:
            st.markdown("### Parametry S3")
            st.caption("Minimalizacja ‖F(u)−x‖_{λ,p}; x — punkt dominujący (utopia).")
            p_label = st.selectbox("p (metryka)", options=list(P_CHOICES.keys()), index=1)
            p_val = P_CHOICES[p_label]
            c1, c2 = st.columns(2)
            ref1 = c1.number_input("x₁ (ref F₁)", value=ref1, format="%.4f")
            ref2 = c2.number_input("x₂ (ref F₂)", value=ref2, format="%.4f")
            if mode == "single":
                lam1 = st.slider("λ₁ (λ₂ = 1−λ₁)", 0.01, 0.99, 0.5, 0.01)

        run = st.button("Uruchom", type="primary", use_container_width=True)
        if st.button("Wyczyść wynik Cw3", use_container_width=True):
            st.session_state.cw3_front = None
            st.session_state.cw3_single = None

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
                # Keep last front if any, so single point overlays reconstructed set.
                # Do not clear front.

    front = st.session_state.cw3_front
    single = st.session_state.cw3_single

    if front is None and single is None:
        st.info(
            "Wybierz metodę i naciśnij **Uruchom**. "
            "Tryb *sweep* odtwarza dyskretną aproksymację P(U,F) i FP(U)."
        )
        dfig, ofig = build_cw3_figures(problem=problem)
        c1, c2 = st.columns(2)
        c1.plotly_chart(dfig, use_container_width=True)
        c2.plotly_chart(ofig, use_container_width=True)
        return

    n_ok = int(front.successes.sum()) if front is not None else 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Metoda", METHOD_LABELS[st.session_state.cw3_method].split("—")[0].strip())
    m2.metric("|P(U,F)| (sweep)", n_ok if front is not None else "—")
    if single is not None:
        m3.metric("F₁(u*)", f"{single.f[0]:.4f}")
        m4.metric("F₂(u*)", f"{single.f[1]:.4f}")
    else:
        m3.metric("Sukcesy SLSQP", f"{n_ok}/{len(front.successes)}" if front else "—")
        m4.metric("Tryb", "sweep Pareta")

    dfig, ofig = build_cw3_figures(
        front=front,
        single=single,
        problem=problem,
        method_label=METHOD_LABELS[st.session_state.cw3_method],
    )
    tab_plots, tab_table = st.tabs(["Wykresy U / F", "Tabela punktów"])
    with tab_plots:
        c1, c2 = st.columns(2)
        c1.plotly_chart(dfig, use_container_width=True)
        c2.plotly_chart(ofig, use_container_width=True)
        st.caption(
            "Linia spline łączy punkty z sweepu posortowane po F₁ "
            "(aproksymacja frontu jak w instrukcji Cw3)."
        )

    with tab_table:
        if front is not None and len(front.decisions):
            df = pd.DataFrame(
                {
                    "x": front.decisions[:, 0],
                    "y": front.decisions[:, 1],
                    "F1": front.objectives[:, 0],
                    "F2": front.objectives[:, 1],
                }
            )
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Pobierz front (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="cw3_pareto_front.csv",
                mime="text/csv",
            )
        if single is not None:
            st.subheader("Pojedyncze rozwiązanie")
            st.write(
                {
                    "x": float(single.u[0]),
                    "y": float(single.u[1]),
                    "F1": float(single.f[0]),
                    "F2": float(single.f[1]),
                    "S(u)": float(single.scalar_value),
                    "success": single.success,
                    "params": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in single.params.items()},
                }
            )
