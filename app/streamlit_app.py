"""OWD — Streamlit GUI: kryteria, dane, algorytmy, wykres, benchmark, eksport."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from owd import __version__
from owd.algorithms import AlgorithmName, find_nondominated
from owd.algorithms.base import default_criteria
from owd.animation import generate_filter_frames
from owd.benchmark import run_benchmark
from owd.export import points_to_dataframe, result_frames, to_excel_bytes
from owd.generators import LAB_EXAMPLE_POINTS, generate_points
from owd.types import Criterion, DistributionName, Sense
from owd.visualization import build_animation_figure, build_benchmark_figure, build_pareto_figure

st.set_page_config(
    page_title="OWD — optymalizacja wielokryterialna",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ALGORITHM_LABELS: dict[AlgorithmName, str] = {
    "naive_no_filter": "Naiwny 1 — bez filtracji",
    "naive_with_filter": "Naiwny 2 — z filtracją",
    "ideal_point": "Punkt idealny",
    "kung_luccio_preparata": "Kung / Luccio & Preparata",
}

DIST_LABELS: dict[DistributionName, str] = {
    "uniform": "Jednostajny",
    "normal": "Normalny (Gauss)",
    "exponential": "Wykładniczy",
    "poisson": "Poisson",
}


def _init_state() -> None:
    if "criteria" not in st.session_state:
        st.session_state.criteria = default_criteria(2)
    if "points" not in st.session_state:
        st.session_state.points = LAB_EXAMPLE_POINTS.copy()
    if "selected_crit" not in st.session_state:
        st.session_state.selected_crit = 0
    if "result" not in st.session_state:
        st.session_state.result = None
    if "benchmark_df" not in st.session_state:
        st.session_state.benchmark_df = None
    if "animation_fig" not in st.session_state:
        st.session_state.animation_fig = None
    if "seed" not in st.session_state:
        st.session_state.seed = 42


def _sync_criteria_count(k: int) -> None:
    criteria: list[Criterion] = list(st.session_state.criteria)
    while len(criteria) < k:
        criteria.append(Criterion(name=f"Kryterium {len(criteria) + 1}", sense=Sense.MIN))
    if len(criteria) > k:
        criteria = criteria[:k]
    # Uzupełnij brakujące automatyczne nazwy
    fixed: list[Criterion] = []
    for i, c in enumerate(criteria):
        name = c.name.strip() or f"Kryterium {i + 1}"
        fixed.append(Criterion(name=name, sense=c.sense))
    st.session_state.criteria = fixed
    st.session_state.selected_crit = min(st.session_state.selected_crit, k - 1)


def _ensure_points_match_criteria() -> None:
    pts = np.asarray(st.session_state.points, dtype=float)
    k = len(st.session_state.criteria)
    if pts.ndim != 2:
        pts = LAB_EXAMPLE_POINTS.copy()
    if pts.shape[1] != k:
        # Przy zmianie liczby kryteriów generujemy nowy zestaw.
        pts = generate_points(max(len(pts), 12), k, "uniform", seed=st.session_state.seed)
    st.session_state.points = pts


_init_state()

# ── Sidebar (zwijany natywnym przełącznikiem Streamlit) ──────────────────────
with st.sidebar:
    st.markdown("## Ustawienia")
    st.caption(f"OWD v{__version__} · panel można schować strzałką u góry")

    # —— Kryteria ——
    st.markdown("### 1. Kryteria")
    st.caption(
        "Dodawanie, usuwanie i edycja kryteriów. Nazwy uzupełniają się automatycznie. "
        "Kierunek Min/Max jest niezależny dla każdego kryterium."
    )

    n_crit = st.number_input(
        "Liczba kryteriów",
        min_value=1,
        max_value=8,
        value=len(st.session_state.criteria),
        step=1,
    )
    _sync_criteria_count(int(n_crit))

    names = [c.name for c in st.session_state.criteria]
    selected_name = st.selectbox(
        "Wybierz kryterium do edycji",
        options=names,
        index=st.session_state.selected_crit,
    )
    st.session_state.selected_crit = names.index(selected_name)
    idx = st.session_state.selected_crit
    current = st.session_state.criteria[idx]

    new_name = st.text_input("Nazwa", value=current.name, key=f"crit_name_{idx}")
    sense_label = st.radio(
        "Kierunek",
        options=["min", "max"],
        format_func=lambda s: "Minimalizacja" if s == "min" else "Maksymalizacja",
        index=0 if current.sense is Sense.MIN else 1,
        horizontal=True,
        key=f"crit_sense_{idx}",
    )
    updated = list(st.session_state.criteria)
    updated[idx] = Criterion(
        name=new_name.strip() or f"Kryterium {idx + 1}",
        sense=Sense.MIN if sense_label == "min" else Sense.MAX,
    )
    st.session_state.criteria = updated

    col_add, col_del = st.columns(2)
    with col_add:
        if st.button("＋ Dodaj", use_container_width=True) and len(st.session_state.criteria) < 8:
            st.session_state.criteria = [
                *st.session_state.criteria,
                Criterion(name=f"Kryterium {len(st.session_state.criteria) + 1}", sense=Sense.MIN),
            ]
            st.rerun()
    with col_del:
        if st.button("－ Usuń", use_container_width=True) and len(st.session_state.criteria) > 1:
            st.session_state.criteria = [
                c for i, c in enumerate(st.session_state.criteria) if i != idx
            ]
            st.session_state.selected_crit = 0
            st.rerun()

    st.divider()

    # —— Generowanie danych ——
    st.markdown("### 2. Generowanie zbioru")
    st.caption(
        "Losowy zbiór o zadanym rozkładzie i liczebności. "
        "Przedział / parametry zależą od rozkładu. Sortuj po wybranym kryterium."
    )

    dist_key = st.selectbox(
        "Rozkład",
        options=list(DIST_LABELS.keys()),
        format_func=lambda d: DIST_LABELS[d],
    )
    n_points = st.number_input("Liczba punktów", min_value=1, max_value=5000, value=25, step=1)
    seed = st.number_input(
        "Ziarno (seed)",
        min_value=0,
        max_value=10_000_000,
        value=int(st.session_state.seed),
    )
    st.session_state.seed = int(seed)

    gen_kwargs: dict = {"seed": int(seed)}
    if dist_key == "uniform":
        c1, c2 = st.columns(2)
        gen_kwargs["low"] = c1.number_input("Dolny zakres", value=0.0)
        gen_kwargs["high"] = c2.number_input("Górny zakres", value=2.0)
    elif dist_key == "normal":
        c1, c2 = st.columns(2)
        gen_kwargs["loc"] = c1.number_input("Średnia (loc)", value=1.0)
        gen_kwargs["scale"] = c2.number_input("Odch. std (scale)", value=1.0, min_value=0.01)
    elif dist_key == "exponential":
        gen_kwargs["scale"] = st.number_input("Skala (scale)", value=1.0, min_value=0.01)
    else:
        gen_kwargs["lam"] = st.number_input("λ (lam)", value=2.0, min_value=0.01)

    if st.button("Generuj zbiór", type="primary", use_container_width=True):
        k = len(st.session_state.criteria)
        st.session_state.points = generate_points(int(n_points), k, dist_key, **gen_kwargs)
        st.session_state.result = None
        st.session_state.benchmark_df = None
        st.session_state.animation_fig = None
        st.success(f"Wygenerowano {n_points} punktów × {k} kryteriów.")

    sort_crit = st.selectbox(
        "Sortuj według kryterium",
        options=list(range(len(st.session_state.criteria))),
        format_func=lambda i: st.session_state.criteria[i].name,
    )
    if st.button("Sortuj", use_container_width=True):
        pts = np.asarray(st.session_state.points, dtype=float)
        order = np.argsort(pts[:, int(sort_crit)], kind="mergesort")
        st.session_state.points = pts[order]
        st.session_state.result = None

    st.divider()

    # —— Dane / plik ——
    st.markdown("### 3. Dane wejściowe")
    st.caption("Podgląd zbioru, wczytanie z pliku CSV lub przykład z instrukcji.")

    if st.button("Załaduj przykład z labów (2D)", use_container_width=True):
        st.session_state.criteria = default_criteria(2)
        st.session_state.criteria = [
            Criterion("Kryterium 1", Sense.MIN),
            Criterion("Kryterium 2", Sense.MIN),
        ]
        st.session_state.points = LAB_EXAMPLE_POINTS.copy()
        st.session_state.result = None
        st.rerun()

    upload = st.file_uploader("Wczytaj CSV / Excel", type=["csv", "xlsx", "xls"])
    if upload is not None:
        try:
            if upload.name.lower().endswith(".csv"):
                df_up = pd.read_csv(upload)
            else:
                df_up = pd.read_excel(upload)
            numeric = df_up.select_dtypes(include=[np.number])
            if numeric.shape[1] < 1:
                st.error("Plik nie zawiera kolumn numerycznych.")
            else:
                st.session_state.points = numeric.to_numpy(dtype=float)
                st.session_state.criteria = [
                    Criterion(name=str(col), sense=Sense.MIN) for col in numeric.columns
                ]
                st.session_state.result = None
                st.success(f"Wczytano {numeric.shape[0]} × {numeric.shape[1]}.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Nie udało się wczytać pliku: {exc}")

    _ensure_points_match_criteria()
    preview = points_to_dataframe(st.session_state.points, st.session_state.criteria)
    st.dataframe(preview.head(30), use_container_width=True, height=220)
    st.caption(f"Pokazano {min(30, len(preview))} / {len(preview)} wierszy.")

    st.divider()

    # —— Algorytm ——
    st.markdown("### 4. Algorytm i wynik")
    st.caption(
        "Dwa warianty naiwne, punkt idealny oraz Kung–Luccio–Preparata. "
        "Rozwiąż · Benchmark · animacja · eksport XLS."
    )

    algo_key: AlgorithmName = st.selectbox(
        "Algorytm",
        options=list(ALGORITHM_LABELS.keys()),
        format_func=lambda a: ALGORITHM_LABELS[a],
        index=3,
    )

    c_solve, c_bench = st.columns(2)
    solve_clicked = c_solve.button("Rozwiąż", type="primary", use_container_width=True)
    bench_clicked = c_bench.button("Benchmark", use_container_width=True)

    c_anim, c_clear = st.columns(2)
    anim_clicked = c_anim.button("Renderuj animację", use_container_width=True)
    if c_clear.button("Wyczyść wynik", use_container_width=True):
        st.session_state.result = None
        st.session_state.benchmark_df = None
        st.session_state.animation_fig = None

    if solve_clicked:
        with st.spinner("Trwa optymalizacja…"):
            st.session_state.result = find_nondominated(
                st.session_state.points,
                algo_key,
                criteria=st.session_state.criteria,
            )
            st.session_state.animation_fig = None

    if bench_clicked:
        with st.spinner("Benchmark wszystkich algorytmów…"):
            st.session_state.benchmark_df = run_benchmark(
                st.session_state.points,
                st.session_state.criteria,
            )
            # Dodatkowo ustaw wynik wybranego algorytmu
            st.session_state.result = find_nondominated(
                st.session_state.points,
                algo_key,
                criteria=st.session_state.criteria,
            )

    if anim_clicked:
        with st.spinner("Budowa klatek animacji…"):
            if st.session_state.result is None:
                st.session_state.result = find_nondominated(
                    st.session_state.points,
                    algo_key,
                    criteria=st.session_state.criteria,
                )
            frames_nd, frames_cmp = generate_filter_frames(
                st.session_state.points,
                st.session_state.criteria,
            )
            st.session_state.animation_fig = build_animation_figure(
                st.session_state.points,
                st.session_state.criteria,
                frames_nd,
                frames_cmp,
                title=f"Animacja · {ALGORITHM_LABELS[algo_key]}",
            )

# ── Main panel ───────────────────────────────────────────────────────────────
st.title("OWD — zbiór punktów niezdominowanych")
st.caption("Panel główny: metryki, znalezione punkty i wykres. Ustawienia po lewej (sidebar).")

criteria: list[Criterion] = st.session_state.criteria
points = np.asarray(st.session_state.points, dtype=float)
result = st.session_state.result

# Kryteria — skrót
chips = " · ".join(f"**{c.name}** ({'min' if c.sense is Sense.MIN else 'max'})" for c in criteria)
st.markdown(f"Kryteria: {chips}")

if result is None:
    st.info("Skonfiguruj dane w panelu bocznym i naciśnij **Rozwiąż**, aby zobaczyć P(X).")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Punkty |X|", len(points))
    m2.metric("Kryteria", len(criteria))
    m3.metric("|P(X)|", "—")
    m4.metric("Porównania", "—")
    fig_preview = build_pareto_figure(points, criteria, [], title="Zbiór wejściowy X")
    st.plotly_chart(fig_preview, use_container_width=True)
    with st.expander("Podgląd danych wejściowych"):
        st.dataframe(points_to_dataframe(points, criteria), use_container_width=True)
else:
    stats = result.stats
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Punkty |X|", len(points))
    m2.metric("|P(X)|", len(result.nondominated))
    m3.metric("Liczba porównań", stats.point_comparisons)
    m4.metric("Czas wykonania", f"{stats.elapsed_seconds * 1000:.4f} ms")

    st.caption(
        f"Algorytm: **{ALGORITHM_LABELS.get(result.algorithm, result.algorithm)}** · "
        f"porównania współrzędnych: {stats.coordinate_comparisons}"
    )

    tab_plot, tab_points, tab_anim, tab_bench, tab_export = st.tabs(
        ["Wykres", "Punkty P(X)", "Animacja", "Benchmark", "Eksport XLS"]
    )

    with tab_plot:
        fig = build_pareto_figure(
            points,
            criteria,
            result.nondominated_indices,
            title=f"{ALGORITHM_LABELS.get(result.algorithm, result.algorithm)} · |X|={len(points)}",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_points:
        _, nd_df, stats_df = result_frames(points, criteria, result)
        st.subheader("Zbiór niezdominowany")
        st.dataframe(nd_df, use_container_width=True)
        st.subheader("Metryki")
        st.dataframe(stats_df, use_container_width=True)
        with st.expander("Wszystkie punkty (flaga niezdominowany)"):
            all_df, _, _ = result_frames(points, criteria, result)
            st.dataframe(all_df, use_container_width=True)

    with tab_anim:
        if st.session_state.animation_fig is None:
            st.warning("Naciśnij **Renderuj animację** w panelu bocznym, aby zbudować podgląd.")
        else:
            st.plotly_chart(st.session_state.animation_fig, use_container_width=True)
            # Eksport HTML animacji
            html_buf = io.StringIO()
            st.session_state.animation_fig.write_html(html_buf, include_plotlyjs="cdn")
            st.download_button(
                "Zapisz animację (HTML)",
                data=html_buf.getvalue(),
                file_name="owd_animacja.html",
                mime="text/html",
            )

    with tab_bench:
        bench = st.session_state.benchmark_df
        if bench is None:
            st.warning("Naciśnij **Benchmark**, aby porównać algorytmy na tych samych danych.")
        else:
            st.dataframe(bench, use_container_width=True)
            st.plotly_chart(build_benchmark_figure(bench), use_container_width=True)

    with tab_export:
        xls_bytes = to_excel_bytes(points, criteria, result, st.session_state.benchmark_df)
        st.download_button(
            "Pobierz wyniki (XLS / XLSX)",
            data=xls_bytes,
            file_name="owd_wyniki.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
        csv_nd = nd_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Pobierz P(X) jako CSV",
            data=csv_nd,
            file_name="owd_pareto.csv",
            mime="text/csv",
        )
