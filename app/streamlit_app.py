"""Stub GUI Streamlit — pełna aplikacja w kolejnych krokach."""

from __future__ import annotations

import streamlit as st

from owd import __version__
from owd.algorithms import ALGORITHMS
from owd.generators import LAB_EXAMPLE_POINTS, generate_points

st.set_page_config(page_title="OWD", layout="wide")

st.title("OWD — optymalizacja wielokryterialna dyskretna")
st.caption(f"wersja {__version__} · szkielet pod ćwiczenie 2 (P(X), benchmark, wizualizacja)")

st.info(
    "Na razie baza projektu. Algorytmy i pełne GUI (kryteria, generatory, wykresy, "
    "benchmark) dołożymy w kolejnych iteracjach."
)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Przykład z instrukcji")
    st.dataframe(LAB_EXAMPLE_POINTS, use_container_width=True)

with col_b:
    st.subheader("Losowa próbka")
    n = st.slider("n", 5, 200, 20)
    k = st.slider("k (kryteria)", 2, 5, 2)
    dist = st.selectbox("rozkład", ["uniform", "normal", "exponential", "poisson"])
    sample = generate_points(n, k, dist, seed=42)  # type: ignore[arg-type]
    st.dataframe(sample, use_container_width=True)

st.subheader("Algorytmy (status)")
for name in ALGORITHMS:
    st.write(f"- `{name}` — stub (`NotImplementedError`)")

st.subheader("Plan paneli (jak w Matlab GUI z labów)")
st.markdown(
    """
1. **Kryteria** — nazwy + kierunek min/max  
2. **Dane** — generator / wczytanie / sortowanie  
3. **Rozwiąż** — wybór algorytmu + wykres P(X)  
4. **Benchmark** — porównania i czas  
5. **Eksport** — CSV / XLS  
"""
)
