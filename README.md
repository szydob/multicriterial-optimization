# OWD — Optymalizacja wielokryterialna dyskretna

Aplikacja (Streamlit) i biblioteka do wyznaczania zbioru punktów niezdominowanych \(P(X)\) oraz benchmarku algorytmów optymalizacji dyskretnej wielokryterialnej.

Kurs: *Optymalizacja wielokryterialna* (AGH, LAiWD KAiR), ćwiczenie 2 — na podstawie instrukcji OWD (algorytmy z/bez filtracji, punkt idealny, Kung–Luccio–Preparata).

## Stack

- **UV** — zarządzanie zależnościami i środowiskiem
- **Python 3.12**
- **NumPy / Pandas** — dane i obliczenia
- **Streamlit + Plotly** — GUI, wykresy, (później) animacje

## Setup

```bash
cd /Users/szydob/Studia/sem_9/optymalizacja_wielokryterialna/owd
uv sync
```

Smoke-test CLI:

```bash
uv run owd
```

Uruchomienie Streamlit:

```bash
uv run streamlit run app/streamlit_app.py
```

Testy:

```bash
uv run pytest
```

## Layout

```
app/                 Streamlit UI (sidebar + panel wyników)
data/examples/       Przykładowe zestawy punktów
docs/                Notatki z labów / decyzje
outputs/             Wyniki, eksporty (gitignored)
src/owd/
  algorithms/        Naiwne, punkt idealny, Kung–Luccio–Preparata
  animation.py       Klatki animacji budowy P(X)
  dominance.py       Porównania dominacji
  export.py          Eksport XLSX / CSV
  generators.py      Generowanie zbiorów losowych
  metrics.py         Licznik porównań + czas
  types.py           Modele danych (kryteria, wynik)
  visualization.py   Wykresy Plotly (2D / 3D)
tests/
```

## Funkcje GUI

1. **Sidebar (zwijany)** — kryteria, generator, sortowanie, wczytywanie pliku, wybór algorytmu
2. **Panel główny** — metryki, P(X), wykres, animacja, benchmark, eksport XLS
3. Algorytmy: naiwny bez/z filtracją, punkt idealny, Kung–Luccio–Preparata

