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

Uruchomienie stubu Streamlit:

```bash
uv run streamlit run app/streamlit_app.py
```

Testy (kontrakty / stuby):

```bash
uv run pytest
```

Opcjonalnie wykresy (Plotly):

```bash
uv sync --extra viz
```
## Layout

```
app/                 Streamlit UI
data/examples/       Przykładowe zestawy punktów
docs/                Notatki z labów / decyzje
notebooks/           Eksperymenty (opcjonalnie)
outputs/             Wyniki, eksporty (gitignored)
src/owd/
  algorithms/        Algorytmy P(X) — na razie stuby
  dominance.py       Porównania dominacji
  generators.py      Generowanie zbiorów losowych
  metrics.py         Licznik porównań + czas
  types.py           Modele danych (kryteria, wynik)
tests/
```

## Plan funkcji (zgodny z labem)

1. Konfiguracja kryteriów (min/max niezależnie)
2. Generowanie / wczytanie zbioru \(X\) (jednostajny, Gauss, wykładniczy, Poisson)
3. Algorytmy:
   - naiwny bez filtracji
   - naiwny z filtracją punktów zdominowanych
   - punkt idealny (sortowanie odległości + filtracja)
   - Kung–Luccio–Preparata
4. Benchmark: liczba porównań punktów / współrzędnych, czas
5. Wizualizacja frontu Pareto (2D / 3D, później animacja)

## Status

Szkielet projektu + kontrakty API. Implementacje algorytmów i pełne GUI — w kolejnych krokach.
