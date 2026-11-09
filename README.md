# OWD — Optymalizacja wielokryterialna dyskretna

Aplikacja (Streamlit) i biblioteka do wyznaczania zbioru punktów niezdominowanych \(P(X)\) oraz skalaryzacji wielokryterialnej (Ćw. 3).

Kurs: *Optymalizacja wielokryterialna* (AGH, LAiWD KAiR):
- **Ćw. 2** — OWD: algorytmy z/bez filtracji, punkt idealny, Kung–Luccio–Preparata
- **Ćw. 3** — skalaryzacja: suma ważona, ε-ograniczenia, odległość ℓ_p od punktu dominującego; sweep parametrów odtwarza P(U,F) / FP(U)

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
app/
  streamlit_app.py   Przełącznik Ćw.2 / Ćw.3 + GUI lab 2
  cw3_page.py        GUI lab 3 (skalaryzacja + sweep Pareta)
data/examples/       Przykładowe zestawy punktów
docs/                Notatki z labów / decyzje
outputs/             Wyniki, eksporty (gitignored)
src/owd/
  algorithms/        Naiwne, punkt idealny, Kung–Luccio–Preparata
  scalarization/     Cw3: problem, S1/S2/S3, sweep, wykresy U/F
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

### Ćw. 2
1. **Sidebar** — kryteria, generator, sortowanie, wczytywanie pliku, wybór algorytmu
2. **Panel główny** — metryki, P(X), wykres, animacja, benchmark, eksport XLS
3. Algorytmy: naiwny bez/z filtracją, punkt idealny, Kung–Luccio–Preparata

### Ćw. 3
1. Przełącznik w sidebarze → **Ćw. 3 — skalaryzacja / zbiór Pareta**
2. Metody: suma ważona (S1), ε-ograniczenia (S2), odległość od punktu dominującego (S3, ℓ_p / Czebyszew)
3. Tryb **sweep** odtwarza dyskretny front P(U,F) / FP(U); tryb pojedynczy liczy jedno rozwiązanie
4. Wykresy: przestrzeń decyzji U=[1,4]² oraz przestrzeń celów F (spline przez punkty sweepu)

