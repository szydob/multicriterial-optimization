# Lab 2 — notatki

Źródło: *Ćwiczenie 2 (Optymalizacja dyskretna). Instrukcja, Przykład OWD*.

## Algorytmy do zaimplementowania

| ID | Opis | Status |
|----|------|--------|
| `naive_no_filter` | Naiwny bez filtracji (pseudokod str. 2) | done |
| `naive_with_filter` | Naiwny z filtracją zdominowanych (str. 3) | done |
| `ideal_point` | Punkt idealny + sort odległości (str. 4) | done |
| `kung_luccio_preparata` | Kung–Luccio–Preparata | done |

## Przykład z instrukcji

```
X = {(5,5), (3,6), (4,4), (5,3), (3,3), (1,8),
     (3,4), (4,5), (3,10), (6,6), (4,1), (3,5)}
```

Oczekiwane niezdominowane (min–min): **(3,3), (1,8), (4,1)**.

Uwaga z instrukcji: przedwczesne usunięcie (3,3) bez filtracji może błędnie uznać (3,6) za niezdominowany.

## Metryki benchmarku

- liczba porównań **punktów**
- liczba porównań **współrzędnych**
- czas obliczeń [s]
- (później) średnia ± odch. std. po K≥50 losowych zbiorach

## Rozkłady (część C instrukcji)

- jednostajny na [0, 2] — wszyscy
- do wyboru: Gauss(1,1), ekspencjalny(1), Poisson(2)

## GUI (Streamlit zamiast Matlab)

Panele: kryteria → dane → algorytm/wynik → benchmark → eksport.
