"""Porównanie algorytmów na tym samym zbiorze X."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from owd.algorithms import ALGORITHMS, AlgorithmName
from owd.types import Criterion, Points


def run_benchmark(
    points: Points,
    criteria: list[Criterion],
    algorithms: list[AlgorithmName] | None = None,
) -> pd.DataFrame:
    """
    Uruchamia wybrane algorytmy i zbiera metryki.

    Zwraca point_comparisons, coordinate_comparisons, elapsed_seconds, |P(X)|.
    Błędy / brak implementacji oznaczane są w kolumnie status.
    """
    if algorithms is None:
        algorithms = list(ALGORITHMS.keys())

    rows: list[dict] = []
    for name in algorithms:
        row: dict = {"algorithm": name, "status": "ok"}
        try:
            result = ALGORITHMS[name](points, criteria=criteria)
            row.update(asdict(result.stats))
            row["n_nondominated"] = int(len(result.nondominated))
        except NotImplementedError as exc:
            row["status"] = "pending"
            row["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 — benchmark ma zebrać błędy, nie padać
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    return pd.DataFrame(rows)
