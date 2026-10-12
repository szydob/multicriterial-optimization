"""Eksport wyników do CSV / Excel."""

from __future__ import annotations

from io import BytesIO

import pandas as pd

from owd.types import Criterion, ParetoResult, Points


def points_to_dataframe(points: Points, criteria: list[Criterion]) -> pd.DataFrame:
    columns = [c.name for c in criteria]
    return pd.DataFrame(points, columns=columns)


def result_frames(
    points: Points,
    criteria: list[Criterion],
    result: ParetoResult,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Zwraca (wszystkie punkty z flagą, niezdominowane, metryki)."""
    all_df = points_to_dataframe(points, criteria)
    nd_mask = [False] * len(all_df)
    for idx in result.nondominated_indices:
        nd_mask[int(idx)] = True
    all_df.insert(0, "indeks", range(len(all_df)))
    all_df["niezdominowany"] = nd_mask

    nd_df = points_to_dataframe(result.nondominated, criteria)
    nd_df.insert(0, "indeks_wejsciowy", result.nondominated_indices.astype(int))

    stats_df = pd.DataFrame(
        [
            {
                "algorytm": result.algorithm,
                "liczba_porownan_punktow": result.stats.point_comparisons,
                "liczba_porownan_wspolrzednych": result.stats.coordinate_comparisons,
                "czas_s": result.stats.elapsed_seconds,
                "czas_ms": result.stats.elapsed_seconds * 1000.0,
                "liczba_niezdominowanych": len(result.nondominated),
                "liczba_punktow": len(points),
            }
        ]
    )
    return all_df, nd_df, stats_df


def to_excel_bytes(
    points: Points,
    criteria: list[Criterion],
    result: ParetoResult,
    benchmark: pd.DataFrame | None = None,
) -> bytes:
    """Buduje plik .xlsx w pamięci (Excel / „xls” w GUI)."""
    all_df, nd_df, stats_df = result_frames(points, criteria, result)
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        all_df.to_excel(writer, sheet_name="punkty", index=False)
        nd_df.to_excel(writer, sheet_name="niezdominowane", index=False)
        stats_df.to_excel(writer, sheet_name="metryki", index=False)
        if benchmark is not None and not benchmark.empty:
            benchmark.to_excel(writer, sheet_name="benchmark", index=False)
    return buffer.getvalue()
