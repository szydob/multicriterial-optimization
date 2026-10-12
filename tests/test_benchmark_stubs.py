from __future__ import annotations

from owd.algorithms.base import default_criteria
from owd.benchmark import run_benchmark
from owd.generators import LAB_EXAMPLE_POINTS


def test_benchmark_all_algorithms_ok() -> None:
    df = run_benchmark(LAB_EXAMPLE_POINTS, default_criteria(2))
    assert df.set_index("algorithm")["status"].to_dict() == {
        "naive_no_filter": "ok",
        "naive_with_filter": "ok",
        "ideal_point": "ok",
        "kung_luccio_preparata": "ok",
    }
    assert len(df) == 4
    assert (df["n_nondominated"] == 3).all()
