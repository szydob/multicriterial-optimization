from __future__ import annotations

from owd.algorithms.base import default_criteria
from owd.benchmark import run_benchmark
from owd.generators import LAB_EXAMPLE_POINTS


def test_benchmark_reports_pending_stubs() -> None:
    df = run_benchmark(LAB_EXAMPLE_POINTS, default_criteria(2))
    assert df.set_index("algorithm")["status"].to_dict() == {
        "naive_no_filter": "ok",
        "naive_with_filter": "ok",
        "ideal_point": "pending",
        "kung_luccio_preparata": "pending",
    }
    assert len(df) == 4
