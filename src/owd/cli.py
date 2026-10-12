"""Prosty CLI do szybkiego smoke-testu środowiska."""

from __future__ import annotations

from owd import __version__
from owd.generators import LAB_EXAMPLE_POINTS, generate_points


def main() -> None:
    sample = generate_points(5, 2, "uniform", seed=0)
    print(f"owd {__version__}")
    print(f"lab example shape: {LAB_EXAMPLE_POINTS.shape}")
    print(f"random sample (5x2, seed=0):\n{sample}")
    print("OK — środowisko działa. Algorytmy: stuby (NotImplementedError).")


if __name__ == "__main__":
    main()
