"""Executable module for one isolated benchmark cell."""

from __future__ import annotations

import sys

from dsa.benchmark import benchmark_cell_worker_main


def main() -> int:
    if sys.argv[1:] != ["--cell-worker"]:
        return 2
    return benchmark_cell_worker_main()


if __name__ == "__main__":
    raise SystemExit(main())
