"""Cross-backend single-protein frustration benchmark.

A small, reusable harness that runs one protein through every frustration backend
available on the machine (the ``lammps`` reference orchestrator, the native CPU core
serial and multicore, and the native CUDA path when built) and reports wall time,
speedup, and a parity check (FrstIndex Spearman vs ``lammps``). It degrades
gracefully: absent backends are skipped, never an error.

Two entry points:

* :func:`run_cross_backend_benchmark` -- end-to-end ``calculate_frustration`` per
  backend; the user-facing table and the parity reference.
* :func:`benchmark_core_scaling` -- times the C++ reduction directly at a range of
  thread counts (isolated from the constant LAMMPS prep), for scaling studies and
  the regression timing gate.

Runnable as a CLI: ``python native/bench/cross_backend.py --pdb tests/data/1crn.pdb``.
"""

from __future__ import annotations

from .cross_backend import (
    BackendRow,
    CoreScalingRow,
    available_backend_labels,
    benchmark_core_scaling,
    format_table,
    run_cross_backend_benchmark,
)

__all__ = [
    "BackendRow",
    "CoreScalingRow",
    "available_backend_labels",
    "benchmark_core_scaling",
    "format_table",
    "run_cross_backend_benchmark",
]
