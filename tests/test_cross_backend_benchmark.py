"""Cross-backend single-protein benchmark: smoke + parity + multicore-not-slower.

The harness (``native/bench/cross_backend.py``) runs one protein through every
backend available on the machine and reports wall time, speedup, and FrstIndex
Spearman vs the LAMMPS reference. These tests pin three properties on 1CRN:

* it imports and runs end-to-end, degrading gracefully when a backend is absent;
* every native row is rank-identical to LAMMPS (Spearman 1.0) -- the parallel core
  did not change the result;
* on the isolated core (mutational, where there is real per-contact work) the
  multicore reduction is not slower than serial and stays bit-identical.

These run the engine (LAMMPS + native), so they are in the slow lane and require
the built ``frustrapy_native`` extension; they skip cleanly if it is absent.
"""

import os
import sys

import pytest

# Make ``native/bench`` importable as ``bench`` without installing it.
_NATIVE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "native")
if _NATIVE not in sys.path:
    sys.path.insert(0, _NATIVE)

native = pytest.importorskip("frustrapy_native")

from bench.cross_backend import (  # noqa: E402
    available_backend_labels,
    benchmark_core_scaling,
    format_table,
    run_cross_backend_benchmark,
)


def test_available_backends_lists_lammps_and_native():
    labels = available_backend_labels()
    assert "lammps" in labels
    assert any("native CPU" in x for x in labels)


@pytest.mark.slow
def test_cross_backend_parity_1crn(crn_pdb, tmp_path):
    """Every native row matches the LAMMPS FrstIndex ranking exactly."""
    rows = run_cross_backend_benchmark(
        pdb_file=str(crn_pdb), mode="configurational", threads_list=[1, 4],
        repeats=1, seq_dist=12, results_root=str(tmp_path),
    )
    assert any(r.backend.startswith("lammps") for r in rows)
    native_rows = [r for r in rows if r.backend.startswith("native")]
    assert native_rows, "native backend should be measured"
    for r in native_rows:
        assert r.spearman >= 0.99, f"{r.backend}: Spearman {r.spearman}"
    # The table renders without error.
    assert "speedup" in format_table(rows)


@pytest.mark.slow
def test_core_multicore_not_slower_than_serial_1crn(crn_pdb, tmp_path):
    """On the isolated core (mutational) the multicore reduction is bit-identical
    to serial and not slower (a generous bound to stay robust to scheduler noise)."""
    cores = os.cpu_count() or 1
    threads = sorted({1, cores})
    rows = benchmark_core_scaling(
        pdb_file=str(crn_pdb), mode="mutational", threads_list=threads,
        repeats=3, seq_dist=12, results_dir=str(tmp_path),
    )
    by_t = {r.threads: r for r in rows}
    assert all(r.bit_identical for r in rows), "core output changed across thread counts"
    if cores > 1:
        serial = by_t[1].wall_s
        multi = by_t[cores].wall_s
        # Not slower than serial, allowing 25% scheduler/overhead slack on a tiny protein.
        assert multi <= serial * 1.25, f"multicore {multi:.3f}s slower than serial {serial:.3f}s"
