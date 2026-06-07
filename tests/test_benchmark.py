"""Smoke tests for the merged ``frustrapy.benchmark`` module.

These lock in the dev_benchmark merge EXIT GATE: ``run_benchmark`` is importable
from a clean install and produces a results DataFrame with the expected columns.
The functional run uses ``cpu_list=[1]`` / ``repeats=1`` on the 1CRN fixture so it
exercises the real parallel mutation path (``mutate_res_parallel``) cheaply.

NOTE on PATH: like the rest of the suite, the functional test
must run with the project's virtualenv on PATH (run ``pytest`` from an activated
venv) or the LAMMPS prep fails with ``ModuleNotFoundError: No module named 'Bio'``.
"""

import warnings

import pytest


def test_benchmark_public_api_importable():
    """``run_benchmark`` and the plotting helpers import from a clean tree."""
    from frustrapy.benchmark import (  # noqa: F401
        run_benchmark,
        get_raw_benchmark_data,
        plot_speedup_linear,
        plot_efficiency,
        plot_execution_time,
    )


@pytest.mark.slow
def test_run_benchmark_minimal(crn_pdb, tmp_path):
    """A minimal single-CPU benchmark returns a DataFrame with the documented columns."""
    from frustrapy.benchmark import run_benchmark

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = run_benchmark(
            pdb_file=str(crn_pdb),
            chain="A",
            residues=10,
            cpu_list=[1],
            results_dir=str(tmp_path / "bench"),
            repeats=1,
        )

    # One row per (residue x cpu count) configuration.
    assert len(df) == 1
    for col in ("residue", "n_cpus", "time_s", "speedup", "efficiency", "pdb_id"):
        assert col in df.columns, f"missing benchmark column: {col}"
    assert int(df.iloc[0]["n_cpus"]) == 1
    # Single-CPU baseline: speedup and efficiency are 1.0 by definition.
    assert df.iloc[0]["speedup"] == pytest.approx(1.0)
