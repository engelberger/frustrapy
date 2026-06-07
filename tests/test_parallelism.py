"""Regression tests for the parallel execution paths.

These pin the structural guarantees of the parallelism without asserting any
wall-clock speedup:

* Flatten - a single-residue scan dispatches all ``N x 20`` mutation tasks through
  ONE persistent pool, and on a > 20-core box the resolved worker count exceeds 20
  (the per-residue 20-way ceiling is structurally gone).
* Cap - the pool never oversubscribes ``min(requested, cores, tasks)``.
* Outer-axis parallelism - a parallel ``dir_frustration`` batch produces
  byte-identical tables to the serial loop and one result per PDB, with a shared
  core budget keeping nested pools below ``cores`` squared.
* The library imports with no ``SyntaxWarning`` (raw-string separators).

The functional tests run on the 1CRN fixture and require the project venv on PATH
(a calculation spawns a bare ``python3`` subprocess), like the rest of the suite.
"""

import filecmp
import os
import warnings

import pytest

from frustrapy.analysis.mutations import (
    AMINO_ACIDS,
    _resolve_pool_size,
    mutate_res_scan_parallel,
)


# ----------------------------------------------------------------------------
# pure, fast, no LAMMPS: the 20-way ceiling is gone.
# ----------------------------------------------------------------------------

def test_amino_acid_order_is_canonical():
    """The fixed merge order must be exactly the 20 canonical identities."""
    assert len(AMINO_ACIDS) == 20
    assert AMINO_ACIDS[0] == "LEU" and AMINO_ACIDS[-1] == "CYS"
    assert len(set(AMINO_ACIDS)) == 20


@pytest.mark.parametrize(
    "requested,n_tasks,cores,expected",
    [
        # 5-residue scan (100 tasks) on a 32-core box, no explicit request:
        # > 20 workers => the per-residue 20-way ceiling is structurally gone.
        (None, 100, 32, 32),
        # Single residue (20 tasks) on a 32-core box: capped at the task count.
        (None, 20, 32, 20),
        # Explicit request honored but capped by cores.
        (8, 100, 32, 8),
        (64, 100, 32, 32),
        # Capped by task count when fewer tasks than cores.
        (None, 5, 32, 5),
        # Never zero / negative.
        (None, 0, 14, 1),
    ],
)
def test_resolve_pool_size(requested, n_tasks, cores, expected):
    assert _resolve_pool_size(requested, n_tasks, cores) == expected


def test_ceiling_is_structurally_gone():
    """On a hypothetical 64-core box a 10-residue scan dispatches > 20 workers."""
    n_tasks = 10 * len(AMINO_ACIDS)  # 200
    assert _resolve_pool_size(None, n_tasks, 64) > 20


# ----------------------------------------------------------------------------
# functional: one persistent pool over a multi-residue grid produces
# the same per-residue tables as N independent single-residue scans.
# ----------------------------------------------------------------------------

def _prepare_pdb(crn_pdb, tmp_path, residues):
    """Run the configurational-independent prep to get a Pdb object whose
    ``job_dir`` is set up, then return it for direct mutation scanning."""
    import frustrapy

    rd = str(tmp_path / "prep")
    pdb, _plots, _dens, _sr = frustrapy.calculate_frustration(
        pdb_file=str(crn_pdb),
        mode="singleresidue",
        residues={"A": residues},
        results_dir=rd,
        graphics=False,
        visualization=False,
        n_cpus=2,
    )
    return pdb


@pytest.mark.slow
def test_flatten_scan_matches_per_residue(crn_pdb, tmp_path):
    """one pool over (res1+res2)×20 == two separate single-residue scans.

    The merged per-residue output tables must be byte-identical whether produced by
    the flattened multi-target scan or by two independent single-target scans.
    """
    import frustrapy

    residues = [1, 2]

    # Reference: two independent single-residue scans (graphics on => parallel path).
    ref_dir = str(tmp_path / "ref")
    frustrapy.calculate_frustration(
        pdb_file=str(crn_pdb), mode="singleresidue",
        residues={"A": residues}, results_dir=ref_dir,
        graphics=True, visualization=False, n_cpus=2,
    )
    ref_mut = os.path.join(ref_dir, "1crn.done", "MutationsData")

    # Flattened scan invoked directly on a freshly-prepared Pdb.
    pdb = _prepare_pdb(crn_pdb, tmp_path, residues)
    mutate_res_scan_parallel(
        pdb, [(r, "A") for r in residues], n_cpus=4,
    )
    scan_mut = os.path.join(pdb.job_dir, "MutationsData")

    for r in residues:
        name = f"singleresidue_Res{r}_threading_A.txt"
        assert filecmp.cmp(
            os.path.join(ref_mut, name),
            os.path.join(scan_mut, name),
            shallow=False,
        ), f"flattened scan diverged from per-residue scan for residue {r}"

    # One persistent scan exposes its metrics with the multi-residue task count.
    assert pdb.MutationAnalysis["n_residues"] == len(residues)
    assert pdb.MutationAnalysis["n_tasks"] == len(residues) * len(AMINO_ACIDS)


# ----------------------------------------------------------------------------
# functional: parallel batch == serial batch, byte for byte.
# ----------------------------------------------------------------------------

@pytest.mark.slow
def test_dir_frustration_parallel_matches_serial(crn_pdb, tmp_path):
    """A parallel (n_procs>1) configurational batch matches the serial loop."""
    from frustrapy.analysis.frustration import dir_frustration

    batch = tmp_path / "batch"
    batch.mkdir()
    names = ["a", "b"]
    for n in names:
        with open(crn_pdb, "rb") as _src:
            (batch / f"prot{n}.pdb").write_bytes(_src.read())

    serial_dir = str(tmp_path / "serial")
    par_dir = str(tmp_path / "par")

    plots_s, _ = dir_frustration(
        pdbs_dir=str(batch), mode="configurational", graphics=False,
        visualization=False, results_dir=serial_dir, n_procs=None,
    )
    plots_p, _ = dir_frustration(
        pdbs_dir=str(batch), mode="configurational", graphics=False,
        visualization=False, results_dir=par_dir, n_procs=2,
    )

    # One result per PDB, both ways.
    assert sorted(plots_s) == sorted(plots_p) == ["prota", "protb"]

    for n in names:
        rel = os.path.join(f"prot{n}.done", "FrustrationData", f"prot{n}.pdb_configurational")
        assert filecmp.cmp(
            os.path.join(serial_dir, rel),
            os.path.join(par_dir, rel),
            shallow=False,
        ), f"parallel batch diverged from serial for prot{n}"


# ----------------------------------------------------------------------------
# no SyntaxWarning from raw-string separators.
# ----------------------------------------------------------------------------

def test_no_syntax_warning_on_import():
    import importlib

    modules = [
        "frustrapy.analysis.mutations",
        "frustrapy.analysis.frustration_calculator",
        "frustrapy.visualization.plots",
        "frustrapy.utils.helpers",
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        for m in modules:
            mod = importlib.import_module(m)
            importlib.reload(mod)
