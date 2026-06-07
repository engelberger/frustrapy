"""Shared pytest fixtures for the FrustraPy regression suite.

The empirical anchor for every golden assertion below is a native-Linux 1CRN
(crambin, 46 residues, single chain) configurational run: it returns a 4-tuple and
writes a 232-row x 14-column ``1crn.pdb_configurational`` table plus a
``tertiary_frustration.dat`` of 32,847 bytes. See ``.devcontainer/HANDOFF.md`` and
``docs/audit/empirical/EMPIRICAL_ANCHOR.md``.

NOTE on PATH (tech-debt P1-22): ``PdbCoords2Lammps.sh`` spawns a bare ``python3``
subprocess, so these tests must run with the project's virtualenv on PATH (i.e. run
``pytest`` from an activated venv), or the calculation fails mid-run with
``ModuleNotFoundError: No module named 'Bio'``.
"""

import os
import warnings

import pytest

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CRN_PDB = os.path.join(DATA_DIR, "1crn.pdb")


@pytest.fixture(scope="session")
def crn_pdb():
    """Absolute path to the committed 1CRN fixture."""
    assert os.path.exists(CRN_PDB), f"missing fixture: {CRN_PDB}"
    return CRN_PDB


def _run_mode(pdb_file, mode, results_dir):
    """Run a single-structure frustration calculation on the Linux graphics=False path."""
    import frustrapy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return frustrapy.calculate_frustration(
            pdb_file=pdb_file,
            mode=mode,
            results_dir=results_dir,
            graphics=False,
            visualization=False,
            debug="ERROR",
        )


@pytest.fixture(scope="session")
def crn_configurational(crn_pdb, tmp_path_factory):
    """Run 1CRN configurational once and share the result across the session.

    Returns a dict with the returned 4-tuple, the job dir, and the parsed-table path.
    """
    results_dir = str(tmp_path_factory.mktemp("crn_config"))
    result = _run_mode(crn_pdb, "configurational", results_dir)
    job_dir = os.path.join(results_dir, "1crn.done")
    table = os.path.join(job_dir, "FrustrationData", "1crn.pdb_configurational")
    return {
        "result": result,
        "results_dir": results_dir,
        "job_dir": job_dir,
        "table": table,
    }


@pytest.fixture
def run_mode(tmp_path):
    """Factory to run an arbitrary mode into a fresh per-test directory."""

    def _factory(pdb_file, mode):
        results_dir = str(tmp_path / mode)
        return _run_mode(pdb_file, mode, results_dir), results_dir

    return _factory
