"""End-to-end example runs on small inputs.

Each example exercises the real calculation path (LAMMPS subprocess + parsing) and
must finish well under 60 seconds, so a release cannot ship a broken pipeline:

  - a single small protein (1CRN, 46 residues) through the three frustration modes
    (configurational, mutational, singleresidue), each writing its output table;
  - a small protein family (3 alpha-globin chains) through the evolution pipeline
    (``analyze_family``), writing its information-content tables.

These are marked ``slow`` (they spawn the engine) and need the project venv on PATH,
like the rest of the functional suite.
"""

import os
import time
import warnings

import pytest

import frustrapy

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EVO_SMOKE_DIR = os.path.join(DATA_DIR, "evo_smoke")
MAX_SECONDS = 60.0


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["configurational", "mutational", "singleresidue"])
def test_example_single_protein_mode(crn_pdb, tmp_path, mode):
    """1CRN runs end to end in under 60s for each mode and writes a non-empty table."""
    results_dir = str(tmp_path / mode)
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdb, *_ = frustrapy.calculate_frustration(
            pdb_file=str(crn_pdb),
            mode=mode,
            results_dir=results_dir,
            graphics=False,
            visualization=False,
            debug="ERROR",
        )
    elapsed = time.time() - start

    table = os.path.join(pdb.job_dir, "FrustrationData", f"1crn.pdb_{mode}")
    assert os.path.isfile(table), f"{mode}: no output table at {table}"
    assert os.path.getsize(table) > 0, f"{mode}: output table is empty"
    assert elapsed < MAX_SECONDS, f"{mode} took {elapsed:.1f}s (limit {MAX_SECONDS:.0f}s)"


@pytest.mark.slow
def test_example_evolution_small_family(tmp_path):
    """A 3-chain alpha-globin family runs through analyze_family in under 60s and
    writes information-content tables."""
    results_dir = str(tmp_path / "evo")
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frustrapy.analyze_family(
            fasta_file=os.path.join(EVO_SMOKE_DIR, "family.fasta"),
            job_id="smoke",
            reference_pdb="2dn1-A",
            pdb_dir=EVO_SMOKE_DIR,
            results_dir=results_dir,
            contact_maps=False,
            debug=False,
        )
    elapsed = time.time() - start

    ic_tables = [f for f in os.listdir(results_dir) if f.startswith("IC_")]
    assert ic_tables, f"evolution wrote no IC_* tables in {results_dir}"
    for name in ic_tables:
        assert os.path.getsize(os.path.join(results_dir, name)) > 0
    assert elapsed < MAX_SECONDS, f"evolution took {elapsed:.1f}s (limit {MAX_SECONDS:.0f}s)"
