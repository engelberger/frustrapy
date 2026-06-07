"""Tests for the ``frustrapy`` command-line tool (``frustrapy.cli.main``).

Two lanes, mirroring the rest of the suite:

* **fast** — exercised entirely in-process via Typer's :class:`CliRunner` without
  touching the LAMMPS/AWSEM engine: ``--help`` for the app and every subcommand,
  ``--version``, and one argument-validation error path per subcommand.
* **slow** (``@pytest.mark.slow``) — drive each subcommand end-to-end on a small
  fixture (1CRN / the 3-member evo_smoke family), assert it exits 0 and writes the
  expected output tree. These run the real toolchain, so — like the rest of the e2e
  lane — they need the project virtualenv on PATH (see ``tests/conftest.py``).

The CLI is a thin Typer wrapper over :mod:`frustrapy.sdk`; these tests check the
wiring (arg parsing, exit codes, output location), not the frustration numbers,
which are locked by the parity/anchor tests.
"""

import os
import warnings

import pytest

from typer.testing import CliRunner

from frustrapy.cli.main import app

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CRN_PDB = os.path.join(DATA_DIR, "1crn.pdb")
EVO_DIR = os.path.join(DATA_DIR, "evo_smoke")
EVO_FASTA = os.path.join(EVO_DIR, "family.fasta")
EVO_REFERENCE = "3lqd-A"

runner = CliRunner()


# --------------------------------------------------------------------------- #
# fast lane — help / version / argument validation (no engine)
# --------------------------------------------------------------------------- #
def test_app_help_lists_every_subcommand():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for sub in ("single", "batch", "evo", "mutate"):
        assert sub in result.output


def test_version_flag():
    import frustrapy

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert frustrapy.__version__ in result.output


@pytest.mark.parametrize("sub", ["single", "batch", "evo", "mutate"])
def test_subcommand_help(sub):
    """Every subcommand renders its own ``--help`` and exits 0."""
    result = runner.invoke(app, [sub, "--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_single_rejects_unknown_mode():
    result = runner.invoke(app, ["single", CRN_PDB, "--mode", "bogus"])
    assert result.exit_code == 1


def test_single_rejects_bad_seq_dist():
    result = runner.invoke(app, ["single", CRN_PDB, "--seq-dist", "7"])
    assert result.exit_code == 1


def test_single_missing_file_is_usage_error():
    """Typer's ``exists=True`` rejects a nonexistent PDB before any work (exit 2)."""
    result = runner.invoke(app, ["single", os.path.join(DATA_DIR, "nope.pdb")])
    assert result.exit_code == 2


def test_batch_empty_directory(tmp_path):
    result = runner.invoke(app, ["batch", str(tmp_path)])
    assert result.exit_code == 1


def test_mutate_rejects_unknown_method():
    result = runner.invoke(app, ["mutate", CRN_PDB, "--res", "10", "--method", "bogus"])
    assert result.exit_code == 1


def test_evo_missing_fasta_is_usage_error(tmp_path):
    """A missing FASTA is rejected by Typer's ``exists=True`` (exit 2)."""
    result = runner.invoke(
        app,
        [
            os.path.join(DATA_DIR, "nope.fasta"),
            "--job-id",
            "x",
            "--pdb-dir",
            str(tmp_path),
        ],
    )
    # First positional is the (missing) FASTA -> usage error, no engine run.
    assert result.exit_code == 2


# --------------------------------------------------------------------------- #
# slow lane — end-to-end on small fixtures (runs the real engine)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_single_end_to_end(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = runner.invoke(
            app,
            ["single", CRN_PDB, "--mode", "configurational", "-o", str(tmp_path)],
        )
    assert result.exit_code == 0, result.output
    table = tmp_path / "1crn.done" / "FrustrationData" / "1crn.pdb_configurational"
    assert table.exists(), f"missing output table: {table}"


@pytest.mark.slow
def test_batch_end_to_end(tmp_path):
    import shutil

    pdbs_dir = tmp_path / "pdbs"
    pdbs_dir.mkdir()
    shutil.copy(CRN_PDB, pdbs_dir / "1crn.pdb")
    out = tmp_path / "out"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = runner.invoke(
            app,
            ["batch", str(pdbs_dir), "--mode", "configurational", "-o", str(out)],
        )
    assert result.exit_code == 0, result.output
    table = out / "1crn.done" / "FrustrationData" / "1crn.pdb_configurational"
    assert table.exists(), f"missing output table: {table}"


@pytest.mark.slow
def test_mutate_end_to_end(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = runner.invoke(
            app,
            ["mutate", CRN_PDB, "--res", "10", "--chain", "A", "-o", str(tmp_path)],
        )
    assert result.exit_code == 0, result.output
    mut_dir = tmp_path / "1crn.done" / "MutationsData"
    assert mut_dir.exists(), f"missing MutationsData dir: {mut_dir}"


@pytest.mark.slow
def test_evo_end_to_end(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = runner.invoke(
            app,
            [
                "evo",
                EVO_FASTA,
                "--job-id",
                "evo_smoke",
                "--pdb-dir",
                EVO_DIR,
                "--reference-pdb",
                EVO_REFERENCE,
                "-o",
                str(tmp_path),
            ],
        )
    assert result.exit_code == 0, result.output
