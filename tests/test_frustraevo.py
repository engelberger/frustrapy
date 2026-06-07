"""End-to-end + structural tests for the FrustraEvo evolution subpackage (Phase 8).

The subpackage was reworked to a single entry path, ``analyze_family``, which runs a
per-member ``calculate_frustration`` and aggregates per-contact information content
across a protein family. The fixture is a 3-member alpha-globin family (real
FrustraEvo example data, ~141 residues each); each member is a single-chain PDB whose
sequence matches its row in the aligned FASTA.

These tests lock the Phase-8 rework:
  * ``analyze_family`` is the one public entry path (the dead pipeline/integrator/
    logo_calculator entry classes were deleted);
  * the FrstIndex column is read correctly (index 11, not DecoyEnergy at index 9) so the
    frustration-state distribution is non-degenerate;
  * the returned result dict has the advertised shape;
  * ``logomaker`` (the declared dependency) resolves for the sequence-logo module.

NOTE on PATH (tech-debt P1-22): the per-member calculation spawns a bare ``python3``
subprocess, so run ``pytest`` from an activated venv (see tests/conftest.py).
"""

import os
import warnings

import pytest

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "frustraevo")
FASTA = os.path.join(DATA_DIR, "family.fasta")
PDB_DIR = os.path.join(DATA_DIR, "pdbs")
REFERENCE = "1fsx-A"

IC_COLUMNS = [
    "Res1", "Res2", "AA1", "AA2", "NumRes1_Ref", "Chain1_Ref", "NumRes2_Ref",
    "Chain2_Ref", "Prot_Ref", "NoContacts", "FreqConts", "pNEU", "pMIN", "pMAX",
    "HNEU", "HMIN", "HMAX", "Htotal", "ICNEU", "ICMIN", "ICMAX", "ICtotal",
    "FstConserved",
]


def test_fixture_present():
    """The checked-in 3-member family fixture exists."""
    assert os.path.exists(FASTA), f"missing fixture FASTA: {FASTA}"
    for member in (REFERENCE, "3cy5-A", "1fhj-A"):
        pdb = os.path.join(PDB_DIR, f"{member}.pdb")
        assert os.path.exists(pdb), f"missing fixture PDB: {pdb}"


def test_analyze_family_exported_top_level():
    """analyze_family is reachable from the top-level package (single entry path)."""
    import frustrapy

    assert hasattr(frustrapy, "analyze_family")
    from frustrapy.evolution import analyze_family

    assert frustrapy.analyze_family is analyze_family


def test_single_entry_path_dead_modules_removed():
    """The competing entry classes were deleted; only analyze_family remains."""
    import importlib

    for dead in ("pipeline", "integrator", "logo_calculator", "contact_calculator",
                 "logo", "msfa"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"frustrapy.evolution.{dead}")


def test_logomaker_dependency_importable():
    """The declared logomaker dep resolves for the canonical sequence-logo module."""
    import logomaker  # noqa: F401
    from frustrapy.evolution.sequence_logo import SequenceLogoGenerator  # noqa: F401


@pytest.fixture(scope="module")
def globin_family(tmp_path_factory):
    """Run analyze_family once on the 3-member fixture; share across assertions."""
    import frustrapy

    results_dir = str(tmp_path_factory.mktemp("frustraevo"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = frustrapy.analyze_family(
            fasta_file=FASTA,
            job_id="globin3",
            reference_pdb=REFERENCE,
            pdb_dir=PDB_DIR,
            results_dir=results_dir,
        )
    return {"result": result, "results_dir": results_dir}


def test_result_dict_shape(globin_family):
    """The result dict has the advertised structure."""
    res = globin_family["result"]
    assert set(res.keys()) == {"job_id", "output_dir", "files", "contacts"}
    assert res["job_id"] == "globin3"
    assert set(res["files"].keys()) == {"data", "contact_maps"}
    # contact_maps=False by default -> no PNG produced
    assert res["files"]["contact_maps"] is None
    assert set(res["contacts"].keys()) == {"information_content", "summary"}


def test_ic_csv_written_with_expected_columns(globin_family):
    """analyze_family writes the per-contact IC table with the documented columns."""
    import pandas as pd

    csv = os.path.join(globin_family["results_dir"], f"IC_Configurational_{REFERENCE}.csv")
    assert os.path.exists(csv), f"missing IC table: {csv}"
    df = pd.read_csv(csv, sep="\t")
    assert list(df.columns) == IC_COLUMNS
    assert len(df) > 0


def test_frustration_state_distribution_non_degenerate(globin_family):
    """Regression for the FrstIndex column bug (read index 11, not 9 = DecoyEnergy).

    Reading DecoyEnergy as the frustration index put EVERY contact at <= -1 -> all MAX.
    With the correct column, a real globin family yields a mix of states dominated by
    NEU/MIN, with MAX a minority -- assert the distribution is genuinely non-degenerate.
    """
    summary = globin_family["result"]["contacts"]["summary"]
    total = summary["total_contacts"]
    assert total > 0
    assert (
        summary["minimally_frustrated"]
        + summary["neutrally_frustrated"]
        + summary["maximally_frustrated"]
        == total
    )
    # No single class may account for everything (the bug produced 100% MAX).
    assert summary["minimally_frustrated"] > 0
    assert summary["neutrally_frustrated"] > 0
    assert summary["maximally_frustrated"] < total
