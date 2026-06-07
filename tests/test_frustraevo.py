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


def test_build_positions_strips_reference_gaps():
    """The shared coordinate drops every alignment column where the REFERENCE is a
    gap, renumbers the survivors 1..N, and maps each to the structure's real
    frustration residue number (parity with FrustraEvo's FinalAlign).

    Regression for the coordinate-convention bug: the old code renumbered PDB_pos
    from a 1-based counter and kept reference-gap columns, which crashed (KeyError)
    on any reference that starts at a residue != 1 or has a leading alignment gap.
    """
    from frustrapy.evolution.information_content import InformationContentCalculator

    # Reference has a LEADING gap (col 0) and starts at PDB residue 2.
    ref_aln = "-LSP"
    sr_lines = [
        "Res ChainRes DensityRes AA NativeEnergy DecoyEnergy SDEnergy FrstIndex\n",
        "2 A 0.0 L -1 -1 1 0.9\n",
        "3 A 0.0 S -1 -1 1 0.1\n",
        "4 A 0.0 P -1 -1 1 -2.0\n",
    ]

    # A structure identical to the reference -> the leading gap is stripped and the
    # three survivors map to the real PDB numbers 2, 3, 4 (NOT 1, 2, 3).
    pos = InformationContentCalculator._build_positions(ref_aln, "-LSP", sr_lines)
    assert pos == ["2", "3", "4"]

    # A structure missing the middle residue (only L,P, numbered 2,3 in its own
    # frustration output) -> 'G' placeholder at the gap, real resnums otherwise.
    sr_two = [
        "Res ChainRes DensityRes AA NativeEnergy DecoyEnergy SDEnergy FrstIndex\n",
        "2 A 0.0 L -1 -1 1 0.9\n",
        "3 A 0.0 P -1 -1 1 -2.0\n",
    ]
    pos_gap = InformationContentCalculator._build_positions(ref_aln, "-L-P", sr_two)
    assert pos_gap == ["2", "G", "3"]


def test_equivalences_use_real_resnums_and_strip_gaps(tmp_path):
    """The written equivalence file carries reference-gap-stripped MSA columns
    (1..N) against the structure's real frustration residue numbers, with N/A for
    structure gaps — the convention the IC table's NumRes*_Ref columns rely on."""
    from frustrapy.evolution.information_content import InformationContentCalculator

    calc = InformationContentCalculator.__new__(InformationContentCalculator)
    # Structure missing the reference's middle residue: its own frustration output
    # numbers the two residues it has (L, P) as 2 and 3.
    sr_lines = [
        "Res ChainRes DensityRes AA NativeEnergy DecoyEnergy SDEnergy FrstIndex\n",
        "2 A 0.0 L -1 -1 1 0.9\n",
        "3 A 0.0 P -1 -1 1 -2.0\n",
    ]
    out = tmp_path / "Equival_x.txt"
    calc._save_structure_equivalences(
        structure_id="x-A",
        msa_seq="-L-P",
        ref_aln_seq="-LSP",
        sr_lines=sr_lines,
        output_file=out,
    )
    rows = [ln.rstrip("\n").split("\t") for ln in out.read_text().splitlines()]
    assert rows[0] == ["MSA_pos", "PDB_pos", "Residue", "Chain", "Structure"]
    # MSA col 1 -> real resnum 2 (L); col 2 -> N/A (structure gap); col 3 -> 3 (P).
    assert rows[1][:3] == ["1", "2", "L"]
    assert rows[2][:2] == ["2", "N/A"]
    assert rows[3][:3] == ["3", "3", "P"]


def test_result_dict_shape(globin_family):
    """The result dict has the advertised structure."""
    res = globin_family["result"]
    assert set(res.keys()) == {"job_id", "output_dir", "files", "contacts"}
    assert res["job_id"] == "globin3"
    assert set(res["files"].keys()) == {"data", "sequence_ic", "contact_maps"}
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


def test_write_sequence_ic_strips_ref_gaps_and_formats(tmp_path):
    """SeqIC = per-column Shannon entropy over the reference-gap-stripped MSA.

    Parity port of FrustraEvo's ``Scripts/Seq_IC.py``. Pins, on a synthetic MSA:
      * columns where the REFERENCE is a gap are dropped, survivors renumbered 1..N;
      * a fully conserved column yields the NumPy ``-0.0`` repr (not ``0.0``);
      * a mixed column yields the exact full-precision ``str(float64)`` repr;
      * entropy is order-independent (sequence order must not change values);
      * the ``Position\\tEntropy`` header and tab layout are byte-exact.
    """
    from frustrapy.evolution.information_content import InformationContentCalculator

    # Reference "ref" has a LEADING gap (col 0) -> that column is stripped.
    # Col 1 is fully conserved (all A); col 2 is mixed (B,B,C,B).
    msa = tmp_path / "msa"
    msa.mkdir()
    (msa / "MSA_Final.fasta").write_text(
        ">ref\n-AB\n>s1\nXAB\n>s2\n-AC\n>s3\n-AB\n"
    )

    calc = InformationContentCalculator.__new__(InformationContentCalculator)
    calc.results_dir = tmp_path
    calc.reference_pdb = "ref"
    calc.msa_dir = msa

    out = calc._write_sequence_ic()
    assert out == tmp_path / "SeqIC_ref.tab"
    # 2 surviving columns (the leading ref-gap column is dropped), 1-based.
    assert out.read_text() == "Position\tEntropy\n1\t-0.0\n2\t0.8112781244591328\n"


def test_seqic_byte_identical_to_fixture(globin_family):
    """End-to-end: analyze_family produces SeqIC byte-identical to the frozen
    fixture (the algorithm is verified byte-identical to the original FrustraEvo
    on the full Alpha-globins (140 cols) and Sars-PlPro (309 cols) example sets;
    this 3-member snapshot is the CI-portable anchor)."""
    produced = os.path.join(globin_family["results_dir"], f"SeqIC_{REFERENCE}.tab")
    assert os.path.exists(produced), f"missing SeqIC table: {produced}"
    expected = os.path.join(DATA_DIR, f"expected_SeqIC_{REFERENCE}.tab")
    with open(produced) as p, open(expected) as e:
        assert p.read() == e.read(), "SeqIC drifted from the frozen parity fixture"


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
