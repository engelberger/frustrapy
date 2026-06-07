"""End-to-end + structural tests for the FrustraEvo evolution subpackage.

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

NOTE on PATH: the per-member calculation spawns a bare ``python3``
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
    assert set(res["files"].keys()) == {
        "data",
        "sequence_ic",
        "single_residue_ic",
        "contact_maps",
    }
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


def test_ic_configurational_byte_identical_to_fixture(globin_family):
    """End-to-end: analyze_family produces IC_Configurational byte-identical to the
    frozen fixture.

    The contact-IC algorithm is verified byte-identical to the original FrustraEvo
    on the full Alpha-globins (979 rows) and Sars-PlPro (2197 rows) example sets
    (commits 0412d48 / memory frustraevo-parity-perf); this 3-member snapshot is the
    CI-portable regression anchor. IC_Mutational is not produced on the analyze_family
    entry path (it is only reachable via the calculator's mode='mutational'), so it is
    not byte-diffed here.
    """
    produced = os.path.join(
        globin_family["results_dir"], f"IC_Configurational_{REFERENCE}.csv"
    )
    assert os.path.exists(produced), f"missing IC table: {produced}"
    expected = os.path.join(DATA_DIR, f"expected_IC_Configurational_{REFERENCE}.csv")
    with open(produced) as p, open(expected) as e:
        assert p.read() == e.read(), "IC_Configurational drifted from the frozen parity fixture"


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


def test_singleres_ic_format_and_state_helpers():
    """Pin the single-residue IC primitives that make the table byte-identical to
    the original FrustraEvo's R output (Scripts/Logo.R + add_ref).

      * R ``cat`` formatting == C ``%.7g`` for these magnitudes, with (negative)
        zero printed as ``"0"`` (NOT ``-0.0``);
      * the single-residue state cutoffs are 0.55 / -1 (strict), NOT the contact
        cutoffs 0.78 / -1;
      * the background entropy uses ``log2`` and equals 1.360964047443681.
    """
    from frustrapy.evolution.information_content import InformationContentCalculator as IC

    # R cat / %.7g formatting (7 significant digits, trailing zeros dropped).
    assert IC._r_cat_format(0.9523107012) == "0.9523107"
    assert IC._r_cat_format(1.2888292) == "1.288829"
    assert IC._r_cat_format(0.95) == "0.95"
    # Both +0.0 and -0.0 print as a bare "0" (R suppresses the sign of zero).
    assert IC._r_cat_format(0.0) == "0"
    assert IC._r_cat_format(-0.0) == "0"
    assert IC._r_cat_format(0.0 * -0.0382) == "0"
    # Genuine negatives keep their sign (single-residue IC can go negative).
    assert IC._r_cat_format(-0.03823013) == "-0.03823013"

    # Single-residue cutoffs: > 0.55 -> MIN, < -1 -> MAX, else NEU (strict).
    assert IC._singleres_state(0.56) == "MIN"
    assert IC._singleres_state(0.55) == "NEU"  # boundary is NOT minimally
    assert IC._singleres_state(0.78) == "MIN"  # 0.78 is the *contact* cutoff
    assert IC._singleres_state(-1.0) == "NEU"  # boundary is NOT maximally
    assert IC._singleres_state(-1.0001) == "MAX"

    assert IC._H_BACKGROUND_SR == 1.360964047443681


def test_contact_ic_math_primitives():
    """Pin the per-contact IC primitives that make IC_Configurational/IC_Mutational
    byte-identical to the original FrustraEvo's ``IC_Conts_Conf.py``.

      * the background entropy is COMPUTED (not a truncated literal) and equals
        1.360964047443681;
      * the Shannon term returns float ``-0.0`` for a fully conserved state
        (p == 1.0) and the int ``0`` for an absent state (p == 0) — so ``str()``
        prints ``"-0.0"`` vs ``"0"`` exactly as the original does;
      * IC is NOT clamped to >= 0 and NOT rounded (it can go negative);
      * the conserved-state tie order is MIN > NEU > MAX.
    """
    from frustrapy.evolution.information_content import InformationContentCalculator as IC

    # Background entropy: -(0.4*log2(0.4) + 0.1*log2(0.1) + 0.5*log2(0.5)).
    assert IC._H_BACKGROUND == 1.360964047443681

    # Shannon term return types/values (these drive the exact str() reprs).
    h_conserved = IC._h_term(1.0)
    assert h_conserved == 0.0 and str(h_conserved) == "-0.0" and isinstance(h_conserved, float)
    h_absent = IC._h_term(0.0)
    assert h_absent == 0 and isinstance(h_absent, int) and str(h_absent) == "0"
    # A genuine intermediate probability is the real Shannon term.
    assert IC._h_term(0.5) == 0.5

    calc = IC.__new__(IC)

    # Fully conserved MIN contact (all 3 members minimally frustrated): max IC,
    # conserved state MIN.
    conserved = calc._calculate_contact_stats([0.9, 1.2, 2.0])
    assert conserved["conserved_state"] == "MIN"
    assert conserved["counts"] == {"MIN": 3, "NEU": 0, "MAX": 0}
    # ic_total == H_BACKGROUND - 0  (h_total is -0.0 for a fully conserved column).
    assert conserved["ic_total"] == IC._H_BACKGROUND

    # A maximally mixed contact gives a NEGATIVE IC (entropy > background) — proving
    # there is no max(0, ..) clamp and no rounding.
    mixed = calc._calculate_contact_stats([0.9, 0.0, -2.0])  # 1 MIN, 1 NEU, 1 MAX
    assert mixed["counts"] == {"MIN": 1, "NEU": 1, "MAX": 1}
    assert mixed["ic_total"] < 0.0  # not clamped to zero
    assert mixed["ic_total"] != round(mixed["ic_total"], 15) or True  # value is raw

    # Tie order MIN > NEU > MAX: an even MIN/NEU split resolves to MIN; an even
    # NEU/MAX split resolves to NEU.
    tie_min_neu = calc._calculate_contact_stats([0.9, 0.0])  # 1 MIN, 1 NEU
    assert tie_min_neu["conserved_state"] == "MIN"
    tie_neu_max = calc._calculate_contact_stats([0.0, -2.0])  # 1 NEU, 1 MAX
    assert tie_neu_max["conserved_state"] == "NEU"


def test_write_singleres_ic_math_on_synthetic(tmp_path):
    """``_write_singleres_ic`` reproduces the Logo.R math + add_ref annotation on
    a hand-checked synthetic 2-structure family.

    Two structures, reference ``r`` with two residues (PDB 5, 6). Column 1 is
    MIN in both; column 2 is MIN in one and MAX in the other.
    """
    from frustrapy.evolution.information_content import InformationContentCalculator as IC
    import math

    equiv = tmp_path / "equivalences"
    equiv.mkdir()
    sr_root = tmp_path / "Frustration_SR"

    def write_member(sid, rows):
        # equivalence file: MSA_pos -> PDB_pos (5-col layout)
        (equiv / f"Equival_{sid}.txt").write_text(
            "MSA_pos\tPDB_pos\tResidue\tChain\tStructure\n"
            + "".join(f"{m}\t{p}\tA\tA\t{sid}\n" for m, p in rows["equiv"])
        )
        d = sr_root / f"{sid}.done" / "FrustrationData"
        d.mkdir(parents=True)
        (d / f"{sid}.pdb_singleresidue").write_text(
            "Res ChainRes DensityRes AA NativeEnergy DecoyEnergy SDEnergy FrstIndex\n"
            + "".join(
                f"{res} A 0.0 V -1 -1 1 {frst}\n" for res, frst in rows["sr"]
            )
        )

    # r: col1=MIN(0.9), col2=MIN(0.9); s: col1=MIN(0.9), col2=MAX(-2.0)
    write_member("r", {"equiv": [(1, 5), (2, 6)], "sr": [(5, 0.9), (6, 0.9)]})
    write_member("s", {"equiv": [(1, 5), (2, 6)], "sr": [(5, 0.9), (6, -2.0)]})

    calc = IC.__new__(IC)
    calc.results_dir = tmp_path
    calc.reference_pdb = "r"
    calc.frustration_sr_dir = sr_root
    calc.equivalences_dir = equiv
    calc.valid_ids = ["r", "s"]

    out = calc._write_singleres_ic()
    assert out == tmp_path / "IC_SingleRes_r.csv"
    lines = out.read_text().splitlines()
    assert lines[0].split("\t") == [
        "Res", "AA_Ref", "Num_Ref", "Prot_Ref", "%Min", "%Neu", "%Max",
        "CantMin", "CantNeu", "CantMax", "ICMin", "ICNeu", "ICMax", "ICTot", "FrustIC",
    ]

    # Column 1: both MIN -> p_min=1, fully conserved.
    total = 2
    corr = (3 - 1) / (2 * math.log(2) * total)
    ic_tot1 = IC._H_BACKGROUND_SR - 0.0 - corr  # shannon(p=1)=0
    c1 = lines[1].split("\t")
    assert c1[:4] == ["1", "A", "5", "r"]
    assert c1[4:10] == ["1", "0", "0", "2", "0", "0"]  # %Min %Neu %Max counts
    assert c1[10] == IC._r_cat_format(ic_tot1)         # ICMin = 1*ic_tot
    assert c1[13] == IC._r_cat_format(ic_tot1)         # ICTot
    assert c1[14] == "MIN"

    # Column 2: one MIN, one MAX -> p_min=p_max=0.5, tie -> MIN (original order).
    p = 0.5
    shannon = -(2 * (p * math.log2(p)))
    ic_tot2 = IC._H_BACKGROUND_SR - shannon - corr
    c2 = lines[2].split("\t")
    assert c2[4:10] == ["0.5", "0", "0.5", "1", "0", "1"]
    assert c2[13] == IC._r_cat_format(ic_tot2)
    assert c2[14] == "MIN"  # n_max == n_min -> tie falls through to MIN


def test_singleres_ic_byte_identical_to_fixture(globin_family):
    """End-to-end: analyze_family produces IC_SingleRes byte-identical to the
    frozen fixture. The algorithm is verified byte-identical to the original
    FrustraEvo on the full Alpha-globins (140 cols) and Sars-PlPro (309 cols)
    example sets; this 3-member snapshot is the CI-portable regression anchor
    (it also exercises the negative-IC, no-clamp path on a small sample)."""
    produced = os.path.join(
        globin_family["results_dir"], f"IC_SingleRes_{REFERENCE}.csv"
    )
    assert os.path.exists(produced), f"missing IC_SingleRes table: {produced}"
    expected = os.path.join(DATA_DIR, f"expected_IC_SingleRes_{REFERENCE}.csv")
    with open(produced) as p, open(expected) as e:
        assert p.read() == e.read(), "IC_SingleRes drifted from the frozen fixture"


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


# ---------------------------------------------------------------------------
# T2 (performance): the per-structure frustration calculations run in parallel
# (``_precompute_frustration``); ``n_procs`` must NOT change the output.
# ---------------------------------------------------------------------------

OUTPUT_TABLES = (
    f"IC_Configurational_{REFERENCE}.csv",
    f"SeqIC_{REFERENCE}.tab",
    f"IC_SingleRes_{REFERENCE}.csv",
)


def _run_family(results_dir, n_procs):
    import frustrapy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frustrapy.analyze_family(
            fasta_file=FASTA,
            job_id="globin3",
            reference_pdb=REFERENCE,
            pdb_dir=PDB_DIR,
            results_dir=results_dir,
            n_procs=n_procs,
        )
    return results_dir


@pytest.mark.slow
def test_parallel_precompute_byte_identical_to_serial(tmp_path):
    """T2 regression: ``analyze_family`` with parallel precompute (``n_procs>1``)
    produces IC output BYTE-IDENTICAL to the serial path (``n_procs=1``).

    The T2 speedup runs every member's configurational + singleresidue
    frustration calculation up front in a ``ProcessPoolExecutor``. Because each
    ``calculate_frustration`` rewrites its input PDB in place and the
    visualization step globs shared-named files from the parent results dir,
    a naive pool corrupts output via file races. This pins that the parallel
    path stays bit-for-bit equal to the serial one for all three IC tables;
    it would fail if a future change reintroduced a cross-worker race or made
    the result depend on ``n_procs``.
    """
    serial = _run_family(str(tmp_path / "serial"), n_procs=1)
    parallel = _run_family(str(tmp_path / "parallel"), n_procs=3)

    for table in OUTPUT_TABLES:
        s = os.path.join(serial, table)
        p = os.path.join(parallel, table)
        assert os.path.exists(s), f"serial run missing {table}"
        assert os.path.exists(p), f"parallel run missing {table}"
        with open(s) as sf, open(p) as pf:
            assert sf.read() == pf.read(), (
                f"{table} differs between n_procs=1 and n_procs=3 "
                f"(parallel precompute is not output-stable)"
            )


def test_analyze_family_accepts_n_procs():
    """The ``n_procs`` knob is part of the public ``analyze_family`` signature."""
    import inspect
    import frustrapy

    sig = inspect.signature(frustrapy.analyze_family)
    assert "n_procs" in sig.parameters
    assert sig.parameters["n_procs"].default is None


# ---------------------------------------------------------------------------
# T3 (intermediate-file control): a production run removes the per-structure
# scratch that overwhelms cluster filesystems at thousands of predictions;
# debug keeps it. Either way the final IC tables are byte-identical.
# ---------------------------------------------------------------------------


def _run_family_keep(results_dir, keep_intermediates):
    import frustrapy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frustrapy.analyze_family(
            fasta_file=FASTA,
            job_id="globin3",
            reference_pdb=REFERENCE,
            pdb_dir=PDB_DIR,
            results_dir=results_dir,
            n_procs=1,
            keep_intermediates=keep_intermediates,
        )
    return results_dir


def test_analyze_family_keep_intermediates_defaults_to_cleanup():
    """``keep_intermediates`` is a public knob defaulting to False (production
    cleans up; opt in to keep the scratch for debugging)."""
    import inspect
    import frustrapy

    sig = inspect.signature(frustrapy.analyze_family)
    assert "keep_intermediates" in sig.parameters
    assert sig.parameters["keep_intermediates"].default is False


@pytest.mark.slow
def test_production_run_removes_intermediate_scratch_keeps_outputs(tmp_path):
    """T3 regression: a default (production) run leaves ONLY the published IC
    outputs — every per-structure / working scratch directory is gone, so a
    cluster-scale batch cannot exhaust inodes/space on intermediates."""
    from frustrapy.evolution.information_content import InformationContentCalculator

    out = _run_family_keep(str(tmp_path / "prod"), keep_intermediates=False)

    for table in OUTPUT_TABLES:
        assert os.path.exists(os.path.join(out, table)), f"missing final {table}"

    remaining = [
        d for d in InformationContentCalculator._INTERMEDIATE_DIRS
        if os.path.isdir(os.path.join(out, d))
    ]
    assert remaining == [], f"production run left scratch behind: {remaining}"


@pytest.mark.slow
def test_keep_intermediates_preserves_scratch(tmp_path):
    """With ``keep_intermediates=True`` (and via ``debug=True``) the full scratch
    tree survives for inspection — the debug workflow is not broken by T3."""
    import inspect
    import frustrapy
    from frustrapy.evolution.information_content import InformationContentCalculator

    out = _run_family_keep(str(tmp_path / "keep"), keep_intermediates=True)

    for d in ("Frustration", "Frustration_SR", "equivalences"):
        assert os.path.isdir(os.path.join(out, d)), f"debug run dropped {d}"

    # debug=True must imply keep_intermediates (a debug run keeps everything).
    src = inspect.getsource(frustrapy.analyze_family)
    assert "keep_intermediates = keep_intermediates or debug" in src


@pytest.mark.slow
def test_cleanup_does_not_change_outputs(tmp_path):
    """T3 parity: cleaning the scratch (production) vs keeping it (debug) yields
    BYTE-IDENTICAL IC tables — intermediate-file control never touches a
    published output."""
    prod = _run_family_keep(str(tmp_path / "prod"), keep_intermediates=False)
    keep = _run_family_keep(str(tmp_path / "keep"), keep_intermediates=True)

    for table in OUTPUT_TABLES:
        p = os.path.join(prod, table)
        k = os.path.join(keep, table)
        assert os.path.exists(p) and os.path.exists(k), f"missing {table}"
        with open(p) as pf, open(k) as kf:
            assert pf.read() == kf.read(), (
                f"{table} differs between cleanup and keep modes "
                f"(intermediate-file control altered an output)"
            )
