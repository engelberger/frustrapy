"""Regression tests for correctness fixes in the calculation and IO paths.

Each test is written to FAIL on the pre-fix code and PASS afterwards, pinning a
specific bug so it cannot silently return:

  - no pickle.load over an os.walk (arbitrary-code-execution risk)
  - glycine H-Beta uses the interpolated H x-coordinate, not the N x-coordinate
  - a residue missing its O atom is skipped, not a crash
  - complete_backbone fails loudly and never overwrites the input PDB
  - get_frustration filters branch on pdb.mode (no KeyError)
  - dir_frustration always returns a 2-tuple
  - dir_frustration initializes density_results before use
  - _download_pdb validates its argument and drops --no-check-certificate
  - PDB records parsed by fixed columns, not whitespace splitting
"""

import inspect
import os
import subprocess
import sys
import warnings

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AWSEM_TOOLS = os.path.join(
    REPO_ROOT, "frustrapy", "core", "scripts", "AWSEMFiles", "AWSEMTools"
)
PDB_TO_COORDS = os.path.join(AWSEM_TOOLS, "PDBToCoordinates.py")

# H-beta interpolation constants (PDBToCoordinates.py:22-24).
aH, bH, cH = -0.946747, 2.50352, -0.620388


def _atom_line(serial, name, resname, chain, resseq, x, y, z, element, altloc=" "):
    """Format one PDB ATOM record at exact column positions."""
    return (
        f"ATOM  {serial:>5} {name:<4}{altloc:1}{resname:>3} {chain:1}{resseq:>4}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2}"
    )


def _run_pdb_to_coords(tmp_path, pdb_lines):
    """Write a PDB and run PDBToCoordinates.py over it; return (returncode, coord_text)."""
    (tmp_path / "in.pdb").write_text("\n".join(pdb_lines) + "\nEND\n")
    proc = subprocess.run(
        [sys.executable, PDB_TO_COORDS, "in", "out"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
    )
    coord_path = tmp_path / "out.coord"
    coord_text = coord_path.read_text() if coord_path.exists() else ""
    return proc, coord_text


# ---------------------------------------------------------------------------
def test_no_pickle_load_in_source():
    """No module deserializes pickle from untrusted/walked files (ACE risk)."""
    offenders = []
    tests_dir = os.path.join(REPO_ROOT, "tests")
    for root, _dirs, files in os.walk(REPO_ROOT):
        if "/.git" in root or "/.venv" in root or "__pycache__" in root:
            continue
        # The scan targets shipped package + scripts, not the test suite itself
        # (which necessarily references the forbidden pattern by name).
        if root == tests_dir or root.startswith(tests_dir + os.sep):
            continue
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            try:
                text = open(path, "r", encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            if "pickle.load(" in text:
                offenders.append(os.path.relpath(path, REPO_ROOT))
    assert not offenders, f"pickle.load found in: {offenders}"


# ---------------------------------------------------------------------------
def test_glycine_hbeta_uses_interpolated_x(tmp_path):
    """The glycine H-Beta pseudo-atom x must be the N/CA/C interpolation, not N.x."""
    n_x, ca_x, c_x = 1.000, 2.000, 3.000
    lines = [
        _atom_line(1, "N", "GLY", "A", 1, n_x, 0.0, 0.0, "N"),
        _atom_line(2, "CA", "GLY", "A", 1, ca_x, 0.0, 0.0, "C"),
        _atom_line(3, "C", "GLY", "A", 1, c_x, 0.0, 0.0, "C"),
        _atom_line(4, "O", "GLY", "A", 1, 4.000, 0.0, 0.0, "O"),
    ]
    proc, coord_text = _run_pdb_to_coords(tmp_path, lines)
    assert proc.returncode == 0, proc.stderr
    hbeta = [ln for ln in coord_text.splitlines() if "H-Beta" in ln]
    assert hbeta, f"no H-Beta atom emitted; got:\n{coord_text}"
    hbeta_x = float(hbeta[0].split()[3])
    expected = aH * n_x + bH * ca_x + cH * c_x
    assert hbeta_x == pytest.approx(expected, abs=1e-5)
    # The pre-fix bug wrote N.x here; ensure we are not doing that.
    assert hbeta_x != pytest.approx(n_x, abs=1e-5)


# ---------------------------------------------------------------------------
def test_residue_missing_o_does_not_crash(tmp_path):
    """A residue lacking its backbone O is skipped, not a hard KeyError crash."""
    lines = [
        # Complete residue 1.
        _atom_line(1, "N", "ALA", "A", 1, 1.0, 0.0, 0.0, "N"),
        _atom_line(2, "CA", "ALA", "A", 1, 2.0, 0.0, 0.0, "C"),
        _atom_line(3, "C", "ALA", "A", 1, 3.0, 0.0, 0.0, "C"),
        _atom_line(4, "O", "ALA", "A", 1, 4.0, 0.0, 0.0, "O"),
        _atom_line(5, "CB", "ALA", "A", 1, 2.5, 1.0, 0.0, "C"),
        # Residue 2 is missing its O atom (would KeyError pre-fix).
        _atom_line(6, "N", "ALA", "A", 2, 5.0, 0.0, 0.0, "N"),
        _atom_line(7, "CA", "ALA", "A", 2, 6.0, 0.0, 0.0, "C"),
        _atom_line(8, "C", "ALA", "A", 2, 7.0, 0.0, 0.0, "C"),
        _atom_line(9, "CB", "ALA", "A", 2, 6.5, 1.0, 0.0, "C"),
    ]
    proc, coord_text = _run_pdb_to_coords(tmp_path, lines)
    assert proc.returncode == 0, f"script crashed on O-less residue:\n{proc.stderr}"
    # The complete residue still produced its atoms.
    assert "C-Alpha" in coord_text


# ---------------------------------------------------------------------------
def test_complete_backbone_validates_and_no_inplace_overwrite():
    """complete_backbone fails loudly and never renames the input file."""
    from frustrapy.utils.helpers import complete_backbone

    src = inspect.getsource(complete_backbone)
    assert "check=True" in src, "subprocess must use check=True"
    assert "getsize" in src, "completed output must be validated as non-empty"
    assert "RuntimeError" in src or "raise" in src, "must raise on failure"
    assert "os.rename(" not in src, "must not rename in place (use os.replace job copy)"


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["configurational", "mutational", "singleresidue"])
def test_get_frustration_filter_branches_on_mode(crn_pdb, run_mode, mode):
    """get_frustration filtering must work in every mode without a KeyError."""
    from frustrapy.analysis.frustration import get_frustration

    result, _ = run_mode(crn_pdb, mode)
    pdb = result[0]
    # These raised KeyError pre-fix (wrong column names for the mode).
    full = get_frustration(pdb)
    by_chain = get_frustration(pdb, chain=["A"])
    by_res = get_frustration(pdb, res_num=[10])
    assert len(full) > 0
    assert len(by_chain) == len(full)  # 1CRN is a single chain A
    assert len(by_res) >= 1


# -----------------------------------------------------------------------
@pytest.mark.slow
def test_dir_frustration_returns_two_tuple_when_skipped(crn_pdb, tmp_path):
    """dir_frustration returns a 2-tuple even when the mode is already logged (skip)."""
    import shutil

    import frustrapy

    pdbs_dir = tmp_path / "pdbs"
    pdbs_dir.mkdir()
    shutil.copy(crn_pdb, pdbs_dir / "1crn.pdb")
    results_dir = str(tmp_path / "res")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = frustrapy.dir_frustration(
            pdbs_dir=str(pdbs_dir),
            mode="configurational",
            results_dir=results_dir,
            graphics=False,
            visualization=False,
            debug="ERROR",
        )
        # Second call: mode now in Modes.log -> calculation skipped.
        second = frustrapy.dir_frustration(
            pdbs_dir=str(pdbs_dir),
            mode="configurational",
            results_dir=results_dir,
            graphics=False,
            visualization=False,
            debug="ERROR",
        )
    assert isinstance(first, tuple) and len(first) == 2
    assert isinstance(second, tuple) and len(second) == 2  # pre-fix returned None


@pytest.mark.slow
def test_dir_frustration_empty_order_list(crn_pdb, tmp_path):
    """An empty order_list returns ({}, None), not an UnboundLocalError on density."""
    import frustrapy

    pdbs_dir = tmp_path / "pdbs"
    pdbs_dir.mkdir()
    results_dir = str(tmp_path / "res")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plots, density = frustrapy.dir_frustration(
            pdbs_dir=str(pdbs_dir),
            order_list=[],
            mode="configurational",
            results_dir=results_dir,
            graphics=False,
            visualization=False,
            debug="ERROR",
        )
    assert plots == {}
    assert density is None


# ---------------------------------------------------------------------------
def test_download_pdb_hardened():
    """_download_pdb validates, bounds, and does not disable TLS verification."""
    from frustrapy.analysis.frustration_calculator import FrustrationCalculator

    src = inspect.getsource(FrustrationCalculator._download_pdb)
    assert "--no-check-certificate" not in src, "must not disable TLS verification"
    assert "check=True" in src, "download must fail loudly"
    assert "timeout=" in src, "download must be time-bounded"


# ---------------------------------------------------------------------------
def test_fixed_width_pdb_parsing(tmp_path):
    """Fixed-column parsing survives altLoc/adjacent fields that shift whitespace splits."""
    from frustrapy.analysis.frustration_calculator import FrustrationCalculator

    # altLoc 'A' abuts the residue name, and a negative x abuts the preceding column --
    # both shift a whitespace-split parser, mislabeling res_name / coordinates.
    lines = [
        _atom_line(1, "N", "ARG", "A", 2, -1.234, 14.099, 3.625, "N", altloc="A"),
        _atom_line(2, "CA", "ARG", "A", 2, 11.234, -14.099, 3.625, "C", altloc="A"),
    ]
    pdb_path = tmp_path / "weird.pdb"
    pdb_path.write_text("\n".join(lines) + "\nEND\n")

    calc = object.__new__(FrustrationCalculator)
    calc.pdb_file = str(pdb_path)
    df = calc._read_and_filter_pdb()

    # Pre-fix (read_csv sep=r"\s+") mislabeled res_name as "AARG" -> dropped by the
    # protein-residue filter, leaving an empty frame. Fixed-width keeps both rows.
    assert len(df) == 2
    assert set(df["res_name"]) == {"ARG"}
    assert df.iloc[0]["x"] == pytest.approx(-1.234)
    assert df.iloc[1]["y"] == pytest.approx(-14.099)
    assert list(df["element"]) == ["N", "C"]


# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_singleresidue_native_energy_is_wt_not_first_row(crn_pdb, tmp_path):
    """Issue #12: SingleResidueData.native_energy is the WILD-TYPE residue's
    frustration at a position, not the first row of the saturation scan."""
    import frustrapy
    from Bio.SeqUtils import seq1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _pdb, _plots, _dens, single_res = frustrapy.calculate_frustration(
            pdb_file=str(crn_pdb),
            mode="singleresidue",
            residues={"A": [1]},
            results_dir=str(tmp_path / "sr"),
            graphics=True,
            visualization=False,
            debug="ERROR",
        )

    rd = single_res["A"][1]
    wt_one_letter = seq1(rd.residue_name)  # 1CRN residue 1 is THR -> "T"
    # native_energy must equal the WT identity's frustration, not an arbitrary row.
    assert rd.native_energy == rd.mutations[wt_one_letter]
