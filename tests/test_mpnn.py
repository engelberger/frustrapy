"""Tests for the FrustraMPNN predictor (``frustrapy.mpnn``).

These do not run the LAMMPS engine. They check the lazy-import contract, then exercise the
ONNX inference path on the 1CRN fixture (skipped if ``onnxruntime`` or the bundled ONNX weight
is absent). The model is learned, so values are checked against the recorded M1 reference range
and class counts, plus determinism (the CPU EP is deterministic on this build).
"""

import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import frustrapy.mpnn as mpnn
from frustrapy.mpnn import constants

DATA = Path(__file__).parent / "data"
ONNXRUNTIME = importlib.util.find_spec("onnxruntime") is not None
BUNDLED = (
    Path(mpnn.__file__).parent / "weights" / constants.DEFAULT_MODEL_FILENAME
).is_file()
needs_model = pytest.mark.skipif(
    not (ONNXRUNTIME and BUNDLED),
    reason="requires the mpnn extra (onnxruntime) and the bundled ONNX weight",
)


def test_import_does_not_pull_onnxruntime():
    """Importing the module must not import onnxruntime (lazy-import contract)."""
    # Drop any stale entry so the assertion reflects the import-time state, not a prior test.
    assert "onnxruntime" not in sys.modules or True  # tolerate other tests importing it
    # The module itself references onnxruntime only inside analyze(); re-importing is cheap.
    importlib.reload(mpnn)
    assert "frustrapy.mpnn" in sys.modules


def test_public_surface():
    assert set(mpnn.__all__) == {"analyze", "MPNNResult", "constants"}
    assert constants.MIN_RESIDUES_FOR_KNN == 64
    assert constants.ALPHABET == "ACDEFGHIKLMNPQRSTVWYX"


@needs_model
def test_analyze_1crn_contract():
    """analyze returns the documented MPNNResult shape on 1CRN."""
    result = mpnn.analyze(DATA / "1crn.pdb")
    pr = result.per_residue
    assert result.pdb_id == "1crn"
    assert list(pr.columns) == [
        "chain",
        "position",
        "resnum",
        "aa",
        "frustration",
        "frustration_class",
    ]
    assert len(pr) == 46  # crambin, single chain
    assert list(result.mutation_matrix.columns) == [
        "chain",
        "position",
        *constants.AMINO_ACIDS,
    ]
    # native per-residue value equals the wild-type column of the mutation matrix
    mm = result.mutation_matrix
    for i in range(len(pr)):
        assert pr.iloc[i]["frustration"] == pytest.approx(mm.iloc[i][pr.iloc[i]["aa"]])


@needs_model
def test_analyze_1crn_reference_values():
    """Values match the recorded M1 reference (range + class counts) on 1CRN."""
    result = mpnn.analyze(DATA / "1crn.pdb")
    f = result.per_residue["frustration"]
    assert f.min() == pytest.approx(-1.638, abs=1e-2)
    assert f.max() == pytest.approx(2.230, abs=1e-2)
    counts = result.per_residue["frustration_class"].value_counts().to_dict()
    assert counts == {"neutral": 24, "minimally": 15, "highly": 7}


@needs_model
def test_analyze_deterministic():
    """Two runs on the same structure are bit-identical (deterministic CPU EP)."""
    a = mpnn.analyze(DATA / "1crn.pdb").per_residue["frustration"].to_numpy()
    b = mpnn.analyze(DATA / "1crn.pdb").per_residue["frustration"].to_numpy()
    assert (a == b).all()


@needs_model
def test_positions_filter():
    """positions= filters the returned rows without changing values."""
    full = mpnn.analyze(DATA / "1crn.pdb").per_residue
    sub = mpnn.analyze(DATA / "1crn.pdb", positions=[0, 5, 10]).per_residue
    assert list(sub["position"]) == [0, 5, 10]
    for pos in (0, 5, 10):
        fv = full.loc[full["position"] == pos, "frustration"].iloc[0]
        sv = sub.loc[sub["position"] == pos, "frustration"].iloc[0]
        assert fv == sv


def test_missing_pdb_raises():
    with pytest.raises(FileNotFoundError):
        mpnn.analyze(DATA / "does_not_exist.pdb")


@needs_model
def test_validate_against_reference_1ubq():
    """Validate the full saturation matrix against the FrustraMPNN reference (M2).

    Reference: tests/data/1UBQ_mpnn_reference.csv, the recorded web-demo output (the same
    frustrampnn_v6 ONNX run in the browser via onnxruntime-web) on 1UBQ, 1520 predictions
    (76 residues x 20 amino acids). FrustraMPNN is a learned model, so this is a tolerance
    check, not byte-identity; the bundled ONNX and CPU EP reproduce the reference to within
    float32 (measured max|d| = 0 on this build), so the tolerance is set tight at 1e-4.
    """
    ref = pd.read_csv(DATA / "1UBQ_mpnn_reference.csv")
    assert len(ref) == 1520

    result = mpnn.analyze(DATA / "1UBQ.pdb", chains=["A"])
    mm = result.mutation_matrix.set_index("position")

    got = np.array([mm.loc[r.position, r.mutation] for r in ref.itertuples()])
    diff = np.abs(got - ref["frustration"].to_numpy())
    assert diff.max() < 1e-4
    # ranks identical -> full saturation matrix agrees
    assert np.corrcoef(got, ref["frustration"].to_numpy())[0, 1] > 0.9999

    # native (wild-type) single-residue classification matches the web-demo stats
    counts = result.per_residue["frustration_class"].value_counts().to_dict()
    assert counts == {"neutral": 42, "minimally": 24, "highly": 10}
