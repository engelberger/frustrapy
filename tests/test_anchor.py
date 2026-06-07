"""Golden / structural assertions on the verified 1CRN empirical anchor.

Linux 1CRN configurational, graphics=False: a 4-tuple return, a 232-row x 14-column
``1crn.pdb_configurational`` table, and a ``tertiary_frustration.dat`` produced by the
LAMMPS binary. These lock the happy path so later refactors (Phases 2/3/6) cannot
silently change the numbers.

The table is read as TEXT (pandas), never via ``pickle.load`` -- avoids re-introducing
the P0-1 arbitrary-code-execution pattern.
"""

import os

import pandas as pd

from frustrapy.visualization.structure import classify_contact_frustration

CONFIG_COLUMNS = [
    "Res1",
    "Res2",
    "ChainRes1",
    "ChainRes2",
    "DensityRes1",
    "DensityRes2",
    "AA1",
    "AA2",
    "NativeEnergy",
    "DecoyEnergy",
    "SDEnergy",
    "FrstIndex",
    "Welltype",
    "FrstState",
]

SINGLERES_COLUMNS = [
    "Res",
    "ChainRes",
    "DensityRes",
    "AA",
    "NativeEnergy",
    "DecoyEnergy",
    "SDEnergy",
    "FrstIndex",
]


def test_returns_four_tuple(crn_configurational):
    """calculate_frustration returns a 4-tuple (Pdb, dict, density|None, single|None)."""
    result = crn_configurational["result"]
    assert isinstance(result, tuple)
    assert len(result) == 4


def test_configurational_table_shape(crn_configurational):
    """232 data rows x 14 columns -- the verified anchor."""
    table = crn_configurational["table"]
    assert os.path.exists(table), f"missing frustration table: {table}"
    df = pd.read_csv(table, sep=r"\s+")
    assert list(df.columns) == CONFIG_COLUMNS
    assert len(df) == 232


def test_tertiary_frustration_dat_present(crn_configurational):
    """The LAMMPS binary wrote its raw output (invisible at import time, P0-C)."""
    dat = os.path.join(
        crn_configurational["job_dir"], "FrustrationData", "tertiary_frustration.dat"
    )
    assert os.path.exists(dat)
    assert os.path.getsize(dat) > 0


def test_frstindex_sign_convention(crn_configurational):
    """The #1 reimplementation hazard: FrstIndex == (DecoyEnergy - NativeEnergy)/SDEnergy.

    AWSEM energies are 'more negative = more favorable', so a favorable native contact
    yields a POSITIVE index (minimally frustrated) -- the negation of the paper's
    published numerator. Confirm the implemented relationship holds row-for-row.
    """
    df = pd.read_csv(crn_configurational["table"], sep=r"\s+")
    expected = (df["DecoyEnergy"] - df["NativeEnergy"]) / df["SDEnergy"]
    # Tolerance covers 3-decimal rounding of the stored energy columns; a FLIPPED sign
    # would diverge by ~2*|FrstIndex| (order 1.9 here), far outside this band.
    assert (expected - df["FrstIndex"]).abs().max() < 5e-3


def test_mutational_table_shape(crn_mutational):
    """Mutational anchor: 232 data rows x 14 columns (same contact set as config)."""
    table = crn_mutational["table"]
    assert os.path.exists(table), f"missing frustration table: {table}"
    df = pd.read_csv(table, sep=r"\s+")
    assert list(df.columns) == CONFIG_COLUMNS
    assert len(df) == 232


def test_singleresidue_table_shape(crn_singleresidue):
    """Single-residue anchor: 46 data rows (one per crambin residue) x 8 columns,
    and NO FrstState column (single-residue tables carry no class column)."""
    table = crn_singleresidue["table"]
    assert os.path.exists(table), f"missing frustration table: {table}"
    df = pd.read_csv(table, sep=r"\s+")
    assert list(df.columns) == SINGLERES_COLUMNS
    assert len(df) == 46
    assert "FrstState" not in df.columns


def test_frststate_matches_contact_cutoffs(crn_configurational):
    """The FrstState column the table classifier (utils/helpers.py) writes agrees,
    row-for-row, with the asymmetric contact cutoffs (-1 / 0.78). This pins the
    cutoff behavior on real output, using the shared 3D-view classifier as oracle."""
    df = pd.read_csv(crn_configurational["table"], sep=r"\s+")
    # helpers writes "highly"/"neutral"/"minimally"; the classifier returns the same.
    expected = df["FrstIndex"].map(classify_contact_frustration)
    assert (df["FrstState"] == expected).all()
