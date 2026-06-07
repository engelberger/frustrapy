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
