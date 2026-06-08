"""In-container parity gate for the atomic Frustratometer post-processor (AA-OUTPUT).

This is the mission with a real local parity gate. The audit regenerated a golden
``tertiary_frustration.dat`` by running the reference ``Frust_Post_public.py`` over
the shipped Rosetta logs, with no Rosetta (``docs/atomic/golden/``). These tests
feed the SAME shipped logs through the ported FrustraPy pipeline
(:mod:`frustrapy.backends.atomic_post`) and prove the result matches the golden:

* the contact SET is identical (geometry / representative-atom contact definition
  reproduced, not just the per-contact numbers);
* the ``FrstIndex`` is bit-for-bit at 3-decimal print precision, with ``max|delta|``
  within print precision and Spearman 1.0;
* the **sign** is correct row-for-row (the AWSEM negation of the reference Z);
* the written file parses through the shared ``process_results`` into the canonical
  14-column contact table with no special-casing.

The shipped reference logs + structure are an external trimmed copy at
``/workspace/atomic_frustratometer_ref/example_output/`` (not committed). Tests that
need them skip cleanly if the directory is absent; override with the
``ATOMIC_FRUST_REF`` environment variable. The pure unit tests (sign convention,
sentinel, contact selection, writer layout) run with no reference data.
"""

import os

import numpy as np
import pytest

from frustrapy.backends import atomic_engine as ae
from frustrapy.backends import atomic_post as ap

# ---------------------------------------------------------------------------
# Reference-data location (external; skip if absent).
# ---------------------------------------------------------------------------

REF_DIR = os.environ.get(
    "ATOMIC_FRUST_REF",
    "/workspace/atomic_frustratometer_ref/example_output",
)
NATIVE_PDB = os.path.join(REF_DIR, "native.pdb")
NATIVE_LOG = os.path.join(REF_DIR, "native.log")
GOLDEN_DAT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "docs", "atomic", "golden", "tertiary_frustration.dat",
)
N_DECOYS = 50

_REF_AVAILABLE = (
    os.path.exists(NATIVE_PDB)
    and os.path.exists(NATIVE_LOG)
    and all(os.path.exists(os.path.join(REF_DIR, f"{i}.log")) for i in range(1, N_DECOYS + 1))
)
requires_ref = pytest.mark.skipif(
    not _REF_AVAILABLE,
    reason=f"reference Rosetta logs/structure not found under {REF_DIR} (set ATOMIC_FRUST_REF)",
)


def _decoy_logs():
    return [os.path.join(REF_DIR, f"{i}.log") for i in range(1, N_DECOYS + 1)]


def _spearman(a, b):
    """Spearman rank correlation, computed with numpy (scipy.stats is not a hard
    dependency and is not fully installed in this container)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _read_golden(path):
    """Golden rows as ``(i, j, r_ij, E_native, decoy_mean, decoy_std)``.

    Reference 16-column layout: ``[0]i [1]j [2]chain_i [3]chain_j [4..6]xyz_i
    [7..9]xyz_j [10]r_ij [11]AA_i [12]AA_j [13]E_native [14]decoy_mean
    [15]decoy_std``.
    """
    rows = []
    with open(path) as f:
        for line in f:
            s = line.split()
            if not s:
                continue
            rows.append(
                (int(s[0]), int(s[1]), float(s[10]), float(s[13]), float(s[14]), float(s[15]))
            )
    return rows


def _read_written(path):
    """Written AWSEM 19-column rows as ``(i0, j0, r_ij, native, decoy, sd, frst)``,
    with ``i0 = Res1 - 1`` so indices line up with the golden's 0-based residues.

    FrustraPy layout: ``[0]Res1 [1]Res2 [2]i_chain [3]j_chain [4..6]xyz_i
    [7..9]xyz_j [10]r_ij [11]DensityRes1 [12]DensityRes2 [13]AA1 [14]AA2
    [15]NativeEnergy [16]DecoyEnergy [17]SDEnergy [18]FrstIndex``.
    """
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            s = line.split()
            if not s:
                continue
            rows.append(
                (
                    int(s[0]) - 1, int(s[1]) - 1, float(s[10]),
                    float(s[15]), float(s[16]), float(s[17]), float(s[18]),
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Pure unit tests (no reference data).
# ---------------------------------------------------------------------------

def test_sign_convention_is_awsem_negation():
    """``FrstIndex = (decoy_mean - native) / sd`` -- the negation of the reference
    Z. A favorable (very negative) native energy must give a POSITIVE index
    (minimally frustrated); an unfavorable native must give a negative index."""
    # Favorable native (well below the decoy mean) -> positive -> minimally.
    fi = ap.atomic_frustration_index(native_energy=-9.0, decoy_mean=-5.0, decoy_std=2.0)
    assert fi == pytest.approx((-5.0 - -9.0) / 2.0)
    assert fi > 0
    # Unfavorable native (above the decoy mean) -> negative -> highly.
    fi2 = ap.atomic_frustration_index(native_energy=-1.0, decoy_mean=-5.0, decoy_std=2.0)
    assert fi2 < 0
    # It is exactly the negation of the reference frust = (native - mean)/sd.
    ref_frust = (-9.0 - -5.0) / 2.0
    assert ap.atomic_frustration_index(-9.0, -5.0, 2.0) == pytest.approx(-ref_frust)


def test_density_sentinel_suppresses_water_mediated():
    """The burial-density sentinel equals the AWSEM water-mediated cutoff so the
    parser's ``density < 2.6`` test is False -- no atomic contact is ever labeled
    water-mediated."""
    from frustrapy.core.constants import WATER_MEDIATED_DENSITY_CUTOFF

    assert ap.BURIAL_DENSITY_SENTINEL == WATER_MEDIATED_DENSITY_CUTOFF
    assert not (ap.BURIAL_DENSITY_SENTINEL < WATER_MEDIATED_DENSITY_CUTOFF)


def _toy_geometry():
    """Four residues on one chain in a line 4 A apart; rep atoms at x = 0,4,8,12."""
    coords = np.array(
        [[0.0, 0, 0], [4.0, 0, 0], [8.0, 0, 0], [12.0, 0, 0]], dtype=float
    )
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    return ap.ContactGeometry(
        cid_list=("A1", "A2", "A3", "A4"),
        aa=("A", "C", "D", "E"),
        rep_coords=coords,
        dist=dist,
    )


def test_select_contacts_distance_and_separation():
    geom = _toy_geometry()
    # seq_sep = 0 means any |i-j| > 0 qualifies; distance cutoff filters the rest.
    contacts = ap.select_contacts(geom, seq_sep=0, distance_cutoff=10.0)
    # i<j pairs with distance <= 10: (0,1)=4, (0,2)=8, (1,2)=4, (1,3)=8, (2,3)=4.
    # (0,3)=12 > 10 is excluded.
    assert contacts == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    # Raising seq_sep to 1 drops the adjacent pairs (|i-j| == 1).
    contacts2 = ap.select_contacts(geom, seq_sep=1, distance_cutoff=10.0)
    assert contacts2 == [(0, 2), (1, 3)]
    # Ordering is i ascending then j ascending (matches the reference row order).
    assert contacts == sorted(contacts)


def test_writer_layout_and_sentinel(tmp_path):
    """The writer emits the 19-column AWSEM layout: 1-based residue indices, the
    density sentinel in columns 11/12, and the sign-flipped FrstIndex in column
    18."""
    geom = _toy_geometry()
    # One decoy whose contact energies are non-zero everywhere.
    engine = ae.EngineResult(
        native_residue_energy={"A1": -2.0, "A2": -1.0, "A3": -3.0, "A4": -1.5},
        decoy_residue_energies=[
            {"A1": 0.5, "A2": 0.3, "A3": 0.4, "A4": 0.2},
            {"A1": 1.0, "A2": 0.8, "A3": 0.9, "A4": 0.7},
        ],
    )
    out = tmp_path / "tertiary_frustration.dat"
    contacts = ap.write_tertiary_frustration(str(out), geom, engine, seq_sep=0)
    lines = [l for l in out.read_text().splitlines() if not l.startswith("#")]
    assert len(lines) == len(contacts)
    first = lines[0].split()
    assert len(first) == 19  # full AWSEM column count
    assert (int(first[0]), int(first[1])) == (1, 2)  # 1-based indices
    assert float(first[11]) == ap.BURIAL_DENSITY_SENTINEL
    assert float(first[12]) == ap.BURIAL_DENSITY_SENTINEL
    # Column 18 is exactly (decoy_mean - native)/sd for the first contact, where the
    # decoy mean/sd is the protein-wide statistic over the SAME contact set the
    # writer used (not a per-contact pool).
    cid_pairs = [(geom.cid_list[i], geom.cid_list[j]) for (i, j) in contacts]
    summ = engine.summarize_contacts(cid_pairs)[0]
    expected = ap.atomic_frustration_index(summ.native_energy, summ.decoy_mean, summ.decoy_std)
    assert float(first[18]) == pytest.approx(expected, abs=5e-4)


# ---------------------------------------------------------------------------
# O2 -- the parity gate against the regenerated golden (needs the shipped logs).
# ---------------------------------------------------------------------------

@requires_ref
@pytest.mark.slow
def test_post_processor_matches_golden(tmp_path):
    """Feed the shipped reference logs through the ported pipeline and diff the
    resulting tertiary_frustration.dat against the golden fixture: contact set
    identical, FrstIndex bit-for-bit at print precision, Spearman 1.0, sign correct
    row-for-row, and native/decoy/sd within print precision."""
    out = tmp_path / "tertiary_frustration.dat"
    ap.write_tertiary_frustration_from_logs(
        str(out), NATIVE_PDB, NATIVE_LOG, _decoy_logs()
    )

    golden = _read_golden(GOLDEN_DAT)
    written = _read_written(str(out))
    assert len(golden) == 328
    assert len(written) == 328

    # 1. Contact set identical (geometry / contact definition reproduced).
    gset = {(i, j) for (i, j, *_rest) in golden}
    mset = {(i, j) for (i, j, *_rest) in written}
    assert gset == mset, f"contact set differs: g-m={gset - mset}, m-g={mset - gset}"

    gmap = {(i, j): (rij, ne, dm, ds) for (i, j, rij, ne, dm, ds) in golden}

    max_dfrst = max_drij = max_dne = max_dde = max_dsd = 0.0
    bit_mismatch = 0
    sign_disagree = 0
    fi_mine, fi_gold = [], []
    for (i, j, rij, ne, de, sd, fi) in written:
        grij, gne, gdm, gds = gmap[(i, j)]
        # The golden omits the FrstIndex column; the AWSEM-sign reference index is
        # (decoy_mean - E_native)/decoy_std, the negation of the reference frust.
        gfi = (gdm - gne) / gds
        max_dfrst = max(max_dfrst, abs(fi - gfi))
        max_drij = max(max_drij, abs(rij - grij))
        max_dne = max(max_dne, abs(ne - gne))
        max_dde = max(max_dde, abs(de - gdm))
        max_dsd = max(max_dsd, abs(sd - gds))
        # Bit-for-bit at 3-decimal print precision: formatting the golden's AWSEM-sign
        # index the same way must reproduce the written column exactly.
        if f"{gfi:8.3f}".strip() != f"{fi:8.3f}".strip():
            bit_mismatch += 1
        # Sign agreement row-for-row (ignore rows that round to exactly 0).
        if round(gfi, 3) != 0 and np.sign(fi) != np.sign(round(gfi, 3)):
            sign_disagree += 1
        fi_mine.append(fi)
        # Compare ranks at the file's own print precision: since the 3-decimal
        # strings match bit-for-bit (asserted above), the rounded series are
        # identical and Spearman is exactly 1.0. (Comparing the written 3-decimal
        # column against full-precision golden would create spurious rank ties.)
        fi_gold.append(round(gfi, 3))

    assert bit_mismatch == 0, f"{bit_mismatch} rows differ at 3-decimal print precision"
    assert sign_disagree == 0, f"{sign_disagree} rows disagree in sign"
    # Within %8.3f print precision (half a ULP at 3 decimals is 5e-4).
    assert max_dfrst < 1e-3, f"max|dFrstIndex| = {max_dfrst}"
    assert max_drij < 1e-3 and max_dne < 1e-3 and max_dde < 1e-3 and max_dsd < 1e-3
    assert _spearman(fi_mine, fi_gold) == pytest.approx(1.0, abs=1e-12)


@requires_ref
def test_geometry_contact_set_matches_golden():
    """The representative-atom geometry + contact selection alone (no energies)
    reproduces the golden contact set and the per-contact distance, proving the
    contact DEFINITION matches, not just the numbers downstream."""
    geom = ap.load_contact_geometry(NATIVE_PDB)
    assert geom.n_residues == 92
    assert geom.cid_list == tuple(f"A{n}" for n in range(1, 93))

    contacts = ap.select_contacts(geom)
    golden = _read_golden(GOLDEN_DAT)
    gset = {(i, j) for (i, j, *_rest) in golden}
    assert set(contacts) == gset
    assert len(contacts) == 328

    # Representative-atom distance matches the golden r_ij column to print precision.
    grij = {(i, j): rij for (i, j, rij, *_rest) in golden}
    max_dr = max(abs(geom.dist[i, j] - grij[(i, j)]) for (i, j) in contacts)
    assert max_dr < 1e-3, f"max|dr_ij| = {max_dr}"


@requires_ref
@pytest.mark.slow
def test_written_file_parses_through_process_results(tmp_path):
    """The AWSEM-format file the writer emits is parsed by the shared
    ``process_results`` (``renum_files``) into the canonical 14-column contact
    table with no special-casing, and the Welltype never comes out
    'water-mediated' (the density sentinel)."""
    import pandas as pd

    from frustrapy.utils.helpers import renum_files

    job = tmp_path
    geom = ap.load_contact_geometry(NATIVE_PDB)
    engine = ae.load_engine_result_from_logs(NATIVE_LOG, _decoy_logs())
    ap.write_tertiary_frustration(str(job / "tertiary_frustration.dat"), geom, engine)

    # Minimal equivalences file: 'chain_letter global_index pdb_resnum' per residue,
    # which is what renum_files maps the AWSEM residue numbers back through.
    with open(job / "native.pdb_equivalences.txt", "w") as f:
        for k, cid in enumerate(geom.cid_list):
            f.write(f"{cid[0]} {k + 1} {cid[1:]}\n")

    renum_files("native", str(job), "configurational")
    df = pd.read_csv(job / "native.pdb_configurational", sep=r"\s+")

    expected_cols = [
        "Res1", "Res2", "ChainRes1", "ChainRes2", "DensityRes1", "DensityRes2",
        "AA1", "AA2", "NativeEnergy", "DecoyEnergy", "SDEnergy", "FrstIndex",
        "Welltype", "FrstState",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 328
    assert set(df["FrstState"]).issubset({"minimally", "neutral", "highly"})
    assert "water-mediated" not in set(df["Welltype"])
    assert set(df["Welltype"]).issubset({"short", "long"})
