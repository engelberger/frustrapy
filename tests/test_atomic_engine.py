"""In-container validation of the atomic (Rosetta) Frustratometer engine helpers.

These tests cover exactly the half of the AA-ENGINE that needs no Rosetta:

* the seedable permutation decoy generator (composition preserved, count, seed
  reproducibility);
* the ``ResResE`` per-pair log parser, against the SHIPPED reference logs;
* the per-residue / per-contact aggregation, validated end-to-end by reproducing
  the golden ``tertiary_frustration.dat`` ``E_native`` / ``decoy_mean`` /
  ``decoy_std`` columns (``docs/atomic/golden/``).

The PyRosetta relax/repack path (``compute_native_pair_energies`` /
``compute_decoy_pair_energies``) is the MAINTAINER step and is only checked here for
a graceful import error when PyRosetta is absent; its numeric validation lives in
``docs/atomic/AA_MAINTAINER_RUNBOOK.md``.

The shipped reference logs are an external trimmed copy at
``/workspace/atomic_frustratometer_ref/example_output/`` (26 MB; not committed to
this repo). Tests that need them skip cleanly if the directory is absent. Override
the location with the ``ATOMIC_FRUST_REF`` environment variable.
"""

import os

import pytest

from frustrapy.backends import atomic_engine as ae

# ---------------------------------------------------------------------------
# Reference-log location (external; skip if absent).
# ---------------------------------------------------------------------------

REF_DIR = os.environ.get(
    "ATOMIC_FRUST_REF",
    "/workspace/atomic_frustratometer_ref/example_output",
)
NATIVE_LOG = os.path.join(REF_DIR, "native.log")
GOLDEN_DAT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "docs", "atomic", "golden", "tertiary_frustration.dat",
)
N_DECOYS = 50

_REF_AVAILABLE = os.path.exists(NATIVE_LOG) and all(
    os.path.exists(os.path.join(REF_DIR, f"{i}.log")) for i in range(1, N_DECOYS + 1)
)
requires_ref = pytest.mark.skipif(
    not _REF_AVAILABLE,
    reason=f"reference Rosetta logs not found under {REF_DIR} (set ATOMIC_FRUST_REF)",
)


def _decoy_logs():
    return [os.path.join(REF_DIR, f"{i}.log") for i in range(1, N_DECOYS + 1)]


# ---------------------------------------------------------------------------
# E2 -- permutation decoy generator (pure; no Rosetta, no reference data)
# ---------------------------------------------------------------------------

def test_decoy_count_and_composition_preserved():
    native = "DIQVQVNIDDNGKNFDYTYTVTTESELQKVL"
    decoys = ae.generate_decoy_sequences(native, 7, seed=0)
    assert len(decoys) == 7
    for d in decoys:
        assert len(d) == len(native)
        # composition-preserving: a permutation has the native multiset of letters.
        assert sorted(d) == sorted(native)


def test_decoy_seed_is_reproducible():
    native = "DIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELDYIKKQGAKRVRISITARTKK"
    a = ae.generate_decoy_sequences(native, 5, seed=42)
    b = ae.generate_decoy_sequences(native, 5, seed=42)
    c = ae.generate_decoy_sequences(native, 5, seed=43)
    assert a == b  # same seed -> identical ensemble
    assert a != c  # different seed -> different ensemble (sequence long enough)


def test_decoy_unseeded_still_valid():
    native = "ACDEFGHIKLMNPQRSTVWY"
    decoys = ae.generate_decoy_sequences(native, 3, seed=None)
    assert len(decoys) == 3
    for d in decoys:
        assert sorted(d) == sorted(native)


def test_decoy_count_edges():
    assert ae.generate_decoy_sequences("ACDE", 0, seed=0) == []
    with pytest.raises(ValueError):
        ae.generate_decoy_sequences("ACDE", -1, seed=0)


# ---------------------------------------------------------------------------
# Scheme energy decomposition (pure; constructed record)
# ---------------------------------------------------------------------------

def test_scheme_energy_decomposition():
    # First native.log pair D_A1 / I_A2: total 0.703, fa_atr -1.547, fa_rep 0.267,
    # pro_close 0.0, dslf_fa13 0.0.
    terms = {name: 0.0 for name in ae.TERM_NAMES}
    terms.update(fa_atr=-1.547, fa_rep=0.267, total=0.703)
    rec = ae.ResPairEnergy("A1", "A2", "D", "I", terms)
    assert rec.scheme_energy("Function1") == pytest.approx(0.703 - 0.267)
    assert rec.scheme_energy("Function2") == pytest.approx(0.703 - 0.267 - (-1.547))
    assert rec.scheme_energy("Packing") == pytest.approx(0.703)
    with pytest.raises(ValueError):
        rec.scheme_energy("nope")


# ---------------------------------------------------------------------------
# E1/E2 -- ResResE parser against the shipped logs
# ---------------------------------------------------------------------------

@requires_ref
def test_parse_native_log():
    records = ae.parse_resrese_log(NATIVE_LOG)
    # native.log has 2122 ResResE lines: 2 headers (Res1 / nonzero) + 2120 data.
    assert len(records) == 2120
    first = records[0]
    assert (first.aa1, first.res1_key) == ("D", "A1")
    assert (first.aa2, first.res2_key) == ("I", "A2")
    assert first.fa_atr == pytest.approx(-1.547)
    assert first.fa_rep == pytest.approx(0.267)
    assert first.total == pytest.approx(0.703)
    assert first.scheme_energy("Function1") == pytest.approx(0.436)
    # Every residue key is chain A, numbered within the 92-residue chain.
    for r in records:
        assert r.res1_key[0] == "A" and r.res2_key[0] == "A"


@requires_ref
def test_parse_decoy_log():
    records = ae.parse_resrese_log(os.path.join(REF_DIR, "1.log"))
    assert records, "decoy log parsed to zero records"
    first = records[0]
    # Decoy 1 threads a permuted sequence: first pair is R_A1 / V_A2, total -0.492.
    assert (first.aa1, first.res1_key) == ("R", "A1")
    assert (first.aa2, first.res2_key) == ("V", "A2")
    assert first.fa_rep == pytest.approx(0.475)
    assert first.total == pytest.approx(-0.492)
    assert first.scheme_energy("Function1") == pytest.approx(-0.492 - 0.475)


@requires_ref
def test_residue_energies_keys():
    records = ae.parse_resrese_log(NATIVE_LOG)
    ene = ae.residue_energies(records, scheme="Function1")
    # Every key is a chain-A residue within 1..92; the clean monomer touches all of
    # them, so all 92 appear.
    assert set(ene) == {f"A{n}" for n in range(1, 93)}
    assert all(isinstance(v, float) for v in ene.values())


# ---------------------------------------------------------------------------
# E3 -- end-to-end: reproduce the golden tertiary_frustration.dat columns
# ---------------------------------------------------------------------------

def _read_golden(path):
    """Read (i, j, E_native, decoy_mean, decoy_std) from the golden .dat rows."""
    rows = []
    with open(path) as f:
        for line in f:
            s = line.split()
            if not s:
                continue
            rows.append((int(s[0]), int(s[1]), float(s[13]), float(s[14]), float(s[15])))
    return rows


@requires_ref
@pytest.mark.slow
def test_engine_reproduces_golden_columns():
    """The engine's native + decoy aggregation reproduces, contact for contact,
    the golden fixture's E_native / decoy_mean / decoy_std columns. This validates
    E1 (native energies), E2 (decoy parsing) and E3 (per-contact summary) end to
    end against the parity oracle, with no Rosetta."""
    golden = _read_golden(GOLDEN_DAT)
    assert len(golden) == 328

    # cid_list ordering: residues sorted by number; for this clean monomer this is
    # the reference get_index order (PDB order), so golden index i -> "A{i+1}".
    native_records = ae.parse_resrese_log(NATIVE_LOG)
    keys = sorted(
        {r.res1_key for r in native_records} | {r.res2_key for r in native_records},
        key=lambda k: int(k[1:]),
    )
    assert keys == [f"A{n}" for n in range(1, 93)]

    result = ae.load_engine_result_from_logs(NATIVE_LOG, _decoy_logs(), scheme="Function1")
    assert result.n_good == 50 and result.n_bad == 0

    contacts = [(keys[i], keys[j]) for (i, j, _, _, _) in golden]
    summaries = result.summarize_contacts(contacts)
    assert len(summaries) == len(golden)

    # The reference assigns one protein-wide decoy mean/sd to every contact.
    ref_means = {round(dm, 9) for (_, _, _, dm, _) in golden}
    assert len(ref_means) == 1

    for (i, j, e_native, decoy_mean, decoy_std), s in zip(golden, summaries):
        assert s.native_energy == pytest.approx(e_native, abs=1e-6)
        assert s.decoy_mean == pytest.approx(decoy_mean, abs=1e-6)
        assert s.decoy_std == pytest.approx(decoy_std, abs=1e-6)
        assert s.n_decoys == 50


# ---------------------------------------------------------------------------
# Lazy / graceful PyRosetta import (engine must never break import frustrapy)
# ---------------------------------------------------------------------------

def test_import_is_pyrosetta_free():
    import sys

    # Importing the engine module must not pull in PyRosetta.
    assert "pyrosetta" not in sys.modules
    assert callable(ae.pyrosetta_available)
    assert isinstance(ae.pyrosetta_available(), bool)


@pytest.mark.skipif(
    ae.pyrosetta_available(), reason="PyRosetta is installed; graceful-error path not exercised"
)
def test_pyrosetta_path_raises_actionable_error():
    # With PyRosetta absent, the maintainer-only compute functions must raise a
    # clear, actionable ImportError (not an AttributeError or a silent failure).
    with pytest.raises(ImportError) as exc:
        ae.compute_native_pair_energies("does_not_matter.pdb")
    assert "pyrosetta-installer" in str(exc.value)
    with pytest.raises(ImportError):
        ae.compute_decoy_pair_energies("does_not_matter.pdb", "ACDEFGHIKL")
