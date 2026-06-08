"""Integration tests for the registered ``atomic`` backend (AA-INTEGRATE).

These cover the plumbing that makes the all-atom Rosetta engine a first-class,
selectable backend: the registry + lazy PyRosetta, the ``requires_lammps_prep``
seam (lammps/native unaffected), the public-API flow (``calculate_frustration`` /
``dir_frustration`` / ``dynamic_frustration``), multi-chain ChainRes mapping, the
parallel-safe decoy loop under the shared core budget, and the visualization
consuming an atomic output dir.

The Rosetta relax/repack half is license-gated (PyRosetta is not in this
container), so the real energies are MOCKED with synthetic ``ResResE`` records and
the parallel decoy loop is exercised with an injected picklable scorer. The
post-processing half (geometry, contact selection, sign-flipped FrstIndex, the
writer, renum_files, the density kernel, the plots) is real and runs end to end.
The bit-for-bit parity of the post-processor lives in ``tests/test_atomic_post.py``.
"""

import os

import numpy as np
import pytest

from frustrapy.backends import (
    DEFAULT_BACKEND,
    AtomicBackend,
    available_backends,
    get_backend,
)
from frustrapy.backends import atomic as atomic_backend
from frustrapy.backends import atomic_engine as ae
from frustrapy.backends import atomic_post as ap


# ---------------------------------------------------------------------------
# Synthetic structure helpers (no Rosetta, no external reference data).
# ---------------------------------------------------------------------------

_ATOM_NAME_FIELD = {"N": " N  ", "CA": " CA ", "C": " C  ", "O": " O  ", "CB": " CB "}


def _atom_line(serial, name, resname, chain, resnum, xyz):
    # Fixed PDB columns: "ATOM  "(1-6) serial(7-11) space(12) name(13-16) altLoc(17)
    # resName(18-20) space(21) chainID(22) resSeq(23-26) iCode+pad(27-30) xyz(31-54)
    # occ(55-60) temp(61-66) pad element(73-78).
    return (
        f"ATOM  {serial:>5d} {_ATOM_NAME_FIELD[name]} {resname:>3s} {chain:1s}{resnum:>4d}    "
        f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}  1.00  0.00          "
        f"{name[0]:>2s}\n"
    )


def _write_synthetic_pdb(path, chains):
    """Write a small all-ALA multi-chain PDB on a compact 3D grid.

    ``chains`` is a list of ``(chain_letter, n_residues)``. Residues sit on a
    spacing-4A grid centred per chain so many residue pairs are within the 10A
    contact cutoff, including pairs far apart in sequence (so intra-chain contacts
    survive the sequence-separation filter). Each residue carries N/CA/C/O/CB so the
    representative-atom geometry (CB) and the CA-based density both have what they
    need.
    """
    serial = 1
    with open(path, "w") as fh:
        for c_idx, (chain, n) in enumerate(chains):
            base = np.array([c_idx * 6.0, 0.0, 0.0])  # offset chains so they touch
            for r in range(n):
                # 3D grid position (spacing 4A) so distant-in-sequence residues can be
                # close in space.
                gx, gy, gz = r % 3, (r // 3) % 3, (r // 9) % 3
                ca = base + np.array([gx * 4.0, gy * 4.0, gz * 4.0])
                n_xyz = ca + np.array([-1.2, 0.5, 0.0])
                c_xyz = ca + np.array([1.2, 0.5, 0.0])
                o_xyz = ca + np.array([1.5, 1.5, 0.0])
                cb = ca + np.array([0.5, -1.3, 0.5])
                resnum = r + 1
                for name, xyz in (
                    ("N", n_xyz), ("CA", ca), ("C", c_xyz), ("O", o_xyz), ("CB", cb)
                ):
                    fh.write(_atom_line(serial, name, "ALA", chain, resnum, xyz))
                    serial += 1
        fh.write("END\n")


def _make_records(cids, total_offset):
    """Synthetic ResResE records for every i<j residue pair, all non-clashing
    (fa_rep=0) with a controllable total energy, so residue_energies sums cleanly."""
    recs = []
    n = len(cids)
    for i in range(n):
        for j in range(i + 1, n):
            terms = {name: 0.0 for name in ae.TERM_NAMES}
            terms["total"] = total_offset
            recs.append(
                ae.ResPairEnergy(
                    res1_key=cids[i], res2_key=cids[j], aa1="A", aa2="A", terms=terms
                )
            )
    return recs


def _install_mock_rosetta(monkeypatch, pdb_path):
    """Replace the two PyRosetta entry points and the parallel decoy loop with
    deterministic synthetic scorers, so a full calculate_frustration(backend='atomic')
    runs end to end without Rosetta and without spawning a pool."""
    geom = ap.load_contact_geometry(pdb_path)
    cids = list(geom.cid_list)

    def fake_native(path, scorefxn=None, repeats=2):
        return _make_records(cids, total_offset=-5.0)

    def fake_score_sequences(path, sequences, *, scorer=None, repeats=2, n_procs=None):
        # Vary the per-decoy energy so the pooled decoy std is > 0 (a zero std would
        # make FrstIndex non-finite).
        out = []
        seqs = list(sequences)
        for k, _seq in enumerate(seqs):
            offset = -4.0 + 0.3 * (k - len(seqs) / 2.0)
            out.append(_make_records(cids, total_offset=offset))
        return out

    monkeypatch.setattr(ae, "compute_native_pair_energies", fake_native)
    monkeypatch.setattr(atomic_backend, "score_sequences", fake_score_sequences)


# ---------------------------------------------------------------------------
# G1 -- registry, lazy import, requires_lammps_prep.
# ---------------------------------------------------------------------------

def test_atomic_is_registered_and_resolves():
    assert "atomic" in available_backends()
    backend = get_backend("atomic")
    assert isinstance(backend, AtomicBackend)
    assert backend.name == "atomic"


def test_default_backend_unchanged():
    assert DEFAULT_BACKEND == "lammps"
    assert get_backend(None).name == "lammps"


def test_requires_lammps_prep_flags():
    from frustrapy.backends import LammpsBackend, NativeBackend

    assert AtomicBackend.requires_lammps_prep is False
    # The AWSEM backends keep the default True (they consume the LAMMPS deck).
    assert LammpsBackend.requires_lammps_prep is True
    assert NativeBackend.requires_lammps_prep is True


def test_importing_atomic_does_not_import_pyrosetta():
    import sys

    # Importing the backend module (done at top of this file) must not have pulled in
    # PyRosetta -- the import is lazy and only happens inside compute_energies.
    assert "pyrosetta" not in sys.modules


def test_compute_energies_without_pyrosetta_raises_clear_error(tmp_path):
    """A real (un-mocked) atomic run raises an actionable ImportError at compute time,
    not at import time, when PyRosetta is absent."""
    if ae.pyrosetta_available():
        pytest.skip("PyRosetta is installed; the missing-dependency path is not exercised")

    pdb_path = tmp_path / "synthetic.pdb"
    _write_synthetic_pdb(str(pdb_path), [("A", 12)])

    class _Calc:
        mode = "configurational"
        n_cpus = 1

    class _Pdb:
        job_dir = str(tmp_path)
        pdb_base = "synthetic"

    backend = get_backend("atomic")
    with pytest.raises(ImportError) as exc:
        backend.compute_energies(_Calc(), _Pdb())
    assert "pyrosetta" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# G3 -- parallel decoy loop under the shared budget.
# ---------------------------------------------------------------------------

def test_resolve_decoy_workers_budget():
    from frustrapy.utils.concurrency import cpu_budget

    cores = cpu_budget()
    # Single task -> no pool.
    assert atomic_backend._resolve_decoy_workers(1) == 1
    # Many tasks, capped by the core budget and the requested width.
    assert atomic_backend._resolve_decoy_workers(1000) <= cores
    assert atomic_backend._resolve_decoy_workers(1000, n_procs=2) == min(2, cores)
    # Never more workers than tasks.
    assert atomic_backend._resolve_decoy_workers(3, n_procs=1000) == min(3, cores)


# Module-level picklable scorer so the ProcessPool can ship it to workers.
def _toy_scorer(pdb_path, sequence, repeats=2):
    # Energy is a deterministic function of the sequence so serial and parallel runs
    # produce identical records.
    total = float(sum(ord(c) for c in sequence))
    terms = {name: 0.0 for name in ae.TERM_NAMES}
    terms["total"] = total
    return [ae.ResPairEnergy(res1_key="A1", res2_key="A2", aa1="A", aa2="A", terms=terms)]


def test_score_sequences_serial_and_parallel_agree():
    seqs = [f"ACDEFG{i:02d}" for i in range(8)]
    serial = atomic_backend.score_sequences("x.pdb", seqs, scorer=_toy_scorer, n_procs=1)
    parallel = atomic_backend.score_sequences("x.pdb", seqs, scorer=_toy_scorer, n_procs=4)
    assert len(serial) == len(parallel) == len(seqs)
    s_tot = [r[0].total for r in serial]
    p_tot = [r[0].total for r in parallel]
    assert s_tot == p_tot  # order preserved, results identical


def test_score_sequences_empty():
    assert atomic_backend.score_sequences("x.pdb", [], scorer=_toy_scorer) == []


# ---------------------------------------------------------------------------
# G2 -- multi-chain ChainRes mapping through the shared process_results.
# ---------------------------------------------------------------------------

def _synthetic_engine_result(geom, n_decoys=6):
    """An EngineResult with deterministic, non-degenerate native + decoy per-residue
    energies (built directly, no Rosetta)."""
    native = {cid: -2.0 - 0.01 * k for k, cid in enumerate(geom.cid_list)}
    decoys = []
    for d in range(n_decoys):
        decoys.append(
            {cid: -1.0 + 0.2 * (d - n_decoys / 2.0) + 0.01 * k
             for k, cid in enumerate(geom.cid_list)}
        )
    return ae.EngineResult(native_residue_energy=native, decoy_residue_energies=decoys)


def test_multichain_chainres_through_renum(tmp_path):
    """A two-chain atomic .dat parses through the shared renum_files into the canonical
    14-column table with correct ChainRes1/ChainRes2 per chain and at least one
    cross-chain contact."""
    import pandas as pd

    from frustrapy.utils.helpers import pdb_equivalences, renum_files

    pdb_path = tmp_path / "duo.pdb"
    _write_synthetic_pdb(str(pdb_path), [("A", 12), ("B", 12)])

    geom = ap.load_contact_geometry(str(pdb_path))
    # cid_list spans both chains.
    assert {c[0] for c in geom.cid_list} == {"A", "B"}

    engine = _synthetic_engine_result(geom)
    out = tmp_path / "tertiary_frustration.dat"
    # Lower seq_sep so the small chains still yield intra-chain contacts too.
    contacts = ap.write_tertiary_frustration(
        str(out), geom, engine_result=engine, seq_sep=2
    )
    assert len(contacts) > 0

    # Real equivalences map from the same PDB (the calculator-level helper).
    pdb_equivalences(str(pdb_path), str(tmp_path))
    renum_files("duo", str(tmp_path), "configurational",
                equivalences_file="duo.pdb_equivalences.txt")

    df = pd.read_csv(tmp_path / "duo.pdb_configurational", sep=r"\s+")
    expected_cols = [
        "Res1", "Res2", "ChainRes1", "ChainRes2", "DensityRes1", "DensityRes2",
        "AA1", "AA2", "NativeEnergy", "DecoyEnergy", "SDEnergy", "FrstIndex",
        "Welltype", "FrstState",
    ]
    assert list(df.columns) == expected_cols
    # Both chains appear in the ChainRes columns.
    chains_seen = set(df["ChainRes1"].astype(str)) | set(df["ChainRes2"].astype(str))
    assert {"A", "B"}.issubset(chains_seen)
    # At least one genuine cross-chain contact.
    cross = df[df["ChainRes1"].astype(str) != df["ChainRes2"].astype(str)]
    assert len(cross) > 0
    # Density sentinel means Welltype never water-mediated.
    assert "water-mediated" not in set(df["Welltype"])
    assert set(df["FrstState"]).issubset({"minimally", "neutral", "highly"})


# ---------------------------------------------------------------------------
# G1 + G4 -- full public-API flow + visualization, with mocked Rosetta.
# ---------------------------------------------------------------------------

def test_calculate_frustration_atomic_end_to_end(tmp_path, monkeypatch):
    """calculate_frustration(backend='atomic') runs the whole pipeline (skip LAMMPS
    prep, atomic compute_energies, shared process_results + density, graphics) on a
    synthetic structure with Rosetta mocked, returns the 4-tuple, and the plots
    consume the atomic output dir without special-casing (G4 smoke)."""
    import frustrapy

    src = tmp_path / "src"
    src.mkdir()
    pdb_path = src / "synth.pdb"
    _write_synthetic_pdb(str(pdb_path), [("A", 27)])

    _install_mock_rosetta(monkeypatch, str(pdb_path))

    results_dir = tmp_path / "out"
    result = frustrapy.calculate_frustration(
        pdb_file=str(pdb_path),
        mode="configurational",
        backend="atomic",
        results_dir=str(results_dir),
        graphics=True,
        visualization=False,
        debug=False,
    )
    # 4-tuple contract preserved.
    assert len(result) == 4
    pdb, plots, density, single_res = result
    assert pdb.mode == "configurational"

    fdata = os.path.join(pdb.job_dir, "FrustrationData")
    table = os.path.join(fdata, f"{pdb.pdb_base}.pdb_configurational")
    assert os.path.exists(table)
    assert os.path.exists(os.path.join(fdata, "tertiary_frustration.dat"))
    assert os.path.exists(os.path.join(fdata, f"{pdb.pdb_base}.pdb_configurational_5adens"))
    assert density is not None

    import pandas as pd

    df = pd.read_csv(table, sep=r"\s+")
    assert len(df) > 0
    assert np.isfinite(df["FrstIndex"]).all()
    # Graphics ran and produced the contact-map plot (G4: viz consumed atomic output).
    assert "plot_contact_map" in plots


def test_dir_frustration_threads_atomic_backend(tmp_path, monkeypatch):
    """dir_frustration(backend='atomic') forwards the selector to each per-structure
    calculation (the serial batch path, Rosetta mocked)."""
    import frustrapy

    pdbs = tmp_path / "pdbs"
    pdbs.mkdir()
    pdb_path = pdbs / "synth.pdb"
    _write_synthetic_pdb(str(pdb_path), [("A", 27)])
    _install_mock_rosetta(monkeypatch, str(pdb_path))

    results_dir = tmp_path / "out"
    plots_dict, density = frustrapy.dir_frustration(
        pdbs_dir=str(pdbs),
        mode="configurational",
        backend="atomic",
        results_dir=str(results_dir),
        graphics=False,
        visualization=False,
    )
    assert "synth" in plots_dict
    table = os.path.join(
        str(results_dir), "synth.done", "FrustrationData", "synth.pdb_configurational"
    )
    assert os.path.exists(table)


def test_no_lammps_prep_means_no_lammps_artifacts(tmp_path, monkeypatch):
    """The atomic path must not have produced any LAMMPS deck files (it skipped the
    PdbCoords2Lammps.sh prep)."""
    import frustrapy

    src = tmp_path / "src"
    src.mkdir()
    pdb_path = src / "synth.pdb"
    _write_synthetic_pdb(str(pdb_path), [("A", 27)])
    _install_mock_rosetta(monkeypatch, str(pdb_path))

    results_dir = tmp_path / "out"
    pdb, _plots, _density, _single = frustrapy.calculate_frustration(
        pdb_file=str(pdb_path),
        mode="configurational",
        backend="atomic",
        results_dir=str(results_dir),
        graphics=False,
        visualization=False,
    )
    # The LAMMPS-only artifacts are absent (no deck prep).
    for artifact in (".coord", ".in"):
        assert not os.path.exists(os.path.join(pdb.job_dir, f"{pdb.pdb_base}{artifact}"))
    assert not os.path.exists(os.path.join(pdb.job_dir, "fix_backbone_coeff.data"))
