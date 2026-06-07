"""Tests for the pluggable mutation backends (T5 — PyRosetta backend).

The saturation-mutagenesis scan can build each point mutant with one of three
backends, selected by the ``method`` argument to ``mutate_res*``:

  * ``threading`` (default, in-house) — keep the native backbone + CB and relabel
    the residue. No extra dependency; the in-container parity reference.
  * ``pyrosetta`` (optional, license-gated) — load the full-atom pose, mutate, and
    repack side chains within a radius using the Rosetta score function.
  * ``modeller`` (license-gated label, geometrically handled in-container).

These tests pin:
  * the backend dispatch / validation surface (accepted methods, error messages,
    the actionable ImportError when PyRosetta is requested but not installed);
  * the 3->1 amino-acid map and the WT PDB serialiser the PyRosetta path relies on;
  * (when PyRosetta is installed) an end-to-end consistency gate: a PyRosetta
    single-residue mutation scan must agree with the threading scan in sign and
    rank — same backbone, AWSEM coarse-grains the side chain, so the indices track
    closely. This is the T5 "sane + consistent with threading" exit gate.

NOTE on PATH (tech-debt P1-22): the per-variant calculation spawns a bare
``python3`` subprocess, so run ``pytest`` from an activated venv (see conftest.py).
"""

import os
import warnings

import pandas as pd
import pytest

from frustrapy.analysis import mutation_backends as mb
from frustrapy.analysis.mutations import mutate_res_parallel

PYROSETTA = mb.pyrosetta_available()
pyrosetta_required = pytest.mark.skipif(
    not PYROSETTA, reason="PyRosetta not installed (optional, license-gated backend)"
)


# --------------------------------------------------------------------------- #
# Unit tests — always run, no PyRosetta needed
# --------------------------------------------------------------------------- #


def test_three_to_one_map_complete():
    """The 3->1 map covers exactly the 20 canonical amino acids."""
    assert len(mb._THREE_TO_ONE) == 20
    assert mb._THREE_TO_ONE["ALA"] == "A"
    assert mb._THREE_TO_ONE["TRP"] == "W"
    assert mb._THREE_TO_ONE["GLY"] == "G"
    # one-letter codes are unique (no collisions)
    assert len(set(mb._THREE_TO_ONE.values())) == 20


def test_method_catalogues():
    """Backend catalogues are stable and consistent."""
    assert set(mb.GEOMETRIC_METHODS) == {"threading", "modeller"}
    assert set(mb.ALL_METHODS) == {"threading", "modeller", "pyrosetta"}


def test_pyrosetta_available_is_bool():
    assert isinstance(mb.pyrosetta_available(), bool)


def _tiny_atom_df():
    """A minimal 2-atom ``pdb.atom``-shaped DataFrame for the serialiser test."""
    return pd.DataFrame(
        {
            "ATOM": ["ATOM", "ATOM"],
            "atom_num": [1, 2],
            "atom_name": ["N", "CA"],
            "alt_loc": [" ", " "],
            "res_name": ["THR", "THR"],
            "chain": ["A", "A"],
            "res_num": [1, 1],
            "insertion_code": [" ", " "],
            "x": [17.047, 16.967],
            "y": [14.099, 12.784],
            "z": [3.625, 4.338],
            "occupancy": [1.0, 1.0],
            "b_factor": [13.79, 10.80],
            "element": ["N", "C"],
        }
    )


def test_write_atom_dataframe_to_pdb(tmp_path):
    """The WT serialiser writes a Biopython-parseable PDB from an atom DataFrame."""
    out = str(tmp_path / "wt.pdb")
    mb.write_atom_dataframe_to_pdb(_tiny_atom_df(), out)
    assert os.path.exists(out)
    from Bio.PDB import PDBParser

    structure = PDBParser(QUIET=True).get_structure("wt", out)
    atoms = list(structure.get_atoms())
    assert {a.get_name() for a in atoms} == {"N", "CA"}


def test_invalid_method_rejected(crn_pdb, tmp_path):
    """An unknown backend name raises a clear ValueError listing the valid set."""
    import frustrapy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdb, _, _, _ = frustrapy.calculate_frustration(
            pdb_file=crn_pdb,
            mode="singleresidue",
            residues={"A": [1]},
            results_dir=str(tmp_path / "inv"),
            graphics=False,
            visualization=False,
            debug="ERROR",
        )
    with pytest.raises(ValueError, match="threading.*modeller.*pyrosetta"):
        mutate_res_parallel(pdb, res_num=1, chain="A", method="banana", n_cpus=1)


def test_pyrosetta_requested_but_missing_raises(crn_pdb, tmp_path, monkeypatch):
    """Requesting method='pyrosetta' without the package gives an actionable error.

    We simulate a bare install by forcing ``pyrosetta_available`` to report False;
    the scan must refuse up front with install guidance rather than failing deep in
    a worker.
    """
    import frustrapy

    monkeypatch.setattr(mb, "pyrosetta_available", lambda: False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdb, _, _, _ = frustrapy.calculate_frustration(
            pdb_file=crn_pdb,
            mode="singleresidue",
            residues={"A": [1]},
            results_dir=str(tmp_path / "miss"),
            graphics=False,
            visualization=False,
            debug="ERROR",
        )
    with pytest.raises(ImportError, match="pyrosetta-installer|PyRosetta"):
        mutate_res_parallel(pdb, res_num=1, chain="A", method="pyrosetta", n_cpus=1)


def test_plot_methods_accept_pyrosetta():
    """The delta/mutate plots validate 'pyrosetta' as a known method."""
    import inspect

    from frustrapy.visualization import plots

    for fn in (plots.plot_delta_frus, plots.plot_mutate_res):
        src = inspect.getsource(fn)
        assert "pyrosetta" in src, f"{fn.__name__} does not accept pyrosetta"


# --------------------------------------------------------------------------- #
# End-to-end consistency gate — only when PyRosetta is installed
# --------------------------------------------------------------------------- #


@pyrosetta_required
def test_pyrosetta_consistent_with_threading(crn_pdb, tmp_path):
    """T5 exit gate: a PyRosetta single-residue scan agrees with threading.

    Both backends share the native backbone; PyRosetta repacks side chains while
    threading relabels, but AWSEM coarse-grains to backbone+CB, so the per-variant
    FrstIndex must track closely: identical sign on every variant and Spearman/
    Pearson near 1. (Not byte-identical — different coordinates by design.)
    """
    import frustrapy

    res, chain = 10, "A"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdb, _, _, _ = frustrapy.calculate_frustration(
            pdb_file=crn_pdb,
            mode="singleresidue",
            residues={chain: [res]},
            results_dir=str(tmp_path / "e2e"),
            graphics=False,
            visualization=False,
            debug="ERROR",
        )
        for method in ("threading", "pyrosetta"):
            mutate_res_parallel(
                pdb, res_num=res, chain=chain, method=method, n_cpus=2
            )

    md = os.path.join(pdb.job_dir, "MutationsData")

    def _load(method):
        f = os.path.join(md, f"singleresidue_Res{res}_{method}_{chain}.txt")
        return pd.read_csv(f, sep=r"\s+").set_index("AA")["FrstIndex"]

    th, pr = _load("threading"), _load("pyrosetta")
    assert len(th) == 20 and len(pr) == 20
    both = pd.DataFrame({"th": th, "pr": pr}).dropna()
    assert len(both) == 20

    import numpy as np

    sign_agree = (np.sign(both["th"]) == np.sign(both["pr"])).mean()
    spearman = both["th"].corr(both["pr"], method="spearman")
    pearson = both["th"].corr(both["pr"], method="pearson")

    assert sign_agree == 1.0, f"sign disagreement: {both}"
    assert spearman > 0.95, f"Spearman too low: {spearman}"
    assert pearson > 0.95, f"Pearson too low: {pearson}"


@pyrosetta_required
def test_build_pyrosetta_mutant_writes_valid_pdb(crn_pdb, tmp_path):
    """build_pyrosetta_mutant produces a parseable full-atom mutant PDB and
    removes its temporary wild-type scratch file."""
    import frustrapy
    from Bio.PDB import PDBParser

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdb, _, _, _ = frustrapy.calculate_frustration(
            pdb_file=crn_pdb,
            mode="singleresidue",
            residues={"A": [10]},
            results_dir=str(tmp_path / "mut"),
            graphics=False,
            visualization=False,
            debug="ERROR",
        )

    out = str(tmp_path / "mutant.pdb")
    mb.build_pyrosetta_mutant(pdb, res_num=10, chain="A", target_aa3="ALA", output_path=out)
    assert os.path.exists(out)
    assert not os.path.exists(out + ".wt.tmp.pdb")

    structure = PDBParser(QUIET=True).get_structure("m", out)
    # residue 10 chain A is now ALA
    res10 = structure[0]["A"][10]
    assert res10.get_resname() == "ALA"
