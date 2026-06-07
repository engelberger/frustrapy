"""Pluggable side-chain mutation backends for the saturation-mutagenesis scan.

Every backend takes a wild-type structure and writes a single point-mutant PDB to
``output_path``. The frustration engine then runs on that file *identically*
regardless of backend, so the backend choice changes only **how the mutant
coordinates are built**, never how frustration is scored. This keeps the parity
spine intact: the AWSEM/LAMMPS energy model and the index math are untouched.

Backends
--------
``threading`` (default, in-house, no extra dependency)
    Keep the native backbone (N, CA, C, O) and CB, relabel the residue to the
    target identity, and synthesise CB for GLY->X from ideal tetrahedral geometry.
    This is the historical FrustraPy behaviour and the in-container parity
    reference (no license, runs in CI). Implemented inline in
    :func:`frustrapy.analysis.mutations._process_amino_acid`.

``pyrosetta`` (optional, license-gated)
    Load the full-atom wild-type pose, mutate the target residue to the target
    identity, and repack side chains within ``pack_radius`` Angstroms using the
    Rosetta score function, then dump the repacked full-atom mutant. Produces
    physically realistic rotamers rather than a relabelled backbone. PyRosetta is
    free for academic/non-commercial use but requires accepting the RosettaCommons
    license and is NOT redistributed with this package; install it separately
    (see ``install_pyrosetta`` below).

``modeller`` (optional, license-gated)
    Retained as an allowed label for backward compatibility / parity with
    ``frustratometeR``'s MODELLER path; not validated in-container (no license).

PyRosetta licensing
-------------------
PyRosetta is distributed by RosettaCommons under a license that is free for
academic and non-commercial users but must be obtained/accepted by the user; it
is not bundled here and is an *optional* dependency declared under the
``pyrosetta`` extra in ``pyproject.toml``. To install it::

    pip install pyrosetta-installer
    python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"

"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Three-letter -> one-letter for the 20 canonical amino acids. PyRosetta's
# ``mutate_residue`` expects the one-letter code of the target identity.
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Methods that build mutant coordinates without PyRosetta. ``modeller`` is kept
# as an accepted label (parity with frustratometeR) but is geometrically handled
# by the threading path in-container (no MODELLER license available here).
GEOMETRIC_METHODS = ("threading", "modeller")
ALL_METHODS = ("threading", "modeller", "pyrosetta")

# One PyRosetta init per process. Each multiprocessing worker is a fresh process
# (pyrosetta is lazy-imported, so the parent never initialises it); the worker
# initialises on its first mutation and reuses the singleton thereafter.
_PYROSETTA_INITED = False


def pyrosetta_available() -> bool:
    """True iff the optional ``pyrosetta`` package can be imported."""
    try:
        import pyrosetta  # noqa: F401

        return True
    except Exception:  # ImportError, or a partial/broken install
        return False


def _ensure_pyrosetta():
    """Import PyRosetta and initialise it once per process (muted).

    Raises ``ImportError`` with an actionable message if PyRosetta is not
    installed, so the caller can surface the install instructions.
    """
    global _PYROSETTA_INITED
    try:
        import pyrosetta
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            "method='pyrosetta' requires the optional PyRosetta package, which is "
            "not installed. PyRosetta is free for academic/non-commercial use under "
            "the RosettaCommons license but must be installed separately:\n"
            "  pip install pyrosetta-installer\n"
            '  python -c "import pyrosetta_installer; '
            'pyrosetta_installer.install_pyrosetta()"'
        ) from exc

    if not _PYROSETTA_INITED:
        # -mute all silences the per-pose banner spam; the structure-tolerance
        # flags let Rosetta load the cleaned/backbone-completed inputs FrustraPy
        # feeds it without aborting on minor PDB quirks.
        pyrosetta.init(
            extra_options="-mute all -ignore_unrecognized_res true "
            "-ignore_zero_occupancy false",
            silent=True,
        )
        _PYROSETTA_INITED = True
    return pyrosetta


def write_atom_dataframe_to_pdb(atom_df, output_path: str) -> None:
    """Serialise a FrustraPy ``pdb.atom`` DataFrame to a fixed-width PDB file.

    Mirrors the manual formatting used by the threading path so the wild-type
    structure handed to PyRosetta is byte-for-byte the same coordinates the
    threading backend would mutate, then normalises columns via Biopython.
    """
    import pandas as pd
    from Bio.PDB import PDBParser, PDBIO

    df = atom_df.copy()
    df["res_name"] = df["res_name"].astype(str)
    df["atom_name"] = df["atom_name"].astype(str)
    df["chain"] = df["chain"].astype(str)
    df["element"] = df["element"].astype(str)
    for col in ["alt_loc", "insertion_code"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
        else:
            df[col] = " "
    df = df.sort_index()

    with open(output_path, "w") as pdb_file:
        for _, row in df.iterrows():
            pdb_file.write(
                f"{row['ATOM']:<6}"
                f"{int(row['atom_num']):>5}"
                f" {row['atom_name']:<4}"
                f"{row['alt_loc']:<1}"
                f"{row['res_name']:<3}"
                f" {row['chain']:<1}"
                f"{int(row['res_num']):>4}"
                f"{row['insertion_code']:<1}"
                f"   "
                f"{float(row['x']):>8.3f}"
                f"{float(row['y']):>8.3f}"
                f"{float(row['z']):>8.3f}"
                f"{float(row['occupancy']):>6.2f}"
                f"{float(row['b_factor']):>6.2f}"
                f"          "
                f"{row['element']:<2}"
                f"\n"
            )

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("wt", output_path)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path)


def build_pyrosetta_mutant(
    pdb,
    res_num: int,
    chain: str,
    target_aa3: str,
    output_path: str,
    pack_radius: float = 8.0,
) -> None:
    """Build a single point-mutant PDB with PyRosetta and write it to ``output_path``.

    The full-atom wild-type pose is loaded from ``pdb.atom``, residue
    ``(chain, res_num)`` is mutated to ``target_aa3`` and side chains within
    ``pack_radius`` Angstroms are repacked with the default Rosetta score
    function; the repacked full-atom mutant is then written to ``output_path``.

    Args:
        pdb: FrustraPy ``Pdb`` carrying the wild-type ``.atom`` DataFrame + job_dir.
        res_num: PDB residue number to mutate.
        chain: chain identifier of the target residue.
        target_aa3: three-letter target amino-acid code (e.g. ``"ALA"``).
        output_path: where to write the mutant PDB.
        pack_radius: side-chain repack radius in Angstroms around the mutation.
    """
    pyrosetta = _ensure_pyrosetta()
    from pyrosetta.toolbox import mutate_residue

    if target_aa3 not in _THREE_TO_ONE:
        raise ValueError(f"Unknown target amino acid {target_aa3!r}")
    target_aa1 = _THREE_TO_ONE[target_aa3]

    # Write the wild-type structure to a temp PDB next to the output, load it,
    # mutate, repack, and dump the mutant. The temp WT is removed afterwards.
    wt_tmp = output_path + ".wt.tmp.pdb"
    write_atom_dataframe_to_pdb(pdb.atom, wt_tmp)
    try:
        pose = pyrosetta.pose_from_pdb(wt_tmp)
        pose_resi = pose.pdb_info().pdb2pose(chain, int(res_num))
        if pose_resi == 0:
            raise ValueError(
                f"Residue {res_num} in chain '{chain}' not found in PyRosetta pose"
            )
        # pack_radius=0 would only rebuild the mutated rotamer; a non-zero radius
        # repacks the local environment so neighbouring side chains relax around
        # the new identity (toolbox.mutate_residue uses the default scorefxn).
        mutate_residue(pose, pose_resi, target_aa1, pack_radius=pack_radius)
        pose.dump_pdb(output_path)
    finally:
        if os.path.exists(wt_tmp):
            os.remove(wt_tmp)
