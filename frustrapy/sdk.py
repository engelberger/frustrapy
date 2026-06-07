"""Public Python SDK for FrustraPy.

This module is a thin, documented facade over the functions that already make up
the FrustraPy public API. It does not reimplement or wrap any logic; it groups the
stable entry points behind one import so application and notebook code has a single,
typed surface to depend on::

    import frustrapy.sdk as fp

    pdb, plots, density, single_res = fp.calculate_frustration(
        pdb_file="1crn.pdb", mode="configurational"
    )
    table = fp.get_frustration(pdb)

Every name re-exported here is the same callable exposed at the top level of the
``frustrapy`` package, so ``frustrapy.calculate_frustration`` and
``frustrapy.sdk.calculate_frustration`` are the same object. The facade exists to
give a single documented place that describes the public surface and its return
contracts; the top-level names remain for backwards compatibility.

Return contracts
----------------
``calculate_frustration`` returns a 4-tuple
``(Pdb, plots: dict, density: FrustrationDensityResults | None, single_residue: dict | None)``.
The density slot is populated only in ``singleresidue`` mode with a non-empty
``residues`` selection; otherwise it is ``None``. The fourth slot is the parsed
single-residue data, also only populated in ``singleresidue`` mode.

``dir_frustration`` returns ``(plots: dict, density: FrustrationDensityResults | None)``.

``dynamic_frustration`` returns a :class:`~frustrapy.core.dynamic.Dynamic` object.

``get_frustration`` returns a :class:`pandas.DataFrame` parsed from the on-disk
frustration table (14 columns for configurational/mutational, 8 for singleresidue).

``analyze_family`` returns a ``dict`` with ``job_id``, ``output_dir``, a ``files``
sub-dict of output paths, and a ``contacts`` sub-dict of per-contact information
content plus MIN/NEU/MAX summary counts.

The mutation-scan functions return the input :class:`~frustrapy.core.pdb.Pdb` with
``pdb.Mutations[method]`` and ``pdb.MutationAnalysis`` populated.

Modes and thresholds
--------------------
Three frustration indices are supported via ``mode``: ``"configurational"``,
``"mutational"``, ``"singleresidue"``. Contact tables (configurational/mutational)
classify ``FrstIndex <= -1`` as highly frustrated, ``>= 0.78`` as minimally
frustrated, and the interval between as neutral.
"""

from __future__ import annotations

# Version of the installed package, mirrored from the top-level namespace.
from . import __version__

# --- Single-structure frustration -----------------------------------------
# calculate_frustration: run one structure through the AWSEM/LAMMPS engine and
# parse its tables. get_frustration: read back the parsed per-residue/per-contact
# table for a finished Pdb, optionally filtered by residue or chain.
from .analysis.frustration import (
    calculate_frustration,
    get_frustration,
)

# --- Batch and trajectory --------------------------------------------------
# dir_frustration: every PDB in a directory (n_procs runs structures concurrently).
# dynamic_frustration: an ordered set of trajectory frames -> a Dynamic object.
from .analysis.frustration import (
    dir_frustration,
    dynamic_frustration,
)

# --- Saturation mutagenesis ------------------------------------------------
# mutate_res_scan_parallel: the whole (residue x amino-acid) grid in one pool.
# mutate_res_parallel: single-residue convenience wrapper over the scan.
# pyrosetta_available: report whether the optional PyRosetta backend is installed.
from .analysis.mutations import (
    mutate_res_parallel,
    mutate_res_scan_parallel,
)
from .analysis.mutation_backends import pyrosetta_available

# --- Evolutionary frustration (FrustraEvo) ---------------------------------
from .evolution import analyze_family

# --- Return-contract types -------------------------------------------------
# Re-exported so callers can annotate against the SDK's return types without
# reaching into private submodules.
from .core.pdb import Pdb
from .core.dynamic import Dynamic
from .core.data_classes import FrustrationDensityResults

__all__ = [
    "__version__",
    # single-structure
    "calculate_frustration",
    "get_frustration",
    # batch / trajectory
    "dir_frustration",
    "dynamic_frustration",
    # mutation scan
    "mutate_res_parallel",
    "mutate_res_scan_parallel",
    "pyrosetta_available",
    # evolution
    "analyze_family",
    # types
    "Pdb",
    "Dynamic",
    "FrustrationDensityResults",
]
