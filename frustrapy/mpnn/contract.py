"""Output data contract for the FrustraMPNN predictor.

Defines ``MPNNResult``, the return type of ``frustrapy.mpnn.analyze``. Pure data containers,
no heavy dependencies imported at module load (pandas is referenced only in type hints).
See ``docs/MPNN_INTEGRATION.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing pandas at module load
    import pandas as pd


@dataclass
class MPNNResult:
    """FrustraMPNN prediction for one structure.

    Attributes:
        per_residue: One row per scored residue with the native (wild-type) prediction.
            Columns share the single-residue vocabulary of the LAMMPS engine where they
            overlap: ``chain``, ``position`` (0-based within chain), ``resnum`` (PDB number),
            ``aa`` (wild-type one-letter), ``frustration`` (native predicted value), and
            ``frustration_class`` (highly / neutral / minimally).
        mutation_matrix: Saturation-mutagenesis scores. One row per scored residue and one
            column per amino acid in ``frustrapy.mpnn.constants.AMINO_ACIDS`` (plus the
            structural ``chain``/``position`` keys), giving the predicted frustration if the
            residue were that amino acid. The native column equals the ``per_residue`` value.
        model_path: Path to the ONNX model used.
        pdb_id: Structure identifier (PDB file stem).
    """

    per_residue: "pd.DataFrame"
    mutation_matrix: "pd.DataFrame"
    model_path: str
    pdb_id: str
