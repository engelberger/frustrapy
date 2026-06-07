"""FrustraMPNN: deep-learning prediction of single-residue local energetic frustration.

A learned predictor integrated alongside FrustraPy's LAMMPS/AWSEM frustration engine. It runs
a ProteinMPNN-derived network (exported to ONNX) on a protein backbone and returns per-residue
frustration plus a saturation-mutagenesis matrix.

Default inference uses ONNX Runtime on CPU. ``onnxruntime`` is an optional dependency declared
in the ``mpnn`` extra and imported lazily inside the inference code, so ``import frustrapy`` and
``import frustrapy.mpnn`` work on a bare install; calling :func:`analyze` without the extra
raises a clear install hint.

Example::

    import frustrapy.mpnn as mpnn
    result = mpnn.analyze("protein.pdb", chains=["A"])
    result.per_residue       # native per-residue frustration + class
    result.mutation_matrix   # L x 21 saturation-mutagenesis scan

See ``docs/MPNN_INTEGRATION.md`` for the data contract and integration design.

Acknowledgment / citation
--------------------------
FrustraMPNN was developed by Beining, Engelberger, Parra, Schoeder,
Ramirez-Sarmiento and Meiler — official repository
https://github.com/RosettaCommons/frustraMPNN . If you use this module, cite the preprint
(bioRxiv 2026, doi:10.64898/2026.01.22.701012) and, for the bundled weights, the Zenodo
record (doi:10.5281/zenodo.17978321, CC BY 4.0). See the project README / CITATION.cff.
"""

from __future__ import annotations

from . import constants
from .analyze import analyze
from .contract import MPNNResult

__all__ = ["analyze", "MPNNResult", "constants"]
