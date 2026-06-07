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
"""

from __future__ import annotations

from . import constants
from .analyze import analyze
from .contract import MPNNResult

__all__ = ["analyze", "MPNNResult", "constants"]
