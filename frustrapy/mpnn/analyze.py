"""FrustraMPNN inference entry point.

M0 skeleton: the public signature and the documented data contract are fixed here; the
inference body (PDB -> backbone tensors -> ONNX Runtime forward -> per-residue scores) is
wired in M1. ``onnxruntime`` is imported lazily inside the implementation, never at import
time, and is provided by the ``mpnn`` extra (``pip install frustrapy[mpnn]``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .contract import MPNNResult

__all__ = ["analyze"]


def analyze(
    pdb_path: str | Path,
    chains: Sequence[str] | None = None,
    positions: Sequence[int] | None = None,
    model_path: str | Path | None = None,
) -> MPNNResult:
    """Predict single-residue local energetic frustration with FrustraMPNN.

    Args:
        pdb_path: Path to a PDB file.
        chains: Chain IDs to analyze; ``None`` analyzes every chain.
        positions: 0-based residue positions to score; ``None`` scores all.
        model_path: Explicit ONNX model path; ``None`` resolves the bundled default
            (see ``docs/MPNN_INTEGRATION.md`` for the resolution order).

    Returns:
        MPNNResult: native per-residue frustration plus the saturation-mutagenesis matrix.

    Raises:
        NotImplementedError: inference is wired in M1; this is the M0 skeleton.
    """
    raise NotImplementedError(
        "frustrapy.mpnn.analyze is wired in M1; M0 ships the design and module skeleton only. "
        "See docs/MPNN_INTEGRATION.md."
    )
