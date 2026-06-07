"""FrustraMPNN inference entry point.

``analyze`` parses a PDB backbone, runs the FrustraMPNN ONNX export on the ONNX Runtime CPU
provider, and returns per-residue frustration plus a saturation-mutagenesis matrix. The call
path mirrors the maintainer's reference (``frustraMPNN-2/scripts/benchmark_onnx_models.py`` and
``web-demo/src/lib/inference.ts``): one ``session.run`` per residue position yields all 21
alphabet scores at once.

``onnxruntime`` is imported lazily inside :func:`analyze`, never at module load, and is provided
by the ``mpnn`` extra (``pip install frustrapy[mpnn]``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from . import constants
from .contract import MPNNResult

__all__ = ["analyze"]


def _resolve_model_path(model_path: str | Path | None) -> Path:
    """Resolve the ONNX model path.

    Order: explicit argument, ``FRUSTRAPY_MPNN_MODEL`` env var, the bundled copy under
    ``frustrapy/mpnn/weights/``. Raises ``FileNotFoundError`` with a pointer to the maintainer's
    ``weights/onnx/`` directory if none resolve.
    """
    if model_path is not None:
        p = Path(model_path)
        if not p.is_file():
            raise FileNotFoundError(f"FrustraMPNN model not found: {p}")
        return p

    env = os.environ.get("FRUSTRAPY_MPNN_MODEL")
    if env:
        p = Path(env)
        if not p.is_file():
            raise FileNotFoundError(
                f"FRUSTRAPY_MPNN_MODEL points at a missing file: {p}"
            )
        return p

    bundled = Path(__file__).resolve().parent / "weights" / constants.DEFAULT_MODEL_FILENAME
    if bundled.is_file():
        return bundled

    raise FileNotFoundError(
        "No FrustraMPNN ONNX model found. Pass model_path=, set FRUSTRAPY_MPNN_MODEL, or "
        "install a build that bundles frustrapy/mpnn/weights/"
        f"{constants.DEFAULT_MODEL_FILENAME} (see the maintainer's weights/onnx/ directory)."
    )


def _import_onnxruntime():
    """Lazily import onnxruntime with a clear install hint."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "frustrapy.mpnn.analyze requires onnxruntime. Install the extra: "
            "pip install 'frustrapy[mpnn]'."
        ) from exc
    return onnxruntime


def _parse_backbone(pdb_path: Path, chains: Sequence[str] | None):
    """Parse N, CA, C, O backbone coordinates and the one-letter sequence per chain.

    Returns an ordered mapping ``chain_id -> (coords, seq, resnums)`` where ``coords`` is an
    ``(L, 4, 3)`` float32 array in (N, CA, C, O) atom order, ``seq`` the one-letter sequence
    (``X`` for unknown residues), and ``resnums`` the PDB residue numbers. Only the first model
    and the first conformer of each residue are kept. Matches the reference parser in
    ``benchmark_onnx_models.py`` (chain filter, atom order, MSE -> M via constants.AA_3_TO_1).
    """
    import numpy as np

    atom_order = {name: i for i, name in enumerate(constants.BACKBONE_ATOMS)}
    wanted = set(chains) if chains is not None else None

    # chain_id -> {res_key: coords(4,3)}, {res_key: aa}, ordered res_keys
    coords_by_chain: dict[str, dict] = {}
    seq_by_chain: dict[str, dict] = {}
    order_by_chain: dict[str, list] = {}
    chain_order: list[str] = []
    in_first_model = True

    with open(pdb_path) as fh:
        for line in fh:
            record = line[:6].strip()
            if record == "ENDMDL":
                in_first_model = False
                continue
            if not in_first_model:
                continue
            if record not in ("ATOM", "HETATM"):
                continue

            atom_name = line[12:16].strip()
            if atom_name not in atom_order:
                continue
            res_name = line[17:20].strip()
            if record == "HETATM" and res_name not in constants.AA_3_TO_1:
                continue

            chain_id = line[21]
            if wanted is not None and chain_id not in wanted:
                continue

            # residue key: (resSeq, insertion code); altLoc handled by first-wins below.
            res_key = (line[22:26].strip(), line[26])
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            if chain_id not in coords_by_chain:
                coords_by_chain[chain_id] = {}
                seq_by_chain[chain_id] = {}
                order_by_chain[chain_id] = []
                chain_order.append(chain_id)

            cc = coords_by_chain[chain_id]
            if res_key not in cc:
                cc[res_key] = np.zeros((4, 3), dtype=np.float32)
                seq_by_chain[chain_id][res_key] = constants.AA_3_TO_1.get(res_name, "X")
                order_by_chain[chain_id].append(res_key)
            cc[res_key][atom_order[atom_name]] = (x, y, z)

    parsed = {}
    for chain_id in chain_order:
        keys = order_by_chain[chain_id]
        coords = np.stack([coords_by_chain[chain_id][k] for k in keys]).astype(np.float32)
        seq = "".join(seq_by_chain[chain_id][k] for k in keys)
        resnums = [k[0] + (k[1].strip()) for k in keys]
        parsed[chain_id] = (coords, seq, resnums)
    return parsed


def _scan_chain(session, coords, seq):
    """Run the model at every position of one chain; return an ``(L, 21)`` score matrix.

    Pads to ``max(L, MIN_RESIDUES_FOR_KNN)`` with a zero mask (the k=64 graph fails on shorter
    inputs) and slices the output back to ``L``. Padding follows the web demo: ``S`` and
    ``residue_idx`` zero on the tail, ``mask`` zero on the tail, ``chain_encoding_all`` all ones.
    """
    import numpy as np

    L = len(seq)
    padded = max(L, constants.MIN_RESIDUES_FOR_KNN)

    X = np.zeros((1, padded, 4, 3), dtype=np.float32)
    X[0, :L] = coords

    S = np.zeros((1, padded), dtype=np.int64)
    for i, aa in enumerate(seq):
        S[0, i] = constants.ALPHABET.index(aa) if aa in constants.ALPHABET else 20

    mask = np.zeros((1, padded), dtype=np.float32)
    mask[0, :L] = 1.0

    residue_idx = np.zeros((1, padded), dtype=np.int64)
    residue_idx[0, :L] = np.arange(L, dtype=np.int64)

    chain_encoding_all = np.ones((1, padded), dtype=np.int64)

    rows = []
    for pos in range(L):
        inputs = {
            "X": X,
            "S": S,
            "mask": mask,
            "residue_idx": residue_idx,
            "chain_encoding_all": chain_encoding_all,
            "position": np.array([pos], dtype=np.int64),
        }
        out = session.run(None, inputs)[0]  # (1, 21)
        rows.append(out[0])
    return np.asarray(rows, dtype=np.float32)


def _classify(value: float) -> str:
    """Single-residue frustration class (highly / neutral / minimally), inclusive cutoffs."""
    if value <= constants.HIGHLY_FRUSTRATED_MAX:
        return "highly"
    if value >= constants.MINIMALLY_FRUSTRATED_MIN:
        return "minimally"
    return "neutral"


def analyze(
    pdb_path: str | Path,
    chains: Sequence[str] | None = None,
    positions: Sequence[int] | None = None,
    model_path: str | Path | None = None,
) -> MPNNResult:
    """Predict single-residue local energetic frustration with FrustraMPNN.

    Args:
        pdb_path: Path to a PDB file.
        chains: Chain IDs to analyze; ``None`` analyzes every chain present.
        positions: 0-based residue positions (within each chain) to keep in the output;
            ``None`` keeps all. The scan still runs over every position (the graph needs the
            full structure); this only filters the returned rows.
        model_path: Explicit ONNX model path; ``None`` resolves the bundled default
            (see ``docs/MPNN_INTEGRATION.md`` for the resolution order).

    Returns:
        MPNNResult: native per-residue frustration plus the saturation-mutagenesis matrix.
    """
    import numpy as np
    import pandas as pd

    pdb_path = Path(pdb_path)
    if not pdb_path.is_file():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    resolved_model = _resolve_model_path(model_path)
    ort = _import_onnxruntime()

    parsed = _parse_backbone(pdb_path, chains)
    if not parsed:
        raise ValueError(
            f"No backbone residues parsed from {pdb_path}"
            + (f" for chains {list(chains)}" if chains is not None else "")
        )

    session = ort.InferenceSession(
        str(resolved_model), providers=["CPUExecutionProvider"]
    )

    pos_filter = set(positions) if positions is not None else None
    per_residue_rows = []
    mutation_rows = []

    for chain_id, (coords, seq, resnums) in parsed.items():
        matrix = _scan_chain(session, coords, seq)  # (L, 21)
        for pos, aa in enumerate(seq):
            if pos_filter is not None and pos not in pos_filter:
                continue
            native_idx = constants.ALPHABET.index(aa) if aa in constants.ALPHABET else 20
            native = float(matrix[pos, native_idx])
            per_residue_rows.append(
                {
                    "chain": chain_id,
                    "position": pos,
                    "resnum": resnums[pos],
                    "aa": aa,
                    "frustration": native,
                    "frustration_class": _classify(native),
                }
            )
            mut_row = {"chain": chain_id, "position": pos}
            for j, mut_aa in enumerate(constants.AMINO_ACIDS):
                mut_row[mut_aa] = float(matrix[pos, j])
            mutation_rows.append(mut_row)

    per_residue = pd.DataFrame(
        per_residue_rows,
        columns=["chain", "position", "resnum", "aa", "frustration", "frustration_class"],
    )
    mutation_matrix = pd.DataFrame(
        mutation_rows, columns=["chain", "position", *constants.AMINO_ACIDS]
    )

    return MPNNResult(
        per_residue=per_residue,
        mutation_matrix=mutation_matrix,
        model_path=str(resolved_model),
        pdb_id=pdb_path.stem,
    )
