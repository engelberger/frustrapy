"""Constants for the FrustraMPNN predictor.

Pure data, no behavior. Mirrors ``frustrampnn.constants`` in the source repository so the
ONNX inputs/outputs are indexed identically. See ``docs/MPNN_INTEGRATION.md`` for the
data contract these constants define.
"""

from __future__ import annotations

# 21-character amino acid alphabet (X = unknown). The ONNX model's ``S`` input indexes into
# this string, and its 21-wide ``frustration`` output is one value per character here.
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
VOCAB_DIM = len(ALPHABET)

# The 20 standard amino acids (the ONNX scan excludes X).
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Three-letter to one-letter, with selenomethionine folded to methionine (matches the source).
AA_3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}

# Backbone atom order expected by the model's ``X`` input: (N, CA, C, O).
BACKBONE_ATOMS = ("N", "CA", "C", "O")

# The exported model builds a k=64 nearest-neighbor graph, so inputs shorter than this fail the
# TopK op. Pad to ``max(L, MIN_RESIDUES_FOR_KNN)`` with a zero mask and slice the output back to
# L. Matches the web demo (``web-demo/src/lib/inference.ts``, ``MIN_RESIDUES_FOR_KNN``).
MIN_RESIDUES_FOR_KNN = 64

# Single-residue frustration classification cutoffs. Same sign convention as FrustraPy's
# ``FrstIndex`` (more positive = minimally frustrated). These are the single-residue cutoffs
# (0.58), distinct from the 0.78 contact cutoff used by the LAMMPS engine's
# configurational/mutational tables. Matches ``frustrampnn.constants.FRUSTRATION_THRESHOLDS``.
HIGHLY_FRUSTRATED_MAX = -1.0  # value <= -1.0  -> highly frustrated
MINIMALLY_FRUSTRATED_MIN = 0.58  # value >= 0.58 -> minimally frustrated
# neutral: -1.0 < value < 0.58

# Default ONNX model file name (bundled under frustrapy/mpnn/weights/ in M1). The dynamic-shape
# export the web demo ships; equivalent to frustrampnn_fireprot_dynamic on the CPU EP.
DEFAULT_MODEL_FILENAME = "frustrampnn_v6_dynamic_fixed.onnx"
