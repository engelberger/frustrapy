# FrustraMPNN integration design (M0)

Design for the `frustrapy.mpnn` module: a deep-learning predictor of single-residue local
energetic frustration, integrated as a first-class FrustraPy capability alongside the
LAMMPS/AWSEM frustration engine. This document is the M0 deliverable (audit + design); the
inference path is wired in M1.

## Source assets

FrustraMPNN is the maintainer's own repository (`/workspace/frustraMPNN-2`), reused freely.
Relevant pieces:

- `src/frustrampnn/` — the PyTorch model (a ProteinMPNN-derived message-passing network with a
  transfer-learning head), Lightning training code, a `FrustraMPNN` torch inference class, and a
  visualization/validation suite.
- `weights/onnx/` — ONNX exports of the trained model. `weights/checkpoints/` — the PyTorch
  Lightning `.ckpt` files (54-61 MB each).
- `web-demo/` — a Next.js app that runs the ONNX model in the browser via onnxruntime-web; its
  `src/lib/inference.ts` is the reference for the ONNX call path (including small-protein padding).
- `scripts/benchmark_onnx_models.py` — a CPU onnxruntime reference path (PDB -> tensors -> session
  -> per-position scores). This is the closest analogue to what `frustrapy.mpnn` does.

The model is trained on FireProt / MegaScale ΔΔG data; the published checkpoints report frustration
Spearman 0.80-0.87 against frustratometeR single-residue frustration (`weights/README.md`).

## Runtime choice: ONNX Runtime (CPU)

We use ONNX Runtime, not PyTorch, for the default inference path:

- The model ships a dynamic-shape ONNX export that supports arbitrary sequence length and runs on
  the ONNX Runtime CPU execution provider with no torch dependency.
- ONNX Runtime is a single small wheel; the torch path pulls torch + lightning + omegaconf and the
  ProteinMPNN vanilla weights, which is a much heavier install for the same numbers.
- The same ONNX file is what the web demo runs, so the Python and browser paths share one artifact.

A torch backend can be added later behind the same API for users who already have the checkpoints;
it is out of scope for M0-M3.

### Bundled model

Default model: `frustrampnn_v6_dynamic_fixed.onnx` (~25 MB, opset 18, dynamic shapes). This is the
file the web demo ships and the one verified below; the "_fixed" export resolves a WebGPU TopK
issue and is equivalent on the CPU EP to `frustrampnn_fireprot_dynamic.onnx`.

Weight resolution order (implemented in M1):

1. explicit `model_path=` argument,
2. `FRUSTRAPY_MPNN_MODEL` environment variable,
3. a bundled copy under `frustrapy/mpnn/weights/` (added via hatch force-include in M1),
4. a clear error pointing at the maintainer's `weights/onnx/` directory.

Bundling the ~25 MB ONNX in the wheel is allowed (maintainer's own model; FrustraPy is
GPL-3.0-or-later). The actual copy + force-include lands in M1 so the M0 commit stays code-only.

## Data contract

Verified by running `frustrampnn_v6_dynamic_fixed.onnx` on `tests/data/1crn.pdb` with onnxruntime
1.26 (CPU EP), 2026-06-07. Inputs and outputs (from the model + the `frustrampnn_fireprot_dynamic`
manifest):

Inputs (batch = 1):

| name | dtype | shape | meaning |
|---|---|---|---|
| `X` | float32 | `[1, L, 4, 3]` | backbone N, CA, C, O coordinates per residue |
| `S` | int64 | `[1, L]` | sequence, indexed into `ACDEFGHIKLMNPQRSTVWYX` (X=20) |
| `mask` | float32 | `[1, L]` | 1 for real residue, 0 for padding |
| `residue_idx` | int64 | `[1, L]` | 0-based residue index (0 on padding) |
| `chain_encoding_all` | int64 | `[1, L]` | per-residue chain id (1 for a single chain) |
| `position` | int64 | `[1]` | the residue position being scored |

Output:

| name | dtype | shape | meaning |
|---|---|---|---|
| `frustration` | float32 | `[1, 21]` | predicted frustration for each of the 21 alphabet AAs at `position` |

A full single-residue scan calls the session once per position (`position = 0..L-1`) and stacks
the rows into an `(L, 21)` matrix. The **native** per-residue frustration is the column at the
wild-type AA index — i.e. `matrix[pos, ALPHABET.index(seq[pos])]`. The off-diagonal columns are the
saturation-mutagenesis predictions (frustration if that position were mutated to each AA).

### Small-protein padding (k-nearest-neighbors)

The exported model builds a k=64 nearest-neighbor graph (`features.TopK`), so it fails on proteins
shorter than 64 residues. Following the web demo (`src/lib/inference.ts`,
`MIN_RESIDUES_FOR_KNN = 64`): pad every input to `max(L, 64)` along the sequence axis with zeros,
set `mask = 0` on the padded tail, and slice the output back to the real `L`. The mask keeps padding
from affecting real residues. Verified: 1crn (46 residues) padded to 64 runs and produces 46 rows.

### Classification thresholds

FrustraMPNN frustration follows the same sign convention as FrustraPy's `FrstIndex` (more positive =
minimally frustrated, more negative = highly frustrated). The single-residue cutoffs match
`frustrampnn.constants.FRUSTRATION_THRESHOLDS` and FrustraPy's single-residue plot:

- highly frustrated: `value <= -1.0`
- minimally frustrated: `value >= 0.58`
- neutral: in between.

(0.58 is the single-residue cutoff, distinct from the 0.78 contact cutoff used by the LAMMPS engine's
configurational/mutational tables — do not collapse them.)

## Module layout

```
frustrapy/mpnn/
  __init__.py     public API (analyze, MPNNResult, constants); no heavy import at module load
  constants.py    alphabet, AA maps, thresholds, MIN_RESIDUES_FOR_KNN  (pure data)
  contract.py     MPNNResult dataclass (the output data contract)
  analyze.py      analyze(pdb, ...) orchestration  (wired in M1)
  weights/        bundled ONNX  (added in M1)
```

`onnxruntime` is imported lazily inside the inference code, never at `import frustrapy` or
`import frustrapy.mpnn` time, and is declared in an `mpnn` extra. A bare install keeps the core
library and `import frustrapy.mpnn` working; calling `analyze()` without the extra raises a clear
install hint.

### Public API (target, wired in M1)

```python
import frustrapy.mpnn as mpnn
result = mpnn.analyze("protein.pdb", chains=["A"])   # -> MPNNResult
result.per_residue      # DataFrame: chain, position, resnum, aa, frustration, class
result.mutation_matrix  # DataFrame or ndarray (L x 21) saturation scan
```

`MPNNResult` shares the residue/chain/IO vocabulary with the LAMMPS engine's single-residue table
(`Res ChainRes AA FrstIndex`) so downstream code and plots can treat both uniformly. If the G1
`FrustrationBackend` interface exists by M3, add an adapter exposing the native per-residue scores
through it; otherwise leave a note (M3).

## Validation plan (M2)

Compare `frustrapy.mpnn` native per-residue scores against the original FrustraMPNN onnxruntime
reference (`scripts/benchmark_onnx_models.py` path) on a small panel (1crn, 1UBQ). Because this is a
learned model, define an explicit numeric tolerance (target: Pearson/Spearman = 1.0 and
`max|Δ| < 1e-4` against the same ONNX file, since both use the CPU EP). Add a regression test and a
usage example. Do not invent accuracy-vs-frustratometeR numbers; the published checkpoint Spearman
(0.80-0.87) comes from the maintainer's `weights/README.md`.

## Verified probe (2026-06-07)

`frustrampnn_v6_dynamic_fixed.onnx`, CPU EP, onnxruntime 1.26, `tests/data/1crn.pdb` (46 residues,
chain A), padded to 64:

- output shape `(46, 21)`; two runs identical (`max|Δ| = 0`) -> deterministic on this CPU build;
- native per-residue frustration range `[-1.638, 2.230]`;
- single-residue classification: 7 highly frustrated, 24 neutral, 15 minimally frustrated.

These are the M0 audit numbers (a single structure on one CPU build), not an accuracy claim.
