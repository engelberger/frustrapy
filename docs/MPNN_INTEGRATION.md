# FrustraMPNN integration design (M0)

Design for the `frustrapy.mpnn` module: a deep-learning predictor of single-residue local
energetic frustration, integrated as a first-class FrustraPy capability alongside the
LAMMPS/AWSEM frustration engine. This document covers the M0 audit + design and the M1
inference path (now wired).

## Status

- M0 (audit + design + module skeleton): done.
- M1 (inference path): done. `frustrapy.mpnn.analyze(pdb)` parses the backbone, runs the bundled
  ONNX export on the ONNX Runtime CPU provider, and returns an `MPNNResult`. The
  `frustrampnn_v6_dynamic_fixed.onnx` weight is bundled under `frustrapy/mpnn/weights/` and
  force-included in the wheel (`pyproject.toml`). The parser produces inputs bit-identical to the
  reference (`frustraMPNN-2/scripts/benchmark_onnx_models.py`); on 1CRN the output matches the
  recorded probe below (range, class counts, determinism). Tests: `tests/test_mpnn.py`.
- M2 (validation vs the reference): done. The full 1UBQ saturation matrix matches the recorded
  FrustraMPNN web-demo output bit-for-bit; see "Validation (M2)" below. Test:
  `tests/test_mpnn.py::test_validate_against_reference_1ubq`.
- M3 (public API + README/wiki): done. `frustrapy.mpnn` is reachable from the top-level package
  and listed in `frustrapy.__all__`; README has a "FrustraMPNN (deep-learning predictor)" section
  and `docs/wiki/FrustraMPNN.md` is in the wiki. See "Public API (M3)" below.

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
  analyze.py      analyze(pdb, ...) orchestration
  weights/        bundled ONNX  (frustrampnn_v6_dynamic_fixed.onnx)
```

`onnxruntime` is imported lazily inside the inference code, never at `import frustrapy` or
`import frustrapy.mpnn` time, and is declared in an `mpnn` extra. A bare install keeps the core
library and `import frustrapy.mpnn` working; calling `analyze()` without the extra raises a clear
install hint.

### Public API

```python
import frustrapy.mpnn as mpnn

result = mpnn.analyze("1UBQ.pdb", chains=["A"])      # -> MPNNResult

# Native single-residue frustration, one row per residue:
#   chain  position  resnum  aa  frustration  frustration_class
result.per_residue.head()

# How many residues fall in each class:
result.per_residue["frustration_class"].value_counts()
# minimally    24
# neutral      42  (1UBQ; matches the FrustraMPNN web demo)
# highly       10

# Saturation-mutagenesis matrix: one row per position, one column per amino acid
# (predicted frustration if that position were mutated to each AA):
result.mutation_matrix.head()

# Score the model only at a few positions (the scan still uses the whole structure):
sub = mpnn.analyze("1UBQ.pdb", chains=["A"], positions=[0, 5, 10])
```

`analyze` needs the `mpnn` extra (`pip install 'frustrapy[mpnn]'`, which pulls onnxruntime). The
ONNX weight is bundled, so no download is required.

`MPNNResult` shares the residue/chain/IO vocabulary with the LAMMPS engine's single-residue table
(`Res ChainRes AA FrstIndex`) so downstream code and plots can treat both uniformly.

## Public API (M3)

The module is exposed at the top level so callers reach it as `frustrapy.mpnn` without a
separate import:

```python
import frustrapy

result = frustrapy.mpnn.analyze("1ubq.pdb", chains=["A"])
```

`mpnn` is a lazy attribute on the `frustrapy` package (resolved in `frustrapy/__init__.py`
`__getattr__` via `importlib.import_module`, the same pattern as `detect_dynamic_clusters`) and
is listed in `frustrapy.__all__`. Accessing `frustrapy.mpnn` imports the submodule but not
`onnxruntime`; the optional runtime is pulled only when `analyze()` runs. The top-level lazy
access is regression-tested by `tests/test_mpnn.py::test_top_level_lazy_access`.

The G1 `FrustrationBackend` interface is not present on this branch (`dev_frustrampnn`), so no
backend adapter is added here. FrustraMPNN is a learned single-residue predictor, not an AWSEM
energy/decoy backend, so it would not be a drop-in `lammps` replacement in any case; if a
`FrustrationBackend` lands, an adapter could expose the native per-residue scores through it. For
now `frustrapy.mpnn.analyze` is the public entry point.

## Validation (M2)

`frustrapy.mpnn` was validated against the FrustraMPNN reference on 1UBQ (ubiquitin, 76 residues,
chain A), the maintainer's regression-test protein. The full saturation matrix (76 residues x 20
amino acids = 1520 predictions) was compared three ways. Measured 2026-06-07, onnxruntime 1.26,
CPU EP. Nothing here is invented; every number was produced by running the code.

| reference | model | max\|Δ\| | Pearson | Spearman |
|---|---|---|---|---|
| `scripts/benchmark_onnx_models.py` recipe, same bundled ONNX | frustrampnn_v6 | 0.0 | 1.0 | 1.0 |
| `1UBQ_results.json` (web demo, onnxruntime-web) | frustrampnn_v6 | 0.0 | 1.0 | 1.0 |
| `test_data/1UBQ_reference_output.csv` | fireprot ckpt (different) | 3.14 | 0.73 | 0.71 |

Reading the table:

- Against the same ONNX file run through the documented reference recipe, and against the recorded
  web-demo output (the same model run in the browser via onnxruntime-web), `frustrapy.mpnn` is
  bit-for-bit identical across the entire 1520-cell matrix. The featurization, k-NN padding, and
  per-position scan reproduce the reference exactly. Native single-residue classification also
  matches the web-demo stats: 10 highly frustrated, 42 neutral, 24 minimally frustrated.
- Against `test_data/1UBQ_reference_output.csv` the values diverge (Pearson 0.73). This is an
  **intentional, expected divergence**: that CSV was generated from a different, earlier PyTorch
  checkpoint (`local_train_raw_fireprot_nosubtract_seed1_epoch=19_..._spearman=0.8.ckpt`, see
  `test_data/README.md`), not from `frustrampnn_v6`. FrustraPy bundles the v6 ONNX export that the
  web demo ships, so it matches the web demo, not the older fireprot checkpoint. The two are
  different model versions; this is a model-choice difference, not a wiring bug.

The regression fixture is `tests/data/1UBQ_mpnn_reference.csv` (the 1520 web-demo predictions:
0-based position, wildtype, mutation, frustration) with `tests/data/1UBQ.pdb`. The test
(`test_validate_against_reference_1ubq`) asserts `max|Δ| < 1e-4` and class-count agreement; the
tolerance is set well above the measured 0 because FrustraMPNN is a learned model and a different
CPU build could differ at the float32 ULP.

Accuracy against frustratometeR is not measured here. The published checkpoint reports frustration
Spearman 0.80-0.87 vs frustratometeR single-residue frustration (maintainer's `weights/README.md`);
that is a property of the trained weights, not of this integration.

## Verified probe (2026-06-07)

`frustrampnn_v6_dynamic_fixed.onnx`, CPU EP, onnxruntime 1.26, `tests/data/1crn.pdb` (46 residues,
chain A), padded to 64:

- output shape `(46, 21)`; two runs identical (`max|Δ| = 0`) -> deterministic on this CPU build;
- native per-residue frustration range `[-1.638, 2.230]`;
- single-residue classification: 7 highly frustrated, 24 neutral, 15 minimally frustrated.

These are the M0 audit numbers (a single structure on one CPU build), not an accuracy claim.
