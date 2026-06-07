# FrustraMPNN

`frustrapy.mpnn` is a learned predictor of single-residue local energetic
frustration, complementary to the AWSEM/LAMMPS engine described in
[Backends](Backends). It runs a ProteinMPNN-derived message-passing network
(exported to ONNX) on a protein backbone and returns per-residue frustration plus a
full saturation-mutagenesis matrix. It does not run the LAMMPS engine and is
independent of the configurational/mutational/singleresidue modes.

## Installation

```bash
uv pip install -e ".[mpnn]"
```

The extra pulls ONNX Runtime (CPU). The ~25 MB `frustrampnn_v6` ONNX weight is
bundled in the wheel, so no download is required. `onnxruntime` is imported lazily
inside `analyze()`, so `import frustrapy` and `import frustrapy.mpnn` work without
the extra; only calling `analyze()` requires it.

## Usage

```python
import frustrapy

result = frustrapy.mpnn.analyze("1ubq.pdb", chains=["A"])

# Native per-residue frustration, one row per residue:
#   chain  position  resnum  aa  frustration  frustration_class
result.per_residue.head()
result.per_residue["frustration_class"].value_counts()

# Saturation-mutagenesis matrix: one row per position, one column per amino acid
# (predicted frustration if that residue were mutated to each AA):
result.mutation_matrix.head()

# Score only a few positions (the scan still uses the whole structure):
sub = frustrapy.mpnn.analyze("1ubq.pdb", chains=["A"], positions=[0, 5, 10])
```

`analyze(pdb_path, chains=None, positions=None, model_path=None)` returns an
`MPNNResult` with two DataFrames (`per_residue`, `mutation_matrix`), the resolved
`model_path`, and the `pdb_id`. With `chains=None` every chain is analyzed.

## Output and classes

Frustration follows the same sign convention as the engine's `FrstIndex` (more
positive = minimally frustrated, more negative = highly frustrated). The
single-residue class cutoffs are the same ones the engine's single-residue plot
uses:

| Class | Condition |
|---|---|
| highly frustrated | `frustration <= -1.0` |
| neutral | in between |
| minimally frustrated | `frustration >= 0.58` |

The `per_residue` columns (`chain`, `position`, `resnum`, `aa`, `frustration`,
`frustration_class`) share the residue/chain vocabulary of the engine's
single-residue table so downstream code can treat both uniformly.

## Weight resolution

`analyze()` resolves the ONNX model in this order:

1. an explicit `model_path=` argument,
2. the `FRUSTRAPY_MPNN_MODEL` environment variable,
3. the bundled copy under `frustrapy/mpnn/weights/`,
4. otherwise a clear error.

## Validation

The bundled `frustrampnn_v6` ONNX export reproduces the FrustraMPNN web-demo output
bit-for-bit on 1UBQ (ubiquitin, 76 residues): the full 1520-cell saturation matrix
matches with max|Δ| = 0 on the CPU provider, and the native single-residue class
counts match the web-demo statistics (10 highly, 42 neutral, 24 minimally). The
regression test is `tests/test_mpnn.py::test_validate_against_reference_1ubq`.

Accuracy against frustratometeR is a property of the trained weights, not of this
integration: the published checkpoint reports frustration Spearman 0.80-0.87 vs
frustratometeR single-residue frustration. The full data contract, validation table,
and intentional-divergence notes are in `docs/MPNN_INTEGRATION.md`.
