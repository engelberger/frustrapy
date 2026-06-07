# FrustraPy

FrustraPy is an unofficial, parallelized Python reimplementation of the
[frustratometeR](https://github.com/proteinphysiologylab/frustratometeR) R package
for computing local energetic frustration in protein structures. It computes the
three standard frustration indices (configurational, mutational, single-residue),
a saturation-mutagenesis workflow, evolutionary frustration over a protein family
(FrustraEvo), and interactive Plotly/py3Dmol visualizations.

These pages document installation, usage, the public API, the parity methodology,
the backend model, and the roadmap. They mirror the project
[README](https://github.com/engelberger/frustrapy/blob/main/README.md); the README
is authoritative where the two differ.

## What "frustration" means

Energetic frustration is the local violation of the principle of minimal
frustration. A native contact is *minimally frustrated* if its native residues and
geometry are near-optimal, and *highly frustrated* if many alternative residues or
geometries would stabilize it more. Highly frustrated regions concentrate at
functional sites such as binding interfaces, allosteric paths, and catalytic
centers.

FrustraPy quantifies this per contact (configurational, mutational) or per site
(single-residue) as a Z-score `FrstIndex` against a decoy ensemble.

## How it computes numbers

FrustraPy does not reimplement the AWSEM energy model in Python. Like
frustratometeR, it shells out to the same precompiled AWSEM/LAMMPS binaries
(`lmp_serial_*`) with the same parameter files, then parses and post-processes the
output. The only frustration math implemented in Python is the 5 Angstrom spatial
density calculation. See [Backends](Backends) and
[Parity Methodology](Parity-Methodology) for the consequences of this design.

## Frustration classes

For configurational and mutational contacts:

| Class | Condition |
|---|---|
| highly frustrated | `FrstIndex <= -1` |
| neutral | `-1 < FrstIndex < 0.78` |
| minimally frustrated | `FrstIndex >= 0.78` |

The `0.78` cutoff is derived analytically (arXiv:1812.05965), not from a p-value.
Single-residue *plots* use a distinct minimally cutoff of `0.58`, matching
frustratometeR's visualization; the two cutoffs are intentional and are not
interchangeable.

## Pages

- [Installation](Installation) — environment, extras, and the runtime requirements
  (Perl, the bundled LAMMPS binaries, the venv-on-PATH requirement).
- [Usage](Usage) — the public Python API, the SDK facade, the command-line tool,
  and the on-disk output contract.
- [Parity Methodology](Parity-Methodology) — how byte-identity against
  frustratometeR and FrustraEvo was validated and how to reproduce it.
- [Backends](Backends) — the `lammps` reference engine and the `threading` /
  `pyrosetta` mutation methods.
- [Roadmap](Roadmap) — planned, not-yet-implemented work (pluggable backends, a
  native GPU backend, FrustraMPNN), each parity-gated before any speed claim.

## Status

Numerical parity against frustratometeR is verified (1CRN, three modes,
`seq_dist` 3 and 12; `FrstIndex` and every energy column diff zero at 3-decimal
print). FrustraEvo information-content outputs are byte-identical to the original
on Alpha-globins and Sars-PlPro. The package is not yet published to PyPI; install
from source.

## License

GPL-3.0-or-later. The bundled AWSEM/LAMMPS binaries are GPL, so the distribution is
GPL regardless. This is an unofficial reimplementation; verify results against the
original frustratometeR where possible.
