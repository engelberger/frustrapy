# Backends

FrustraPy separates two notions that are sometimes conflated: the **compute
backend** that produces the frustration energies and indices, and the **mutation
method** that builds each point mutant for a saturation scan.

## Compute backend: `lammps` (reference)

There is one compute backend today, and it is the numerical reference: the
precompiled AWSEM/LAMMPS engine. The native energy, the decoy-ensemble statistics,
and the `FrstIndex` are all computed by `lmp_serial_{3,12}_{Linux,MacOS}`, which
FrustraPy drives by subprocess. FrustraPy is an orchestrator and parser around that
binary; the only frustration math in Python is the 5 Angstrom density calculation.

The binaries are byte-for-byte identical to the upstream frustratometeR copies, and
the coefficient/gamma/Perl files are identical after CRLF normalization. This is the
basis for the parity guarantee — see [Parity Methodology](Parity-Methodology).

Practical notes:

- Linux is the supported path (absolute paths plus a stdin redirect). The macOS
  path is x86_64-only and runs under Rosetta 2 on Apple Silicon.
- Only `seq_dist=3` and `seq_dist=12` binaries are provided; 12 is the default.
- Perl must be on `PATH` for the visualization and charge-file steps.
- A calculation spawns a bare `python3` subprocess, so the environment must be
  activated (venv on `PATH`) — see [Installation](Installation).

A pluggable `FrustrationBackend` interface that would let the engine be swapped for
alternative implementations is planned, not implemented; see [Roadmap](Roadmap).
Today the `lammps` engine is the only path, and it is the reference everything else
will be measured against.

## Mutation methods

The saturation-mutagenesis scan builds each point mutant with a selectable method,
passed as `method=` to `mutate_res_parallel` / `mutate_res_scan_parallel`. Both
methods then score frustration with the same `lammps` engine, so the difference is
only in how the mutant structure is constructed.

### `threading` (default)

Keeps the native backbone and CB and relabels the target residue to the new
identity, synthesizing a CB for `GLY -> X` from ideal geometry. It has no extra
dependency and is the in-container parity reference for the mutation workflow.

### `pyrosetta` (optional)

Loads the full-atom pose, mutates the target residue, and repacks side chains
within a radius using the Rosetta score function, then scores frustration on the
repacked mutant. It produces physically realistic rotamers. On 1CRN it tracks the
`threading` method closely: per-variant `FrstIndex` sign agreement 100% and Spearman
~0.996.

PyRosetta is not bundled. It is free for academic/non-commercial use under the
RosettaCommons license and is installed separately:

```bash
uv pip install -e ".[pyrosetta]"
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
```

`frustrapy.sdk.pyrosetta_available` reports whether the backend can be used in the
current environment.

### Choosing a method

Use `threading` for a dependency-free, fast scan and as the parity reference. Use
`pyrosetta` when you want repacked side chains around the mutation and can accept
the extra dependency and runtime; the two agree in sign and rank-order on the
structures tested, so the choice is about physical realism of the mutant model
rather than a different frustration definition.
