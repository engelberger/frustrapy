# FrustraPy: A Python Implementation of the Protein Frustratometer

[![CI](https://github.com/engelberger/frustrapy/actions/workflows/ci.yml/badge.svg)](https://github.com/engelberger/frustrapy/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/frustrapy.svg)](https://pypi.org/project/frustrapy/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/engelberger/frustrapy/blob/main/frustrapy_colab.ipynb)

FrustraPy is an unofficial, parallelized Python reimplementation of the
[frustratometeR package](https://github.com/proteinphysiologylab/frustratometeR)
(see [Parra et al.](https://academic.oup.com/nar/article/44/W1/W356/2499321),
[Rausch et al.](https://academic.oup.com/bioinformatics/article/37/18/3038/6171179),
[Jenik et al.](https://academic.oup.com/nar/article/40/W1/W348/1075768)) for
computing **local energetic frustration** in protein structures. It adds interactive
Plotly visualizations and parallel mutation analysis on top of the original method.

**Disclaimer:** This is an unofficial reimplementation. Use at your own risk and
verify results against the original frustratometeR when possible.

## How it works

FrustraPy does **not** reimplement the AWSEM energy model in Python. Like
frustratometeR, it shells out to the **same precompiled AWSEM/LAMMPS binaries**
(`lmp_serial_*`) with the **same parameter files**, then post-processes the output.
This is what makes numerical agreement tractable: parity reduces to *glue-code parity*.

> **Numerical parity is verified.** On 1CRN, all three modes × `seq_dist ∈ {3, 12}`,
> the `FrstIndex` and every energy column match frustratometeR bit-for-bit at
> 3-decimal print (Spearman = 1.0, class-agreement = 100 %); an MSE structure (1B6W)
> also agrees. Speed has been benchmarked but headline "faster than R" ratios are
> reported only for parity-passing configurations (see `docs/audit/benchmark/`).

## Features

- **Three frustration modes**: configurational, mutational, and single-residue —
  all run end-to-end on the Linux path.
- **Parallel mutation analysis** via `multiprocessing` (`mutate_res_parallel`,
  `mutate_res_scan_parallel`) and parallel batch/trajectory processing
  (`dir_frustration(..., n_procs=K)`). Pluggable mutation backends: the default
  in-house `threading` (no extra dependency) and an optional `pyrosetta` backend
  that repacks side chains around the mutation.
- **Interactive Plotly figures**: contact map, 5 Å frustration-density plots, density
  proportions, and delta-frustration plots (with `graphics=True`).
- **Evolutionary frustration (FrustraEvo)** via `frustrapy.analyze_family`.

## Requirements

- **Python 3.10–3.12.**
- **Perl** on `PATH` (`/usr/bin/perl`) — used by the visualization/charge-file steps.
- **A LAMMPS `lmp_serial` binary**, shipped precompiled in `frustrapy/core/scripts/`
  as `lmp_serial_{3,12}_{Linux,MacOS}`. Only `seq_dist=3` and `seq_dist=12` are
  provided (12 is the default). The macOS binaries are x86_64 only; on Apple Silicon
  they run under Rosetta 2 (`softwareupdate --install-rosetta`).

## Installation

> **PyPI status:** publishing to PyPI is set up (PEP 621 / Hatchling build, OIDC
> trusted-publishing workflow) but the first release has not been pushed yet. Install
> from source for now. Unlike older revisions, a source install **does** resolve the
> runtime dependencies automatically.

Using [uv](https://github.com/astral-sh/uv) (recommended) or plain `pip`:

```bash
git clone https://github.com/engelberger/frustrapy.git
cd frustrapy

# Create and ACTIVATE an environment. Activation matters: a calculation spawns a
# bare `python3` subprocess, which must find Biopython on PATH.
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install the package — the core runtime deps are pulled in automatically.
uv pip install -e .

# Optional extras:
uv pip install -e ".[viz]"         # 3D viewers (py3Dmol) + static PNG export (kaleido)
uv pip install -e ".[clustering]"  # detect_dynamic_clusters (scipy/sklearn/igraph/leidenalg/statsmodels)
uv pip install -e ".[perf]"        # memory logging (psutil)
uv pip install -e ".[pyrosetta]"   # pyrosetta-installer helper for the PyRosetta mutation backend
uv pip install -e ".[cli]"         # the `frustrapy` command-line tool (Typer + Rich)
uv pip install -e ".[all]"         # everything above
```

Verify:

```bash
python -c "import frustrapy; print(frustrapy.__version__)"
```

### Mutation backends

The saturation-mutagenesis scan builds each point mutant with a selectable
backend, passed as `method=` to `mutate_res_parallel` /
`mutate_res_scan_parallel`:

- **`threading`** (default) — keeps the native backbone and CB and relabels the
  residue to the target identity (synthesising CB for `GLY→X` from ideal
  geometry). No extra dependency; this is the in-container parity reference.
- **`pyrosetta`** — loads the full-atom pose, mutates the target residue, and
  repacks side chains within a radius using the Rosetta score function, then
  scores frustration on the repacked mutant. On 1CRN this tracks the `threading`
  backend closely (per-variant `FrstIndex` sign agreement 100 %, Spearman ≈ 0.996),
  while producing physically realistic rotamers.

PyRosetta is **not** bundled — it is free for academic/non-commercial use under
the [RosettaCommons license](https://www.rosettacommons.org/software/license-and-download)
but must be installed separately:

```bash
uv pip install -e ".[pyrosetta]"   # pulls the pyrosetta-installer helper
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
```

```python
import frustrapy
from frustrapy.analysis.mutations import mutate_res_parallel

pdb, *_ = frustrapy.calculate_frustration(
    pdb_file="1crn.pdb", mode="singleresidue",
    residues={"A": [10]}, graphics=False, visualization=False,
)
mutate_res_parallel(pdb, res_num=10, chain="A", method="pyrosetta")
```

## Quickstart

```python
import frustrapy

# calculate_frustration returns a 4-TUPLE:
#   (pdb, plots, density_results, single_residue_data)
pdb, plots, density, single_res = frustrapy.calculate_frustration(
    pdb_file="protein.pdb",
    mode="configurational",   # or "mutational" / "singleresidue"
    results_dir="results",
    graphics=False,           # set True to also produce Plotly figures
)

# The authoritative results are written to disk under:
#   results/protein.done/FrustrationData/
#     protein.pdb_configurational           # per-contact frustration table (14 cols)
#     protein.pdb_configurational_5adens    # 5 Angstrom density table
#     protein.pdb_configurational_density.pkl
import pandas as pd
table = pd.read_csv(
    "results/protein.done/FrustrationData/protein.pdb_configurational", sep=r"\s+"
)
print(table.head())
```

> With `graphics=False`, `plots` is an empty dict. Set `graphics=True` to build Plotly
> figures (returned in `plots`, keyed by name, and written as HTML/PNG into the job
> directory; PNG export needs the `viz` extra for `kaleido`).

### Single-residue mode

```python
pdb, plots, density, single_res = frustrapy.calculate_frustration(
    pdb_file="protein.pdb",
    mode="singleresidue",
    residues={"A": [10]},
    results_dir="results",
)
# Per-residue table: results/protein.done/FrustrationData/protein.pdb_singleresidue
```

### A directory of structures (parallel batch)

```python
# dir_frustration returns a 2-tuple: (plots_by_pdb, density_results)
plots_by_pdb, density = frustrapy.dir_frustration(
    pdbs_dir="pdbs_directory",
    mode="configurational",
    results_dir="results",
    n_procs=4,   # process structures concurrently under a shared core budget
)
```

## Command-line tool

Installing the `cli` extra (`uv pip install -e ".[cli]"`) exposes a `frustrapy`
console command — thin wrappers over the same functions used above, with a styled
Rich interface. The extra adds `typer` for argument parsing (`rich` is already a
core dependency); `frustrapy/cli/main.py` imports `typer` lazily, so the core
library never requires it and plain `import frustrapy` works without the extra.

```bash
frustrapy --version
frustrapy --help            # lists every subcommand
frustrapy single --help     # per-subcommand help
```

| Subcommand | What it runs | Example |
|---|---|---|
| `single` | one PDB through one mode (`calculate_frustration`) | `frustrapy single protein.pdb -m configurational -o results` |
| `batch` | every `.pdb` in a directory (`dir_frustration`) | `frustrapy batch pdbs/ -m configurational -o results --n-procs 4` |
| `evo` | evolutionary frustration for a family (`analyze_family`) | `frustrapy evo family.fasta -j fam1 -p pdbs/ -r 3lqd-A -o results` |
| `mutate` | saturation-mutagenesis scan of one residue (`mutate_res_parallel`) | `frustrapy mutate protein.pdb --res 10 -c A --method threading -o results` |

Each subcommand writes the same on-disk output described below, prints a summary
panel/table, and returns a non-zero exit code with an actionable error message on
failure. Run any subcommand with `--help` for the full option list.

## Output contract

For a structure `protein.pdb` and `results_dir="results"`, outputs land under
`results/protein.done/`:

- `FrustrationData/protein.pdb_<mode>` — the main table. **14 columns** for
  configurational/mutational (`Res1 Res2 ChainRes1 ChainRes2 DensityRes1 DensityRes2
  AA1 AA2 NativeEnergy DecoyEnergy SDEnergy FrstIndex Welltype FrstState`); **8 columns**
  for singleresidue (no `FrstState`).
- `FrustrationData/protein.pdb_<mode>_5adens` — 5 Å density table.
- `FrustrationData/protein.pdb_<mode>_density.pkl` — pickled density results.
- With `graphics=True`: Plotly HTML/PNG figures in the job directory.

## Frustration classes

For configurational/mutational contacts (single source of truth:
`frustrapy/core/constants.py`):

| Class | Condition |
|---|---|
| highly frustrated | `FrstIndex ≤ -1` |
| neutral | `-1 < FrstIndex < 0.78` |
| minimally frustrated | `FrstIndex ≥ 0.78` |

The `0.78` cutoff is derived analytically (arXiv:1812.05965). Single-residue *plots*
use a distinct minimally cutoff of `0.58`, matching frustratometeR's visualization.

## Parallelism and resource limits

FrustraPy runs work concurrently on three axes — a batch of structures
(`dir_frustration(..., n_procs=K)` and `dynamic_frustration` over trajectory
frames), the per-residue saturation scan (`mutate_res_scan_parallel`), and the
per-structure precompute in the evolution module. These can nest: a parallel
batch can hand each structure a singleresidue scan that opens its own pool. To
stop that nest from multiplying into a fork bomb, every pool draws from one
shared budget instead of calling `cpu_count()` on its own.

The budget lives in `frustrapy/utils/concurrency.py` and works as follows:

- **One core budget.** `cpu_budget()` is the single source of how many cores
  FrustraPy may use (physical cores, never less than 1). No call site invents a
  larger bound.
- **Nested pools split the budget, they do not multiply it.**
  `resolve_concurrency(n_procs, n_items)` returns an `(outer, inner)` pair where
  `outer` items run concurrently and each gets `inner = cores // outer` cores
  for its own inner pool. The invariant `outer * inner <= cores` holds for any
  input, so two levels of nesting can never exceed the core budget. A flat
  (non-nested) pool uses `resolve_pool_size(...)`, which is
  `min(requested, cores, n_tasks)`.
- **Each worker is pinned to one native-math thread.** `apply_thread_limits()`
  (in the dependency-free `frustrapy/_threadlimits.py`, applied at the top of
  `frustrapy/__init__.py` before numpy loads) sets `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and two
  related vars to `1` via `setdefault` — so `cores` worker processes cannot each
  spin up `cores` BLAS threads (which would be `cores**2` threads). If you
  export your own thread counts, FrustraPy leaves them untouched.
- **Pools fork safely and shut down cleanly.** Worker pools are created from a
  `forkserver` (then `spawn`) start method via `get_pool_context()`, so forking
  from a possibly multi-threaded parent cannot deadlock the child or orphan a
  LAMMPS subprocess. Pools are always closed and joined, including on a worker
  error or timeout.

In practice this means `n_procs=K` is a ceiling on how many structures run at
once, not a multiplier: a user cannot fork-bomb the machine by combining a large
`n_procs` with a singleresidue batch or an evolution run. To cap usage further,
set `n_procs` to a small number; to give each native-math library more than one
thread, export e.g. `OMP_NUM_THREADS` before importing FrustraPy. The full audit
of every parallel construct and its nesting paths is in
`docs/parallelism_inventory.md`.

## Known limitations

- **Not yet on PyPI** — install from source (the release workflow is ready).
- The LAMMPS binaries are bundled in the wheel for now; a `fetch_lammps()` /
  download-on-first-use delivery (to slim the wheel) is planned.
- On Apple Silicon the macOS binaries require Rosetta 2.

## Roadmap

The following are planned and **not yet implemented**. Each will be parity-gated
against the `lammps` reference (FrstIndex agreement, CPU/GPU determinism) before any
speedup is claimed:

- **Pluggable compute backends.** Refactor the engine behind a `FrustrationBackend`
  interface so the current AWSEM/LAMMPS path (`lammps`, the numerical reference) can be
  swapped for alternative implementations behind one public API.
- **Native GPU backend.** A C++ core with two device paths — CUDA (NVIDIA) and Apple
  MPS/Metal (Apple Silicon) — plus a CPU fallback, exposed to Python, computing the
  AWSEM energy, decoy ensemble and 5 Å density reductions on the device.
- **FrustraMPNN.** Integrate the message-passing-network frustration model as a
  first-class module with full parity against its reference outputs.

A separate, browser-native WebGPU engine and an interactive web demo are tracked in
their own repository, outside this Python package.

## Citation

If you use FrustraPy in your research, please cite the following papers:

```bibtex
@article{parra2016protein,
  title={Protein Frustratometer 2: a tool to localize energetic frustration in protein molecules, now with electrostatics},
  author={Parra, R Gonzalo and Schafer, Nicholas P and Radusky, Leandro G and Tsai, Min-Yeh and Guzovsky, A Brenda and Wolynes, Peter G and Ferreiro, Diego U},
  journal={Nucleic acids research},
  volume={44},
  number={W1},
  pages={W356--W360},
  year={2016},
  publisher={Oxford University Press}
}
@article{jenik2012protein,
  title={Protein frustratometer: a tool to localize energetic frustration in protein molecules},
  author={Jenik, Michael and Parra, R Gonzalo and Radusky, Leandro G and Turjanski, Adrian and Wolynes, Peter G and Ferreiro, Diego U},
  journal={Nucleic acids research},
  volume={40},
  number={W1},
  pages={W348--W351},
  year={2012},
  publisher={Oxford University Press}
}
@article{rausch2021frustratometer,
  title={FrustratometeR: an R-package to compute local frustration in protein structures, point mutants and MD simulations},
  author={Rausch, Atilio O and Freiberger, Maria I and Leonetti, Cesar O and Luna, Diego M and Radusky, Leandro G and Wolynes, Peter G and Ferreiro, Diego U and Parra, R Gonzalo},
  journal={Bioinformatics},
  volume={37},
  number={18},
  pages={3038--3040},
  year={2021},
  publisher={Oxford University Press}
}
```

A Zenodo DOI for FrustraPy itself will be added here with the first tagged release.
See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## License

GPL-3.0-or-later. See the [`LICENSE`](LICENSE) file (GNU General Public License v3).
The bundled AWSEM/LAMMPS binaries are GPL, so the distribution is GPL regardless.

## Acknowledgments

- The original Frustratometer developers.
