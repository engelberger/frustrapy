# Installation

FrustraPy is not yet on PyPI; install from source. The release workflow (PEP 621 /
Hatchling build, OIDC trusted publishing) is in place but the first release has not
been pushed. A source install resolves the core runtime dependencies automatically.

## Requirements

- **Python 3.10-3.12.**
- **Perl** on `PATH` (`/usr/bin/perl`) — invoked by the visualization and
  charge-file steps (`GenerateVisualizations.pl`, `GenerateChargeFile.pl`). It is a
  runtime dependency and is not installed by pip.
- **A LAMMPS `lmp_serial` binary**, shipped precompiled in
  `frustrapy/core/scripts/` as `lmp_serial_{3,12}_{Linux,MacOS}`. Only `seq_dist=3`
  and `seq_dist=12` are provided (12 is the default). The macOS binaries are
  x86_64-only; on Apple Silicon they run under Rosetta 2
  (`softwareupdate --install-rosetta`). Linux is the supported path.

## Install from source

Using [uv](https://github.com/astral-sh/uv) (recommended) or plain `pip`:

```bash
git clone https://github.com/engelberger/frustrapy.git
cd frustrapy

# Create and ACTIVATE an environment. Activation matters: see "venv on PATH" below.
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install the package — the 9 core runtime deps are pulled in automatically.
uv pip install -e .
```

Verify:

```bash
python -c "import frustrapy; print(frustrapy.__version__)"
```

## Optional extras

```bash
uv pip install -e ".[viz]"         # 3D viewers (py3Dmol) + static PNG export (kaleido)
uv pip install -e ".[clustering]"  # detect_dynamic_clusters (scipy/sklearn/igraph/leidenalg/statsmodels)
uv pip install -e ".[perf]"        # memory logging (psutil)
uv pip install -e ".[pyrosetta]"   # pyrosetta-installer helper for the PyRosetta mutation backend
uv pip install -e ".[cli]"         # the `frustrapy` command-line tool (Typer + Rich)
uv pip install -e ".[all]"         # everything above
```

PyRosetta is not bundled. It is free for academic/non-commercial use under the
[RosettaCommons license](https://www.rosettacommons.org/software/license-and-download)
and is installed separately after the extra:

```bash
uv pip install -e ".[pyrosetta]"
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
```

See [Backends](Backends) for when each mutation method applies.

## venv on PATH (required for any calculation)

A calculation spawns a bare `python3` subprocess from `PdbCoords2Lammps.sh`. That
subprocess must find Biopython on `PATH`, or the run fails mid-way with
`ModuleNotFoundError: No module named 'Bio'`. Activating the environment
(`source .venv/bin/activate`, or exporting `PATH=.venv/bin:$PATH`) satisfies this.
Invoking the interpreter as `.venv/bin/python` without activation is not enough,
because the child process inherits the un-activated `PATH`.

## HPC and Singularity

On a cluster, build a container once and run the binaries inside it. A typical
Singularity/Apptainer recipe installs Python 3.12, Perl, and the package
(`pip install -e .`) into the image; activation is handled by the image's entry
point so the venv-on-PATH requirement is met. For array jobs over many structures
see `docs/HPC_WORKFLOW.md` in the repository, which covers per-task output layout
and the HDF5 store for aggregating results across shards.

## Parallelism and thread limits

FrustraPy runs work concurrently across structures, residues, and per-structure
precompute, and these can nest. All pools draw from one shared core budget rather
than each calling `cpu_count()`, and each worker is pinned to a single
native-math thread, so nesting cannot oversubscribe the machine. To cap usage,
pass a small `n_procs`; to give native-math libraries more than one thread, export
`OMP_NUM_THREADS` before importing FrustraPy. Details are in the README section
"Parallelism and resource limits" and in `docs/parallelism_inventory.md`.

## Clean-import and dependency notes

After `uv pip install -e .`, `import frustrapy` works with no manual dependency
installation; the build declares all nine runtime packages. The CI `build` job
asserts the wheel ships the runtime assets (`lmp_serial`, `PdbCoords2Lammps.sh`,
coefficient files, the Perl scripts) and declares at least nine runtime
dependencies, so a packaging regression fails the build.
