# Usage and public API

This page covers the public Python API, the `frustrapy.sdk` facade, the
command-line tool, and the on-disk output contract. See [Installation](Installation)
for setup and the venv-on-PATH requirement.

## Public API surface

The names exported at the top level of the `frustrapy` package:

| Name | Kind | Purpose |
|---|---|---|
| `calculate_frustration` | function | Run one structure through one mode. |
| `dir_frustration` | function | Run every `.pdb` in a directory (parallel batch). |
| `dynamic_frustration` | function | Run frustration over trajectory frames. |
| `get_frustration` | function | Read back a parsed frustration table as a DataFrame. |
| `mutate_res`, `mutate_res_parallel` | function | Saturation-mutagenesis scan of one residue. |
| `detect_dynamic_clusters` | function | Cluster residues by frustration over a trajectory (needs the `clustering` extra). |
| `analyze_family` | function | Evolutionary frustration over a family (FrustraEvo). |
| `plot_contact_map`, `plot_5andens`, `plot_5adens_proportions`, `plot_delta_frus` | function | Plotly figures. |
| `view_frustration_pymol` | function | Structure view. |
| `Pdb`, `Dynamic` | class | Result/handle types. |

The same callables are also re-exported from `frustrapy.sdk` (see
[The SDK facade](#the-sdk-facade)).

## Single structure

`calculate_frustration` returns a **4-tuple**
`(Pdb, plots, density, single_residue_data)`. Always unpack four values.

```python
import frustrapy

pdb, plots, density, single_res = frustrapy.calculate_frustration(
    pdb_file="protein.pdb",
    mode="configurational",   # or "mutational" / "singleresidue"
    seq_dist=12,              # 12 (default) or 3
    results_dir="results",
    graphics=False,           # True to also build Plotly figures
)
```

Selected parameters:

- `pdb_file` / `pdb_id` — a local PDB path, or an identifier to fetch.
- `chain` — a chain ID or list of chains to restrict to.
- `residues` — `{chain: [resnums]}`; required to populate the single-residue data
  slot in `singleresidue` mode.
- `mode` — `"configurational"`, `"mutational"`, or `"singleresidue"`.
- `seq_dist` — sequence separation for contacts; `3` or `12` (the two binaries
  provided).
- `graphics` — build Plotly figures into `plots`. With `graphics=False`, `plots`
  is an empty dict.
- `results_dir` — output root; results land under `{results_dir}/{pdb_base}.done/`.
- `n_cpus` — inner CPU budget for the mutation pool.

In configurational and mutational mode the `density` and `single_residue_data`
slots are `None`; only `singleresidue` with a non-empty `residues` selection
populates them.

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

### Reading tables back

```python
import pandas as pd
table = pd.read_csv(
    "results/protein.done/FrustrationData/protein.pdb_configurational", sep=r"\s+"
)
# or, from a finished Pdb object:
df = frustrapy.get_frustration(pdb)                 # full table
df_a = frustrapy.get_frustration(pdb, chain="A")    # filtered by chain
```

`get_frustration` branches its filter columns on `pdb.mode` (`ChainRes` for
single-residue, `ChainRes1`/`ChainRes2` for contacts), so pass filters consistent
with the mode.

## Directory of structures (parallel batch)

`dir_frustration` returns a **2-tuple** `(plots_by_pdb, density)`.

```python
plots_by_pdb, density = frustrapy.dir_frustration(
    pdbs_dir="pdbs_directory",
    mode="configurational",
    results_dir="results",
    n_procs=4,   # process structures concurrently under a shared core budget
)
```

`n_procs` is a ceiling on how many structures run at once, not a multiplier:
each inner mutation pool is throttled so the two nested levels never exceed the
core budget. `n_procs=None` or `1` keeps the historic serial loop.

## Trajectory

`dynamic_frustration` runs frustration over the frames of a trajectory and returns
a `Dynamic` object; it passes `n_procs` through to process frames concurrently
under the same shared budget. `detect_dynamic_clusters` then clusters residues by
their frustration profile across frames (requires the `clustering` extra). The
clustering is a faithful port of frustratometeR's residue-level method.

## Evolutionary frustration (FrustraEvo)

```python
result = frustrapy.analyze_family(
    fasta_file="family.fasta",
    job_id="fam1",
    reference_pdb="3lqd-A",
    pdb_dir="pdbs/",
    results_dir="results",
    n_procs=4,             # per-structure frustration calcs in parallel; output is identical
    keep_intermediates=False,  # production default: drop scratch, keep only IC_*/SeqIC_*
)
```

`analyze_family` returns a dict with `job_id`, `output_dir`, a `files` sub-dict of
output paths, and a `contacts` sub-dict of per-contact information content plus
MIN/NEU/MAX summary counts. `n_procs` only affects speed; outputs are
byte-identical regardless of its value. With `keep_intermediates=False` the
per-member scratch (the ~27 MB binary and the LAMMPS deck) is dropped as the run
proceeds and the whole working tree is removed once the IC tables are written.

## The SDK facade

`frustrapy.sdk` is a thin, documented facade that re-exports the stable public
entry points behind one import. Every name is the *same object* as the top-level
name (`frustrapy.calculate_frustration is frustrapy.sdk.calculate_frustration`); it
adds no logic. Use it when you want a single typed surface to depend on:

```python
import frustrapy.sdk as fp

pdb, plots, density, single_res = fp.calculate_frustration(
    pdb_file="1crn.pdb", mode="configurational"
)
table = fp.get_frustration(pdb)
```

It also exposes `mutate_res_scan_parallel`, `pyrosetta_available`, and the
`FrustrationDensityResults` type. The module docstring documents every return
contract in one place.

## Command-line tool

Installing the `cli` extra (`uv pip install -e ".[cli]"`) exposes a `frustrapy`
console command — thin wrappers over the same functions, with a Rich interface.
The extra adds `typer`; `frustrapy/cli/main.py` imports it lazily, so plain
`import frustrapy` never requires it.

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
| `mutate` | saturation scan of one residue (`mutate_res_parallel`) | `frustrapy mutate protein.pdb --res 10 -c A --method threading -o results` |

Each subcommand writes the same on-disk output described below, prints a summary,
and returns a non-zero exit code with an actionable message on failure.

## Mutation scan

```python
import frustrapy
from frustrapy.analysis.mutations import mutate_res_parallel

pdb, *_ = frustrapy.calculate_frustration(
    pdb_file="1crn.pdb", mode="singleresidue",
    residues={"A": [10]}, graphics=False, visualization=False,
)
mutate_res_parallel(pdb, res_num=10, chain="A", method="threading")  # or "pyrosetta"
```

The scan returns the input `Pdb` with `pdb.Mutations[method]` and
`pdb.MutationAnalysis` populated. See [Backends](Backends) for the difference
between the `threading` and `pyrosetta` methods.

## Output contract

For `protein.pdb` and `results_dir="results"`, outputs land under
`results/protein.done/`:

- `FrustrationData/protein.pdb_<mode>` — the main table. **14 columns** for
  configurational/mutational (`Res1 Res2 ChainRes1 ChainRes2 DensityRes1
  DensityRes2 AA1 AA2 NativeEnergy DecoyEnergy SDEnergy FrstIndex Welltype
  FrstState`); **8 columns** for singleresidue (no `FrstState`).
- `FrustrationData/protein.pdb_<mode>_5adens` — 5 Angstrom density table.
- `FrustrationData/protein.pdb_<mode>_density.pkl` — pickled density results.
- `FrustrationData/tertiary_frustration.dat` — the raw binary output the tables are
  parsed from.
- With `graphics=True`: Plotly HTML/PNG figures in the job directory.

Energy and index columns are copied verbatim as strings with no reformatting; if a
value looks wrong, the input or the binary is the cause, not the parser. The
on-disk tables are the authoritative result, not the in-memory return value.
