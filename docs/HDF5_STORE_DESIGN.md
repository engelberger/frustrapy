# HDF5 store design (high-throughput output)

This document specifies the optional HDF5 store that FrustraPy can write instead of
thousands of small text files when running many predictions on a cluster. The text
tables under `{results_dir}/{base}.done/FrustrationData/` stay the default output and
the numerical reference; the HDF5 store is an opt-in alternative that must round-trip
value-identical to them (the parity gate enforced in O3).

`h5py` is an optional dependency (the `hdf5` extra). It is imported lazily inside the
store, so `import frustrapy` and the text path work with `h5py` absent.

## Why a single store

A per-structure run writes ~5 files (`*_configurational`, `*_singleresidue`,
`*_5adens`, `*_density.pkl`, `tertiary_frustration.dat`). A genome- or family-scale
batch of 10^4-10^6 structures therefore creates millions of inodes. On a shared
cluster filesystem (Lustre/GPFS/NFS) this is the bottleneck: metadata operations
(create/stat/list), per-file overhead, and quota inode limits dominate over the few
hundred bytes of actual data per table. One compressed HDF5 file per batch (or a small
number of shards) collapses that to a handful of inodes and lets the data compress
across structures.

## File and group layout

One HDF5 file holds many structures. The hierarchy is structure -> mode -> table:

```
/                                  root
  attrs: format_version            int   (this layout's version; bumped on breaking change)
         frustrapy_version         str
         created_utc               str   (ISO-8601; supplied by caller, not generated in-store)

/{structure_id}/                   one group per structure (default id = pdb base name)
    attrs: pdb_base                str
           source_pdb_sha256       str   (sha256 of the input PDB bytes)

  /{structure_id}/{mode}/          one group per calculation mode for that structure
      attrs: mode                  str   (configurational | mutational | singleresidue)
             seq_dist              int   (12 or 3)
             n_residues            int
             timestamp             str   (ISO-8601; caller-supplied)

      .../contact                  dataset, present for configurational/mutational
      .../singleresidue            dataset, present for singleresidue
      .../density_5adens           dataset, the 5 A density table
```

The dataset names are the schema keys (`contact`, `singleresidue`, `density_5adens`)
from `frustrapy/output/schema.py`, so the store and the contract share one vocabulary.
A structure analysed in two modes lands two mode subgroups under the same structure
group; the data is not duplicated across modes.

`structure_id` defaults to the PDB base name. Callers running a corpus with colliding
base names pass an explicit id (e.g. an accession) so groups do not clash. Group names
are HDF5 paths, so any `/` in an id is rejected by the writer.

## Dataset encoding (the parity crux)

Each table is stored as a single dataset whose rows are records of a NumPy
**structured (compound) dtype** built from the table schema. Field order equals the
schema column order; field names equal the schema column names. The per-column dtype
maps from the schema's logical dtype:

| schema dtype | stored NumPy field dtype | rationale |
|--------------|--------------------------|-----------|
| `int`        | `int64`                  | exact     |
| `float`      | `float64`                | exact for the round-trip (see below) |
| `str`        | `h5py.string_dtype("utf-8")` (variable-length) | exact bytes |

The text tables print energies and indices as decimal strings emitted by the
AWSEM/LAMMPS binary. Parsing such a printed `double` with `float()` and storing it as
`float64` is exact: the decimal came from a `double`, so `float64 -> text -> float64`
is the identity. The round-trip comparison (O3) is therefore value-identical: read the
text table and the HDF5 dataset both into a DataFrame and compare numeric columns by
exact `float64` equality and string columns by exact string equality. String columns
(`ChainRes`, `AA`, `Welltype`, `FrstState`, FrustraEvo state labels) are stored as
their on-disk bytes, so they are byte-identical, not merely value-identical.

The store does not reformat numbers back into the binary's exact print format, so the
HDF5 path is value-identical, not byte-identical, on the numeric columns. The text path
remains the byte reference for anyone who needs the exact printed digits.

FrustraEvo `IC_*` / `SeqIC_*` tables use the same compound-dtype encoding and live in a
parallel family layout (`/family/{reference}/ic_configurational`, etc.); they are in
scope for the store but the first writer (O3) targets the per-structure tables, which
are the verified 1CRN parity anchor.

## Compression and chunking

* **Compression:** gzip level 4, with the shuffle filter on numeric datasets. gzip is
  in every HDF5 build (unlike blosc/lzf-via-plugins), so a store written on the cluster
  reads anywhere. Shuffle reorders bytes by significance before gzip and markedly helps
  the many near-equal `float64` energies. Level 4 is the size/CPU knee for this data;
  it is a single constant (`GZIP_LEVEL`) and easy to retune.
* **Chunking:** datasets are chunked by whole rows — chunk shape `(n_rows,)` for a
  table written in one shot, so each table is one chunk. Tables are small (hundreds of
  rows) and always read whole, so one chunk per table avoids partial-chunk read
  overhead. HDF5 requires chunking for compression, so even tiny tables are chunked.

## Parallel write at scale

The store composes with the shared concurrency budget rather than opening one HDF5
handle across many processes (HDF5 is not safe for concurrent writers without MPI-IO or
SWMR, and parallel HDF5 is not assumed on every cluster). The strategy is
**per-worker shards then merge**:

1. An array job (or a `ProcessPoolExecutor` under the existing shared core budget,
   `inner = max(1, cpu_count // n_procs)` — no nested fork-bomb) assigns each worker a
   slice of the structure list.
2. Each worker writes its own shard file `batch.part-{rank}.h5` (one writer per file,
   no contention, no lock).
3. A cheap final merge concatenates the shards into one `batch.h5` by copying groups
   (`h5py` `group.copy`), or the shards are kept and read together by the reader, which
   can open a list of shards as one logical store.

This keeps every HDF5 file single-writer, needs no MPI build, and the merge is metadata
movement (no recompression when groups are copied whole). The merge step is itself
serial and cheap relative to the LAMMPS runs.

## Reader (O4)

The reader opens one file or a list of shards and exposes: load one structure/mode as a
DataFrame, iterate all predictions, and aggregate a column across the corpus. A
`to_training_dataset(...)` exporter (O4) emits ML-ready per-residue / per-contact
feature arrays for the FrustraMPNN-pkl training-data use case. Reading a dataset back
reconstructs the DataFrame with the schema's column order and dtypes, which is what the
parity test compares against the text table.

## Module surface

`frustrapy/output/store.py`:

* `FORMAT_VERSION` — layout version integer.
* `GZIP_LEVEL` — compression level constant.
* `compound_dtype_for_schema(schema)` — build the NumPy structured dtype for a table
  schema (pure; no `h5py`). Used by both writer and tests.
* `structure_group_path(structure_id, mode)` / `dataset_path(structure_id, mode, key)`
  — HDF5 path builders (pure).
* `require_h5py()` — import `h5py` or raise `ImportError` with the
  `pip install frustrapy[hdf5]` hint.
* `FrustrationStore` — the writer/reader. O2 ships the importable skeleton (constructor
  + method signatures); O3 implements `write_structure`/`read_structure` against the
  parity gate.

## Scope of O2

Design (this document) plus the importable skeleton. No data is written yet; the writer
body and the round-trip parity test land in O3, the reader and training-data export in
O4, and the HPC benchmark in O5. The text path is untouched and stays the default.
