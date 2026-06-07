# High-throughput workflow and storage benchmark

This document covers running FrustraPy at batch scale (10^4-10^6 structures) on a
cluster: why the default per-structure text output becomes a filesystem problem, how the
optional HDF5 store fixes it, a measured text-vs-HDF5 comparison, and a Slurm-style
array-job recipe. It assumes the store from `docs/HDF5_STORE_DESIGN.md` and the
parallel-safety budget from the README ("Parallelism and resource limits").

The text tables stay the default and the numerical reference; the HDF5 store is opt-in
and round-trips value-identical to them (the O3 parity gate, re-checked by the benchmark
on every run).

## The problem at scale

A single prediction writes one `{base}.done/FrustrationData/` directory holding the
parsed table for its mode, the `*_5adens` density table, the `*_density.pkl`, and the raw
`tertiary_frustration.dat` — five files plus two directories, about seven inodes per
structure. A genome- or family-scale batch of 10^5-10^6 structures is then millions of
inodes. On a shared parallel filesystem (Lustre, GPFS, NFS) the cost is the metadata
path: file create/stat/list operations and per-directory locking, plus inode quotas that
cap a project long before its byte quota. The actual frustration data is only a few
hundred kilobytes per structure; the bottleneck is the file count, not the volume.

The HDF5 store collapses the parsed tables for a whole batch into one compressed file
(one inode), or a handful of per-worker shards. See `docs/HDF5_STORE_DESIGN.md` for the
group layout, the compound-dtype encoding, and the gzip/chunking choices.

## Measured comparison (text vs one HDF5 store)

`frustrapy.output.benchmark.benchmark_text_vs_hdf5` replicates one real, parity-passing
`done` directory `n_structures` times and measures both layouts: inode count, on-disk
bytes, write time, and read/aggregate time. Replicating a single verified structure
isolates the storage-format cost from structure-to-structure variation; the function
asserts the HDF5 read-back is value-identical to the text aggregate before returning, so
the numbers below are parity-passing.

Numbers from a 1CRN (46-residue) run, `seq_dist=12`, in the dev container (local overlay
filesystem):

| mode | N | text inodes | HDF5 inodes | text MiB | HDF5 MiB | inode ratio | size ratio |
|------|----|-------------|-------------|----------|----------|-------------|------------|
| configurational | 1000 | 7001 | 1 | 89.8 | 50.1 | 7001x | 1.8x |
| mutational      | 1000 | 7001 | 1 | 89.8 | 53.2 | 7001x | 1.7x |
| singleresidue   | 1000 | 5001 | 1 | 31.0 |  9.1 | 5001x | 3.4x |

The inode reduction is the headline: thousands of files and directories become one file.
gzip level 4 with the shuffle filter shrinks the data 1.7x-3.4x on top of that (the
single-residue table compresses best — fewer, more regular columns).

Honest caveat on the time columns. On this local fast filesystem the text **write** is a
cheap file copy and the HDF5 write is slower (it parses the tables and gzip-compresses),
and the text **read/aggregate** is likewise faster than decompressing the store. The
benchmark does not reproduce a shared parallel filesystem, where creating and stat-ing
millions of small files — not raw bytes — is the dominant cost and where the single-file
store wins on write and read as well. Treat the inode and size columns as the portable
result and the time columns as local-filesystem-specific. To measure on real cluster
storage, run the function with `keep=True` against a path on the target filesystem.

Reproduce:

```python
from frustrapy.output import benchmark_text_vs_hdf5
r = benchmark_text_vs_hdf5(
    source_done_dir="results/1crn.done",   # any real, parity-passing run
    mode="configurational",
    n_structures=1000,
    workdir="/scratch/bench",
)
print(r.summary())   # parity_ok, inodes, bytes, write/read seconds
print(r.to_frame())  # one row per layout
```

## Array-job workflow: shards then merge or read-together

HDF5 is not safe for concurrent writers without an MPI-IO build, which is not assumed on
every cluster. The scalable pattern is one writer per file: each worker writes its own
shard, then the shards are either merged once or read together unmerged.

1. **Partition.** Split the structure list into one slice per array task (e.g. `$SLURM_ARRAY_TASK_ID`).
2. **Write a shard.** Each task runs its slice and writes the results into its own
   `batch.part-{rank}.h5` — one writer per file, no lock, no contention:

   ```python
   from frustrapy.output import FrustrationStore
   with FrustrationStore(f"batch.part-{rank}.h5", mode="w") as store:
       for sid, done_dir in slice_results:          # done_dir from a normal run
           store.write_structure(sid, mode, done_dir, seq_dist=12)
   ```

3. **Combine, two options.**
   - **Read together (no merge):** open the shards as one logical store. Shards are
     assumed disjoint in their structure ids (each worker owns a slice):

     ```python
     from frustrapy.output import FrustrationCorpus
     import glob
     with FrustrationCorpus(sorted(glob.glob("batch.part-*.h5"))) as corpus:
         contacts = corpus.read_corpus("configurational", "contact")
         td = corpus.to_training_dataset(level="residue")
     ```
   - **Merge once:** copy each shard's groups into one `batch.h5` with `h5py`
     (`dst.copy(src[sid], sid)`). The merge is metadata movement — groups are copied
     whole, with no recompression — and is serial and cheap relative to the LAMMPS runs.

Each array task is itself the unit the shared concurrency budget governs: within a task,
`dir_frustration(..., n_procs=K)` runs structures concurrently under
`inner = max(1, cpu_count // K)` inner threads, so the task never oversubscribes its
allocated cores, and the BLAS/OpenMP thread pinning keeps the per-structure LAMMPS prep
single-threaded (see the README parallelism section). The HDF5 write is single-threaded
per shard and adds no extra processes, so it composes with that budget without a
fork-bomb.

## What the store holds vs the text directory

The store keeps the parsed, analysis-ready tables (the contact or single-residue table
and the `*_5adens` density table) plus per-structure and per-mode attributes (mode,
`seq_dist`, residue count, source PDB sha256, timestamp, FrustraPy version). It does not
keep the raw `tertiary_frustration.dat` or the `*_density.pkl`; those stay only in the
text output for anyone who needs the exact printed digits or the raw decoy dump. For a
training corpus or a batch analysis the parsed tables are what is read, so the store is
the durable artifact and the text directories can be discarded after ingestion.
