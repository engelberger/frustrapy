"""HPC benchmark: many text files vs one compressed HDF5 store (O5).

A high-throughput batch writes one ``FrustrationData/`` directory per structure (the
contact/single-residue table, the ``*_5adens`` density table, the ``*_density.pkl``, and
the raw ``tertiary_frustration.dat``). At 10^4-10^6 structures that is millions of
inodes on a shared cluster filesystem, where metadata operations and inode quotas
dominate over the few hundred bytes of data per table. The HDF5 store collapses the same
parsed tables into one compressed file. This module measures the difference: inode count,
on-disk bytes, write time, and read/aggregate time, for both layouts.

The benchmark replicates ONE real, parity-passing ``done`` directory ``n_structures``
times. Replicating a single verified structure isolates the storage-format cost (inodes,
compression, metadata operations) from structure-to-structure variation; it is not a
claim about runtime across diverse proteins. Every run asserts the HDF5 aggregate is
value-identical to the text aggregate (the same parity gate as O3), so only honest,
parity-passing numbers are reported.

``h5py`` is the optional ``hdf5`` extra; this module imports it lazily through the store.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from typing import List

import pandas as pd

from .schema import read_table, schema_for_mode
from .store import FrustrationStore, read_corpus_from


def count_inodes(path: str) -> int:
    """Count filesystem entries (the path itself plus every dir and file under it).

    This is the metric a cluster inode quota counts. An empty directory still costs one
    inode; the store's single file costs one regardless of how many structures it holds.
    """
    total = 1  # the path entry itself
    for _root, dirs, files in os.walk(path):
        total += len(dirs) + len(files)
    return total


def tree_size_bytes(path: str) -> int:
    """Total size in bytes of all regular files under ``path`` (symlinks skipped)."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def _read_text_corpus(done_dirs: List[str], mode: str) -> pd.DataFrame:
    """Aggregate the main ``mode`` table across many text ``done`` dirs into one frame.

    Mirrors :meth:`FrustrationStore.read_corpus`: prepend a ``StructureId`` column and
    concatenate, so the text and HDF5 aggregates are directly comparable.
    """
    schema = schema_for_mode(mode)
    frames = []
    for done_dir in done_dirs:
        base = _base_of(done_dir)
        sid = os.path.basename(os.path.normpath(done_dir))
        if sid.endswith(".done"):
            sid = sid[: -len(".done")]
        path = os.path.join(done_dir, "FrustrationData", schema.filename(base=base, mode=mode))
        df = read_table(path, schema)
        df.insert(0, "StructureId", sid)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["StructureId", *schema.column_names])
    return pd.concat(frames, ignore_index=True)


def _base_of(done_dir: str) -> str:
    base = os.path.basename(os.path.normpath(done_dir))
    if base.endswith(".done"):
        base = base[: -len(".done")]
    return base


def _assert_value_identical(text_df: pd.DataFrame, hdf5_df: pd.DataFrame, mode: str) -> None:
    """Raise if the two corpus aggregates differ on any schema column (the parity gate)."""
    schema = schema_for_mode(mode)
    cols = ["StructureId", *schema.column_names]
    if list(text_df.columns) != cols or list(hdf5_df.columns) != cols:
        raise AssertionError("benchmark corpus columns differ between text and HDF5")
    if len(text_df) != len(hdf5_df):
        raise AssertionError(
            f"benchmark row counts differ: text {len(text_df)} vs HDF5 {len(hdf5_df)}"
        )
    for col in schema.columns:
        t = text_df[col.name].tolist()
        h = hdf5_df[col.name].tolist()
        if col.dtype == "str":
            ok = h == [str(v) for v in t]
        elif col.dtype == "int":
            ok = h == [int(float(v)) for v in t]
        else:
            ok = h == [float(v) for v in t]
        if not ok:
            raise AssertionError(f"benchmark parity failed on column {col.name!r}")


@dataclass
class BenchmarkResult:
    """Storage-format comparison for ``n_structures`` predictions (text vs HDF5).

    Inode and byte counts are measured on disk; the times are wall-clock seconds from
    :func:`time.perf_counter`. ``parity_ok`` records that the HDF5 read-back aggregate was
    value-identical to the text aggregate (the benchmark raises otherwise, so a returned
    result always has ``parity_ok`` True). Ratios are convenience views.
    """

    n_structures: int
    mode: str
    text_inodes: int
    text_bytes: int
    text_write_seconds: float
    text_read_seconds: float
    hdf5_inodes: int
    hdf5_bytes: int
    hdf5_write_seconds: float
    hdf5_read_seconds: float
    parity_ok: bool = True

    @property
    def inode_ratio(self) -> float:
        """Text inodes per HDF5 inode (higher means the store saves more metadata)."""
        return self.text_inodes / self.hdf5_inodes if self.hdf5_inodes else float("nan")

    @property
    def size_ratio(self) -> float:
        """Text bytes per HDF5 byte (higher means the store compresses better)."""
        return self.text_bytes / self.hdf5_bytes if self.hdf5_bytes else float("nan")

    def to_frame(self) -> pd.DataFrame:
        """One-row-per-layout DataFrame (text, hdf5) for tabulating or printing."""
        return pd.DataFrame(
            [
                {
                    "layout": "text",
                    "inodes": self.text_inodes,
                    "bytes": self.text_bytes,
                    "write_seconds": self.text_write_seconds,
                    "read_seconds": self.text_read_seconds,
                },
                {
                    "layout": "hdf5",
                    "inodes": self.hdf5_inodes,
                    "bytes": self.hdf5_bytes,
                    "write_seconds": self.hdf5_write_seconds,
                    "read_seconds": self.hdf5_read_seconds,
                },
            ]
        )

    def summary(self) -> str:
        """Plain multi-line summary suitable for a benchmark log."""
        return (
            f"n_structures={self.n_structures} mode={self.mode} parity_ok={self.parity_ok}\n"
            f"  inodes : text {self.text_inodes:>8d}  hdf5 {self.hdf5_inodes:>6d}"
            f"  ({self.inode_ratio:.1f}x fewer)\n"
            f"  bytes  : text {self.text_bytes:>8d}  hdf5 {self.hdf5_bytes:>6d}"
            f"  ({self.size_ratio:.1f}x smaller)\n"
            f"  write_s: text {self.text_write_seconds:>8.3f}  hdf5 {self.hdf5_write_seconds:>6.3f}\n"
            f"  read_s : text {self.text_read_seconds:>8.3f}  hdf5 {self.hdf5_read_seconds:>6.3f}"
        )


def benchmark_text_vs_hdf5(
    source_done_dir: str,
    mode: str,
    n_structures: int,
    workdir: str,
    seq_dist: int = 12,
    keep: bool = False,
) -> BenchmarkResult:
    """Benchmark ``n_structures`` predictions as text dirs vs one HDF5 store.

    ``source_done_dir`` is one real, parity-passing ``{base}.done`` directory (produced by
    a normal run); it is replicated ``n_structures`` times under ``workdir`` so the only
    variable is the storage format. Returns a :class:`BenchmarkResult` with inode count,
    on-disk bytes, write time, and read/aggregate time for each layout. Raises
    ``AssertionError`` if the HDF5 read-back is not value-identical to the text aggregate.

    Set ``keep=True`` to leave the generated ``workdir/text`` and ``workdir/store.h5`` in
    place for inspection; by default the text replicas are removed after measurement.
    """
    if n_structures < 1:
        raise ValueError("n_structures must be >= 1")
    base = _base_of(source_done_dir)
    src_data = os.path.join(source_done_dir, "FrustrationData")
    if not os.path.isdir(src_data):
        raise FileNotFoundError(f"no FrustrationData under {source_done_dir}")

    os.makedirs(workdir, exist_ok=True)
    text_root = os.path.join(workdir, "text")
    if os.path.exists(text_root):
        shutil.rmtree(text_root)
    os.makedirs(text_root)
    h5_path = os.path.join(workdir, "store.h5")
    if os.path.exists(h5_path):
        os.remove(h5_path)

    sids = [f"{base}_{i:06d}" for i in range(n_structures)]

    # --- text layout: one FrustrationData dir per structure --------------------- #
    done_dirs = []
    t0 = time.perf_counter()
    for sid in sids:
        done_dir = os.path.join(text_root, f"{sid}.done")
        dst_data = os.path.join(done_dir, "FrustrationData")
        # Copy the parsed tables under the structure's own base name, mirroring a real
        # per-structure run (so filenames embed the structure id, as on a cluster).
        os.makedirs(dst_data)
        for name in os.listdir(src_data):
            new_name = name.replace(base, sid, 1) if name.startswith(base) else name
            shutil.copy2(os.path.join(src_data, name), os.path.join(dst_data, new_name))
        done_dirs.append(done_dir)
    text_write_seconds = time.perf_counter() - t0
    text_inodes = count_inodes(text_root)
    text_bytes = tree_size_bytes(text_root)

    # The copied tables carry the per-structure base; read them under that base.
    t0 = time.perf_counter()
    text_corpus = _read_text_corpus(done_dirs, mode)
    text_read_seconds = time.perf_counter() - t0

    # --- HDF5 layout: one compressed store, one group per structure ------------- #
    t0 = time.perf_counter()
    with FrustrationStore(h5_path, mode="w") as store:
        for sid, done_dir in zip(sids, done_dirs):
            store.write_structure(sid, mode, done_dir, seq_dist=seq_dist)
    hdf5_write_seconds = time.perf_counter() - t0
    hdf5_inodes = count_inodes(h5_path)
    hdf5_bytes = tree_size_bytes(h5_path)

    schema = schema_for_mode(mode)
    t0 = time.perf_counter()
    with FrustrationStore(h5_path) as store:
        hdf5_corpus = read_corpus_from(store, mode, schema.key)
    hdf5_read_seconds = time.perf_counter() - t0

    _assert_value_identical(text_corpus, hdf5_corpus, mode)

    if not keep:
        shutil.rmtree(text_root)
        os.remove(h5_path)

    return BenchmarkResult(
        n_structures=n_structures,
        mode=mode,
        text_inodes=text_inodes,
        text_bytes=text_bytes,
        text_write_seconds=text_write_seconds,
        text_read_seconds=text_read_seconds,
        hdf5_inodes=hdf5_inodes,
        hdf5_bytes=hdf5_bytes,
        hdf5_write_seconds=hdf5_write_seconds,
        hdf5_read_seconds=hdf5_read_seconds,
        parity_ok=True,
    )


__all__ = [
    "BenchmarkResult",
    "benchmark_text_vs_hdf5",
    "count_inodes",
    "tree_size_bytes",
]
