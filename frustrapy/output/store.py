"""Optional HDF5 store for high-throughput FrustraPy output.

For batches of many predictions, writing thousands of small text files saturates a
cluster filesystem's metadata path. This module collects the same tables into one
compressed HDF5 file (or a few shards) instead. See ``docs/HDF5_STORE_DESIGN.md`` for
the layout, compression, and parallel-write strategy.

The text tables under ``{results_dir}/{base}.done/FrustrationData/`` stay the default
output and the numerical reference. The HDF5 store is opt-in and must round-trip
value-identical to them (the O3 parity gate).

``h5py`` is an optional dependency (the ``hdf5`` extra). It is imported lazily inside
the methods that touch a file, so importing this module and using the text path work
with ``h5py`` absent. The pure helpers (dtype mapping, path builders) need only NumPy.

This is the O2 skeleton: the design, the constants, the pure helpers, and the class
surface. ``write_structure`` / ``read_structure`` are implemented in O3 against the
round-trip parity test.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .schema import (
    CONTACT_FEATURE_COLUMNS,
    CONTACT_ID_COLUMNS,
    CONTACT_TABLE,
    DENSITY_5ADENS_TABLE,
    RESIDUE_FEATURE_COLUMNS,
    RESIDUE_ID_COLUMNS,
    SINGLERESIDUE_TABLE,
    TableSchema,
    TABLE_SCHEMAS,
    read_table,
    schema_for_mode,
)

#: Layout version written to the file root ``attrs``; bump on a breaking change to the
#: group/dataset layout so a reader can refuse an incompatible store.
FORMAT_VERSION = 1

#: gzip compression level for datasets. gzip is in every HDF5 build, so a store written
#: on a cluster reads anywhere; level 4 is the size/CPU knee for this data.
GZIP_LEVEL = 4

#: NumPy field dtype for each schema logical dtype. ``str`` resolves to a variable-length
#: UTF-8 dtype only when ``h5py`` is available (see :func:`compound_dtype_for_schema`),
#: because the vlen-string dtype is provided by ``h5py``.
_NUMERIC_NP_DTYPE = {"int": "i8", "float": "f8"}


def require_h5py():
    """Return the imported ``h5py`` module, or raise a helpful ``ImportError``.

    ``h5py`` is the optional ``hdf5`` extra; a bare install does not have it and the
    text path does not need it.
    """
    try:
        import h5py  # noqa: WPS433 (deliberately lazy/optional)
    except ImportError as exc:  # pragma: no cover - exercised only without h5py
        raise ImportError(
            "the HDF5 store needs h5py, which is an optional dependency. "
            "Install it with `pip install frustrapy[hdf5]` (or `pip install h5py`). "
            "The default text output does not require it."
        ) from exc
    return h5py


def _string_dtype():
    """Variable-length UTF-8 dtype for string columns (needs ``h5py``)."""
    h5py = require_h5py()
    return h5py.string_dtype("utf-8")


def compound_dtype_for_schema(schema: TableSchema) -> np.dtype:
    """Build the NumPy structured dtype for a table schema.

    Field order and names equal the schema column order and names. ``int`` -> ``int64``,
    ``float`` -> ``float64``, ``str`` -> variable-length UTF-8 (so this needs ``h5py``
    only for the string case). Used by the writer and by the parity tests.
    """
    fields = []
    for col in schema.columns:
        if col.dtype == "str":
            fields.append((col.name, _string_dtype()))
        else:
            fields.append((col.name, _NUMERIC_NP_DTYPE[col.dtype]))
    return np.dtype(fields)


def _valid_id(structure_id: str) -> str:
    """Validate a structure id used as an HDF5 group name (no path separator)."""
    if not structure_id or "/" in structure_id:
        raise ValueError(f"invalid structure id {structure_id!r}: must be non-empty and contain no '/'")
    return structure_id


def structure_group_path(structure_id: str, mode: str) -> str:
    """HDF5 path of the per-structure, per-mode group, e.g. ``/1crn/configurational``."""
    return f"/{_valid_id(structure_id)}/{mode}"


def dataset_path(structure_id: str, mode: str, key: str) -> str:
    """HDF5 path of a table dataset within a structure/mode group.

    ``key`` is a schema key (``contact``, ``singleresidue``, ``density_5adens``).
    """
    if key not in TABLE_SCHEMAS:
        raise ValueError(f"unknown table key {key!r}; expected one of {sorted(TABLE_SCHEMAS)}")
    return f"{structure_group_path(structure_id, mode)}/{key}"


def family_group_path(family_id: str, reference: str) -> str:
    """HDF5 path of a FrustraEvo family/reference group, e.g. ``/globin3/1fsx-A``.

    The IC tables (``ic_configurational``, ``ic_mutational``, ``ic_singleres``,
    ``seqic``) live as datasets under this group, mirroring the per-structure layout.
    """
    return f"/{_valid_id(family_id)}/{_valid_id(reference)}"


# --------------------------------------------------------------------------- #
# DataFrame <-> structured-array conversion (the parity crux)
# --------------------------------------------------------------------------- #


def dataframe_to_records(df: pd.DataFrame, schema: TableSchema) -> np.ndarray:
    """Convert a string-valued table DataFrame to a structured array per ``schema``.

    Columns are cast by the schema's logical dtype: ``int`` -> ``int64`` (via
    ``int(float(v))`` so an integer printed as ``"12.0"`` is accepted), ``float`` ->
    ``float64`` (exact: the decimal text was printed from a ``double``), ``str`` ->
    variable-length UTF-8. The result is what the writer stores; reading it back and
    decoding the string fields reproduces the table value-identically.
    """
    if list(df.columns) != schema.column_names:
        raise ValueError(
            f"{schema.key}: columns {list(df.columns)} != schema {schema.column_names}"
        )
    dtype = compound_dtype_for_schema(schema)
    arr = np.empty(len(df), dtype=dtype)
    for col in schema.columns:
        values = df[col.name].tolist()
        if col.dtype == "int":
            arr[col.name] = [int(float(v)) for v in values]
        elif col.dtype == "float":
            arr[col.name] = [float(v) for v in values]
        else:
            arr[col.name] = [str(v) for v in values]
    return arr


def records_to_dataframe(arr: np.ndarray, schema: TableSchema) -> pd.DataFrame:
    """Reconstruct a typed DataFrame from a stored structured array per ``schema``.

    Numeric fields keep their ``int64`` / ``float64`` dtype; string fields (stored as
    vlen UTF-8, read back as ``bytes``) are decoded to ``str``. Column order is the
    schema order.
    """
    data = {}
    for col in schema.columns:
        values = arr[col.name]
        if col.dtype == "str":
            data[col.name] = [
                v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
                for v in values
            ]
        else:
            data[col.name] = values
    df = pd.DataFrame(data, columns=schema.column_names)
    for col in schema.columns:
        if col.dtype == "int":
            df[col.name] = df[col.name].astype("int64")
        elif col.dtype == "float":
            df[col.name] = df[col.name].astype("float64")
        else:
            df[col.name] = df[col.name].astype("object")
    return df


class FrustrationStore:
    """Writer/reader for the compressed HDF5 output store.

    :meth:`write_structure` parses the text tables for one structure/mode and stores
    them as compressed compound-dtype datasets; :meth:`read_structure` reads one back as
    a typed DataFrame. Values round-trip value-identical to the text tables (numeric
    columns exact, string columns byte-identical) — the O3 parity gate. FrustraEvo IC
    tables go through :meth:`write_family` / :meth:`read_family`. Corpus-level
    aggregation and training-data export land in O4.

    Usage::

        with FrustrationStore("batch.h5", mode="w") as store:
            store.write_structure("1crn", "configurational", done_dir, seq_dist=12)
        with FrustrationStore("batch.h5") as store:
            df = store.read_structure("1crn", "configurational", "contact")
    """

    def __init__(self, path: str, mode: str = "r"):
        self.path = path
        self.file_mode = mode
        self._h5 = None  # opened lazily on __enter__ / open()

    # -- lifecycle ---------------------------------------------------------- #

    def open(self) -> "FrustrationStore":
        """Open the underlying HDF5 file (requires ``h5py``)."""
        h5py = require_h5py()
        self._h5 = h5py.File(self.path, self.file_mode)
        if self.file_mode in ("w", "a") and "format_version" not in self._h5.attrs:
            self._h5.attrs["format_version"] = FORMAT_VERSION
        return self

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self) -> "FrustrationStore":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _require_open(self):
        if self._h5 is None:
            raise RuntimeError("store is not open; use `with FrustrationStore(path) as s:`")
        return self._h5

    def _write_dataset(self, group, key: str, schema: TableSchema, df: pd.DataFrame) -> None:
        """Write one table as a compressed compound-dtype dataset under ``group``."""
        records = dataframe_to_records(df, schema)
        nrows = records.shape[0]
        if nrows == 0:
            # Compression requires chunking, which requires a non-zero chunk; an empty
            # table is stored uncompressed (it carries no data to compress anyway).
            group.create_dataset(key, data=records)
            return
        group.create_dataset(
            key,
            data=records,
            chunks=(nrows,),
            compression="gzip",
            compression_opts=GZIP_LEVEL,
            shuffle=True,
        )

    # -- writer (O3) -------------------------------------------------------- #

    def write_structure(
        self,
        structure_id: str,
        mode: str,
        done_dir: str,
        seq_dist: int,
        source_pdb_sha256: Optional[str] = None,
        timestamp: Optional[str] = None,
        frustrapy_version: Optional[str] = None,
    ) -> None:
        """Write one structure's tables (parsed from ``done_dir``) into the store.

        Reads the text tables for ``mode`` (the contact or single-residue table, plus
        the ``*_5adens`` density table when present) from ``{done_dir}/FrustrationData/``
        and stores each as a compressed compound-dtype dataset under
        :func:`structure_group_path`, with the per-structure and per-mode attrs from the
        design doc. The values written round-trip value-identical to the text tables.
        """
        h5 = self._require_open()
        base = os.path.basename(os.path.normpath(done_dir))
        if base.endswith(".done"):
            base = base[: -len(".done")]
        data_dir = os.path.join(done_dir, "FrustrationData")

        table_schema = schema_for_mode(mode)
        main_path = os.path.join(data_dir, table_schema.filename(base=base, mode=mode))
        if not os.path.exists(main_path):
            raise FileNotFoundError(f"expected {mode} table not found: {main_path}")
        main_df = read_table(main_path, table_schema)

        group = h5.require_group(structure_group_path(structure_id, mode))
        self._write_dataset(group, table_schema.key, table_schema, main_df)

        dens_path = os.path.join(
            data_dir, DENSITY_5ADENS_TABLE.filename(base=base, mode=mode)
        )
        n_residues = None
        if os.path.exists(dens_path):
            dens_df = read_table(dens_path, DENSITY_5ADENS_TABLE)
            self._write_dataset(group, DENSITY_5ADENS_TABLE.key, DENSITY_5ADENS_TABLE, dens_df)
            n_residues = len(dens_df)

        # Per-structure attrs live on the parent group; per-mode attrs on this group.
        parent = h5.require_group(f"/{_valid_id(structure_id)}")
        parent.attrs["pdb_base"] = base
        if source_pdb_sha256 is not None:
            parent.attrs["source_pdb_sha256"] = source_pdb_sha256
        group.attrs["mode"] = mode
        group.attrs["seq_dist"] = int(seq_dist)
        if n_residues is not None:
            group.attrs["n_residues"] = int(n_residues)
        if timestamp is not None:
            group.attrs["timestamp"] = timestamp
        if frustrapy_version is not None and "frustrapy_version" not in h5.attrs:
            h5.attrs["frustrapy_version"] = frustrapy_version

    def write_family(
        self,
        family_id: str,
        reference: str,
        results_dir: str,
        timestamp: Optional[str] = None,
        frustrapy_version: Optional[str] = None,
    ) -> List[str]:
        """Write a FrustraEvo family's information-content tables into the store.

        Reads whichever ``IC_*``/``SeqIC_*`` tables for ``reference`` exist directly in
        ``results_dir`` and stores each as a dataset under
        :func:`family_group_path`. Returns the schema keys written.
        """
        h5 = self._require_open()
        group = h5.require_group(family_group_path(family_id, reference))
        written: List[str] = []
        for key in ("ic_configurational", "ic_mutational", "ic_singleres", "seqic"):
            schema = TABLE_SCHEMAS[key]
            path = os.path.join(results_dir, schema.filename(reference=reference))
            if not os.path.exists(path):
                continue
            df = read_table(path, schema)
            self._write_dataset(group, key, schema, df)
            written.append(key)
        group.attrs["reference"] = reference
        if timestamp is not None:
            group.attrs["timestamp"] = timestamp
        if frustrapy_version is not None and "frustrapy_version" not in h5.attrs:
            h5.attrs["frustrapy_version"] = frustrapy_version
        return written

    # -- reader (O3 round-trip / O4 analysis) ------------------------------ #

    def _read_dataset(self, group_path: str, key: str) -> pd.DataFrame:
        h5 = self._require_open()
        if key not in TABLE_SCHEMAS:
            raise ValueError(f"unknown table key {key!r}; expected one of {sorted(TABLE_SCHEMAS)}")
        full = f"{group_path}/{key}"
        if full not in h5:
            raise KeyError(f"dataset not found in store: {full}")
        return records_to_dataframe(h5[full][:], TABLE_SCHEMAS[key])

    def read_structure(self, structure_id: str, mode: str, key: str) -> pd.DataFrame:
        """Read one stored per-structure table back as a typed pandas DataFrame.

        ``key`` is a schema key (``contact``/``singleresidue``/``density_5adens``). The
        returned frame has the schema column order and dtypes; numeric columns are
        value-identical to the text table, string columns byte-identical.
        """
        return self._read_dataset(structure_group_path(structure_id, mode), key)

    def read_family(self, family_id: str, reference: str, key: str) -> pd.DataFrame:
        """Read one stored FrustraEvo IC table back as a typed pandas DataFrame."""
        return self._read_dataset(family_group_path(family_id, reference), key)

    def list_structures(self) -> List[str]:
        """List the structure ids (top-level groups) present in the store."""
        h5 = self._require_open()
        h5py = require_h5py()
        return [k for k in h5.keys() if isinstance(h5[k], h5py.Group)]

    # -- corpus traversal / aggregation (O4) -------------------------------- #

    def list_modes(self, structure_id: str) -> List[str]:
        """List the modes (per-structure subgroups) stored for ``structure_id``."""
        h5 = self._require_open()
        h5py = require_h5py()
        grp = h5.get(f"/{_valid_id(structure_id)}")
        if grp is None:
            raise KeyError(f"structure not found in store: {structure_id}")
        return [k for k in grp.keys() if isinstance(grp[k], h5py.Group)]

    def list_tables(self, structure_id: str, mode: str) -> List[str]:
        """List the table dataset keys stored for ``structure_id`` in ``mode``."""
        h5 = self._require_open()
        h5py = require_h5py()
        grp = h5.get(structure_group_path(structure_id, mode))
        if grp is None:
            raise KeyError(f"structure/mode not found in store: {structure_id}/{mode}")
        return [k for k in grp.keys() if isinstance(grp[k], h5py.Dataset)]

    def iter_tables(
        self, mode: Optional[str] = None, key: Optional[str] = None
    ) -> Iterator[Tuple[str, str, str, pd.DataFrame]]:
        """Iterate stored per-structure tables across the whole store.

        Yields ``(structure_id, mode, key, df)`` for every table, optionally filtered to
        one ``mode`` (e.g. ``"configurational"``) and/or one table ``key`` (e.g.
        ``"contact"``). This is the corpus primitive the reader/exporter build on.
        """
        for sid in self.list_structures():
            for m in self.list_modes(sid):
                if mode is not None and m != mode:
                    continue
                for k in self.list_tables(sid, m):
                    if key is not None and k != key:
                        continue
                    yield sid, m, k, self.read_structure(sid, m, k)

    def read_corpus(self, mode: str, key: str) -> pd.DataFrame:
        """Concatenate one table across every structure into a single DataFrame.

        Prepends a ``StructureId`` column identifying the source structure; the rest of
        the columns are the schema columns. Useful for an aggregate analysis over a
        whole batch (e.g. the distribution of ``FrstIndex`` across a corpus).
        """
        return read_corpus_from(self, mode, key)

    def to_training_dataset(
        self,
        level: str = "residue",
        mode: Optional[str] = None,
        feature_columns: Optional[Sequence[str]] = None,
    ) -> "TrainingDataset":
        """Export ML-ready feature arrays over the whole store (see :func:`build_training_dataset`)."""
        return build_training_dataset(self, level=level, mode=mode, feature_columns=feature_columns)


# --------------------------------------------------------------------------- #
# Multi-shard reader (O4): open a list of shard files as one logical store
# --------------------------------------------------------------------------- #


class FrustrationCorpus:
    """Read-only view over one or more HDF5 shard files as a single logical store.

    The parallel-write strategy is per-worker shards then (optionally) merge; this
    reader lets the un-merged shards be read together. Shards are assumed disjoint in
    their structure ids (each worker writes its own slice) — a structure appearing in
    two shards is iterated twice. Exposes the same traversal/aggregation surface as
    :class:`FrustrationStore` (:meth:`list_structures`, :meth:`iter_tables`,
    :meth:`read_corpus`, :meth:`to_training_dataset`).

    Usage::

        with FrustrationCorpus(["batch.part-0.h5", "batch.part-1.h5"]) as corpus:
            df = corpus.read_corpus("configurational", "contact")
    """

    def __init__(self, paths: Iterable[str]):
        self.paths = list(paths)
        self._stores: List[FrustrationStore] = []

    def open(self) -> "FrustrationCorpus":
        self._stores = [FrustrationStore(p, "r").open() for p in self.paths]
        return self

    def close(self) -> None:
        for store in self._stores:
            store.close()
        self._stores = []

    def __enter__(self) -> "FrustrationCorpus":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _require_open(self) -> List[FrustrationStore]:
        if not self._stores:
            raise RuntimeError("corpus is not open; use `with FrustrationCorpus(paths) as c:`")
        return self._stores

    def _store_for(self, structure_id: str) -> FrustrationStore:
        for store in self._require_open():
            if structure_id in store.list_structures():
                return store
        raise KeyError(f"structure not found in any shard: {structure_id}")

    def list_structures(self) -> List[str]:
        """Union of structure ids across all shards (first-seen order, de-duplicated)."""
        seen: List[str] = []
        for store in self._require_open():
            for sid in store.list_structures():
                if sid not in seen:
                    seen.append(sid)
        return seen

    def list_modes(self, structure_id: str) -> List[str]:
        return self._store_for(structure_id).list_modes(structure_id)

    def list_tables(self, structure_id: str, mode: str) -> List[str]:
        return self._store_for(structure_id).list_tables(structure_id, mode)

    def read_structure(self, structure_id: str, mode: str, key: str) -> pd.DataFrame:
        return self._store_for(structure_id).read_structure(structure_id, mode, key)

    def iter_tables(
        self, mode: Optional[str] = None, key: Optional[str] = None
    ) -> Iterator[Tuple[str, str, str, pd.DataFrame]]:
        for store in self._require_open():
            yield from store.iter_tables(mode=mode, key=key)

    def read_corpus(self, mode: str, key: str) -> pd.DataFrame:
        return read_corpus_from(self, mode, key)

    def to_training_dataset(
        self,
        level: str = "residue",
        mode: Optional[str] = None,
        feature_columns: Optional[Sequence[str]] = None,
    ) -> "TrainingDataset":
        return build_training_dataset(self, level=level, mode=mode, feature_columns=feature_columns)


# --------------------------------------------------------------------------- #
# Corpus aggregation + training-data export (O4)
# --------------------------------------------------------------------------- #


def read_corpus_from(reader, mode: str, key: str) -> pd.DataFrame:
    """Concatenate one table across a reader's structures into a single DataFrame.

    ``reader`` is anything exposing :meth:`iter_tables` (a :class:`FrustrationStore` or
    a :class:`FrustrationCorpus`). The result has a leading ``StructureId`` column and
    the table's schema columns; an empty corpus yields a typed empty frame.
    """
    if key not in TABLE_SCHEMAS:
        raise ValueError(f"unknown table key {key!r}; expected one of {sorted(TABLE_SCHEMAS)}")
    schema = TABLE_SCHEMAS[key]
    frames = []
    for sid, _m, _k, df in reader.iter_tables(mode=mode, key=key):
        df = df.copy()
        df.insert(0, "StructureId", sid)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["StructureId", *schema.column_names])
    return pd.concat(frames, ignore_index=True)


@dataclass
class _LevelSpec:
    """How a training-data ``level`` maps to a table, its modes, and its columns."""

    key: str
    default_mode: str
    modes: Tuple[str, ...]
    schema: TableSchema
    id_columns: Tuple[str, ...]
    feature_columns: Tuple[str, ...]


#: Training-data levels: per-residue (single-residue table) and per-contact (contact
#: table). ``contact`` defaults to configurational mode but accepts mutational.
_LEVEL_SPECS = {
    "residue": _LevelSpec(
        key="singleresidue",
        default_mode="singleresidue",
        modes=("singleresidue",),
        schema=SINGLERESIDUE_TABLE,
        id_columns=RESIDUE_ID_COLUMNS,
        feature_columns=RESIDUE_FEATURE_COLUMNS,
    ),
    "contact": _LevelSpec(
        key="contact",
        default_mode="configurational",
        modes=("configurational", "mutational"),
        schema=CONTACT_TABLE,
        id_columns=CONTACT_ID_COLUMNS,
        feature_columns=CONTACT_FEATURE_COLUMNS,
    ),
}


@dataclass
class TrainingDataset:
    """ML-ready feature arrays exported from the store (the FrustraMPNN-pkl use case).

    ``X`` is a ``(n_samples, n_features)`` ``float64`` array of the frustration
    measurements; ``feature_names`` labels its columns; ``ids`` is a DataFrame locating
    each row (``StructureId`` plus the level's identifier columns, e.g. ``Res``/``AA``
    for residues). ``level`` is ``"residue"`` or ``"contact"`` and ``mode`` the
    calculation mode the features came from. Numeric values are value-identical to the
    text tables (the same exact ``float64`` the round-trip parity gate compares).
    """

    X: np.ndarray
    feature_names: List[str]
    ids: pd.DataFrame
    level: str
    mode: str

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def to_frame(self) -> pd.DataFrame:
        """Combine ids and features into one DataFrame (ids first, then features)."""
        df = self.ids.copy().reset_index(drop=True)
        for j, name in enumerate(self.feature_names):
            df[name] = self.X[:, j]
        return df

    def save_npz(self, path: str) -> None:
        """Save the arrays to a compressed ``.npz`` (X, feature_names, and id columns).

        ``.npz`` is preferred over pickle for training data: it loads without executing
        arbitrary code. Reload with ``numpy.load(path, allow_pickle=False)`` for ``X``
        and ``feature_names``; the id columns are saved as named string arrays.
        """
        arrays = {
            "X": self.X,
            "feature_names": np.asarray(self.feature_names, dtype="U"),
            "level": np.asarray(self.level, dtype="U"),
            "mode": np.asarray(self.mode, dtype="U"),
        }
        for col in self.ids.columns:
            arrays[f"id__{col}"] = self.ids[col].to_numpy()
        np.savez_compressed(path, **arrays)


def build_training_dataset(
    reader,
    level: str = "residue",
    mode: Optional[str] = None,
    feature_columns: Optional[Sequence[str]] = None,
) -> TrainingDataset:
    """Assemble a :class:`TrainingDataset` from a reader's stored tables.

    ``reader`` is a :class:`FrustrationStore` or :class:`FrustrationCorpus`. ``level``
    selects per-residue or per-contact samples; ``mode`` defaults to the level's natural
    mode (``singleresidue`` / ``configurational``) and is validated against the level.
    ``feature_columns`` overrides the default numeric feature set (each must be a numeric
    schema column). Rows are stacked across the corpus in structure-then-row order.
    """
    if level not in _LEVEL_SPECS:
        raise ValueError(f"unknown level {level!r}; expected one of {sorted(_LEVEL_SPECS)}")
    spec = _LEVEL_SPECS[level]
    resolved_mode = mode if mode is not None else spec.default_mode
    if resolved_mode not in spec.modes:
        raise ValueError(
            f"mode {resolved_mode!r} is not valid for level {level!r}; expected one of {list(spec.modes)}"
        )

    by_name = {c.name: c for c in spec.schema.columns}
    feats = list(feature_columns) if feature_columns is not None else list(spec.feature_columns)
    for name in feats:
        if name not in by_name:
            raise ValueError(f"feature column {name!r} not in {spec.schema.key} schema")
        if by_name[name].dtype == "str":
            raise ValueError(f"feature column {name!r} is non-numeric and cannot be a feature")
    id_cols = [c for c in spec.id_columns if c in by_name]

    id_blocks: List[pd.DataFrame] = []
    x_blocks: List[np.ndarray] = []
    for sid, _m, _k, df in reader.iter_tables(mode=resolved_mode, key=spec.key):
        if len(df) == 0:
            continue
        block = pd.DataFrame({"StructureId": [sid] * len(df)})
        for col in id_cols:
            block[col] = df[col].to_numpy()
        id_blocks.append(block)
        x_blocks.append(df[feats].to_numpy(dtype="float64"))

    if x_blocks:
        X = np.concatenate(x_blocks, axis=0)
        ids = pd.concat(id_blocks, ignore_index=True)
    else:
        X = np.empty((0, len(feats)), dtype="float64")
        ids = pd.DataFrame(columns=["StructureId", *id_cols])

    return TrainingDataset(X=X, feature_names=feats, ids=ids, level=level, mode=resolved_mode)


def schema_for_table_key(key: str) -> TableSchema:
    """Schema for a stored dataset key (``contact`` / ``singleresidue`` / ``density_5adens``)."""
    if key not in TABLE_SCHEMAS:
        raise ValueError(f"unknown table key {key!r}; expected one of {sorted(TABLE_SCHEMAS)}")
    return TABLE_SCHEMAS[key]


# Re-export so callers can resolve a mode to its contact/singleresidue schema without
# reaching into .schema directly.
__all__ = [
    "FORMAT_VERSION",
    "GZIP_LEVEL",
    "FrustrationStore",
    "FrustrationCorpus",
    "TrainingDataset",
    "build_training_dataset",
    "read_corpus_from",
    "compound_dtype_for_schema",
    "dataframe_to_records",
    "records_to_dataframe",
    "structure_group_path",
    "dataset_path",
    "family_group_path",
    "require_h5py",
    "schema_for_table_key",
    "schema_for_mode",
]
