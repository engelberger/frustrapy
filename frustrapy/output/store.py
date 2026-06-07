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
from typing import List, Optional

import numpy as np
import pandas as pd

from .schema import (
    DENSITY_5ADENS_TABLE,
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
