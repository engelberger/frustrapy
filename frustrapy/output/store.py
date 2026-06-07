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

from typing import List, Optional

import numpy as np

from .schema import TableSchema, TABLE_SCHEMAS, schema_for_mode

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


class FrustrationStore:
    """Writer/reader for the compressed HDF5 output store.

    O2 ships this skeleton: the constructor and the method surface. The ``h5py``-backed
    bodies of :meth:`write_structure` and :meth:`read_structure` are implemented in O3
    against the round-trip parity gate (values written then read back must equal the
    text tables exactly). The reader/aggregation and training-data export land in O4.

    Usage (target API)::

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

    # -- writer (O3) -------------------------------------------------------- #

    def write_structure(
        self,
        structure_id: str,
        mode: str,
        done_dir: str,
        seq_dist: int,
        source_pdb_sha256: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Write one structure's tables (parsed from ``done_dir``) into the store.

        Implemented in O3 against the round-trip parity gate. Reads the text tables for
        ``mode`` (plus ``density_5adens``) from
        ``{done_dir}/FrustrationData/`` and stores them under
        :func:`structure_group_path`, with the per-mode attrs from the design doc.
        """
        raise NotImplementedError("write_structure lands in O3 (writer + parity gate)")

    # -- reader (O3 round-trip / O4 analysis) ------------------------------ #

    def read_structure(self, structure_id: str, mode: str, key: str):
        """Read one stored table back as a pandas DataFrame.

        Implemented in O3 (used by the round-trip parity test) and extended in O4 with
        corpus iteration / aggregation and the training-data exporter.
        """
        raise NotImplementedError("read_structure lands in O3 (round-trip parity)")

    def list_structures(self) -> List[str]:
        """List the structure ids (top-level groups) present in the store."""
        if self._h5 is None:
            raise RuntimeError("store is not open; use `with FrustrationStore(path) as s:`")
        h5py = require_h5py()
        return [k for k in self._h5.keys() if isinstance(self._h5[k], h5py.Group)]


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
    "structure_group_path",
    "dataset_path",
    "require_h5py",
    "schema_for_table_key",
    "schema_for_mode",
]
