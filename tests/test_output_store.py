"""O2 skeleton checks for the optional HDF5 output store.

These are fast, h5py-independent tests of the store's design surface: the module
imports with h5py absent, the pure helpers (dtype mapping, path builders) behave, and
the h5py-backed entry points fail with a clear message when the optional dependency is
missing. The writer/reader bodies and the round-trip parity gate are O3.
"""

import importlib.util

import numpy as np
import pytest

from frustrapy.output import (
    FORMAT_VERSION,
    GZIP_LEVEL,
    FrustrationStore,
    compound_dtype_for_schema,
    structure_group_path,
    dataset_path,
    require_h5py,
)
from frustrapy.output import schema as S
from frustrapy.output import store as store_mod


H5PY_PRESENT = importlib.util.find_spec("h5py") is not None


def test_module_imports_without_h5py():
    """The store module and its public symbols import regardless of h5py."""
    assert FrustrationStore is not None
    assert isinstance(FORMAT_VERSION, int)
    assert isinstance(GZIP_LEVEL, int)


def test_constants_are_sane():
    assert FORMAT_VERSION >= 1
    assert 0 <= GZIP_LEVEL <= 9


# --------------------------------------------------------------------------- #
# Pure path builders (no h5py)
# --------------------------------------------------------------------------- #


def test_structure_group_path():
    assert structure_group_path("1crn", "configurational") == "/1crn/configurational"


def test_dataset_path_uses_schema_key():
    assert dataset_path("1crn", "mutational", "contact") == "/1crn/mutational/contact"
    assert (
        dataset_path("1crn", "configurational", "density_5adens")
        == "/1crn/configurational/density_5adens"
    )


def test_dataset_path_rejects_unknown_key():
    with pytest.raises(ValueError):
        dataset_path("1crn", "configurational", "not_a_table")


def test_structure_id_rejects_path_separator():
    with pytest.raises(ValueError):
        structure_group_path("a/b", "configurational")
    with pytest.raises(ValueError):
        structure_group_path("", "configurational")


# --------------------------------------------------------------------------- #
# Compound dtype mapping
# --------------------------------------------------------------------------- #


def test_numeric_only_compound_dtype_needs_no_h5py():
    """SeqIC is int+float only, so its structured dtype builds without h5py."""
    dt = compound_dtype_for_schema(S.SEQIC_TABLE)
    assert dt.names == tuple(S.SEQIC_TABLE.column_names)
    assert dt["Position"] == np.dtype("i8")
    assert dt["Entropy"] == np.dtype("f8")


@pytest.mark.skipif(not H5PY_PRESENT, reason="string columns need h5py vlen dtype")
def test_contact_compound_dtype_preserves_schema_order():
    dt = compound_dtype_for_schema(S.CONTACT_TABLE)
    assert dt.names == tuple(S.CONTACT_TABLE.column_names)
    # numeric and string fields map as designed
    assert dt["Res1"] == np.dtype("i8")
    assert dt["NativeEnergy"] == np.dtype("f8")


# --------------------------------------------------------------------------- #
# Optional-dependency behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(H5PY_PRESENT, reason="checks the h5py-absent error message")
def test_require_h5py_raises_helpful_error_when_absent():
    with pytest.raises(ImportError) as exc:
        require_h5py()
    assert "frustrapy[hdf5]" in str(exc.value)


@pytest.mark.skipif(H5PY_PRESENT, reason="open needs h5py")
def test_open_without_h5py_raises_importerror():
    with pytest.raises(ImportError):
        FrustrationStore("nonexistent.h5", mode="r").open()


def test_construction_does_not_touch_h5py():
    """Constructing a store is cheap and does not import or open anything."""
    store = FrustrationStore("batch.h5", mode="w")
    assert store.path == "batch.h5"
    assert store.file_mode == "w"
    assert store._h5 is None


def test_writer_and_reader_are_stubbed_for_o2():
    """O2 ships the skeleton; the bodies land in O3."""
    store = FrustrationStore("batch.h5", mode="w")
    with pytest.raises(NotImplementedError):
        store.write_structure("1crn", "configurational", "done_dir", seq_dist=12)
    with pytest.raises(NotImplementedError):
        store.read_structure("1crn", "configurational", "contact")


def test_schema_for_table_key_round_trips():
    assert store_mod.schema_for_table_key("contact") is S.CONTACT_TABLE
    assert store_mod.schema_for_table_key("singleresidue") is S.SINGLERESIDUE_TABLE
    with pytest.raises(ValueError):
        store_mod.schema_for_table_key("nope")
