"""O2 skeleton checks for the optional HDF5 output store.

These are fast, h5py-independent tests of the store's design surface: the module
imports with h5py absent, the pure helpers (dtype mapping, path builders) behave, and
the h5py-backed entry points fail with a clear message when the optional dependency is
missing. The writer/reader bodies and the round-trip parity gate are O3.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from frustrapy.output import (
    FORMAT_VERSION,
    GZIP_LEVEL,
    FrustrationStore,
    FrustrationCorpus,
    TrainingDataset,
    build_training_dataset,
    read_corpus_from,
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


def test_writer_and_reader_require_open_store():
    """The h5py-backed methods refuse to run on an unopened store with a clear error."""
    store = FrustrationStore("batch.h5", mode="w")
    with pytest.raises(RuntimeError):
        store.write_structure("1crn", "configurational", "done_dir", seq_dist=12)
    with pytest.raises(RuntimeError):
        store.read_structure("1crn", "configurational", "contact")


def test_schema_for_table_key_round_trips():
    assert store_mod.schema_for_table_key("contact") is S.CONTACT_TABLE
    assert store_mod.schema_for_table_key("singleresidue") is S.SINGLERESIDUE_TABLE
    with pytest.raises(ValueError):
        store_mod.schema_for_table_key("nope")


# --------------------------------------------------------------------------- #
# Corpus aggregation + training-data export (O4) — pure, no h5py needed
# --------------------------------------------------------------------------- #


class _FakeReader:
    """Minimal reader exposing iter_tables, for the pure-logic O4 tests."""

    def __init__(self, tables):
        # tables: list of (structure_id, mode, key, DataFrame)
        self._tables = tables

    def iter_tables(self, mode=None, key=None):
        for sid, m, k, df in self._tables:
            if mode is not None and m != mode:
                continue
            if key is not None and k != key:
                continue
            yield sid, m, k, df


def _singleres_df(res_start=1):
    return pd.DataFrame(
        {
            "Res": [res_start, res_start + 1],
            "ChainRes": ["A", "A"],
            "DensityRes": [1.0, 2.0],
            "AA": ["M", "K"],
            "NativeEnergy": [-1.5, -2.5],
            "DecoyEnergy": [0.0, 0.1],
            "SDEnergy": [1.0, 1.0],
            "FrstIndex": [0.5, -1.2],
        }
    )


def test_read_corpus_from_prepends_structure_id_and_concatenates():
    reader = _FakeReader(
        [
            ("a", "singleresidue", "singleresidue", _singleres_df(1)),
            ("b", "singleresidue", "singleresidue", _singleres_df(10)),
        ]
    )
    df = read_corpus_from(reader, "singleresidue", "singleresidue")
    assert list(df.columns)[0] == "StructureId"
    assert len(df) == 4
    assert df["StructureId"].tolist() == ["a", "a", "b", "b"]


def test_read_corpus_from_empty_returns_typed_frame():
    df = read_corpus_from(_FakeReader([]), "singleresidue", "singleresidue")
    assert len(df) == 0
    assert list(df.columns) == ["StructureId", *S.SINGLERESIDUE_TABLE.column_names]


def test_read_corpus_from_rejects_unknown_key():
    with pytest.raises(ValueError):
        read_corpus_from(_FakeReader([]), "singleresidue", "not_a_table")


def test_build_training_dataset_residue_level():
    reader = _FakeReader(
        [
            ("a", "singleresidue", "singleresidue", _singleres_df(1)),
            ("b", "singleresidue", "singleresidue", _singleres_df(10)),
        ]
    )
    td = build_training_dataset(reader, level="residue")
    assert isinstance(td, TrainingDataset)
    assert td.n_samples == 4
    assert td.feature_names == list(S.RESIDUE_FEATURE_COLUMNS)
    assert td.n_features == len(S.RESIDUE_FEATURE_COLUMNS)
    assert list(td.ids.columns) == ["StructureId", "Res", "ChainRes", "AA"]
    # feature values are exact float64 from the source table
    fi = td.feature_names.index("FrstIndex")
    assert td.X[:, fi].tolist() == [0.5, -1.2, 0.5, -1.2]
    # to_frame puts ids first, features after
    frame = td.to_frame()
    assert list(frame.columns) == ["StructureId", "Res", "ChainRes", "AA", *td.feature_names]


def test_build_training_dataset_custom_features():
    reader = _FakeReader([("a", "singleresidue", "singleresidue", _singleres_df(1))])
    td = build_training_dataset(reader, level="residue", feature_columns=["FrstIndex"])
    assert td.feature_names == ["FrstIndex"]
    assert td.X.shape == (2, 1)


def test_build_training_dataset_empty_corpus():
    td = build_training_dataset(_FakeReader([]), level="residue")
    assert td.n_samples == 0
    assert td.n_features == len(S.RESIDUE_FEATURE_COLUMNS)


def test_build_training_dataset_rejects_bad_level():
    with pytest.raises(ValueError):
        build_training_dataset(_FakeReader([]), level="atom")


def test_build_training_dataset_rejects_mode_level_mismatch():
    with pytest.raises(ValueError):
        build_training_dataset(_FakeReader([]), level="residue", mode="configurational")


def test_build_training_dataset_rejects_nonnumeric_feature():
    with pytest.raises(ValueError):
        build_training_dataset(_FakeReader([]), level="residue", feature_columns=["AA"])


def test_build_training_dataset_rejects_unknown_feature():
    with pytest.raises(ValueError):
        build_training_dataset(_FakeReader([]), level="residue", feature_columns=["Nope"])


def test_training_dataset_save_npz_roundtrips(tmp_path):
    reader = _FakeReader([("a", "singleresidue", "singleresidue", _singleres_df(1))])
    td = build_training_dataset(reader, level="residue")
    path = str(tmp_path / "train.npz")
    td.save_npz(path)
    loaded = np.load(path, allow_pickle=False)
    assert loaded["X"].shape == td.X.shape
    assert loaded["feature_names"].tolist() == td.feature_names
    assert str(loaded["level"]) == "residue"


def test_contact_level_default_mode_is_configurational():
    spec = store_mod._LEVEL_SPECS["contact"]
    assert spec.default_mode == "configurational"
    assert set(spec.modes) == {"configurational", "mutational"}


def test_corpus_construction_does_not_open():
    corpus = FrustrationCorpus(["a.h5", "b.h5"])
    assert corpus.paths == ["a.h5", "b.h5"]
    with pytest.raises(RuntimeError):
        corpus.list_structures()
