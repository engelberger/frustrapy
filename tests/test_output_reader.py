"""O4 reader / corpus / training-data tests for the HDF5 output store.

These exercise the multi-prediction reader surface (corpus iteration, cross-structure
aggregation, the ``to_training_dataset`` exporter, and the multi-shard
:class:`FrustrationCorpus`) against the real LAMMPS/AWSEM output of the shared ``crn_*``
fixtures, so they are auto-tagged ``slow`` and need the optional ``h5py`` extra. The
values exported are checked against the text tables — the O3 parity guarantee carried
through into O4.
"""

import importlib.util
import os

import numpy as np
import pytest

from frustrapy.output import (
    FrustrationStore,
    FrustrationCorpus,
    read_table,
)
from frustrapy.output.schema import (
    RESIDUE_FEATURE_COLUMNS,
    SINGLERESIDUE_TABLE,
    schema_for_mode,
)

H5PY_PRESENT = importlib.util.find_spec("h5py") is not None
pytestmark = pytest.mark.skipif(not H5PY_PRESENT, reason="HDF5 store needs the h5py extra")


def _two_structure_store(job_dir, base, mode, h5_path):
    """Write the same crn run twice under two structure ids to make a small corpus."""
    with FrustrationStore(h5_path, mode="w") as store:
        store.write_structure(f"{base}_a", mode, job_dir, seq_dist=12)
        store.write_structure(f"{base}_b", mode, job_dir, seq_dist=12)
    return h5_path


def test_listing_modes_and_tables(crn_singleresidue, tmp_path):
    h5_path = _two_structure_store(
        crn_singleresidue["job_dir"], "1crn", "singleresidue", str(tmp_path / "list.h5")
    )
    with FrustrationStore(h5_path) as store:
        assert sorted(store.list_structures()) == ["1crn_a", "1crn_b"]
        assert store.list_modes("1crn_a") == ["singleresidue"]
        tables = store.list_tables("1crn_a", "singleresidue")
        assert "singleresidue" in tables


def test_iter_tables_filters(crn_singleresidue, tmp_path):
    h5_path = _two_structure_store(
        crn_singleresidue["job_dir"], "1crn", "singleresidue", str(tmp_path / "iter.h5")
    )
    with FrustrationStore(h5_path) as store:
        keyed = list(store.iter_tables(mode="singleresidue", key="singleresidue"))
        ids = sorted(row[0] for row in keyed)
        assert ids == ["1crn_a", "1crn_b"]
        # filtering to a mode that is not present yields nothing
        assert list(store.iter_tables(mode="configurational")) == []


def test_read_corpus_concatenates_across_structures(crn_singleresidue, tmp_path):
    job_dir = crn_singleresidue["job_dir"]
    h5_path = _two_structure_store(job_dir, "1crn", "singleresidue", str(tmp_path / "corpus.h5"))

    text = read_table(
        os.path.join(job_dir, "FrustrationData", "1crn.pdb_singleresidue"),
        SINGLERESIDUE_TABLE,
    )
    with FrustrationStore(h5_path) as store:
        corpus = store.read_corpus("singleresidue", "singleresidue")

    assert list(corpus.columns)[0] == "StructureId"
    assert len(corpus) == 2 * len(text)
    assert corpus["StructureId"].tolist() == ["1crn_a"] * len(text) + ["1crn_b"] * len(text)
    # the per-structure block equals the text table value-for-value on FrstIndex
    block_a = corpus[corpus["StructureId"] == "1crn_a"]
    assert block_a["FrstIndex"].tolist() == [float(v) for v in text["FrstIndex"].tolist()]


def test_to_training_dataset_residue(crn_singleresidue, tmp_path):
    job_dir = crn_singleresidue["job_dir"]
    h5_path = _two_structure_store(job_dir, "1crn", "singleresidue", str(tmp_path / "train.h5"))

    text = read_table(
        os.path.join(job_dir, "FrustrationData", "1crn.pdb_singleresidue"),
        SINGLERESIDUE_TABLE,
    )
    with FrustrationStore(h5_path) as store:
        td = store.to_training_dataset(level="residue")

    assert td.feature_names == list(RESIDUE_FEATURE_COLUMNS)
    # 46 residues for 1CRN, twice over (two structure ids)
    assert td.n_samples == 2 * len(text)
    assert td.X.shape == (2 * len(text), len(RESIDUE_FEATURE_COLUMNS))
    assert list(td.ids.columns) == ["StructureId", "Res", "ChainRes", "AA"]
    # feature values are exact float64 from the text table
    fi = td.feature_names.index("FrstIndex")
    expected = [float(v) for v in text["FrstIndex"].tolist()]
    assert td.X[: len(text), fi].tolist() == expected
    # save/reload as npz preserves the arrays
    npz_path = str(tmp_path / "td.npz")
    td.save_npz(npz_path)
    loaded = np.load(npz_path, allow_pickle=False)
    assert loaded["X"].shape == td.X.shape


def test_to_training_dataset_contact(crn_configurational, tmp_path):
    job_dir = crn_configurational["job_dir"]
    h5_path = str(tmp_path / "contact.h5")
    with FrustrationStore(h5_path, mode="w") as store:
        store.write_structure("1crn", "configurational", job_dir, seq_dist=12)

    text = read_table(
        os.path.join(job_dir, "FrustrationData", "1crn.pdb_configurational"),
        schema_for_mode("configurational"),
    )
    with FrustrationStore(h5_path) as store:
        td = store.to_training_dataset(level="contact", mode="configurational")

    assert td.n_samples == len(text)
    fi = td.feature_names.index("FrstIndex")
    assert td.X[:, fi].tolist() == [float(v) for v in text["FrstIndex"].tolist()]


def test_frustration_corpus_reads_multiple_shards(crn_singleresidue, tmp_path):
    """Two shard files read together as one logical corpus."""
    job_dir = crn_singleresidue["job_dir"]
    shard0 = str(tmp_path / "batch.part-0.h5")
    shard1 = str(tmp_path / "batch.part-1.h5")
    with FrustrationStore(shard0, mode="w") as store:
        store.write_structure("struct0", "singleresidue", job_dir, seq_dist=12)
    with FrustrationStore(shard1, mode="w") as store:
        store.write_structure("struct1", "singleresidue", job_dir, seq_dist=12)

    text = read_table(
        os.path.join(job_dir, "FrustrationData", "1crn.pdb_singleresidue"),
        SINGLERESIDUE_TABLE,
    )
    with FrustrationCorpus([shard0, shard1]) as corpus:
        assert corpus.list_structures() == ["struct0", "struct1"]
        # a structure resolves to the shard that holds it
        df1 = corpus.read_structure("struct1", "singleresidue", "singleresidue")
        assert len(df1) == len(text)
        full = corpus.read_corpus("singleresidue", "singleresidue")
        assert len(full) == 2 * len(text)
        td = corpus.to_training_dataset(level="residue")
        assert td.n_samples == 2 * len(text)
