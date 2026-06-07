"""O3 round-trip parity gate for the HDF5 output store.

The HDF5 store must round-trip value-identical to the default text tables: a table
written into the store and read back equals the on-disk text table exactly — numeric
columns by exact ``float64``/``int64`` equality, string columns byte-identical. These
tests run the real LAMMPS/AWSEM toolchain (via the shared ``crn_*`` and
``globin_family`` fixtures, so they are auto-tagged ``slow``) and need the optional
``h5py`` extra.
"""

import importlib.util
import os

import pytest

from frustrapy.output import FrustrationStore, read_table
from frustrapy.output.schema import (
    DENSITY_5ADENS_TABLE,
    IC_CONFIGURATIONAL_TABLE,
    IC_SINGLERES_TABLE,
    SEQIC_TABLE,
    schema_for_mode,
)

H5PY_PRESENT = importlib.util.find_spec("h5py") is not None
pytestmark = pytest.mark.skipif(not H5PY_PRESENT, reason="HDF5 store needs the h5py extra")


def _assert_value_identical(text_df, got_df, schema):
    """The store DataFrame equals the text table value-for-value, per schema dtype."""
    assert list(got_df.columns) == schema.column_names
    assert len(got_df) == len(text_df)
    for col in schema.columns:
        text_vals = text_df[col.name].tolist()
        got_vals = got_df[col.name].tolist()
        if col.dtype == "str":
            assert got_vals == [str(v) for v in text_vals], f"{schema.key}.{col.name}"
        elif col.dtype == "int":
            assert got_vals == [int(float(v)) for v in text_vals], f"{schema.key}.{col.name}"
        else:  # exact float64 equality: the text was printed from a double
            assert got_vals == [float(v) for v in text_vals], f"{schema.key}.{col.name}"


def _roundtrip_mode(job_dir, base, mode, h5_path):
    """Write base/mode into a store, read each table back, assert value-identical."""
    data_dir = os.path.join(job_dir, "FrustrationData")
    with FrustrationStore(h5_path, mode="w") as store:
        store.write_structure(base, mode, job_dir, seq_dist=12)

    schema = schema_for_mode(mode)
    text_main = read_table(
        os.path.join(data_dir, schema.filename(base=base, mode=mode)), schema
    )
    with FrustrationStore(h5_path) as store:
        got_main = store.read_structure(base, mode, schema.key)
    _assert_value_identical(text_main, got_main, schema)

    dens_path = os.path.join(data_dir, DENSITY_5ADENS_TABLE.filename(base=base, mode=mode))
    if os.path.exists(dens_path):
        text_dens = read_table(dens_path, DENSITY_5ADENS_TABLE)
        with FrustrationStore(h5_path) as store:
            got_dens = store.read_structure(base, mode, DENSITY_5ADENS_TABLE.key)
        _assert_value_identical(text_dens, got_dens, DENSITY_5ADENS_TABLE)


def test_roundtrip_configurational(crn_configurational, tmp_path):
    _roundtrip_mode(crn_configurational["job_dir"], "1crn", "configurational", str(tmp_path / "c.h5"))


def test_roundtrip_mutational(crn_mutational, tmp_path):
    _roundtrip_mode(crn_mutational["job_dir"], "1crn", "mutational", str(tmp_path / "m.h5"))


def test_roundtrip_singleresidue(crn_singleresidue, tmp_path):
    _roundtrip_mode(crn_singleresidue["job_dir"], "1crn", "singleresidue", str(tmp_path / "s.h5"))


def test_store_attrs_and_listing(crn_configurational, tmp_path):
    """Per-structure / per-mode attrs are written and the structure is listed."""
    h5_path = str(tmp_path / "attrs.h5")
    with FrustrationStore(h5_path, mode="w") as store:
        store.write_structure(
            "1crn", "configurational", crn_configurational["job_dir"], seq_dist=12,
            source_pdb_sha256="deadbeef", timestamp="2026-06-07T00:00:00Z",
            frustrapy_version="9.9.9",
        )
    import h5py

    with h5py.File(h5_path) as h:
        assert h.attrs["format_version"] >= 1
        assert h.attrs["frustrapy_version"] == "9.9.9"
        assert h["1crn"].attrs["pdb_base"] == "1crn"
        assert h["1crn"].attrs["source_pdb_sha256"] == "deadbeef"
        grp = h["1crn/configurational"]
        assert grp.attrs["mode"] == "configurational"
        assert int(grp.attrs["seq_dist"]) == 12
        assert int(grp.attrs["n_residues"]) == 46
    with FrustrationStore(h5_path) as store:
        assert store.list_structures() == ["1crn"]


def test_roundtrip_frustraevo_ic_tables(globin_family, tmp_path):
    """FrustraEvo IC_Configurational / IC_SingleRes / SeqIC round-trip value-identical."""
    rd = globin_family["results_dir"]
    ref = "1fsx-A"
    h5_path = str(tmp_path / "family.h5")
    with FrustrationStore(h5_path, mode="w") as store:
        written = store.write_family("globin3", ref, rd)
    assert "ic_configurational" in written

    for schema in (IC_CONFIGURATIONAL_TABLE, IC_SINGLERES_TABLE, SEQIC_TABLE):
        text_path = os.path.join(rd, schema.filename(reference=ref))
        if not os.path.exists(text_path):
            continue
        text_df = read_table(text_path, schema)
        with FrustrationStore(h5_path) as store:
            got_df = store.read_family("globin3", ref, schema.key)
        _assert_value_identical(text_df, got_df, schema)
