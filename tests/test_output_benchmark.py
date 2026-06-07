"""Tests for the O5 HPC benchmark (text files vs one HDF5 store).

The fast tests cover the pure filesystem helpers (no h5py, no LAMMPS). The slow test
runs the real benchmark on a replicated 1CRN ``done`` directory and asserts the store
saves inodes and round-trips value-identical (the benchmark raises otherwise, so it only
ever returns parity-passing numbers).
"""

import importlib.util
import os

import pytest

from frustrapy.output import count_inodes, tree_size_bytes

H5PY_PRESENT = importlib.util.find_spec("h5py") is not None


def test_count_inodes_counts_dirs_and_files(tmp_path):
    # one root + one subdir + two files = 4 entries
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub" / "b.txt").write_text("bb")
    assert count_inodes(str(tmp_path)) == 4


def test_count_inodes_single_file(tmp_path):
    f = tmp_path / "only.h5"
    f.write_text("x")
    assert count_inodes(str(f)) == 1


def test_tree_size_bytes_sums_files(tmp_path):
    (tmp_path / "a.txt").write_text("abc")  # 3 bytes
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("de")  # 2 bytes
    assert tree_size_bytes(str(tmp_path)) == 5


def test_tree_size_bytes_single_file(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"01234")
    assert tree_size_bytes(str(f)) == 5


@pytest.mark.skipif(not H5PY_PRESENT, reason="HDF5 store needs the h5py extra")
def test_benchmark_text_vs_hdf5_saves_inodes_and_parity(crn_configurational, tmp_path):
    """A small replicated batch: the store is fewer inodes and value-identical."""
    from frustrapy.output import benchmark_text_vs_hdf5

    n = 8
    result = benchmark_text_vs_hdf5(
        source_done_dir=crn_configurational["job_dir"],
        mode="configurational",
        n_structures=n,
        workdir=str(tmp_path / "bench"),
        seq_dist=12,
    )
    assert result.parity_ok is True
    assert result.n_structures == n
    # One HDF5 file is a single inode; n text dirs cost many more.
    assert result.hdf5_inodes == 1
    assert result.text_inodes > result.hdf5_inodes
    assert result.inode_ratio > 1.0
    # By default the generated text/store are cleaned up.
    assert not os.path.exists(str(tmp_path / "bench" / "store.h5"))
    # to_frame has the two layout rows.
    frame = result.to_frame()
    assert list(frame["layout"]) == ["text", "hdf5"]
    assert "fewer" in result.summary()


@pytest.mark.skipif(not H5PY_PRESENT, reason="HDF5 store needs the h5py extra")
def test_benchmark_keep_leaves_artifacts(crn_configurational, tmp_path):
    from frustrapy.output import benchmark_text_vs_hdf5

    workdir = str(tmp_path / "kept")
    benchmark_text_vs_hdf5(
        source_done_dir=crn_configurational["job_dir"],
        mode="configurational",
        n_structures=2,
        workdir=workdir,
        keep=True,
    )
    assert os.path.exists(os.path.join(workdir, "store.h5"))
    assert os.path.isdir(os.path.join(workdir, "text"))
