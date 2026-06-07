"""Smoke tests for the native core skeleton (N1).

These run only if the ``frustrapy_native`` extension was built (``pip install ./native``).
They exercise the build, the zero-copy binding path, and the two real geometry kernels;
the energy reduction is a stub until N2 and is asserted to raise.
"""

import numpy as np
import pytest

native = pytest.importorskip("frustrapy_native")


def _toy_structure():
    # Four residues on a line 5 A apart, single chain. CB == CA for the toy.
    coords = np.array(
        [[0, 0, 0], [5, 0, 0], [10, 0, 0], [15, 0, 0]], dtype=np.float64
    )
    res_type = np.zeros(4, dtype=np.int32)
    chain_id = np.zeros(4, dtype=np.int32)
    res_seqid = np.array([1, 2, 3, 4], dtype=np.int32)
    return coords, res_type, chain_id, res_seqid


def test_has_cuda_is_bool():
    assert native.has_cuda() in (True, False)


def test_core_version_present():
    assert isinstance(native.__core_version__, str)


def test_contact_map_geometry_and_seq_dist():
    coords, res_type, chain_id, res_seqid = _toy_structure()
    # cutoff 6 A, seq_dist 1: adjacent residues 5 A apart are contacts.
    pairs = native.contact_map(coords, coords, res_type, chain_id, res_seqid, 6.0, 1)
    assert pairs.dtype == np.int32 and pairs.shape[1] == 2
    got = {tuple(p) for p in pairs}
    assert got == {(0, 1), (1, 2), (2, 3)}

    # seq_dist 2 drops the adjacent pairs (separation 1 < 2); 10 A pairs exceed cutoff.
    pairs2 = native.contact_map(coords, coords, res_type, chain_id, res_seqid, 6.0, 2)
    assert pairs2.shape[0] == 0


def test_local_density_runs_and_is_monotone_at_ends():
    coords, res_type, chain_id, res_seqid = _toy_structure()
    rho = native.local_density(coords, coords, res_type, chain_id, res_seqid, 4.5, 6.5)
    assert rho.shape == (4,) and rho.dtype == np.float64
    # Interior residues (1, 2) have neighbours on both sides; ends have one -> higher.
    assert rho[1] > rho[0]
    assert rho[2] > rho[3]


def test_compute_frustration_is_stub_until_n2():
    coords, res_type, chain_id, res_seqid = _toy_structure()
    g20 = np.zeros((20, 20), dtype=np.float64)
    b = np.zeros((20, 3), dtype=np.float64)
    with pytest.raises(RuntimeError, match="not implemented"):
        native.compute_frustration(
            coords, coords, res_type, chain_id, res_seqid,
            g20, g20, g20, b, "configurational", 12, 1000, 1,
        )


def test_compute_frustration_validates_param_shapes():
    coords, res_type, chain_id, res_seqid = _toy_structure()
    bad = np.zeros((19, 20), dtype=np.float64)
    g20 = np.zeros((20, 20), dtype=np.float64)
    b = np.zeros((20, 3), dtype=np.float64)
    with pytest.raises((ValueError, RuntimeError)):
        native.compute_frustration(
            coords, coords, res_type, chain_id, res_seqid,
            bad, g20, g20, b, "configurational", 12, 1000, 1,
        )
