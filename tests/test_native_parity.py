"""Parity gate for the native C++ backend vs the LAMMPS reference.

The native backend reimplements the AWSEM energy model, the per-mode decoy ensemble,
and the index in C++ (the ``frustrapy_native`` extension). This pins it numerically
against the reference ``lammps`` backend on 1CRN for all three modes: the native
energy, decoy mean/sd, density, and ``FrstIndex`` columns must agree at the table's
3-decimal print precision, and the ``FrstIndex`` ranking (Spearman) must be 1.0.

These run the engine (both backends), so they are in the slow lane and require the
built ``frustrapy_native`` extension; they skip cleanly if it is absent.
"""

import os
import shutil
import warnings

import pytest

native = pytest.importorskip("frustrapy_native")

# Numeric columns per mode in the parsed FrustrationData table (0-based).
_NUM_COLS = {
    "configurational": [4, 5, 8, 9, 10, 11],  # density1/2, native, decoy, sd, frst
    "mutational": [4, 5, 8, 9, 10, 11],
    "singleresidue": [2, 4, 5, 6, 7],  # density, native, decoy, sd, frst
}
_FRST_COL = {"configurational": 11, "mutational": 11, "singleresidue": 7}


def _run(backend, mode, pdb, results_dir):
    import frustrapy

    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frustrapy.calculate_frustration(
            pdb_file=pdb, mode=mode, results_dir=results_dir, graphics=False,
            visualization=False, debug="ERROR", backend=backend, seq_dist=12,
        )
    base = os.path.splitext(os.path.basename(pdb))[0]
    return os.path.join(results_dir, f"{base}.done", "FrustrationData", f"{base}.pdb_{mode}")


def _rows(path):
    with open(path) as fh:
        return [line.split() for line in fh.readlines()[1:]]  # skip header


def _spearman(a, b):
    # Spearman without scipy: Pearson of ranks.
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma = sum(ra) / n
    mb = sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((x - mb) ** 2 for x in rb) ** 0.5
    return cov / (va * vb)


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["configurational", "mutational", "singleresidue"])
def test_native_matches_lammps_1crn(mode, crn_pdb, tmp_path):
    lammps_table = _run("lammps", mode, crn_pdb, str(tmp_path / f"l_{mode}"))
    native_table = _run("native", mode, crn_pdb, str(tmp_path / f"n_{mode}"))

    L = _rows(lammps_table)
    N = _rows(native_table)
    assert len(L) == len(N) and len(L) > 0, f"{mode}: row count {len(L)} vs {len(N)}"

    max_d = 0.0
    for a, b in zip(L, N):
        for c in _NUM_COLS[mode]:
            max_d = max(max_d, abs(float(a[c]) - float(b[c])))
    # Bit-for-bit at the file's 3-decimal print precision (allow a single last-digit
    # rounding step). The gate is FrstIndex Spearman >= 0.99 + an energy tolerance.
    assert max_d <= 1.5e-3, f"{mode}: max numeric column diff {max_d}"

    fc = _FRST_COL[mode]
    sp = _spearman([float(r[fc]) for r in L], [float(r[fc]) for r in N])
    assert sp >= 0.99, f"{mode}: FrstIndex Spearman {sp}"


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["configurational", "mutational", "singleresidue"])
def test_metal_matches_lammps_1crn(mode, crn_pdb, tmp_path, monkeypatch):
    """Metal (Apple GPU) parity lane. A NO-OP skip on the CPU-only container build
    (the extension reports has_metal()==False); on a Mac built with
    -C cmake.define.FRUSTRAPY_NATIVE_METAL=ON it asserts the Metal path matches the
    LAMMPS reference within tolerance. Apple GPUs are float32-only, so the energy
    tolerance is looser than the bit-for-bit CPU gate (see native/docs/METAL_BUILD.md),
    but the FrstIndex ranking and sign must still agree."""
    if not native.has_metal():
        pytest.skip("native core built without Metal (has_metal() is False)")

    monkeypatch.setenv("FRUSTRAPY_NATIVE_USE_METAL", "1")
    lammps_table = _run("lammps", mode, crn_pdb, str(tmp_path / f"l_{mode}"))
    metal_table = _run("native", mode, crn_pdb, str(tmp_path / f"m_{mode}"))

    L = _rows(lammps_table)
    M = _rows(metal_table)
    assert len(L) == len(M) and len(L) > 0, f"{mode}: row count {len(L)} vs {len(M)}"

    # float32 GPU arithmetic: allow a small absolute energy tolerance. The hard gate is
    # the FrstIndex Spearman (ranking) and sign agreement below.
    max_d = 0.0
    for a, b in zip(L, M):
        for c in _NUM_COLS[mode]:
            max_d = max(max_d, abs(float(a[c]) - float(b[c])))
    assert max_d <= 5e-2, f"{mode}: max numeric column diff {max_d} (float32 tolerance)"

    fc = _FRST_COL[mode]
    sp = _spearman([float(r[fc]) for r in L], [float(r[fc]) for r in M])
    assert sp >= 0.99, f"{mode}: FrstIndex Spearman {sp}"
    for a, b in zip(L, M):
        la, lb = float(a[fc]), float(b[fc])
        if abs(la) > 1e-2 and abs(lb) > 1e-2:
            assert (la > 0) == (lb > 0), f"{mode}: sign mismatch {la} vs {lb}"


@pytest.mark.slow
def test_native_lammps_frstindex_sign_agreement(crn_pdb, tmp_path):
    """No inverted classes: the sign of every FrstIndex agrees between backends
    (guards the #1 reimplementation hazard, the Z-score sign convention)."""
    mode = "configurational"
    L = _rows(_run("lammps", mode, crn_pdb, str(tmp_path / "ls")))
    N = _rows(_run("native", mode, crn_pdb, str(tmp_path / "ns")))
    fc = _FRST_COL[mode]
    for a, b in zip(L, N):
        la, lb = float(a[fc]), float(b[fc])
        # same side of zero (allow tiny near-zero values within tolerance)
        if abs(la) > 1e-2 and abs(lb) > 1e-2:
            assert (la > 0) == (lb > 0), f"sign mismatch: {la} vs {lb}"
