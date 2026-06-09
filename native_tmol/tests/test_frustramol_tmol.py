"""Tests for the torch-free CPU energy kernels (frustramol_tmol._kernels).

These run only if the ``frustramol_tmol`` extension was built (``pip install ./native_tmol``).
The synthetic tests need no external data; the parity test is gated on the tmol-webgpu oracle
fixtures (Rosetta-derived params folded in, so NOT shipped here) and is skipped when they are
absent. The authoritative torch-free parity gate is the standalone C++ harness
(``tests/parity_main.cpp``); this mirrors it through the Python binding.
"""

import json
import os

import numpy as np
import pytest

fk = pytest.importorskip("frustramol_tmol")

# Default location of the tmol-webgpu oracle fixtures in the maintainer container; override
# with the FRUSTRAMOL_TMOL_FIXTURES env var.
_FIXTURE_DIR = os.environ.get("FRUSTRAMOL_TMOL_FIXTURES", "/workspace/tmol-webgpu/test/fixtures")


def test_has_openmp_is_bool():
    assert fk.has_openmp() in (True, False)


def test_has_metal_is_bool():
    # False on the CPU-only / non-Apple build; True only in a Metal build with a device.
    assert fk.has_metal() in (True, False)


def test_use_metal_without_build_raises():
    # On a CPU-only build, use_metal=True must raise a clear error, never silently run CPU.
    if fk.has_metal():
        pytest.skip("Metal build present; the raise-path test only applies CPU-only")
    with pytest.raises(Exception):
        fk.compute_pair_energies(use_metal=True, **_two_atoms(3.2))


def test_effective_threads_contract():
    # 0 -> all cores (>= 1); a positive request is honored verbatim; without OpenMP it is 1.
    assert fk.effective_threads(0) >= 1
    if fk.has_openmp():
        assert fk.effective_threads(1) == 1
        assert fk.effective_threads(4) == 4
    else:
        assert fk.effective_threads(4) == 1


def _two_atoms(distance):
    coords = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=np.float64)
    block = np.array([0, 1], dtype=np.int32)
    charge = np.array([0.5, -0.5], dtype=np.float64)
    is_heavy = np.array([True, True])
    # plausible carbon-like ljlk params (values are not asserted against an oracle here)
    ljr = np.array([2.0, 2.0]); ljw = np.array([0.1, 0.1])
    lkd = np.array([1.0, 1.0]); lkl = np.array([3.5, 3.5]); lkv = np.array([16.0, 16.0])
    f = np.array([False, False])
    pi = np.array([0], dtype=np.int32); pj = np.array([1], dtype=np.int32)
    sep = np.array([5], dtype=np.int32)
    ljg = np.array([3.0, 2.6, 1.75], dtype=np.float64)
    eg = np.array([78.0, 1.0, 0.36, 1.45, 5.5], dtype=np.float64)
    return dict(coords=coords, block=block, charge=charge, is_heavy=is_heavy,
                lj_radius=ljr, lj_wdepth=ljw, lk_dgfree=lkd, lk_lambda=lkl, lk_volume=lkv,
                is_donor=f, is_hydroxyl=f, is_polarh=f, is_acceptor=f,
                pair_i=pi, pair_j=pj, sep_ljlk=sep, sep_elec=sep,
                ljlk_global=ljg, elec_global=eg, n_blocks=2)


def test_far_pair_is_zero():
    # Beyond the 6 A ljlk cutoff and 5.5 A elec max_dis, every subterm is zero.
    res = fk.compute_pair_energies(n_threads=0, **_two_atoms(12.0))
    assert set(res.keys()) == {"fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec"}
    for name, m in res.items():
        assert m.shape == (2, 2)
        assert np.all(m == 0.0), name


def test_near_pair_nonzero_and_thread_stable():
    args = _two_atoms(3.2)
    res0 = fk.compute_pair_energies(n_threads=0, **args)
    res1 = fk.compute_pair_energies(n_threads=1, **args)
    # something is scored at contact distance
    assert any(np.any(res0[t] != 0.0) for t in res0)
    # single vs multi thread agree to floating-point reduction order
    for t in res0:
        assert np.max(np.abs(res0[t] - res1[t])) < 1e-12, t


def _load_fixture(path):
    fx = json.load(open(path))
    atoms = fx["atoms"]

    def col(key, dt=np.float64):
        return np.array([a[key] for a in atoms], dtype=dt)

    coords = np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=np.float64)
    npr = fx["neighbor_pairs"]
    pack = dict(
        coords=coords, block=col("block", np.int32), charge=col("charge"),
        is_heavy=col("is_heavy", bool), lj_radius=col("lj_radius"), lj_wdepth=col("lj_wdepth"),
        lk_dgfree=col("lk_dgfree"), lk_lambda=col("lk_lambda"), lk_volume=col("lk_volume"),
        is_donor=col("is_donor", bool), is_hydroxyl=col("is_hydroxyl", bool),
        is_polarh=col("is_polarh", bool), is_acceptor=col("is_acceptor", bool),
        pair_i=np.array([p["i"] for p in npr], np.int32),
        pair_j=np.array([p["j"] for p in npr], np.int32),
        sep_ljlk=np.array([p["sep_ljlk"] for p in npr], np.int32),
        sep_elec=np.array([p["sep_elec"] for p in npr], np.int32),
        ljlk_global=np.array(fx["ljlk_global"][0], np.float64),
        elec_global=np.array(fx["elec_global"][0], np.float64),
        n_blocks=fx["n_blocks"],
    )
    return fx, pack


@pytest.mark.parametrize("name", ["1ubq", "small_1lu6"])
def test_ljlk_parity_vs_tmol_oracle(name):
    path = os.path.join(_FIXTURE_DIR, f"{name}.json")
    if not os.path.exists(path):
        pytest.skip(f"tmol-webgpu oracle fixture not present: {path}")
    fx, pack = _load_fixture(path)
    res = fk.compute_pair_energies(n_threads=0, **pack)
    ref_bp = fx["reference"]["block_pair"]
    ref_wp = fx["reference"]["whole_pose"]
    wp_true = fx["reference"].get("whole_pose_true", {})
    for t in ("fa_ljatr", "fa_ljrep", "fa_lk"):
        got = np.asarray(res[t])
        ref = np.array(ref_bp[t])
        assert np.max(np.abs(got - ref)) <= 1e-3, f"{name} {t} per-pair"
        assert abs(got.sum() - ref_wp[t]) <= 1e-3 + 1e-4 * abs(ref_wp[t]), f"{name} {t} whole-pose"
    if "fa_elec" in wp_true:
        got_elec = float(np.asarray(res["fa_elec"]).sum())
        assert abs(got_elec - wp_true["fa_elec"]) <= 1e-3 + 1e-4 * abs(wp_true["fa_elec"])


# --- optional CUDA forward path (M6) -------------------------------------------------
# has_cuda() is always callable; the parity tests run only when the module was built with
# CUDA (a maintainer GPU host). The authoritative GPU gate is the standalone CUDA harness
# (tests/cuda_parity_main.cu), which diffs the CUDA path against the CPU driver AND the
# tmol oracle on a fixture; these mirror it through the Python binding.

def test_has_cuda_is_bool():
    assert fk.has_cuda() in (True, False)


def test_use_cuda_on_cpu_build_raises():
    # use_cuda=True must be a clear rebuild error on a CPU-only build, never a silent
    # CPU fall-back that would hide that the GPU path was not exercised.
    if fk.has_cuda():
        pytest.skip("built with CUDA; the CPU-build guard does not apply")
    with pytest.raises(Exception):
        fk.compute_pair_energies(use_cuda=True, n_threads=0, **_two_atoms(3.2))


@pytest.mark.skipif(not fk.has_cuda(), reason="frustramol_tmol built without CUDA")
def test_cuda_matches_cpu_pair_energies():
    args = _two_atoms(3.2)
    cpu = fk.compute_pair_energies(use_cuda=False, n_threads=0, **args)
    gpu = fk.compute_pair_energies(use_cuda=True, **args)
    assert set(cpu.keys()) == set(gpu.keys())
    for t in cpu:
        # atomicAdd reduction order differs from the CPU sum only in the last ULPs.
        assert np.max(np.abs(np.asarray(cpu[t]) - np.asarray(gpu[t]))) <= 1e-9, t


@pytest.mark.skipif(not fk.has_cuda(), reason="frustramol_tmol built without CUDA")
@pytest.mark.parametrize("name", ["1ubq", "small_1lu6"])
def test_cuda_ljlk_parity_vs_tmol_oracle(name):
    path = os.path.join(_FIXTURE_DIR, f"{name}.json")
    if not os.path.exists(path):
        pytest.skip(f"tmol-webgpu oracle fixture not present: {path}")
    _fx, pack = _load_fixture(path)
    cpu = fk.compute_pair_energies(use_cuda=False, n_threads=0, **pack)
    gpu = fk.compute_pair_energies(use_cuda=True, **pack)
    for t in ("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec"):
        assert np.max(np.abs(np.asarray(cpu[t]) - np.asarray(gpu[t]))) <= 1e-3, t


# ---------------------------------------------------------------------------
# Metal (Apple GPU) parity. These run ONLY on a Metal build with a device present
# (has_metal() True); in the CPU-only container they skip. They check that the GPU path
# reproduces the CPU path AND the tmol oracle within the G2 tolerance (1e-3): the float32
# on-GPU energy with host-side double block-pair accumulation must not drift past it.
# ---------------------------------------------------------------------------
_METAL_TOL = 1e-3


def _max_dict_diff(a, b):
    return max(float(np.max(np.abs(np.asarray(a[t]) - np.asarray(b[t])))) for t in a)


def _load_lk_ball_pack(fx, max_water=4):
    atoms = fx["atoms"]
    n = len(atoms)
    waters = np.zeros((n, max_water, 3), dtype=np.float64)
    present = np.zeros((n, max_water), dtype=bool)
    for k, a in enumerate(atoms):
        ws = a.get("waters")
        if isinstance(ws, list):
            for w, wp in enumerate(ws[:max_water]):
                if isinstance(wp, list):
                    waters[k, w] = wp
                    present[k, w] = True

    def col(key, dt=np.float64):
        return np.array([a[key] for a in atoms], dtype=dt)

    npr = fx["neighbor_pairs"]
    return dict(
        coords=np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=np.float64),
        block=col("block", np.int32), charge=col("charge"), is_heavy=col("is_heavy", bool),
        lj_radius=col("lj_radius"), lj_wdepth=col("lj_wdepth"), lk_dgfree=col("lk_dgfree"),
        lk_lambda=col("lk_lambda"), lk_volume=col("lk_volume"),
        is_donor=col("is_donor", bool), is_hydroxyl=col("is_hydroxyl", bool),
        is_polarh=col("is_polarh", bool), is_acceptor=col("is_acceptor", bool),
        waters=waters, water_present=present,
        pair_i=np.array([p["i"] for p in npr], np.int32),
        pair_j=np.array([p["j"] for p in npr], np.int32),
        sep_ljlk=np.array([p["sep_ljlk"] for p in npr], np.int32),
        sep_elec=np.array([p["sep_elec"] for p in npr], np.int32),
        ljlk_global=np.array(fx["ljlk_global"][0], np.float64),
        lk_ball_global=np.array(fx["lk_ball_global"], np.float64),
        n_blocks=fx["n_blocks"],
    )


def _load_hbond_pack(fx):
    hps = fx.get("hbond_pairs", [])

    def poly(key, field):
        return np.array([hp[key][field] for hp in hps], dtype=np.float64)

    def vec(key):
        return np.array([hp[key] for hp in hps], dtype=np.float64)

    # The H/A coordinates live on the synthetic atoms the binding builds; the fixture stores
    # them directly on each hbond pair via its h/a atom indices.
    atoms = fx["atoms"]
    H = np.array([[atoms[hp["h"]]["x"], atoms[hp["h"]]["y"], atoms[hp["h"]]["z"]] for hp in hps],
                 dtype=np.float64).reshape(-1, 3)
    A = np.array([[atoms[hp["a"]]["x"], atoms[hp["a"]]["y"], atoms[hp["a"]]["z"]] for hp in hps],
                 dtype=np.float64).reshape(-1, 3)
    return dict(
        hp_H=H, hp_A=A, hp_D=vec("D").reshape(-1, 3), hp_B=vec("B").reshape(-1, 3),
        hp_B0=vec("B0").reshape(-1, 3),
        hp_block_h=np.array([atoms[hp["h"]]["block"] for hp in hps], np.int32),
        hp_block_a=np.array([atoms[hp["a"]]["block"] for hp in hps], np.int32),
        hp_hyb=np.array([hp["hyb"] for hp in hps], np.int32),
        hp_ad_weight=np.array([hp["ad_weight"] for hp in hps], np.float64),
        hp_sep=np.array([hp["sep"] for hp in hps], np.int32),
        ahdist_coeffs=poly("AHdist", "coeffs"), ahdist_range=poly("AHdist", "range"),
        ahdist_bound=poly("AHdist", "bound"),
        cosbah_coeffs=poly("cosBAH", "coeffs"), cosbah_range=poly("cosBAH", "range"),
        cosbah_bound=poly("cosBAH", "bound"),
        cosahd_coeffs=poly("cosAHD", "coeffs"), cosahd_range=poly("cosAHD", "range"),
        cosahd_bound=poly("cosAHD", "bound"),
        hbond_global=np.array(fx["hbond_global"], np.float64), n_blocks=fx["n_blocks"],
    )


def _require_metal(name):
    if not fk.has_metal():
        pytest.skip("Metal build / device not present (maintainer runs this on a host Mac)")
    path = os.path.join(_FIXTURE_DIR, f"{name}.json")
    if not os.path.exists(path):
        pytest.skip(f"tmol-webgpu oracle fixture not present: {path}")
    return json.load(open(path)), path


@pytest.mark.parametrize("name", ["1ubq", "small_1lu6"])
def test_metal_ljlk_elec_matches_cpu_and_oracle(name):
    fx, path = _require_metal(name)
    _, pack = _load_fixture(path)
    cpu = fk.compute_pair_energies(n_threads=0, **pack)
    gpu = fk.compute_pair_energies(use_metal=True, **pack)
    assert _max_dict_diff(cpu, gpu) <= _METAL_TOL, f"{name} metal-vs-cpu ljlk/elec"
    ref_bp = fx["reference"]["block_pair"]
    for t in ("fa_ljatr", "fa_ljrep", "fa_lk"):
        assert np.max(np.abs(np.asarray(gpu[t]) - np.array(ref_bp[t]))) <= _METAL_TOL, \
            f"{name} metal-vs-oracle {t}"


@pytest.mark.parametrize("name", ["1ubq", "small_1lu6"])
def test_metal_lk_ball_matches_cpu(name):
    fx, _ = _require_metal(name)
    if "lk_ball_global" not in fx:
        pytest.skip(f"fixture {name} has no lk_ball_global")
    pack = _load_lk_ball_pack(fx)
    cpu = fk.compute_lk_ball(n_threads=0, **pack)
    gpu = fk.compute_lk_ball(use_metal=True, **pack)
    assert _max_dict_diff(cpu, gpu) <= _METAL_TOL, f"{name} metal-vs-cpu lk_ball"


@pytest.mark.parametrize("name", ["1ubq", "small_1lu6"])
def test_metal_hbond_matches_cpu(name):
    fx, _ = _require_metal(name)
    if not fx.get("hbond_pairs"):
        pytest.skip(f"fixture {name} has no hbond_pairs")
    pack = _load_hbond_pack(fx)
    cpu = fk.compute_hbond(n_threads=0, **pack)
    gpu = fk.compute_hbond(use_metal=True, **pack)
    assert _max_dict_diff(cpu, gpu) <= _METAL_TOL, f"{name} metal-vs-cpu hbond"
