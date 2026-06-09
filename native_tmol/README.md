# native_tmol

Torch-free C++ CPU energy kernels for the FrustraPy all-atom (ref2015) backend.

This component computes per-residue-pair single-point energies for the in-scope ref2015
pairwise terms (`fa_ljatr`, `fa_ljrep`, `fa_lk`, `fa_elec`, `lk_ball_iso`, `lk_ball`,
`lk_bridge`, `lk_bridge_uncpl`, `hbond`) without PyTorch. The scalar math is ported verbatim
from the Apache-2.0 tmol-webgpu reference (itself a verbatim port of `uw-ipd/tmol`); no energy
model is reimplemented. Single-point forward only (no autograd). See `NOTICE` for attribution.

Like the AWSEM `native/` core, this is CPU-only by default (plain C++17 + optional OpenMP). An
optional CUDA forward path (`src/cuda/kernels.cu`) shares the same kernel math and is built only
on a CUDA host (see "CUDA forward path" below); the Metal lane is still out of scope here.

## Layout

- `include/frustramol_tmol/*.hpp` - the ported scalar kernels (one header per term, plus the
  geometry/interpolation helpers) and the CPU drivers that loop the neighbor / hbond lists and
  accumulate the per-residue-pair (block-pair) matrices.
- `src/cuda/kernels.cu` - the optional CUDA forward kernels (`ljlk`, `fa_elec`, `lk_ball`,
  `hbond`), mirroring the CPU kernel headers term-for-term; single-point forward only, no torch.
  Compiled only when `FRUSTRAMOL_TMOL_CUDA=ON` and nvcc is present.
- `src/bindings.cpp` - an optional nanobind Python module (`frustramol_tmol._kernels`), built
  when nanobind and a Python development module are present. Each `compute_*` takes a `use_cuda`
  flag (default `False`); `has_cuda()` reports whether the GPU path was compiled.
- `tests/parity_main.cpp` + `tests/mini_json.hpp` - a standalone, dependency-free CPU parity
  harness that validates the kernels against a tmol oracle fixture.
- `tests/cuda_parity_main.cu` - the standalone CUDA parity harness (maintainer GPU host): diffs
  the CUDA path against the CPU driver AND the tmol oracle. Built only with CUDA on.

## Parallelism and the shared core budget

The only concurrency is OpenMP worker threads, bounded by the explicit `n_threads` argument the
caller passes (`0` = all hardware cores, `1` = serial); the kernels never read the machine core
count internally. There is no process fork. A caller running structures concurrently (an outer
process pool, e.g. `dir_frustration(n_procs=K)`) passes the shared-budget inner thread count
(`cores // K`), so `outer * inner <= cores` and the nested process+thread fan-out can never
oversubscribe, matching the discipline in `frustrapy/utils/concurrency.py` and the AWSEM
`native/` core. Single-thread and multi-thread results agree to floating-point reduction order
(observed `< 4e-15` on the validation fixtures), not a formula difference.

## Build and validate (in-container, no torch)

The standalone harness is the torch-free validation path:

```
g++ -std=c++17 -O2 -fopenmp -Iinclude -Itests tests/parity_main.cpp -o tmol_parity_harness
./tmol_parity_harness <fixture.json> [more.json ...]
```

or via CMake (`cmake -S native_tmol -B build && cmake --build build`), which always builds the
harness and additionally builds the nanobind module when nanobind is found.

The harness loads a tmol-webgpu fixture (the per-structure energy oracle exported from tmol)
and asserts each subterm reproduces tmol's per-residue-pair block-pair matrices and whole-pose
totals to the tmol gate tolerance (`atol 1e-3`, `rtol 1e-4`). The fixtures carry Rosetta-derived
parameter values and are not shipped with this package; the maintainer supplies them.

## Python module (optional)

```
pip install ./native_tmol
```

builds `frustramol_tmol._kernels` (CPU-only). It exposes `compute_pair_energies`,
`compute_lk_ball`, `compute_hbond` (each returning a dict of `(n_blocks, n_blocks)` NumPy
matrices), plus `has_openmp()`, `has_cuda()`, and `effective_threads(n_threads)`. The maintainer
builds and runs this on a host; it is not required for the standalone validation above.

## CUDA forward path (optional, maintainer GPU build)

The CUDA path (`src/cuda/kernels.cu`) reproduces the four terms on the GPU (single-point
forward only, no torch). The device math mirrors the CPU kernel headers term-for-term; the only
numerical difference from the CPU result is the floating-point order of the atomicAdd reduction
(last ULPs), never a formula difference. It is OFF by default, so the CPU lane above is the
in-container path; nvcc is not present in the dev container, so the CUDA path is built and
validated by the maintainer on a CUDA host (Colab / cluster), exactly like the AWSEM `native/`
CUDA lane.

```
# A CUDA toolkit (nvcc) and a GPU must be present.
nvcc --version

# Standalone CUDA parity harness (CPU vs CUDA vs tmol oracle):
cmake -S native_tmol -B build -DFRUSTRAMOL_TMOL_CUDA=ON
cmake --build build
./build/tmol_cuda_parity_harness <fixture.json> [more.json ...]

# Or the Python module with the GPU path compiled in:
pip install ./native_tmol -C cmake.define.FRUSTRAMOL_TMOL_CUDA=ON
python -c "import frustramol_tmol as fk; print('cuda:', fk.has_cuda())"   # -> cuda: True
```

When nvcc is absent the build falls back to CPU-only and `has_cuda()` returns `False` (graceful,
never a hard failure). With the GPU path compiled, pass `use_cuda=True` to any `compute_*` to run
the kernels on the GPU; `use_cuda=True` on a CPU-only build raises a clear rebuild error rather
than silently falling back. The CUDA harness asserts the GPU result agrees with the CPU driver
(the parity reference) and with the tmol oracle to the gate tolerance; record the numbers it
prints. See `docs/tmol/M6_NATIVE_CUDA.md` for the full validation procedure.
