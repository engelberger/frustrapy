# frustrapy-native

Native C++ (and optional CUDA) compute core for FrustraPy frustration backends. It
reimplements the AWSEM energy reductions that the reference `lammps` backend gets from
the precompiled binary, behind the `FrustrationBackend` interface
(`frustrapy/backends/`). Parity-gated against the `lammps` reference.

This is a self-contained subproject so the main `frustrapy` package keeps its pure-Python
build. The compiled module installs as the top-level `frustrapy_native`.

Status: CPU reference core (`compute_frustration`, all three modes, parity-gated against
`lammps`), parallelized across all cores with OpenMP, plus optional CUDA kernels. See
`../docs/NATIVE_BACKEND_DESIGN.md` and the profile/scaling numbers in `docs/PROFILE.md`.

## CPU parallelism

The energy reductions and the density kernel are parallelized with OpenMP over the
independent work units (per residue / per contact). The result is bit-identical for any
thread count: the decoy random draws are materialized serially in the reference order
(preserving the exact glibc stream), then the per-unit reductions run in parallel into
disjoint output slots. `compute_frustration(..., n_threads=N)` controls the thread count
(0 = all cores, 1 = serial); the `native` backend resolves it automatically so it
composes safely with FrustraPy's outer process pools (one inner thread inside a pool
worker, all cores in the main process). OpenMP is optional in the build
(`find_package(OpenMP)`); without it the core compiles serial. `has_openmp()` and
`effective_threads(n)` report the build's capability and the resolved thread count.

## Build

CPU (default):

```
pip install ./native
# or, from inside native/: pip install .
```

CUDA path (a machine with nvcc; the kernels are placeholders until N3):

```
pip install ./native -C cmake.define.FRUSTRAPY_NATIVE_CUDA=ON
```

ASan/UBSan build for the N2 parity gate:

```
pip install ./native -C cmake.define.FRUSTRAPY_NATIVE_SANITIZE=ON -C cmake.build-type=Debug
```

## Requirements

CMake >= 3.15, a C++20 compiler (C++23 used where available), nanobind >= 2.0,
scikit-build-core >= 0.10, NumPy. For the CUDA path, the CUDA Toolkit (nvcc); validated
by the maintainer on Colab / cluster GPUs.

## Test

```
pip install ./native
python -m pytest native/tests
```

The smoke tests `importorskip` the extension, so they are inert until it is built.

## Benchmark

`native/bench/cross_backend.py` runs one protein through every backend available on the
machine (lammps, native CPU serial and multicore, native CUDA if built) and reports wall
time, speedup, and FrstIndex Spearman vs the lammps reference. It degrades gracefully
(absent backends are skipped). As a CLI:

```
python native/bench/cross_backend.py --pdb tests/data/1crn.pdb --mode mutational \
    --threads 1,2,4,8 --repeats 3 --core-scaling
```

or from Python via `run_cross_backend_benchmark` / `benchmark_core_scaling`. The pytest
form (`tests/test_cross_backend_benchmark.py`) asserts parity and that multicore is not
slower than serial on 1CRN. Measured scaling and profiling live in `docs/PROFILE.md`.
