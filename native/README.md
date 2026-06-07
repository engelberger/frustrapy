# frustrapy-native

Native C++ (and optional CUDA) compute core for FrustraPy frustration backends. It
reimplements the AWSEM energy reductions that the reference `lammps` backend gets from
the precompiled binary, behind the `FrustrationBackend` interface
(`frustrapy/backends/`). Parity-gated against the `lammps` reference.

This is a self-contained subproject so the main `frustrapy` package keeps its pure-Python
build. The compiled module installs as the top-level `frustrapy_native`.

Status: N1 (build + binding surface + geometry kernels). The energy reductions
(`compute_frustration`) land in N2; the CUDA kernels in N3. See
`../docs/NATIVE_BACKEND_DESIGN.md`.

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
