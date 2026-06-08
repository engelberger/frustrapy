"""Native frustration core (compiled extension).

A self-contained nanobind/CMake extension built by the ``native/`` subproject. It
reimplements the AWSEM energy reductions that the reference ``lammps`` backend gets from
the precompiled binary, on CPU now and (optionally) CUDA later. N1 ships the build, the
binding surface, and one real geometry kernel; the energy reductions land in N2.

Build with ``pip install ./native`` (CPU). For the CUDA path:
``pip install ./native -C cmake.define.FRUSTRAPY_NATIVE_CUDA=ON`` on a machine with nvcc.
For the Metal (Apple GPU) path:
``pip install ./native -C cmake.define.FRUSTRAPY_NATIVE_METAL=ON`` on a Mac (see
``native/docs/METAL_BUILD.md``).

See ``docs/NATIVE_BACKEND_DESIGN.md``.
"""

from __future__ import annotations

from ._core import (  # type: ignore[import-not-found]
    __core_version__,
    compute_frustration,
    contact_map,
    has_cuda,
    has_metal,
    local_density,
)

__all__ = [
    "__core_version__",
    "compute_frustration",
    "contact_map",
    "has_cuda",
    "has_metal",
    "local_density",
]
