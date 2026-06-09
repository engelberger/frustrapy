"""Torch-free CPU energy kernels for the FrustraPy all-atom (ref2015) backend.

Re-exports the compiled :mod:`frustramol_tmol._kernels` functions. The math is ported
verbatim from the Apache-2.0 tmol-webgpu reference (single-point forward only); no energy
model is reimplemented here. See the package ``NOTICE`` for attribution.

Each ``compute_*`` returns a ``dict`` mapping subterm name to its ``(n_blocks, n_blocks)``
per-residue-pair energy matrix (NumPy float64). ``n_threads`` controls CPU parallelism
(``0`` = all hardware cores, ``1`` = serial); pass the shared-budget inner count
(``cores // outer_procs``) when running under an outer process pool.

Two optional GPU paths share the same kernel math and are off in the default CPU-only
build. When the CUDA forward path is compiled (``has_cuda()`` is ``True``; a maintainer
build on a CUDA host, ``-C cmake.define.FRUSTRAMOL_TMOL_CUDA=ON``), passing ``use_cuda=True``
to a ``compute_*`` runs the GPU kernels instead of the CPU drivers; the result matches the
CPU path to floating-point reduction order. Likewise ``use_metal=True`` routes to the Metal
(Apple GPU) path (single-point forward, float32 on-GPU energy with the block-pair
accumulation kept in double on the host), available only in the Metal build (a Mac with
metal-cpp + ``xcrun metal``); ``has_metal()`` reports whether it is usable. Requesting a GPU
path that was not built raises a clear error; the default CPU path is unchanged.
"""

from ._kernels import (  # noqa: F401
    compute_pair_energies,
    compute_lk_ball,
    compute_hbond,
    has_openmp,
    has_cuda,
    has_metal,
    effective_threads,
)

__all__ = [
    "compute_pair_energies",
    "compute_lk_ball",
    "compute_hbond",
    "has_openmp",
    "has_cuda",
    "has_metal",
    "effective_threads",
]
