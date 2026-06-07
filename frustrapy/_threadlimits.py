"""Native-math thread pinning — a dependency-free leaf module.

Kept separate from :mod:`frustrapy.utils.concurrency` (and importing nothing
from the package) so it can run at the very top of ``frustrapy/__init__.py``,
before any numpy/BLAS-importing submodule loads. Importing it cannot trigger the
``utils -> decorators -> analysis`` import cycle, because it imports only the
standard library. ``utils.concurrency`` re-exports these names so callers have a
single import surface for all concurrency helpers.
"""

import os

# BLAS/OpenMP thread-pool environment variables, pinned to one thread per
# process so a pool of `cores` workers cannot each spin `cores` native threads
# (cores**2). BLAS reads these only at its first import, so they must be set
# before numpy/scipy/pandas/PyRosetta load — hence this leaf module is imported
# first in `frustrapy/__init__.py`.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def apply_thread_limits(value: str = "1") -> None:
    """Pin every native-math thread pool to ``value`` threads per process.

    Uses ``setdefault`` so an operator who exported their own thread counts
    keeps control; FrustraPy only fills in the unset ones. Idempotent and
    side-effect-free beyond ``os.environ`` — safe to call at import time and
    again from a pool-worker initializer.
    """
    for var in THREAD_ENV_VARS:
        os.environ.setdefault(var, value)
