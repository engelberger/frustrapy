from importlib.metadata import PackageNotFoundError, version as _pkg_version

# Pin native-math thread pools (OpenBLAS/MKL/OpenMP) to one thread per process
# BEFORE any numpy-importing submodule loads below — this is the only point that
# is guaranteed to run ahead of the first BLAS import, and BLAS reads these env
# vars only at import. With it in place, a pool of `cores` worker processes can
# never each spin `cores` threads (cores**2). It uses setdefault, so an operator
# who exported their own thread counts keeps them. See utils/concurrency.py and
# the "Parallelism and resource limits" note in the README.
from ._threadlimits import apply_thread_limits as _apply_thread_limits

_apply_thread_limits()

try:
    # Single source of truth for the version: the installed package metadata
    # (pyproject [project].version). Avoids a second hard-coded version string.
    __version__ = _pkg_version("frustrapy")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

# Import core classes
from .core.pdb import Pdb
from .core.dynamic import Dynamic

# Import analysis functions
from .analysis.frustration import (
    calculate_frustration,
    dir_frustration,
    dynamic_frustration,
    get_frustration,
)
from .analysis.mutations import mutate_res, mutate_res_parallel

# NOTE: detect_dynamic_clusters is intentionally NOT imported eagerly here. Its
# module (analysis/clustering.py) pulls in the heavy optional clustering stack
# (scipy, scikit-learn, python-igraph, leidenalg, statsmodels), which is shipped as
# the `clustering` extra. Eagerly importing it would make a bare `pip install
# frustrapy` unable to `import frustrapy` (lazify-before-demote, Phase 7). It is
# resolved on first access via the module __getattr__ below, so the public API and
# `from frustrapy import detect_dynamic_clusters` still work when the extra is
# installed, and raise a clear error otherwise.

# Import visualization functions
from .visualization.plots import (
    plot_contact_map,
    plot_5andens,
    plot_5adens_proportions,
    plot_delta_frus,
)
from .visualization.structure import view_frustration_pymol

# New Benchmark Module
from . import benchmark

# Evolutionary frustration (FrustraEvo). Imported AFTER calculate_frustration is
# bound above: evolution.information_content does `from frustrapy import
# calculate_frustration` at module load, so the name must already exist in this
# partially-initialized module's namespace.
from .evolution import analyze_family

# Define what's available when using "from frustrapy import *"
__all__ = [
    "__version__",
    # Core classes
    "Pdb",
    "Dynamic",
    # Analysis functions
    "calculate_frustration",
    "dir_frustration",
    "dynamic_frustration",
    "get_frustration",
    "mutate_res",
    "mutate_res_parallel",
    "detect_dynamic_clusters",
    # Visualization functions
    "plot_contact_map",
    "plot_5andens",
    "plot_5adens_proportions",
    "plot_delta_frus",
    "view_frustration_pymol",
    "benchmark",
    # Evolution (FrustraEvo)
    "analyze_family",
    # Deep-learning frustration predictor (lazy submodule, mpnn extra)
    "mpnn",
]


def __getattr__(name):
    """Lazily resolve optional, heavy-dependency attributes (PEP 562).

    Keeps `detect_dynamic_clusters` in the public API without importing the
    `clustering` extra at `import frustrapy` time.
    """
    if name == "detect_dynamic_clusters":
        from .analysis.clustering import detect_dynamic_clusters

        return detect_dynamic_clusters
    if name == "mpnn":
        # Deep-learning frustration predictor. Imported on first access so `import frustrapy`
        # never loads it; the submodule itself defers onnxruntime to the mpnn extra. Use
        # import_module rather than `from . import mpnn`: the latter resolves the name through
        # this same __getattr__ (via _handle_fromlist) and recurses forever.
        import importlib

        return importlib.import_module("frustrapy.mpnn")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
