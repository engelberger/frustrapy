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
from .analysis.clustering import detect_dynamic_clusters

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
]
