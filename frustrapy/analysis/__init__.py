from .frustration import (
    calculate_frustration,
    dir_frustration,
    dynamic_frustration,
    get_frustration,
)

from .mutations import (
    mutate_res,
    mutate_res_parallel,
)

# detect_dynamic_clusters is lazified (clustering extra; lazify-before-demote,
# Phase 7) — resolved on first access via __getattr__ below, never at import time.

__all__ = [
    "calculate_frustration",
    "dir_frustration",
    "dynamic_frustration",
    "get_frustration",
    "mutate_res",
    "mutate_res_parallel",
    "detect_dynamic_clusters",
]


def __getattr__(name):
    if name == "detect_dynamic_clusters":
        from .clustering import detect_dynamic_clusters

        return detect_dynamic_clusters
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
