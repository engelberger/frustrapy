"""Smoke tests for the public SDK facade (``frustrapy.sdk``).

These do not run the LAMMPS engine; they check that the facade imports, exposes the
documented public surface, and re-exports the same callables as the top-level
``frustrapy`` package (the facade must not shadow or diverge from the existing API).
"""

import frustrapy
from frustrapy import sdk


# The names the facade promises to expose. Kept in sync with sdk.__all__.
EXPECTED_NAMES = {
    "calculate_frustration",
    "get_frustration",
    "dir_frustration",
    "dynamic_frustration",
    "mutate_res_parallel",
    "mutate_res_scan_parallel",
    "pyrosetta_available",
    "analyze_family",
    "Pdb",
    "Dynamic",
    "FrustrationDensityResults",
}


def test_all_names_present():
    """Every documented name is importable from the facade."""
    for name in EXPECTED_NAMES:
        assert hasattr(sdk, name), f"frustrapy.sdk is missing {name!r}"
    assert EXPECTED_NAMES.issubset(set(sdk.__all__))


def test_facade_matches_toplevel():
    """Names shared with the top-level package are the SAME object, not a copy.

    This is what makes the facade a non-breaking organization of the existing API
    rather than a parallel reimplementation.
    """
    for name in [
        "calculate_frustration",
        "get_frustration",
        "dir_frustration",
        "dynamic_frustration",
        "mutate_res_parallel",
        "analyze_family",
        "Pdb",
        "Dynamic",
    ]:
        assert getattr(sdk, name) is getattr(frustrapy, name), name


def test_version_mirrored():
    """The facade exposes the same version string as the package."""
    assert sdk.__version__ == frustrapy.__version__


def test_callables_are_callable():
    """The re-exported entry points are callable and the types are classes."""
    for name in [
        "calculate_frustration",
        "get_frustration",
        "dir_frustration",
        "dynamic_frustration",
        "mutate_res_parallel",
        "mutate_res_scan_parallel",
        "pyrosetta_available",
        "analyze_family",
    ]:
        assert callable(getattr(sdk, name)), name
    for name in ["Pdb", "Dynamic", "FrustrationDensityResults"]:
        assert isinstance(getattr(sdk, name), type), name
