"""Packaging guardrails: the install must resolve all runtime deps and import cleanly.

These lock in the Phase 0 fixes:
  * P0-A -- a stock install resolved ZERO runtime deps (the Poetry backend ignored
    setup.py's install_requires). The PEP 621 / hatchling pyproject must declare the
    full core runtime set.
  * P0-B -- ``tqdm`` was a phantom dependency: imported at ``frustration.py`` and
    ``mutations.py`` but declared nowhere, so ``import frustrapy`` blew up on a clean
    install.

And the Phase 7 demotion:
  * The four clustering-only deps (scipy, scikit-learn, python-igraph, leidenalg)
    moved out of the core set into the ``clustering`` optional extra (lazify-before-
    demote). They must be in the extra and NOT in the core set, and ``import
    frustrapy`` must succeed without them.
"""

import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# The core runtime set every clean install must provide to `import frustrapy` and run
# the no-graphics analysis path. Phase 7 (lazify-before-demote) moved the four
# clustering-only deps OUT of this set into the `clustering` extra below.
REQUIRED_RUNTIME = {
    "numpy",
    "pandas",
    "biopython",
    "plotly",
    "tqdm",
    "rich",
    "matplotlib",
    "seaborn",
    "logomaker",
}

# Phase 7: demoted to the `clustering` optional extra. Imported lazily inside
# detect_dynamic_clusters, so they must NOT be in the core set (else the demotion is
# a no-op and a bare install still drags in the heavy clustering stack).
CLUSTERING_EXTRA = {
    "scipy",
    "scikit-learn",
    "python-igraph",
    "leidenalg",
}


def _strip_to_name(spec):
    name = spec.split(";")[0]
    for sep in ("==", ">=", "<=", "~=", ">", "<", "[", " "):
        name = name.split(sep)[0]
    return name.strip().lower()


def _declared_runtime_deps():
    with open(PYPROJECT, "rb") as fh:
        data = tomllib.load(fh)
    deps = data["project"]["dependencies"]
    return {_strip_to_name(spec) for spec in deps}


def _declared_extra(extra):
    with open(PYPROJECT, "rb") as fh:
        data = tomllib.load(fh)
    deps = data["project"].get("optional-dependencies", {}).get(extra, [])
    return {_strip_to_name(spec) for spec in deps}


def test_pyproject_declares_full_runtime_set():
    """P0-A + P0-B: every core runtime dependency, including tqdm, is declared."""
    declared = _declared_runtime_deps()
    missing = REQUIRED_RUNTIME - declared
    assert not missing, f"pyproject.toml is missing runtime deps: {sorted(missing)}"


def test_clustering_deps_demoted_to_extra():
    """Phase 7: clustering deps live in the `clustering` extra, not the core set."""
    extra = _declared_extra("clustering")
    core = _declared_runtime_deps()
    missing = CLUSTERING_EXTRA - extra
    assert not missing, f"clustering extra is missing: {sorted(missing)}"
    leaked = CLUSTERING_EXTRA & core
    assert not leaked, f"clustering deps must not be core (demotion no-op): {sorted(leaked)}"


def test_tqdm_is_declared_not_phantom():
    """P0-B: tqdm specifically (the former phantom import) must be declared."""
    assert "tqdm" in _declared_runtime_deps()


def test_import_frustrapy_clean():
    """A clean import must succeed with no manual dependency installation."""
    import frustrapy  # noqa: F401


@pytest.mark.parametrize("module", sorted(["numpy", "pandas", "Bio", "plotly", "tqdm"]))
def test_runtime_modules_importable(module):
    """The runtime stack the live calc path imports is actually importable."""
    __import__(module)
