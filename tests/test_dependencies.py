"""Packaging guardrails: the install must resolve all runtime deps and import cleanly.

These lock in the install fixes:
  * a stock install resolved ZERO runtime deps (the Poetry backend ignored
    setup.py's install_requires). The PEP 621 / hatchling pyproject must declare the
    full core runtime set.
  * ``tqdm`` was a phantom dependency: imported at ``frustration.py`` and
    ``mutations.py`` but declared nowhere, so ``import frustrapy`` blew up on a clean
    install.

And the optional-extra demotion:
  * The four clustering-only deps (scipy, scikit-learn, python-igraph, leidenalg)
    moved out of the core set into the ``clustering`` optional extra (lazify-before-
    demote). They must be in the extra and NOT in the core set, and ``import
    frustrapy`` must succeed without them.
"""

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# The core runtime set every clean install must provide to `import frustrapy` and run
# the no-graphics analysis path. A later change moved the four
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

# Demoted to the `clustering` optional extra. Imported lazily inside
# detect_dynamic_clusters, so they must NOT be in the core set (else the demotion is
# a no-op and a bare install still drags in the heavy clustering stack).
CLUSTERING_EXTRA = {
    "scipy",
    "scikit-learn",
    "python-igraph",
    "leidenalg",
}

# The default install must stay PyTorch-free: torch is only the all-atom tmol
# VALIDATION ORACLE's runtime (frustrapy.backends.atomic_tmol_engine), imported lazily,
# so it must live in the `torch` extra and NEVER in the core set. A bare install runs
# lammps + the frustrapy-native AWSEM CPU core + the torch-free frustramol-tmol all-atom
# CPU kernels with no torch present.
TORCH_ONLY_EXTRA = {"torch"}


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
    """Every core runtime dependency, including tqdm, is declared."""
    declared = _declared_runtime_deps()
    missing = REQUIRED_RUNTIME - declared
    assert not missing, f"pyproject.toml is missing runtime deps: {sorted(missing)}"


def test_clustering_deps_demoted_to_extra():
    """Clustering deps live in the `clustering` extra, not the core set."""
    extra = _declared_extra("clustering")
    core = _declared_runtime_deps()
    missing = CLUSTERING_EXTRA - extra
    assert not missing, f"clustering extra is missing: {sorted(missing)}"
    leaked = CLUSTERING_EXTRA & core
    assert not leaked, f"clustering deps must not be core (demotion no-op): {sorted(leaked)}"


def test_tqdm_is_declared_not_phantom():
    """tqdm specifically (the former phantom import) must be declared."""
    assert "tqdm" in _declared_runtime_deps()


def test_import_frustrapy_clean():
    """A clean import must succeed with no manual dependency installation."""
    import frustrapy  # noqa: F401


def test_torch_is_not_a_core_dependency():
    """The default install is PyTorch-free: torch must not be in the core set."""
    leaked = TORCH_ONLY_EXTRA & _declared_runtime_deps()
    assert not leaked, f"torch must not be a core dep (default install stays torch-free): {sorted(leaked)}"


def test_torch_oracle_lives_in_torch_extra():
    """The tmol-oracle torch runtime is installable via the `torch` extra only."""
    extra = _declared_extra("torch")
    missing = TORCH_ONLY_EXTRA - extra
    assert not missing, f"`torch` extra is missing: {sorted(missing)}"


def test_accelerator_marker_extras_declared():
    """The optional accelerator tiers are declared so `pip install frustrapy[<tier>]` resolves."""
    with open(PYPROJECT, "rb") as fh:
        extras = tomllib.load(fh)["project"].get("optional-dependencies", {})
    for tier in ("torch", "cuda", "metal", "pyrosetta"):
        assert tier in extras, f"optional accelerator/oracle tier `{tier}` is not declared"


def test_torch_not_in_all_extra():
    """`all` stays light: it must not drag in the heavy oracle-only torch runtime."""
    with open(PYPROJECT, "rb") as fh:
        all_specs = tomllib.load(fh)["project"]["optional-dependencies"].get("all", [])
    assert not any("torch" in spec for spec in all_specs), (
        "`all` must not pull torch (oracle-only); keep it out of the recursive extra"
    )


@pytest.mark.parametrize("module", sorted(["numpy", "pandas", "Bio", "plotly", "tqdm"]))
def test_runtime_modules_importable(module):
    """The runtime stack the live calc path imports is actually importable."""
    __import__(module)
