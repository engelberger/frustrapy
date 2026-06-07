"""Packaging guardrails: the install must resolve all runtime deps and import cleanly.

These lock in the Phase 0 fixes:
  * P0-A -- a stock install resolved ZERO runtime deps (the Poetry backend ignored
    setup.py's install_requires). The PEP 621 / hatchling pyproject must declare the
    full 9-package runtime set.
  * P0-B -- ``tqdm`` was a phantom dependency: imported at ``frustration.py`` and
    ``mutations.py`` but declared nowhere, so ``import frustrapy`` blew up on a clean
    install.
"""

import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# The 9-package runtime set every clean install must provide (CLAUDE.md S7).
REQUIRED_RUNTIME = {
    "numpy",
    "pandas",
    "biopython",
    "plotly",
    "scipy",
    "scikit-learn",
    "python-igraph",
    "leidenalg",
    "tqdm",
}


def _declared_runtime_deps():
    with open(PYPROJECT, "rb") as fh:
        data = tomllib.load(fh)
    deps = data["project"]["dependencies"]
    # Strip version specifiers / extras to the bare distribution name.
    names = set()
    for spec in deps:
        name = spec.split(";")[0]
        for sep in ("==", ">=", "<=", "~=", ">", "<", "[", " "):
            name = name.split(sep)[0]
        names.add(name.strip().lower())
    return names


def test_pyproject_declares_full_runtime_set():
    """P0-A + P0-B: every runtime dependency, including tqdm, is declared."""
    declared = _declared_runtime_deps()
    missing = REQUIRED_RUNTIME - declared
    assert not missing, f"pyproject.toml is missing runtime deps: {sorted(missing)}"


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
