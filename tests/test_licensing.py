"""Tests for the one-time PyRosetta non-commercial license notice (TMOL2-M3).

PyRosetta is licensed for non-commercial use only. FrustraPy must emit a clear,
one-time notice whenever a PyRosetta-backed path actually runs, and must stay
silent for the license-clean lammps/native/tmol and threading/modeller paths.

PyRosetta is NOT installed in the container, so these tests exercise:
  * the error path (PyRosetta absent) -- an actionable ImportError, and crucially
    NO license notice (you cannot be "using" a component that failed to load);
  * the notice path (PyRosetta mocked importable) -- the notice fires exactly once
    per process, at the _ensure_pyrosetta boundary, through logging (not print);
  * suppression via FRUSTRAPY_SUPPRESS_PYROSETTA_NOTICE;
  * that a bare pyrosetta_available() probe never emits the notice.
"""

import logging
import sys
import types

import pytest

from frustrapy import _licensing
from frustrapy.analysis import mutation_backends as mb


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Reset the once-per-process guards and the init flag around each test."""
    _licensing._reset_notice_for_tests()
    monkeypatch.setattr(mb, "_PYROSETTA_INITED", False, raising=False)
    monkeypatch.delenv(_licensing.SUPPRESS_ENV_VAR, raising=False)
    yield
    _licensing._reset_notice_for_tests()


def _install_fake_pyrosetta(monkeypatch):
    """Put a minimal importable ``pyrosetta`` stub in sys.modules.

    Only ``init`` is needed by _ensure_pyrosetta; it is a no-op recorder so the
    init boundary runs end-to-end without the real, license-gated package.
    """
    fake = types.ModuleType("pyrosetta")
    calls = []
    fake.init = lambda *a, **k: calls.append((a, k))
    monkeypatch.setitem(sys.modules, "pyrosetta", fake)
    return fake, calls


# --------------------------------------------------------------------------- #
# The helper in isolation
# --------------------------------------------------------------------------- #


def test_notice_emitted_once(caplog):
    with caplog.at_level(logging.WARNING, logger="frustrapy"):
        _licensing.warn_pyrosetta_noncommercial()
        _licensing.warn_pyrosetta_noncommercial()
    notices = [r for r in caplog.records if "NON-COMMERCIAL" in r.getMessage()]
    assert len(notices) == 1
    msg = notices[0].getMessage()
    assert "PyRosetta" in msg and "RosettaCommons" in msg
    assert notices[0].levelno == logging.WARNING


def test_notice_suppressed_by_env(monkeypatch, caplog):
    monkeypatch.setenv(_licensing.SUPPRESS_ENV_VAR, "1")
    with caplog.at_level(logging.WARNING, logger="frustrapy"):
        _licensing.warn_pyrosetta_noncommercial()
    assert not [r for r in caplog.records if "NON-COMMERCIAL" in r.getMessage()]


def test_notice_force_reemits(caplog):
    with caplog.at_level(logging.WARNING, logger="frustrapy"):
        _licensing.warn_pyrosetta_noncommercial()
        _licensing.warn_pyrosetta_noncommercial(force=True)
    assert len([r for r in caplog.records if "NON-COMMERCIAL" in r.getMessage()]) == 2


# --------------------------------------------------------------------------- #
# Wired at the _ensure_pyrosetta boundary
# --------------------------------------------------------------------------- #


def test_ensure_pyrosetta_emits_notice_when_present(monkeypatch, caplog):
    fake, calls = _install_fake_pyrosetta(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="frustrapy"):
        got = mb._ensure_pyrosetta()
        # Second call in the same process must not re-emit.
        mb._ensure_pyrosetta()
    assert got is fake
    assert len(calls) == 1  # init runs once per process
    notices = [r for r in caplog.records if "NON-COMMERCIAL" in r.getMessage()]
    assert len(notices) == 1


def test_ensure_pyrosetta_missing_raises_and_is_silent(monkeypatch, caplog):
    # Force the import to fail regardless of the host.
    monkeypatch.setitem(sys.modules, "pyrosetta", None)
    with caplog.at_level(logging.WARNING, logger="frustrapy"):
        with pytest.raises(ImportError, match="pyrosetta-installer|RosettaCommons"):
            mb._ensure_pyrosetta()
    # A path that could not load PyRosetta must not claim it is using PyRosetta.
    assert not [r for r in caplog.records if "NON-COMMERCIAL" in r.getMessage()]


def test_availability_probe_does_not_emit(caplog):
    with caplog.at_level(logging.WARNING, logger="frustrapy"):
        mb.pyrosetta_available()
    assert not [r for r in caplog.records if "NON-COMMERCIAL" in r.getMessage()]
