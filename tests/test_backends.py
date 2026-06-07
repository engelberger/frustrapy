"""Backend interface and selection wiring.

These pin the pluggable-backend seam introduced in the G1.0 refactor: the energy
model is selectable behind ``FrustrationBackend``, the default is ``lammps``, and the
selection threads from the public API into the engine. The numerical reference path
(``LammpsBackend`` wrapping the AWSEM/LAMMPS subprocess) is exercised byte-for-byte by
the 1CRN anchor in ``test_anchor.py``; here we test the wiring without running LAMMPS.
"""

import pytest

from frustrapy.backends import (
    DEFAULT_BACKEND,
    FrustrationBackend,
    LammpsBackend,
    available_backends,
    get_backend,
    register_backend,
)
from frustrapy.analysis.frustration_calculator import FrustrationCalculator


def test_default_backend_is_lammps():
    assert DEFAULT_BACKEND == "lammps"
    assert LammpsBackend.name == "lammps"
    assert "lammps" in available_backends()


def test_get_backend_resolves_default_and_name():
    assert isinstance(get_backend(), LammpsBackend)
    assert isinstance(get_backend(None), LammpsBackend)
    assert isinstance(get_backend("lammps"), LammpsBackend)


def test_get_backend_passthrough_instance():
    inst = LammpsBackend()
    assert get_backend(inst) is inst


def test_get_backend_unknown_name_raises():
    with pytest.raises(ValueError):
        get_backend("does-not-exist")


def test_get_backend_bad_type_raises():
    with pytest.raises(TypeError):
        get_backend(123)


def test_calculator_default_backend_is_lammps():
    """The engine resolves to LammpsBackend by default (selection wired into the
    calculator), without running any calculation."""
    calc = FrustrationCalculator(pdb_file="x.pdb", mode="configurational")
    assert isinstance(calc.backend, LammpsBackend)


def test_calculator_accepts_backend_selector():
    calc = FrustrationCalculator(pdb_file="x.pdb", mode="configurational", backend="lammps")
    assert isinstance(calc.backend, LammpsBackend)
    inst = LammpsBackend()
    calc2 = FrustrationCalculator(pdb_file="x.pdb", mode="configurational", backend=inst)
    assert calc2.backend is inst


def test_register_backend_roundtrip():
    class _StubBackend(FrustrationBackend):
        name = "stub-test-backend"

        def compute_energies(self, calculator, pdb):  # pragma: no cover - not run
            raise NotImplementedError

    try:
        register_backend(_StubBackend)
        assert "stub-test-backend" in available_backends()
        assert isinstance(get_backend("stub-test-backend"), _StubBackend)
    finally:
        from frustrapy.backends import _REGISTRY

        _REGISTRY.pop("stub-test-backend", None)


def test_register_backend_rejects_non_backend():
    with pytest.raises(TypeError):
        register_backend(object)


@pytest.mark.slow
def test_explicit_lammps_backend_matches_default(crn_pdb, tmp_path):
    """Explicitly selecting backend='lammps' yields the byte-identical table the
    default path produces (the default IS lammps)."""
    import warnings

    import frustrapy

    def _run(backend, sub):
        out = str(tmp_path / sub)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frustrapy.calculate_frustration(
                pdb_file=crn_pdb,
                mode="configurational",
                results_dir=out,
                graphics=False,
                visualization=False,
                debug="ERROR",
                backend=backend,
            )
        table = (
            f"{out}/1crn.done/FrustrationData/1crn.pdb_configurational"
        )
        with open(table, "rb") as fh:
            return fh.read()

    assert _run(None, "default") == _run("lammps", "explicit")
