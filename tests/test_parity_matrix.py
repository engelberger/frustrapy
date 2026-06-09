"""Tests for the cross-backend parity + cost harness (``native/bench/parity_matrix.py``).

Fast tests pin the load-bearing logic with no engine run: the energy-function taxonomy,
the parity-vs-correlation classification (the G1 fork), availability probing, the cited
cells, and the report renderer. They guarantee the harness can NEVER print "parity" for
two different energy functions.

The slow test runs the real AWSEM backends (lammps + native) on 1CRN and asserts the
measured cell is exact AWSEM parity (Spearman 1.0, byte-identical energies). It needs the
built ``frustrapy_native`` extension and skips cleanly without it.
"""

import os
import sys

import pytest

# Make ``native/bench`` importable as ``bench`` without installing it.
_NATIVE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "native")
if _NATIVE not in sys.path:
    sys.path.insert(0, _NATIVE)

from bench.parity_matrix import (  # noqa: E402
    BackendStatus,
    CitedCell,
    ParityCell,
    RunRecord,
    VarianceRecord,
    MatrixResult,
    ENERGY_FUNCTION,
    agreement_kind,
    cited_cells,
    probe_backends,
    render_report,
    run_parity_matrix,
)


# --------------------------------------------------------------------------- #
# The G1 fork: same energy function -> parity, different -> correlation.
# --------------------------------------------------------------------------- #
def test_same_energy_function_is_parity():
    assert agreement_kind("lammps", "native") == "parity"          # AWSEM vs AWSEM
    assert agreement_kind("atomic", "atomic") == "parity"          # REF2015 vs REF2015
    assert agreement_kind("atomic-tmol", "native_tmol-cpu") == "parity"  # beta vs beta


def test_different_energy_function_is_correlation():
    # AWSEM vs REF2015, AWSEM vs beta_nov2016, REF2015 vs beta_nov2016 are all
    # cross-method correlation, never parity.
    assert agreement_kind("lammps", "atomic") == "cross-method correlation"
    assert agreement_kind("native", "atomic-tmol") == "cross-method correlation"
    assert agreement_kind("atomic", "atomic-tmol") == "cross-method correlation"
    assert agreement_kind("atomic", "native_tmol-cpu") == "cross-method correlation"


def test_beta_is_not_ref2015_parity():
    """The honesty rule: a beta_nov2016-vs-REF2015 agreement must not be labeled parity."""
    assert ENERGY_FUNCTION["atomic"] == "REF2015"
    assert ENERGY_FUNCTION["atomic-tmol"] == "beta_nov2016"
    assert ENERGY_FUNCTION["native_tmol-cpu"] == "beta_nov2016"
    assert agreement_kind("atomic", "atomic-tmol") != "parity"


def test_unknown_backend_raises_not_defaults_to_parity():
    with pytest.raises(KeyError):
        agreement_kind("lammps", "totally-new-backend")


# --------------------------------------------------------------------------- #
# Availability probing.
# --------------------------------------------------------------------------- #
def test_probe_lists_all_backends_with_a_reason_when_skipped():
    statuses = probe_backends()
    names = {s.name for s in statuses}
    # Every known backend is reported (no silent drop).
    assert {"lammps", "native", "atomic", "atomic-tmol", "native_tmol-cpu"} <= names
    # lammps always runs; every non-runnable backend carries a concrete reason.
    by = {s.name: s for s in statuses}
    assert by["lammps"].runnable
    for s in statuses:
        if not s.runnable:
            assert s.skip_reason, f"{s.name} skipped without a reason"


def test_atomic_paths_are_gated_in_container():
    by = {s.name: s for s in probe_backends()}
    # No PyRosetta / torch in the container -> these do not run end-to-end here.
    assert not by["atomic"].runnable
    assert not by["atomic-tmol"].runnable


# --------------------------------------------------------------------------- #
# Cited cells: cross-method numbers come labeled as correlation, with a source.
# --------------------------------------------------------------------------- #
def test_cited_cells_label_beta_vs_ref2015_as_correlation():
    cells = {(c.backend_a, c.backend_b): c for c in cited_cells()}
    tmol_vs_ros = cells[("atomic", "atomic-tmol")]
    assert tmol_vs_ros.kind == "cross-method correlation"
    assert tmol_vs_ros.source  # always carries a provenance
    assert abs(tmol_vs_ros.spearman - 0.917) < 1e-9


def test_cited_cells_carry_source_and_status():
    for c in cited_cells():
        assert c.source, f"{c} missing source"
        assert c.status in {"cited", "untested"}
        if c.status == "untested":
            assert c.spearman is None


# --------------------------------------------------------------------------- #
# Report renderer: structure + the honesty guarantees.
# --------------------------------------------------------------------------- #
def _synthetic_result() -> MatrixResult:
    statuses = [
        BackendStatus("lammps", "AWSEM", True),
        BackendStatus("native", "AWSEM", True),
        BackendStatus("atomic", "REF2015", False, "PyRosetta not installed"),
        BackendStatus("atomic-tmol", "beta_nov2016", False, "torch not installed"),
        BackendStatus("native_tmol-cpu", "beta_nov2016", False, "full run not wired"),
    ]
    runs = [
        RunRecord("lammps", "1crn", "configurational", 1, 0.80, 232,
                  [0.1] * 232, [-1.0] * 232, [-2.0] * 232, ["neutral"] * 232),
        RunRecord("native", "1crn", "configurational", 14, 0.60, 232,
                  [0.1] * 232, [-1.0] * 232, [-2.0] * 232, ["neutral"] * 232),
    ]
    cells = [ParityCell("1crn", "configurational", "lammps", "native", "parity",
                        1.0, 1.0, 0.0, 0.0, 0.0, 232)]
    variance = [VarianceRecord("native (AWSEM)", "1crn", "configurational", 3,
                               0.0, 0.0, "deterministic")]
    return MatrixResult(["1crn"], ["configurational"], statuses, runs, cells,
                        cited_cells(), variance, 14)


def test_report_never_labels_cross_method_as_parity():
    report = render_report(_synthetic_result())
    # The cross-method tmol-vs-REF2015 line is present and labeled correlation.
    assert "cross-method correlation" in report
    # It must never claim "atomic vs atomic-tmol | parity".
    assert "atomic vs atomic-tmol | parity " not in report


def test_report_has_all_sections_and_discloses_skips():
    report = render_report(_synthetic_result())
    for section in ("Backend availability", "Measured agreement", "Gated agreement",
                    "Cost", "Repack / decoy variance", "Skips"):
        assert section in report, f"missing section: {section}"
    # Every skipped backend appears in the disclosure with its reason.
    assert "PyRosetta not installed" in report
    assert "torch not installed" in report


# --------------------------------------------------------------------------- #
# Slow: the real AWSEM run is exact parity.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_awsem_measured_parity_is_exact_1crn(crn_pdb, tmp_path):
    pytest.importorskip("frustrapy_native")
    result = run_parity_matrix(
        structures={"1crn": str(crn_pdb)},
        modes=("configurational",),
        seq_dist=12, repeats=1, results_root=str(tmp_path / "pm"),
        variance_repeats=2,
    )
    awsem_cells = [c for c in result.cells if c.kind == "parity"]
    assert awsem_cells, "lammps-vs-native AWSEM cell should be measured"
    for c in awsem_cells:
        assert c.spearman >= 0.99, f"AWSEM Spearman {c.spearman}"
        assert c.class_agreement == 1.0, f"AWSEM class agreement {c.class_agreement}"
        # AWSEM lammps vs native is byte-identical on 1CRN.
        assert c.max_dnative <= 1.5e-3
        assert c.max_ddecoy <= 1.5e-3
    # The decoy ensemble is deterministic: zero re-run spread.
    assert result.variance, "variance should be measured for the AWSEM path"
    for v in result.variance:
        assert v.max_ddecoy == 0.0, f"AWSEM decoy spread {v.max_ddecoy} (expected 0)"

    # The rendered report is well-formed and honest.
    report = render_report(result)
    assert "cross-method correlation" in report  # cited tmol-vs-REF2015 row
    assert "## Skips" in report
