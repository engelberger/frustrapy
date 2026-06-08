"""Numerical-parity gates for the tmol atomic backend (TMOL-PARITY, #39).

Three layers, each skipping cleanly when its prerequisite is absent:

* **in-container, no license** -- the backend-vs-golden post-processor parity (scope
  item 2). Needs only the shipped reference ``ResResE`` logs, NO tmol, so it runs in
  the fast lane. Asserts the shared post-processor reproduces the AA golden FrstIndex.

* **in-container, tmol** -- the evaluator-vs-oracle parity (scope item 1) and the
  tmol-vs-frozen-Rosetta energy cross-check (scope item 3, in-container half). Skipped
  unless tmol is importable (``TMOL_USE_JIT=1`` + ninja; see PY_BACKEND_NOTES.md). The
  cross-check scores 50 poses, so it is marked ``slow``.

* **maintainer, PyRosetta** -- the live-PyRosetta cross-check (scope item 3, full).
  ``skipif`` no PyRosetta; asserts the live diff stays within the documented tolerance.
  PyRosetta is not installed in the container, so this skips here and the maintainer
  runs it on a licensed host.

The reproducible scripts these import from (``frustrapy.backends.tmol_eval``) are the
same ones that write the committed reports under ``docs/tmol/parity/``; the tests assert
the gate thresholds those reports record.
"""

import os

import pytest

from frustrapy.backends import atomic_engine as ae
from frustrapy.backends import atomic_tmol_engine as E
from frustrapy.backends.tmol_eval import validate_in_container as vic
from frustrapy.backends.tmol_eval import validate_vs_pyrosetta as vvp

_HAVE_TMOL = E.tmol_available()
_HAVE_PYROSETTA = ae.pyrosetta_available()
_HAVE_REF = os.path.isdir(vic.DEFAULT_REF_DIR) and os.path.exists(
    os.path.join(vic.DEFAULT_REF_DIR, "native.log")
)

needs_tmol = pytest.mark.skipif(
    not _HAVE_TMOL, reason="optional tmol not importable (see docs/tmol/PY_BACKEND_NOTES.md)"
)
needs_ref = pytest.mark.skipif(
    not _HAVE_REF, reason=f"atomic reference logs absent under {vic.DEFAULT_REF_DIR}"
)
needs_pyrosetta = pytest.mark.skipif(
    not _HAVE_PYROSETTA, reason="PyRosetta not installed (license-gated; maintainer host)"
)


# ---------------------------------------------------------------------------
# In-container, no license: backend post-processor vs the AA golden (scope item 2).
# ---------------------------------------------------------------------------

@needs_ref
def test_backend_postprocessor_matches_golden():
    """The path the atomic-tmol backend reuses (EngineResult -> write_tertiary_frustration)
    reproduces the AA golden FrstIndex bit-for-bit at print precision."""
    res = vic.run_golden_parity()
    assert res["status"] == "ok", res
    m = res["metrics"]
    assert res["only_in_golden"] == 0 and res["only_in_regenerated"] == 0
    assert m["spearman"] >= vic.GOLDEN_GATE["min_spearman"]
    assert m["class_agreement"] >= vic.GOLDEN_GATE["min_class_agreement"]
    assert m["max_abs"] < vic.GOLDEN_GATE["max_max_abs"]
    assert res["passed"]


# ---------------------------------------------------------------------------
# In-container, tmol: evaluator vs the shipped 1ubq oracle (scope item 1).
# ---------------------------------------------------------------------------

@needs_tmol
def test_evaluator_matches_1ubq_oracle():
    """Every in-scope pairwise subterm reproduces tmol's shipped 1ubq baseline within
    tmol's own whole-pose test tolerance (atol=1e-3, rtol=1e-3)."""
    res = vic.run_oracle_parity()
    if res["status"] == "skipped":
        pytest.skip(res["reason"])
    assert res["passed"], res["rows"]
    assert res["max_abs_diff"] <= vic.ORACLE_ATOL + vic.ORACLE_RTOL * 1e3


# ---------------------------------------------------------------------------
# In-container, tmol: tmol energies vs frozen Rosetta logs (scope item 3, frozen half).
# ---------------------------------------------------------------------------

@pytest.mark.slow
@needs_tmol
@needs_ref
def test_tmol_vs_frozen_rosetta_cross_check():
    """tmol (beta_nov2016) vs the frozen Rosetta (ref2015) per-pair logs on the IDENTICAL
    shipped poses: the per-residue Function1 energy and the engine-swap FrstIndex agree
    in rank and class to the documented in-container gate. This is a strong-correlation,
    not bit-exact, check: the two energy functions differ by construction."""
    res = vvp.run_frozen_cross_check(n_decoys=50, native_idx=1)
    if res["status"] == "skipped":
        pytest.skip(res["reason"])
    rm = res["residue_metrics"]
    fm = res["frstindex_metrics"]
    assert rm["spearman"] >= vvp.FROZEN_GATE["min_residue_spearman"], rm
    assert fm["spearman"] >= vvp.FROZEN_GATE["min_frstindex_spearman"], fm
    assert fm["class_agreement"] >= vvp.FROZEN_GATE["min_class_agreement"], fm
    assert res["passed"]


@pytest.mark.slow
@needs_tmol
@needs_ref
def test_per_term_breakdown_is_term_local_not_uniform():
    """The energy-function gap is term-local, not a uniform offset: the LJ/solvation/elec
    terms track Rosetta strongly (Spearman high) while hbond/lk_ball diverge more. This
    documents WHERE the beta_nov2016-vs-ref2015 difference lives (feeds LABEL_DECISION.md)."""
    res = vvp.run_frozen_cross_check(n_decoys=50, native_idx=1)
    if res["status"] == "skipped":
        pytest.skip(res["reason"])
    by_term = {r["term"]: r for r in res["term_rows"]}
    # The dominant Lennard-Jones/solvation/electrostatic terms port faithfully.
    for term in ("fa_atr", "fa_rep", "fa_sol", "fa_elec"):
        assert by_term[term]["spearman"] >= 0.90, by_term[term]
    # hbond/lk_ball are present and quantified (the term-local gap); we assert they were
    # measured over the full pair set rather than pinning a fragile divergence number.
    for term in ("hbond", "lk_ball"):
        assert by_term[term]["n"] > 0


# ---------------------------------------------------------------------------
# Maintainer, PyRosetta: live cross-check (skips cleanly in-container).
# ---------------------------------------------------------------------------

@needs_pyrosetta
def test_tmol_vs_live_pyrosetta_within_tolerance():
    """When PyRosetta is present (maintainer host): the live-PyRosetta re-score
    reproduces the frozen logs and the tmol energies track live PyRosetta in rank, both
    within the documented tolerance. Skipped in-container (no PyRosetta)."""
    res = vvp.run_pyrosetta_cross_check(pose_idxs=[1, 2, 3])
    if res["status"] == "skipped":
        pytest.skip(res["reason"])
    for p in res["poses"]:
        # The shipped logs must reproduce a fresh single-point Rosetta score closely.
        assert p["live_vs_frozen"]["spearman"] >= 0.95, p
        # tmol must track live PyRosetta at least as well as it tracks the frozen logs.
        assert p["tmol_vs_live"]["spearman"] >= vvp.FROZEN_GATE["min_residue_spearman"], p


def test_pyrosetta_cross_check_skips_cleanly_without_pyrosetta():
    """The maintainer cross-check returns a clear SKIPPED marker (never raises) when
    PyRosetta is absent, which is the in-container state."""
    if _HAVE_PYROSETTA:
        pytest.skip("PyRosetta present; the skip-path is not exercised here")
    res = vvp.run_pyrosetta_cross_check()
    assert res["status"] == "skipped"
    assert "PyRosetta" in res["reason"]
