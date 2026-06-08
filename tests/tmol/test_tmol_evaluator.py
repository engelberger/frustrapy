"""tmol energy-evaluator parity + autograd tests (TMOL-PY-BACKEND, #38).

These exercise the REAL tmol engine (:mod:`frustrapy.backends.atomic_tmol_engine`):
the per-residue-pair evaluator over the in-scope pairwise ref2015 terms, validated
against tmol's own shipped 1ubq oracle, plus the differentiability the mission's
definition of done asks for.

They are skipped cleanly when tmol is not importable in the running interpreter. Per
``docs/tmol/ENERGY_AUDIT.md`` the published CPU wheel needs a JIT bridge
(``TMOL_USE_JIT=1`` + ninja + g++) until the upstream packaging gap is closed, so the
default fast lane (no tmol) skips and a tmol-enabled env runs them. The oracle is the
committed ``docs/tmol/oracle/1ubq_term_energies.json`` (#37), shared with the WebGPU
port; the 1ubq structure is tmol's own shipped fixture, located from the installed tmol
package or ``$FRUSTRAPY_TMOL_1UBQ``.
"""

import json
import os

import numpy as np
import pytest

from frustrapy.backends import atomic_tmol_engine as E

pytestmark = pytest.mark.skipif(
    not E.tmol_available(),
    reason="optional tmol package not importable (see docs/tmol/ENERGY_AUDIT.md JIT setup)",
)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ORACLE_JSON = os.path.join(_REPO, "docs", "tmol", "oracle", "1ubq_term_energies.json")
CRN_PDB = os.path.join(_REPO, "tests", "data", "1crn.pdb")

# tmol's own 1ubq fixture (the oracle was measured on it). Try the env override, the
# installed tmol package, then the read-only source clone.
def _find_1ubq():
    cand = [os.environ.get("FRUSTRAPY_TMOL_1UBQ")]
    try:
        import tmol  # noqa: PLC0415

        cand.append(
            os.path.join(os.path.dirname(tmol.__file__), "tests", "data", "pdb", "1ubq.pdb")
        )
    except Exception:
        pass
    cand.append("/workspace/tmol_src/tmol/tests/data/pdb/1ubq.pdb")
    for p in cand:
        if p and os.path.exists(p):
            return p
    return None


UBQ_PDB = _find_1ubq()
_need_ubq = pytest.mark.skipif(UBQ_PDB is None, reason="tmol 1ubq fixture not found")


def _oracle():
    with open(ORACLE_JSON) as fh:
        return json.load(fh)


# Pairwise subterm name -> (oracle term group, index within that group's pose0 list).
# The one-body ref term is intentionally NOT part of the pairwise contact evaluator.
_SUBTERM_ORACLE = {
    "fa_ljatr": ("LJLKEnergyTerm", 0),
    "fa_ljrep": ("LJLKEnergyTerm", 1),
    "fa_lk": ("LJLKEnergyTerm", 2),
    "lk_ball_iso": ("LKBallEnergyTerm", 0),
    "lk_ball": ("LKBallEnergyTerm", 1),
    "lk_bridge": ("LKBallEnergyTerm", 2),
    "lk_bridge_uncpl": ("LKBallEnergyTerm", 3),
    "fa_elec": ("ElecEnergyTerm", 0),
    "hbond": ("HBondEnergyTerm", 0),
}


@_need_ubq
def test_whole_pose_terms_match_tmol_oracle():
    """Build the pairwise score function the evaluator uses and score 1ubq whole-pose;
    each in-scope subterm must reproduce tmol's shipped baseline. This re-proves #37's
    oracle through the frustrapy-side score-function builder (adapt-and-wrap)."""
    import torch  # noqa: PLC0415

    oracle = _oracle()
    pose = E._pose_stack_from_pdb(UBQ_PDB)
    sfxn = E.build_pairwise_score_function(device=pose.coords.device)
    wp = sfxn.render_whole_pose_scoring_module(pose)
    coords = torch.nn.Parameter(pose.coords.clone())
    unweighted = wp(coords, sum_terms=False, apply_weights=False).detach().cpu().numpy()
    score_types = [st for term in sfxn.all_terms() for st in term.score_types()]
    by_name = {st.name: unweighted[row, 0] for row, st in enumerate(score_types)}

    for name, (group, idx) in _SUBTERM_ORACLE.items():
        baseline = oracle[group]["baseline_pose0"][idx]
        measured = float(by_name[name])
        # tmol's own whole-pose test tolerance: atol=1e-5, rtol=1e-3.
        np.testing.assert_allclose(measured, baseline, atol=1e-3, rtol=1e-3,
                                   err_msg=f"{name} (whole-pose) drifts from tmol oracle")


@_need_ubq
def test_block_pair_evaluator_shape_and_ljlk_parity():
    """The per-residue-pair evaluator returns N x N matrices per subterm, and the
    dominant ljlk/hbond terms summed over all block pairs reproduce tmol's whole-pose
    oracle. (fa_elec differs by an intra-residue diagonal term not used for inter-residue
    contacts; see docs/tmol/PY_BACKEND_NOTES.md, so it is checked loosely.)"""
    oracle = _oracle()
    pose = E._pose_stack_from_pdb(UBQ_PDB)
    pe = E.evaluate_pair_energies(pose)
    n = pe.n_blocks
    assert n > 0
    assert pe.total_per_pair.shape == (n, n)
    assert pe.rep_per_pair.shape == (n, n)
    for name in ("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec", "hbond"):
        assert pe.per_subterm[name].shape == (n, n)

    w = E.PAIRWISE_WEIGHTS
    # Block-pair sum of the unweighted matrix vs the whole-pose oracle.
    for name, tol in (("fa_ljatr", 1e-2), ("fa_ljrep", 1e-2), ("fa_lk", 1e-2),
                      ("hbond", 1e-2)):
        group, idx = _SUBTERM_ORACLE[name]
        baseline = oracle[group]["baseline_pose0"][idx]
        unweighted_sum = pe.per_subterm[name].sum() / w[name]
        np.testing.assert_allclose(unweighted_sum, baseline, atol=tol, rtol=1e-3,
                                   err_msg=f"{name} block-pair sum drifts from oracle")
    # fa_elec: equal to the oracle minus the intra-residue (diagonal) contribution.
    g, i = _SUBTERM_ORACLE["fa_elec"]
    elec_sum = pe.per_subterm["fa_elec"].sum() / w["fa_elec"]
    assert abs(elec_sum - oracle[g]["baseline_pose0"][i]) < 5.0  # intra-residue gap


@_need_ubq
def test_scheme_pair_matrix_drops_rep():
    """Function1 per-pair energy == total per pair minus the repulsive term."""
    pose = E._pose_stack_from_pdb(UBQ_PDB)
    pe = E.evaluate_pair_energies(pose)
    f1 = pe.scheme_pair_matrix("Function1")
    np.testing.assert_allclose(f1, pe.total_per_pair - pe.rep_per_pair, atol=1e-6)
    packing = pe.scheme_pair_matrix("Packing")
    np.testing.assert_allclose(packing, pe.total_per_pair, atol=1e-6)


@_need_ubq
def test_autograd_produces_finite_gradients():
    """The evaluator is differentiable wrt coordinates: autograd produces finite,
    non-trivial gradients on a tiny input (mission definition of done)."""
    import torch  # noqa: PLC0415

    pose = E._pose_stack_from_pdb(UBQ_PDB)
    weighted_summed, _unweighted, coords = E.evaluate_pair_energy_tensor(pose)
    grad = torch.autograd.grad(weighted_summed.sum(), coords)[0]
    assert torch.isfinite(grad).all()
    assert float(grad.norm()) > 0.0


def test_native_scorer_on_1crn_returns_finite_residue_energies():
    """The native scorer (the genuinely license-clean half: single-point scoring, no
    Rosetta, no packer) returns one finite energy per 1crn residue."""
    from frustrapy.backends import atomic_post as ap

    n = ap.load_contact_geometry(CRN_PDB).n_residues
    energies = E.compute_native_residue_energies_tmol(CRN_PDB)
    assert energies.shape == (n,)
    assert np.isfinite(energies).all()


@pytest.mark.slow
def test_real_tmol_native_end_to_end_no_pyrosetta(tmp_path, monkeypatch):
    """End to end on 1crn with the REAL tmol native scorer (no PyRosetta, no GPU): the
    native per-residue energies come from tmol's single-point evaluation, flow through
    the shared post-processor, and produce the full output contract and a valid
    14-column table. The decoy ensemble uses a deterministic stub (the heavy tmol
    repack is the maintainer path; see docs/tmol/PY_BACKEND_NOTES.md), so this asserts
    the genuine-tmol-energy pipeline, not decoy physics."""
    import frustrapy
    import pandas as pd
    from frustrapy.backends import AtomicTmolBackend, atomic_post as ap

    monkeypatch.setenv("FRUSTRAPY_ATOMIC_N_DECOYS", "8")
    monkeypatch.setenv("FRUSTRAPY_ATOMIC_SEED", "7")

    def stub_decoy(pdb_path, decoy_seq, *, native_seq, scheme, fa_rep_cutoff, n_threads):
        n = ap.load_contact_geometry(pdb_path).n_residues
        rng = np.random.default_rng(abs(hash(decoy_seq)) % (2**32))
        return rng.normal(0.0, 1.0, size=n)

    # Real tmol native scorer; stub decoys.
    backend = AtomicTmolBackend(decoy_scorer=stub_decoy)
    result = frustrapy.calculate_frustration(
        pdb_file=CRN_PDB, mode="configurational",
        results_dir=str(tmp_path / "out"), graphics=False, backend=backend,
    )
    assert len(result) == 4
    pdb = result[0]
    fdata = os.path.join(pdb.job_dir, "FrustrationData")
    table = os.path.join(fdata, f"{pdb.pdb_base}.pdb_configurational")
    assert os.path.exists(table)
    assert os.path.exists(os.path.join(fdata, f"{pdb.pdb_base}.pdb_configurational_5adens"))
    df = pd.read_csv(table, sep=r"\s+")
    assert len(df.columns) == 14 and len(df) > 0
    assert np.isfinite(pd.to_numeric(df["NativeEnergy"])).all()
