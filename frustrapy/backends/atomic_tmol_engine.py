"""tmol energy provider for the all-atom Frustratometer backend (TMOL-PY-BACKEND, #38).

This module is the license-clean (Apache-2.0) energy engine for the atomic backend.
It mirrors the LAMMPS-backend philosophy exactly: the energy model is NOT
reimplemented here. The per-residue-pair ref2015 energies are evaluated by **tmol**
(``engelberger/tmol``, an Apache-2.0 PyTorch reimplementation of the Rosetta
``beta_nov2016`` energy function), which #37 (``docs/tmol/ENERGY_AUDIT.md``) confirmed
runs on CPU in-container and reproduces tmol's own shipped 1ubq oracle to ~1e-5. So
this is **adapt-and-wrap, not a reimplementation**: we build a tmol ``ScoreFunction``
over the in-scope pairwise terms and read back its per-block-pair energy tensor.

What this replaces, and what it does not (the load-bearing honesty point):

* It REPLACES the PyRosetta dependency for **energy evaluation** (scoring). The native
  pose is scored directly from its all-atom coordinates with no Rosetta, no GPU, no
  license-gated binary.
* It does NOT remove the need for a **side-chain packer** when scoring decoys. The
  reference's permutation decoys are threaded onto the fixed backbone and *repacked*;
  placing decoy side chains needs a rotamer optimizer + the Dunbrack rotamer library.
  tmol ships a packer (:mod:`tmol.pack.pack_rotamers` /
  :func:`tmol.pack.build_missing_sidechains.build_missing_sidechains`), so this can be
  done license-tier-gated, but it is slow and the Dunbrack library is in the
  non-commercial Rosetta parameter tier (``docs/tmol/PARAM_SOURCING.md``). The decoy
  packing path is therefore implemented here but treated as the heavy/maintainer
  analogue of the PyRosetta engine, exactly as in :mod:`.atomic_engine`.

In-scope terms (ENERGY_AUDIT section 3): the **pairwise (two-body)** contact set is
``ljlk`` (fa_ljatr, fa_ljrep, fa_lk) + ``lk_ball`` (lk_ball_iso, lk_ball, lk_bridge,
lk_bridge_uncpl) + ``elec`` (fa_elec) + ``hbond`` (hbond). The one-body ``ref`` term is
a per-residue baseline and is NOT part of an inter-residue contact energy, so it is
excluded from the per-pair sum (kept available for a per-residue total).

Sign convention is NOT decided here. This module emits raw per-residue energies; the
AWSEM-sign flip (``FrstIndex = (DecoyEnergy - NativeEnergy) / SDEnergy``), the contact
geometry, the 19-column ``tertiary_frustration.dat`` adapter, and the FrstState
classification all belong to the shared :mod:`.atomic_post` (reused unchanged).

Parallelism (mission scope item 1): this module spawns NO process pool. tmol uses
torch intra-op threads for its CPU kernels. The decoy loop in :mod:`.atomic_tmol` is a
plain serial loop so it never nests a pool inside the calculator's pool. The caller
sets the thread budget via :func:`configure_torch_threads` (the calculator's resolved
inner core budget); see ``docs/tmol/PY_BACKEND_NOTES.md`` and the package
parallel-safety notes.

tmol is imported lazily and only when an energy is actually evaluated, so importing
this module, ``import frustrapy``, and the default LAMMPS path never import tmol.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Term set and beta_nov2016 weights (ENERGY_AUDIT sections 0/1/3).
#
# The pairwise (two-body) contact terms and their canonical beta_nov2016 weights,
# verbatim from tmol/score/__init__.py:_non_memoized_beta2016 (read in #37). The
# one-body `ref` term is intentionally excluded from the per-pair contact sum.
# ---------------------------------------------------------------------------

#: Pairwise score-type name -> beta_nov2016 weight. Names are tmol ScoreType members.
PAIRWISE_WEIGHTS: Dict[str, float] = {
    "fa_ljatr": 1.0,
    "fa_ljrep": 0.55,
    "fa_lk": 1.0,
    "fa_elec": 1.0,
    "hbond": 1.0,
    "lk_ball_iso": -0.38,
    "lk_ball": 0.92,
    "lk_bridge": -0.33,
    "lk_bridge_uncpl": -0.33,
}

#: The repulsive subterm (Rosetta ``fa_rep``). Two roles, matching the atomic
#: reference: it is dropped from the per-pair "Function1" energy, and it is the column
#: the clashing-pair filter reads (a pair contributes to a residue only if its weighted
#: fa_rep is at or below :data:`FA_REP_CUTOFF`).
REP_TERM = "fa_ljrep"

#: Clashing-pair cutoff on the weighted repulsive term (atomic reference
#: ``Frust_Post_public.py``: ``fa_rep <= 5.0``). Pairs above it are excluded from the
#: per-residue energy. Applied to the weighted fa_ljrep so it matches the reference,
#: which reads the weighted ResResE column.
FA_REP_CUTOFF = 5.0

#: Energy schemes, mirroring :data:`frustrapy.backends.atomic_engine.ENERGY_SCHEMES`.
#: tmol has no ``pro_close`` / ``dslf_fa13`` two-body terms (those are not in the
#: pairwise contact set), so the reference's ``total - pro_close - dslf`` is automatic;
#: only the fa_rep handling distinguishes the schemes here.
#:   Function1 -> drop fa_rep (the reference default, job.sh line 32)
#:   Function2 -> drop fa_rep and fa_atr
#:   Packing   -> keep all pairwise terms
ENERGY_SCHEMES: Tuple[str, ...] = ("Function1", "Function2", "Packing")
DEFAULT_SCHEME = "Function1"

#: Environment variable the calculator/user can set to pin torch intra-op threads for
#: the tmol kernels (the inner core budget). Honored by :func:`configure_torch_threads`
#: when no explicit count is passed.
THREADS_ENV = "FRUSTRAPY_TMOL_THREADS"


# ---------------------------------------------------------------------------
# Lazy tmol import + availability (mirrors the PyRosetta plumbing in
# frustrapy/analysis/mutation_backends.py and frustrapy/backends/atomic_engine.py).
# ---------------------------------------------------------------------------

_TMOL_SETUP_HELP = (
    "The 'atomic-tmol' backend needs the optional 'tmol' package (Apache-2.0).\n"
    "tmol evaluates the ref2015/beta_nov2016 energy on CPU. Per docs/tmol/ENERGY_AUDIT.md\n"
    "the published CPU wheel ships without one pybind module, so until that gap is closed\n"
    "upstream a C++ toolchain + JIT bridge is required. Maintainer setup:\n"
    "  uv venv /tmp/tmolprobe --python 3.12 && source /tmp/tmolprobe/bin/activate\n"
    "  uv pip install 'torch>=2.5,<3' --index-url https://download.pytorch.org/whl/cpu\n"
    "  uv pip install <tmol cp312 cpu wheel> ninja\n"
    "  export TMOL_USE_JIT=1   # and ensure 'ninja' is on PATH\n"
    "Preferred long-term: build tmol from source (compiles all pybind modules) or have\n"
    "upstream include the missing module in the AOT wheel. See docs/tmol/PY_BACKEND_NOTES.md."
)


def tmol_available() -> bool:
    """True iff the optional ``tmol`` package can be imported in this process.

    Importing tmol may JIT-compile kernels on first use (see :data:`_TMOL_SETUP_HELP`),
    so this can be slow on a cold cache; it is cached thereafter.
    """
    try:
        import tmol  # noqa: F401
    except Exception:
        return False
    return True


def _ensure_tmol():
    """Import tmol or raise a clear, actionable error. The import is done here (not at
    module top) so importing this module never imports tmol."""
    try:
        import tmol  # noqa: F401
    except Exception as exc:  # ImportError or a JIT/toolchain error
        raise ImportError(_TMOL_SETUP_HELP) from exc
    return tmol


def configure_torch_threads(n_threads: Optional[int] = None) -> int:
    """Pin torch intra-op threads for the tmol CPU kernels and return the value set.

    The tmol backend parallelizes only via torch intra-op threads; it spawns no
    process pool (so it never nests inside the calculator's pool). The CALLER owns the
    budget: the calculator passes its resolved inner core budget here before scoring,
    so ``outer_pool * torch_threads <= cores``. With ``n_threads=None`` the
    :data:`THREADS_ENV` environment variable is used if set; otherwise torch's current
    default is left untouched. Returns the effective thread count.
    """
    import torch  # noqa: PLC0415

    if n_threads is None:
        env = os.environ.get(THREADS_ENV)
        if env:
            try:
                n_threads = int(env)
            except ValueError:
                n_threads = None
    if n_threads is not None and n_threads >= 1:
        torch.set_num_threads(int(n_threads))
    return torch.get_num_threads()


# ---------------------------------------------------------------------------
# The pairwise score function and the per-residue-pair evaluator.
# ---------------------------------------------------------------------------

def build_pairwise_score_function(device=None, param_db=None):
    """Build a tmol ``ScoreFunction`` carrying ONLY the in-scope pairwise terms
    (:data:`PAIRWISE_WEIGHTS`) at their beta_nov2016 weights.

    The one-body ``ref`` term is deliberately omitted (it is not part of an
    inter-residue contact energy). This is the energy model the evaluator wraps; it is
    the swappable part, exactly like the AWSEM coefficient file is for the LAMMPS
    backend.
    """
    _ensure_tmol()
    import torch  # noqa: PLC0415
    from tmol.score.score_function import ScoreFunction  # noqa: PLC0415
    from tmol.score.score_types import ScoreType  # noqa: PLC0415
    from tmol.database import ParameterDatabase  # noqa: PLC0415

    if device is None:
        device = torch.device("cpu")
    if param_db is None:
        param_db = ParameterDatabase.get_default()
    sfxn = ScoreFunction(param_db, device)
    for name, weight in PAIRWISE_WEIGHTS.items():
        sfxn.set_weight(getattr(ScoreType, name), float(weight))
    return sfxn


@dataclass
class PairEnergies:
    """Per-residue-pair energies for one pose, the output of :func:`evaluate_pair_energies`.

    ``total_per_pair[i, j]`` is the weighted sum of every in-scope pairwise term for the
    contact between block (residue) ``i`` and block ``j`` (a symmetric ``n x n`` matrix
    whose off-diagonal is the inter-residue contact energy; the diagonal carries the
    intra-residue two-body contribution and is not used for contacts). ``rep_per_pair``
    is the weighted repulsive term alone (the clashing-pair filter column).
    ``per_subterm`` maps each term name to its weighted ``n x n`` matrix. ``total`` is
    the scalar pose total over all pairs.

    All arrays are plain numpy (detached). For autograd, use the tensors returned by
    :func:`evaluate_pair_energy_tensor`.
    """

    total_per_pair: np.ndarray
    rep_per_pair: np.ndarray
    per_subterm: Dict[str, np.ndarray]
    n_blocks: int

    @property
    def total(self) -> float:
        return float(self.total_per_pair.sum())

    def scheme_pair_matrix(self, scheme: str = DEFAULT_SCHEME) -> np.ndarray:
        """The per-pair contact energy under ``scheme`` (the Function1 analogue).

        ``Function1`` drops the repulsive term; ``Function2`` drops repulsive and
        attractive; ``Packing`` keeps all pairwise terms. Returns the ``n x n`` matrix.
        """
        if scheme == "Packing":
            return self.total_per_pair
        if scheme == "Function1":
            return self.total_per_pair - self.rep_per_pair
        if scheme == "Function2":
            atr = self.per_subterm.get("fa_ljatr")
            base = self.total_per_pair - self.rep_per_pair
            return base if atr is None else base - atr
        raise ValueError(
            f"unknown energy scheme {scheme!r}; expected one of {ENERGY_SCHEMES}"
        )

    def residue_energies(
        self, scheme: str = DEFAULT_SCHEME, fa_rep_cutoff: float = FA_REP_CUTOFF
    ) -> np.ndarray:
        """Per-residue energy array (0-based block index), porting the reference's
        ``ene_res`` split: for every off-diagonal pair whose weighted fa_rep is at or
        below ``fa_rep_cutoff``, half of the scheme pair energy is added to each of the
        two residues (``ene_res[i] += 0.5*ene``; ``ene_res[j] += 0.5*ene``).

        This produces exactly the per-residue quantity
        :func:`frustrapy.backends.atomic_engine.residue_energies` produces from a
        Rosetta ResResE log, so the shared :class:`~frustrapy.backends.atomic_engine.EngineResult`
        aggregation and :mod:`.atomic_post` writer consume it unchanged.
        """
        n = self.n_blocks
        scheme_mat = self.scheme_pair_matrix(scheme)
        ene = np.zeros(n, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                if self.rep_per_pair[i, j] > fa_rep_cutoff:
                    continue
                half = 0.5 * scheme_mat[i, j]
                ene[i] += half
                ene[j] += half
        return ene


def _pose_stack_from_pdb(pdb_path: str, device=None):
    """Build a tmol ``PoseStack`` from a PDB file on CPU."""
    _ensure_tmol()
    import torch  # noqa: PLC0415
    from tmol.io import pose_stack_from_pdb  # noqa: PLC0415

    if device is None:
        device = torch.device("cpu")
    with open(pdb_path) as fh:
        content = fh.read()
    return pose_stack_from_pdb(content, device)


def evaluate_pair_energy_tensor(pose_stack, sfxn=None, coords=None):
    """Evaluate the per-residue-pair energy as differentiable torch tensors.

    Returns ``(weighted_summed, unweighted_subterms, coords)`` where
    ``weighted_summed`` has shape ``(n_poses, n_blocks, n_blocks)`` (the full pairwise
    contact energy, weights applied, summed over subterms) and ``unweighted_subterms``
    has shape ``(n_subterms, n_poses, n_blocks, n_blocks)``. ``coords`` is the
    (possibly newly created) ``torch.nn.Parameter`` the scoring was run on, so callers
    can take gradients (``torch.autograd.grad(weighted_summed.sum(), coords)``).

    This is the differentiable form of the evaluator (mission scope item 1). When
    ``coords`` is ``None`` a fresh ``Parameter`` is cloned from the pose so a gradient
    can flow; pass your own leaf tensor to differentiate through it.
    """
    _ensure_tmol()
    import torch  # noqa: PLC0415

    if sfxn is None:
        sfxn = build_pairwise_score_function(device=pose_stack.coords.device)
    bp = sfxn.render_block_pair_scoring_module(pose_stack)
    if coords is None:
        coords = torch.nn.Parameter(pose_stack.coords.clone())
    weighted_summed = bp(coords, sum_terms=True, apply_weights=True)
    unweighted = bp(coords, sum_terms=False, apply_weights=False)
    return weighted_summed, unweighted, coords


def evaluate_pair_energies(pose_stack, sfxn=None) -> PairEnergies:
    """Evaluate the in-scope per-residue-pair energies for one pose (CPU).

    The single evaluator interface the mission asks for: given a tmol ``PoseStack``
    (all-atom coordinates + atom/residue types), return the per-residue-pair energy
    matrix plus the per-subterm breakdown and the pose total, for the pairwise terms
    (ljlk + lk_ball + elec + hbond). Uses tmol's ``render_block_pair_scoring_module``,
    whose output is shape ``(n_poses, n_blocks, n_blocks)`` (per #38 probe), summed over
    the requested terms; here we score a single pose.

    Numbers come straight from tmol (validated against tmol's 1ubq oracle in
    ``tests/tmol/test_tmol_evaluator.py``); this function does not reimplement any
    energy term.
    """
    _ensure_tmol()
    import torch  # noqa: PLC0415
    from tmol.score.score_types import ScoreType  # noqa: PLC0415

    if sfxn is None:
        sfxn = build_pairwise_score_function(device=pose_stack.coords.device)

    weighted_summed, unweighted, _ = evaluate_pair_energy_tensor(pose_stack, sfxn=sfxn)
    # pose 0 only (single structure)
    total_per_pair = weighted_summed[0].detach().cpu().numpy()
    n_blocks = total_per_pair.shape[0]

    # Recover the per-subterm weighted matrices. The unweighted block-pair rows are in
    # the flattened term/score-type order tmol assembled (one row per score type, terms
    # in all_terms() order), which is the SAME order weights_tensor() uses; so row r maps
    # to score_types[r] and weight weights[r]. This keeps the per-subterm breakdown and
    # the repulsive matrix (the clashing-pair filter column) explicit.
    score_types = [st for term in sfxn.all_terms() for st in term.score_types()]
    weights = sfxn.weights_tensor()
    per_subterm: Dict[str, np.ndarray] = {}
    for row, st in enumerate(score_types):
        name = st.name
        if name not in PAIRWISE_WEIGHTS:
            continue
        w = float(weights[row])
        per_subterm[name] = (unweighted[row, 0].detach().cpu().numpy()) * w

    rep_per_pair = per_subterm.get(
        REP_TERM, np.zeros((n_blocks, n_blocks), dtype=float)
    )
    return PairEnergies(
        total_per_pair=total_per_pair,
        rep_per_pair=rep_per_pair,
        per_subterm=per_subterm,
        n_blocks=n_blocks,
    )


# ---------------------------------------------------------------------------
# Native + decoy scorers (the engine's two execution regimes), returning per-residue
# energy arrays in block (PDB-residue) order. The backend maps block index -> residue
# key via the shared ContactGeometry, so these stay tmol-only and label-free.
# ---------------------------------------------------------------------------

def compute_native_residue_energies_tmol(
    pdb_path: str,
    *,
    scheme: str = DEFAULT_SCHEME,
    fa_rep_cutoff: float = FA_REP_CUTOFF,
    n_threads: Optional[int] = None,
    device=None,
) -> np.ndarray:
    """Native per-residue energies via a single tmol single-point evaluation.

    Scores the structure at ``pdb_path`` AS-IS (the native all-atom coordinates), with
    NO Rosetta and NO repack, and returns the per-residue energy array in block (PDB
    residue) order. Fully runnable in-container (this is the half tmol genuinely makes
    license-clean). The caller aligns the array to residue keys via the shared
    :class:`~frustrapy.backends.atomic_post.ContactGeometry` (same PDB order).
    """
    configure_torch_threads(n_threads)
    pose = _pose_stack_from_pdb(pdb_path, device=device)
    pair = evaluate_pair_energies(pose)
    return pair.residue_energies(scheme=scheme, fa_rep_cutoff=fa_rep_cutoff)


def compute_decoy_residue_energies_tmol(
    pdb_path: str,
    decoy_seq: str,
    *,
    native_seq: Optional[str] = None,
    scheme: str = DEFAULT_SCHEME,
    fa_rep_cutoff: float = FA_REP_CUTOFF,
    n_threads: Optional[int] = None,
    device=None,
) -> np.ndarray:
    """Decoy per-residue energies: thread ``decoy_seq`` onto the fixed native backbone,
    rebuild side chains, score with tmol, and return per-residue energies in block
    order.

    HEAVY / parameter-tier-gated (see the module docstring). Side-chain placement uses
    tmol's packer (:func:`tmol.pack.build_missing_sidechains.build_missing_sidechains`,
    Dunbrack rotamer library). This is the tmol analogue of the PyRosetta repack in
    :func:`frustrapy.backends.atomic_engine.compute_decoy_pair_energies`: it removes
    PyRosetta, but it is slow and the Dunbrack params are in the non-commercial Rosetta
    tier (``docs/tmol/PARAM_SOURCING.md``). For fast/CI runs the backend accepts an
    injectable scorer instead (the AA-lane pattern), and the in-container end-to-end
    test uses that. This function raises a clear error if the packing path cannot run.
    """
    configure_torch_threads(n_threads)
    pose = _thread_and_repack(pdb_path, decoy_seq, native_seq=native_seq, device=device)
    pair = evaluate_pair_energies(pose)
    return pair.residue_energies(scheme=scheme, fa_rep_cutoff=fa_rep_cutoff)


def _permute_res_types(res_types, native_seq: str, decoy_seq: str):
    """Reorder the canonical ``res_types`` row to realize a permutation decoy.

    The reference decoy model is a composition-preserving PERMUTATION of the native
    sequence (``atomic_engine.generate_decoy_sequences`` / ``RandSeq.py``). Rather than
    map one-letter codes to canonical indices, we realize that permutation directly on
    the native ``res_types`` tensor: the decoy is the native multiset reordered, so for
    each decoy position we take the native res_type of the position the decoy borrowed
    its identity from. This is exact and avoids a code-to-index table.
    """
    import torch  # noqa: PLC0415

    # Greedy stable assignment: for each decoy character, consume the next native
    # position carrying that identity. Composition-preserving guarantees a full match.
    from collections import defaultdict, deque  # noqa: PLC0415

    buckets = defaultdict(deque)
    for pos, aa in enumerate(native_seq):
        buckets[aa].append(pos)
    order = []
    for aa in decoy_seq:
        order.append(buckets[aa].popleft())
    idx = torch.as_tensor(order, dtype=torch.long, device=res_types.device)
    return res_types[:, idx].clone()


def _thread_and_repack(pdb_path: str, decoy_seq: str, native_seq: Optional[str] = None, device=None):
    """Build a tmol pose for ``decoy_seq`` on the native backbone, side chains rebuilt.

    HEAVY / maintainer-validated, exactly like
    :func:`frustrapy.backends.atomic_engine.compute_decoy_pair_energies` (the PyRosetta
    analogue). Threads the decoy identities onto the fixed native backbone by permuting
    the canonical ``res_types`` (composition-preserving), drops side chains so the new
    identities have none, and rebuilds them with tmol's packer
    (:func:`build_missing_sidechains`, Dunbrack rotamer library). The packer JIT-compiles
    its rotamer/anneal kernels on first use (slow cold start) and the Dunbrack library is
    in the non-commercial parameter tier (``docs/tmol/PARAM_SOURCING.md``); this is why
    the in-container tests inject a scorer instead. Lazy imports keep tmol out of module
    import.
    """
    _ensure_tmol()
    import torch  # noqa: PLC0415
    from tmol.io.canonical_ordering import (  # noqa: PLC0415
        default_canonical_ordering,
        default_packed_block_types,
        canonical_form_from_pdb,
    )
    from tmol.io.pose_stack_construction import pose_stack_from_canonical_form  # noqa: PLC0415
    from tmol.pack.build_missing_sidechains import build_missing_sidechains  # noqa: PLC0415
    from tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler import (  # noqa: PLC0415
        create_dunbrack_sampler_from_database,
    )
    from tmol.database import ParameterDatabase  # noqa: PLC0415

    if device is None:
        device = torch.device("cpu")
    co = default_canonical_ordering()
    pbt = default_packed_block_types(device)
    cf = canonical_form_from_pdb(co, pdb_path, device)

    if native_seq is not None and len(native_seq) == len(decoy_seq):
        decoy_res_types = _permute_res_types(cf.res_types, native_seq, decoy_seq)
    else:
        # No native sequence given: score the native identities as a fallback (the
        # caller should pass native_seq for a true permutation decoy).
        decoy_res_types = cf.res_types
    # Changing res_types leaves the new identities' side-chain atoms absent, so the
    # construction reports them missing and the packer rebuilds them.
    pose_stack, opt = pose_stack_from_canonical_form(
        co, pbt, cf.chain_id, decoy_res_types, cf.coords,
        cf.res_labels, cf.residue_insertion_codes, cf.chain_labels,
        return_block_has_missing_atoms=True,
    )
    block_has_missing = opt["block_has_missing_atoms"]
    db = ParameterDatabase.get_default()
    sfxn = build_pairwise_score_function(device=device, param_db=db)
    dun = create_dunbrack_sampler_from_database(db, device)
    return build_missing_sidechains(
        pose_stack, sfxn, dun, block_has_missing, None, no_optH=True
    )


__all__ = [
    "PAIRWISE_WEIGHTS",
    "REP_TERM",
    "FA_REP_CUTOFF",
    "ENERGY_SCHEMES",
    "DEFAULT_SCHEME",
    "THREADS_ENV",
    "tmol_available",
    "configure_torch_threads",
    "build_pairwise_score_function",
    "PairEnergies",
    "evaluate_pair_energies",
    "evaluate_pair_energy_tensor",
    "compute_native_residue_energies_tmol",
    "compute_decoy_residue_energies_tmol",
]
