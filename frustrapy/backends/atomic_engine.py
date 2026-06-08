"""Engine helpers for the all-atom (Rosetta) Frustratometer backend.

This module is the AA-ENGINE deliverable. It mirrors the LAMMPS-backend
philosophy exactly: the energy model lives in Rosetta, not here. We drive
PyRosetta to reproduce what the reference ``job.sh`` / ``native.xml`` / ``test.xml``
do (fixed-backbone FastRelax with side-chain repack; composition-preserving
permutation decoys threaded onto the native backbone), read back the per-residue-pair
``ResResE`` energies, and hand them to the post-processor (AA-OUTPUT). We do NOT
reimplement ref2015.

Reference: the atomic packing Frustratometer (Chen et al.), trimmed copy at
``/workspace/atomic_frustratometer_ref/``. The audit + decision gate is
``docs/atomic/AA_DESIGN_DECISION.md``; the parity oracle for the post-processing
half is ``docs/atomic/golden/``.

Two execution regimes, kept deliberately separate so each half is validated where
it can be:

* Pure post-processing (NO Rosetta needed, validated in-container against the
  shipped reference logs and the golden fixture): the permutation decoy generator
  (:func:`generate_decoy_sequences`), the ``ResResE`` per-pair log parser
  (:func:`parse_resrese_log`), the per-residue and per-contact aggregation
  (:func:`residue_energies`, :class:`EngineResult`, :func:`summarize_contacts`).

* The Rosetta relax/repack (PyRosetta-driven, license-gated, NOT runnable in this
  container, validated by the maintainer per ``docs/atomic/AA_MAINTAINER_RUNBOOK.md``):
  :func:`compute_native_pair_energies`, :func:`compute_decoy_pair_energies`. These
  return the *same* :class:`ResPairEnergy` records the parser yields, so the
  aggregation path is identical whether the energies come from a shipped log or a
  live Rosetta run.

PyRosetta is imported lazily and gracefully (reusing the plumbing in
``frustrapy/analysis/mutation_backends.py``); importing this module never imports
PyRosetta, so ``import frustrapy`` and the default LAMMPS path are unaffected.

Sign convention and the .dat layout are NOT decided here. The engine emits raw
native energies and decoy mean/sd; the AWSEM-sign flip
(``FrstIndex = (DecoyEnergy - NativeEnergy) / SDEnergy``), the 19-column
``tertiary_frustration.dat`` adapter, the density columns, and the FrstState
classification all belong to AA-OUTPUT.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# ResResE log layout (from native.xml / test.xml ScoreCutoffFilter
# report_residue_pair_energies="1"; header verified on the shipped native.log).
#
# Each data line is:
#   ResResE  <Res1>  <Res2>  fa_atr fa_rep fa_sol fa_intra_r fa_intra_s
#            lk_ball_wt fa_elec pro_close hbond_sr_b hbond_lr_b hbond_bb_s
#            hbond_sc dslf_fa13 omega fa_dun p_aa_pp yhh_planar ref rama_prepr total
#
# A residue token is "<aa1>_<chain><resnum>", e.g. "D_A1" -> aa "D", key "A1".
# The reference slices strs[1][2:] for the key and strs[1][0] for the amino acid,
# which assumes a single-character chain id; we mirror that assumption faithfully.
# ---------------------------------------------------------------------------

#: The 20 weighted energy terms in column order (the final one is the row total).
TERM_NAMES: Tuple[str, ...] = (
    "fa_atr", "fa_rep", "fa_sol", "fa_intra_r", "fa_intra_s", "lk_ball_wt",
    "fa_elec", "pro_close", "hbond_sr_b", "hbond_lr_b", "hbond_bb_s", "hbond_sc",
    "dslf_fa13", "omega", "fa_dun", "p_aa_pp", "yhh_planar", "ref", "rama_prepr",
    "total",
)

#: Energy decomposition schemes (Frust_Post_public.py:166-171). Each drops a
#: different subset of Rosetta terms from the row ``total`` before the per-residue
#: split. ``Function1`` is the reference default (``job.sh`` line 32).
ENERGY_SCHEMES: Tuple[str, ...] = ("Function1", "Function2", "Packing")

#: Reference default scheme.
DEFAULT_SCHEME = "Function1"

#: A pair contributes to the per-residue energy only if its ``fa_rep`` is at or
#: below this cutoff (Frust_Post_public.py:152,172). Pairs above it are clashing
#: and excluded.
FA_REP_CUTOFF = 5.0

# The two non-numeric header rows the reference skips: ``Res1`` (column header)
# and ``nonzero``/``weights`` (the weights line).
_HEADER_RES1_TOKENS = frozenset({"Res1", "nonzero"})


@dataclass(frozen=True)
class ResPairEnergy:
    """One ``ResResE`` per-residue-pair line: the two residues and the 20 weighted
    Rosetta energy terms.

    ``res1_key`` / ``res2_key`` are the ``chain+resnum`` strings (e.g. ``"A1"``)
    used to index residues, matching the reference ``cid_list`` entries. ``aa1`` /
    ``aa2`` are the one-letter residue identities. ``terms`` maps every name in
    :data:`TERM_NAMES` to its weighted value.
    """

    res1_key: str
    res2_key: str
    aa1: str
    aa2: str
    terms: Dict[str, float]

    @property
    def fa_atr(self) -> float:
        return self.terms["fa_atr"]

    @property
    def fa_rep(self) -> float:
        return self.terms["fa_rep"]

    @property
    def pro_close(self) -> float:
        return self.terms["pro_close"]

    @property
    def dslf_fa13(self) -> float:
        return self.terms["dslf_fa13"]

    @property
    def total(self) -> float:
        return self.terms["total"]

    def scheme_energy(self, scheme: str = DEFAULT_SCHEME) -> float:
        """The per-pair energy under ``scheme`` (Frust_Post_public.py:166-171).

        ``Function1`` drops ``fa_rep``; ``Function2`` drops ``fa_rep`` and
        ``fa_atr``; ``Packing`` drops neither but always drops ``pro_close`` and
        ``dslf_fa13`` (as do the other two).
        """
        total = self.terms["total"]
        pro_close = self.terms["pro_close"]
        dslf = self.terms["dslf_fa13"]
        if scheme == "Function1":
            return total - self.terms["fa_rep"] - pro_close - dslf
        if scheme == "Function2":
            return total - self.terms["fa_rep"] - self.terms["fa_atr"] - pro_close - dslf
        if scheme == "Packing":
            return total - pro_close - dslf
        raise ValueError(
            f"unknown energy scheme {scheme!r}; expected one of {ENERGY_SCHEMES}"
        )


def _split_residue_token(token: str) -> Tuple[str, str]:
    """``"D_A1"`` -> ``("D", "A1")`` (amino acid, chain+resnum key).

    Mirrors the reference slicing (``strs[1][0]`` and ``strs[1][2:]``), which
    assumes a one-letter amino-acid code, an underscore, and a single-character
    chain id followed by the residue number.
    """
    return token[0], token[2:]


# ---------------------------------------------------------------------------
# E1/E2 parsing -- the ResResE log parser (validated in-container)
# ---------------------------------------------------------------------------

def parse_resrese_log(path: str) -> List[ResPairEnergy]:
    """Parse a Rosetta ``ResResE`` log into a list of :class:`ResPairEnergy`.

    Reads the per-residue-pair lines written by the reference
    ``ScoreCutoffFilter report_residue_pair_energies="1"`` (``native.log`` and the
    decoy ``*.log`` files). The two header rows (``Res1`` column header and the
    ``nonzero weights`` line) and any non-``ResResE`` lines are skipped. No
    filtering on ``fa_rep`` happens here -- that is applied during aggregation
    (:func:`residue_energies`), exactly as the reference does.

    This is the in-container-validated parse half of E1 (native) and E2 (decoys);
    the same records are produced by the PyRosetta path
    (:func:`compute_native_pair_energies` / :func:`compute_decoy_pair_energies`).
    """
    records: List[ResPairEnergy] = []
    with open(path) as fin:
        for line in fin:
            strs = line.split()
            if not strs or strs[0] != "ResResE":
                continue
            if len(strs) < 1 + 2 + len(TERM_NAMES):
                continue
            if strs[1] in _HEADER_RES1_TOKENS:
                continue
            aa1, key1 = _split_residue_token(strs[1])
            aa2, key2 = _split_residue_token(strs[2])
            # The 20 term values follow the two residue tokens (strs[3:23]). The
            # reference reads strs[4]=fa_rep, strs[10]=pro_close, strs[15]=dslf_fa13,
            # strs[-1]=total, which line up with this fixed offset of 3.
            values = strs[3:3 + len(TERM_NAMES)]
            terms = {name: float(v) for name, v in zip(TERM_NAMES, values)}
            records.append(
                ResPairEnergy(res1_key=key1, res2_key=key2, aa1=aa1, aa2=aa2, terms=terms)
            )
    return records


# ---------------------------------------------------------------------------
# E2 -- permutation decoy generator (validated in-container)
# ---------------------------------------------------------------------------

def generate_decoy_sequences(
    native_seq: str, n_decoys: int, seed: Optional[int] = None
) -> List[str]:
    """Generate ``n_decoys`` composition-preserving permutation decoys.

    Python 3 port of the reference ``RandSeq.py`` (``''.join(random.sample(seq,
    len(seq)))``): each decoy is a random permutation of ``native_seq``, so it has
    exactly the native amino-acid composition, reordered. This is the reference's
    single decoy model (audit section 1.2 / 3), distinct from the three AWSEM
    modes.

    Improvement over the reference: the RNG is **seedable**. The reference is
    unseeded; passing ``seed`` makes a decoy ensemble reproducible for parity runs.
    ``seed=None`` reproduces the reference's unseeded behaviour.

    Args:
        native_seq: the native one-letter sequence (e.g. from the input PDB CA
            records).
        n_decoys: number of decoy sequences to draw (the paper recommends >= 200
            for convergence; the shipped demo used 50).
        seed: optional RNG seed for reproducibility.

    Returns:
        A list of ``n_decoys`` permuted sequences, each the same length as
        ``native_seq``.
    """
    if n_decoys < 0:
        raise ValueError(f"n_decoys must be non-negative, got {n_decoys}")
    seq = list(native_seq)
    rng = random.Random(seed)
    return ["".join(rng.sample(seq, len(seq))) for _ in range(n_decoys)]


# ---------------------------------------------------------------------------
# E1/E2/E3 -- aggregation (validated in-container against the golden fixture)
# ---------------------------------------------------------------------------

def residue_energies(
    records: Sequence[ResPairEnergy],
    scheme: str = DEFAULT_SCHEME,
    fa_rep_cutoff: float = FA_REP_CUTOFF,
) -> Dict[str, float]:
    """Per-residue energy from a parsed log (the reference ``ene_res`` array,
    keyed by residue rather than index).

    Faithful port of ``Frust_Post_public.read_log`` / ``read_nat_log``: for every
    pair whose ``fa_rep <= fa_rep_cutoff``, half of the scheme energy is added to
    each of the two contacting residues (``ene_res[i] += 0.5*ene``). Residues with
    no surviving pair are simply absent from the returned dict (treated as 0.0 by
    the contact lookups), matching the reference zero-initialised array.

    Args:
        records: parsed :class:`ResPairEnergy` lines for one structure/log.
        scheme: energy decomposition scheme (see :data:`ENERGY_SCHEMES`).
        fa_rep_cutoff: drop pairs with ``fa_rep`` above this (clashing pairs).

    Returns:
        ``{res_key: energy}`` for residues with at least one surviving pair.
    """
    ene: Dict[str, float] = {}
    for rec in records:
        if rec.fa_rep <= fa_rep_cutoff:
            half = 0.5 * rec.scheme_energy(scheme)
            ene[rec.res1_key] = ene.get(rec.res1_key, 0.0) + half
            ene[rec.res2_key] = ene.get(rec.res2_key, 0.0) + half
    return ene


@dataclass
class ContactEnergySummary:
    """E3 per-contact result: the native contact energy plus the decoy ensemble
    mean/sd. This is exactly the input AA-OUTPUT turns into ``FrstIndex`` and the
    AWSEM-format ``tertiary_frustration.dat`` (after the sign flip)."""

    i_key: str
    j_key: str
    native_energy: float
    decoy_mean: float
    decoy_std: float
    n_decoys: int


@dataclass
class EngineResult:
    """The engine's per-structure output: native + decoy per-residue energies,
    from which any contact's native energy and decoy statistics follow.

    Built from parsed logs (:func:`build_engine_result` /
    :func:`load_engine_result_from_logs`) or, on the maintainer path, from
    PyRosetta runs. The contact set itself (which residue pairs are in contact) is
    a geometric/post-processing concern owned by AA-OUTPUT, so the per-contact
    reductions here take the contact list as an argument.
    """

    native_residue_energy: Dict[str, float]
    decoy_residue_energies: List[Dict[str, float]]
    scheme: str = DEFAULT_SCHEME
    n_decoys_requested: int = 0

    def native_contact_energy(self, i_key: str, j_key: str) -> float:
        """``mat_nat[i,j] = ene_res[i] + ene_res[j]`` (the symmetric per-residue
        decomposition; Frust_Post_public.py:176-177,180-181)."""
        nat = self.native_residue_energy
        return nat.get(i_key, 0.0) + nat.get(j_key, 0.0)

    def _good_decoys(self) -> List[Dict[str, float]]:
        """Decoys whose energies are not all zero (the reference's ``temp.sum() !=
        0`` test for a usable sequence; Frust_Post_public.py:192-196)."""
        return [d for d in self.decoy_residue_energies if any(v != 0.0 for v in d.values())]

    @property
    def n_bad(self) -> int:
        return len(self.decoy_residue_energies) - len(self._good_decoys())

    @property
    def n_good(self) -> int:
        return len(self._good_decoys())

    def protein_decoy_statistics(
        self, contacts: Sequence[Tuple[str, str]]
    ) -> Tuple[float, float, int]:
        """Protein-wide decoy mean/sd over a contact set (Frust_Post_public.py
        decoy_stat, lines 205-212).

        Pools the decoy contact energy ``ene_res_decoy[i] + ene_res_decoy[j]`` over
        every contact and every *good* decoy, keeping only non-zero contributions
        (``if mat_all[i,j,k] != 0.0``), then takes the population mean and standard
        deviation (``np.mean`` / ``np.std`` with ``ddof=0``). The reference assigns
        this single protein-wide statistic to every protein residue, so it is the
        decoy mean/sd of every contact.

        Returns:
            ``(mean, std, n_values)`` over the pooled non-zero decoy energies.
        """
        good = self._good_decoys()
        pool: List[float] = []
        for i_key, j_key in contacts:
            for d in good:
                val = d.get(i_key, 0.0) + d.get(j_key, 0.0)
                if val != 0.0:
                    pool.append(val)
        if not pool:
            return float("nan"), float("nan"), 0
        arr = np.array(pool, dtype=float)
        return float(arr.mean()), float(arr.std()), len(pool)

    def summarize_contacts(
        self, contacts: Sequence[Tuple[str, str]]
    ) -> List[ContactEnergySummary]:
        """E3: per-contact ``(native_E, decoy_mean, decoy_sd, n_decoys)`` over a
        contact set.

        ``decoy_mean`` / ``decoy_sd`` are the protein-wide statistic
        (:meth:`protein_decoy_statistics`), matching the reference's per-residue
        assignment; ``n_decoys`` is the number of good decoy sequences. The native
        energy is per contact.
        """
        mean, std, _ = self.protein_decoy_statistics(contacts)
        n_good = self.n_good
        return [
            ContactEnergySummary(
                i_key=i_key,
                j_key=j_key,
                native_energy=self.native_contact_energy(i_key, j_key),
                decoy_mean=mean,
                decoy_std=std,
                n_decoys=n_good,
            )
            for i_key, j_key in contacts
        ]


def build_engine_result(
    native_records: Sequence[ResPairEnergy],
    decoy_records: Sequence[Sequence[ResPairEnergy]],
    scheme: str = DEFAULT_SCHEME,
    fa_rep_cutoff: float = FA_REP_CUTOFF,
) -> EngineResult:
    """Aggregate parsed native + decoy ``ResResE`` records into an
    :class:`EngineResult`.

    Source-agnostic: ``native_records`` / ``decoy_records`` come from
    :func:`parse_resrese_log` (shipped logs, validated here) or from the PyRosetta
    path (maintainer). The aggregation is identical either way.
    """
    nat = residue_energies(native_records, scheme=scheme, fa_rep_cutoff=fa_rep_cutoff)
    decoys = [
        residue_energies(recs, scheme=scheme, fa_rep_cutoff=fa_rep_cutoff)
        for recs in decoy_records
    ]
    return EngineResult(
        native_residue_energy=nat,
        decoy_residue_energies=decoys,
        scheme=scheme,
        n_decoys_requested=len(decoys),
    )


def load_engine_result_from_logs(
    native_log: str,
    decoy_logs: Sequence[str],
    scheme: str = DEFAULT_SCHEME,
    fa_rep_cutoff: float = FA_REP_CUTOFF,
) -> EngineResult:
    """Convenience: parse a native log + a list of decoy logs and aggregate them
    into an :class:`EngineResult`. Used by the in-container parity tests and by the
    maintainer when validating a real Rosetta run against the shipped logs."""
    native_records = parse_resrese_log(native_log)
    decoy_records = [parse_resrese_log(p) for p in decoy_logs]
    return build_engine_result(
        native_records, decoy_records, scheme=scheme, fa_rep_cutoff=fa_rep_cutoff
    )


# ---------------------------------------------------------------------------
# E1/E2 -- the PyRosetta relax/repack path (MAINTAINER step, license-gated).
#
# These reproduce native.xml / test.xml under PyRosetta and return the same
# ResPairEnergy records the parser yields, so the aggregation above is identical
# whether energies come from a shipped log or a live run. They CANNOT be run in
# this container (PyRosetta is not installed); the maintainer validates them by
# diffing against the shipped native.log / *.log per
# docs/atomic/AA_MAINTAINER_RUNBOOK.md. The PyRosetta import is lazy and graceful
# (reusing frustrapy/analysis/mutation_backends.py), so importing this module
# never imports PyRosetta.
# ---------------------------------------------------------------------------

# Rosetta protocol constants, matching native.xml / test.xml exactly.
RELAX_SCRIPT = "rosettacon2018"
RELAX_REPEATS = 2
THREAD_NEIGHBOR_DISTANCE = 10.0
THREAD_PACK_ROUNDS = 5


def _init_pyrosetta():
    """Lazily import + initialise PyRosetta, reusing the mutation-backend plumbing.

    Raises a clear, actionable ``ImportError`` if PyRosetta is missing (the same
    message the mutation backend surfaces). Importing happens inside this function
    so the module top stays PyRosetta-free.
    """
    from ..analysis.mutation_backends import _ensure_pyrosetta

    return _ensure_pyrosetta()


def pyrosetta_available() -> bool:
    """True iff the optional PyRosetta package can be imported (delegates to the
    mutation backend's check)."""
    from ..analysis.mutation_backends import pyrosetta_available as _avail

    return _avail()


def _fixed_backbone_repack(pose, scorefxn, repeats: int = RELAX_REPEATS):
    """Apply FastRelax with the backbone FIXED and side chains repacked only.

    Reproduces native.xml / test.xml: ``RestrictToRepacking``, a MoveMap with
    ``bb=0 chi=1``, ``relaxscript="rosettacon2018"``, ``repeats=2``. The reference
    also prevents repacking outside a ``Neighborhood`` over the whole chain, which
    is a no-op for a single-chain input; for multi-chain inputs the maintainer
    should add the equivalent task operation (see the runbook).
    """
    pyrosetta = _init_pyrosetta()
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking

    move_map = MoveMap()
    move_map.set_bb(False)
    move_map.set_chi(True)

    task_factory = TaskFactory()
    task_factory.push_back(RestrictToRepacking())

    fast_relax = FastRelax(scorefxn, repeats)
    fast_relax.set_movemap(move_map)
    fast_relax.set_task_factory(task_factory)
    # The relaxscript is set via constructor on some PyRosetta builds and via a
    # setter on others; guard so either works.
    if hasattr(fast_relax, "set_script_to_use"):
        fast_relax.set_script_to_use(RELAX_SCRIPT)
    fast_relax.apply(pose)
    return pose


def _extract_pair_energies(pose, scorefxn) -> List[ResPairEnergy]:
    """Read the per-residue-pair (two-body) weighted energies from a scored pose.

    Reproduces ``ScoreCutoffFilter report_residue_pair_energies="1"`` (the
    ``ResResE`` lines): for each interacting residue pair in the energy graph,
    weight each two-body term by the score function's weight and report the per-term
    values plus the row total. The maintainer validates these against native.log.
    """
    pyrosetta = _init_pyrosetta()
    from pyrosetta.rosetta.core.scoring import ScoreType

    scorefxn(pose)  # ensure the pose is scored and the energy graph is current
    energies = pose.energies()
    graph = energies.energy_graph()
    weights = scorefxn.weights()
    info = pose.pdb_info()

    # Map our term names to Rosetta ScoreType enum members. Names match the
    # native.log header. Any term unknown to a given Rosetta build is treated as 0.
    score_types = {}
    for name in TERM_NAMES:
        if name == "total":
            continue
        score_types[name] = getattr(ScoreType, name, None)

    records: List[ResPairEnergy] = []
    n_res = pose.total_residue()
    for i in range(1, n_res + 1):
        for j in range(i + 1, n_res + 1):
            edge = graph.find_energy_edge(i, j)
            if edge is None:
                continue
            emap = edge.fill_energy_map()  # unweighted two-body EnergyMap
            terms: Dict[str, float] = {}
            total = 0.0
            for name in TERM_NAMES:
                if name == "total":
                    continue
                st = score_types[name]
                if st is None:
                    terms[name] = 0.0
                    continue
                weighted = float(emap[st]) * float(weights[st])
                terms[name] = weighted
                total += weighted
            terms["total"] = total
            key1 = f"{info.chain(i)}{info.number(i)}"
            key2 = f"{info.chain(j)}{info.number(j)}"
            records.append(
                ResPairEnergy(
                    res1_key=key1,
                    res2_key=key2,
                    aa1=pose.residue(i).name1(),
                    aa2=pose.residue(j).name1(),
                    terms=terms,
                )
            )
    return records


def compute_native_pair_energies(
    pdb_path: str, scorefxn=None, repeats: int = RELAX_REPEATS
) -> List[ResPairEnergy]:
    """E1 (MAINTAINER): native pose energy via fixed-backbone repack.

    Loads ``pdb_path``, runs the fixed-backbone FastRelax/repack (native.xml), and
    extracts the per-residue-pair energies as :class:`ResPairEnergy` records --
    the PyRosetta equivalent of the reference ``native.log``. The score function
    defaults to Rosetta's full-atom default (ref2015), matching the empty
    ``<SCOREFXNS>`` block in native.xml.

    NOT runnable in this container (PyRosetta license-gated). The maintainer
    validates the output against the shipped ``native.log`` per the runbook.
    """
    pyrosetta = _init_pyrosetta()
    if scorefxn is None:
        scorefxn = pyrosetta.get_score_function()  # ref2015 default
    pose = pyrosetta.pose_from_pdb(pdb_path)
    _fixed_backbone_repack(pose, scorefxn, repeats=repeats)
    return _extract_pair_energies(pose, scorefxn)


def compute_decoy_pair_energies(
    pdb_path: str,
    decoy_seq: str,
    scorefxn=None,
    repeats: int = RELAX_REPEATS,
    start_position: str = "1A",
) -> List[ResPairEnergy]:
    """E2 (MAINTAINER): one decoy's pose energy via threading + fixed-backbone repack.

    Threads ``decoy_seq`` onto the native backbone (the ``SimpleThreadingMover``
    that ``test.xml`` applies before relax; ``pack_neighbors=1 neighbor_dis=10
    pack_rounds=5``), then runs the same fixed-backbone repack as the native pose
    and extracts the per-pair energies -- the PyRosetta equivalent of one decoy
    ``*.log``.

    This is the expensive inner loop. It is a pure function of ``(pdb_path,
    decoy_seq)`` with no shared mutable state beyond the per-process PyRosetta
    singleton, so AA-INTEGRATE can run decoys concurrently under the shared core
    budget (one pool, ``inner = cores // outer``; no nested fork bomb -- see the
    package parallelism conventions). NOT runnable in this container; maintainer
    validates against the shipped ``*.log`` files.
    """
    pyrosetta = _init_pyrosetta()
    if scorefxn is None:
        scorefxn = pyrosetta.get_score_function()
    pose = pyrosetta.pose_from_pdb(pdb_path)

    from pyrosetta.rosetta.protocols.simple_moves import SimpleThreadingMover

    threader = SimpleThreadingMover(start_position, decoy_seq)
    if hasattr(threader, "set_pack_neighbors"):
        threader.set_pack_neighbors(True)
    if hasattr(threader, "set_neighbor_distance"):
        threader.set_neighbor_distance(THREAD_NEIGHBOR_DISTANCE)
    if hasattr(threader, "set_pack_rounds"):
        threader.set_pack_rounds(THREAD_PACK_ROUNDS)
    threader.apply(pose)

    _fixed_backbone_repack(pose, scorefxn, repeats=repeats)
    return _extract_pair_energies(pose, scorefxn)


__all__ = [
    "TERM_NAMES",
    "ENERGY_SCHEMES",
    "DEFAULT_SCHEME",
    "FA_REP_CUTOFF",
    "ResPairEnergy",
    "parse_resrese_log",
    "generate_decoy_sequences",
    "residue_energies",
    "ContactEnergySummary",
    "EngineResult",
    "build_engine_result",
    "load_engine_result_from_logs",
    "pyrosetta_available",
    "compute_native_pair_energies",
    "compute_decoy_pair_energies",
    "RELAX_SCRIPT",
    "RELAX_REPEATS",
]
