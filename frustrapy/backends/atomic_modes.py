"""Mode -> decoy-strategy mapping for the all-atom (Rosetta) Frustratometer (AA-MODES).

This module maps FrustraPy's three AWSEM frustration modes (configurational,
mutational, singleresidue) onto atomic decoy-generation strategies. It parameterizes
**only the decoy-generation step**, exactly as the LAMMPS backend swaps a single
keyword in the coefficient file for the mode: the per-pair energy extraction
(``atomic_engine``) and the post-processor (``atomic_post``) are shared, never forked
per mode.

READ THIS FIRST - parity vs extension (the honesty framing this mission requires):

    The Rosetta reference (Chen et al.) implements EXACTLY ONE decoy scheme:
    composition-preserving PERMUTATION of the whole native sequence on a fixed
    backbone (``RandSeq.py``; audit ``docs/atomic/AA_DESIGN_DECISION.md`` section 3).
    FrustraPy's three AWSEM modes come from THREE DIFFERENT decoy ensembles, which
    do NOT correspond one-to-one to that single scheme.

    * ``configurational`` -> the reference permutation scheme. This is the ONE
      PARITY-BACKED atomic mode: it is the paper's own decoy model, and its
      post-processing half is gated bit-for-bit against the golden fixture
      (``tests/test_atomic_post.py``). Ship it.

    * ``mutational`` and ``singleresidue`` -> EXTENSIONS BEYOND THE PUBLISHED METHOD,
      designed by analogy with the AWSEM definitions (the project conventions), NOT a
      reproduction of any reference. There is no atomic reference for them and no
      parity oracle. They are EXPERIMENTAL. Their decoy-generation logic is
      well-defined and unit-tested here; their end-to-end scoring (PyRosetta relax/
      repack) and the per-site / per-contact aggregation are maintainer-gated and
      UNVALIDATED. Do not let a user mistake either for the published method.

The mode's :attr:`ModeInfo.parity_status` carries this distinction into code; the
docs are in ``docs/atomic/AA_MODES.md``. The cutoffs are unchanged and not the mode's
to set: contacts use ``-1`` / ``0.78`` (``FrstState`` in :mod:`atomic_post`), the
single-residue plot uses ``0.58`` (the project conventions); the ``0.78`` and ``0.58``
splits are intentional and must never be collapsed.

The AWSEM definitions this maps onto (the project conventions): the three indices share
ONE energy model and ONE Z-score and differ ONLY in what the decoy ensemble
randomizes -- configurational randomizes identities + geometry + density,
mutational randomizes only the identities of the contacting pair i,j (geometry
frozen), singleresidue randomizes only the identity at site i. The atomic backend
holds geometry fixed by construction (fixed backbone, side-chain repack only), so
"geometry" never varies on the atomic path; the atomic modes therefore differ only
in WHICH sequence positions are re-identified, which is exactly what these strategies
encode.

This module imports nothing from PyRosetta and only the pure helpers from
``atomic_engine``; importing it never imports PyRosetta, so ``import frustrapy`` and
the default LAMMPS path are unaffected.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .atomic_engine import generate_decoy_sequences

# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------

#: The 20 standard one-letter amino-acid identities, in fixed alphabetical order so
#: an exhaustive identity scan is deterministic (no RNG needed). Used as the decoy
#: identity alphabet for the experimental mutational / singleresidue modes.
ATOMIC_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWY"

#: Parity-status labels (kept as constants so call sites and tests never typo them).
PARITY_BACKED = "parity-backed"
EXPERIMENTAL = "experimental"

#: The three FrustraPy modes the atomic backend recognizes (same names as the
#: LAMMPS/AWSEM path).
CONFIGURATIONAL = "configurational"
MUTATIONAL = "mutational"
SINGLERESIDUE = "singleresidue"


@dataclass(frozen=True)
class ModeInfo:
    """Static description of one atomic mode.

    ``parity_status`` is the load-bearing field: :data:`PARITY_BACKED` only for
    ``configurational`` (the reference's own permutation scheme, post-processing
    gated against the golden fixture); :data:`EXPERIMENTAL` for the two extension
    modes. ``randomizes`` states the decoy randomization domain; ``table`` names the
    downstream output table shape the shared post-processor produces.
    """

    mode: str
    parity_status: str
    randomizes: str
    table: str
    note: str

    @property
    def is_parity_backed(self) -> bool:
        return self.parity_status == PARITY_BACKED

    @property
    def is_experimental(self) -> bool:
        return self.parity_status == EXPERIMENTAL


#: The mode registry. ``configurational`` is the single parity-backed atomic mode
#: (the published permutation scheme); the other two are clearly-labeled extensions.
MODES: Dict[str, ModeInfo] = {
    CONFIGURATIONAL: ModeInfo(
        mode=CONFIGURATIONAL,
        parity_status=PARITY_BACKED,
        randomizes=(
            "the whole sequence at once: each decoy is a composition-preserving "
            "permutation of the native sequence, threaded onto the fixed native "
            "backbone and repacked"
        ),
        table="14-column contact table (Res1 Res2 ... FrstIndex Welltype FrstState)",
        note=(
            "The reference's single decoy scheme (RandSeq.py). PARITY-BACKED: the "
            "post-processing half is gated bit-for-bit against docs/atomic/golden/. "
            "Pools one protein-wide decoy mean/sd over all contacts (the reference "
            "decoy_stat)."
        ),
    ),
    MUTATIONAL: ModeInfo(
        mode=MUTATIONAL,
        parity_status=EXPERIMENTAL,
        randomizes=(
            "only the identities of the two contacting residues i,j for each contact; "
            "every other position stays native, the backbone stays fixed"
        ),
        table="14-column contact table (per-contact decoy statistic)",
        note=(
            "EXTENSION BEYOND THE PAPER, by analogy with AWSEM mutational. No atomic "
            "reference, no parity oracle. The decoy ensemble is per-contact: a "
            "separate identity scan/sample over (i, j). EXPERIMENTAL."
        ),
    ),
    SINGLERESIDUE: ModeInfo(
        mode=SINGLERESIDUE,
        parity_status=EXPERIMENTAL,
        randomizes=(
            "only the identity at site i for each site; every other position stays "
            "native, the backbone stays fixed"
        ),
        table="8-column single-residue table (Res ChainRes DensityRes AA ... FrstIndex)",
        note=(
            "EXTENSION BEYOND THE PAPER, by analogy with AWSEM singleresidue. No "
            "atomic reference, no parity oracle. The decoy ensemble is per-site: a "
            "separate identity scan/sample at site i. EXPERIMENTAL. The single-residue "
            "table carries no FrstState; classification is plot-only at the 0.58 "
            "cutoff (do not collapse it with the 0.78 contact cutoff)."
        ),
    ),
}


def mode_info(mode: str) -> ModeInfo:
    """Return the :class:`ModeInfo` for ``mode`` or raise a clear error."""
    try:
        return MODES[mode]
    except KeyError:
        raise ValueError(
            f"unknown atomic mode {mode!r}; expected one of {sorted(MODES)}"
        )


def is_experimental_mode(mode: str) -> bool:
    """True iff ``mode`` is an EXPERIMENTAL extension (mutational / singleresidue)."""
    return mode_info(mode).is_experimental


# ---------------------------------------------------------------------------
# Decoy specification model (mode-agnostic; the engine consumes these uniformly).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecoySpec:
    """One decoy to be threaded onto the fixed native backbone and scored.

    ``sequence`` is the full one-letter sequence (every mode threads a whole
    sequence; the modes differ only in how it was perturbed). ``varied_positions``
    is the decoy's randomization DOMAIN -- the 0-based positions the mode is allowed
    to re-identify. Every position OUTSIDE ``varied_positions`` is guaranteed equal
    to the native identity; positions inside it may or may not differ from native
    (an exhaustive identity scan includes the native identity as one decoy).
    """

    sequence: str
    varied_positions: Tuple[int, ...]


@dataclass(frozen=True)
class DecoyGroup:
    """A decoy ensemble whose energies are pooled into ONE mean/sd at aggregation.

    ``key`` identifies the group and is the unit the downstream statistic is keyed
    to: ``None`` for the protein-wide configurational ensemble, an ``int`` site index
    for singleresidue, an ``(i, j)`` index pair for a mutational contact.
    """

    key: Optional[object]
    specs: Tuple[DecoySpec, ...]


def _thread_identities(native_seq: str, substitutions: Dict[int, str]) -> str:
    """Return ``native_seq`` with ``substitutions`` (0-based position -> identity)
    applied; all other positions untouched."""
    chars = list(native_seq)
    for pos, aa in substitutions.items():
        chars[pos] = aa
    return "".join(chars)


def positions_differing_from_native(native_seq: str, decoy_seq: str) -> Tuple[int, ...]:
    """0-based positions where ``decoy_seq`` differs from ``native_seq``."""
    if len(native_seq) != len(decoy_seq):
        raise ValueError("native and decoy sequences must be the same length")
    return tuple(i for i, (a, b) in enumerate(zip(native_seq, decoy_seq)) if a != b)


def spec_respects_domain(native_seq: str, spec: DecoySpec) -> bool:
    """True iff ``spec`` changes native ONLY within its ``varied_positions`` domain.

    The core correctness invariant for every mode: a mode must never perturb a
    position outside the randomization domain it claims. Used by the unit tests and
    cheap enough to assert at generation time.
    """
    allowed = set(spec.varied_positions)
    for pos in positions_differing_from_native(native_seq, spec.sequence):
        if pos not in allowed:
            return False
    return True


# ---------------------------------------------------------------------------
# Decoy strategies -- one per mode. The strategy is the ONLY thing the mode
# parameterizes; everything downstream (scoring, aggregation math, writing) is
# shared.
# ---------------------------------------------------------------------------

class DecoyStrategy(ABC):
    """A mode's decoy-generation strategy.

    Subclasses implement :meth:`generate`, returning the mode's decoy ensemble as a
    list of :class:`DecoyGroup`. The engine threads + scores every
    :class:`DecoySpec` identically regardless of mode (the shared per-pair
    extraction); the grouping tells the post-processor how to pool the decoy
    energies (protein-wide, per-site, or per-contact).
    """

    #: The mode name this strategy serves.
    mode: str = ""

    @property
    def info(self) -> ModeInfo:
        return mode_info(self.mode)

    @property
    def parity_status(self) -> str:
        return self.info.parity_status

    @abstractmethod
    def generate(self, native_seq: str, *, seed: Optional[int] = None, **kwargs) -> List[DecoyGroup]:
        """Return the mode's decoy ensemble for ``native_seq``."""
        raise NotImplementedError


class ConfigurationalDecoyStrategy(DecoyStrategy):
    """PARITY-BACKED. The reference permutation scheme: each decoy is a
    composition-preserving permutation of the whole native sequence (randomization
    domain = every position). Reuses :func:`atomic_engine.generate_decoy_sequences`
    verbatim, so it is the SAME decoy model the golden fixture validates. Produces a
    single protein-wide :class:`DecoyGroup` (``key=None``), which the engine pools
    into the one protein-wide decoy mean/sd the reference uses.
    """

    mode = CONFIGURATIONAL

    def generate(
        self, native_seq: str, *, n_decoys: int = 0, seed: Optional[int] = None, **kwargs
    ) -> List[DecoyGroup]:
        seqs = generate_decoy_sequences(native_seq, n_decoys, seed=seed)
        all_positions = tuple(range(len(native_seq)))
        specs = tuple(DecoySpec(sequence=s, varied_positions=all_positions) for s in seqs)
        return [DecoyGroup(key=None, specs=specs)]


class SingleResidueDecoyStrategy(DecoyStrategy):
    """EXPERIMENTAL (extension beyond the paper). By analogy with AWSEM
    singleresidue: for each site i, the decoys re-identify ONLY site i (domain =
    {i}); every other position stays native. Produces one :class:`DecoyGroup` per
    site, keyed by the site index.

    Default ensemble: the exhaustive 20-identity scan at the site (deterministic, no
    RNG, the most reproducible choice and the closest atomic analogue of AWSEM's
    exhaustive single-site identity randomization). Passing ``n_decoys_per_site``
    samples that many identities instead (seeded for reproducibility).
    """

    mode = SINGLERESIDUE

    def generate(
        self,
        native_seq: str,
        *,
        sites: Optional[Sequence[int]] = None,
        alphabet: str = ATOMIC_ALPHABET,
        n_decoys_per_site: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[DecoyGroup]:
        target_sites = range(len(native_seq)) if sites is None else sites
        rng = random.Random(seed)
        groups: List[DecoyGroup] = []
        for i in target_sites:
            if not 0 <= i < len(native_seq):
                raise ValueError(f"site index {i} out of range for length {len(native_seq)}")
            identities = _select_identities(alphabet, n_decoys_per_site, rng)
            specs = tuple(
                DecoySpec(
                    sequence=_thread_identities(native_seq, {i: aa}),
                    varied_positions=(i,),
                )
                for aa in identities
            )
            groups.append(DecoyGroup(key=i, specs=specs))
        return groups


class MutationalDecoyStrategy(DecoyStrategy):
    """EXPERIMENTAL (extension beyond the paper). By analogy with AWSEM mutational:
    for each contact (i, j), the decoys re-identify ONLY positions i and j (domain =
    {i, j}); every other position stays native, the geometry stays frozen. Produces
    one :class:`DecoyGroup` per contact, keyed by the ``(i, j)`` index pair.

    Default ensemble: the exhaustive identity-PAIR scan over the contacting sites
    (``len(alphabet)**2`` decoys, deterministic, the closest analogue of AWSEM's
    exhaustive pairwise identity randomization). Passing ``n_decoys_per_contact``
    samples that many identity pairs instead (seeded). Contacts are 0-based index
    pairs, the same indices :func:`atomic_post.select_contacts` returns.
    """

    mode = MUTATIONAL

    def generate(
        self,
        native_seq: str,
        *,
        contacts: Optional[Sequence[Tuple[int, int]]] = None,
        alphabet: str = ATOMIC_ALPHABET,
        n_decoys_per_contact: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[DecoyGroup]:
        if contacts is None:
            raise ValueError(
                "mutational decoys are per-contact; pass contacts=[(i, j), ...] "
                "(0-based index pairs from atomic_post.select_contacts)"
            )
        rng = random.Random(seed)
        groups: List[DecoyGroup] = []
        n = len(native_seq)
        for (i, j) in contacts:
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError(f"contact ({i}, {j}) out of range for length {n}")
            pairs = _select_identity_pairs(alphabet, n_decoys_per_contact, rng)
            specs = tuple(
                DecoySpec(
                    sequence=_thread_identities(native_seq, {i: ai, j: aj}),
                    varied_positions=(i, j),
                )
                for (ai, aj) in pairs
            )
            groups.append(DecoyGroup(key=(i, j), specs=specs))
        return groups


def _select_identities(
    alphabet: str, n: Optional[int], rng: random.Random
) -> List[str]:
    """Exhaustive identity list (``n is None``) or ``n`` sampled identities."""
    letters = list(alphabet)
    if n is None:
        return letters
    if n < 0:
        raise ValueError(f"decoy count must be non-negative, got {n}")
    return [rng.choice(letters) for _ in range(n)]


def _select_identity_pairs(
    alphabet: str, n: Optional[int], rng: random.Random
) -> List[Tuple[str, str]]:
    """Exhaustive identity-pair grid (``n is None``) or ``n`` sampled pairs."""
    letters = list(alphabet)
    if n is None:
        return [(a, b) for a in letters for b in letters]
    if n < 0:
        raise ValueError(f"decoy count must be non-negative, got {n}")
    return [(rng.choice(letters), rng.choice(letters)) for _ in range(n)]


#: Strategy instances, one per mode. Stateless, so module-level singletons are safe.
_STRATEGIES: Dict[str, DecoyStrategy] = {
    CONFIGURATIONAL: ConfigurationalDecoyStrategy(),
    SINGLERESIDUE: SingleResidueDecoyStrategy(),
    MUTATIONAL: MutationalDecoyStrategy(),
}


def get_decoy_strategy(mode: str) -> DecoyStrategy:
    """Resolve a mode name to its :class:`DecoyStrategy` (the LAMMPS-keyword-swap
    analogue: one selector picks the decoy step, everything downstream is shared)."""
    try:
        return _STRATEGIES[mode]
    except KeyError:
        raise ValueError(
            f"unknown atomic mode {mode!r}; expected one of {sorted(_STRATEGIES)}"
        )


# ---------------------------------------------------------------------------
# Shared aggregation seam: score every decoy spec with the SAME per-pair
# extraction (Rosetta-gated, injected as a callable so it is mockable), then pool
# each group's energies into one mean/sd. This is the only downstream step the
# modes share a signature for; the scoring callable is identical across modes,
# mirroring how the AWSEM binary runs the same energy model for every mode.
# ---------------------------------------------------------------------------

#: A group's pooled decoy statistic: ``(mean, std, n_values)`` over its non-zero
#: decoy energies (population std, ``ddof=0``).
GroupStatistic = Tuple[float, float, int]

#: The per-spec scoring callable: ``score_fn(spec, group) -> energy``. On the live
#: path this wraps the engine's Rosetta thread+repack+extract (license-gated); in
#: tests it is mocked. It is the SAME callable for every mode (only the decoy specs
#: differ), which is the whole point of parameterizing only the decoy step.
ScoreFn = Callable[["DecoySpec", "DecoyGroup"], float]


def pooled_statistics(values: Sequence[float]) -> GroupStatistic:
    """Population mean/std (``ddof=0``) over the NON-ZERO values, with ``n``.

    Mirrors the reference decoy pooling exactly
    (:meth:`atomic_engine.EngineResult.protein_decoy_statistics`: keep
    ``!= 0`` values, ``np.mean`` / ``np.std`` with ``ddof=0``). For the
    PARITY-BACKED configurational mode this is the protein-wide statistic the golden
    fixture validates; applied PER-SITE or PER-CONTACT (the experimental modes) the
    same math has no parity oracle.
    """
    pool = [v for v in values if v != 0.0]
    if not pool:
        return float("nan"), float("nan"), 0
    arr = np.asarray(pool, dtype=float)
    return float(arr.mean()), float(arr.std()), len(pool)


def aggregate_groups(
    groups: Sequence[DecoyGroup], score_fn: ScoreFn
) -> "Dict[Optional[object], GroupStatistic]":
    """Score every spec in every group with the shared ``score_fn`` and pool each
    group's energies into a :data:`GroupStatistic`, keyed by ``group.key``.

    The scoring callable is identical for all modes; the modes differ only in the
    decoy specs the strategy produced (``configurational`` -> one ``None``-keyed
    group, ``singleresidue`` -> one group per site index, ``mutational`` -> one group
    per ``(i, j)`` contact). The caller turns the returned per-group statistics into
    the post-processor's summaries (protein-wide for configurational; per-contact /
    per-site for the experimental modes).
    """
    out: Dict[Optional[object], GroupStatistic] = {}
    for group in groups:
        energies = [score_fn(spec, group) for spec in group.specs]
        out[group.key] = pooled_statistics(energies)
    return out


__all__ = [
    "ATOMIC_ALPHABET",
    "PARITY_BACKED",
    "EXPERIMENTAL",
    "CONFIGURATIONAL",
    "MUTATIONAL",
    "SINGLERESIDUE",
    "ModeInfo",
    "MODES",
    "mode_info",
    "is_experimental_mode",
    "DecoySpec",
    "DecoyGroup",
    "positions_differing_from_native",
    "spec_respects_domain",
    "DecoyStrategy",
    "ConfigurationalDecoyStrategy",
    "SingleResidueDecoyStrategy",
    "MutationalDecoyStrategy",
    "get_decoy_strategy",
    "GroupStatistic",
    "ScoreFn",
    "pooled_statistics",
    "aggregate_groups",
]
