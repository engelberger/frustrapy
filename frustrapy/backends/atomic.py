"""The all-atom (Rosetta) frustration backend (AA-INTEGRATE).

``AtomicBackend`` makes the all-atom Frustratometer a first-class, selectable
energy backend that flows through the existing public API exactly like ``lammps``
and ``native``. It does NOT reimplement the energy model: the energy lives in
Rosetta (ref2015), driven through PyRosetta by :mod:`.atomic_engine`; the decoy
ensemble per mode comes from :mod:`.atomic_modes`; and the AWSEM-format
``tertiary_frustration.dat`` is written by :mod:`.atomic_post`, so the shared
:meth:`process_results` / :meth:`compute_density` post-processing, the canonical
tables, and the Plotly/py3Dmol visualizations all run unchanged.

This mirrors the parity spine: the energy model is the swappable part, while
parsing, the threshold classification, and the 5 Angstrom density kernel are common
glue. Two facts make this backend different from ``lammps``/``native`` in HOW it is
prepared, not in WHAT it produces:

* **No LAMMPS deck.** The Rosetta path reads the cleaned ``{base}.pdb`` (and the
  ``{base}.pdb_equivalences.txt`` residue map) the calculator writes anyway; it does
  not read ``fix_backbone_coeff.data`` / ``gamma.dat`` / the ``.in`` deck. So this
  backend sets :attr:`requires_lammps_prep` ``= False`` and the calculator skips the
  ``PdbCoords2Lammps.sh`` subprocess prep. The on-disk output contract is identical.

* **PyRosetta is license-gated and not in this container.** The relax/repack half
  (native pose + decoy poses) needs PyRosetta, imported lazily and only when
  :meth:`compute_energies` actually runs. Importing this module, ``import frustrapy``,
  and the default ``lammps`` path never import PyRosetta. A missing PyRosetta raises a
  clear, actionable error at compute time, never at import time. The post-processing
  half (geometry, contact selection, the sign-flipped ``FrstIndex``, the writer) needs
  no Rosetta and is parity-gated against the golden fixture (``tests/test_atomic_post.py``).

Modes (see :mod:`.atomic_modes`): ``configurational`` is the reference's own
permutation decoy scheme and is the one PARITY-BACKED atomic mode; ``mutational`` and
``singleresidue`` are EXPERIMENTAL extensions beyond the published method (no atomic
reference, no parity oracle). All three share one energy model and one Z-score and
differ only in the decoy ensemble, exactly like the AWSEM modes.

Parallelism (G3): the expensive axis is the decoy loop (one Rosetta thread+repack per
decoy). :func:`score_sequences` parallelizes it under the package's SHARED core budget
(:mod:`frustrapy.utils.concurrency`): one pool, sized ``min(requested, cores,
n_tasks)``, and -- when this backend already runs inside an outer pool worker
(``dir_frustration(n_procs=K)`` or the mutation scan) -- it falls back to serial so the
nested pools never multiply past the core budget (``outer * 1 <= cores``; the same rule
:mod:`.native` uses for its thread count). The native pose energy is computed ONCE per
structure and reused across every contact, mirroring the prep-amortization lever.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence

from .base import FrustrationBackend

if TYPE_CHECKING:
    from ..core import Pdb
    from ..analysis.frustration_calculator import FrustrationCalculator
    from .atomic_engine import ResPairEnergy

# Defaults for the decoy ensemble. The paper recommends >= 200 permutation decoys for
# convergence (the shipped demo used 50). Overridable per run via calculator attributes
# (``atomic_*``) or the matching ``FRUSTRAPY_ATOMIC_*`` environment variables.
DEFAULT_N_DECOYS = 200


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


@dataclass(frozen=True)
class AtomicOptions:
    """Resolved per-run atomic options.

    Read from the owning calculator's optional ``atomic_*`` attributes, then the
    matching ``FRUSTRAPY_ATOMIC_*`` environment variables, then documented defaults.
    Nothing here changes the LAMMPS/native paths; these only parameterize the Rosetta
    decoy step and the atomic contact geometry.
    """

    n_decoys: int
    seed: Optional[int]
    seq_sep: int
    scheme: str
    repeats: int
    n_procs: Optional[int]
    distance_cutoff: float
    n_decoys_per_site: Optional[int]
    n_decoys_per_contact: Optional[int]

    @classmethod
    def from_calculator(cls, calculator: "FrustrationCalculator") -> "AtomicOptions":
        from .atomic_engine import RELAX_REPEATS  # noqa: PLC0415
        from .atomic_post import CONTACT_DISTANCE_CUTOFF, DEFAULT_SEQ_SEP  # noqa: PLC0415
        from .atomic_engine import DEFAULT_SCHEME  # noqa: PLC0415

        def attr(name, default):
            return getattr(calculator, f"atomic_{name}", None) or default

        return cls(
            n_decoys=int(attr("n_decoys", _env_int("FRUSTRAPY_ATOMIC_N_DECOYS", DEFAULT_N_DECOYS))),
            seed=getattr(calculator, "atomic_seed", None)
            if getattr(calculator, "atomic_seed", None) is not None
            else _env_int("FRUSTRAPY_ATOMIC_SEED", None),
            # The atomic contact definition follows the reference (representative-atom
            # distance <= cutoff and |i-j| > seq_sep), not the AWSEM seq_dist; the
            # parity gate (the golden fixture) uses the reference sep = 9.
            seq_sep=int(attr("seq_sep", DEFAULT_SEQ_SEP)),
            scheme=str(getattr(calculator, "atomic_scheme", None) or DEFAULT_SCHEME),
            repeats=int(attr("repeats", RELAX_REPEATS)),
            n_procs=getattr(calculator, "atomic_n_procs", None) or getattr(calculator, "n_cpus", None),
            distance_cutoff=float(attr("distance_cutoff", CONTACT_DISTANCE_CUTOFF)),
            n_decoys_per_site=getattr(calculator, "atomic_n_decoys_per_site", None),
            n_decoys_per_contact=getattr(calculator, "atomic_n_decoys_per_contact", None),
        )


def _resolve_decoy_workers(n_tasks: int, n_procs: Optional[int] = None) -> int:
    """Worker count for the decoy pool, composing safely with outer parallelism.

    The decoy scoring is independent per decoy, so the only constraint is not
    oversubscribing cores. Mirrors :func:`frustrapy.backends.native._resolve_native_threads`:

    * inside an outer pool worker (``multiprocessing.parent_process()`` is not ``None``)
      the outer pool already saturates the cores, so score serially (1). With the outer
      pool capped at ``<= cores`` (the shared budget) this keeps ``outer * 1 <= cores`` --
      no ``cores**2`` fork bomb.
    * in the main process, use :func:`resolve_pool_size` (``min(requested, cores,
      n_tasks)``) so one pool never exceeds the core budget or the work available.
    """
    import multiprocessing  # noqa: PLC0415

    if n_tasks <= 1:
        return 1
    if multiprocessing.parent_process() is not None:
        return 1
    from ..utils.concurrency import resolve_pool_size  # noqa: PLC0415

    return resolve_pool_size(n_procs, n_tasks)


# Module-level worker so the decoy pool payload is picklable (the scorer is a
# function of two strings; ResPairEnergy is a frozen dataclass of floats). Stashed in a
# module global by score_sequences before the pool runs, matching how the package's
# other pools pass a read-only callable to workers.
_SCORE_WORKER_FN: Optional[Callable] = None
_SCORE_WORKER_PDB: Optional[str] = None
_SCORE_WORKER_REPEATS: int = 2


def _score_worker(decoy_seq: str):
    """Pool worker: score one decoy sequence with the configured scorer."""
    return _SCORE_WORKER_FN(_SCORE_WORKER_PDB, decoy_seq, repeats=_SCORE_WORKER_REPEATS)


def _score_pool_init(scorer, pdb_path, repeats):
    global _SCORE_WORKER_FN, _SCORE_WORKER_PDB, _SCORE_WORKER_REPEATS
    _SCORE_WORKER_FN = scorer
    _SCORE_WORKER_PDB = pdb_path
    _SCORE_WORKER_REPEATS = repeats
    from ..utils.concurrency import pool_worker_initializer  # noqa: PLC0415

    pool_worker_initializer()


def score_sequences(
    pdb_path: str,
    sequences: Sequence[str],
    *,
    scorer: Optional[Callable[..., "List[ResPairEnergy]"]] = None,
    repeats: int = 2,
    n_procs: Optional[int] = None,
) -> "List[List[ResPairEnergy]]":
    """Score every decoy ``sequence`` (thread onto the fixed backbone, repack, extract
    the per-residue-pair energies), returning one :class:`ResPairEnergy` list per
    sequence in input order.

    This is the parallel decoy loop (G3). It runs under the shared core budget
    (:func:`_resolve_decoy_workers`): one pool, never nested past the budget. The
    ``scorer`` is the per-sequence energy callable -- by default
    :func:`frustrapy.backends.atomic_engine.compute_decoy_pair_energies` (PyRosetta,
    license-gated). It is injectable so the parallel-safety and aggregation can be
    tested without Rosetta. The scorer must be a top-level (picklable) function with
    signature ``scorer(pdb_path, sequence, repeats=...) -> List[ResPairEnergy]``.
    """
    if scorer is None:
        from .atomic_engine import compute_decoy_pair_energies  # noqa: PLC0415

        scorer = compute_decoy_pair_energies
    sequences = list(sequences)
    if not sequences:
        return []
    workers = _resolve_decoy_workers(len(sequences), n_procs)
    if workers <= 1:
        return [scorer(pdb_path, seq, repeats=repeats) for seq in sequences]

    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    from ..utils.concurrency import get_pool_context  # noqa: PLC0415

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_pool_context(),
        initializer=_score_pool_init,
        initargs=(scorer, pdb_path, repeats),
    ) as ex:
        return list(ex.map(_score_worker, sequences))


class AtomicBackend(FrustrationBackend):
    """All-atom (Rosetta) energy backend (AA-INTEGRATE).

    Produces ``tertiary_frustration.dat`` in the same AWSEM column layout the LAMMPS
    binary emits, so the shared post-processing, tables, density, and viz run
    unchanged. The energy model is Rosetta (ref2015) via PyRosetta (lazy, license-gated);
    the decoy strategy per mode and the post-processor are the in-container-validated
    halves.
    """

    name = "atomic"
    requires_lammps_prep = False

    def compute_energies(self, calculator: "FrustrationCalculator", pdb: "Pdb") -> None:
        """Drive the Rosetta engine for the requested mode and write
        ``tertiary_frustration.dat`` in the job directory.

        The native pose energy is computed ONCE (amortized across every contact); the
        decoy loop is parallel-safe under the shared budget. Raises a clear error (via
        the lazy PyRosetta import) if PyRosetta is unavailable.
        """
        from .atomic_post import load_contact_geometry  # noqa: PLC0415

        job_dir = pdb.job_dir
        pdb_path = os.path.join(job_dir, f"{pdb.pdb_base}.pdb")
        out_path = os.path.join(job_dir, "tertiary_frustration.dat")
        mode = calculator.mode
        opts = AtomicOptions.from_calculator(calculator)

        # Geometry / contact set / sequence are pure (no Rosetta).
        geom = load_contact_geometry(pdb_path, pdb_code=pdb.pdb_base)

        if mode == "configurational":
            self._compute_configurational(pdb_path, geom, out_path, opts)
        elif mode == "mutational":
            self._compute_mutational(pdb_path, geom, out_path, opts)
        elif mode == "singleresidue":
            self._compute_singleresidue(pdb_path, geom, out_path, opts)
        else:
            raise ValueError(
                f"atomic backend does not support mode {mode!r}; "
                "expected configurational, mutational, or singleresidue"
            )

    # -- mode implementations -------------------------------------------------

    def _compute_configurational(self, pdb_path, geom, out_path, opts: AtomicOptions) -> None:
        """PARITY-BACKED. The reference permutation scheme: one protein-wide decoy
        mean/sd over the contact set. Native pose scored once; permutation decoys
        scored in parallel."""
        from .atomic_engine import compute_native_pair_energies, build_engine_result  # noqa: PLC0415
        from .atomic_modes import get_decoy_strategy  # noqa: PLC0415
        from .atomic_post import write_tertiary_frustration  # noqa: PLC0415

        native_seq = "".join(geom.aa)
        native_records = compute_native_pair_energies(pdb_path, repeats=opts.repeats)
        groups = get_decoy_strategy("configurational").generate(
            native_seq, n_decoys=opts.n_decoys, seed=opts.seed
        )
        # Configurational is one protein-wide group: every spec is a full-sequence
        # permutation; pool all decoy records into the single mean/sd.
        decoy_seqs = [spec.sequence for group in groups for spec in group.specs]
        decoy_records = score_sequences(
            pdb_path, decoy_seqs, repeats=opts.repeats, n_procs=opts.n_procs
        )
        engine_result = build_engine_result(
            native_records, decoy_records, scheme=opts.scheme
        )
        write_tertiary_frustration(
            out_path,
            geom,
            engine_result=engine_result,
            seq_sep=opts.seq_sep,
            distance_cutoff=opts.distance_cutoff,
        )

    def _compute_mutational(self, pdb_path, geom, out_path, opts: AtomicOptions) -> None:
        """EXPERIMENTAL. Per-contact decoy ensemble (re-identify only the contacting
        pair). No atomic reference / no parity oracle."""
        from .atomic_engine import (  # noqa: PLC0415
            ContactEnergySummary,
            compute_native_pair_energies,
            residue_energies,
        )
        from .atomic_modes import get_decoy_strategy  # noqa: PLC0415
        from .atomic_post import select_contacts, write_tertiary_frustration  # noqa: PLC0415

        native_seq = "".join(geom.aa)
        native_records = compute_native_pair_energies(pdb_path, repeats=opts.repeats)
        native_res_e = residue_energies(native_records, scheme=opts.scheme)
        contacts = select_contacts(
            geom, seq_sep=opts.seq_sep, distance_cutoff=opts.distance_cutoff
        )
        groups = get_decoy_strategy("mutational").generate(
            native_seq,
            contacts=contacts,
            n_decoys_per_contact=opts.n_decoys_per_contact,
            seed=opts.seed,
        )
        stats = self._aggregate_groups_parallel(pdb_path, geom, groups, opts)
        summaries = []
        for (i, j) in contacts:
            mean, std, _ = stats[(i, j)]
            i_key, j_key = geom.cid_list[i], geom.cid_list[j]
            native_e = native_res_e.get(i_key, 0.0) + native_res_e.get(j_key, 0.0)
            summaries.append(
                ContactEnergySummary(
                    i_key=i_key, j_key=j_key, native_energy=native_e,
                    decoy_mean=mean, decoy_std=std, n_decoys=len(groups),
                )
            )
        write_tertiary_frustration(
            out_path,
            geom,
            summaries=summaries,
            seq_sep=opts.seq_sep,
            distance_cutoff=opts.distance_cutoff,
        )

    def _compute_singleresidue(self, pdb_path, geom, out_path, opts: AtomicOptions) -> None:
        """EXPERIMENTAL. Per-site decoy ensemble (re-identify only site i). No atomic
        reference / no parity oracle. Writes the 8-column single-residue table."""
        from .atomic_engine import compute_native_pair_energies, residue_energies  # noqa: PLC0415
        from .atomic_modes import get_decoy_strategy  # noqa: PLC0415
        from .atomic_post import SiteEnergySummary, write_singleresidue_dat  # noqa: PLC0415

        native_seq = "".join(geom.aa)
        native_records = compute_native_pair_energies(pdb_path, repeats=opts.repeats)
        native_res_e = residue_energies(native_records, scheme=opts.scheme)
        groups = get_decoy_strategy("singleresidue").generate(
            native_seq, n_decoys_per_site=opts.n_decoys_per_site, seed=opts.seed
        )
        stats = self._aggregate_groups_parallel(pdb_path, geom, groups, opts)
        site_summaries = []
        for group in groups:
            i = group.key
            mean, std, _ = stats[i]
            native_e = native_res_e.get(geom.cid_list[i], 0.0)
            site_summaries.append(
                SiteEnergySummary(
                    site_index=i, native_energy=native_e, decoy_mean=mean, decoy_std=std,
                )
            )
        write_singleresidue_dat(out_path, geom, site_summaries)

    # -- shared experimental-mode aggregation --------------------------------

    def _aggregate_groups_parallel(self, pdb_path, geom, groups, opts: AtomicOptions):
        """Score every decoy spec across all groups in ONE parallel pass, then pool
        each group's energies into ``(mean, std, n)`` keyed by ``group.key``.

        Used by the experimental per-contact / per-site modes. Each decoy sequence is
        scored once (the parallel decoy loop); the per-group statistic is the pooled
        contact (i+j) or site (i) energy over that group's decoys. The native pose is
        already amortized by the caller (computed once, passed as native residue
        energies). EXPERIMENTAL: no parity oracle.
        """
        from .atomic_engine import residue_energies  # noqa: PLC0415
        from .atomic_modes import pooled_statistics  # noqa: PLC0415

        # Score each spec sequence once; cache by sequence to avoid rescoring repeats.
        all_specs = [spec for group in groups for spec in group.specs]
        unique_seqs = list(dict.fromkeys(spec.sequence for spec in all_specs))
        records = score_sequences(
            pdb_path, unique_seqs, repeats=opts.repeats, n_procs=opts.n_procs
        )
        res_e_by_seq = {
            seq: residue_energies(recs, scheme=opts.scheme)
            for seq, recs in zip(unique_seqs, records)
        }

        stats = {}
        for group in groups:
            energies = []
            for spec in group.specs:
                ene = res_e_by_seq[spec.sequence]
                if isinstance(group.key, tuple):  # mutational contact (i, j)
                    i, j = group.key
                    energies.append(
                        ene.get(geom.cid_list[i], 0.0) + ene.get(geom.cid_list[j], 0.0)
                    )
                else:  # singleresidue site index i
                    energies.append(ene.get(geom.cid_list[group.key], 0.0))
            stats[group.key] = pooled_statistics(energies)
        return stats


__all__ = [
    "AtomicBackend",
    "AtomicOptions",
    "DEFAULT_N_DECOYS",
    "score_sequences",
]
