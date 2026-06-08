"""The tmol-based all-atom frustration backend (TMOL-PY-BACKEND, #38).

``AtomicTmolBackend`` is the license-clean (Apache-2.0) sibling of the PyRosetta
``AtomicBackend``. It computes the all-atom frustration indices with the SAME design,
the SAME decoy schemes, the SAME post-processor, and the SAME sign convention as the
AA lane; it differs ONLY in the energy engine: per-residue-pair ref2015 energies come
from **tmol** (:mod:`.atomic_tmol_engine`) instead of Rosetta via PyRosetta. This is
the gate's recommended role (``docs/tmol/TMOL_LANE_DECISION.md`` section 3): tmol
replaces the PyRosetta engine half of the atomic backend for the academic tier.

It reuses the AA lane verbatim (no fork):

* :mod:`.atomic_post` -- the representative-atom contact geometry, contact selection,
  the sign-flipped ``FrstIndex``, the AWSEM-format ``tertiary_frustration.dat`` writer
  (gated bit-for-bit against ``docs/atomic/golden/`` by ``tests/test_atomic_post.py``);
* :mod:`.atomic_modes` -- the per-mode decoy strategies (configurational =
  parity-backed permutation; mutational / singleresidue = experimental extensions);
* :mod:`.atomic_engine` -- :class:`~frustrapy.backends.atomic_engine.EngineResult` and
  :class:`~frustrapy.backends.atomic_engine.ContactEnergySummary` for the per-contact
  decoy-statistics aggregation.

Two facts make it different from ``lammps`` / ``native`` in HOW it is prepared, not in
WHAT it produces (identical to ``AtomicBackend``):

* **No LAMMPS deck** -- it reads only the cleaned ``{base}.pdb`` the calculator writes,
  so :attr:`requires_lammps_prep` is ``False``.
* **The engine is optional + lazy** -- tmol is imported only when an energy is
  evaluated. ``import frustrapy`` and the default LAMMPS path never import tmol. A
  missing tmol raises a clear, actionable error at compute time (see
  :mod:`.atomic_tmol_engine`), never at import time.

Honesty (load-bearing): tmol removes PyRosetta for **energy evaluation**. The NATIVE
pose is scored directly from its coordinates in-container with no Rosetta. The DECOY
poses need side-chain placement (the permutation decoys are threaded + repacked);
that packing needs a rotamer optimizer + the Dunbrack library, which tmol ships but is
slow and non-commercial-tier (``docs/tmol/PARAM_SOURCING.md``). So the decoy scorer is
the heavy, maintainer-analogue path -- and, exactly like the AA lane's
``score_sequences(scorer=...)``, it is **injectable**, so the wiring, the output
contract, and the sign can be tested without running the packer. See
``docs/tmol/PY_BACKEND_NOTES.md``.

Parallelism (mission scope item 1): this backend spawns NO process pool. tmol uses
torch intra-op threads; the decoy loop here is a plain serial loop, so it never nests
a pool inside the calculator's pool. The thread budget is the caller's
(:func:`.atomic_tmol_engine.configure_torch_threads`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np

from .base import FrustrationBackend

if TYPE_CHECKING:
    from ..core import Pdb
    from ..analysis.frustration_calculator import FrustrationCalculator
    from .atomic_post import ContactGeometry

#: Decoy-count default. The reference recommends >= 200 permutation decoys for
#: convergence (the shipped demo used 50). Overridable via calculator ``atomic_*``
#: attributes or the matching ``FRUSTRAPY_ATOMIC_*`` environment variables, exactly as
#: in :mod:`.atomic`.
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
class AtomicTmolOptions:
    """Resolved per-run options for the tmol atomic backend.

    Read from the owning calculator's optional ``atomic_*`` attributes, then the
    matching ``FRUSTRAPY_ATOMIC_*`` / ``FRUSTRAPY_TMOL_*`` environment variables, then
    documented defaults. These parameterize only the decoy step, the contact geometry,
    and the tmol thread budget; nothing here touches the LAMMPS/native paths.
    """

    n_decoys: int
    seed: Optional[int]
    seq_sep: int
    scheme: str
    distance_cutoff: float
    fa_rep_cutoff: float
    n_threads: Optional[int]
    n_decoys_per_site: Optional[int]
    n_decoys_per_contact: Optional[int]

    @classmethod
    def from_calculator(cls, calculator: "FrustrationCalculator") -> "AtomicTmolOptions":
        from .atomic_post import CONTACT_DISTANCE_CUTOFF, DEFAULT_SEQ_SEP  # noqa: PLC0415
        from .atomic_tmol_engine import DEFAULT_SCHEME, FA_REP_CUTOFF  # noqa: PLC0415

        def attr(name, default):
            return getattr(calculator, f"atomic_{name}", None) or default

        seed = getattr(calculator, "atomic_seed", None)
        if seed is None:
            seed = _env_int("FRUSTRAPY_ATOMIC_SEED", None)
        return cls(
            n_decoys=int(attr("n_decoys", _env_int("FRUSTRAPY_ATOMIC_N_DECOYS", DEFAULT_N_DECOYS))),
            seed=seed,
            seq_sep=int(attr("seq_sep", DEFAULT_SEQ_SEP)),
            scheme=str(getattr(calculator, "atomic_scheme", None) or DEFAULT_SCHEME),
            distance_cutoff=float(attr("distance_cutoff", CONTACT_DISTANCE_CUTOFF)),
            fa_rep_cutoff=float(attr("fa_rep_cutoff", FA_REP_CUTOFF)),
            n_threads=getattr(calculator, "atomic_n_threads", None)
            or getattr(calculator, "n_cpus", None),
            n_decoys_per_site=getattr(calculator, "atomic_n_decoys_per_site", None),
            n_decoys_per_contact=getattr(calculator, "atomic_n_decoys_per_contact", None),
        )


def _default_native_scorer(pdb_path, *, scheme, fa_rep_cutoff, n_threads):
    from .atomic_tmol_engine import compute_native_residue_energies_tmol  # noqa: PLC0415

    return compute_native_residue_energies_tmol(
        pdb_path, scheme=scheme, fa_rep_cutoff=fa_rep_cutoff, n_threads=n_threads
    )


def _default_decoy_scorer(pdb_path, decoy_seq, *, native_seq, scheme, fa_rep_cutoff, n_threads):
    from .atomic_tmol_engine import compute_decoy_residue_energies_tmol  # noqa: PLC0415

    return compute_decoy_residue_energies_tmol(
        pdb_path, decoy_seq, native_seq=native_seq, scheme=scheme,
        fa_rep_cutoff=fa_rep_cutoff, n_threads=n_threads,
    )


def _residue_array_to_keyed(geom: "ContactGeometry", energies: np.ndarray) -> dict:
    """Map a per-residue energy array (block / PDB-residue order) to the residue-key
    dict :class:`~frustrapy.backends.atomic_engine.EngineResult` expects.

    The tmol scorers return energies in block order; the shared
    :class:`~frustrapy.backends.atomic_post.ContactGeometry` enumerates residues in the
    SAME PDB order (single chain), so block ``i`` maps to ``geom.cid_list[i]``. The
    length is asserted; multi-chain residue ordering across the tmol pose vs the
    geometry is a maintainer note (see docs/tmol/PY_BACKEND_NOTES.md), consistent with
    the AA lane's multi-chain caveat.
    """
    if len(energies) != geom.n_residues:
        raise ValueError(
            f"tmol returned {len(energies)} residue energies but the contact geometry "
            f"has {geom.n_residues} residues; the pose and the PDB must enumerate "
            "residues in the same order (single-chain assumption)"
        )
    return {geom.cid_list[i]: float(energies[i]) for i in range(geom.n_residues)}


class AtomicTmolBackend(FrustrationBackend):
    """All-atom frustration backend with a tmol (Apache-2.0) energy engine.

    Produces ``tertiary_frustration.dat`` in the same AWSEM column layout the LAMMPS
    binary emits, so the shared post-processing, tables, density, and viz run
    unchanged. ``configurational`` is the reference's parity-backed permutation scheme;
    ``mutational`` / ``singleresidue`` are EXPERIMENTAL extensions (no atomic reference,
    no parity oracle), exactly as documented in :mod:`.atomic_modes`.
    """

    name = "atomic-tmol"
    requires_lammps_prep = False

    def __init__(
        self,
        native_scorer: Optional[Callable] = None,
        decoy_scorer: Optional[Callable] = None,
    ) -> None:
        # Injectable so the wiring, output contract, and sign can be tested without
        # running the heavy tmol packer (the AA-lane score_sequences(scorer=...) pattern).
        self._native_scorer = native_scorer or _default_native_scorer
        self._decoy_scorer = decoy_scorer or _default_decoy_scorer

    # -- the backend seam -----------------------------------------------------

    def compute_energies(self, calculator: "FrustrationCalculator", pdb: "Pdb") -> None:
        from .atomic_post import load_contact_geometry  # noqa: PLC0415

        job_dir = pdb.job_dir
        pdb_path = os.path.join(job_dir, f"{pdb.pdb_base}.pdb")
        out_path = os.path.join(job_dir, "tertiary_frustration.dat")
        mode = calculator.mode
        opts = AtomicTmolOptions.from_calculator(calculator)

        geom = load_contact_geometry(pdb_path, pdb_code=pdb.pdb_base)

        if mode == "configurational":
            self._compute_configurational(pdb_path, geom, out_path, opts)
        elif mode == "mutational":
            self._compute_mutational(pdb_path, geom, out_path, opts)
        elif mode == "singleresidue":
            self._compute_singleresidue(pdb_path, geom, out_path, opts)
        else:
            raise ValueError(
                f"atomic-tmol backend does not support mode {mode!r}; "
                "expected configurational, mutational, or singleresidue"
            )

    # -- scoring helpers ------------------------------------------------------

    def _score_native(self, pdb_path, geom, opts) -> dict:
        arr = self._native_scorer(
            pdb_path, scheme=opts.scheme, fa_rep_cutoff=opts.fa_rep_cutoff,
            n_threads=opts.n_threads,
        )
        return _residue_array_to_keyed(geom, np.asarray(arr, dtype=float))

    def _score_decoys(self, pdb_path, geom, sequences) -> List[dict]:
        """Score every decoy sequence serially (torch intra-op threads only; no nested
        pool). Returns one residue-key energy dict per sequence, in input order."""
        native_seq = "".join(geom.aa)
        out: List[dict] = []
        for seq in sequences:
            arr = self._decoy_scorer(
                pdb_path, seq, native_seq=native_seq, scheme=self._opts.scheme,
                fa_rep_cutoff=self._opts.fa_rep_cutoff, n_threads=self._opts.n_threads,
            )
            out.append(_residue_array_to_keyed(geom, np.asarray(arr, dtype=float)))
        return out

    # -- mode implementations -------------------------------------------------

    def _compute_configurational(self, pdb_path, geom, out_path, opts: AtomicTmolOptions) -> None:
        """PARITY-BACKED. The reference permutation scheme: one protein-wide decoy
        mean/sd over the contact set. Native scored once; permutation decoys scored
        serially with tmol."""
        from .atomic_engine import EngineResult  # noqa: PLC0415
        from .atomic_modes import get_decoy_strategy  # noqa: PLC0415
        from .atomic_post import write_tertiary_frustration  # noqa: PLC0415

        self._opts = opts
        native_seq = "".join(geom.aa)
        native_res_e = self._score_native(pdb_path, geom, opts)
        groups = get_decoy_strategy("configurational").generate(
            native_seq, n_decoys=opts.n_decoys, seed=opts.seed
        )
        decoy_seqs = [spec.sequence for group in groups for spec in group.specs]
        decoy_res_e = self._score_decoys(pdb_path, geom, decoy_seqs)
        engine_result = EngineResult(
            native_residue_energy=native_res_e,
            decoy_residue_energies=decoy_res_e,
            scheme=opts.scheme,
            n_decoys_requested=len(decoy_res_e),
        )
        write_tertiary_frustration(
            out_path,
            geom,
            engine_result=engine_result,
            seq_sep=opts.seq_sep,
            distance_cutoff=opts.distance_cutoff,
        )

    def _compute_mutational(self, pdb_path, geom, out_path, opts: AtomicTmolOptions) -> None:
        """EXPERIMENTAL. Per-contact decoy ensemble (re-identify only the contacting
        pair). No atomic reference / no parity oracle."""
        from .atomic_engine import ContactEnergySummary  # noqa: PLC0415
        from .atomic_modes import get_decoy_strategy  # noqa: PLC0415
        from .atomic_post import select_contacts, write_tertiary_frustration  # noqa: PLC0415

        self._opts = opts
        native_seq = "".join(geom.aa)
        native_res_e = self._score_native(pdb_path, geom, opts)
        contacts = select_contacts(
            geom, seq_sep=opts.seq_sep, distance_cutoff=opts.distance_cutoff
        )
        groups = get_decoy_strategy("mutational").generate(
            native_seq, contacts=contacts,
            n_decoys_per_contact=opts.n_decoys_per_contact, seed=opts.seed,
        )
        stats = self._aggregate_groups(pdb_path, geom, groups)
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
            out_path, geom, summaries=summaries,
            seq_sep=opts.seq_sep, distance_cutoff=opts.distance_cutoff,
        )

    def _compute_singleresidue(self, pdb_path, geom, out_path, opts: AtomicTmolOptions) -> None:
        """EXPERIMENTAL. Per-site decoy ensemble (re-identify only site i). Writes the
        8-column single-residue table. No atomic reference / no parity oracle."""
        from .atomic_modes import get_decoy_strategy  # noqa: PLC0415
        from .atomic_post import SiteEnergySummary, write_singleresidue_dat  # noqa: PLC0415

        self._opts = opts
        native_seq = "".join(geom.aa)
        native_res_e = self._score_native(pdb_path, geom, opts)
        groups = get_decoy_strategy("singleresidue").generate(
            native_seq, n_decoys_per_site=opts.n_decoys_per_site, seed=opts.seed
        )
        stats = self._aggregate_groups(pdb_path, geom, groups)
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

    def _aggregate_groups(self, pdb_path, geom, groups):
        """Score every unique decoy sequence once, then pool each group's energies into
        ``(mean, std, n)`` keyed by ``group.key``. Mirrors
        :meth:`AtomicBackend._aggregate_groups_parallel` but serial + tmol."""
        from .atomic_modes import pooled_statistics  # noqa: PLC0415

        all_specs = [spec for group in groups for spec in group.specs]
        unique_seqs = list(dict.fromkeys(spec.sequence for spec in all_specs))
        scored = self._score_decoys(pdb_path, geom, unique_seqs)
        res_e_by_seq = {seq: e for seq, e in zip(unique_seqs, scored)}

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
    "AtomicTmolBackend",
    "AtomicTmolOptions",
    "DEFAULT_N_DECOYS",
]
