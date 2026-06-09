"""Post-processor for the all-atom (Rosetta) Frustratometer backend (AA-OUTPUT).

This is the parity-testable half of the atomic backend. AA-ENGINE
(:mod:`frustrapy.backends.atomic_engine`) turns the Rosetta ``ResResE`` logs into
per-contact ``(native_energy, decoy_mean, decoy_std)`` records; this module turns
those records into the Z-score (``FrstIndex``) and writes a
``tertiary_frustration.dat`` in the **same AWSEM column layout the LAMMPS binary
emits**, so the shared ``process_results`` parses it into the canonical 14-column
contact table with no special-casing.

Unlike the engine, this module needs **no Rosetta**. The audit produced a golden
``tertiary_frustration.dat`` by running the reference ``Frust_Post_public.py`` over
the shipped logs (``docs/atomic/golden/``); this module ports that post-processor
into FrustraPy and is gated bit-for-bit (within print precision) against that
golden by ``tests/test_atomic_post.py``.

Three things this module reproduces from the reference
(``docs/atomic/golden/Frust_Post_public_py3.py``), in order:

* **the representative-atom contact geometry** (``get_index`` / ``calc_dist_matrix``):
  one representative atom per residue (CB, or CA for GLY / residues with no CB),
  and the contact distance matrix between them;
* **the contact selection** (``frust_map``): a pair ``(i, j)`` with ``i < j`` is a
  contact iff the representative-atom distance is ``<= 10 A`` and the two residues
  are sequence-separated (``|i - j| > sep``) or on different chains;
* **the per-contact Z-score**, with the **sign flipped** to FrustraPy's convention.

**Sign convention (the #1 hazard).** The atomic reference computes
``frust = (E_native - decoy_mean) / decoy_std`` and, because Rosetta energies are
"lower = more favorable", reads a very negative Z as minimally frustrated (the
published-paper sign). FrustraPy/AWSEM uses the **opposite** implemented relation
``FrstIndex = (DecoyEnergy - NativeEnergy) / SDEnergy`` where **positive = minimally
frustrated** (the project conventions). So this module **negates** the atomic Z:
``FrstIndex = (decoy_mean - E_native) / decoy_std``. That keeps the existing
``FrstState`` classifier (``>= 0.78`` minimal, ``<= -1`` highly) correct. The flip
is verified row-for-row against the golden by the parity test, not by reasoning
alone.

**Density / Welltype columns.** The AWSEM ``DensityRes1`` / ``DensityRes2`` columns
are the AWSEM 5 A *burial* density; the atomic method has no such quantity (the
backbone is fixed; there is no AWSEM density field). We fill those two columns with
a documented sentinel (:data:`BURIAL_DENSITY_SENTINEL`) chosen so that
``process_results`` never labels an atomic contact ``water-mediated`` (a
burial-density claim the atomic model cannot support): atomic contacts come out
``short`` (< 6.5 A) or ``long`` (>= 6.5 A) by representative-atom distance alone.
The separate 5 A *spatial* density kernel (``_calculate_frustration_density`` ->
the ``_5adens`` proportions table) IS meaningful for the atomic backend because it
is a backend-agnostic geometric reduction over the written contacts and their
``FrstIndex``; see ``docs/atomic/AA_OUTPUT_NOTES.md`` for the full verdict.

This module imports only numpy + Biopython (both hard FrustraPy deps); it never
imports PyRosetta, so ``import frustrapy`` and the default LAMMPS path are
unaffected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import sqrt
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .atomic_engine import (
    ContactEnergySummary,
    DEFAULT_SCHEME,
    EngineResult,
    load_engine_result_from_logs,
)

# ---------------------------------------------------------------------------
# Reference constants (Frust_Post_public.py).
# ---------------------------------------------------------------------------

#: Standard residue three-letter codes the reference treats as protein, including
#: MSE (selenomethionine), in the reference's order (``get_index`` se_map).
SE_MAP_3: Tuple[str, ...] = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
    "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE",
)

#: One-letter codes aligned positionally to :data:`SE_MAP_3` (MSE -> "M", as the
#: reference se_map_b does).
SE_MAP_1: Tuple[str, ...] = (
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P",
    "S", "T", "W", "Y", "V", "M",
)

_SE_MAP_3 = frozenset(SE_MAP_3)

#: Contact distance cutoff between representative atoms (``frust_map`` /
#: ``decoy_stat``: ``contact_map[i, j] <= 10.0``).
CONTACT_DISTANCE_CUTOFF = 10.0

#: Default sequence separation (``job.sh`` line 32: ``sep = 9``). A pair contributes
#: only if ``|i - j| > sep`` (intra-chain) or the residues are on different chains.
DEFAULT_SEQ_SEP = 9

#: Sentinel written into the two AWSEM burial-density columns (DensityRes1/2). The
#: atomic method produces no AWSEM burial density. The value equals the AWSEM
#: water-mediated cutoff (``WATER_MEDIATED_DENSITY_CUTOFF`` = 2.6), so the parser's
#: ``density < 2.6`` test is False and no atomic contact is ever mislabeled
#: ``water-mediated``; contacts are ``short`` / ``long`` by distance alone.
BURIAL_DENSITY_SENTINEL = 2.6


# ---------------------------------------------------------------------------
# Geometry: representative atoms + contact distance matrix (ports get_index /
# calc_dist_matrix for the protein-only case; ligand handling is out of scope, see
# below).
# ---------------------------------------------------------------------------

def _vector(p1, p2) -> List[float]:
    return [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]


def _vabs(a) -> float:
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _one_letter(resname: str) -> str:
    """Map a three-letter code in :data:`SE_MAP_3` to its one-letter code."""
    return SE_MAP_1[SE_MAP_3.index(resname)]


@dataclass(frozen=True)
class ContactGeometry:
    """Representative-atom geometry of a structure, in the reference's residue
    order (PDB order within each chain).

    ``cid_list[i]`` is ``chain_letter + str(resnum)`` (e.g. ``"A1"``), matching the
    :class:`~frustrapy.backends.atomic_engine.ResPairEnergy` residue keys; ``aa[i]``
    is the one-letter identity; ``rep_coords[i]`` is the representative-atom
    coordinate (CB, or CA for GLY / no-CB); ``dist[i, j]`` is the Euclidean distance
    between representative atoms ``i`` and ``j``.
    """

    cid_list: Tuple[str, ...]
    aa: Tuple[str, ...]
    rep_coords: np.ndarray  # shape (n, 3)
    dist: np.ndarray  # shape (n, n)

    @property
    def n_residues(self) -> int:
        return len(self.cid_list)

    def chain_of(self, i: int) -> str:
        return self.cid_list[i][0]


def load_contact_geometry(
    pdb_path: str, pdb_code: Optional[str] = None
) -> ContactGeometry:
    """Port of the reference ``get_index`` + ``calc_dist_matrix`` for protein input.

    For every residue (in PDB order, per chain) picks one representative atom:

    * GLY, or any residue with no ``CB`` but a ``CA`` -> ``CA``;
    * any residue with a ``CB`` -> ``CB``.

    The representative-atom choice is identical to the atom ``calc_dist_matrix``
    uses for each residue, so the distance matrix is exactly the pairwise Euclidean
    distance between representative atoms (the reference computes the same value via
    its per-pair ``CB``/``CA`` branch). This avoids the reference's second PDB parse
    while staying numerically identical for protein-only structures.

    Ligand / non-standard residues (``resname not in`` :data:`SE_MAP_3`) are **not**
    supported here: the reference's per-ligand decoy statistics have no analogue in
    :class:`~frustrapy.backends.atomic_engine.EngineResult` and the parity fixture
    (1QYS / native.pdb) has none. A clear ``NotImplementedError`` is raised if any
    are present, rather than silently diverging.

    Args:
        pdb_path: path to the scored structure (the reference's ``3gso.pdb`` ==
            ``native.pdb``).
        pdb_code: optional structure id passed to the parser (cosmetic).

    Returns:
        A :class:`ContactGeometry`.
    """
    from Bio.PDB.PDBParser import PDBParser

    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    structure = parser.get_structure(pdb_code or "structure", pdb_path)
    model = structure[0]

    cid_list: List[str] = []
    aa: List[str] = []
    coords: List[np.ndarray] = []

    for chain in model.get_list():
        for res in chain:
            resname = res.get_resname()
            if resname not in _SE_MAP_3:
                raise NotImplementedError(
                    "atomic_post.load_contact_geometry supports protein-only "
                    f"structures; residue {resname} {res.get_id()} in chain "
                    f"{chain.id} is not a standard residue. Ligand handling is a "
                    "maintainer extension (see docs/atomic/AA_DESIGN_DECISION.md)."
                )
            cid_list.append(f"{chain.id}{res.get_id()[1]}")
            aa.append(_one_letter(resname))
            # Representative atom: CA for GLY or any residue lacking CB, else CB.
            if resname == "GLY" or not res.has_id("CB"):
                if not res.has_id("CA"):
                    raise NotImplementedError(
                        f"residue {resname} {res.get_id()} has neither CB nor CA; "
                        "the protein-only post-processor cannot place a "
                        "representative atom for it."
                    )
                coords.append(np.asarray(res["CA"].get_coord(), dtype=float))
            else:
                coords.append(np.asarray(res["CB"].get_coord(), dtype=float))

    rep = np.asarray(coords, dtype=float)
    # Pairwise Euclidean distance between representative atoms == the reference
    # calc_dist_matrix for protein-only input (proven in the parity test against the
    # golden r_ij column).
    diff = rep[:, None, :] - rep[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))

    return ContactGeometry(
        cid_list=tuple(cid_list), aa=tuple(aa), rep_coords=rep, dist=dist
    )


def select_contacts(
    geom: ContactGeometry,
    seq_sep: int = DEFAULT_SEQ_SEP,
    distance_cutoff: float = CONTACT_DISTANCE_CUTOFF,
) -> List[Tuple[int, int]]:
    """Port of the reference ``frust_map`` protein-protein contact filter.

    Returns the ordered list of contact index pairs ``(i, j)`` with ``i < j``,
    iterated ``i`` ascending then ``j`` ascending (the reference's
    ``for i: for j in range(i, reslen)`` order, so the written rows match the golden
    row order). A pair is a contact iff the representative-atom distance is
    ``<= distance_cutoff`` and the residues are sequence-separated
    (``|i - j| > seq_sep``) or on different chains. This is the same set the
    reference pools for the decoy statistics (``decoy_stat``), so the decoy mean/sd
    are computed over exactly the written contacts.
    """
    n = geom.n_residues
    contacts: List[Tuple[int, int]] = []
    for i in range(n):
        chain_i = geom.chain_of(i)
        for j in range(i, n):
            if abs(i - j) <= seq_sep and chain_i == geom.chain_of(j):
                continue
            if geom.dist[i, j] <= distance_cutoff:
                contacts.append((i, j))
    return contacts


# ---------------------------------------------------------------------------
# The Z-score (with the AWSEM sign flip) and the .dat writer.
# ---------------------------------------------------------------------------

def atomic_frustration_index(
    native_energy: float, decoy_mean: float, decoy_std: float
) -> float:
    """FrustraPy/AWSEM ``FrstIndex`` for an atomic contact.

    ``FrstIndex = (DecoyEnergy - NativeEnergy) / SDEnergy`` (the project conventions),
    i.e. the **negation** of the atomic reference's own
    ``frust = (E_native - decoy_mean) / decoy_std``. Positive = minimally
    frustrated (favorable native contact), matching the existing ``FrstState``
    classifier.
    """
    return (decoy_mean - native_energy) / decoy_std


#: Header lines written ahead of the data, matching the AWSEM/native layout that
#: ``process_results`` skips (``tertiary_frustration[2:]``).
_DAT_HEADER = (
    "# i j i_chain j_chain xi yi zi xj yj zj r_ij rho_i rho_j a_i a_j "
    "native_energy <decoy_energies> std(decoy_energies) f_ij\n"
    "# timestep: 0\n"
)

#: Header for the single-residue ``.dat`` layout (the AWSEM/native singleresidue
#: layout that ``process_results`` parses for ``mode == "singleresidue"``:
#: ``[0]i [1]i_chain [2..4]xyz_i [5]rho_i [6]a_i [7]native [8]decoy [9]std [10]f_i``).
_SINGLERESIDUE_DAT_HEADER = (
    "# i i_chain xi yi zi rho_i a_i native_energy <decoy_energies> "
    "std(decoy_energies) f_i\n"
    "# timestep: 0\n"
)


def _chain_numbers(geom: ContactGeometry) -> dict:
    """Map each chain letter to a 1-based integer, in first-appearance order (the
    AWSEM chain-number convention ``process_results`` maps back via the
    equivalences file)."""
    numbers: dict = {}
    for cid in geom.cid_list:
        letter = cid[0]
        if letter not in numbers:
            numbers[letter] = len(numbers) + 1
    return numbers


def write_tertiary_frustration(
    out_path: str,
    geom: ContactGeometry,
    engine_result: Optional[EngineResult] = None,
    seq_sep: int = DEFAULT_SEQ_SEP,
    distance_cutoff: float = CONTACT_DISTANCE_CUTOFF,
    res_offset: int = 1,
    summaries: Optional[Sequence[ContactEnergySummary]] = None,
) -> List[Tuple[int, int]]:
    """Write an AWSEM-format ``tertiary_frustration.dat`` from engine output.

    Selects the contacts from ``geom`` (:func:`select_contacts`), gets each
    contact's ``(native_energy, decoy_mean, decoy_std)``, computes the sign-flipped
    ``FrstIndex`` (:func:`atomic_frustration_index`), and writes the 19-column
    layout the shared ``process_results`` parses:

    ``[0]Res1 [1]Res2 [2]i_chain [3]j_chain [4..6]xyz_i [7..9]xyz_j [10]r_ij
    [11]DensityRes1 [12]DensityRes2 [13]AA1 [14]AA2 [15]NativeEnergy
    [16]DecoyEnergy [17]SDEnergy [18]FrstIndex``

    The per-contact statistic comes from exactly one of two sources, mirroring how
    the AWSEM backend swaps only the decoy step and reuses one writer:

    * ``engine_result`` (the PARITY-BACKED configurational path): the one
      protein-wide decoy mean/sd over the contact set
      (:meth:`EngineResult.summarize_contacts`).
    * ``summaries`` (the EXPERIMENTAL mutational path): per-contact
      ``(native, decoy_mean, decoy_std)`` already aggregated by the caller, one per
      selected contact in :func:`select_contacts` order. Used when the decoy
      ensemble is per-contact rather than protein-wide.

    Provide exactly one of ``engine_result`` / ``summaries``. Floats are formatted
    ``%8.3f`` (the AWSEM/native print precision); residue indices are written 1-based
    (``i + res_offset``, the AWSEM convention); DensityRes1/2 are the documented
    :data:`BURIAL_DENSITY_SENTINEL`.

    Returns the list of contact index pairs written (same as
    :func:`select_contacts`), so callers can align the rows.
    """
    if (engine_result is None) == (summaries is None):
        raise ValueError(
            "provide exactly one of engine_result (configurational, protein-wide "
            "statistic) or summaries (mutational, per-contact statistic)"
        )
    contacts = select_contacts(geom, seq_sep=seq_sep, distance_cutoff=distance_cutoff)
    if engine_result is not None:
        cid_pairs = [(geom.cid_list[i], geom.cid_list[j]) for (i, j) in contacts]
        summaries = engine_result.summarize_contacts(cid_pairs)
    elif len(summaries) != len(contacts):
        raise ValueError(
            f"summaries ({len(summaries)}) must align 1:1 with the selected contacts "
            f"({len(contacts)}) in select_contacts order"
        )
    chain_num = _chain_numbers(geom)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(_DAT_HEADER)
        for (i, j), summ in zip(contacts, summaries):
            xi = geom.rep_coords[i]
            xj = geom.rep_coords[j]
            rij = geom.dist[i, j]
            ne = summ.native_energy
            de = summ.decoy_mean
            sd = summ.decoy_std
            fi = atomic_frustration_index(ne, de, sd)
            ci = chain_num[geom.chain_of(i)]
            cj = chain_num[geom.chain_of(j)]
            dens = BURIAL_DENSITY_SENTINEL
            fh.write(
                f"{i + res_offset:5d} {j + res_offset:5d} {ci:3d} {cj:3d} "
                f"{xi[0]:8.3f} {xi[1]:8.3f} {xi[2]:8.3f} "
                f"{xj[0]:8.3f} {xj[1]:8.3f} {xj[2]:8.3f} {rij:8.3f} "
                f"{dens:8.3f} {dens:8.3f} {geom.aa[i]:s} {geom.aa[j]:s} "
                f"{ne:8.3f} {de:8.3f} {sd:8.3f} {fi:8.3f}\n"
            )
    return contacts


@dataclass(frozen=True)
class SiteEnergySummary:
    """EXPERIMENTAL. One single-residue site's native energy plus its per-site decoy
    ensemble mean/sd, the input the singleresidue ``.dat`` writer turns into the
    site ``FrstIndex``. ``site_index`` is the 0-based residue index into a
    :class:`ContactGeometry` (``geom.cid_list[site_index]`` is its residue key)."""

    site_index: int
    native_energy: float
    decoy_mean: float
    decoy_std: float


def write_singleresidue_dat(
    out_path: str,
    geom: ContactGeometry,
    site_summaries: Sequence[SiteEnergySummary],
    res_offset: int = 1,
) -> List[int]:
    """EXPERIMENTAL (extension beyond the paper). Write a single-residue
    ``tertiary_frustration.dat`` from per-site decoy statistics.

    This is the singleresidue analogue of :func:`write_tertiary_frustration`: it
    reuses the SAME sign-flipped ``FrstIndex`` (:func:`atomic_frustration_index`)
    and the SAME geometry, and emits the AWSEM/native single-residue ``.dat`` layout
    so the shared ``process_results`` (``mode == "singleresidue"``) parses it into
    the canonical 8-column single-residue table
    (``Res ChainRes DensityRes AA NativeEnergy DecoyEnergy SDEnergy FrstIndex``,
    no ``FrstState`` -- single-residue classification is plot-only at the 0.58
    cutoff, never the 0.78 contact cutoff).

    The atomic single-residue mode is EXPERIMENTAL: the per-site decoy ensemble
    (re-identify only site i; the AWSEM singleresidue definition) has no atomic reference
    and no parity oracle. ``site_summaries`` carries the per-site
    ``(native, decoy_mean, decoy_std)`` the (maintainer-gated) engine produces; this
    writer is the pure post-processing half.

    The single-residue layout written per row is
    ``[0]i [1]i_chain [2..4]xyz_i [5]rho_i [6]a_i [7]native [8]decoy [9]std [10]f_i``.
    ``rho_i`` is the documented :data:`BURIAL_DENSITY_SENTINEL` (the atomic model has
    no AWSEM burial density; the single-residue table copies it through without
    classifying on it).

    Returns the list of 0-based site indices written, in input order.
    """
    chain_num = _chain_numbers(geom)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    written: List[int] = []
    with open(out_path, "w") as fh:
        fh.write(_SINGLERESIDUE_DAT_HEADER)
        for summ in site_summaries:
            i = summ.site_index
            xi = geom.rep_coords[i]
            ne = summ.native_energy
            de = summ.decoy_mean
            sd = summ.decoy_std
            fi = atomic_frustration_index(ne, de, sd)
            ci = chain_num[geom.chain_of(i)]
            dens = BURIAL_DENSITY_SENTINEL
            fh.write(
                f"{i + res_offset:5d} {ci:5d} "
                f"{xi[0]:8.3f} {xi[1]:8.3f} {xi[2]:8.3f} "
                f"{dens:8.3f} {geom.aa[i]:s} "
                f"{ne:8.3f} {de:8.3f} {sd:8.3f} {fi:8.3f}\n"
            )
            written.append(i)
    return written


def write_tertiary_frustration_from_logs(
    out_path: str,
    pdb_path: str,
    native_log: str,
    decoy_logs: Sequence[str],
    scheme: str = DEFAULT_SCHEME,
    seq_sep: int = DEFAULT_SEQ_SEP,
    distance_cutoff: float = CONTACT_DISTANCE_CUTOFF,
    pdb_code: Optional[str] = None,
) -> List[Tuple[int, int]]:
    """End-to-end convenience: parse the shipped ``ResResE`` logs, build the engine
    result, derive the geometry from ``pdb_path``, and write the AWSEM-format
    ``tertiary_frustration.dat``.

    This is the in-container parity path (no Rosetta): the logs already hold the
    Rosetta energies; everything from here is pure numpy + Biopython. Used by
    ``tests/test_atomic_post.py`` to gate against ``docs/atomic/golden/``.
    """
    engine_result = load_engine_result_from_logs(native_log, decoy_logs, scheme=scheme)
    geom = load_contact_geometry(pdb_path, pdb_code=pdb_code)
    return write_tertiary_frustration(
        out_path,
        geom,
        engine_result,
        seq_sep=seq_sep,
        distance_cutoff=distance_cutoff,
    )


__all__ = [
    "SE_MAP_3",
    "SE_MAP_1",
    "CONTACT_DISTANCE_CUTOFF",
    "DEFAULT_SEQ_SEP",
    "BURIAL_DENSITY_SENTINEL",
    "ContactGeometry",
    "load_contact_geometry",
    "select_contacts",
    "atomic_frustration_index",
    "write_tertiary_frustration",
    "SiteEnergySummary",
    "write_singleresidue_dat",
    "write_tertiary_frustration_from_logs",
]
