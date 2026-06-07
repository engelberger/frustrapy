"""The native C++ / CUDA frustration backend.

``NativeBackend`` reimplements the AWSEM energy reductions that :class:`LammpsBackend`
gets from the precompiled binary, using the compiled ``frustrapy_native`` extension
(the ``native/`` subproject: CPU now, optional CUDA later). It reproduces
``tertiary_frustration.dat`` from the prepared job directory and is parity-gated
against the ``lammps`` reference (native energy, decoy mean/sd, and ``FrstIndex``
match bit-for-bit at the file's 3-decimal print precision; FrstIndex Spearman = 1.0).

How it works: after the calculator has run ``PdbCoords2Lammps.sh``, the job
directory holds the cleaned ``{base}.pdb`` (whose CB/CA coordinates are exactly what
the binary uses), ``fix_backbone_coeff.data``, ``gamma.dat`` and ``burial_gamma.dat``.
This backend parses those, calls :func:`frustrapy_native.compute_frustration`, and
writes ``tertiary_frustration.dat`` in the binary's column layout, so the shared
:meth:`process_results` / :meth:`compute_density` post-processing runs unchanged.

The extension is an optional build artifact, so the import is lazy and graceful:
nothing is imported at module load, and a missing build raises a clear, actionable
error. Installing ``frustrapy`` without a compiler still works and the ``lammps``
default is unaffected.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, List, Tuple

from .base import FrustrationBackend

if TYPE_CHECKING:
    from ..core import Pdb
    from ..analysis.frustration_calculator import FrustrationCalculator

# Alphabet index (letter - 'A') -> AWSEM gamma index 0..19 (fix_backbone.cpp se_map).
_SE_MAP = [0, 0, 4, 3, 6, 13, 7, 8, 9, 0, 11, 10, 12, 2, 0, 14, 5, 1, 15, 16, 0, 19, 17, 0, 18, 0]
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def _load_native():
    """Import the compiled native core, or raise an actionable error if unbuilt."""
    try:
        import frustrapy_native  # noqa: PLC0415 - optional, intentionally lazy
    except ImportError as exc:  # pragma: no cover - exercised only without the build
        raise ImportError(
            "The native frustration core is not built. Install it with "
            "`pip install ./native` (CPU) or "
            "`pip install ./native -C cmake.define.FRUSTRAPY_NATIVE_CUDA=ON` (CUDA). "
            "See docs/NATIVE_BACKEND_DESIGN.md."
        ) from exc
    return frustrapy_native


def _parse_structure(pdb_path: str):
    """Parse the cleaned job PDB into per-residue arrays in file order.

    Returns (coord, res_type, chain_id, seqid, chain_num, letters) where coord is the
    AWSEM interaction coordinate (CB, or CA for glycine), seqid is the sequential
    residue position (1..n, matching the equivalences file the parser uses), and
    chain_num is the 1-based chain index for the output file.
    """
    import numpy as np  # noqa: PLC0415

    ca: Dict[Tuple[str, int], Tuple[float, float, float]] = {}
    cb: Dict[Tuple[str, int], Tuple[float, float, float]] = {}
    resname: Dict[Tuple[str, int], str] = {}
    order: List[Tuple[str, int]] = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom = line[12:16].strip()
            chain = line[21]
            resseq = int(line[22:26])
            key = (chain, resseq)
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            if key not in resname:
                resname[key] = line[17:20].strip()
            if atom == "CA":
                if key not in ca:
                    order.append(key)
                ca[key] = xyz
            elif atom == "CB":
                cb[key] = xyz

    coord = []
    res_type = []
    chain_id = []
    seqid = []
    chain_num = []
    letters = []
    chain_index: Dict[str, int] = {}
    for pos, key in enumerate(order, start=1):
        chain, _ = key
        if chain not in chain_index:
            chain_index[chain] = len(chain_index) + 1
        rn = resname[key]
        xyz = ca[key] if rn == "GLY" else cb.get(key, ca[key])
        one = _THREE_TO_ONE.get(rn, "G")
        coord.append(xyz)
        res_type.append(_SE_MAP[ord(one) - ord("A")])
        chain_id.append(chain_index[chain])
        chain_num.append(chain_index[chain])
        seqid.append(pos)
        letters.append(one)
    return (
        np.ascontiguousarray(coord, dtype=np.float64),
        np.ascontiguousarray(res_type, dtype=np.int32),
        np.ascontiguousarray(chain_id, dtype=np.int32),
        np.ascontiguousarray(seqid, dtype=np.int32),
        chain_num,
        letters,
    )


def _read_coeff(coeff_path: str) -> dict:
    """Parse the AWSEM [Water]/[Burial]/[Tertiary_Frustratometer] blocks we need."""
    blocks: Dict[str, List[str]] = {}
    current = None
    with open(coeff_path) as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("[") and "]" in line:
                current = line[: line.index("]") + 1]
                blocks[current] = []
            elif current is not None and line:
                blocks[current].append(line)

    water = blocks["[Water]"]
    well_kappa, kappa_sigma = (float(x) for x in water[1].split())
    treshold = float(water[2])
    contact_min_sep = int(float(water[3]))
    n_wells = int(float(water[4]))
    wells = [water[5 + k].split() for k in range(n_wells)]
    well_r_min0, well_r_max0 = float(wells[0][0]), float(wells[0][1])
    well_r_min1, well_r_max1 = float(wells[1][0]), float(wells[1][1])

    burial = blocks["[Burial]"]
    k_burial = float(burial[0])
    burial_kappa = float(burial[1])

    tert = blocks["[Tertiary_Frustratometer]"]
    contact_cutoff = float(tert[0])
    n_decoys = int(float(tert[1]))
    mode = tert[3]
    return dict(
        well_kappa=well_kappa, kappa_sigma=kappa_sigma, treshold=treshold,
        contact_min_sep=contact_min_sep, well_r_min0=well_r_min0,
        well_r_max0=well_r_max0, well_r_min1=well_r_min1, well_r_max1=well_r_max1,
        k_burial=k_burial, burial_kappa=burial_kappa, contact_cutoff=contact_cutoff,
        n_decoys=n_decoys, mode=mode,
    )


def _read_gammas(job_dir: str):
    """Read gamma.dat / burial_gamma.dat in the AWSEM order and build the 20x20
    direct/water/protein tables and the 20x3 burial table."""
    import numpy as np  # noqa: PLC0415

    nums: List[float] = []
    with open(os.path.join(job_dir, "gamma.dat")) as fh:
        for line in fh:
            nums.extend(float(x) for x in line.split())
    it = iter(nums)
    wg = np.zeros((2, 20, 20, 2))
    for iw in range(2):
        for i in range(20):
            for j in range(i, 20):
                a = next(it)
                b = next(it)
                wg[iw, i, j, 0] = a
                wg[iw, i, j, 1] = b
                wg[iw, j, i, 0] = a
                wg[iw, j, i, 1] = b
    gamma_direct = np.ascontiguousarray((wg[0, :, :, 0] + wg[0, :, :, 1]) / 2.0)
    gamma_protein = np.ascontiguousarray(wg[1, :, :, 0])
    gamma_water = np.ascontiguousarray(wg[1, :, :, 1])

    burial = np.zeros((20, 3))
    with open(os.path.join(job_dir, "burial_gamma.dat")) as fh:
        rows = [r.split() for r in fh if r.split()]
    for i in range(20):
        for k in range(3):
            burial[i, k] = float(rows[i][k])
    return gamma_direct, gamma_water, gamma_protein, np.ascontiguousarray(burial)


def _write_dat(out_path: str, mode: str, result, coord, chain_num, letters):
    """Write tertiary_frustration.dat in the AWSEM binary's column layout."""
    ui = result["unit_i"]
    uj = result["unit_j"]
    ne = result["native_energy"]
    de = result["decoy_energy"]
    sd = result["sd_energy"]
    fi = result["frst_index"]
    rho = result["rho"]
    with open(out_path, "w") as fh:
        if mode == "singleresidue":
            fh.write("# i i_chain xi yi zi rho_i a_i native_energy <decoy_energies> "
                     "std(decoy_energies) f_i\n")
            fh.write("# timestep: 0\n")
            for k in range(len(ui)):
                i = int(ui[k])
                xi = coord[i]
                fh.write(
                    f"{i + 1:5d} {chain_num[i]:5d} {xi[0]:8.3f} {xi[1]:8.3f} {xi[2]:8.3f} "
                    f"{rho[i]:8.3f} {letters[i]:s} {ne[k]:8.3f} {de[k]:8.3f} "
                    f"{sd[k]:8.3f} {fi[k]:8.3f}\n"
                )
        else:
            fh.write("# i j i_chain j_chain xi yi zi xj yj zj r_ij rho_i rho_j a_i a_j "
                     "native_energy <decoy_energies> std(decoy_energies) f_ij\n")
            fh.write("# timestep: 0\n")
            for k in range(len(ui)):
                i = int(ui[k])
                j = int(uj[k])
                xi = coord[i]
                xj = coord[j]
                rij = (
                    (xi[0] - xj[0]) ** 2 + (xi[1] - xj[1]) ** 2 + (xi[2] - xj[2]) ** 2
                ) ** 0.5
                fh.write(
                    f"{i + 1:5d} {j + 1:5d} {chain_num[i]:3d} {chain_num[j]:3d} "
                    f"{xi[0]:8.3f} {xi[1]:8.3f} {xi[2]:8.3f} "
                    f"{xj[0]:8.3f} {xj[1]:8.3f} {xj[2]:8.3f} {rij:8.3f} "
                    f"{rho[i]:8.3f} {rho[j]:8.3f} {letters[i]:s} {letters[j]:s} "
                    f"{ne[k]:8.3f} {de[k]:8.3f} {sd[k]:8.3f} {fi[k]:8.3f}\n"
                )


class NativeBackend(FrustrationBackend):
    """Native C++/CUDA energy backend (parity-gated against ``lammps``).

    The on-disk output contract is identical to the ``lammps`` path: this backend
    writes ``tertiary_frustration.dat`` in the same column layout, and the shared
    :meth:`process_results` / :meth:`compute_density` post-processing run unchanged.
    """

    name = "native"

    def compute_energies(self, calculator: "FrustrationCalculator", pdb: "Pdb") -> None:
        """Run the native energy reduction, writing ``tertiary_frustration.dat``."""
        native = _load_native()
        job_dir = pdb.job_dir
        pdb_path = os.path.join(job_dir, f"{pdb.pdb_base}.pdb")
        coeff = _read_coeff(os.path.join(job_dir, "fix_backbone_coeff.data"))
        mode = calculator.mode
        if coeff["mode"] != mode:
            # The coefficient file's keyword is the source of truth the binary reads;
            # keep them consistent.
            mode = coeff["mode"]
        coord, res_type, chain_id, seqid, chain_num, letters = _parse_structure(pdb_path)
        gamma_direct, gamma_water, gamma_protein, burial_gamma = _read_gammas(job_dir)

        # Opt into the GPU path via env when the core was built with CUDA. The CPU
        # path is the default and the parity reference.
        use_cuda = os.environ.get("FRUSTRAPY_NATIVE_USE_CUDA", "") not in ("", "0", "false", "False")

        result = native.compute_frustration(
            coord, res_type, chain_id, seqid,
            gamma_direct, gamma_water, gamma_protein, burial_gamma, mode,
            well_kappa=coeff["well_kappa"], kappa_sigma=coeff["kappa_sigma"],
            treshold=coeff["treshold"], well_r_min0=coeff["well_r_min0"],
            well_r_max0=coeff["well_r_max0"], well_r_min1=coeff["well_r_min1"],
            well_r_max1=coeff["well_r_max1"], burial_kappa=coeff["burial_kappa"],
            k_burial=coeff["k_burial"], contact_cutoff=coeff["contact_cutoff"],
            contact_min_sep=coeff["contact_min_sep"], seq_dist=int(calculator.seq_dist),
            n_decoys=coeff["n_decoys"], seed=1, use_cuda=use_cuda,
        )
        _write_dat(
            os.path.join(job_dir, "tertiary_frustration.dat"),
            mode, result, coord, chain_num, letters,
        )
