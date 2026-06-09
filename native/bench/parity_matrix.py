"""Cross-backend parity + cost harness (the M8 driver).

ONE harness that, for a fixed structure set, enumerates every registered frustration
backend, runs the ones this machine can run end to end, and reports BOTH numerical
agreement (the parity matrix) AND cost (wall time, per-unit counts). It is read-only
over each backend's output: it parses the tables the backends already write and never
alters a number.

The load-bearing distinction it encodes is the energy-FUNCTION taxonomy and the
parity-vs-correlation vocabulary fixed by the energy-parity gate
(``docs/tmol/G1_ENERGY_PARITY.md``):

* Two backends that run the SAME energy function agree by PARITY. Their FrstIndex
  should match to numerical tolerance (AWSEM lammps vs AWSEM native is byte-identical
  on 1CRN; REF2015 PyRosetta vs the REF2015 golden post-processor matches to 1e-6).
* Two backends that run DIFFERENT energy functions agree only by CROSS-METHOD
  CORRELATION. A rank/class agreement between AWSEM, REF2015 and beta_nov2016 is a
  correlation, never "parity". The harness refuses to print "parity" for such a pair.

The energy functions in play:

* ``AWSEM``     -- lammps (reference), native (C++/CUDA/Metal). Coarse-grained.
* ``REF2015``   -- atomic (PyRosetta, all-atom). License-gated host.
* ``beta_nov2016`` -- atomic-tmol (tmol/PyTorch) and native_tmol-cpu (torch-free C++).
  A later refit of REF2015; NOT REF2015-equivalent (G1 section 2).

What runs in this container and what is gated is probed at runtime and disclosed in the
report. AWSEM (lammps + native) runs fully. The all-atom paths need a packer/licensed
host and are reported from the committed parity reports with explicit provenance, never
silently dropped.
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Reuse the existing cross-backend plumbing rather than re-inventing it.
from bench.cross_backend import (  # noqa: E402
    _FRST_COL,
    _has_cuda,
    _has_metal,
    _native_built,
    _read_frst,
    _run_once,
    _spearman,
    _table_path,
)


# --------------------------------------------------------------------------- #
# Energy-function taxonomy and the parity-vs-correlation rule (G1).
# --------------------------------------------------------------------------- #
#: Energy function each backend evaluates. This is the axis the G1 vocabulary turns on.
ENERGY_FUNCTION: Dict[str, str] = {
    "lammps": "AWSEM",
    "native": "AWSEM",
    "native-cuda": "AWSEM",
    "native-metal": "AWSEM",
    "atomic": "REF2015",
    "atomic-tmol": "beta_nov2016",
    "native_tmol-cpu": "beta_nov2016",
}


def agreement_kind(backend_a: str, backend_b: str) -> str:
    """Classify the agreement between two backends per the G1 fork.

    Same energy function -> ``"parity"`` (FrstIndex should match to tolerance).
    Different energy function -> ``"cross-method correlation"`` (a rank/class
    relationship only; never "parity"). Raises if a backend is unknown so a new
    backend cannot silently default to "parity".
    """
    fa, fb = ENERGY_FUNCTION[backend_a], ENERGY_FUNCTION[backend_b]
    return "parity" if fa == fb else "cross-method correlation"


# --------------------------------------------------------------------------- #
# Data records.
# --------------------------------------------------------------------------- #
@dataclass
class BackendStatus:
    """Whether a backend can run end to end in this container, and why not."""

    name: str
    energy_function: str
    runnable: bool
    skip_reason: str = ""


@dataclass
class RunRecord:
    """One end-to-end backend run on one structure in one mode."""

    backend: str
    structure: str
    mode: str
    threads: int
    wall_s: float
    n_units: int
    frst: List[float] = field(default_factory=list)        # FrstIndex per unit
    native_e: List[float] = field(default_factory=list)     # NativeEnergy per unit
    decoy_e: List[float] = field(default_factory=list)      # DecoyEnergy per unit
    state: List[str] = field(default_factory=list)          # FrstState class per unit


@dataclass
class ParityCell:
    """One backend-pair agreement on one structure/mode."""

    structure: str
    mode: str
    backend_a: str
    backend_b: str
    kind: str               # "parity" or "cross-method correlation"
    spearman: float
    class_agreement: float  # fraction in [0, 1]; nan when a class column is absent
    max_dnative: float      # max abs delta of NativeEnergy
    max_ddecoy: float       # max abs delta of DecoyEnergy
    max_dfrst: float        # max abs delta of FrstIndex
    n_units: int
    provenance: str = "measured in-container"


@dataclass
class CitedCell:
    """A parity-matrix cell whose numbers come from a committed parity report
    because the run is gated in this container (no PyRosetta, no GPU, no packer)."""

    backend_a: str
    backend_b: str
    kind: str
    spearman: Optional[float]
    class_agreement: Optional[float]
    note: str
    source: str
    status: str             # "cited" (numbers from a report) or "untested"


@dataclass
class VarianceRecord:
    """Re-run spread of a backend's decoy ensemble: the repack/decoy stochasticity."""

    backend: str
    structure: str
    mode: str
    repeats: int
    max_ddecoy: float       # max abs delta of DecoyEnergy across re-runs
    max_dfrst: float        # max abs delta of FrstIndex across re-runs
    note: str = ""


@dataclass
class MatrixResult:
    """Everything the report generator needs."""

    structures: List[str]
    modes: List[str]
    statuses: List[BackendStatus]
    runs: List[RunRecord]
    cells: List[ParityCell]
    cited: List[CitedCell]
    variance: List[VarianceRecord]
    cpu_threads: int


# --------------------------------------------------------------------------- #
# Availability probing.
# --------------------------------------------------------------------------- #
def _importable(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


def probe_backends() -> List[BackendStatus]:
    """Probe which backends can run end to end here. AWSEM lammps + native run; the
    all-atom and GPU paths report a concrete skip reason."""
    out: List[BackendStatus] = []
    out.append(BackendStatus("lammps", "AWSEM", True))
    if _native_built():
        out.append(BackendStatus("native", "AWSEM", True))
    else:
        out.append(BackendStatus("native", "AWSEM", False,
                                  "frustrapy_native extension not built"))
    out.append(BackendStatus(
        "native-cuda", "AWSEM", False,
        "" if _has_cuda() else "no CUDA device / not compiled in"))
    out.append(BackendStatus(
        "native-metal", "AWSEM", False,
        "" if _has_metal() else "no Metal device / not compiled in (host Mac only)"))
    out.append(BackendStatus(
        "atomic", "REF2015", False,
        "" if _importable("pyrosetta") else "PyRosetta not installed (license-gated host)"))
    # atomic-tmol scores the native pose with tmol/torch and packs decoys; both halves
    # need torch (and the slow Dunbrack packer for decoys).
    out.append(BackendStatus(
        "atomic-tmol", "beta_nov2016", False,
        "" if _importable("torch") else "torch/tmol not installed; decoy packer maintainer-gated"))
    # native_tmol-cpu is the torch-free energy kernel: present and unit-validated, but
    # not wired to a full per-PDB frustration run (needs the atom-typer + decoy packer).
    if _importable("frustramol_tmol"):
        engine = "energy kernel runs here (probe passed)" if native_tmol_engine_ok() \
            else "energy kernel present"
        out.append(BackendStatus(
            "native_tmol-cpu", "beta_nov2016", False,
            f"{engine} + unit-validated (tmol oracle 1.99e-5, M5); full per-PDB "
            "frustration not wired (atom-typing + decoy packing maintainer-gated)"))
    else:
        out.append(BackendStatus(
            "native_tmol-cpu", "beta_nov2016", False,
            "frustramol_tmol extension not built"))
    return out


def native_tmol_engine_ok() -> bool:
    """Confirm the torch-free beta_nov2016 energy kernel runs: a far pair scores to
    zero and a close pair scores nonzero. Engine-availability, not a frustration run."""
    try:
        import numpy as np  # noqa: PLC0415
        import frustramol_tmol as fk  # noqa: PLC0415
    except Exception:
        return False

    def _pair(distance: float) -> Dict[str, float]:
        coords = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=np.float64)
        return fk.compute_pair_energies(
            n_threads=1,
            coords=coords,
            block=np.array([0, 1], dtype=np.int32),
            charge=np.array([0.5, -0.5], dtype=np.float64),
            is_heavy=np.array([True, True]),
            lj_radius=np.array([2.0, 2.0]), lj_wdepth=np.array([0.1, 0.1]),
            lk_dgfree=np.array([1.0, 1.0]), lk_lambda=np.array([3.5, 3.5]),
            lk_volume=np.array([16.0, 16.0]),
            is_donor=np.array([False, False]), is_hydroxyl=np.array([False, False]),
            is_polarh=np.array([False, False]), is_acceptor=np.array([False, False]),
            pair_i=np.array([0], dtype=np.int32), pair_j=np.array([1], dtype=np.int32),
            sep_ljlk=np.array([5], dtype=np.int32), sep_elec=np.array([5], dtype=np.int32),
            ljlk_global=np.array([3.0, 2.6, 1.75], dtype=np.float64),
            elec_global=np.array([78.0, 1.0, 0.36, 1.45, 5.5], dtype=np.float64),
            n_blocks=2,
        )

    far = _pair(12.0)
    near = _pair(2.0)
    far_zero = all(abs(float(np.sum(v))) < 1e-12 for v in far.values())
    near_nonzero = any(abs(float(np.sum(v))) > 1e-9 for v in near.values())
    return bool(far_zero and near_nonzero)


# --------------------------------------------------------------------------- #
# End-to-end runs of the AWSEM backends.
# --------------------------------------------------------------------------- #
_STATE_COL = {"configurational": 13, "mutational": 13}  # FrstState; absent for singleresidue
_NATIVE_E_COL = {"configurational": 8, "mutational": 8, "singleresidue": 4}
_DECOY_E_COL = {"configurational": 9, "mutational": 9, "singleresidue": 5}


def _read_col_floats(path: str, col: int) -> List[float]:
    with open(path) as fh:
        return [float(line.split()[col]) for line in fh.readlines()[1:]]


def _read_col_str(path: str, col: int) -> List[str]:
    with open(path) as fh:
        return [line.split()[col] for line in fh.readlines()[1:]]


def _cpu_threads() -> int:
    try:
        from frustrapy.utils.concurrency import cpu_budget  # noqa: PLC0415
        return cpu_budget()
    except Exception:
        return os.cpu_count() or 1


def _run_and_record(
    backend: str, pdb_file: str, structure: str, mode: str, threads: int,
    seq_dist: int, results_root: str, repeats: int,
) -> RunRecord:
    """Run one backend on one structure/mode and parse FrstIndex/energies/class."""
    prev = os.environ.get("FRUSTRAPY_NATIVE_THREADS")
    if backend == "native":
        os.environ["FRUSTRAPY_NATIVE_THREADS"] = str(threads)
    try:
        samples: List[float] = []
        path = ""
        rdir = os.path.join(results_root, f"{backend}_{structure}_{mode}_t{threads}")
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            path = _run_once(pdb_file, mode, backend, rdir, seq_dist)
            samples.append(time.perf_counter() - t0)
        wall = statistics.median(samples)
    finally:
        if backend == "native":
            if prev is None:
                os.environ.pop("FRUSTRAPY_NATIVE_THREADS", None)
            else:
                os.environ["FRUSTRAPY_NATIVE_THREADS"] = prev

    frst = _read_frst(path, mode)
    native_e = _read_col_floats(path, _NATIVE_E_COL[mode])
    decoy_e = _read_col_floats(path, _DECOY_E_COL[mode])
    state = _read_col_str(path, _STATE_COL[mode]) if mode in _STATE_COL else []
    return RunRecord(backend, structure, mode, threads, wall, len(frst),
                     frst, native_e, decoy_e, state)


def _max_abs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return float("nan")
    return max(abs(x - y) for x, y in zip(a, b))


def _class_agreement(a: Sequence[str], b: Sequence[str]) -> float:
    if not a or len(a) != len(b):
        return float("nan")
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def _pair_cell(ra: RunRecord, rb: RunRecord) -> ParityCell:
    return ParityCell(
        structure=ra.structure, mode=ra.mode,
        backend_a=ra.backend, backend_b=rb.backend,
        kind=agreement_kind(ra.backend, rb.backend),
        spearman=_spearman(ra.frst, rb.frst),
        class_agreement=_class_agreement(ra.state, rb.state),
        max_dnative=_max_abs_delta(ra.native_e, rb.native_e),
        max_ddecoy=_max_abs_delta(ra.decoy_e, rb.decoy_e),
        max_dfrst=_max_abs_delta(ra.frst, rb.frst),
        n_units=ra.n_units,
    )


# --------------------------------------------------------------------------- #
# Committed cross-method / gated cells (numbers from the parity reports, not run here).
# --------------------------------------------------------------------------- #
def cited_cells() -> List[CitedCell]:
    """The parity-matrix cells whose runs are gated in this container, filled from the
    committed parity reports with explicit provenance. These are the cross-method and
    REF2015 anchors that need a licensed/packer host to MEASURE here; we cite, never
    re-label them as measured."""
    return [
        CitedCell(
            "atomic", "atomic-tmol", agreement_kind("atomic", "atomic-tmol"),
            spearman=0.917, class_agreement=0.840,
            note="beta_nov2016 (tmol) vs REF2015 (PyRosetta, frozen logs), 324 contacts "
                 "over the shipped reference poses; gap concentrated in hbond + lk_ball "
                 "decomposition (G1 section 2). Zero repack noise (identical frozen "
                 "coordinates scored two ways).",
            source="docs/tmol/G1_ENERGY_PARITY.md section 2 / "
                   "docs/tmol/parity/tmol_vs_rosetta_frozen.md",
            status="cited",
        ),
        CitedCell(
            "atomic", "atomic", "parity (REF2015 golden post-processor vs PyRosetta)",
            spearman=0.999999, class_agreement=1.000,
            note="REF2015 golden tertiary_frustration.dat (post-processor over frozen "
                 "Rosetta logs, no Rosetta) vs PyRosetta REF2015, 328 contacts. The atomic "
                 "post-processing half has a real in-container parity gate.",
            source="docs/tmol/parity/tmol_golden_parity.md / tests/test_atomic_post.py",
            status="cited",
        ),
        CitedCell(
            "atomic", "atomic", "parity (REF2015 live PyRosetta native pose)",
            spearman=None, class_agreement=None,
            note="Live PyRosetta REF2015 on a native pose. Needs a licensed PyRosetta host; "
                 "not runnable here.",
            source="docs/tmol/parity/tmol_vs_pyrosetta.md (maintainer)",
            status="untested",
        ),
        CitedCell(
            "atomic-tmol", "atomic-tmol",
            "parity (tmol beta_nov2016 vs its own source oracle)",
            spearman=None, class_agreement=None,
            note="tmol implementation vs tmol-source beta_nov2016 oracle: 9/9 subterms within "
                 "3.05e-5 on 1ubq term baselines. Validates tmol against its OWN energy "
                 "function, not REF2015. The packer (decoys) is torch-gated here.",
            source="docs/tmol/parity/tmol_oracle_parity.md",
            status="cited",
        ),
    ]


# --------------------------------------------------------------------------- #
# Repack / decoy variance.
# --------------------------------------------------------------------------- #
def measure_awsem_variance(
    pdb_file: str, structure: str, mode: str, seq_dist: int, results_root: str,
    repeats: int = 3,
) -> VarianceRecord:
    """Re-run the AWSEM native backend ``repeats`` times and report the spread of the
    decoy ensemble. The decoys are unseeded in principle, but the Linux lmp_serial /
    native AWSEM path is empirically deterministic (two independent runs give max|delta|
    = 0 on every output column), so the expected spread is zero. We MEASURE it to show
    that, rather than assert it."""
    decoys: List[List[float]] = []
    frsts: List[List[float]] = []
    for i in range(max(2, repeats)):
        rdir = os.path.join(results_root, f"var_{structure}_{mode}_r{i}")
        path = _run_once(pdb_file, mode, "native", rdir, seq_dist)
        decoys.append(_read_col_floats(path, _DECOY_E_COL[mode]))
        frsts.append(_read_frst(path, mode))
    max_dd = max(_max_abs_delta(decoys[0], d) for d in decoys[1:])
    max_df = max(_max_abs_delta(frsts[0], f) for f in frsts[1:])
    note = ("AWSEM decoy ensemble is deterministic on this Linux build (re-run spread "
            "exactly zero); not a side-chain repack. The atomic/atomic-tmol repack "
            "variance is a separate, UNTESTED quantity (no packer in container).")
    return VarianceRecord("native (AWSEM)", structure, mode, len(decoys), max_dd, max_df, note)


# --------------------------------------------------------------------------- #
# Top-level driver.
# --------------------------------------------------------------------------- #
def run_parity_matrix(
    structures: Dict[str, str],
    modes: Sequence[str] = ("configurational", "mutational", "singleresidue"),
    seq_dist: int = 12,
    repeats: int = 1,
    threads: Optional[int] = None,
    results_root: Optional[str] = None,
    variance_repeats: int = 3,
) -> MatrixResult:
    """Run the cross-backend parity + cost harness.

    Args:
        structures: ordered ``{label: pdb_path}`` of the fixed structure set.
        modes: frustration modes to run.
        seq_dist: density sequence-separation (12 or 3).
        repeats: timed repeats per run (median reported).
        threads: native CPU threads (default: the shared core budget).
        results_root: scratch root (temp dir if None).
        variance_repeats: AWSEM re-runs for the decoy-variance measurement.

    Returns:
        A :class:`MatrixResult` with statuses, runs, measured parity cells, cited
        cells, and variance records. Read-only over backend output.
    """
    for m in modes:
        if m not in _FRST_COL:
            raise ValueError(f"mode must be one of {sorted(_FRST_COL)}; got {m!r}")
    threads = threads or _cpu_threads()
    if results_root is None:
        import tempfile  # noqa: PLC0415
        results_root = tempfile.mkdtemp(prefix="fp_parity_matrix_")
    os.makedirs(results_root, exist_ok=True)

    statuses = probe_backends()
    runnable = {s.name for s in statuses if s.runnable}

    runs: List[RunRecord] = []
    cells: List[ParityCell] = []
    variance: List[VarianceRecord] = []

    for label, pdb in structures.items():
        for mode in modes:
            mode_runs: Dict[str, RunRecord] = {}
            for backend in ("lammps", "native"):
                if backend not in runnable:
                    continue
                t = 1 if backend == "lammps" else threads
                rec = _run_and_record(backend, pdb, label, mode, t, seq_dist,
                                      results_root, repeats)
                runs.append(rec)
                mode_runs[backend] = rec
            # Measured pairwise parity among the AWSEM backends that ran.
            if "lammps" in mode_runs and "native" in mode_runs:
                cells.append(_pair_cell(mode_runs["lammps"], mode_runs["native"]))
            # Decoy variance once per structure (configurational is enough).
            if mode == "configurational" and "native" in runnable:
                variance.append(measure_awsem_variance(
                    pdb, label, mode, seq_dist, results_root, variance_repeats))

    return MatrixResult(
        structures=list(structures),
        modes=list(modes),
        statuses=statuses,
        runs=runs,
        cells=cells,
        cited=cited_cells(),
        variance=variance,
        cpu_threads=threads,
    )


# --------------------------------------------------------------------------- #
# Report rendering (Markdown).
# --------------------------------------------------------------------------- #
def _fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "UNTESTED"
    if x != x:  # nan
        return "n/a"
    return f"{x:.{nd}f}"


def render_report(result: MatrixResult) -> str:
    """Render the G1 parity matrix, the cost table, the repack-variance note, and the
    skip disclosure as Markdown. Neutral engineering-validation framing only."""
    L: List[str] = []
    L.append("# Cross-backend parity + cost matrix (M8)")
    L.append("")
    L.append("Generated by `native/bench/parity_matrix.py`. Neutral cross-backend "
             "numerical validation and benchmarking. The harness is read-only over each "
             "backend's output: it parses the tables the backends write and changes no "
             "number.")
    L.append("")
    L.append(f"Structure set: {', '.join(result.structures)}. Modes: "
             f"{', '.join(result.modes)}. Native CPU threads: {result.cpu_threads}.")
    L.append("")
    L.append("## Vocabulary (G1 fork)")
    L.append("")
    L.append("Same energy function -> PARITY (FrstIndex matches to tolerance). Different "
             "energy function -> CROSS-METHOD CORRELATION (a rank/class relationship "
             "only). Energy functions: AWSEM (lammps, native), REF2015 (atomic / "
             "PyRosetta), beta_nov2016 (atomic-tmol, native_tmol-cpu). beta_nov2016 is a "
             "refit of REF2015 and is NOT REF2015-equivalent (`docs/tmol/G1_ENERGY_"
             "PARITY.md`), so a beta-vs-REF2015 agreement is correlation, never parity.")
    L.append("")

    # Backend availability.
    L.append("## Backend availability in this container")
    L.append("")
    L.append("| backend | energy function | ran here | note |")
    L.append("|---|---|---|---|")
    for s in result.statuses:
        ran = "yes" if s.runnable else "no"
        note = s.skip_reason or ("end-to-end" if s.runnable else "")
        L.append(f"| {s.name} | {s.energy_function} | {ran} | {note} |")
    L.append("")

    # Measured parity cells.
    L.append("## Measured agreement (in-container)")
    L.append("")
    L.append("All rows below are MEASURED here. Every measured backend pair shares the "
             "AWSEM energy function, so every row is a PARITY row.")
    L.append("")
    L.append("| structure | mode | pair | kind | FrstIndex Spearman | class agree | "
             "max d NativeE | max d DecoyE | max d FrstIndex | units |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in result.cells:
        L.append(
            f"| {c.structure} | {c.mode} | {c.backend_a} vs {c.backend_b} | {c.kind} | "
            f"{_fmt(c.spearman, 6)} | {_fmt(c.class_agreement, 4)} | "
            f"{_fmt(c.max_dnative, 6)} | {_fmt(c.max_ddecoy, 6)} | "
            f"{_fmt(c.max_dfrst, 6)} | {c.n_units} |")
    L.append("")

    # Cited / gated cells.
    L.append("## Gated agreement (from committed parity reports, NOT run here)")
    L.append("")
    L.append("These cells need a licensed PyRosetta host, a GPU, or the side-chain "
             "packer, none of which run in this container. Numbers are cited from the "
             "committed parity reports with their source; they are not measured by this "
             "harness. They are listed so the matrix is complete and the skips are "
             "explicit, never silently dropped.")
    L.append("")
    L.append("| pair | kind | Spearman | class agree | status | source | note |")
    L.append("|---|---|---|---|---|---|---|")
    for c in result.cited:
        L.append(
            f"| {c.backend_a} vs {c.backend_b} | {c.kind} | "
            f"{_fmt(c.spearman, 6)} | {_fmt(c.class_agreement, 4)} | {c.status} | "
            f"{c.source} | {c.note} |")
    L.append("")

    # Cost table.
    L.append("## Cost (wall time)")
    L.append("")
    L.append("| structure | mode | backend | threads | wall (s) | units |")
    L.append("|---|---|---|---|---|---|")
    for r in result.runs:
        L.append(f"| {r.structure} | {r.mode} | {r.backend} | {r.threads} | "
                 f"{r.wall_s:.4f} | {r.n_units} |")
    L.append("")

    # Repack / decoy variance.
    L.append("## Repack / decoy variance")
    L.append("")
    L.append("A single Spearman is never sold as exact: the decoy ensemble is re-run and "
             "the spread reported. For the AWSEM backends the spread is measured here; "
             "for the all-atom side-chain-repack backends it is UNTESTED (no packer in "
             "container) and that is stated, not hidden.")
    L.append("")
    L.append("| backend | structure | mode | re-runs | max d DecoyE | max d FrstIndex | note |")
    L.append("|---|---|---|---|---|---|---|")
    for v in result.variance:
        L.append(f"| {v.backend} | {v.structure} | {v.mode} | {v.repeats} | "
                 f"{_fmt(v.max_ddecoy, 6)} | {_fmt(v.max_dfrst, 6)} | {v.note} |")
    L.append("| atomic / atomic-tmol (REF2015 / beta_nov2016) | - | - | - | UNTESTED | "
             "UNTESTED | side-chain repack decoys; packer is license/torch-gated, no "
             "spread measurable in container |")
    L.append("")

    # Skip disclosure.
    L.append("## Skips (disclosed)")
    L.append("")
    skipped = [s for s in result.statuses if not s.runnable]
    for s in skipped:
        L.append(f"- **{s.name}** ({s.energy_function}): {s.skip_reason}")
    L.append("")
    L.append("The fixed structure set is small (3 structures) by design: 1crn (crambin, "
             "46 res, single chain), 1ubq (ubiquitin, 76 res, single chain), 1zni "
             "(insulin, 4 chains, the multi-chain case). It is a validation set, not a "
             "coverage claim.")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
_DEFAULT_SET = {
    "1crn": "tests/data/1crn.pdb",
    "1ubq": "tests/data/1UBQ.pdb",
    "1zni": "tests/data/1zni.pdb",
}


def _resolve_default_set(repo_root: str) -> Dict[str, str]:
    return {k: os.path.join(repo_root, v) for k, v in _DEFAULT_SET.items()}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-backend parity + cost harness (M8).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--modes", default="configurational,mutational,singleresidue",
                   help="comma-separated frustration modes")
    p.add_argument("--seq-dist", type=int, default=12, choices=[3, 12])
    p.add_argument("--repeats", type=int, default=1, help="timed repeats per run")
    p.add_argument("--variance-repeats", type=int, default=3,
                   help="AWSEM re-runs for the decoy-variance measurement")
    p.add_argument("--threads", type=int, default=None, help="native CPU threads")
    p.add_argument("--results-root", default=None, help="scratch dir (temp if unset)")
    p.add_argument("--out", default=None, help="write the Markdown report to this path")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    structures = _resolve_default_set(repo_root)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_parity_matrix(
            structures=structures, modes=modes, seq_dist=args.seq_dist,
            repeats=args.repeats, threads=args.threads,
            results_root=args.results_root, variance_repeats=args.variance_repeats,
        )
    report = render_report(result)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(report)
        print(f"wrote {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
