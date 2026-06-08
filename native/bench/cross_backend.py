"""Cross-backend single-protein benchmark (implementation).

See the package docstring (``native/bench/__init__.py``) for the overview. This
module has no hard dependency on a GPU or on the native build: it probes what is
available at runtime and skips the rest.
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# FrstIndex column (0-based) in the parsed FrustrationData table, per mode.
_FRST_COL = {"configurational": 11, "mutational": 11, "singleresidue": 7}


@dataclass
class BackendRow:
    """One end-to-end backend measurement."""

    backend: str          # display label, e.g. "native (CPU x14)"
    threads: int          # CPU threads used (1 for lammps/cuda rows)
    wall_s: float         # median wall time over repeats
    speedup: float        # vs the serial baseline (native CPU x1, else lammps)
    spearman: float       # FrstIndex Spearman vs the lammps reference (1.0 = identical rank)
    n_units: int          # rows in the output table (contacts or residues)


@dataclass
class CoreScalingRow:
    """One isolated-core measurement at a fixed thread count."""

    threads: int
    wall_s: float          # median wall time of compute_frustration over repeats
    speedup: float         # vs the 1-thread time
    bit_identical: bool    # output bit-identical to the 1-thread output


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman correlation without scipy (Pearson of ranks). Returns 1.0 for
    a constant pair (degenerate but rank-identical)."""
    def rank(xs: Sequence[float]) -> List[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r

    n = len(a)
    if n == 0:
        return float("nan")
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((x - mb) ** 2 for x in rb) ** 0.5
    if va == 0 or vb == 0:
        return 1.0
    return cov / (va * vb)


def _read_frst(path: str, mode: str) -> List[float]:
    col = _FRST_COL[mode]
    with open(path) as fh:
        return [float(line.split()[col]) for line in fh.readlines()[1:]]


def _table_path(results_dir: str, pdb_file: str, mode: str) -> str:
    base = os.path.splitext(os.path.basename(pdb_file))[0]
    return os.path.join(results_dir, f"{base}.done", "FrustrationData", f"{base}.pdb_{mode}")


def _run_once(pdb_file: str, mode: str, backend: str, results_dir: str, seq_dist: int) -> str:
    """Run one calculate_frustration and return the parsed-table path.

    The input PDB is copied into the scratch directory first: some prep paths rewrite
    the structure in place, so the benchmark must never touch the caller's file.
    """
    import frustrapy  # noqa: PLC0415

    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)
    local_pdb = os.path.join(results_dir, os.path.basename(pdb_file))
    shutil.copy2(pdb_file, local_pdb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frustrapy.calculate_frustration(
            pdb_file=local_pdb, mode=mode, results_dir=results_dir, graphics=False,
            visualization=False, debug="ERROR", backend=backend, seq_dist=seq_dist,
        )
    return _table_path(results_dir, local_pdb, mode)


def _native_built() -> bool:
    try:
        import frustrapy_native  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


def _has_cuda() -> bool:
    try:
        import frustrapy_native  # noqa: PLC0415
        return bool(frustrapy_native.has_cuda())
    except Exception:
        return False


def _has_metal() -> bool:
    try:
        import frustrapy_native  # noqa: PLC0415
        return bool(getattr(frustrapy_native, "has_metal", lambda: False)())
    except Exception:
        return False


def _cpu_count() -> int:
    try:
        from frustrapy.utils.concurrency import cpu_budget  # noqa: PLC0415
        return cpu_budget()
    except Exception:
        return os.cpu_count() or 1


def available_backend_labels() -> List[str]:
    """Human-readable list of the backends this machine can actually run."""
    labels = ["lammps"]
    if _native_built():
        labels.append("native CPU (serial + multicore)")
    if _has_cuda():
        labels.append("native CUDA")
    if _has_metal():
        labels.append("native Metal")
    return labels


# --------------------------------------------------------------------------- #
# End-to-end cross-backend benchmark
# --------------------------------------------------------------------------- #
def run_cross_backend_benchmark(
    pdb_file: str,
    mode: str = "configurational",
    threads_list: Optional[Sequence[int]] = None,
    repeats: int = 1,
    seq_dist: int = 12,
    results_root: Optional[str] = None,
) -> List[BackendRow]:
    """Run ``pdb_file`` through every available backend end-to-end.

    Args:
        pdb_file: structure to benchmark.
        mode: frustration mode (configurational | mutational | singleresidue).
        threads_list: native CPU thread counts to measure. Defaults to ``[1, all]``.
        repeats: timed repeats per configuration; the median is reported.
        seq_dist: density sequence-separation (12 or 3).
        results_root: scratch directory root (a temp dir if ``None``).

    Returns:
        One :class:`BackendRow` per measured configuration. The ``lammps`` row is
        the parity reference (Spearman 1.0 by definition). Absent backends are
        silently skipped.
    """
    if mode not in _FRST_COL:
        raise ValueError(f"mode must be one of {sorted(_FRST_COL)}")
    cores = _cpu_count()
    if threads_list is None:
        threads_list = [1, cores] if cores > 1 else [1]
    threads_list = sorted({max(1, int(t)) for t in threads_list})

    if results_root is None:
        import tempfile  # noqa: PLC0415
        results_root = tempfile.mkdtemp(prefix="fp_xbench_")

    def _time(fn) -> Tuple[float, str]:
        path = ""
        samples = []
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            path = fn()
            samples.append(time.perf_counter() - t0)
        return statistics.median(samples), path

    rows: List[BackendRow] = []

    # 1) lammps reference.
    lam_wall, lam_path = _time(
        lambda: _run_once(pdb_file, mode, "lammps", os.path.join(results_root, "lammps"), seq_dist)
    )
    ref_frst = _read_frst(lam_path, mode)
    n_units = len(ref_frst)

    # 2) native CPU at each thread count (env override drives the inner thread count).
    serial_wall: Optional[float] = None
    if _native_built():
        prev = os.environ.get("FRUSTRAPY_NATIVE_THREADS")
        try:
            for t in threads_list:
                os.environ["FRUSTRAPY_NATIVE_THREADS"] = str(t)
                wall, path = _time(
                    lambda t=t: _run_once(
                        pdb_file, mode, "native",
                        os.path.join(results_root, f"native_t{t}"), seq_dist
                    )
                )
                if serial_wall is None:
                    serial_wall = wall
                sp = _spearman(ref_frst, _read_frst(path, mode))
                label = "native CPU x1 (serial)" if t == 1 else f"native CPU x{t}"
                rows.append(BackendRow(label, t, wall, serial_wall / wall, sp, n_units))
        finally:
            if prev is None:
                os.environ.pop("FRUSTRAPY_NATIVE_THREADS", None)
            else:
                os.environ["FRUSTRAPY_NATIVE_THREADS"] = prev

    # 3) native CUDA (if compiled in and a GPU is present).
    if _has_cuda():
        prev_cuda = os.environ.get("FRUSTRAPY_NATIVE_USE_CUDA")
        os.environ["FRUSTRAPY_NATIVE_USE_CUDA"] = "1"
        try:
            wall, path = _time(
                lambda: _run_once(
                    pdb_file, mode, "native",
                    os.path.join(results_root, "native_cuda"), seq_dist
                )
            )
            sp = _spearman(ref_frst, _read_frst(path, mode))
            base = serial_wall if serial_wall else wall
            rows.append(BackendRow("native CUDA", 1, wall, base / wall, sp, n_units))
        finally:
            if prev_cuda is None:
                os.environ.pop("FRUSTRAPY_NATIVE_USE_CUDA", None)
            else:
                os.environ["FRUSTRAPY_NATIVE_USE_CUDA"] = prev_cuda

    # lammps speedup baseline = native serial when present, else itself.
    base = serial_wall if serial_wall else lam_wall
    rows.insert(0, BackendRow("lammps (reference)", 1, lam_wall, base / lam_wall, 1.0, n_units))
    return rows


# --------------------------------------------------------------------------- #
# Isolated-core scaling benchmark
# --------------------------------------------------------------------------- #
def _prepare_core_inputs(pdb_file: str, seq_dist: int, results_dir: str):
    """Prepare the native core inputs once: run the native pipeline to materialize
    the coefficient/gamma files, then parse the structure and parameter tables.

    Returns a kwargs dict ready to splat into ``frustrapy_native.compute_frustration``
    (minus ``mode`` and ``n_threads``).
    """
    from frustrapy.backends import native as nb  # noqa: PLC0415

    # One native run to produce gamma.dat / burial_gamma.dat / fix_backbone_coeff.data.
    _run_once(pdb_file, "configurational", "native", results_dir, seq_dist)
    base = os.path.splitext(os.path.basename(pdb_file))[0]
    job = os.path.join(results_dir, f"{base}.done")

    coord, res_type, chain_id, seqid, _chain_num, _letters = nb._parse_structure(pdb_file)
    coeff = nb._read_coeff(os.path.join(job, "fix_backbone_coeff.data"))
    gd, gw, gp, bg = nb._read_gammas(job)
    return dict(
        coord=coord, res_type=res_type, chain_id=chain_id, res_seqid=seqid,
        gamma_direct=gd, gamma_water=gw, gamma_protein=gp, burial_gamma=bg,
        well_kappa=coeff["well_kappa"], kappa_sigma=coeff["kappa_sigma"],
        treshold=coeff["treshold"], well_r_min0=coeff["well_r_min0"],
        well_r_max0=coeff["well_r_max0"], well_r_min1=coeff["well_r_min1"],
        well_r_max1=coeff["well_r_max1"], burial_kappa=coeff["burial_kappa"],
        k_burial=coeff["k_burial"], contact_cutoff=coeff["contact_cutoff"],
        contact_min_sep=coeff["contact_min_sep"], seq_dist=seq_dist,
        n_decoys=coeff["n_decoys"], seed=1, use_cuda=False,
    )


def benchmark_core_scaling(
    pdb_file: str,
    mode: str = "mutational",
    threads_list: Optional[Sequence[int]] = None,
    repeats: int = 3,
    seq_dist: int = 12,
    results_dir: Optional[str] = None,
) -> List[CoreScalingRow]:
    """Time the C++ reduction directly at each thread count, isolated from the
    constant LAMMPS prep, and verify the output is bit-identical across counts.

    Requires the native build; raises ImportError otherwise.
    """
    import numpy as np  # noqa: PLC0415
    import frustrapy_native as fn  # noqa: PLC0415

    if mode not in _FRST_COL:
        raise ValueError(f"mode must be one of {sorted(_FRST_COL)}")
    cores = _cpu_count()
    if threads_list is None:
        threads_list = sorted({1, max(1, cores // 2), cores})
    threads_list = sorted({max(1, int(t)) for t in threads_list})

    cleanup = False
    if results_dir is None:
        import tempfile  # noqa: PLC0415
        results_dir = tempfile.mkdtemp(prefix="fp_core_scale_")
        cleanup = True
    try:
        kwargs = _prepare_core_inputs(pdb_file, seq_dist, results_dir)
        keys = ["unit_i", "unit_j", "native_energy", "decoy_energy", "sd_energy",
                "frst_index", "rho"]

        def run(t: int):
            return fn.compute_frustration(mode=mode, n_threads=t, **kwargs)

        ref = run(1)
        rows: List[CoreScalingRow] = []
        serial_wall: Optional[float] = None
        for t in threads_list:
            samples = []
            out = None
            for _ in range(max(1, repeats)):
                t0 = time.perf_counter()
                out = run(t)
                samples.append(time.perf_counter() - t0)
            wall = statistics.median(samples)
            if serial_wall is None or t == 1:
                serial_wall = wall if t == 1 else serial_wall
            bit = all(np.array_equal(ref[k], out[k]) for k in keys)
            rows.append(CoreScalingRow(t, wall, 0.0, bit))
        base = next((r.wall_s for r in rows if r.threads == 1), rows[0].wall_s)
        for r in rows:
            r.speedup = base / r.wall_s if r.wall_s else float("nan")
        return rows
    finally:
        if cleanup and os.path.exists(results_dir):
            shutil.rmtree(results_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Formatting + CLI
# --------------------------------------------------------------------------- #
def format_table(rows: Sequence[BackendRow]) -> str:
    head = f"{'backend':24s} {'threads':>7s} {'wall(s)':>10s} {'speedup':>8s} {'spearman':>9s} {'units':>6s}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r.backend:24s} {r.threads:>7d} {r.wall_s:>10.4f} {r.speedup:>8.2f} "
            f"{r.spearman:>9.4f} {r.n_units:>6d}"
        )
    return "\n".join(lines)


def _format_core_table(rows: Sequence[CoreScalingRow]) -> str:
    head = f"{'threads':>7s} {'wall(s)':>10s} {'speedup':>8s} {'bit-identical':>14s}"
    lines = ["core scaling (compute_frustration only):", head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r.threads:>7d} {r.wall_s:>10.4f} {r.speedup:>8.2f} {str(r.bit_identical):>14s}")
    return "\n".join(lines)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-protein cross-backend frustration benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pdb", required=True, help="PDB file to benchmark")
    p.add_argument("--mode", default="configurational",
                   choices=["configurational", "mutational", "singleresidue"])
    p.add_argument("--threads", default="", help="comma-separated CPU thread counts (default: 1,all)")
    p.add_argument("--repeats", type=int, default=1, help="timed repeats per configuration")
    p.add_argument("--seq-dist", type=int, default=12, choices=[3, 12])
    p.add_argument("--results-root", default=None, help="scratch directory (temp if unset)")
    p.add_argument("--core-scaling", action="store_true",
                   help="also run the isolated-core scaling benchmark")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    threads = [int(x) for x in args.threads.split(",") if x.strip()] or None
    print(f"available backends: {', '.join(available_backend_labels())}")
    print(f"protein: {args.pdb}  mode: {args.mode}  seq_dist: {args.seq_dist}\n")

    rows = run_cross_backend_benchmark(
        pdb_file=args.pdb, mode=args.mode, threads_list=threads,
        repeats=args.repeats, seq_dist=args.seq_dist, results_root=args.results_root,
    )
    print(format_table(rows))

    if args.core_scaling and _native_built():
        print()
        core = benchmark_core_scaling(
            pdb_file=args.pdb, mode=args.mode if args.mode != "configurational" else "mutational",
            threads_list=threads, repeats=max(3, args.repeats), seq_dist=args.seq_dist,
        )
        print(_format_core_table(core))
    return 0


if __name__ == "__main__":
    sys.exit(main())
