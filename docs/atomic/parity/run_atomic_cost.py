#!/usr/bin/env python3
"""Cost benchmark: the atomic (Rosetta) backend vs the coarse-grain (LAMMPS) path.

This is the AA-PARITY-BENCH B2 deliverable. The all-atom path is far more expensive
than the AWSEM/LAMMPS path: every decoy is a full-atom side-chain repack (FastRelax +
RestrictToRepacking) rather than a single AWSEM single-point energy. This script
quantifies that honestly and ties into the existing cross-backend benchmark narrative
(``native/bench/``).

Two halves, matching the two parity tiers:

* The **coarse-grain baseline is measured here, in-container** (``--lammps-baseline``):
  it times the default LAMMPS backend on a structure, which is the real reference
  point the atomic cost is expressed relative to.

* The **atomic wall time needs PyRosetta and is a MAINTAINER measurement**. Rather
  than fabricate a number, this script uses an explicit cost MODEL parameterized by
  one measured quantity: the per-decoy repack wall time ``t_repack`` (seconds),
  which the maintainer measures once on their hardware (see the runbook). Everything
  else (decoy count, protein size, mode multiplicities, the native-pose amortization)
  is computed from that one input, so the projection is transparent and auditable.

The cost model (per structure, one mode)
-----------------------------------------
Let ``D`` = number of decoys, ``t_repack`` = wall time of one full-atom repack +
energy extraction (seconds), ``t_native`` = one native repack (~= ``t_repack``),
``N`` = number of residues, ``C`` = number of selected contacts.

* **configurational**: one protein-wide decoy ensemble. The native pose is relaxed
  once; each of ``D`` decoys is threaded onto the fixed backbone and repacked.

      T_config ~= t_native + D * t_repack

* **mutational**: a per-contact decoy ensemble (only the two contacting identities
  vary). Without amortization each contact pays its own native + decoys; WITH the
  AA-INTEGRATE G3 native-pose amortization the native relax is paid ONCE and shared
  read-only across contacts:

      T_mut(no amort)  ~= C * (t_native + D * t_repack)
      T_mut(amort)     ~= t_native + C * D * t_repack

* **singleresidue**: the most expensive case, a per-site ensemble over identities.
  For a saturation scan this is ``N`` sites x up to 19 substitution identities x
  ``D`` decoys. Native-pose amortization again removes the repeated native relax:

      T_single(no amort) ~= N * (t_native + 19 * D * t_repack)
      T_single(amort)    ~= t_native + N * 19 * D * t_repack

The LAMMPS path, by contrast, is ONE single-point AWSEM energy per structure
(the ~1000-decoy statistics are computed inside the binary in a single ``run 0``
call), so its per-structure cost does not carry the per-decoy full-atom repack
factor at all. That factor -- ``D`` independent side-chain optimizations -- is the
whole reason all-atom costs orders of magnitude more, and what it buys is side-chain
packing resolution (a real ref2015 repacked rotamer ensemble) instead of the AWSEM
coarse-grain approximation.

Usage
-----
Measure the coarse-grain baseline in-container::

    python docs/atomic/parity/run_atomic_cost.py --lammps-baseline --pdb tests/data/<x>.pdb

Project the atomic cost from a maintainer-measured per-decoy repack time::

    python docs/atomic/parity/run_atomic_cost.py \
        --t-repack 8.0 --n-decoys 200 --n-res 92 --n-contacts 328 \
        --lammps-wall 0.98

(With ``--lammps-wall`` given, the atomic-vs-LAMMPS ratio is printed too.)
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Optional


def _measure_lammps_baseline(pdb: str, mode: str, repeats: int) -> float:
    """Time the default LAMMPS backend on ``pdb`` (median of ``repeats``).

    Real in-container measurement -- the coarse-grain reference the atomic cost is
    expressed against. Imports frustrapy lazily so ``--help`` works without the deps.
    """
    import tempfile

    import frustrapy  # noqa: PLC0415

    samples = []
    for _ in range(repeats):
        out = tempfile.mkdtemp(prefix="atomic_cost_lammps_")
        t0 = time.perf_counter()
        frustrapy.calculate_frustration(
            pdb_file=pdb, mode=mode, results_dir=out, graphics=False, debug=False
        )
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _project_atomic(
    t_repack: float,
    n_decoys: int,
    n_res: int,
    n_contacts: int,
    mode: str,
    amortize: bool,
    n_identities: int = 19,
) -> float:
    """Projected atomic wall time (seconds) from the cost model above."""
    t_native = t_repack  # one native relax ~= one decoy repack
    if mode == "configurational":
        return t_native + n_decoys * t_repack
    if mode == "mutational":
        if amortize:
            return t_native + n_contacts * n_decoys * t_repack
        return n_contacts * (t_native + n_decoys * t_repack)
    if mode == "singleresidue":
        if amortize:
            return t_native + n_res * n_identities * n_decoys * t_repack
        return n_res * (t_native + n_identities * n_decoys * t_repack)
    raise ValueError(f"unknown mode {mode!r}")


def _fmt_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.2f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.2f} h"
    return f"{seconds / 86400:.2f} d"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--lammps-baseline",
        action="store_true",
        help="measure the LAMMPS backend wall time in-container (real measurement).",
    )
    p.add_argument("--pdb", help="structure for --lammps-baseline.")
    p.add_argument("--repeats", type=int, default=3, help="repeats for the baseline median.")
    p.add_argument(
        "--t-repack",
        type=float,
        help="MAINTAINER-measured per-decoy full-atom repack + extract wall time (s). "
        "Drives the atomic cost projection.",
    )
    p.add_argument("--n-decoys", type=int, default=200, help="decoys per ensemble (paper: >=200).")
    p.add_argument("--n-res", type=int, default=92, help="residue count (1QYS/TOP7: 92).")
    p.add_argument("--n-contacts", type=int, default=328, help="selected contacts (1QYS: 328).")
    p.add_argument(
        "--lammps-wall",
        type=float,
        help="LAMMPS wall time (s) to print the atomic-vs-LAMMPS ratio. If omitted "
        "and --lammps-baseline ran, the measured value is used.",
    )
    p.add_argument(
        "--mode",
        default="configurational",
        choices=["configurational", "mutational", "singleresidue"],
    )
    args = p.parse_args(argv)

    lammps_wall: Optional[float] = args.lammps_wall

    if args.lammps_baseline:
        if not args.pdb:
            p.error("--lammps-baseline requires --pdb")
        print("COARSE-GRAIN BASELINE (LAMMPS, measured in-container)")
        measured = _measure_lammps_baseline(args.pdb, args.mode, args.repeats)
        print(f"  pdb           : {args.pdb}")
        print(f"  mode          : {args.mode}")
        print(f"  wall (median) : {_fmt_time(measured)}  (over {args.repeats} repeats)")
        if lammps_wall is None:
            lammps_wall = measured

    if args.t_repack is not None:
        print("\nATOMIC COST PROJECTION (maintainer-measured t_repack; PyRosetta path)")
        print(f"  t_repack (per decoy) : {args.t_repack:.3f} s")
        print(f"  n_decoys             : {args.n_decoys}")
        print(f"  n_res / n_contacts   : {args.n_res} / {args.n_contacts}")
        print(f"  mode                 : {args.mode}")
        for mode in ("configurational", "mutational", "singleresidue"):
            no_amort = _project_atomic(
                args.t_repack, args.n_decoys, args.n_res, args.n_contacts, mode, amortize=False
            )
            amort = _project_atomic(
                args.t_repack, args.n_decoys, args.n_res, args.n_contacts, mode, amortize=True
            )
            tag = "  <- selected" if mode == args.mode else ""
            if mode == "configurational":
                print(f"  T[{mode:15s}] = {_fmt_time(no_amort)}{tag}")
            else:
                saved = (1.0 - amort / no_amort) * 100.0 if no_amort else 0.0
                print(
                    f"  T[{mode:15s}] = {_fmt_time(amort)} (amortized) "
                    f"vs {_fmt_time(no_amort)} (naive); native-pose amortization "
                    f"saves {saved:.1f}%{tag}"
                )
        selected = _project_atomic(
            args.t_repack, args.n_decoys, args.n_res, args.n_contacts, args.mode, amortize=True
        )
        if lammps_wall:
            print(
                f"\n  atomic / LAMMPS ratio ({args.mode}, amortized): "
                f"{selected / lammps_wall:.0f}x "
                f"({_fmt_time(selected)} vs {_fmt_time(lammps_wall)})"
            )
        print(
            "\n  Interpretation: the ratio is dominated by the per-decoy full-atom "
            "repack. The LAMMPS path computes its whole decoy statistic in one "
            "single-point call; the atomic path pays D independent side-chain "
            "optimizations. The extra cost buys ref2015 side-chain packing "
            "resolution. Single-residue saturation is the worst case (N x 19 x D "
            "repacks); native-pose amortization (AA-INTEGRATE G3) removes the "
            "repeated native relax but not the decoy repacks."
        )

    if not args.lammps_baseline and args.t_repack is None:
        p.error("provide --lammps-baseline (with --pdb) and/or --t-repack.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
