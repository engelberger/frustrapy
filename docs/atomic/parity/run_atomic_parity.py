#!/usr/bin/env python3
"""Statistical parity harness for the atomic (Rosetta) Frustratometer backend.

This is the AA-PARITY-BENCH B1 deliverable. It computes the standard parity metric
table (Spearman, Pearson R-squared, max|delta|, RMSE, class-agreement) on the
per-contact ``FrstIndex`` of an atomic-backend output against a reference output,
reusing the frozen benchmark metrics in :mod:`native.bench.metrics` (do not rewrite
those; this script only aligns the two contact sets and calls ``compare``).

There are TWO parity tiers and this one script serves both:

* **Tier 1 (in-container, no Rosetta):** ``--golden``. Regenerate the
  post-processor output from the shipped reference ``ResResE`` logs (the same path
  ``tests/test_atomic_post.py`` gates) and diff its ``FrstIndex`` against the
  committed golden ``docs/atomic/golden/tertiary_frustration.dat``. This runs HERE,
  today, and must stay green. It validates the post-processing half: the geometry,
  the contact selection, the sign-flipped Z-score, and the writer.

* **Tier 2 (maintainer, needs PyRosetta):** ``--reference REF --test T1 [T2 T3 ...]``.
  Compare a full ``AtomicBackend`` run (``backend="atomic"``) against the published
  atomic method's own output on the same structure (1QYS / TOP7). Because the Rosetta
  repack is stochastic and the reference is unseeded, parity here is STATISTICAL, not
  bit-exact. Pass three independently-seeded ``--test`` replicates to also get a
  mean/std on each metric across the triplicate. Seed the Rosetta RNG for the
  AtomicBackend run with ``frustrapy.analysis.mutation_backends.set_pyrosetta_seed``
  before running (see the runbook below and ``docs/atomic/parity/MAINTAINER_RUNBOOK.md``).

Input formats are auto-detected by column count, so the same script reads either:

* the **reference** 16-column ``tertiary_frustration.dat`` (no FrstIndex column;
  ``FrstIndex`` is reconstructed with the AWSEM sign as ``(decoy_mean - native)/std``),
  layout ``i j ci cj xi yi zi xj yj zj r_ij AA_i AA_j E_native decoy_mean decoy_std``,
  0-based residue indices; or
* FrustraPy's **19-column** AWSEM ``tertiary_frustration.dat`` (``FrstIndex`` in the
  last column, 1-based residue indices), as written by
  :func:`frustrapy.backends.atomic_post.write_tertiary_frustration`.

Both are keyed by the 0-based contact pair ``(i, j)`` so the two contact sets are
aligned before metrics are computed; a mismatch in the contact SET is reported and
is itself a parity failure.

Usage
-----
Tier 1 (here, now)::

    python docs/atomic/parity/run_atomic_parity.py --golden

Tier 2 (maintainer, after a seeded AtomicBackend run)::

    python docs/atomic/parity/run_atomic_parity.py \
        --reference /path/to/reference/tertiary_frustration.dat \
        --test run_seed1/tertiary_frustration.dat \
               run_seed2/tertiary_frustration.dat \
               run_seed3/tertiary_frustration.dat \
        --mode configurational

Exit status is non-zero if the contact sets differ or (with ``--gate``) if the
metrics fall below the tier-1 pass thresholds.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Tuple

# Reuse the frozen benchmark metrics; do not reimplement them here. ``native/`` has
# no top-level __init__, so (matching native/bench/flag_sweep.py) we add native/ to
# the path and import ``bench.metrics`` rather than ``native.bench.metrics``.
# this file is docs/atomic/parity/run_atomic_parity.py -> four dirnames to the root.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "native")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from bench.metrics import compare, summarize_times  # noqa: E402

GOLDEN_DAT = os.path.join(
    _REPO_ROOT, "docs", "atomic", "golden", "tertiary_frustration.dat"
)
DEFAULT_REF_DIR = os.environ.get(
    "ATOMIC_FRUST_REF", "/workspace/atomic_frustratometer_ref/example_output"
)


# ---------------------------------------------------------------------------
# Readers (auto-detect the two layouts; key everything by 0-based (i, j)).
# ---------------------------------------------------------------------------

def _read_frstindex(path: str) -> Dict[Tuple[int, int], float]:
    """Read a ``tertiary_frustration.dat`` into ``{(i0, j0): FrstIndex}``.

    Auto-detects the reference 16-column layout vs FrustraPy's 19-column AWSEM
    layout by the field count of the first data row, and returns the AWSEM-sign
    ``FrstIndex`` for both (reconstructing it for the reference, where it is not a
    written column). Residue indices are normalised to 0-based.
    """
    out: Dict[Tuple[int, int], float] = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            s = line.split()
            if not s:
                continue
            ncol = len(s)
            if ncol == 16:
                # Reference: 0-based indices; FrstIndex = (decoy_mean - native)/std.
                i, j = int(s[0]), int(s[1])
                ne, dm, ds = float(s[13]), float(s[14]), float(s[15])
                fi = (dm - ne) / ds
            elif ncol == 19:
                # FrustraPy AWSEM layout: 1-based indices; FrstIndex is column 18.
                i, j = int(s[0]) - 1, int(s[1]) - 1
                fi = float(s[18])
            else:
                raise ValueError(
                    f"{path}: unexpected column count {ncol} (expected 16 reference "
                    "or 19 FrustraPy AWSEM); is this a tertiary_frustration.dat?"
                )
            out[(min(i, j), max(i, j))] = fi
    if not out:
        raise ValueError(f"{path}: no data rows found")
    return out


def _aligned_series(
    ref: Dict[Tuple[int, int], float], test: Dict[Tuple[int, int], float]
) -> Tuple[List[float], List[float], List[Tuple[int, int]], set, set]:
    """Align two ``{(i,j): FrstIndex}`` maps on their shared contact keys.

    Returns ``(ref_values, test_values, shared_keys, only_in_ref, only_in_test)``
    with the value lists in a fixed (sorted-key) order so the metric call is
    deterministic.
    """
    rk, tk = set(ref), set(test)
    shared = sorted(rk & tk)
    only_ref = rk - tk
    only_test = tk - rk
    rv = [ref[k] for k in shared]
    tv = [test[k] for k in shared]
    return rv, tv, shared, only_ref, only_test


# ---------------------------------------------------------------------------
# Tier 1: regenerate the post-processor output from the shipped logs.
# ---------------------------------------------------------------------------

def _regenerate_from_logs(ref_dir: str, n_decoys: int) -> str:
    """Run the in-container post-processor over the shipped reference logs and
    return the path to the written ``tertiary_frustration.dat`` (a temp file).

    No Rosetta: the logs already hold the per-residue-pair energies. This is the
    exact path ``tests/test_atomic_post.py`` gates.
    """
    import tempfile

    from frustrapy.backends import atomic_post as ap  # noqa: PLC0415

    native_pdb = os.path.join(ref_dir, "native.pdb")
    native_log = os.path.join(ref_dir, "native.log")
    decoy_logs = [os.path.join(ref_dir, f"{i}.log") for i in range(1, n_decoys + 1)]
    missing = [p for p in [native_pdb, native_log, *decoy_logs] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "missing reference logs/structure for tier-1 regeneration:\n  "
            + "\n  ".join(missing)
            + f"\n(set ATOMIC_FRUST_REF; looked under {ref_dir})"
        )
    out = tempfile.NamedTemporaryFile(
        prefix="atomic_parity_", suffix=".dat", delete=False
    ).name
    ap.write_tertiary_frustration_from_logs(out, native_pdb, native_log, decoy_logs)
    return out


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

def _print_metric_table(label: str, metrics: Dict[str, float]) -> None:
    print(f"\n=== {label} ===")
    print(f"  contacts (n)      : {int(metrics['n'])}")
    print(f"  Spearman          : {metrics['spearman']:.6f}")
    print(f"  Pearson R^2       : {metrics['r2']:.6f}")
    print(f"  max|delta|        : {metrics['max_abs']:.6e}")
    print(f"  RMSE              : {metrics['rmse']:.6e}")
    print(f"  class-agreement   : {metrics['class_agreement'] * 100:.2f}%")


def _print_triplicate_summary(per_metric: Dict[str, List[float]]) -> None:
    print("\n=== triplicate summary (mean +/- sample std across replicates) ===")
    for name in ("spearman", "r2", "max_abs", "rmse", "class_agreement"):
        vals = per_metric[name]
        stats = summarize_times(vals)
        std = stats["std"]
        print(f"  {name:16s}: {stats['mean']:.6f} +/- {std:.6f}  (n={stats['n']})")


# Tier-1 pass thresholds: the post-processor must reproduce the golden FrstIndex to
# within %8.3f print precision (half a ULP at 3 decimals is 5e-4), with perfect rank
# order and class agreement, over the identical contact set.
_GATE = {
    "min_spearman": 0.9999,
    "min_class_agreement": 1.0,
    "max_max_abs": 1e-3,
}


def _check_gate(metrics: Dict[str, float], n_only_ref: int, n_only_test: int) -> List[str]:
    failures: List[str] = []
    if n_only_ref or n_only_test:
        failures.append(
            f"contact set differs (only-in-ref={n_only_ref}, only-in-test={n_only_test})"
        )
    if metrics["spearman"] < _GATE["min_spearman"]:
        failures.append(f"Spearman {metrics['spearman']:.6f} < {_GATE['min_spearman']}")
    if metrics["class_agreement"] < _GATE["min_class_agreement"]:
        failures.append(
            f"class-agreement {metrics['class_agreement']:.4f} < {_GATE['min_class_agreement']}"
        )
    if not (metrics["max_abs"] < _GATE["max_max_abs"]):
        failures.append(f"max|delta| {metrics['max_abs']:.3e} >= {_GATE['max_max_abs']}")
    return failures


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--golden",
        action="store_true",
        help="tier 1: regenerate from the shipped reference logs and diff vs the "
        "committed golden tertiary_frustration.dat (runs in-container, no Rosetta).",
    )
    p.add_argument(
        "--reference",
        help="tier 2: path to the reference method's tertiary_frustration.dat "
        "(16-col) or a frozen FrustraPy atomic output (19-col).",
    )
    p.add_argument(
        "--test",
        nargs="+",
        help="tier 2: one or more AtomicBackend output tertiary_frustration.dat "
        "files. Pass >=3 seeded replicates for a triplicate mean/std.",
    )
    p.add_argument(
        "--mode",
        default="configurational",
        choices=["configurational", "mutational", "singleresidue"],
        help="frustration mode (selects the class-agreement cutoffs: 0.78 contact "
        "vs 0.58 single-residue). Default configurational.",
    )
    p.add_argument("--ref-dir", default=DEFAULT_REF_DIR, help="tier-1 shipped-logs dir.")
    p.add_argument("--n-decoys", type=int, default=50, help="tier-1 decoy count (golden=50).")
    p.add_argument(
        "--gate",
        action="store_true",
        help="exit non-zero if metrics fall below the tier-1 pass thresholds "
        "(implied by --golden).",
    )
    args = p.parse_args(argv)

    if args.golden:
        regenerated = _regenerate_from_logs(args.ref_dir, args.n_decoys)
        ref_map = _read_frstindex(GOLDEN_DAT)
        test_map = _read_frstindex(regenerated)
        rv, tv, shared, only_ref, only_test = _aligned_series(ref_map, test_map)
        metrics = compare(rv, tv, args.mode)
        print("TIER 1 (in-container, no Rosetta): post-processor vs committed golden")
        print(f"  golden    : {GOLDEN_DAT}")
        print(f"  regenerated: {regenerated}")
        print(f"  shared contacts: {len(shared)}  "
              f"(only-in-golden={len(only_ref)}, only-in-regenerated={len(only_test)})")
        _print_metric_table(f"FrstIndex parity ({args.mode})", metrics)
        failures = _check_gate(metrics, len(only_ref), len(only_test))
        if failures:
            print("\nTIER 1 GATE FAILED:")
            for f in failures:
                print(f"  - {f}")
            return 1
        print("\nTIER 1 GATE PASSED.")
        return 0

    if not (args.reference and args.test):
        p.error("provide --golden (tier 1), or both --reference and --test (tier 2).")

    print("TIER 2 (maintainer, needs PyRosetta): AtomicBackend vs reference method")
    print(f"  reference: {args.reference}")
    ref_map = _read_frstindex(args.reference)

    per_metric: Dict[str, List[float]] = {
        k: [] for k in ("spearman", "r2", "max_abs", "rmse", "class_agreement")
    }
    worst_setdiff = (0, 0)
    last_metrics = None
    for idx, test_path in enumerate(args.test, 1):
        test_map = _read_frstindex(test_path)
        rv, tv, shared, only_ref, only_test = _aligned_series(ref_map, test_map)
        metrics = compare(rv, tv, args.mode)
        last_metrics = metrics
        worst_setdiff = (
            max(worst_setdiff[0], len(only_ref)),
            max(worst_setdiff[1], len(only_test)),
        )
        _print_metric_table(
            f"replicate {idx}: {os.path.basename(test_path)} ({args.mode}), "
            f"shared={len(shared)}",
            metrics,
        )
        for k in per_metric:
            per_metric[k].append(metrics[k])

    if len(args.test) >= 2:
        _print_triplicate_summary(per_metric)
    if any(worst_setdiff):
        print(
            f"\nWARNING: contact set not identical across all replicates "
            f"(max only-in-ref={worst_setdiff[0]}, max only-in-test={worst_setdiff[1]}). "
            "The atomic contact set is geometry-only and should match exactly; a "
            "difference points to a structure-preparation discrepancy, not stochastic "
            "repack."
        )

    if args.gate and last_metrics is not None:
        failures = _check_gate(last_metrics, *worst_setdiff)
        if failures:
            print("\nGATE FAILED:")
            for f in failures:
                print(f"  - {f}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
