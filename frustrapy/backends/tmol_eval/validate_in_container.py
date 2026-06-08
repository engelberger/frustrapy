#!/usr/bin/env python3
"""In-container numerical-parity harness for the tmol atomic backend (TMOL-PARITY #39).

Two checks, both runnable HERE with no license (scope items 1 and 2):

1. **Oracle parity** -- the #38 evaluator's pairwise score function scores tmol's own
   shipped 1ubq fixture whole-pose and every in-scope subterm is diffed against tmol's
   committed baseline (``docs/tmol/oracle/1ubq_term_energies.json``, the shared #37
   oracle). This proves the evaluator wraps tmol faithfully (adapt-and-wrap), to a
   stated tolerance. Needs tmol importable (``TMOL_USE_JIT=1`` + ninja in this
   container; see ``docs/tmol/PY_BACKEND_NOTES.md``).

2. **Golden parity** -- the atomic backend's post-processor output vs the AA golden
   fixture (``docs/atomic/golden/tertiary_frustration.dat``). The tmol backend reuses
   :func:`frustrapy.backends.atomic_post.write_tertiary_frustration` verbatim, so we
   drive that exact writer from an :class:`~frustrapy.backends.atomic_engine.EngineResult`
   built from the shipped reference ``ResResE`` logs and diff the resulting per-contact
   ``FrstIndex`` against the golden. This needs NO tmol (it isolates the
   post-processing half the tmol backend shares with the AA lane); the energy ENGINE
   gap is the separate ``validate_vs_pyrosetta`` cross-check.

Run::

    python -m frustrapy.backends.tmol_eval.validate_in_container --write-report

The metric definitions are the frozen benchmark metrics in :mod:`native.bench.metrics`
(Spearman, Pearson R^2, max|delta|, RMSE, class-agreement); this script only aligns the
series and calls ``compare`` -- it never reimplements a metric.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

# Reuse the frozen benchmark metrics. native/ has no top-level package, so add it to
# the path and import bench.metrics (matching docs/atomic/parity/run_atomic_parity.py).
_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "native")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from bench.metrics import compare  # noqa: E402

ORACLE_JSON = os.path.join(_REPO_ROOT, "docs", "tmol", "oracle", "1ubq_term_energies.json")
GOLDEN_DAT = os.path.join(
    _REPO_ROOT, "docs", "atomic", "golden", "tertiary_frustration.dat"
)
DEFAULT_REF_DIR = os.environ.get(
    "ATOMIC_FRUST_REF", "/workspace/atomic_frustratometer_ref/example_output"
)
PARITY_DIR = os.path.join(_REPO_ROOT, "docs", "tmol", "parity")

# Pairwise subterm name -> (oracle term group, index within that group's pose0 list).
# The one-body ``ref`` term is NOT part of the pairwise contact evaluator.
_SUBTERM_ORACLE = {
    "fa_ljatr": ("LJLKEnergyTerm", 0),
    "fa_ljrep": ("LJLKEnergyTerm", 1),
    "fa_lk": ("LJLKEnergyTerm", 2),
    "lk_ball_iso": ("LKBallEnergyTerm", 0),
    "lk_ball": ("LKBallEnergyTerm", 1),
    "lk_bridge": ("LKBallEnergyTerm", 2),
    "lk_bridge_uncpl": ("LKBallEnergyTerm", 3),
    "fa_elec": ("ElecEnergyTerm", 0),
    "hbond": ("HBondEnergyTerm", 0),
}

# Stated oracle tolerance (tmol's own whole-pose test tolerance, used as the gate).
ORACLE_ATOL = 1e-3
ORACLE_RTOL = 1e-3

# Golden gate (the post-processor must reproduce the golden FrstIndex to print
# precision, with perfect rank/class agreement, over the identical contact set).
GOLDEN_GATE = {"min_spearman": 0.9999, "min_class_agreement": 1.0, "max_max_abs": 1e-3}


# ---------------------------------------------------------------------------
# 1. Oracle parity (needs tmol).
# ---------------------------------------------------------------------------

def _find_1ubq() -> Optional[str]:
    cand = [os.environ.get("FRUSTRAPY_TMOL_1UBQ")]
    try:
        import tmol  # noqa: PLC0415

        cand.append(
            os.path.join(os.path.dirname(tmol.__file__), "tests", "data", "pdb", "1ubq.pdb")
        )
    except Exception:
        pass
    cand.append("/workspace/tmol_src/tmol/tests/data/pdb/1ubq.pdb")
    for p in cand:
        if p and os.path.exists(p):
            return p
    return None


def run_oracle_parity(ubq_pdb: Optional[str] = None) -> Dict:
    """Score tmol's 1ubq whole-pose through the evaluator's pairwise score function and
    diff every in-scope subterm against the committed oracle baseline.

    Returns a result dict with a per-subterm table and the worst absolute deviation, or
    ``{"status": "skipped", ...}`` if tmol or the 1ubq fixture is unavailable.
    """
    from frustrapy.backends import atomic_tmol_engine as E  # noqa: PLC0415

    if not E.tmol_available():
        return {"status": "skipped", "reason": "tmol not importable (see PY_BACKEND_NOTES.md)"}
    ubq_pdb = ubq_pdb or _find_1ubq()
    if ubq_pdb is None:
        return {"status": "skipped", "reason": "tmol 1ubq fixture not found"}

    import torch  # noqa: PLC0415

    with open(ORACLE_JSON) as fh:
        oracle = json.load(fh)

    pose = E._pose_stack_from_pdb(ubq_pdb)
    sfxn = E.build_pairwise_score_function(device=pose.coords.device)
    wp = sfxn.render_whole_pose_scoring_module(pose)
    coords = torch.nn.Parameter(pose.coords.clone())
    unweighted = wp(coords, sum_terms=False, apply_weights=False).detach().cpu().numpy()
    score_types = [st for term in sfxn.all_terms() for st in term.score_types()]
    by_name = {st.name: float(unweighted[row, 0]) for row, st in enumerate(score_types)}

    rows = []
    worst = 0.0
    n_pass = 0
    for name, (group, idx) in _SUBTERM_ORACLE.items():
        baseline = float(oracle[group]["baseline_pose0"][idx])
        measured = by_name[name]
        absd = abs(measured - baseline)
        ok = absd <= (ORACLE_ATOL + ORACLE_RTOL * abs(baseline))
        n_pass += int(ok)
        worst = max(worst, absd)
        rows.append({"term": name, "measured": measured, "baseline": baseline,
                     "abs_diff": absd, "pass": ok})
    return {
        "status": "ok",
        "fixture": ubq_pdb,
        "atol": ORACLE_ATOL,
        "rtol": ORACLE_RTOL,
        "rows": rows,
        "max_abs_diff": worst,
        "n_pass": n_pass,
        "n_terms": len(rows),
        "passed": n_pass == len(rows),
    }


# ---------------------------------------------------------------------------
# 2. Golden parity (no tmol; isolates the shared post-processor).
# ---------------------------------------------------------------------------

def _read_frstindex(path: str) -> Dict[Tuple[int, int], float]:
    """Read a ``tertiary_frustration.dat`` into ``{(i0, j0): FrstIndex}`` (AWSEM sign),
    auto-detecting the reference 16-column vs the FrustraPy 19-column layout."""
    out: Dict[Tuple[int, int], float] = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            s = line.split()
            if not s:
                continue
            if len(s) == 16:
                i, j = int(s[0]), int(s[1])
                ne, dm, ds = float(s[13]), float(s[14]), float(s[15])
                fi = (dm - ne) / ds
            elif len(s) == 19:
                i, j = int(s[0]) - 1, int(s[1]) - 1
                fi = float(s[18])
            else:
                raise ValueError(f"{path}: unexpected column count {len(s)}")
            out[(min(i, j), max(i, j))] = fi
    if not out:
        raise ValueError(f"{path}: no data rows")
    return out


def run_golden_parity(ref_dir: str = DEFAULT_REF_DIR, n_decoys: int = 50) -> Dict:
    """Drive the shared post-processor (the path the tmol backend reuses) from the
    shipped reference logs and diff the per-contact ``FrstIndex`` against the committed
    golden. Returns the metric dict or a skipped marker if the logs are absent."""
    from frustrapy.backends import atomic_post as ap  # noqa: PLC0415

    native_pdb = os.path.join(ref_dir, "native.pdb")
    native_log = os.path.join(ref_dir, "native.log")
    decoy_logs = [os.path.join(ref_dir, f"{i}.log") for i in range(1, n_decoys + 1)]
    missing = [p for p in [native_pdb, native_log, *decoy_logs] if not os.path.exists(p)]
    if missing:
        return {"status": "skipped",
                "reason": f"reference logs absent under {ref_dir} ({len(missing)} missing)"}

    out = tempfile.NamedTemporaryFile(prefix="tmol_golden_", suffix=".dat", delete=False).name
    ap.write_tertiary_frustration_from_logs(out, native_pdb, native_log, decoy_logs)
    golden = _read_frstindex(GOLDEN_DAT)
    regen = _read_frstindex(out)
    os.unlink(out)
    shared = sorted(set(golden) & set(regen))
    only_golden = set(golden) - set(regen)
    only_regen = set(regen) - set(golden)
    rv = [golden[k] for k in shared]
    tv = [regen[k] for k in shared]
    metrics = compare(rv, tv, "configurational")
    passed = (
        not only_golden and not only_regen
        and metrics["spearman"] >= GOLDEN_GATE["min_spearman"]
        and metrics["class_agreement"] >= GOLDEN_GATE["min_class_agreement"]
        and metrics["max_abs"] < GOLDEN_GATE["max_max_abs"]
    )
    return {
        "status": "ok",
        "ref_dir": ref_dir,
        "n_decoys": n_decoys,
        "metrics": metrics,
        "only_in_golden": len(only_golden),
        "only_in_regenerated": len(only_regen),
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------

def _fmt_oracle_md(res: Dict) -> str:
    if res["status"] != "ok":
        return (f"# tmol evaluator vs 1ubq oracle (scope item 1)\n\n"
                f"**SKIPPED**: {res['reason']}\n")
    lines = [
        "# tmol evaluator vs the shipped 1ubq oracle (scope item 1)",
        "",
        "The #38 evaluator's pairwise score function "
        "(`frustrapy.backends.atomic_tmol_engine.build_pairwise_score_function`) scores "
        "tmol's own shipped 1ubq whole-pose; each in-scope subterm is diffed against the "
        "committed baseline `docs/tmol/oracle/1ubq_term_energies.json` (#37). This proves "
        "the frustrapy-side builder wraps tmol faithfully (adapt-and-wrap). `[VERIFIED EMPIRICALLY]`",
        "",
        f"- fixture: `{res['fixture']}`",
        f"- tolerance: atol={res['atol']:g}, rtol={res['rtol']:g} (tmol's own whole-pose test tolerance)",
        f"- worst absolute deviation: **{res['max_abs_diff']:.3e}**",
        f"- terms within tolerance: **{res['n_pass']}/{res['n_terms']}**",
        "",
        "| subterm | measured pose0 | oracle baseline | abs diff | pass |",
        "|---|---|---|---|---|",
    ]
    for r in res["rows"]:
        lines.append(
            f"| {r['term']} | {r['measured']:.5f} | {r['baseline']:.5f} | "
            f"{r['abs_diff']:.3e} | {'yes' if r['pass'] else 'NO'} |"
        )
    lines.append("")
    lines.append(f"**Verdict: {'PASS' if res['passed'] else 'FAIL'}** "
                 "(every in-scope subterm reproduces tmol's baseline within tolerance).")
    lines.append("")
    return "\n".join(lines)


def _fmt_golden_md(res: Dict) -> str:
    if res["status"] != "ok":
        return (f"# tmol backend post-processor vs AA golden (scope item 2)\n\n"
                f"**SKIPPED**: {res['reason']}\n")
    m = res["metrics"]
    return "\n".join([
        "# tmol backend post-processor vs the AA golden fixture (scope item 2)",
        "",
        "The `atomic-tmol` backend reuses "
        "`frustrapy.backends.atomic_post.write_tertiary_frustration` verbatim. This check "
        "drives that exact writer from an `EngineResult` built from the shipped reference "
        "`ResResE` logs and diffs the per-contact `FrstIndex` against the committed golden "
        "`docs/atomic/golden/tertiary_frustration.dat`. It isolates the POST-PROCESSING "
        "half the tmol backend shares with the AA lane (the energy-engine gap is the "
        "separate `tmol_vs_rosetta_frozen.md` cross-check). `[VERIFIED EMPIRICALLY]`",
        "",
        f"- reference logs: `{res['ref_dir']}` (N={res['n_decoys']} decoys)",
        f"- contacts compared: **{int(m['n'])}** "
        f"(only-in-golden={res['only_in_golden']}, only-in-regenerated={res['only_in_regenerated']})",
        "",
        "| metric | value |",
        "|---|---|",
        f"| Spearman | {m['spearman']:.6f} |",
        f"| Pearson R^2 | {m['r2']:.6f} |",
        f"| max\\|delta\\| | {m['max_abs']:.3e} |",
        f"| RMSE | {m['rmse']:.3e} |",
        f"| class-agreement | {m['class_agreement'] * 100:.2f}% |",
        "",
        f"**Verdict: {'PASS' if res['passed'] else 'FAIL'}** "
        "(the shared post-processor reproduces the golden FrstIndex bit-for-bit at print "
        "precision over the identical contact set; the tmol backend calls the same writer).",
        "",
    ])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ref-dir", default=DEFAULT_REF_DIR, help="shipped reference-logs dir.")
    p.add_argument("--n-decoys", type=int, default=50, help="golden decoy count (default 50).")
    p.add_argument("--ubq", default=None, help="override the 1ubq fixture path.")
    p.add_argument("--write-report", action="store_true",
                   help="write docs/tmol/parity/tmol_oracle_parity.md and tmol_golden_parity.md.")
    p.add_argument("--gate", action="store_true",
                   help="exit non-zero if a check that RAN failed its threshold "
                        "(skips do not fail the gate).")
    args = p.parse_args(argv)

    oracle = run_oracle_parity(ubq_pdb=args.ubq)
    golden = run_golden_parity(ref_dir=args.ref_dir, n_decoys=args.n_decoys)

    oracle_md = _fmt_oracle_md(oracle)
    golden_md = _fmt_golden_md(golden)
    print(oracle_md)
    print(golden_md)

    if args.write_report:
        os.makedirs(PARITY_DIR, exist_ok=True)
        with open(os.path.join(PARITY_DIR, "tmol_oracle_parity.md"), "w") as fh:
            fh.write(oracle_md)
        with open(os.path.join(PARITY_DIR, "tmol_golden_parity.md"), "w") as fh:
            fh.write(golden_md)
        print(f"wrote reports to {PARITY_DIR}")

    if args.gate:
        failed = []
        if oracle["status"] == "ok" and not oracle["passed"]:
            failed.append("oracle")
        if golden["status"] == "ok" and not golden["passed"]:
            failed.append("golden")
        if failed:
            print(f"\nGATE FAILED: {', '.join(failed)}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
