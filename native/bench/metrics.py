"""Parity metrics for backend/flag comparisons against a frozen reference.

All comparisons are apples-to-apples: same protein, same mode, same n_decoys, same
algorithm; the backends differ only in implementation. Between bit-stable builds
Spearman and R^2 saturate at 1.0 (that saturation IS the correctness proof) -- the
discriminating metrics on aggressive compiler flags are max|delta|, RMSE, and
class-agreement (what a user actually acts on: the -1/0.78 contact classes, 0.58
single-residue plot cutoff).
"""
import math
from typing import Sequence, Dict

# Frustration classes. Contacts (configurational/mutational): <=-1 highly, >=0.78 minimal.
# Single-residue PLOT cutoff is 0.58 (distinct -- see cutoff-audit-vs-r).
def _contact_class(x: float) -> int:
    if x <= -1.0:
        return -1
    if x >= 0.78:
        return 1
    return 0


def _single_class(x: float) -> int:
    if x <= -1.0:
        return -1
    if x >= 0.58:
        return 1
    return 0


def _rank(xs: Sequence[float]):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    ma = sum(a) / n
    mb = sum(b) / n
    sa = sum((x - ma) ** 2 for x in a)
    sb = sum((x - mb) ** 2 for x in b)
    if sa == 0 or sb == 0:
        return float("nan")
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / math.sqrt(sa * sb)


def compare(ref: Sequence[float], test: Sequence[float], mode: str) -> Dict[str, float]:
    """All five metrics for `test` vs the frozen `ref` (same units, same order)."""
    n = min(len(ref), len(test))
    ref = list(ref[:n]); test = list(test[:n])
    ra, rb = _rank(ref), _rank(test)
    spearman = _pearson(ra, rb)
    pearson = _pearson(ref, test)
    r2 = pearson * pearson if pearson == pearson else float("nan")
    diffs = [abs(ref[i] - test[i]) for i in range(n)]
    max_abs = max(diffs) if diffs else float("nan")
    rmse = math.sqrt(sum(d * d for d in diffs) / n) if n else float("nan")
    cls = _single_class if mode == "singleresidue" else _contact_class
    agree = sum(1 for i in range(n) if cls(ref[i]) == cls(test[i])) / n if n else float("nan")
    return {
        "n": n, "spearman": spearman, "pearson": pearson, "r2": r2,
        "max_abs": max_abs, "rmse": rmse, "class_agreement": agree,
    }


def summarize_times(samples: Sequence[float]) -> Dict[str, float]:
    """Mean + sample std + median + IQR over timed repeats (discard the cold rep first).

    Triplicate (n>=3) gives a usable sample std for error bars and confident extrapolation.
    """
    s = sorted(samples)
    n = len(s)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"),
                "q1": float("nan"), "q3": float("nan"), "n": 0}
    mean = sum(s) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in s) / (n - 1)) if n > 1 else 0.0
    cv = (std / mean) if mean else float("nan")
    def q(p):
        idx = p * (n - 1)
        lo = int(math.floor(idx)); hi = int(math.ceil(idx))
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)
    return {"mean": mean, "std": std, "cv": cv, "median": q(0.5),
            "q1": q(0.25), "q3": q(0.75), "min": s[0], "max": s[-1], "n": n}
