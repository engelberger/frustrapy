"""Parallel-efficiency (CPU scaling) + bottleneck attribution.

Two questions the maintainer asked:
  1. How does the kernel scale as cores are added? Ideal is directly proportional
     (efficiency = speedup/cores = 1.0). We sweep a fine thread grid, report
     speedup + efficiency, and fit Amdahl's law speedup(p)=1/(s+(1-s)/p) to recover
     the serial fraction s (the ceiling on multicore speedup).
  2. Where does end-to-end time actually go, so we know whether to optimise IO / data
     prep rather than the kernel? We decompose one per-variant calculation into:
       prep      = _prepare_core_inputs (PdbCoords2Lammps subprocess + parse + gamma read)
       density   = configurational kernel (5A density + native energy; the light part)
       decoy     = mutational kernel minus configurational (the decoy ensemble cost)
       io+glue   = end-to-end (_run_once) minus (prep + kernel)
     Everything is logged so bottlenecks are visible, not inferred.
"""
import argparse, json, math, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import cross_backend as xb           # noqa: E402
import frustrapy_native as nat                  # noqa: E402


def med_std(fn, reps, discard=1):
    s = []
    out = None
    for i in range(reps + discard):
        t0 = time.perf_counter(); out = fn(); dt = time.perf_counter() - t0
        if i >= discard:
            s.append(dt)
    return statistics.median(s), (statistics.stdev(s) if len(s) > 1 else 0.0), out


def amdahl_serial_fraction(threads, speedups):
    """Least-squares fit of s in speedup(p)=1/(s+(1-s)/p) over a grid of (p, speedup)."""
    best_s, best_err = 0.0, float("inf")
    s = 0.0
    while s <= 1.0:
        err = sum((1.0 / (s + (1 - s) / p) - sp) ** 2 for p, sp in zip(threads, speedups))
        if err < best_err:
            best_err, best_s = err, s
        s += 0.001
    max_speedup = 1.0 / best_s if best_s > 0 else float("inf")
    return best_s, max_speedup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--threads", default="1,2,3,4,6,8,10,12,14")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    threads = [int(t) for t in a.threads.split(",")]
    N = sum(1 for ln in open(a.pdb) if ln.startswith("ATOM") and ln[12:16].strip() == "CA")

    # --- bottleneck: time prep explicitly, then the kernels, then end-to-end ---
    t0 = time.perf_counter()
    kw = xb._prepare_core_inputs(a.pdb, 12, "/tmp/sb_prep")
    prep_s = time.perf_counter() - t0
    phases = {}
    for mode in ("configurational", "mutational", "singleresidue"):
        m, sd, _ = med_std(lambda: nat.compute_frustration(mode=mode, n_threads=0, **kw), a.reps)
        phases[mode + "_kernel_allcores_s"] = m
        m1, _, _ = med_std(lambda: nat.compute_frustration(mode=mode, n_threads=1, **kw), a.reps)
        phases[mode + "_kernel_serial_s"] = m1
    # end-to-end for one mutational variant (full pipeline incl IO)
    os.environ["FRUSTRAPY_NATIVE_THREADS"] = "0"
    e2e_m, e2e_sd, _ = med_std(lambda: xb._run_once(a.pdb, "mutational", "native", "/tmp/sb_e2e", 12), a.reps)
    os.environ.pop("FRUSTRAPY_NATIVE_THREADS", None)
    density = phases["configurational_kernel_allcores_s"]
    decoy = max(0.0, phases["mutational_kernel_allcores_s"] - density)
    io_glue = max(0.0, e2e_m - prep_s - phases["mutational_kernel_allcores_s"])
    bottleneck = {
        "n_res": N, "prep_s": prep_s, "density_kernel_s": density, "decoy_kernel_s": decoy,
        "io_glue_s": io_glue, "e2e_mutational_s": e2e_m, "e2e_std_s": e2e_sd,
        "kernel_total_s": phases["mutational_kernel_allcores_s"],
        "phases": phases,
        "note": "prep = PdbCoords2Lammps subprocess + parse + gamma; per-variant e2e is dominated by "
                "whichever of {prep, kernel, io+glue} is largest -- if prep/io dominate, optimise data "
                "prep (reuse scaffolding across the N*20 variants) not the kernel.",
    }

    # --- CPU scaling efficiency on the heavy mode (mutational) ---
    serial = None
    scaling = []
    for p in threads:
        m, sd, _ = med_std(lambda p=p: nat.compute_frustration(mode="mutational", n_threads=p, **kw), a.reps)
        if serial is None:
            serial = m
        sp = serial / m
        scaling.append({"threads": p, "mean_s": m, "std_s": sd, "speedup": sp, "efficiency": sp / p})
        print(f"  threads={p:2d}  {m*1000:8.2f}ms  speedup={sp:5.2f}x  efficiency={sp/p:.2f}", flush=True)
    s_frac, max_sp = amdahl_serial_fraction([r["threads"] for r in scaling], [r["speedup"] for r in scaling])
    print(f"\nAmdahl serial fraction s={s_frac:.4f} -> ceiling speedup ~{max_sp:.1f}x", flush=True)
    print("bottleneck (allcores):", {k: round(v, 4) for k, v in bottleneck.items()
                                      if isinstance(v, float)}, flush=True)

    out = {"pdb": os.path.basename(a.pdb), "n_res": N, "cores": nat.effective_threads(0),
           "scaling_mutational": scaling, "amdahl_serial_fraction": s_frac,
           "amdahl_ceiling_speedup": max_sp, "bottleneck": bottleneck}
    json.dump(out, open(a.out, "w"), indent=2)
    print("\nWROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
