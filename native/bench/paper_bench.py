"""Paper-grade size-series benchmark, centered on the deep mutational saturation scan.

Headline workload: single-residue saturation mutational scanning = for each of N
positions, mutate to all 20 amino acids and recompute frustration (N*20 per-variant
calculations). This is FrustraPy's most expensive real workflow and the FrustraMPNN-style
proteome-scale use case. Each per-variant unit is one frustration calculation on the
(mutated) structure, so DMS-scan cost = N * 20 * (per-variant calc).

We measure the per-variant calc at three time-bases on ONE machine (host arm64):
  kernel-only  : the AWSEM compute ceiling (frustrapy_native.compute_frustration).
  end-to-end   : what a user feels per mutant (prep + parse + gamma + compute).
and project the full DMS scan = N*20*per-variant. Backends: native CPU x1, native CPU
xAll, Metal GPU. Reps discard the cold first run; median + IQR reported. Parity vs the
CPU x1 frozen reference per size: Spearman, Pearson R^2, max|delta|, RMSE, class-agreement.

Apples-to-apples: same protein, same n_decoys, same algorithm; backends differ only in
implementation (~1e-6 agreement confirms this). AF2 monomers give a clean size axis.
"""
import argparse, glob, json, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import cross_backend as xb           # noqa: E402
from bench import metrics as M                  # noqa: E402
import frustrapy_native as nat                  # noqa: E402

N_AA = 20  # saturation: 20 amino acids per position


def timed(fn, reps, discard=1):
    samples, out = [], None
    for i in range(reps + discard):
        t0 = time.perf_counter(); out = fn(); dt = time.perf_counter() - t0
        if i >= discard:
            samples.append(dt)
    return samples, out


def n_ca(pdb):
    return sum(1 for ln in open(pdb) if ln.startswith("ATOM") and ln[12:16].strip() == "CA")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdbs", nargs="+", required=True)
    p.add_argument("--mode", default="configurational",
                   help="per-variant frustration mode (FrustraPy mutate_res default = configurational)")
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--seq-dist", type=int, default=12)
    p.add_argument("--out", required=True)
    p.add_argument("--max-serial-res", type=int, default=99999,
                   help="skip CPU x1 above this size (serial too slow to time many reps)")
    a = p.parse_args()

    import platform
    recs = []
    pdbs = sorted(a.pdbs, key=n_ca)
    for pdb in pdbs:
        N = n_ca(pdb)
        tag = os.path.basename(pdb)
        try:
            kw = xb._prepare_core_inputs(pdb, a.seq_dist, f"/tmp/pb_{tag}")
        except Exception as e:
            print(f"[{tag} N={N}] prep FAILED: {repr(e)[:120]}", flush=True)
            continue
        reps = a.reps if N <= 700 else max(3, a.reps // 2)
        # frozen reference = CPU x1 output (same machine)
        ref = None
        row = {"pdb": tag, "n_res": N}
        backends = [("cpu_x1", dict(n_threads=1)), ("cpu_xAll", dict(n_threads=0)),
                    ("metal", dict(use_metal=True))]
        for name, kwargs in backends:
            if name == "cpu_x1" and N > a.max_serial_res:
                continue
            try:
                samples, out = timed(lambda: nat.compute_frustration(mode=a.mode, **kwargs, **kw), reps)
                frst = list(out["frst_index"])
                if name == "cpu_x1" and ref is None:
                    ref = frst
                ts = M.summarize_times(samples)
                met = M.compare(ref, frst, a.mode) if ref is not None else {}
                rec = {"pdb": tag, "n_res": N, "mode": a.mode, "backend": name,
                       "kernel_median_s": ts["median"], "kernel_q1_s": ts["q1"], "kernel_q3_s": ts["q3"],
                       "reps": ts["n"], "metrics": met,
                       "dms_scan_proj_s": N * N_AA * ts["median"]}
                recs.append(rec)
                cell = (f"[{tag} N={N}] {name:9s} kernel={ts['median']*1000:8.2f}ms "
                        f"DMS_proj={N*N_AA*ts['median']:8.1f}s")
                if met:
                    cell += (f"  Sp={met['spearman']:.4f} R2={met['r2']:.4f} "
                             f"maxd={met['max_abs']:.2e} rmse={met['rmse']:.2e} cls={met['class_agreement']:.4f}")
                print(cell, flush=True)
            except Exception as e:
                print(f"[{tag} N={N}] {name} FAILED: {repr(e)[:140]}", flush=True)
    meta = {"machine": "host-arm64", "uname": platform.machine(), "mode": a.mode,
            "reps": a.reps, "n_aa": N_AA, "cores": nat.effective_threads(0),
            "has_metal": nat.has_metal(), "has_openmp": nat.has_openmp(),
            "note": "DMS-scan = N*20*per-variant kernel time; per-variant = one frustration calc (mutate_res model)"}
    json.dump({"meta": meta, "records": recs}, open(a.out, "w"), indent=2)
    print("\nWROTE", a.out, "with", len(recs), "records", flush=True)


if __name__ == "__main__":
    main()
