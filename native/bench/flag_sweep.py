"""Compiler-flag sweep for the native CPU core: speed vs numerical parity.

For a parity-gated tool the headline question is whether aggressive flags (notably
-ffast-math) buy speed at the cost of reproducibility. We rebuild the native core under
several flag sets, benchmark the mutational kernel (triplicate) at serial and all-cores,
and compare every variant's FrstIndex against a single FROZEN -O2 reference (never against
itself). Metrics: Spearman, R2, max|delta|, RMSE, class-agreement. -ffast-math is checked
at BOTH thread counts because it can also break the bit-stable OpenMP reduction.

Each flag is verified to actually reach the compiler (grep the verbose build) so a mistyped
define cannot masquerade as "no effect". -mcpu=native makes a non-portable binary; reported
as its own row, never the silent default.

Run on the host (native arm64, real timings). The build defines for Metal/OpenMP come from
the environment (FLAG_SWEEP_EXTRA_DEFINES), so the same script works on an x86 box with
different -march flags.
"""
import argparse, json, os, subprocess, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # native/ -> import bench

# (name, extra CXX flags appended after the Release -O3, portable?)
DEFAULT_CONFIGS = [
    ("O2", "-O2", True),
    ("O3", "-O3", True),
    ("O3_native", "-O3 -mcpu=native", False),
    ("O3_unroll", "-O3 -funroll-loops", True),
    ("O3_ffast_math", "-O3 -ffast-math", True),
]

WORKER = r'''
import json, sys, time, statistics, os
sys.path.insert(0, "native")
from bench import cross_backend as xb
import frustrapy_native as nat
pdbs = json.loads(sys.argv[1]); mode = sys.argv[2]; reps = int(sys.argv[3]); out = sys.argv[4]
res = []
for pdb in pdbs:
    kw = xb._prepare_core_inputs(pdb, 12, "/tmp/fs_"+os.path.basename(pdb))
    for thr in (1, 0):
        s = []; frst = None
        for i in range(reps+1):
            t0 = time.perf_counter(); o = nat.compute_frustration(mode=mode, n_threads=thr, **kw); dt = time.perf_counter()-t0
            if i >= 1: s.append(dt)
            frst = list(o["frst_index"])
        res.append({"pdb": os.path.basename(pdb), "threads": thr,
                    "mean_s": statistics.fmean(s), "std_s": (statistics.stdev(s) if len(s)>1 else 0.0),
                    "frst": frst})
json.dump(res, open(out, "w"))
'''


def build(flags, log):
    defines = os.environ.get("FLAG_SWEEP_EXTRA_DEFINES", "")
    cmd = (f'pip install --force-reinstall --no-deps -v ./native '
           f'-C cmake.define.CMAKE_CXX_FLAGS="{flags}" {defines}')
    with open(log, "w") as f:
        p = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def flag_reached(log, flags):
    """Confirm at least one distinctive token of `flags` appears in a compile line."""
    token = [t for t in flags.split() if t not in ("-O2", "-O3")]
    token = token[0] if token else flags.split()[0]
    try:
        txt = open(log, errors="ignore").read()
    except OSError:
        return False
    return any(token in ln and "core.cpp" in ln for ln in txt.splitlines()) or token in txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdbs", nargs="+", required=True)
    ap.add_argument("--mode", default="mutational")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from bench import metrics as M

    cfg_env = os.environ.get("FLAG_SWEEP_CONFIGS")
    configs = [tuple(c) for c in json.loads(cfg_env)] if cfg_env else DEFAULT_CONFIGS
    tmp = tempfile.mkdtemp(prefix="flagsweep_")
    raw = {}
    for name, flags, portable in configs:
        log = os.path.join(tmp, f"build_{name}.log")
        print(f"\n=== building {name}: {flags} ===", flush=True)
        rc = build(flags, log)
        reached = flag_reached(log, flags)
        if rc != 0:
            print(f"  BUILD FAILED rc={rc} (flag_reached={reached}); skipping", flush=True)
            continue
        print(f"  built ok; flag_reached_compiler={reached}", flush=True)
        frag = os.path.join(tmp, f"frag_{name}.json")
        rc2 = subprocess.run([sys.executable, "-c", WORKER, json.dumps(a.pdbs), a.mode,
                              str(a.reps), frag]).returncode
        if rc2 != 0 or not os.path.exists(frag):
            print(f"  worker FAILED rc={rc2}; skipping", flush=True)
            continue
        raw[name] = {"flags": flags, "portable": portable, "reached": reached,
                     "results": json.load(open(frag))}
        for r in raw[name]["results"]:
            print(f"  {name:14s} {r['pdb']:16s} t={r['threads']:<2d} "
                  f"{r['mean_s']*1000:8.2f}+/-{r['std_s']*1000:.2f}ms", flush=True)

    # frozen reference = O2, per (pdb, threads)
    ref = {}
    for r in raw.get("O2", {}).get("results", []):
        ref[(r["pdb"], r["threads"])] = r["frst"]
    records = []
    for name, blk in raw.items():
        for r in blk["results"]:
            key = (r["pdb"], r["threads"])
            met = M.compare(ref[key], r["frst"], a.mode) if key in ref else {}
            base = None
            for rr in raw.get("O3", {}).get("results", []):
                if rr["pdb"] == r["pdb"] and rr["threads"] == r["threads"]:
                    base = rr["mean_s"]
            records.append({"config": name, "flags": blk["flags"], "portable": blk["portable"],
                            "flag_reached_compiler": blk["reached"], "pdb": r["pdb"],
                            "threads": r["threads"], "mean_s": r["mean_s"], "std_s": r["std_s"],
                            "speedup_vs_O3": (base / r["mean_s"]) if base else None, "metrics": met})
    json.dump({"configs": [c[0] for c in configs], "records": records}, open(a.out, "w"), indent=2)
    print("\nWROTE", a.out, "with", len(records), "records", flush=True)


if __name__ == "__main__":
    main()
