"""Benchmark driver: each frustration mode x {lammps, native-CPU @ N cores, metal}.

Writes a JSON with per-(machine,backend,mode,threads) wall-time, speedup, and parity
Spearman. Container is x86_64-emulated on an arm64 host, so absolute times there are
indicative only; ratios (core-scaling shape, lammps-vs-native) are valid within a machine.
Run on each machine separately; combine + plot afterwards.

Usage:
  python native/bench/run_matrix.py --pdb 3pgk.pdb --machine container --threads 1,2,4,8,14 \
      --repeats 5 --modes configurational,mutational,singleresidue --out bench_container.json
  python native/bench/run_matrix.py --pdb 3pgk.pdb --machine host-arm64 --threads 1,2,4,8 \
      --repeats 5 --modes ... --no-lammps --metal --out bench_host.json
"""
import argparse, json, os, platform, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import cross_backend as xb  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pdb", required=True)
    p.add_argument("--machine", required=True, help="label, e.g. container-x86_64-emulated / host-arm64")
    p.add_argument("--threads", default="1,2,4,8,14")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seq-dist", type=int, default=12, choices=[3, 12])
    p.add_argument("--modes", default="configurational,mutational,singleresidue")
    p.add_argument("--no-lammps", action="store_true", help="skip the lammps reference (broken on host macOS)")
    p.add_argument("--metal", action="store_true", help="also time the Metal backend (host w/ Metal build)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    threads = sorted({max(1, int(t)) for t in a.threads.split(",") if t.strip()})
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    n_ca = sum(1 for ln in open(a.pdb) if ln.startswith("ATOM") and ln[12:16].strip() == "CA")

    meta = {
        "machine": a.machine, "uname_machine": platform.machine(),
        "cpu_count": xb._cpu_count(), "pdb": os.path.basename(a.pdb), "n_ca": n_ca,
        "seq_dist": a.seq_dist, "repeats": a.repeats, "threads": threads, "modes": modes,
        "native_built": xb._native_built(), "has_metal": xb._has_metal(), "has_cuda": xb._has_cuda(),
        "available_backends": xb.available_backend_labels(),
    }
    print("META:", json.dumps(meta), flush=True)

    records = []
    for mode in modes:
        print(f"\n===== mode={mode} =====", flush=True)
        # Isolated kernel core-scaling (the clean scaling signal: prep done once, excluded).
        try:
            core_rows = xb.benchmark_core_scaling(
                a.pdb, mode=mode, threads_list=threads, repeats=a.repeats, seq_dist=a.seq_dist)
            for r in core_rows:
                rec = {"machine": a.machine, "mode": mode, "kind": "kernel",
                       "backend": "native-cpu", "threads": r.threads,
                       "wall_s": r.wall_s, "speedup": r.speedup, "bit_identical": r.bit_identical}
                records.append(rec)
                print("  kernel", mode, f"t={r.threads}", f"{r.wall_s*1000:.2f}ms", f"x{r.speedup:.2f}", flush=True)
        except Exception as e:
            print("  kernel-scaling FAILED:", repr(e), flush=True)

        # End-to-end (full pipeline) incl. lammps reference + native-cpu sweep (+ metal/cuda if present).
        if not a.no_lammps:
            try:
                rows = xb.run_cross_backend_benchmark(
                    a.pdb, mode=mode, threads_list=threads, repeats=a.repeats, seq_dist=a.seq_dist)
                for r in rows:
                    rec = {"machine": a.machine, "mode": mode, "kind": "e2e",
                           "backend_label": r.backend, "threads": r.threads,
                           "wall_s": r.wall_s, "speedup": r.speedup,
                           "spearman": r.spearman, "n_units": r.n_units}
                    records.append(rec)
                    print("  e2e   ", mode, r.backend, f"{r.wall_s:.3f}s", f"x{r.speedup:.2f}", f"sp={r.spearman:.4f}", flush=True)
            except Exception as e:
                print("  e2e cross-backend FAILED:", repr(e), flush=True)

        # Metal end-to-end (host only; native-cpu serial as the speedup base + parity ref).
        if a.metal and xb._has_metal():
            try:
                import tempfile
                root = tempfile.mkdtemp(prefix="fp_metal_")
                os.environ["FRUSTRAPY_NATIVE_USE_METAL"] = "1"
                base, _ = (None, None)
                def _t(fn):
                    import time, statistics
                    s = []
                    path = ""
                    for _ in range(a.repeats):
                        t0 = time.perf_counter(); path = fn(); s.append(time.perf_counter()-t0)
                    return statistics.median(s), path
                wall, path = _t(lambda: xb._run_once(a.pdb, mode, "native", os.path.join(root, "metal"), a.seq_dist))
                os.environ.pop("FRUSTRAPY_NATIVE_USE_METAL", None)
                # parity vs native-cpu serial (itself parity-gated vs lammps in the container)
                os.environ["FRUSTRAPY_NATIVE_THREADS"] = "1"
                _, cpath = _t(lambda: xb._run_once(a.pdb, mode, "native", os.path.join(root, "cpu1"), a.seq_dist))
                sp = xb._spearman(xb._read_frst(cpath, mode), xb._read_frst(path, mode))
                records.append({"machine": a.machine, "mode": mode, "kind": "e2e",
                                "backend_label": "metal", "threads": 1, "wall_s": wall,
                                "speedup": None, "spearman": sp})
                print("  metal ", mode, f"{wall:.3f}s", f"parity_vs_cpu_sp={sp:.4f}", flush=True)
            except Exception as e:
                print("  metal FAILED:", repr(e), flush=True)

    out = {"meta": meta, "records": records}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\nWROTE", a.out, "with", len(records), "records", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
