#!/usr/bin/env python3
"""Maintainer GPU parity + timing harness for the native frustration core (N3).

Run on a GPU machine (Colab / cluster) AFTER building the extension with CUDA:

    pip install ./native -C cmake.define.FRUSTRAPY_NATIVE_CUDA=ON

It (1) prepares a structure through the normal FrustraPy pipeline (so the job dir
holds the cleaned PDB + coefficient + gamma files), (2) calls the native core twice
on the identical inputs -- CPU then CUDA -- and (3) reports max column differences,
FrstIndex Spearman, and wall-clock for each. It asserts parity (Spearman >= 0.99,
energy within tolerance); it prints timings but asserts nothing about speed -- the
maintainer records the measured numbers. No timing is fabricated in the repo.

Usage:
    python native/colab/bench_cuda.py PDB [--mode mutational] [--seq-dist 12] [--repeats 3]
"""

import argparse
import os
import shutil
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb")
    ap.add_argument("--mode", default="mutational",
                    choices=["configurational", "mutational", "singleresidue"])
    ap.add_argument("--seq-dist", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workdir", default="/tmp/frustra_native_bench")
    args = ap.parse_args()

    import numpy as np
    import frustrapy
    import frustrapy_native as fn
    from frustrapy.backends.native import _parse_structure, _read_coeff, _read_gammas

    if not fn.has_cuda():
        print("ERROR: frustrapy_native was built WITHOUT CUDA. Rebuild with "
              "-C cmake.define.FRUSTRAPY_NATIVE_CUDA=ON on a GPU machine.", file=sys.stderr)
        return 2

    # 1) Prepare the structure through the pipeline (produces the job dir inputs).
    if os.path.exists(args.workdir):
        shutil.rmtree(args.workdir)
    os.makedirs(args.workdir)
    frustrapy.calculate_frustration(
        pdb_file=args.pdb, mode=args.mode, results_dir=args.workdir, graphics=False,
        visualization=False, debug="ERROR", backend="native", seq_dist=args.seq_dist,
    )
    base = os.path.splitext(os.path.basename(args.pdb))[0]
    job_dir = os.path.join(args.workdir, f"{base}.done")

    coord, res_type, chain_id, seqid, _, _ = _parse_structure(os.path.join(job_dir, f"{base}.pdb"))
    coeff = _read_coeff(os.path.join(job_dir, "fix_backbone_coeff.data"))
    gd, gw, gp, bg = _read_gammas(job_dir)
    common = dict(
        well_kappa=coeff["well_kappa"], kappa_sigma=coeff["kappa_sigma"],
        treshold=coeff["treshold"], well_r_min0=coeff["well_r_min0"],
        well_r_max0=coeff["well_r_max0"], well_r_min1=coeff["well_r_min1"],
        well_r_max1=coeff["well_r_max1"], burial_kappa=coeff["burial_kappa"],
        k_burial=coeff["k_burial"], contact_cutoff=coeff["contact_cutoff"],
        contact_min_sep=coeff["contact_min_sep"], seq_dist=args.seq_dist,
        n_decoys=coeff["n_decoys"], seed=1,
    )

    def call(use_cuda):
        return fn.compute_frustration(
            coord, res_type, chain_id, seqid, gd, gw, gp, bg, coeff["mode"],
            use_cuda=use_cuda, **common,
        )

    def timeit(use_cuda):
        best = float("inf")
        out = None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            out = call(use_cuda)
            best = min(best, time.perf_counter() - t0)
        return out, best

    print(f"# {base} mode={args.mode} seq_dist={args.seq_dist} n_res={len(coord)} "
          f"n_decoys={coeff['n_decoys']}")
    cpu, t_cpu = timeit(False)
    gpu, t_gpu = timeit(True)

    # Parity (CPU is the validated reference).
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    max_native = float(np.max(np.abs(cpu["native_energy"] - gpu["native_energy"])))
    max_decoy = float(np.max(np.abs(cpu["decoy_energy"] - gpu["decoy_energy"])))
    max_sd = float(np.max(np.abs(cpu["sd_energy"] - gpu["sd_energy"])))
    sp = spearman(cpu["frst_index"], gpu["frst_index"])
    print(f"parity  max|dNative|={max_native:.2e} max|dDecoy|={max_decoy:.2e} "
          f"max|dSD|={max_sd:.2e} SpearmanFI={sp:.5f}")
    print(f"timing  CPU best={t_cpu*1e3:.2f} ms   CUDA best={t_gpu*1e3:.2f} ms   "
          f"speedup={t_cpu / t_gpu:.2f}x  (record these; not committed)")

    # The decoy stream is host-generated with the exact glibc sequence, so CPU and GPU
    # should agree to floating-point reduction order (loosen if the GPU reduction reorders).
    assert sp >= 0.99, f"FrstIndex Spearman below gate: {sp}"
    assert max_native < 1e-6, f"native energy mismatch CPU vs CUDA: {max_native}"
    print("PARITY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
