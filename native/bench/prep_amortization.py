"""Prep-amortization micro-benchmark.

Measures the win from computing the per-structure data prep once and reusing it
across a saturation scan's variants, instead of redoing it per mutant.

For K variants the current scan pays K x (prep + kernel): every mutant re-runs the
PdbCoords2Lammps subprocess, re-parses the structure, re-reads the gammas, and runs
the backend. The amortized path pays prep_once + K x kernel: one PdbCoords2Lammps
run materializes the coefficient/gamma files, prepare_structure parses and caches the
geometry once, and each variant is then a res_type swap plus the energy kernel
(reusing the cached density and contact list).

This times the three components on a real structure and reports the amortized scan
time and the prep-amortization factor for a chosen scan size (positions x 20 amino
acids). The amortized per-variant FrstIndex is parity-checked against a fresh native
recompute so the speedup is reported only when the numbers are unchanged.

Run: python native/bench/prep_amortization.py --pdb tests/data/1zni.pdb --chain B --res 25
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
import warnings
from typing import Optional, Sequence

# Run the in-tree frustrapy (this worktree), not any globally installed copy.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# The 20 canonical identities, scan order (matches frustrapy.analysis.mutations).
AMINO_ACIDS = [
    "LEU", "ASP", "ILE", "ASN", "THR", "VAL", "ALA", "GLY", "GLU", "ARG",
    "LYS", "HIS", "GLN", "SER", "PRO", "PHE", "TYR", "MET", "TRP", "CYS",
]


def _median_std(fn, reps, discard=1):
    samples = []
    out = None
    for i in range(reps + discard):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        if i >= discard:
            samples.append(dt)
    return statistics.median(samples), (statistics.stdev(samples) if len(samples) > 1 else 0.0), out


def _wt_run(pdb_file, mode, chain, seq_dist, results_dir):
    """One full WT calculate_frustration (PdbCoords2Lammps + parse + gamma + kernel),
    optionally chain-restricted as the saturation scan is. Returns the Pdb."""
    import frustrapy  # noqa: PLC0415

    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)
    local = os.path.join(results_dir, os.path.basename(pdb_file))
    shutil.copy2(pdb_file, local)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdb, _, _, _ = frustrapy.calculate_frustration(
            pdb_file=local, mode=mode, chain=chain, results_dir=results_dir,
            graphics=False, visualization=False, debug="ERROR", backend="native",
            seq_dist=seq_dist,
        )
    return pdb


def _first_non_gly_residue(pdb, chain):
    """A reasonable default scan target: the first non-glycine residue in `chain`."""
    atom = pdb.atom
    sel = atom[(atom["chain"] == chain) & (atom["atom_name"] == "CA")]
    for _, row in sel.iterrows():
        if row["res_name"] != "GLY":
            return int(row["res_num"])
    return int(sel.iloc[0]["res_num"])


def run(pdb_file, chain, res, positions, reps, seq_dist, threads):
    from frustrapy.backends import native as nbk  # noqa: PLC0415

    root = tempfile.mkdtemp(prefix="fp_prep_amort_")
    try:
        # Pick the target chain/residue.
        probe = _wt_run(pdb_file, "singleresidue", chain, seq_dist, os.path.join(root, "probe"))
        if chain is None:
            chain = str(probe.atom.iloc[0]["chain"])
        if res is None:
            res = _first_non_gly_residue(probe, chain)

        # Component 1: one full WT calc (the per-variant cost the current scan pays).
        t_wt, t_wt_sd, pdb = _median_std(
            lambda: _wt_run(pdb_file, "singleresidue", chain, seq_dist,
                            os.path.join(root, "wt")),
            reps,
        )
        job, pbase = pdb.job_dir, pdb.pdb_base

        # Component 2: prepare_structure once (parse + gamma + geometry cache).
        t_prep, t_prep_sd, prep = _median_std(
            lambda: nbk.prepare_structure(job, pbase, seq_dist), reps
        )
        idx = prep.site_index(res, chain)

        # Component 3: one amortized variant (res_type swap + kernel, reuse geometry).
        def _variant(aa):
            return nbk.compute_variant_frustration(
                prep, "singleresidue", site=(res, chain), new_aa=aa, n_threads=threads
            )

        t_kernel, t_kernel_sd, _ = _median_std(lambda: _variant("ALA"), reps)

        # Parity: the amortized variant matches a fresh native recompute (no cached
        # geometry) bit-for-bit for a non-glycine swap. Report so the speedup is honest.
        import numpy as np  # noqa: PLC0415
        import frustrapy_native as fn  # noqa: PLC0415
        rt = np.array(prep.res_type, dtype=np.int32)
        rt[idx] = nbk._aa_to_res_type("ALA")
        gd, gw, gp, bg = prep.gammas
        fresh = fn.compute_frustration(
            coord=prep.coord, res_type=rt, chain_id=prep.chain_id, res_seqid=prep.seqid,
            gamma_direct=gd, gamma_water=gw, gamma_protein=gp, burial_gamma=bg,
            mode="singleresidue", well_kappa=prep.coeff["well_kappa"],
            kappa_sigma=prep.coeff["kappa_sigma"], treshold=prep.coeff["treshold"],
            well_r_min0=prep.coeff["well_r_min0"], well_r_max0=prep.coeff["well_r_max0"],
            well_r_min1=prep.coeff["well_r_min1"], well_r_max1=prep.coeff["well_r_max1"],
            burial_kappa=prep.coeff["burial_kappa"], k_burial=prep.coeff["k_burial"],
            contact_cutoff=prep.coeff["contact_cutoff"],
            contact_min_sep=prep.coeff["contact_min_sep"], seq_dist=seq_dist,
            n_decoys=prep.coeff["n_decoys"], seed=1, n_threads=threads,
        )
        amort = _variant("ALA")
        bit_identical = bool(np.array_equal(amort["frst_index"], fresh["frst_index"]))

        n_res = len(prep.resnames)
        k = positions * len(AMINO_ACIDS)
        old_total = k * t_wt
        new_total = t_wt + t_prep + k * t_kernel  # one WT run materializes the prep
        factor = old_total / new_total if new_total else float("nan")
        ceiling = t_wt / t_kernel if t_kernel else float("nan")

        result = {
            "pdb": os.path.basename(pdb_file), "chain": chain, "res": res,
            "n_res_chain": n_res, "positions": positions, "variants_K": k,
            "seq_dist": seq_dist, "threads": fn.effective_threads(threads),
            "t_full_wt_calc_s": t_wt, "t_full_wt_calc_std_s": t_wt_sd,
            "t_prepare_once_s": t_prep, "t_prepare_once_std_s": t_prep_sd,
            "t_amortized_kernel_s": t_kernel, "t_amortized_kernel_std_s": t_kernel_sd,
            "old_scan_total_s": old_total, "amortized_scan_total_s": new_total,
            "prep_amortization_factor": factor, "asymptotic_ceiling": ceiling,
            "amortized_bit_identical_to_recompute": bit_identical,
        }
        return result
    finally:
        shutil.rmtree(root, ignore_errors=True)


def format_report(r) -> str:
    lines = [
        f"prep-amortization micro-benchmark: {r['pdb']} chain {r['chain']} "
        f"res {r['res']} ({r['n_res_chain']} residues, seq_dist {r['seq_dist']}, "
        f"{r['threads']} thread(s))",
        "-" * 72,
        f"  full WT calc (prep + kernel)   {r['t_full_wt_calc_s']*1000:9.2f} ms "
        f"(+/- {r['t_full_wt_calc_std_s']*1000:.2f})",
        f"  prepare_structure once         {r['t_prepare_once_s']*1000:9.2f} ms "
        f"(+/- {r['t_prepare_once_std_s']*1000:.2f})",
        f"  amortized variant (kernel)     {r['t_amortized_kernel_s']*1000:9.2f} ms "
        f"(+/- {r['t_amortized_kernel_std_s']*1000:.2f})",
        "",
        f"  scan of {r['positions']} position(s) x 20 AA = {r['variants_K']} variants:",
        f"    current   K x (prep + kernel) = {r['old_scan_total_s']:8.2f} s",
        f"    amortized prep_once + K x kernel = {r['amortized_scan_total_s']:8.2f} s",
        f"    prep-amortization factor       {r['prep_amortization_factor']:8.1f}x "
        f"(asymptotic ceiling {r['asymptotic_ceiling']:.1f}x)",
        f"    amortized == fresh recompute   {r['amortized_bit_identical_to_recompute']}",
    ]
    return "\n".join(lines)


def _pick_targets(pdb, chain, positions):
    """First `positions` residues (by CA) in `chain` as (res_num, chain) targets."""
    atom = pdb.atom
    sel = atom[(atom["chain"] == chain) & (atom["atom_name"] == "CA")]
    resnums = list(dict.fromkeys(int(r) for r in sel["res_num"]))
    return [(r, chain) for r in resnums[:positions]]


def _read_scan_values(job_dir, targets, mode="singleresidue"):
    """Read a singleresidue scan's per-variant FrstIndex: {(res, chain, AA): value}."""
    md = os.path.join(job_dir, "MutationsData")
    vals = {}
    for res, chain in targets:
        with open(os.path.join(md, f"{mode}_Res{res}_threading_{chain}.txt")) as fh:
            for ln in fh.read().splitlines()[1:]:
                t = ln.split()
                vals[(res, chain, t[2])] = float(t[3])
    return vals


def run_measured_scan(pdb_file, chain, positions, seq_dist, n_cpus, root):
    """Measure the END-TO-END saturation scan both ways through the wired path:
    the per-variant native scan (amortize off, prep redone per mutant) and the
    amortized native scan (prep once per chain). No projection: both totals are
    measured wall times for the same set of variants, and the per-variant FrstIndex
    is diffed between the two so the speedup is reported only when the numbers agree.
    """
    import frustrapy  # noqa: PLC0415
    from frustrapy.analysis.mutations import mutate_res_scan_parallel  # noqa: PLC0415

    rd = os.path.join(root, "measured_scan")
    if os.path.exists(rd):
        shutil.rmtree(rd)
    os.makedirs(rd)
    local = os.path.join(rd, os.path.basename(pdb_file))
    shutil.copy2(pdb_file, local)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdb, _, _, _ = frustrapy.calculate_frustration(
            pdb_file=local, mode="singleresidue", chain=chain, results_dir=rd,
            graphics=False, visualization=False, debug="ERROR", backend="native",
            seq_dist=seq_dist,
        )
    target_chain = chain if chain is not None else str(pdb.atom.iloc[0]["chain"])
    targets = _pick_targets(pdb, target_chain, positions)
    k = len(targets) * len(AMINO_ACIDS)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        mutate_res_scan_parallel(pdb, targets=list(targets), split=True,
                                 method="threading", n_cpus=n_cpus,
                                 backend="native", amortize=False)
        t_before = time.perf_counter() - t0
        before_vals = _read_scan_values(pdb.job_dir, targets)

        t0 = time.perf_counter()
        mutate_res_scan_parallel(pdb, targets=list(targets), split=True,
                                 method="threading", n_cpus=n_cpus,
                                 backend="native", amortize=True)
        t_after = time.perf_counter() - t0
        after_vals = _read_scan_values(pdb.job_dir, targets)

    keys = sorted(set(before_vals) & set(after_vals))
    max_diff = max((abs(before_vals[k2] - after_vals[k2]) for k2 in keys), default=0.0)
    return {
        "pdb": os.path.basename(pdb_file), "chain": target_chain,
        "positions": len(targets), "variants_K": k, "seq_dist": seq_dist,
        "n_cpus": (1 if n_cpus == 1 else "auto"),
        "measured_before_per_variant_scan_s": t_before,
        "measured_amortized_scan_s": t_after,
        "measured_prep_amortization_factor": (t_before / t_after if t_after else float("nan")),
        "before_per_variant_s": t_before / k if k else float("nan"),
        "after_per_variant_s": t_after / k if k else float("nan"),
        "max_frstindex_diff_before_vs_after": max_diff,
    }


def format_measured(r) -> str:
    return "\n".join([
        f"measured end-to-end scan: {r['pdb']} chain {r['chain']}, "
        f"{r['positions']} position(s) x 20 AA = {r['variants_K']} variants, "
        f"seq_dist {r['seq_dist']}, n_cpus={r['n_cpus']}",
        "-" * 72,
        f"  per-variant native scan (prep per mutant) {r['measured_before_per_variant_scan_s']:8.2f} s "
        f"({r['before_per_variant_s']*1000:.1f} ms/variant)",
        f"  amortized native scan (prep once/chain)   {r['measured_amortized_scan_s']:8.2f} s "
        f"({r['after_per_variant_s']*1000:.1f} ms/variant)",
        f"  measured prep-amortization factor         {r['measured_prep_amortization_factor']:8.1f}x",
        f"  max |FrstIndex| diff before vs after       {r['max_frstindex_diff_before_vs_after']:.3e}",
    ])


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prep-amortization micro-benchmark (prep once vs prep per variant).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pdb", required=True, help="PDB file to benchmark")
    p.add_argument("--chain", default=None, help="target chain (default: first chain)")
    p.add_argument("--res", type=int, default=None,
                   help="target residue number (default: first non-glycine in the chain)")
    p.add_argument("--positions", type=int, default=10,
                   help="number of scan positions to project the scan total over")
    p.add_argument("--reps", type=int, default=3, help="timed repeats per component")
    p.add_argument("--seq-dist", type=int, default=12, choices=[3, 12])
    p.add_argument("--threads", type=int, default=1,
                   help="native kernel threads (1 = serial, 0 = all cores)")
    p.add_argument("--measured-scan", action="store_true",
                   help="measure the end-to-end scan both ways (no projection) "
                        "instead of the component micro-benchmark")
    p.add_argument("--n-cpus", type=int, default=1,
                   help="outer pool width for the measured scan (1 = serial)")
    p.add_argument("--out", default=None, help="write the result JSON here")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.measured_scan:
        root = tempfile.mkdtemp(prefix="fp_prep_measured_")
        n_cpus = None if args.n_cpus <= 0 else args.n_cpus  # 0 = auto (all cores)
        try:
            r = run_measured_scan(args.pdb, args.chain, args.positions,
                                  args.seq_dist, n_cpus, root)
        finally:
            shutil.rmtree(root, ignore_errors=True)
        print(format_measured(r))
    else:
        r = run(args.pdb, args.chain, args.res, args.positions, args.reps,
                args.seq_dist, args.threads)
        print(format_report(r))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(r, fh, indent=2)
        print(f"\nWROTE {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
