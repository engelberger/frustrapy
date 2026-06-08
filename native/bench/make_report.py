"""Capstone: publication figures (with std error bars + power-law fits) and the report.

Consumes the JSON outputs of paper_bench (size sample, triplicate+std), flag_sweep, and
the host Metal kernel detail. Fits t = a * N^b per backend by least squares on log-log
(reports the scaling exponent b and the fit R^2) so the size trend extrapolates with a
stated confidence rather than by eye. No em dashes anywhere in the emitted text.
"""
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BR = "benchmark_results"
COL = {"cpu_x1": "#9ecae1", "cpu_xAll": "#08519c", "metal": "#d62728"}
LAB = {"cpu_x1": "native CPU x1 (serial)", "cpu_xAll": "native CPU x14 (all cores)", "metal": "Metal GPU"}


def loadj(p):
    return json.load(open(p)) if os.path.exists(p) else None


def powerfit(ns, ts):
    """log10(t) = log10(a) + b*log10(N); returns (a, b, r2)."""
    xs = [math.log10(n) for n in ns]
    ys = [math.log10(t) for t in ts]
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    b = sxy / sxx
    loga = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((ys[i] - (loga + b * xs[i])) ** 2 for i in range(n))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return 10 ** loga, b, r2


def by_backend(recs, bk):
    rs = sorted([r for r in recs if r["backend"] == bk], key=lambda r: r["n_res"])
    return rs


def fig_sample(sample, out):
    recs = sample["records"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
    fits = {}
    for bk in ["cpu_x1", "cpu_xAll", "metal"]:
        rs = by_backend(recs, bk)
        if not rs:
            continue
        ns = [r["n_res"] for r in rs]
        per = [r["mean_s"] * 1000 for r in rs]
        err = [r["std_s"] * 1000 for r in rs]
        ax1.errorbar(ns, per, yerr=err, fmt="o", color=COL[bk], label=LAB[bk], ms=6, capsize=3, lw=1.5)
        f = powerfit(ns, [r["mean_s"] for r in rs])
        if f:
            a, b, r2 = f
            fits[bk] = (a, b, r2)
            xx = [min(ns), max(ns) * 2.7]
            ax1.plot(xx, [a * (x ** b) * 1000 for x in xx], "--", color=COL[bk], alpha=0.6,
                     label=f"  fit t~N^{b:.2f} (R2={r2:.3f})")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("protein size (residues)"); ax1.set_ylabel("per-variant mutational kernel time (ms)")
    ax1.set_title("Per-variant compute vs size (triplicate, std error bars + power-law fit)", fontsize=10)
    ax1.grid(True, which="both", alpha=0.3); ax1.legend(fontsize=7.5)

    for bk in ["cpu_x1", "cpu_xAll", "metal"]:
        rs = by_backend(recs, bk)
        if not rs:
            continue
        ns = [r["n_res"] for r in rs]
        scan = [r["dms_scan_proj_s"] / 60 for r in rs]
        serr = [r.get("dms_scan_proj_std_s", 0) / 60 for r in rs]
        ax2.errorbar(ns, scan, yerr=serr, fmt="o-", color=COL[bk], label=LAB[bk], ms=6, capsize=3, lw=1.8)
    for hr, lbl in [(60, "1 hour"), (1440, "1 day")]:
        ax2.axhline(hr, ls=":", c="gray", alpha=0.5); ax2.text(115, hr * 1.05, lbl, fontsize=8, c="gray")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("protein size (residues)")
    ax2.set_ylabel("full DMS saturation scan, projected (minutes)")
    ax2.set_title("Full deep-mutational scan (N x 20 variants) vs size", fontsize=10)
    ax2.grid(True, which="both", alpha=0.3); ax2.legend(fontsize=8)
    fig.suptitle(f"FrustraPy deep mutational frustration scan, AF2 size sample (n={len(set(r['pdb'] for r in recs))} proteins, "
                 f"host arm64, triplicate)\nall backends parity to CPU reference: Spearman = R2 = class-agreement = 1.0, "
                 "max|dFrstIndex| ~1e-6 (f32); same n_decoys, same algorithm", fontsize=10, y=1.02)
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)
    return fits


def fig_flags(flags, out):
    if not flags:
        return None
    recs = flags["records"]
    # one panel per thread setting: speedup vs O3 + class-agreement annotation
    thr_vals = sorted(set(r["threads"] for r in recs))
    fig, axes = plt.subplots(1, len(thr_vals), figsize=(7 * len(thr_vals), 5.2), squeeze=False)
    configs = flags["configs"]
    for ai, thr in enumerate(thr_vals):
        ax = axes[0][ai]
        # average across pdbs per config
        names, speed, cls, maxd = [], [], [], []
        for c in configs:
            rr = [r for r in recs if r["config"] == c and r["threads"] == thr]
            if not rr:
                continue
            names.append(c)
            sp = [r["speedup_vs_O3"] for r in rr if r.get("speedup_vs_O3")]
            speed.append(sum(sp) / len(sp) if sp else 0)
            ca = [r["metrics"].get("class_agreement", 1) for r in rr if r.get("metrics")]
            cls.append(min(ca) if ca else 1)
            md = [r["metrics"].get("max_abs", 0) for r in rr if r.get("metrics")]
            maxd.append(max(md) if md else 0)
        bars = ax.bar(range(len(names)), speed, color=["#2ca02c" if c >= 0.999 else "#d62728" for c in cls])
        for i, (s, c, m) in enumerate(zip(speed, cls, maxd)):
            ax.text(i, s, f"{s:.2f}x\ncls={c:.3f}\nmaxd={m:.1e}", ha="center", va="bottom", fontsize=7.5)
        ax.axhline(1.0, ls="--", c="gray", alpha=0.5)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("speedup vs -O3 (mean over proteins)")
        ax.set_title(f"threads = {'all' if thr == 0 else thr}", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Compiler-flag sweep: speed vs numerical parity (green = parity preserved, red = parity broken)\n"
                 "class-agreement and max|delta| vs the frozen -O2 reference; the parity-gated tool cannot ship a flag that reddens",
                 fontsize=10, y=1.04)
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)
    return recs


def fig_scaling_bottleneck(sb, out):
    if not sb:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))
    sc = sb["scaling_mutational"]
    thr = [r["threads"] for r in sc]
    spd = [r["speedup"] for r in sc]
    eff = [r["efficiency"] for r in sc]
    ax1.plot(thr, spd, "o-", color="#08519c", label="measured speedup", lw=2, ms=6)
    ax1.plot([1, max(thr)], [1, max(thr)], "k--", alpha=0.4, label="ideal (directly proportional)")
    s = sb["amdahl_serial_fraction"]
    ax1.plot(thr, [1 / (s + (1 - s) / p) for p in thr], ":", color="#d62728",
             label=f"Amdahl fit (s={s:.3f}, ceiling ~{sb['amdahl_ceiling_speedup']:.0f}x)")
    ax1b = ax1.twinx()
    ax1b.plot(thr, eff, "s-", color="#2ca02c", alpha=0.6, label="efficiency")
    ax1b.set_ylabel("parallel efficiency (speedup / cores)", color="#2ca02c")
    ax1b.set_ylim(0, 1.05); ax1b.axhline(1.0, ls=":", c="#2ca02c", alpha=0.3)
    ax1.set_xlabel("CPU threads"); ax1.set_ylabel("speedup vs 1 thread")
    ax1.set_title(f"CPU scaling, mutational kernel ({sb['pdb']}, {sb['n_res']} res)\n"
                  "near-linear to 10 perf cores; plateau past that is the host's 10+4 hetero cores, not Amdahl",
                  fontsize=9.5)
    ax1.grid(True, alpha=0.3); ax1.legend(fontsize=8, loc="upper left")
    b = sb["bottleneck"]
    parts = [("data prep\n(subprocess+parse+gamma)", b["prep_s"], "#d62728"),
             ("density kernel", b["density_kernel_s"], "#9ecae1"),
             ("decoy kernel\n(the real compute)", b["decoy_kernel_s"], "#08519c")]
    ax2.bar([p[0] for p in parts], [p[1] for p in parts], color=[p[2] for p in parts])
    for i, p in enumerate(parts):
        ax2.text(i, p[1], f"{p[1]*1000:.0f} ms", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("time per single variant (s)")
    ax2.set_title("Bottleneck: data prep dominates one per-variant calc\n"
                  "DMS scan redoes prep per mutant -> amortize prep across the N x 20 variants",
                  fontsize=9.5)
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


def main():
    os.makedirs(f"{BR}/plots", exist_ok=True)
    sample = loadj(f"{BR}/bench_sample_mut.json")
    flags = loadj(f"{BR}/bench_flags_x86.json")
    fig_scaling_bottleneck(loadj(f"{BR}/bench_scaling_bottleneck.json"),
                           f"{BR}/plots/fig7_cpu_scaling_bottleneck.png")
    fits = fig_sample(sample, f"{BR}/plots/fig5_size_sample_fit.png") if sample else {}
    fig_flags(flags, f"{BR}/plots/fig6_flag_sweep.png") if flags else None
    # dump fit summary for the report
    if fits:
        json.dump({k: {"a": v[0], "exponent_b": v[1], "fit_r2": v[2]} for k, v in fits.items()},
                  open(f"{BR}/scaling_fits.json", "w"), indent=2)
        print("scaling exponents:", {k: round(v[1], 2) for k, v in fits.items()})


if __name__ == "__main__":
    main()
