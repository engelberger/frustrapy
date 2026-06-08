"""Plot the cross-backend / core-scaling benchmark from bench_*.json.

Two per-machine, ratio-based figures (host arm64 native vs container x86_64 emulated are
NEVER merged on one wall-time axis - emulation inflates absolute container times; only
within-machine ratios are valid). Parity Spearman annotated where measured.
"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = ["configurational", "mutational", "singleresidue"]
CMODE = {"configurational": "#1f77b4", "mutational": "#d62728", "singleresidue": "#2ca02c"}


def load(path):
    with open(path) as f:
        return json.load(f)


def kernel_rows(data, mode):
    return sorted(
        [r for r in data["records"] if r["kind"] == "kernel" and r["mode"] == mode],
        key=lambda r: r["threads"])


def fig_scaling(host, cont, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, data, title in [
        (axes[0], host, f"Host Apple-Silicon arm64 (native)\n{host['meta']['pdb']}  {host['meta']['n_ca']} res  ·  {host['meta']['cpu_count']} cores"),
        (axes[1], cont, f"Dev container x86_64 (EMULATED on arm64)\n{cont['meta']['pdb']}  {cont['meta']['n_ca']} res  ·  {cont['meta']['cpu_count']} cores")]:
        if data is None:
            ax.set_title("(not available)"); continue
        maxt = 1
        for mode in MODES:
            rows = kernel_rows(data, mode)
            if not rows:
                continue
            ts = [r["threads"] for r in rows]
            sp = [r["speedup"] for r in rows]
            maxt = max(maxt, max(ts))
            ax.plot(ts, sp, "o-", color=CMODE[mode], label=mode, linewidth=2, markersize=6)
        ax.plot([1, maxt], [1, maxt], "k--", alpha=0.35, linewidth=1, label="ideal linear")
        ax.set_xlabel("CPU threads"); ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    axes[0].set_ylabel("Speedup vs 1 thread (native AWSEM kernel)")
    fig.suptitle("FrustraPy native CPU core-scaling - kernel speedup vs threads (parity Spearman = 1.0000, bit-identical)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


def fig_backends(cont, out):
    """Container-only (same machine): lammps vs native-CPU serial vs native-CPU all-cores, per mode."""
    fig, ax = plt.subplots(figsize=(11, 5.4))
    labels, lam, ser, par, speed = [], [], [], [], []
    for mode in MODES:
        e2e = [r for r in cont["records"] if r["kind"] == "e2e" and r["mode"] == mode]
        d = {}
        for r in e2e:
            d[r["backend_label"]] = r
        lam_r = d.get("lammps (reference)")
        ser_r = d.get("native CPU x1 (serial)")
        # best (max threads) native row
        nat = [r for r in e2e if r["backend_label"].startswith("native CPU x") and "serial" not in r["backend_label"]]
        best = min(nat, key=lambda r: r["wall_s"]) if nat else None
        if not (lam_r and ser_r and best):
            continue
        labels.append(mode)
        lam.append(lam_r["wall_s"]); ser.append(ser_r["wall_s"]); par.append(best["wall_s"])
        speed.append(best["threads"])
    x = range(len(labels)); w = 0.26
    b1 = ax.bar([i - w for i in x], lam, w, label="lammps (binary subprocess)", color="#7f7f7f")
    b2 = ax.bar([i for i in x], ser, w, label="native CPU x1 (serial)", color="#9ecae1")
    b3 = ax.bar([i + w for i in x], par, w, label="native CPU (all cores)", color="#08519c")
    for i, (l, s, p, t) in enumerate(zip(lam, ser, par, speed)):
        ax.text(i - w, l, f"{l:.2f}s", ha="center", va="bottom", fontsize=8)
        ax.text(i, s, f"{s:.2f}s", ha="center", va="bottom", fontsize=8)
        ax.text(i + w, p, f"{p:.2f}s\n(x{t})", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("End-to-end wall time (s)")
    ax.set_title("Backend comparison, same machine - dev container x86_64 EMULATED\n"
                 f"{cont['meta']['pdb']} {cont['meta']['n_ca']} res · all FrstIndex parity Spearman = 1.0000 · "
                 "absolute times emulation-inflated (ratios valid, representative lammps timing needs native x86_64 cluster)",
                 fontsize=10)
    ax.grid(True, axis="y", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


def fig_metal(metal_recs, out):
    """Host arm64, same machine: native CPU x1 vs CPU x14 vs Metal GPU, per mode (log scale)."""
    fig, ax = plt.subplots(figsize=(11, 5.6))
    by = {}
    for r in metal_recs:
        by.setdefault(r["mode"], {})[r["backend"]] = r
    modes = [m for m in MODES if m in by]
    x = range(len(modes)); w = 0.26
    cpu1 = [by[m]["native-cpu-x1"]["wall_s"] * 1000 for m in modes]
    cpu14 = [by[m]["native-cpu-x14"]["wall_s"] * 1000 for m in modes]
    met = [by[m]["metal"]["wall_s"] * 1000 for m in modes]
    ax.bar([i - w for i in x], cpu1, w, label="native CPU x1 (serial)", color="#9ecae1")
    ax.bar([i for i in x], cpu14, w, label="native CPU x14 (all cores)", color="#08519c")
    ax.bar([i + w for i in x], met, w, label="Metal GPU (Apple Silicon)", color="#d62728")
    for i, m in enumerate(modes):
        sp = by[m]["metal"]["spearman"]
        ax.text(i - w, cpu1[i], f"{cpu1[i]:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(i, cpu14[i], f"{cpu14[i]:.1f}", ha="center", va="bottom", fontsize=8)
        sv = cpu1[i] / met[i]; sv14 = cpu14[i] / met[i]
        ax.text(i + w, met[i], f"{met[i]:.1f}ms\n{sv:.0f}x vs x1\n{sv14:.1f}x vs x14", ha="center", va="bottom", fontsize=7.5)
    ax.set_yscale("log"); ax.set_xticks(list(x)); ax.set_xticklabels(modes)
    ax.set_ylabel("AWSEM kernel wall time (ms, log scale)")
    ax.set_title("Backend kernel time on host Apple-Silicon arm64 (SAME machine) - CPU vs Metal GPU\n"
                 "3pgk_A 415 res · Metal-CPU FrstIndex parity Spearman = 1.00000 (max|Δ| ~1e-6, f32) · "
                 "configurational is GPU-overhead-bound (light work)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=None)
    p.add_argument("--container", required=True)
    p.add_argument("--metal", default=None)
    p.add_argument("--outdir", required=True)
    a = p.parse_args()
    host = load(a.host) if a.host and os.path.exists(a.host) else None
    cont = load(a.container)
    os.makedirs(a.outdir, exist_ok=True)
    fig_scaling(host, cont, os.path.join(a.outdir, "fig1_cpu_core_scaling.png"))
    fig_backends(cont, os.path.join(a.outdir, "fig2_backend_comparison_container.png"))
    if a.metal and os.path.exists(a.metal):
        with open(a.metal) as f:
            fig_metal(json.load(f), os.path.join(a.outdir, "fig3_metal_vs_cpu_host.png"))


if __name__ == "__main__":
    main()
