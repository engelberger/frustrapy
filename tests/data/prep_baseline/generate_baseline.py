"""Generate the golden prep-amortization regression baseline.

This freezes the current (pre-optimization) FrustraPy outputs so the prep
amortization work (missions 2 and 3) can be diffed against a fixed oracle and
prove zero numeric regression. It is committed alongside the fixtures it writes
so the baseline is reproducible.

What it captures, under ``tests/data/prep_baseline/``:

  * Full FrustrationData tables for a single-chain structure (1crn) and a genuine
    multi-chain structure (1zni, insulin, 4 chains with cross-chain contacts), for
    all three modes. For the contact modes it also captures the 5 Angstrom density
    table. Tables are written by the default ``lammps`` backend (the parity
    reference); the native backend is checked against them, not stored separately.
  * Per-variant FrstIndex arrays from a small saturation-mutagenesis scan (a few
    positions x 20 amino acids), single-chain (1crn) and multi-chain (1zni, with
    target sites in two different chains to pin chain-correct site selection).
  * ``manifest.json`` recording, per fixture, the structure, mode, backend,
    seq_dist, row count, and sha256, plus the native-vs-lammps parity summary.

Run from an activated venv (PdbCoords2Lammps.sh spawns a bare ``python3`` that
needs Bio on PATH), with the native core built (``pip install ./native``) so the
parity summary is populated:

    python tests/data/prep_baseline/generate_baseline.py

The regression test ``tests/test_prep_amortization_parity.py`` reads these
fixtures and re-runs the current code against them.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(HERE)
# Ensure this worktree's frustrapy is imported, not an editable install pointing
# elsewhere (the package is resolved by cwd when pytest runs from the repo root;
# this script can run from any directory).
_REPO_ROOT = os.path.dirname(os.path.dirname(DATA_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
SEQ_DIST = 12
MODES = ("configurational", "mutational", "singleresidue")

# (structure base name, source PDB path). 1crn = single chain; 1zni = multi-chain.
STRUCTURES = (
    ("1crn", os.path.join(DATA_DIR, "1crn.pdb")),
    ("1zni", os.path.join(DATA_DIR, "1zni.pdb")),
)

# Saturation-scan targets: (base, [(res_num, chain), ...]). The 1zni targets sit
# in two different chains to exercise chain-correct site selection. They use
# distinct residue numbers on purpose: with split=True the per-variant mutant PDB
# filename omits the chain (mutations.py:_process_amino_acid), so two targets that
# share a residue number across chains would collide on one file and race. That
# latent multi-chain limitation is documented in docs/PREP_AMORTIZATION_PLAN.md;
# the baseline stays on the working path.
SCAN_TARGETS = {
    "1crn": [(10, "A"), (25, "A")],
    "1zni": [(5, "A"), (25, "B")],
}

# FrstIndex column (0-based) in the parsed FrustrationData table, per mode.
FRST_COL = {"configurational": 11, "mutational": 11, "singleresidue": 7}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(pdb_src: str, mode: str, backend: str, results_dir: str) -> str:
    """Run one calculate_frustration into a fresh scratch dir; return the table path.

    The input PDB is copied into the scratch dir first because some prep paths
    rewrite the structure in place.
    """
    import frustrapy

    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)
    local = os.path.join(results_dir, os.path.basename(pdb_src))
    shutil.copy2(pdb_src, local)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frustrapy.calculate_frustration(
            pdb_file=local, mode=mode, results_dir=results_dir, graphics=False,
            visualization=False, debug="ERROR", backend=backend, seq_dist=SEQ_DIST,
        )
    base = os.path.splitext(os.path.basename(pdb_src))[0]
    return os.path.join(results_dir, f"{base}.done", "FrustrationData", f"{base}.pdb_{mode}")


def _scan(pdb_src: str, targets, results_dir: str):
    """Run a saturation-mutagenesis scan; return {(res,chain): table_text}."""
    import frustrapy
    from frustrapy.analysis.mutations import mutate_res_scan_parallel

    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)
    local = os.path.join(results_dir, os.path.basename(pdb_src))
    shutil.copy2(pdb_src, local)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        residues = {}
        for res, chain in targets:
            residues.setdefault(chain, []).append(res)
        pdb, _, _, _ = frustrapy.calculate_frustration(
            pdb_file=local, mode="singleresidue", residues=residues,
            results_dir=results_dir, graphics=False, visualization=False,
            debug="ERROR", seq_dist=SEQ_DIST,
        )
        mutate_res_scan_parallel(
            pdb, targets=list(targets), split=True, method="threading", n_cpus=None
        )
    md = os.path.join(pdb.job_dir, "MutationsData")
    out = {}
    for res, chain in targets:
        f = os.path.join(md, f"singleresidue_Res{res}_threading_{chain}.txt")
        with open(f) as fh:
            out[(res, chain)] = fh.read()
    return out


def _spearman(a, b):
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((x - mb) ** 2 for x in rb) ** 0.5
    return cov / (va * vb) if va and vb else 1.0


def main():
    import tempfile

    manifest = {
        "description": "Golden prep-amortization regression baseline.",
        "seq_dist": SEQ_DIST,
        "backend": "lammps",
        "tables": {},
        "scans": {},
        "native_parity": {},
    }
    scratch = tempfile.mkdtemp(prefix="prep_baseline_")
    try:
        # 1) Full FrustrationData tables (lammps), all modes, both structures.
        for base, src in STRUCTURES:
            for mode in MODES:
                table = _run(src, mode, "lammps", os.path.join(scratch, f"{base}_{mode}"))
                dst = os.path.join(HERE, f"{base}.pdb_{mode}")
                shutil.copy2(table, dst)
                with open(dst) as fh:
                    n_rows = sum(1 for _ in fh) - 1
                manifest["tables"][f"{base}.pdb_{mode}"] = {
                    "structure": base, "mode": mode, "n_rows": n_rows,
                    "sha256": _sha256(dst),
                }
                # 5A density table for the contact modes.
                if mode in ("configurational", "mutational"):
                    adens = table + "_5adens"
                    if os.path.exists(adens):
                        dst_a = os.path.join(HERE, f"{base}.pdb_{mode}_5adens")
                        shutil.copy2(adens, dst_a)
                        with open(dst_a) as fh:
                            n_a = sum(1 for _ in fh) - 1
                        manifest["tables"][f"{base}.pdb_{mode}_5adens"] = {
                            "structure": base, "mode": f"{mode}_5adens",
                            "n_rows": n_a, "sha256": _sha256(dst_a),
                        }
                print(f"captured {base} {mode}: {n_rows} rows")

        # 2) Saturation scans (per-variant FrstIndex), single- and multi-chain.
        for base, targets in SCAN_TARGETS.items():
            src = dict(STRUCTURES)[base]
            tables = _scan(src, targets, os.path.join(scratch, f"{base}_scan"))
            for (res, chain), text in tables.items():
                name = f"scan_{base}_Res{res}_{chain}.txt"
                dst = os.path.join(HERE, name)
                with open(dst, "w") as fh:
                    fh.write(text)
                n_rows = len(text.splitlines()) - 1
                manifest["scans"][name] = {
                    "structure": base, "res_num": res, "chain": chain,
                    "n_variants": n_rows, "sha256": _sha256(dst),
                }
                print(f"captured scan {base} Res{res}/{chain}: {n_rows} variants")

        # 3) Native-vs-lammps parity summary (informational; the test enforces it).
        try:
            import frustrapy_native  # noqa: F401

            for base, src in STRUCTURES:
                for mode in MODES:
                    lam = os.path.join(HERE, f"{base}.pdb_{mode}")
                    nat = _run(src, mode, "native", os.path.join(scratch, f"{base}_{mode}_n"))
                    L = [ln.split() for ln in open(lam).read().splitlines()[1:]]
                    N = [ln.split() for ln in open(nat).read().splitlines()[1:]]
                    fc = FRST_COL[mode]
                    max_d = max(
                        (abs(float(a[fc]) - float(b[fc])) for a, b in zip(L, N)),
                        default=0.0,
                    )
                    sp = _spearman([float(a[fc]) for a in L], [float(b[fc]) for b in N])
                    manifest["native_parity"][f"{base}.pdb_{mode}"] = {
                        "rows_lammps": len(L), "rows_native": len(N),
                        "max_abs_dFrstIndex": max_d, "spearman": sp,
                    }
                    print(f"native parity {base} {mode}: max|dFI|={max_d:.2e} spearman={sp:.4f}")
        except ImportError:
            manifest["native_parity"]["note"] = "native core not built; skipped"
            print("native core not built; parity summary skipped")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    with open(os.path.join(HERE, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("wrote manifest.json")


if __name__ == "__main__":
    main()
