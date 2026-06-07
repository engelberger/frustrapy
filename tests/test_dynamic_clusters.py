"""Regression tests for dynamic-trajectory clustering parity with frustratometeR.

These pin frustrapy's ``detect_dynamic_clusters`` (a Python port of frustratometeR's
``detect_dynamic_clusters``, R/functions.R:1052) against ground-truth values produced by
the R reference on a frozen synthetic single-residue trajectory
(``tests/data/dynamic_clusters/``). They guard the T4 fixes:

  * clustering operates on *residues* (not principal components) — the prior code
    clustered the <=10 PCs and crashed when n_residues != n_frames;
  * the FactoMineR PCA (scaling + sign convention) is reproduced component-for-component;
  * the Hmisc::rcorr residue correlation, the dynamic-range / mean filter, and the
    igraph ``mode="max"`` graph construction match R.

The reference numbers were generated with R 4.3.3 + frustratometeR / FactoMineR / Hmisc /
igraph; see the data-generation notes in the FrustraEvo/MD-parity work. Only the loess
smoother differs across implementations (R ``loess`` vs statsmodels ``lowess``); it agrees
to ~1e-2, which is well within the margin that preserves the residue filter decision.
"""

import os
import numpy as np
import pandas as pd
import pytest

# detect_dynamic_clusters needs the optional `clustering` extra.
pytest.importorskip("statsmodels")
pytest.importorskip("igraph")
pytest.importorskip("leidenalg")
pytest.importorskip("scipy")

from frustrapy.core import Dynamic  # noqa: E402
from frustrapy.analysis.clustering import (  # noqa: E402
    detect_dynamic_clusters,
    _factominer_pca,
    _corr_pvalue_matrix,
    _aa123,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "dynamic_clusters")


def _load_matrix():
    """Frozen FrstIndex matrix: returns (aa list, res list, values[res x frames])."""
    df = pd.read_csv(os.path.join(DATA_DIR, "frstindex_matrix.csv"))
    aa = df["AA"].tolist()
    res = df["Res"].tolist()
    frame_cols = [c for c in df.columns if c.startswith("frame")]
    # keep the verbatim string reprs so the rebuilt tables feed identical floats
    raw = pd.read_csv(
        os.path.join(DATA_DIR, "frstindex_matrix.csv"), dtype=str
    )[frame_cols]
    return aa, res, frame_cols, raw


def _build_trajectory(tmp_path):
    """Write per-frame ``*.pdb_singleresidue`` tables and return a Dynamic object."""
    aa, res, frame_cols, raw = _load_matrix()
    results_dir = os.path.join(str(tmp_path), "results")
    order_list = []
    for f, col in enumerate(frame_cols):
        base = f"frame{f}"
        order_list.append(f"{base}.pdb")
        d = os.path.join(results_dir, f"{base}.done", "FrustrationData")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{base}.pdb_singleresidue"), "w") as fh:
            fh.write(
                "Res ChainRes DensityRes AA NativeEnergy DecoyEnergy SDEnergy FrstIndex\n"
            )
            for i in range(len(aa)):
                fh.write(
                    f"{res[i]} A 5.0 {aa[i]} -1.0 -1.0 0.5 {raw.iloc[i, f]}\n"
                )
    dyn = Dynamic(
        pdbs_dir=str(tmp_path),
        order_list=order_list,
        mode="singleresidue",
        results_dir=results_dir + "/",
    )
    return dyn, aa, res


def test_aa123():
    assert _aa123("A") == "ALA"
    assert _aa123("g") == "GLY"
    assert _aa123("W") == "TRP"
    # unknown codes pass through (no crash)
    assert _aa123("X") == "X"


def test_pca_matches_factominer():
    """_factominer_pca reproduces R's pca$ind$coord component-for-component (sign incl.)."""
    aa, res, frame_cols, raw = _load_matrix()
    values = raw.astype(float).to_numpy()  # [residues x frames]
    stats = pd.read_csv(os.path.join(DATA_DIR, "expected_loess_stats.tsv"), sep="\t")
    keep = stats["keep"].to_numpy()
    coord = _factominer_pca(values[keep], 10)
    expected = pd.read_csv(
        os.path.join(DATA_DIR, "expected_pca_coord.tsv"), sep="\t", index_col=0
    ).to_numpy()
    assert coord.shape == expected.shape
    # No manual sign alignment: the FactoMineR sign convention must make these agree.
    assert np.abs(coord - expected).max() < 1e-8


def test_correlation_matches_rcorr():
    """_corr_pvalue_matrix reproduces Hmisc::rcorr given identical PCA coordinates."""
    expected_coord = pd.read_csv(
        os.path.join(DATA_DIR, "expected_pca_coord.tsv"), sep="\t", index_col=0
    ).to_numpy()
    corr, _ = _corr_pvalue_matrix(expected_coord, "spearman")
    expected_corr = pd.read_csv(
        os.path.join(DATA_DIR, "expected_corr.tsv"), sep="\t", index_col=0
    ).to_numpy()
    assert np.abs(corr - expected_corr).max() < 1e-8


def test_filter_matches_r(tmp_path):
    """The dynamic-range / mean residue filter selects exactly R's residue set."""
    dyn, aa, res = _build_trajectory(tmp_path)
    dyn = detect_dynamic_clusters(dyn, min_corr=0.6)
    fr = np.asarray(dyn.clusters["FrstRange"])
    me = np.asarray(dyn.clusters["Means"])
    range_cut = np.quantile(fr, 0.7)
    keep = (fr > range_cut) & ((me < -0.15) | (me > 0.15))
    labels = [f"{_aa123(a)}_{r}" for a, r in zip(aa, res)]
    py_filtered = sorted(l for l, k in zip(labels, keep) if k)
    stats = pd.read_csv(os.path.join(DATA_DIR, "expected_loess_stats.tsv"), sep="\t")
    r_filtered = sorted(stats.loc[stats["keep"], "name"].tolist())
    assert py_filtered == r_filtered


def test_loess_within_tolerance_preserves_filter(tmp_path):
    """statsmodels lowess differs from R loess only at ~1e-2 and keeps the same residues."""
    dyn, aa, res = _build_trajectory(tmp_path)
    dyn = detect_dynamic_clusters(dyn, min_corr=0.6)
    stats = pd.read_csv(os.path.join(DATA_DIR, "expected_loess_stats.tsv"), sep="\t")
    fr = np.asarray(dyn.clusters["FrstRange"])
    me = np.asarray(dyn.clusters["Means"])
    assert np.abs(fr - stats["FrstRange"].to_numpy()).max() < 5e-2
    assert np.abs(me - stats["Means"].to_numpy()).max() < 5e-2


def test_graph_edges_match_r(tmp_path):
    """End-to-end graph (residue nodes, weights) is byte-for-byte R's at min_corr=0.6."""
    dyn, aa, res = _build_trajectory(tmp_path)
    dyn = detect_dynamic_clusters(dyn, min_corr=0.6, seed=0)
    net = dyn.clusters["Graph"]
    py_edges = sorted(
        tuple(sorted((net.vs[e.source]["name"], net.vs[e.target]["name"])))
        + (round(e["weight"], 7),)
        for e in net.es
    )
    ref = pd.read_csv(
        os.path.join(DATA_DIR, "expected_graph_edges_mc0.6.tsv"), sep="\t"
    )
    r_edges = sorted(
        tuple(sorted((a, b))) + (round(w, 7),)
        for a, b, w in zip(ref["from"], ref["to"], ref["weight"])
    )
    assert py_edges == r_edges
    # the nodes are residues (the structural fix), labelled <AA3>_<Resno>
    assert all("_" in name for name in net.vs["name"])


def test_clusters_are_residue_indexed(tmp_path):
    """LeidenClusters must be indexed by residues, and connected pairs co-cluster."""
    dyn, aa, res = _build_trajectory(tmp_path)
    dyn = detect_dynamic_clusters(dyn, min_corr=0.6, seed=0)
    lc = dyn.clusters["LeidenClusters"]
    # one row per non-isolated residue (3 connected pairs -> 6 residues)
    assert len(lc) == 6
    assert all("_" in str(idx) for idx in lc.index)
    mapping = lc["cluster"].to_dict()
    for a, b in [("ALA_1", "GLU_4"), ("ILE_8", "LEU_10"), ("PRO_13", "GLN_14")]:
        assert mapping[a] == mapping[b]
    # the three pairs are distinct modules
    assert len({mapping["ALA_1"], mapping["ILE_8"], mapping["PRO_13"]}) == 3


def test_high_min_corr_yields_empty_graph_without_crashing(tmp_path):
    """Default min_corr=0.95 leaves no edges here; must degrade gracefully (R crashes)."""
    dyn, aa, res = _build_trajectory(tmp_path)
    dyn = detect_dynamic_clusters(dyn)  # default min_corr=0.95
    assert dyn.clusters["Graph"].ecount() == 0
    assert len(dyn.clusters["LeidenClusters"]) == 0


def test_requires_singleresidue_mode(tmp_path):
    dyn, aa, res = _build_trajectory(tmp_path)
    dyn.mode = "configurational"
    with pytest.raises(ValueError, match="singleresidue"):
        detect_dynamic_clusters(dyn)


def test_rejects_unknown_corr_type(tmp_path):
    dyn, aa, res = _build_trajectory(tmp_path)
    with pytest.raises(ValueError, match="Correlation type"):
        detect_dynamic_clusters(dyn, corr_type="kendall")
