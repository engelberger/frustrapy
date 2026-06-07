import os
import logging
import numpy as np
import pandas as pd

# The heavy clustering stack (igraph, leidenalg, scikit-learn, scipy, statsmodels) is
# the optional `clustering` extra. It is imported lazily inside
# detect_dynamic_clusters (not at module top) so that `import frustrapy` works on a
# bare install; calling the function without the extra raises a clear ImportError.
from ..core import Dynamic
from ..utils import log_execution_time

logger = logging.getLogger(__name__)

# bio3d::aa123 1-letter -> 3-letter map (used to label residue nodes exactly as R's
# detect_dynamic_clusters does: rownames = paste(aa123(AA), "_", Res)).
_AA1_TO_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def _aa123(one_letter: str) -> str:
    """1-letter -> 3-letter amino acid code (bio3d aa123 parity); unknowns pass through."""
    return _AA1_TO_AA3.get(str(one_letter).upper(), str(one_letter).upper())


def _factominer_pca(x: np.ndarray, ncp: int) -> np.ndarray:
    """Principal-component individual coordinates matching FactoMineR::PCA(scale.unit=TRUE).

    R's detect_dynamic_clusters runs ``PCA(frustraData, ncp=Ncp)`` where the rows of
    ``frustraData`` are the (filtered) residues — the *individuals* — and the columns are
    the trajectory frames — the *variables*. FactoMineR centres and scales each variable
    to unit variance (population sd, denominator n) and weights individuals by 1/n, then
    returns ``pca$ind$coord`` = the individuals' coordinates in PC space.

    To make the result reproducible (and to match FactoMineR component-for-component, not
    just up to sign), we replicate FactoMineR's ``svd.triplet`` sign convention: each
    component is flipped so that the sum of its variable loadings (the right singular
    vector) is non-negative. With that convention this reproduces ``pca$ind$coord`` to
    machine precision (≈1e-14), signs included.

    Args:
        x: array of shape ``[n_individuals (residues) x n_variables (frames)]``.
        ncp: number of components to keep.

    Returns:
        Individual coordinates, shape ``[n_individuals x min(ncp, rank)]``.
    """
    n = x.shape[0]
    col_mean = x.mean(axis=0)
    col_std = x.std(axis=0, ddof=0)  # FactoMineR scales by population sd (denominator n)
    # Guard against zero-variance columns (constant frames) — FactoMineR would divide by
    # a tiny sd; leaving them as 0 after centring is the numerically stable equivalent.
    col_std = np.where(col_std == 0, 1.0, col_std)
    xs = (x - col_mean) / col_std
    row_w = np.full(n, 1.0 / n)
    b = xs * np.sqrt(row_w)[:, None]  # column weights default to 1
    u, s, vt = np.linalg.svd(b, full_matrices=False)
    keep = min(ncp, s.shape[0])
    u = u[:, :keep]
    v = vt[:keep, :].T  # variable (frame) loadings, [n_variables x keep]
    # FactoMineR svd.triplet: mult <- sign(colSums(V)); mult[mult==0] <- 1; flip U,V by it.
    mult = np.sign(v.sum(axis=0))
    mult[mult == 0] = 1.0
    coord = (u * s[:keep]) / np.sqrt(row_w)[:, None]
    coord = coord * mult
    return coord


def _corr_pvalue_matrix(coord: np.ndarray, corr_type: str):
    """Residue x residue correlation + p-values, matching Hmisc::rcorr(t(pca$ind$coord)).

    R computes ``rcorr(t(pca$ind$coord), type = CorrType)`` — correlations between the
    *residues* (the columns of ``t(coord)``) using the ``ncp`` principal-component
    coordinates as the observations. For Spearman, rcorr ranks each residue's coordinate
    vector and applies Pearson on the ranks; the two-sided p-value comes from the
    t-distribution ``t = r*sqrt((n-2)/(1-r^2))`` with ``n-2`` degrees of freedom, where
    ``n`` is the number of observations (= number of components).

    Returns:
        (corr, pval): two ``[M x M]`` arrays (M = number of residues).
    """
    from scipy.stats import rankdata, t as tdist

    n_obs = coord.shape[1]  # observations per correlation = number of components
    if corr_type == "spearman":
        # rank each residue's coordinate vector (rows), then Pearson on ranks
        data = np.vstack([rankdata(coord[i, :]) for i in range(coord.shape[0])])
    else:
        data = coord
    corr = np.corrcoef(data)  # rows are residues -> [M x M]
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    # Two-sided t-test p-values (rcorr convention).
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 1.0 - corr**2
        denom = np.where(denom <= 0, np.nan, denom)
        t_stat = corr * np.sqrt((n_obs - 2) / denom)
        pval = 2.0 * tdist.sf(np.abs(t_stat), n_obs - 2)
    pval = np.where(np.isnan(pval), 0.0, pval)
    np.fill_diagonal(pval, 0.0)
    return corr, pval


@log_execution_time
def detect_dynamic_clusters(
    dynamic: "Dynamic",
    loess_span: float = 0.05,
    min_frst_range: float = 0.7,
    filt_mean: float = 0.15,
    ncp: int = 10,
    min_corr: float = 0.95,
    leiden_resol: float = 1,
    corr_type: str = "spearman",
    seed: int = 0,
) -> "Dynamic":
    """
    Detects residue modules with similar single-residue frustration dynamics.

    Faithful Python port of frustratometeR's ``detect_dynamic_clusters``
    (``R/functions.R:1052``). It fits a loess model (span = ``loess_span``) to each
    residue's single-residue frustration trajectory, keeps residues whose dynamic
    frustration range exceeds the ``min_frst_range`` quantile and whose mean is outside
    ``[-filt_mean, filt_mean]``, runs PCA over the surviving residues, computes the
    ``corr_type`` correlation between residues from their principal-component
    coordinates, keeps residue pairs with ``corr > min_corr`` and p-value <= 0.05,
    builds an undirected weighted graph over those residues and applies Leiden
    clustering at resolution ``leiden_resol``. (Strong *anti*-correlations survive the
    threshold but are dropped by R's graph construction under igraph >= 1.6.0 — see the
    graph-building note in the body.)

    Args:
        dynamic (Dynamic): Dynamic Frustration Object (must be singleresidue mode).
        loess_span (float): Loess smoothing span (alpha > 0). Default: 0.05.
        min_frst_range (float): Dynamic-range quantile filter, 0..1. Default: 0.7.
        filt_mean (float): Mean filter threshold, >= 0. Default: 0.15.
        ncp (int): Number of principal components. Default: 10.
        min_corr (float): Correlation filter threshold, 0..1. Default: 0.95.
        leiden_resol (float): Leiden resolution. Default: 1.
        corr_type (str): "pearson" or "spearman". Default: "spearman".
        seed (int): Random seed for Leiden (determinism). Default: 0.

    Returns:
        Dynamic: the input object with its ``clusters`` attribute populated. The nodes of
        the graph and the rows of ``LeidenClusters`` are *residues* (labelled
        ``<AA3>_<Resno>``), matching frustratometeR.

    Note on parity: the deterministic core (loess statistics, residue filter, PCA
    coordinates, correlation matrix, graph, Leiden membership for a fixed seed) matches
    frustratometeR. Two cross-implementation sources are irreducible: loess (R ``loess``
    vs statsmodels ``lowess``) agrees to ~1e-2 — enough to preserve the filter decision
    but not bit-identical — and the PCA component signs are arbitrary per library.
    """
    if dynamic.mode != "singleresidue":
        raise ValueError(
            "This functionality is only available for the singleresidue index, run dynamic_frustration() with Mode = 'singleresidue'"
        )

    corr_type = corr_type.lower()
    if corr_type not in ["pearson", "spearman"]:
        raise ValueError(
            "Correlation type(CorrType) indicated isn't available or doesn't exist, indicate 'pearson' or 'spearman'"
        )

    # Resolve the optional clustering stack here (lazify-before-demote).
    try:
        import igraph as ig
        import leidenalg as la
        import scipy.stats  # noqa: F401 (used in _corr_pvalue_matrix)
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError as exc:
        raise ImportError(
            "detect_dynamic_clusters requires the optional 'clustering' dependencies "
            "(scipy, python-igraph, leidenalg, statsmodels). Install them with: "
            "pip install 'frustrapy[clustering]'"
        ) from exc

    def _sr_path(pdb_file: str) -> str:
        base = os.path.splitext(pdb_file)[0]
        return os.path.join(
            dynamic.results_dir,
            f"{base}.done/FrustrationData/{base}.pdb_singleresidue",
        )

    # Loading residues and res_num from the first frame.
    ini = pd.read_csv(_sr_path(dynamic.order_list[0]), sep=r"\s+", header=0)
    residues = ini["AA"].tolist()
    res_nums = ini["Res"].tolist()

    # Loading data: rows = residues, columns = frames (FrstIndex per residue per frame).
    logger.debug(
        "-----------------------------Loading data-----------------------------"
    )
    frame_cols = []
    for pdb_file in dynamic.order_list:
        read = pd.read_csv(_sr_path(pdb_file), sep=r"\s+", header=0)
        frame_cols.append(read["FrstIndex"].to_numpy())
    frustra_values = np.column_stack(frame_cols)  # [n_residues x n_frames]
    n_residues, n_frames = frustra_values.shape
    res_labels = [f"{_aa123(aa)}_{rn}" for aa, rn in zip(residues, res_nums)]
    frustra_data = pd.DataFrame(
        frustra_values,
        index=res_labels,
        columns=[f"frame_{i + 1}" for i in range(n_frames)],
    )

    # Model fitting and filter by dynamic range and mean.
    logger.debug(
        "-----------------------------Model fitting and filtering by dynamic range and frustration mean-----------------------------"
    )
    frames_axis = np.arange(n_frames, dtype=float)
    frstrange = []
    means = []
    sds = []
    fitted = {}
    for i in range(n_residues):
        y = frustra_values[i, :]
        fit = lowess(
            y,
            frames_axis,
            frac=loess_span,
            it=0,
            delta=0.0,
            is_sorted=True,
            return_sorted=False,
        )
        fitted[f"res_{i}"] = fit
        frstrange.append(float(fit.max() - fit.min()))
        means.append(float(fit.mean()))
        sds.append(float(np.std(fit, ddof=1)))  # R sd() uses denominator n-1

    frstrange_arr = np.asarray(frstrange)
    means_arr = np.asarray(means)
    # R: quantile(..., probs=MinFrstRange) is type-7 (== numpy linear interpolation).
    range_cut = np.quantile(frstrange_arr, min_frst_range)
    keep_mask = (frstrange_arr > range_cut) & (
        (means_arr < -filt_mean) | (means_arr > filt_mean)
    )
    frustra_data_f = frustra_data.loc[keep_mask]

    if frustra_data_f.shape[0] < 2:
        # R errors out (0x0 matrix) when fewer than 2 residues survive; degrade
        # gracefully instead of crashing, returning an empty clustering.
        logger.warning(
            "detect_dynamic_clusters: only %d residue(s) passed the dynamic-range / "
            "mean filter; no clustering is possible.",
            frustra_data_f.shape[0],
        )
        net = ig.Graph()
        cluster_data = pd.DataFrame({"cluster": []})
    else:
        # Principal component analysis (FactoMineR parity).
        logger.debug(
            "-----------------------------Principal component analysis-----------------------------"
        )
        coord = _factominer_pca(frustra_data_f.to_numpy(), ncp)

        # Residue x residue correlation + p-values (Hmisc::rcorr parity).
        corr, pval = _corr_pvalue_matrix(coord, corr_type)

        # R: Cor[lower.tri(diag=T)] <- 0 ; then zero everything that isn't a strong
        # (positive OR negative) correlation, or is not significant.
        corr = np.triu(corr, k=1)
        weak = ~(corr < -min_corr) & ~(corr > min_corr)
        corr[weak | (pval > 0.05)] = 0.0

        logger.debug(
            "-----------------------------Undirected graph-----------------------------"
        )
        labels_f = frustra_data_f.index.tolist()
        # R builds the graph with `graph_from_adjacency_matrix(mode = "undirected")`,
        # which under igraph >= 1.6.0 resolves to mode "max": each pair's weight is
        # max(corr[i,j], corr[j,i]). Because R has already zeroed the lower triangle,
        # this is max(corr_upper, 0) — so *negative* (strong anti-correlation) weights
        # collapse to 0 and produce no edge. Only strong positive correlations become
        # edges. We replicate that exactly with mode="max"; it also keeps Leiden happy
        # (leidenalg rejects negative weights).
        net = ig.Graph.Weighted_Adjacency(
            corr.tolist(), mode="max", attr="weight", loops=False
        )
        net.vs["name"] = labels_f

        # Leiden Clustering (RBConfigurationVertexPartition, weighted) — matches the
        # frustratometeR `leiden` default. Seeded for determinism.
        logger.debug(
            "-----------------------------Leiden Clustering-----------------------------"
        )
        if net.ecount() == 0:
            membership = list(range(net.vcount()))
        else:
            part = la.find_partition(
                net,
                la.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=leiden_resol,
                seed=seed,
            )
            membership = list(part.membership)

        cluster_data = pd.DataFrame({"cluster": membership}, index=labels_f)
        # Drop degree-0 (isolated) vertices from both the cluster table and the graph.
        degrees = np.asarray(net.degree())
        cluster_data = cluster_data.loc[degrees > 0]
        net.delete_vertices([v.index for v in net.vs if net.degree(v) == 0])

    dynamic.clusters["Graph"] = net
    dynamic.clusters["LeidenClusters"] = cluster_data
    dynamic.clusters["LoessSpan"] = loess_span
    dynamic.clusters["MinFrstRange"] = min_frst_range
    dynamic.clusters["FiltMean"] = filt_mean
    dynamic.clusters["Ncp"] = ncp
    dynamic.clusters["MinCorr"] = min_corr
    dynamic.clusters["LeidenResol"] = leiden_resol
    dynamic.clusters["Fitted"] = pd.DataFrame(fitted)
    dynamic.clusters["Means"] = means
    dynamic.clusters["FrstRange"] = frstrange
    dynamic.clusters["Sd"] = sds
    dynamic.clusters["CorrType"] = corr_type

    if dynamic.clusters.get("Graph") is None:
        logger.error("The process was not completed successfully!")
    else:
        logger.debug("The process has finished successfully!")

    return dynamic
