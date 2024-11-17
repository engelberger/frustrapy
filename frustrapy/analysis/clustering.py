import logging
import numpy as np
import pandas as pd
import igraph as ig
import leidenalg as la
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr

# from statsmodels.nonparametric.smoothers_lowess import lowess
from ..core import Dynamic
from ..utils import log_execution_time

logger = logging.getLogger(__name__)


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
) -> "Dynamic":
    """
    Detects residue modules with similar single-residue frustration dynamics.
    It filters out the residuals with variable dynamics, for this, it adjusts a loess
    model with span = LoessSpan and calculates the dynamic range of frustration and the mean of single-residue frustration.
    It is left with the residuals with a dynamic frustration range greater than the quantile defined by MinFrstRange and with a mean Mean <(-FiltMean) or Mean> FiltMean.
    Performs an analysis of main components and keeps Ncp components, to compute the correlation(CorrType) between them and keep the residues that have greater correlation MinCorr and p-value> 0.05.
    An undirected graph is generated and Leiden clustering is applied with LeidenResol resolution.

    Args:
        dynamic (Dynamic): Dynamic Frustration Object.
        loess_span (float): Parameter α > 0 that controls the degree of smoothing of the loess() function of model fit. Default: 0.05.
        min_frst_range (float): Frustration dynamic range filter threshold. 0 <= MinFrstRange <= 1. Default: 0.7.
        filt_mean (float): Frustration Mean Filter Threshold. FiltMean >= 0. Default: 0.15.
        ncp (int): Number of principal components to be used in PCA(). Ncp >= 1. Default: 10.
        min_corr (float): Correlation filter threshold. 0 <= MinCorr <= 1. Default: 0.95.
        leiden_resol (float): Parameter that defines the coarseness of the cluster. LeidenResol > 0. Default: 1.
        corr_type (str): Type of correlation index to compute. Values: "pearson" or "spearman". Default: "spearman".

    Returns:
        Dynamic: Dynamic Frustration Object and its Clusters attribute.
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

    required_libraries = ["leidenalg", "igraph", "sklearn", "scipy", "numpy", "pandas"]
    missing_libraries = [
        library for library in required_libraries if library not in globals()
    ]
    if missing_libraries:
        raise ImportError(
            f"Please install the following libraries to continue: {', '.join(missing_libraries)}"
        )

    # Loading residues and res_num
    ini = pd.read_csv(
        os.path.join(
            dynamic.results_dir,
            f"{os.path.splitext(dynamic.order_list[0])[0]}.done/FrustrationData/{os.path.splitext(dynamic.order_list[0])[0]}.pdb_singleresidue",
        ),
        sep="\s+",
        header=0,
    )
    residues = ini["AA"].tolist()
    res_nums = ini["Res"].tolist()

    # Loading data
    logger.debug(
        "-----------------------------Loading data-----------------------------"
    )
    frustra_data = pd.DataFrame()
    for pdb_file in dynamic.order_list:
        read = pd.read_csv(
            os.path.join(
                dynamic.results_dir,
                f"{os.path.splitext(pdb_file)[0]}.done/FrustrationData/{os.path.splitext(pdb_file)[0]}.pdb_singleresidue",
            ),
            sep="\s+",
            header=0,
        )
        frustra_data[f"frame_{len(frustra_data.columns)}"] = read["FrstIndex"]

    frustra_data.index = [
        f"{residue}_{res_num}" for residue, res_num in zip(residues, res_nums)
    ]

    # Model fitting and filter by difference and mean
    logger.debug(
        "-----------------------------Model fitting and filtering by dynamic range and frustration mean-----------------------------"
    )
    frstrange = []
    means = []
    sds = []
    fitted = pd.DataFrame()
    for i in range(len(residues)):
        res = pd.DataFrame(
            {"Frustration": frustra_data.iloc[i], "Frames": range(len(frustra_data))}
        )
        modelo = lowess(
            res["Frustration"],
            res["Frames"],
            frac=loess_span,
            it=0,
            delta=0.0,
            is_sorted=False,
        )
        fitted[f"res_{i}"] = modelo[:, 1]
        frstrange.append(modelo[:, 1].max() - modelo[:, 1].min())
        means.append(modelo[:, 1].mean())
        sds.append(modelo[:, 1].std())

    estadistics = pd.DataFrame({"Diferences": frstrange, "Means": means})
    frustra_data = frustra_data[
        (
            estadistics["Diferences"]
            > np.quantile(estadistics["Diferences"], min_frst_range)
        )
        & ((estadistics["Means"] < -filt_mean) | (estadistics["Means"] > filt_mean))
    ]

    # Principal component analysis
    logger.debug(
        "-----------------------------Principal component analysis-----------------------------"
    )
    pca = PCA(n_components=ncp)
    pca_result = pca.fit_transform(frustra_data.T)

    if corr_type == "spearman":
        corr_func = spearmanr
    else:
        corr_func = pearsonr

    corr_matrix = np.zeros((pca_result.shape[1], pca_result.shape[1]))
    p_values = np.zeros((pca_result.shape[1], pca_result.shape[1]))
    for i in range(pca_result.shape[1]):
        for j in range(i, pca_result.shape[1]):
            corr, p_value = corr_func(pca_result[:, i], pca_result[:, j])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
            p_values[i, j] = p_value
            p_values[j, i] = p_value

    np.fill_diagonal(corr_matrix, 0)
    corr_matrix[
        (corr_matrix < min_corr) & (corr_matrix > -min_corr) | (p_values > 0.05)
    ] = 0
    logger.debug(
        "-----------------------------Undirected graph-----------------------------"
    )
    net = ig.Graph.Adjacency((corr_matrix > 0).tolist(), mode="undirected")

    logger.debug(
        "-----------------------------Leiden Clustering-----------------------------"
    )
    leiden_clusters = la.find_partition(
        net, la.RBConfigurationVertexPartition, resolution_parameter=leiden_resol
    )
    cluster_data = pd.DataFrame({"cluster": leiden_clusters.membership})
    cluster_data = cluster_data.loc[net.degree() > 0]

    net.delete_vertices(net.vs.select(_degree=0))

    dynamic.clusters["Graph"] = net
    dynamic.clusters["LeidenClusters"] = cluster_data
    dynamic.clusters["LoessSpan"] = loess_span
    dynamic.clusters["MinFrstRange"] = min_frst_range
    dynamic.clusters["FiltMean"] = filt_mean
    dynamic.clusters["Ncp"] = ncp
    dynamic.clusters["MinCorr"] = min_corr
    dynamic.clusters["LeidenResol"] = leiden_resol
    dynamic.clusters["Fitted"] = fitted
    dynamic.clusters["Means"] = means
    dynamic.clusters["FrstRange"] = frstrange
    dynamic.clusters["Sd"] = sds
    dynamic.clusters["CorrType"] = corr_type

    if "Graph" not in dynamic.clusters or dynamic.clusters["Graph"] is None:
        logger.error("The process was not completed successfully!")
    else:
        logger.debug("The process has finished successfully!")

    return dynamic
