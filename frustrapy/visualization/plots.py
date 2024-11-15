import os
import logging
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ..core import Pdb

logger = logging.getLogger(__name__)


def plot_contact_map(pdb, chain=None, save=False, show=False):
    """
    Generates an interactive contact map plot to visualize the frustration values assigned to each contact.

    Args:
        pdb (Pdb): Pdb Frustration object.
        chain (str, optional): Chain of residue. Default: None.
        save (bool, optional): If True, saves the graph; otherwise, it does not. Default: False.

    Returns:
        plotly.graph_objects.Figure: The generated interactive plot.
    """
    if save not in [True, False]:
        raise ValueError("Save must be a boolean value!")

    if chain is not None:
        if len(chain) > 1:
            raise ValueError("You must enter only one Chain!")
        if chain not in pdb.atom["chain"].unique():
            raise ValueError(
                f"The Chain {chain} doesn't exist! The Chains are: {', '.join(pdb.atom['chain'].unique())}"
            )

    if not os.path.exists(os.path.join(pdb.job_dir, "Images")):
        os.makedirs(os.path.join(pdb.job_dir, "Images"))

    adens_table = pd.read_csv(
        os.path.join(
            pdb.job_dir, "FrustrationData", f"{pdb.pdb_base}.pdb_{pdb.mode}_5adens"
        ),
        sep=" ",
        header=0,
    )

    if chain is not None:
        adens_table = adens_table[adens_table["Chains"] == chain]

    datos = pd.read_csv(
        os.path.join(pdb.job_dir, "FrustrationData", f"{pdb.pdb_base}.pdb_{pdb.mode}"),
        sep=" ",
        engine="python",
    )

    if chain is not None:
        datos = datos[datos["ChainRes1"] == chain]

    datos["ChainRes1"] = datos["ChainRes1"].apply(
        lambda x: str(x) if pd.notnull(x) else "Unknown"
    )
    datos["ChainRes2"] = datos["ChainRes2"].apply(
        lambda x: str(x) if pd.notnull(x) else "Unknown"
    )

    datos["Res1"] = pd.to_numeric(datos["Res1"], errors="coerce")
    datos["Res2"] = pd.to_numeric(datos["Res2"], errors="coerce")

    chains_two = sorted(set(datos["ChainRes1"].tolist() + datos["ChainRes2"].tolist()))

    positions = []
    aux_pos_vec = []
    for c in chains_two:
        if c != "Unknown":
            datos["Res1"] = pd.to_numeric(datos["Res1"], errors="coerce")
            datos["Res2"] = pd.to_numeric(datos["Res2"], errors="coerce")

            datos.dropna(subset=["Res1", "Res2"], inplace=True)

            res1_range = datos[datos["ChainRes1"] == c]["Res1"]
            res2_range = datos[datos["ChainRes2"] == c]["Res2"]

            min_pos = min(res1_range.min(), res2_range.min())
            max_pos = max(res1_range.max(), res2_range.max())
            positions.append([min_pos, max_pos, max_pos - min_pos + 1])
            aux_pos_vec.extend(range(int(min_pos), int(max_pos) + 1))

    datos["pos1"] = 0
    datos["pos2"] = 0
    bias = 0
    for i, c in enumerate(chains_two):
        if i > 0:
            bias = sum(pos[2] for pos in positions[:i])
        idx1 = datos["ChainRes1"] == c
        datos.loc[idx1, "pos1"] = datos.loc[idx1, "Res1"] - positions[i][0] + bias + 1
        idx2 = datos["ChainRes2"] == c
        datos.loc[idx2, "pos2"] = datos.loc[idx2, "Res2"] - positions[i][0] + bias + 1

    pos_new = []
    for c in chains_two:
        pos1_range = datos[datos["ChainRes1"] == c]["pos1"]
        pos2_range = datos[datos["ChainRes2"] == c]["pos2"]
        min_pos = min(pos1_range.min(), pos2_range.min())
        max_pos = max(pos1_range.max(), pos2_range.max())
        pos_new.append([min_pos, max_pos, max_pos - min_pos + 1])

    total_positions = sum(pos[2] for pos in pos_new)

    matrz = pd.DataFrame(
        index=range(1, total_positions + 1), columns=range(1, total_positions + 1)
    )

    for _, row in datos.iterrows():
        matrz.loc[row["pos2"], row["pos1"]] = row["FrstIndex"]
        matrz.loc[row["pos1"], row["pos2"]] = row["FrstIndex"]

    matrz = matrz.fillna(0)
    matrz.values[np.tril_indices(matrz.shape[0], k=0)] = 0

    fig = go.Figure(
        data=go.Heatmap(
            z=matrz.values,
            x=list(matrz.columns),
            y=list(matrz.index),
            colorscale=[[0, "red"], [0.5, "white"], [1, "green"]],
            zmin=-4,
            zmax=4,
            colorbar=dict(title="Frustration Index"),
        )
    )

    fig.update_layout(
        title=f"Contact map {pdb.pdb_base}",
        xaxis_title="Residue i",
        yaxis_title="Residue j",
        font=dict(family="Arial", size=12, color="black"),
    )

    if len(chains_two) > 1:
        breaks = [round(x) for x in np.linspace(1, total_positions, 15)]
        labels = [str(aux_pos_vec[b - 1]) for b in breaks]

        for pos in np.cumsum([p[2] for p in pos_new[:-1]]):
            fig.add_shape(
                type="line",
                x0=pos,
                y0=0,
                x1=pos,
                y1=total_positions,
                line=dict(color="gray", width=0.5, dash="dash"),
            )
            fig.add_shape(
                type="line",
                x0=0,
                y0=pos,
                x1=total_positions,
                y1=pos,
                line=dict(color="gray", width=0.5, dash="dash"),
            )

        for i, c in enumerate(chains_two):
            mean_pos = np.mean(pos_new[i][:2])
            fig.add_annotation(
                x=mean_pos,
                y=total_positions + 0.5,
                text=c,
                showarrow=False,
                font=dict(color="gray"),
            )
            fig.add_annotation(
                x=total_positions + 0.5,
                y=mean_pos,
                text=c,
                showarrow=False,
                font=dict(color="gray"),
            )

        fig.update_xaxes(tickvals=breaks, ticktext=labels)
        fig.update_yaxes(tickvals=breaks, ticktext=labels)

    if save:
        # Change the widthe of the plot to 1800x 1800 and save a png
        fig.update_layout(width=1000, height=1000)
        fig.write_image(
            os.path.join(pdb.job_dir, "Images", f"{pdb.pdb_base}_{pdb.mode}_map.png")
        )
        fig.write_html(
            os.path.join(pdb.job_dir, "Images", f"{pdb.pdb_base}_{pdb.mode}_map.html")
        )
        print(
            f"Contact map is stored in {os.path.join(pdb.job_dir, 'Images', f'{pdb.pdb_base}_{pdb.mode}_map.html')}"
        )

    return fig


def plot_5andens(pdb, chain=None, save=False, show=False):
    """
    Generates an interactive plot to analyze the density of contacts around a sphere of 5 Armstrongs,
    centered in the C-alfa atom from the residue. The different classes of contacts
    based on the mutational frustration index are counted in absolute terms.

    Args:
        pdb (Pdb): Pdb Frustration object.
        chain (str, optional): Chain of residue. Default: None.
        save (bool, optional): If True, saves the graph; otherwise, it does not. Default: False.

    Returns:
        plotly.graph_objects.Figure: The generated interactive plot.
    """
    if save not in [True, False]:
        raise ValueError("Save must be a boolean value!")

    if chain is not None:
        if len(chain) > 1:
            raise ValueError("You must enter only one Chain!")
        if chain not in pdb.atom["chain"].unique():
            raise ValueError(
                f"The Chain {chain} doesn't exist! The Chains are: {', '.join(pdb.atom['chain'].unique())}"
            )

    if not os.path.exists(os.path.join(pdb.job_dir, "Images")):
        os.makedirs(os.path.join(pdb.job_dir, "Images"))

    adens_table = pd.read_csv(
        os.path.join(
            pdb.job_dir, "FrustrationData", f"{pdb.pdb_base}.pdb_{pdb.mode}_5adens"
        ),
        sep=" ",
        header=0,
    )

    adens_table["PositionsTotal"] = range(1, len(adens_table) + 1)

    if chain is None:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=adens_table["PositionsTotal"],
                y=adens_table["HighlyFrst"],
                mode="lines",
                name="Highly frustrated",
                line=dict(color="red"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=adens_table["PositionsTotal"],
                y=adens_table["NeutrallyFrst"],
                mode="lines",
                name="Neutral",
                line=dict(color="gray"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=adens_table["PositionsTotal"],
                y=adens_table["MinimallyFrst"],
                mode="lines",
                name="Minimally frustrated",
                line=dict(color="green"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=adens_table["PositionsTotal"],
                y=adens_table["Total"],
                mode="lines",
                name="Total",
                line=dict(color="black"),
            )
        )

        fig.update_layout(
            title=f"Density around 5A sphere (%) in {pdb.pdb_base}",
            xaxis_title="Position",
            yaxis_title="Local frustration density (5A sphere)",
            legend_title="",
            font=dict(family="Arial", size=12, color="black"),
        )

        if save:
            # Change the widthe of the plot to 1800 and save a png
            fig.update_layout(width=1800)
            fig.write_html(
                os.path.join(
                    pdb.job_dir, "Images", f"{pdb.pdb_base}_{pdb.mode}.html_5Adens.html"
                )
            )
            print(
                f"5Adens plot is stored in {os.path.join(pdb.job_dir, 'Images', f'{pdb.pdb_base}_{pdb.mode}.html_5Adens.html')}"
            )
            fig.write_image(
                os.path.join(
                    pdb.job_dir, "Images", f"{pdb.pdb_base}_{pdb.mode}.png_5Adens.png"
                )
            )

    else:
        adens_table = adens_table[adens_table["Chains"] == chain]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=adens_table["Positions"],
                y=adens_table["HighlyFrst"],
                mode="lines",
                name="Highly frustrated",
                line=dict(color="red"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=adens_table["Positions"],
                y=adens_table["NeutrallyFrst"],
                mode="lines",
                name="Neutral",
                line=dict(color="gray"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=adens_table["Positions"],
                y=adens_table["MinimallyFrst"],
                mode="lines",
                name="Minimally frustrated",
                line=dict(color="green"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=adens_table["Positions"],
                y=adens_table["Total"],
                mode="lines",
                name="Total",
                line=dict(color="black"),
            )
        )

        fig.update_layout(
            title=f"Density around 5A sphere (%) in {pdb.pdb_base} chain {chain}",
            xaxis_title="Position",
            yaxis_title="Local frustration density (5A sphere)",
            legend_title="",
            font=dict(family="Arial", size=12, color="black"),
        )

        if save:
            # Change the widthe of the plot to 1800 and save a png
            fig.update_layout(width=1800)
            fig.write_image(
                os.path.join(
                    pdb.job_dir,
                    "Images",
                    f"{pdb.pdb_base}_{pdb.mode}_5Adens__chain{chain}.png",
                )
            )
            fig.write_html(
                os.path.join(
                    pdb.job_dir,
                    "Images",
                    f"{pdb.pdb_base}_{pdb.mode}_5Adens__chain{chain}.html",
                )
            )
            print(
                f"5Adens plot is stored in {os.path.join(pdb.job_dir, 'Images', f'{pdb.pdb_base}_{pdb.mode}_5Adens__chain{chain}.html')}"
            )

    return fig


def plot_5adens_proportions(pdb, chain=None, save=False, show=False):
    """
    Generates an interactive plot to analyze the density of contacts around a sphere of 5 Armstrongs,
    centered in the C-alfa atom from the residue. The different classes of contacts
    based on the mutational frustration index are counted in relative terms.

    Args:
        pdb (Pdb): Pdb Frustration object.
        chain (str, optional): Chain of residue. Default: None.
        save (bool, optional): If True, saves the graph; otherwise, it does not. Default: False.

    Returns:
        plotly.graph_objects.Figure: The generated interactive plot.
    """
    if save not in [True, False]:
        raise ValueError("Save must be a boolean value!")

    if chain is not None:
        if len(chain) > 1:
            raise ValueError("You must enter only one Chain!")
        if chain not in pdb.atom["chain"].unique():
            raise ValueError(
                f"The Chain {chain} doesn't exist! The Chains are: {', '.join(pdb.atom['chain'].unique())}"
            )

    if not os.path.exists(os.path.join(pdb.job_dir, "Images")):
        os.makedirs(os.path.join(pdb.job_dir, "Images"))

    adens_table = pd.read_csv(
        os.path.join(
            pdb.job_dir, "FrustrationData", f"{pdb.pdb_base}.pdb_{pdb.mode}_5adens"
        ),
        sep=" ",
        header=0,
    )

    if chain is not None:
        adens_table = adens_table[adens_table["Chains"] == chain]

    minimally_frst = adens_table["MinimallyFrst"].astype(float)
    neutrally_frst = adens_table["NeutrallyFrst"].astype(float)
    maximally_frst = adens_table["HighlyFrst"].astype(float)

    frustration_data = pd.DataFrame(
        {
            "Positions": adens_table["Res"],
            "Highly frustrated": maximally_frst,
            "Neutral": neutrally_frst,
            "Minimally frustrated": minimally_frst,
        }
    )

    frustration_data = frustration_data.melt(
        id_vars="Positions", var_name="Frustration", value_name="Density"
    )

    fig = go.Figure()

    for frustration in ["Highly frustrated", "Neutral", "Minimally frustrated"]:
        data = frustration_data[frustration_data["Frustration"] == frustration]
        fig.add_trace(
            go.Bar(
                x=data["Positions"],
                y=data["Density"],
                name=frustration,
                marker_color={
                    "Highly frustrated": "red",
                    "Neutral": "gray",
                    "Minimally frustrated": "green",
                }[frustration],
            )
        )

    fig.update_layout(
        barmode="stack",
        xaxis=dict(
            title="Positions",
            tickmode="array",
            tickvals=list(range(0, len(adens_table), (len(adens_table) - 1) // 10)),
            ticktext=[
                str(pos)
                for pos in adens_table["Res"][:: ((len(adens_table) - 1) // 10)]
            ],
        ),
        yaxis=dict(title="Density around 5A sphere (%)"),
        legend=dict(title=""),
        font=dict(family="Arial", size=12, color="black"),
    )

    if chain is None:
        fig.update_layout(title=f"Density around 5A sphere (%) in {pdb.pdb_base}")
    else:
        fig.update_layout(
            title=f"Density around 5A sphere (%) in {pdb.pdb_base} chain {chain}"
        )

    if save:
        if chain is None:
            # Change the widthe of the plot to 1800 and save a png
            fig.update_layout(width=1800)
            fig.write_image(
                os.path.join(
                    pdb.job_dir,
                    "Images",
                    f"{pdb.pdb_base}_{pdb.mode}_5Adens_around.png",
                )
            )
            fig.write_html(
                os.path.join(
                    pdb.job_dir,
                    "Images",
                    f"{pdb.pdb_base}_{pdb.mode}_5Adens_around.html",
                )
            )
            print(
                f"5Adens proportion plot is stored in {os.path.join(pdb.job_dir, 'Images', f'{pdb.pdb_base}_{pdb.mode}_5Adens_around.html')}"
            )
        else:
            # Change the widthe of the plot to 1800 and save a png
            fig.update_layout(width=1800)
            fig.write_image(
                os.path.join(
                    pdb.job_dir,
                    "Images",
                    f"{pdb.pdb_base}_{pdb.mode}_5Adens_around_chain{chain}.png",
                )
            )
            fig.write_html(
                os.path.join(
                    pdb.job_dir,
                    "Images",
                    f"{pdb.pdb_base}_{pdb.mode}_5Adens_around_chain{chain}.html",
                )
            )
            print(
                f"5Adens proportion plot is stored in {os.path.join(pdb.job_dir, 'Images', f'{pdb.pdb_base}_{pdb.mode}_5Adens_around_chain{chain}.html')}"
            )

    return fig


def plot_delta_frus(pdb, res_num, chain, method="threading", save=True, show=False):
    """
    Generate an interactive plot of the single residue frustration difference for mutations.

    Args:
        pdb (Pdb): Pdb frustration object
        res_num (int): Specific residue number
        chain (str): Chain identifier
        method (str): Mutation method ("threading" or "modeller")
        save (bool): Whether to save the plot
        show (bool): Whether to show the plot

    Returns:
        plotly.graph_objects.Figure: Interactive plot of delta frustration
    """
    # Input validation
    if save not in [True, False]:
        raise ValueError("Save must be a boolean value!")
    if pdb.mode != "singleresidue":
        raise ValueError(
            "This graph is only available for singleresidue mode. Run calculate_frustration() with mode='singleresidue'"
        )
    method = method.lower()
    if method not in ["threading", "modeller"]:
        raise ValueError(
            f"{method} is not a valid mutation method. Available methods are: threading or modeller"
        )

    # Check if mutation data exists
    mutation_key = f"Res_{res_num}_{chain}"
    if method not in pdb.Mutations or mutation_key not in pdb.Mutations[method]:
        raise ValueError(
            f"No mutation data found for residue {res_num} chain {chain} using {method} method"
        )

    # Get mutation data
    mutation = pdb.Mutations[method][mutation_key]

    # Read frustration data
    data_frus = pd.read_csv(
        mutation["File"],
        sep="\s+",
        header=0,
        names=["Res1", "Chain1", "AA1", "FrstIndex"],
    )

    # Classify frustration states
    data_frus["FrstState"] = pd.cut(
        data_frus["FrstIndex"],
        bins=[-float("inf"), -1.0, 0.58, float("inf")],
        labels=["highly", "neutral", "minimally"],
    )

    # Get native residue and its frustration
    logger = logging.getLogger(__name__)

    # Get native residue from pdb structure
    native_res = pdb.atom[
        (pdb.atom["res_num"] == mutation["Res"])
        & (pdb.atom["chain"] == mutation["Chain"])
    ]["res_name"].iloc[0]

    # Convert 3-letter code to 1-letter code if needed
    if len(native_res) == 3:
        aa_codes = {
            "ALA": "A",
            "CYS": "C",
            "ASP": "D",
            "GLU": "E",
            "PHE": "F",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LYS": "K",
            "LEU": "L",
            "MET": "M",
            "ASN": "N",
            "PRO": "P",
            "GLN": "Q",
            "ARG": "R",
            "SER": "S",
            "THR": "T",
            "VAL": "V",
            "TRP": "W",
            "TYR": "Y",
        }
        native_res = aa_codes[native_res]

    # Find native residue's frustration
    native_mask = data_frus["AA1"] == native_res
    if not native_mask.any():
        logger.error(f"No native residue {native_res} found in frustration data")
        raise ValueError(f"No native residue {native_res} found in frustration data")

    native_frst = data_frus.loc[native_mask, "FrstIndex"].values[0]
    logger.debug(f"Native frustration value: {native_frst}")
    data_frus["DeltaFrst"] = data_frus["FrstIndex"] - native_frst

    # Create plot
    fig = go.Figure()

    # Add traces for each frustration state
    colors = {"highly": "red", "neutral": "gray", "minimally": "green"}

    # Plot non-native residues first
    for state in ["highly", "neutral", "minimally"]:
        mask = (data_frus["FrstState"] == state) & (data_frus["AA1"] != native_res)
        fig.add_trace(
            go.Scatter(
                x=data_frus[mask]["Res1"],
                y=data_frus[mask]["DeltaFrst"],
                mode="text",
                name=f"{state.capitalize()} frustrated",
                text=data_frus[mask]["AA1"],
                textfont=dict(
                    color=colors[state],
                    size=14,
                    family="Arial",
                ),
                showlegend=True,
            )
        )

    # Plot native residue last (on top)
    native_mask = data_frus["AA1"] == native_res
    fig.add_trace(
        go.Scatter(
            x=data_frus[native_mask]["Res1"],
            y=data_frus[native_mask]["DeltaFrst"],
            mode="text",
            name="Native state",
            text=data_frus[native_mask]["AA1"],
            textfont=dict(
                color="blue",
                size=14,
                family="Arial",
            ),
            showlegend=True,
        )
    )

    # Update layout for publication quality
    fig.update_layout(
        title=dict(
            text=f"Delta Frustration for Residue {res_num} Chain {chain}",
            font=dict(size=16, family="Arial"),
            x=0.5,  # Center the title
            y=0.95,
        ),
        xaxis=dict(
            title="Residue Position",
            titlefont=dict(size=14, family="Arial"),
            tickfont=dict(size=12, family="Arial"),
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.2)",
            dtick=1,  # Force integer ticks
            tick0=res_num,  # Start from the actual residue number
            range=[
                res_num - 0.5,
                res_num + 0.5,
            ],  # Limit x-axis range to just show the residue
        ),
        yaxis=dict(
            title="ΔFrustration",  # Using proper delta symbol
            titlefont=dict(size=14, family="Arial"),
            tickfont=dict(size=12, family="Arial"),
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="rgba(0,0,0,0.2)",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            font=dict(size=12, family="Arial"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
        width=600,
        height=500,
    )

    # Add horizontal lines at important thresholds with improved styling
    fig.add_hline(
        y=0.58,
        line_dash="dash",
        line_color="rgba(128,128,128,0.5)",
        line_width=1,
    )
    fig.add_hline(
        y=-1.0,
        line_dash="dash",
        line_color="rgba(128,128,128,0.5)",
        line_width=1,
    )

    # Add subtle box around the plot
    fig.update_layout(
        xaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True),
        yaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True),
    )

    if save:
        # Create output directory if it doesn't exist
        output_dir = os.path.join(pdb.job_dir, "MutationsData", "Images")
        os.makedirs(output_dir, exist_ok=True)

        # Base filename
        output_base = os.path.join(
            output_dir, f"Delta_frus_res{int(res_num)}_chain{chain}"
        )

        # Always save HTML
        fig.write_html(f"{output_base}.html")

        # Try to save PNG, but don't fail if it doesn't work
        try:
            fig.write_image(f"{output_base}.png", scale=4)
        except (ImportError, TypeError) as e:
            warnings.warn(
                f"Could not save PNG image due to missing dependencies: {str(e)}\n"
                "Only HTML file was saved. To save static images, install kaleido:\n"
                "pip install -U kaleido"
            )

    if show:
        fig.show()

    return fig
