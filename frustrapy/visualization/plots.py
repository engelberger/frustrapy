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
        show (bool, optional): If True, displays the plot. Default: False.

    Returns:
        plotly.graph_objects.Figure: The generated interactive plot.
    """
    logger = logging.getLogger(__name__)
    logger.debug("Starting contact map generation")

    # Input validation
    if save not in [True, False]:
        raise ValueError("Save must be a boolean value!")

    if chain is not None:
        if len(chain) > 1:
            raise ValueError("You must enter only one Chain!")
        if chain not in pdb.atom["chain"].unique():
            available_chains = ", ".join(pdb.atom["chain"].unique())
            raise ValueError(
                f"The Chain {chain} doesn't exist! Available chains: {available_chains}"
            )

    # Create output directory if needed
    output_dir = os.path.join(pdb.job_dir, "Images")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.debug(f"Created output directory: {output_dir}")

    # Read frustration density data
    density_file = os.path.join(
        pdb.job_dir, "FrustrationData", f"{pdb.pdb_base}.pdb_{pdb.mode}_5adens"
    )
    logger.debug(f"Reading density data from: {density_file}")
    density_data = pd.read_csv(density_file, sep=" ", header=0)

    # Read frustration contact data
    contacts_file = os.path.join(
        pdb.job_dir, "FrustrationData", f"{pdb.pdb_base}.pdb_{pdb.mode}"
    )
    logger.debug(f"Reading contact data from: {contacts_file}")
    contact_data = pd.read_csv(contacts_file, sep=" ", engine="python")

    # Filter by chain if specified
    if chain is not None:
        contact_data = contact_data[contact_data["ChainRes1"] == chain]
        logger.debug(f"Filtered contacts for chain {chain}")

    # Convert chain residue identifiers to strings, handling missing values
    for chain_col in ["ChainRes1", "ChainRes2"]:
        contact_data[chain_col] = contact_data[chain_col].apply(
            lambda x: str(x) if pd.notnull(x) else "Unknown"
        )

    # Convert residue numbers to numeric, allowing for missing values
    for res_col in ["Res1", "Res2"]:
        contact_data[res_col] = pd.to_numeric(contact_data[res_col], errors="coerce")

    # Get unique chains in sorted order
    unique_chains = sorted(
        set(contact_data["ChainRes1"].tolist() + contact_data["ChainRes2"].tolist())
    )
    logger.debug(f"Found chains: {unique_chains}")

    # Calculate position ranges for each chain
    chain_positions = []  # Stores [min_pos, max_pos, range] for each chain
    residue_positions = []  # Stores actual residue numbers for axis labels

    for chain_id in unique_chains:
        if chain_id != "Unknown":
            # Ensure residue numbers are numeric
            contact_data["Res1"] = pd.to_numeric(contact_data["Res1"], errors="coerce")
            contact_data["Res2"] = pd.to_numeric(contact_data["Res2"], errors="coerce")

            # Remove rows with missing residue numbers
            contact_data.dropna(subset=["Res1", "Res2"], inplace=True)

            # Get residue ranges for this chain
            chain_res1 = contact_data[contact_data["ChainRes1"] == chain_id]["Res1"]
            chain_res2 = contact_data[contact_data["ChainRes2"] == chain_id]["Res2"]

            min_position = min(chain_res1.min(), chain_res2.min())
            max_position = max(chain_res1.max(), chain_res2.max())
            position_range = max_position - min_position + 1

            chain_positions.append([min_position, max_position, position_range])
            residue_positions.extend(range(int(min_position), int(max_position) + 1))

            logger.debug(f"Chain {chain_id}: range {min_position}-{max_position}")

    # Initialize position columns for contact mapping
    contact_data["pos1"] = 0
    contact_data["pos2"] = 0

    # Calculate adjusted positions with chain offsets
    cumulative_offset = 0
    for i, chain_id in enumerate(unique_chains):
        if i > 0:
            cumulative_offset = sum(pos[2] for pos in chain_positions[:i])

        # Adjust positions for current chain
        chain_mask1 = contact_data["ChainRes1"] == chain_id
        chain_mask2 = contact_data["ChainRes2"] == chain_id

        contact_data.loc[chain_mask1, "pos1"] = (
            contact_data.loc[chain_mask1, "Res1"]
            - chain_positions[i][0]
            + cumulative_offset
            + 1
        )
        contact_data.loc[chain_mask2, "pos2"] = (
            contact_data.loc[chain_mask2, "Res2"]
            - chain_positions[i][0]
            + cumulative_offset
            + 1
        )

    # Calculate final position ranges for visualization
    final_positions = []
    for chain_id in unique_chains:
        chain_pos1 = contact_data[contact_data["ChainRes1"] == chain_id]["pos1"]
        chain_pos2 = contact_data[contact_data["ChainRes2"] == chain_id]["pos2"]

        min_pos = min(chain_pos1.min(), chain_pos2.min())
        max_pos = max(chain_pos1.max(), chain_pos2.max())
        final_positions.append([min_pos, max_pos, max_pos - min_pos + 1])

    total_positions = sum(pos[2] for pos in final_positions)
    logger.debug(f"Total matrix size: {total_positions}x{total_positions}")

    # Create contact matrix with explicit float dtype
    contact_matrix = pd.DataFrame(
        np.zeros((total_positions, total_positions), dtype=np.float64),
        index=range(1, total_positions + 1),
        columns=range(1, total_positions + 1),
    )

    # Fill matrix with frustration values
    for _, contact in contact_data.iterrows():
        pos1_idx = int(contact["pos1"] - 1)
        pos2_idx = int(contact["pos2"] - 1)
        frustration_value = contact["FrstIndex"]

        contact_matrix.iloc[pos2_idx, pos1_idx] = frustration_value
        contact_matrix.iloc[pos1_idx, pos2_idx] = frustration_value

    # Set lower triangle to zero for visualization
    contact_matrix.values[np.tril_indices(contact_matrix.shape[0], k=0)] = 0.0

    # Create plotly heatmap
    logger.debug("Generating plotly heatmap")
    fig = go.Figure(
        data=go.Heatmap(
            z=contact_matrix.values,
            x=list(contact_matrix.columns),
            y=list(contact_matrix.index),
            colorscale=[[0, "red"], [0.5, "white"], [1, "green"]],
            zmin=-4,
            zmax=4,
            colorbar=dict(title="Frustration Index"),
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Contact map {pdb.pdb_base}",
        xaxis_title="Residue i",
        yaxis_title="Residue j",
        font=dict(family="Arial", size=12, color="black"),
    )

    # Add chain separators and labels for multiple chains
    if len(unique_chains) > 1:
        # Calculate tick positions and labels
        tick_positions = [round(x) for x in np.linspace(1, total_positions, 15)]
        tick_labels = [str(residue_positions[pos - 1]) for pos in tick_positions]

        # Add chain separator lines
        chain_boundaries = np.cumsum([p[2] for p in final_positions[:-1]])
        for boundary in chain_boundaries:
            # Add vertical separator
            fig.add_shape(
                type="line",
                x0=boundary,
                y0=0,
                x1=boundary,
                y1=total_positions,
                line=dict(color="gray", width=0.5, dash="dash"),
            )
            # Add horizontal separator
            fig.add_shape(
                type="line",
                x0=0,
                y0=boundary,
                x1=total_positions,
                y1=boundary,
                line=dict(color="gray", width=0.5, dash="dash"),
            )

        # Add chain labels
        for i, chain_id in enumerate(unique_chains):
            mean_position = np.mean(final_positions[i][:2])
            # Add top label
            fig.add_annotation(
                x=mean_position,
                y=total_positions + 0.5,
                text=chain_id,
                showarrow=False,
                font=dict(color="gray"),
            )
            # Add right label
            fig.add_annotation(
                x=total_positions + 0.5,
                y=mean_position,
                text=chain_id,
                showarrow=False,
                font=dict(color="gray"),
            )

        # Update axis labels
        fig.update_xaxes(tickvals=tick_positions, ticktext=tick_labels)
        fig.update_yaxes(tickvals=tick_positions, ticktext=tick_labels)

    # Save plot if requested
    if save:
        logger.debug("Saving contact map")
        fig.update_layout(width=1000, height=1000)

        # Save PNG version
        png_path = os.path.join(output_dir, f"{pdb.pdb_base}_{pdb.mode}_map.png")
        fig.write_image(png_path)
        logger.debug(f"Saved PNG to: {png_path}")

        # Save HTML version
        html_path = os.path.join(output_dir, f"{pdb.pdb_base}_{pdb.mode}_map.html")
        fig.write_html(html_path)
        logger.debug(f"Saved HTML to: {html_path}")

        logger.debug(f"Contact map is stored in {html_path}")

    if show:
        fig.show()

    logger.debug("Contact map generation complete")
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
            logger.debug(
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
            logger.debug(
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
            logger.debug(
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
            logger.debug(
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
