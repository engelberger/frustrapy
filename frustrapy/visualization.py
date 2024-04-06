import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.PDB import PDBParser, Select


def plot_5andens(pdb, chain=None, save=False):
    """
    Generates plot to analyze the density of contacts around a sphere of 5 Armstrongs,
    centered in the C-alfa atom from the residue. The different classes of contacts
    based on the mutational frustration index are counted in absolute terms.

    Args:
        pdb (Pdb): Pdb Frustration object.
        chain (str, optional): Chain of residue. Default: None.
        save (bool, optional): If True, saves the graph; otherwise, it does not. Default: False.

    Returns:
        matplotlib.figure.Figure: The generated plot.
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
    print(
        os.path.join(
            pdb.job_dir, "FrustrationData", f"{pdb.pdb_base}.pdb_{pdb.mode}_5adens"
        )
    )
    adens_table = pd.read_csv(
        os.path.join(
            pdb.job_dir, "FrustrationData", f"{pdb.pdb_base}.pdb_{pdb.mode}_5adens"
        ),
        sep=" ",
        header=0,

    )
    adens_table["PositionsTotal"] = range(1, len(adens_table) + 1)

    if chain is None:
        maximum = max(
            adens_table[
                ["HighlyFrst", "MinimallyFrst", "NeutrallyFrst", "Total"]
            ].max()
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            adens_table["PositionsTotal"],
            adens_table["HighlyFrst"],
            color="red",
            label="Highly frustrated",
        )
        ax.plot(
            adens_table["PositionsTotal"],
            adens_table["NeutrallyFrst"],
            color="gray",
            label="Neutral",
        )
        ax.plot(
            adens_table["PositionsTotal"],
            adens_table["MinimallyFrst"],
            color="green",
            label="Minimally frustrated",
        )
        ax.plot(
            adens_table["PositionsTotal"],
            adens_table["Total"],
            color="black",
            label="Total",
        )
        ax.set_title(f"Density around 5A sphere (%) in {pdb.pdb_base}")
        ax.set_ylabel("Local frustration density (5A sphere)")
        ax.set_xlabel("Position")
        ax.legend()
        ax.set_yticks(range(0, int(maximum) + 1, 5))
        ax.set_xticks(range(1, len(adens_table) + 1, (len(adens_table) - 1) // 10))

        if save:
            plt.savefig(
                os.path.join(
                    pdb.job_dir, "Images", f"{pdb.pdb_base}_{pdb.mode}.png_5Adens.png"
                )
            )
            print(
                f"5Adens plot is stored in {os.path.join(pdb.job_dir, 'Images', f'{pdb.pdb_base}_{pdb.mode}.png_5Adens.png')}"
            )
    else:
        adens_table = adens_table[adens_table["Chains"] == chain]
        maximum = max(
            adens_table[
                ["HighlyFrst", "MinimallyFrst", "NeutrallyFrst", "Total"]
            ].max()
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            adens_table["Positions"],
            adens_table["HighlyFrst"],
            color="red",
            label="Highly frustrated",
        )
        ax.plot(
            adens_table["Positions"],
            adens_table["NeutrallyFrst"],
            color="gray",
            label="Neutral",
        )
        ax.plot(
            adens_table["Positions"],
            adens_table["MinimallyFrst"],
            color="green",
            label="Minimally frustrated",
        )
        ax.plot(
            adens_table["Positions"], adens_table["Total"], color="black", label="Total"
        )
        ax.set_title(f"Density around 5A sphere (%) in {pdb.pdb_base} chain {chain}")
        ax.set_ylabel("Local frustration density (5A sphere)")
        ax.set_xlabel("Position")
        ax.legend()
        ax.set_yticks(range(0, maximum + 1, 5))
        ax.set_xticks(
            range(
                adens_table["Positions"].iloc[0],
                adens_table["Positions"].iloc[-1] + 1,
                (len(adens_table) - 1) // 10,
            )
        )

        if save:
            plt.savefig(
                os.path.join(
                    pdb.job_dir,
                    "Images",
                    f"{pdb.pdb_base}_{pdb.mode}_5Adens__chain{chain}.png",
                )
            )
            print(
                f"5Adens plot is stored in {os.path.join(pdb.job_dir, 'Images', f'{pdb.pdb_base}_{pdb.mode}_5Adens__chain{chain}.png')}"
            )

    return fig


def plot_5adens_proportions(pdb, chain=None, save=False):
    """
    Generates plot to analyze the density of contacts around a sphere of 5 Armstrongs,
    centered in the C-alfa atom from the residue. The different classes of contacts
    based on the mutational frustration index are counted in relative terms.

    Args:
        pdb (Pdb): Pdb Frustration object.
        chain (str, optional): Chain of residue. Default: None.
        save (bool, optional): If True, saves the graph; otherwise, it does not. Default: False.

    Returns:
        matplotlib.figure.Figure: The generated plot.
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

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="Positions",
        y="Density",
        hue="Frustration",
        data=frustration_data,
        palette=["red", "gray", "green"],
        ax=ax,
    )
    ax.set_xticks(range(0, len(adens_table), (len(adens_table) - 1) // 10))
    ax.set_xlabel("Positions")
    ax.set_ylabel("Density around 5A sphere (%)")
    if chain is None:
        ax.set_title(f"Density around 5A sphere (%) in {pdb.pdb_base}")
    else:
        ax.set_title(f"Density around 5A sphere (%) in {pdb.pdb_base} chain {chain}")
    ax.legend(title="")

    if save:
        if chain is None:
            plt.savefig(
                os.path.join(
                    pdb.job_dir,
                    "Images",
                    f"{pdb.pdb_base}_{pdb.mode}_5Adens_around.png",
                )
            )
            print(
                f"5Adens proportion plot is stored in {os.path.join(pdb.job_dir, 'Images', f'{pdb.pdb_base}_{pdb.mode}_5Adens_around.png')}"
            )
        else:
            plt.savefig(
                os.path.join(
                    pdb.job_dir,
                    "Images",
                    f"{pdb.pdb_base}_{pdb.mode}_5Adens_around_chain{chain}.png",
                )
            )
            print(
                f"5Adens proportion plot is stored in {os.path.join(pdb.job_dir, 'Images', f'{pdb.pdb_base}_{pdb.mode}_5Adens_around_chain{chain}.png')}"
            )

    return fig


def plot_contact_map(pdb, chain=None, save=False):
    """
    Generates contact map plot to visualize the frustration values assigned to each contact.

    Args:
        pdb (Pdb): Pdb Frustration object.
        chain (str, optional): Chain of residue. Default: None.
        save (bool, optional): If True, saves the graph; otherwise, it does not. Default: False.

    Returns:
        matplotlib.figure.Figure: The generated plot.
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
    # datos = pd.read_csv(
    #    os.path.join(pdb.job_dir, "FrustrationData", f"{pdb.pdb_base}.pdb_{pdb.mode}"),
    #    sep="\s+",
    # )
    datos = pd.read_csv(
        os.path.join(pdb.job_dir, "FrustrationData", f"{pdb.pdb_base}.pdb_{pdb.mode}"),
        sep=" ",  # Use a single space as the separator
        engine="python",  # Use the Python parsing engine
    )
    # Assert that the columns are as expected
    columns = [
        "Res1",
        "Res2",
        "ChainRes1",
        "ChainRes2",
        "AA1",
        "AA2",
        "NativeEnergy",
        "DecoyEnergy",
        "SDEnergy",
        "FrstIndex",
        "FrstState",
    ]

    # assert datos.columns.tolist() == columns

    if chain is not None:
        datos = datos[datos["ChainRes1"] == chain]

    # Convert chain identifiers to strings, ignoring NaNs or converting them to a placeholder
    datos["ChainRes1"] = datos["ChainRes1"].apply(
        lambda x: str(x) if pd.notnull(x) else "Unknown"
    )
    datos["ChainRes2"] = datos["ChainRes2"].apply(
        lambda x: str(x) if pd.notnull(x) else "Unknown"
    )
    # Convert 'Res1' and 'Res2' to numeric, coercing errors to NaN
    datos["Res1"] = pd.to_numeric(datos["Res1"], errors="coerce")
    datos["Res2"] = pd.to_numeric(datos["Res2"], errors="coerce")

    # Now attempt to sort the unique set of chain identifiers
    chains = sorted(set(datos["ChainRes1"].tolist() + datos["ChainRes2"].tolist()))
    chains_two = sorted(set(datos["ChainRes1"].tolist() + datos["ChainRes2"].tolist()))
    print(chains_two)
    # chains = sorted(set(datos["ChainRes1"].tolist() + datos["ChainRes2"].tolist()))
    positions = []
    aux_pos_vec = []
    for c in chains_two:
        # If c is not 'Unknown'
        if c != "Unknown":
            # Convert 'Res1' and 'Res2' to numeric, coercing errors to NaN
            datos["Res1"] = pd.to_numeric(datos["Res1"], errors="coerce")
            datos["Res2"] = pd.to_numeric(datos["Res2"], errors="coerce")

            # Optionally, handle rows with NaN in 'Res1' or 'Res2' after conversion
            # For example, you could drop these rows
            datos.dropna(subset=["Res1", "Res2"], inplace=True)
            #
            res1_range = datos[datos["ChainRes1"] == c]["Res1"]
            res2_range = datos[datos["ChainRes2"] == c]["Res2"]
            # Now your comparison should work without errors
            min_pos = min(res1_range.min(), res2_range.min())
            # min_pos = min(res1_range.min(), res2_range.min())
            max_pos = max(res1_range.max(), res2_range.max())
            positions.append([min_pos, max_pos, max_pos - min_pos + 1])
            aux_pos_vec.extend(range(int(min_pos), int(max_pos) + 1))

    datos["pos1"] = 0
    datos["pos2"] = 0
    bias = 0
    for i, c in enumerate(chains):
        if i > 0:
            bias = sum(pos[2] for pos in positions[:i])
        idx1 = datos["ChainRes1"] == c
        datos.loc[idx1, "pos1"] = datos.loc[idx1, "Res1"] - positions[i][0] + bias + 1
        idx2 = datos["ChainRes2"] == c
        datos.loc[idx2, "pos2"] = datos.loc[idx2, "Res2"] - positions[i][0] + bias + 1

    pos_new = []
    for c in chains:
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

    # Fill NaN values with a default number, e.g., 0, as seaborn's heatmap cannot handle NaN values
    matrz = matrz.fillna(0)

    # Make the matrix upper triangular
    matrz.values[np.tril_indices(matrz.shape[0], k=0)] = 0

    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["red", "grey", "green"], N=256
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrz, cmap=cmap, vmin=-4, vmax=4, square=True, linewidths=0.5, ax=ax)
    ax.set_xlabel("Residue i")
    ax.set_ylabel("Residue j")
    ax.set_title(f"Contact map {pdb.pdb_base}")

    if len(chains) > 1:
        breaks = [round(x) for x in np.linspace(1, total_positions, 15)]
        labels = [str(aux_pos_vec[b - 1]) for b in breaks]
        for pos in np.cumsum([p[2] for p in pos_new[:-1]]):
            ax.axvline(pos, color="gray", linestyle="--", linewidth=0.5)
            ax.axhline(pos, color="gray", linestyle="--", linewidth=0.5)

        for i, c in enumerate(chains):
            mean_pos = np.mean(pos_new[i][:2])
            ax.text(
                mean_pos,
                total_positions + 0.5,
                c,
                color="gray",
                ha="center",
                va="center",
            )
            ax.text(
                total_positions + 0.5,
                mean_pos,
                c,
                color="gray",
                ha="center",
                va="center",
            )

        ax.set_xticks(breaks)
        ax.set_xticklabels(labels)
        ax.set_yticks(breaks)
        ax.set_yticklabels(labels)

    if save:
        plt.savefig(
            os.path.join(pdb.job_dir, "Images", f"{pdb.pdb_base}_{pdb.mode}_map.png")
        )
        print(
            f"Contact map is stored in {os.path.join(pdb.job_dir, 'Images', f'{pdb.pdb_base}_{pdb.mode}_map.png')}"
        )

    return fig

def view_frustration_pymol(pdb):
    """
    Generates a PyMOL session to observe the frustration patterns on top of the PDB protein structure.

    Args:
        pdb (Pdb): Pdb Frustration object.
    """
    try:
        import pymol
    except ImportError:
        raise ImportError(
            "PyMOL is required to use this function. Please install PyMOL and try again."
        )

    pymol.cmd.load(os.path.join(pdb.job_dir, "FrustrationData", f"{pdb.pdb_base}.pdb"))
    pymol.cmd.hide("everything")
    pymol.cmd.show("cartoon")
    pymol.cmd.color("gray", "all")

    pymol.cmd.select(
        "highly_frustrated",
        f"resi {'+'.join(map(str, pdb.highly_frustrated_residues))}",
    )
    pymol.cmd.select("neutral", f"resi {'+'.join(map(str, pdb.neutral_residues))}")
    pymol.cmd.select(
        "minimally_frustrated",
        f"resi {'+'.join(map(str, pdb.minimally_frustrated_residues))}",
    )

    pymol.cmd.color("red", "highly_frustrated")
    pymol.cmd.color("gray", "neutral")
    pymol.cmd.color("green", "minimally_frustrated")

    pymol.cmd.zoom()
    pymol.cmd.orient()
    pymol.cmd.center("all")
    pymol.cmd.set("ray_opaque_background", 0)
    pymol.cmd.png(
        os.path.join(pdb.job_dir, "Images", f"{pdb.pdb_base}_{pdb.mode}_pymol.png"),
        width=1200,
        height=800,
        dpi=300,
        ray=1,
    )
    print(
        f"PyMOL session is stored in {os.path.join(pdb.job_dir, 'Images', f'{pdb.pdb_base}_{pdb.mode}_pymol.png')}"
    )


def plot_delta_frus(pdb, resno, chain, method="threading", save=False):
    """
    Generate a plot of the single residue frustration difference for the mutation of the specific residue given by mutate_res.

    Args:
        pdb (Pdb): Pdb frustration object.
        resno (int): Specific residue.
        chain (str): Specific chain.
        method (str): Method indicates the method to use to perform the mutation (Threading or Modeller). Default: "threading".
        save (bool): If True, saves the graph; otherwise, it does not. Default: False.

    Returns:
        matplotlib.figure.Figure: The generated plot.
    """
    if save not in [True, False]:
        raise ValueError("Save must be a boolean value!")
    if pdb.mode != "singleresidue":
        raise ValueError(
            "This graph is available for singleresidue index, run calculate_frustration() with Mode='singleresidue' and mutate_res()"
        )
    method = method.lower()
    if method not in ["threading", "modeller"]:
        raise ValueError(
            f"{method} is not a valid mutation method. The available methods are: threading or modeller!"
        )
    if (
        method not in pdb.mutations
        or f"Res_{resno}_{chain}" not in pdb.mutations[method]
    ):
        raise ValueError(
            f"Residue {resno} from chain {chain} was not mutated using the {method} method."
        )

    mutation = pdb.mutations[method][f"Res_{resno}_{chain}"]
    data_frus = pd.read_csv(
        mutation["File"],
        sep="\s+",
        header=0,
        names=["Res1", "Chain1", "AA1", "FrstIndex"],
    )
    data_frus["FrstState"] = pd.cut(
        data_frus["FrstIndex"],
        bins=[-float("inf"), -1.0, 0.58, float("inf")],
        labels=["highly", "neutral", "minimally"],
    )
    data_frus["Color"] = data_frus["FrstState"].map(
        {"neutral": "gray", "highly": "red", "minimally": "green"}
    )

    # Native residues
    native = pdb.atom[
        (pdb.atom["resno"] == mutation["Res"])
        & (pdb.atom["chain"] == mutation["Chain"])
    ]["resid"].unique()[0]
    data_frus.loc[data_frus["AA1"] == native, "Color"] = "blue"

    # Delta frustration
    frst_index_native = data_frus.loc[data_frus["AA1"] == native, "FrstIndex"].values[0]
    data_frus["FrstIndex"] = data_frus["FrstIndex"] - frst_index_native

    x1, x2 = data_frus["Res1"].min() - 3, data_frus["Res1"].max() + 3
    y1, y2 = min(data_frus["FrstIndex"].min(), -4), max(data_frus["FrstIndex"].max(), 4)

    data_frus = pd.concat(
        [
            data_frus[data_frus["Color"] != "blue"],
            data_frus[data_frus["Color"] == "blue"],
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x="Res1", y="FrstIndex", hue="Color", style="AA1", data=data_frus, s=100, ax=ax
    )
    ax.set_xlabel("Residue Position")
    ax.set_ylabel("Delta Frustration")
    ax.set_xticks(data_frus["Res1"].unique())
    ax.set_xticklabels(data_frus["Res1"].unique())
    ax.set_yticks(np.arange(y1, y2 + 0.5, 0.5))
    ax.set_yticklabels(np.arange(y1, y2 + 0.5, 0.5))
    ax.set_ylim(y1, y2)
    ax.legend(
        title="Frustration",
        labels=["Native state", "Neutral", "Minimally frustrated", "Highly frustrated"],
    )
    plt.xticks(rotation=90)
    plt.tight_layout()

    if save:
        os.makedirs(os.path.join(pdb.job_dir, "MutationsData/Images"), exist_ok=True)
        plt.savefig(
            os.path.join(
                pdb.job_dir, f"MutationsData/Images/Delta_frus_{resno}_{chain}.png"
            )
        )
        print(
            f"Delta frus plot is stored in {os.path.join(pdb.job_dir, f'MutationsData/Images/Delta_frus_{resno}_{chain}.png')}"
        )

    return fig


def plot_mutate_res(pdb, resno, chain, method="threading", save=False):
    """
    Plot the frustration for each of the 20 residue variants at a given position in the structure.

    Args:
        pdb (Pdb): Pdb frustration object.
        resno (int): Specific residue.
        chain (str): Specific chain.
        method (str): Method indicates the method to use to perform the mutation (Threading or Modeller). Default: "threading".
        save (bool): If True, saves the graph; otherwise, it does not. Default: False.

    Returns:
        matplotlib.figure.Figure: The generated plot.
    """
    if save not in [True, False]:
        raise ValueError("Save must be a boolean value!")
    method = method.lower()
    if method not in ["threading", "modeller"]:
        raise ValueError(
            f"{method} is not a valid mutation method. The available methods are: threading or modeller!"
        )
    if (
        method not in pdb.mutations
        or f"Res_{resno}_{chain}" not in pdb.mutations[method]
    ):
        raise ValueError(
            f"Residue {resno} from chain {chain} was not mutated using the {method} method."
        )

    mutation = pdb.mutations[method][f"Res_{resno}_{chain}"]

    if pdb.mode in ["configurational", "mutational"]:
        data_frus = pd.read_csv(
            mutation["File"],
            sep="\s+",
            header=0,
            names=[
                "Res1",
                "Res2",
                "Chain1",
                "Chain2",
                "AA1",
                "AA2",
                "FrstIndex",
                "FrstState",
            ],
        )
        data_frus["FrstState"] = pd.cut(
            data_frus["FrstIndex"],
            bins=[-float("inf"), -1.0, 0.78, float("inf")],
            labels=["highly", "neutral", "minimally"],
        )
        data_frus.loc[
            data_frus["Res2"] == mutation["Res"], ["Res2", "Chain2", "AA1", "AA2"]
        ] = data_frus.loc[
            data_frus["Res2"] == mutation["Res"], ["Res1", "Chain1", "AA2", "AA1"]
        ].values
        data_frus.loc[data_frus["Chain1"] != mutation["Chain"], "Chain1"] = mutation[
            "Chain"
        ]
        data_frus.loc[data_frus["Res1"] != mutation["Res"], "Res1"] = mutation["Res"]
    elif pdb.mode == "singleresidue":
        data_frus = pd.read_csv(
            mutation["File"],
            sep="\s+",
            header=0,
            names=["Res1", "Chain1", "AA1", "FrstIndex"],
        )
        data_frus["FrstState"] = pd.cut(
            data_frus["FrstIndex"],
            bins=[-float("inf"), -1.0, 0.58, float("inf")],
            labels=["highly", "neutral", "minimally"],
        )

    # Native residue
    native = pdb.atom[
        (pdb.atom["resno"] == mutation["Res"])
        & (pdb.atom["chain"] == mutation["Chain"])
    ]["resid"].unique()[0]

    data_frus["Color"] = data_frus["FrstState"].map(
        {"neutral": "gray", "highly": "red", "minimally": "green"}
    )
    data_frus.loc[data_frus["AA1"] == native, "Color"] = "blue"

    if pdb.mode in ["configurational", "mutational"]:
        contacts = (
            data_frus[["Res2", "Chain2"]]
            .drop_duplicates()
            .sort_values("Res2")
            .reset_index(drop=True)
        )
        contacts["Index"] = contacts.index + 1
        contacts["Res2"] = contacts["Res2"].astype(int)

        resid = pdb.atom[
            (pdb.atom["resno"].isin(contacts["Res2"]))
            & (pdb.atom["chain"] == mutation["Chain"])
            & (pdb.atom["elety"] == "CA")
        ]["resid"].values

        data_frus["Res2"] = (
            contacts.set_index("Res2")["Index"].reindex(data_frus["Res2"]).values
        )

        data_frus = pd.concat(
            [
                data_frus[data_frus["Color"] != "blue"],
                data_frus[data_frus["Color"] == "blue"],
            ]
        )

        y1, y2 = -4, 4
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x="Res2",
            y="FrstIndex",
            hue="Color",
            style="AA1",
            data=data_frus,
            s=100,
            ax=ax,
        )
        ax.set_xlabel("Contact residue")
        ax.set_ylabel("Frustration Index")
        ax.set_xticks(contacts["Index"])
        ax.set_xticklabels([f"{r} {c}" for r, c in zip(resid, contacts["Res2"])])
        ax.set_yticks(np.arange(y1, y2 + 0.5, 0.5))
        ax.set_yticklabels(np.arange(y1, y2 + 0.5, 0.5))
        ax.set_ylim(y1, y2)
        ax.axhline(y=0.78, color="gray", linestyle="--")
        ax.axhline(y=-1, color="gray", linestyle="--")
        ax.set_title(
            f"Contact Frustration {pdb.mode} of residue {native}_{data_frus['Res1'].iloc[0]}"
        )
        ax.legend(
            title="",
            labels=["Minimally frustrated", "Neutral", "Highly frustrated", "Native"],
        )
        plt.xticks(rotation=90)
        plt.tight_layout()

    elif pdb.mode == "singleresidue":
        y1, y2 = -4, 4
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x="AA1", y="FrstIndex", hue="Color", data=data_frus, s=100, ax=ax
        )
        ax.set_xlabel("Residue")
        ax.set_ylabel("Frustration Index")
        ax.set_yticks(np.arange(y1, y2 + 0.5, 0.5))
        ax.set_yticklabels(np.arange(y1, y2 + 0.5, 0.5))
        ax.set_ylim(y1, y2)
        ax.axhline(y=0.58, color="gray", linestyle="--")
        ax.axhline(y=-1, color="gray", linestyle="--")
        ax.set_title(
            f"Frustration of the 20 variants in position {mutation['Res']} of the structure"
        )
        ax.legend(
            title="",
            labels=["Minimally frustrated", "Neutral", "Highly frustrated", "Native"],
        )
        plt.xticks(rotation=90)
        plt.tight_layout()

    if save:
        os.makedirs(os.path.join(pdb.job_dir, "MutationsData/Images"), exist_ok=True)
        plt.savefig(
            os.path.join(
                pdb.job_dir,
                f"MutationsData/Images/{pdb.mode}_{data_frus['Res1'].iloc[0]}_{mutation['Method']}_{mutation['Chain']}.png",
            )
        )
        # print(f"Mutate res plot is stored in {os.path.join(pdb.job_dir, f"MutationsData/Images/{pdb.mode}_{data_frus['Res1'].iloc[0]}_{mutation['Method']}_{mutation['Chain']}.png")}")

    return fig

