from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)


@dataclass
class GeneratorData:
    """Stores data for histogram generation"""

    residue: int
    aa_ref: str
    num_ref: str
    prot_ref: str
    min_percent: float
    neu_percent: float
    max_percent: float
    min_count: int
    neu_count: int
    max_count: int
    min_ic: float
    neu_ic: float
    max_ic: float
    total_ic: float
    frust_state: str


@dataclass
class Contact:
    """Data class for contact information"""

    residue1: int
    residue2: int
    frustration_index: float
    frustration_state: str
    structure: str


class HistogramGenerator:
    """Generates frustration histograms and contact map visualizations"""

    def __init__(
        self, results_dir: Path, msa_file: Path, reference_pdb: Optional[str] = None
    ):
        """
        Initialize histogram generator.

        Args:
            results_dir: Directory for results
            msa_file: Path to MSA file
            reference_pdb: Reference PDB identifier
        """
        self.results_dir = Path(results_dir)
        self.msa_file = Path(msa_file)
        self.reference_pdb = reference_pdb
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

    def generate_contact_maps(self, ic_results: pd.DataFrame) -> None:
        """
        Generate contact map visualizations.

        Args:
            ic_results: DataFrame containing information content results
        """
        try:
            logger.info("Generating contact maps")
            # Log the columns in the DataFrame
            logger.debug(f"Columns in ic_results: {ic_results.columns}")

            # Create contact matrix
            max_residue = max(max(ic_results["Res1"]), max(ic_results["Res2"]))
            contact_matrix = np.zeros((max_residue + 1, max_residue + 1))
            state_matrix = np.full((max_residue + 1, max_residue + 1), "UNK", dtype=str)

            # Fill matrices
            for _, row in ic_results.iterrows():
                i, j = int(row["Res1"]), int(row["Res2"])
                contact_matrix[i, j] = row["ICtotal"]
                contact_matrix[j, i] = row["ICtotal"]  # Symmetric
                state_matrix[i, j] = row["FstConserved"]
                state_matrix[j, i] = row["FstConserved"]

            # Create figure
            plt.figure(figsize=(15, 15))

            # Define colors for all possible states
            colors = {
                "MIN": "green",
                "NEU": "grey",
                "MAX": "red",
                "UNK": "white",
                "U": "white",  # Add unknown/undefined state
                "N": "grey",  # Add neutral state alternative
                "minimally": "green",  # Add legacy states
                "neutral": "grey",
                "highly": "red",
            }

            # Create custom colormap
            cmap = LinearSegmentedColormap.from_list(
                "frustration", ["white", "grey", "red", "green"]
            )

            # Plot heatmap
            sns.heatmap(
                contact_matrix,
                cmap=cmap,
                center=0,
                square=True,
                cbar_kws={"label": "Information Content"},
            )

            # Add state markers with fallback color
            for i in range(max_residue + 1):
                for j in range(max_residue + 1):
                    if state_matrix[i, j] != "UNK":
                        plt.plot(
                            j + 0.5,
                            i + 0.5,
                            "o",
                            color=colors.get(
                                state_matrix[i, j], "white"
                            ),  # Use get() with default
                            markersize=3,
                        )

            plt.title(f"Contact Map with Frustration States - {self.reference_pdb}")
            plt.xlabel("Residue Position")
            plt.ylabel("Residue Position")

            # Save plot
            output_file = self.plots_dir / "contact_map.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved contact map to {output_file}")

        except Exception as e:
            logger.error(f"Failed to generate contact maps: {str(e)}")
            logger.error(
                f"State matrix unique values: {np.unique(state_matrix)}"
            )  # Log unique states
            raise
