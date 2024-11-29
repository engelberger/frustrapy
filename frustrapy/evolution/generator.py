from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from Bio import SeqIO
import seaborn as sns
from .logo import LogoData
from .contacts import ContactInformation
from matplotlib.colors import LinearSegmentedColormap
from .data_classes import PositionInformation

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
    """Generates frustration histograms and visualizations"""

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

    def generate_visualization(self, logo_data: List[LogoData]) -> None:
        """
        Generate combined visualization of sequence logo and frustration data.

        Args:
            logo_data: List of logo data objects
        """
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 10))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5])

            # Generate sequence logo
            ax_logo = fig.add_subplot(gs[0])
            self._plot_sequence_logo(ax_logo)

            # Generate frustration histogram
            ax_hist = fig.add_subplot(gs[1])
            self._plot_frustration_histogram(ax_hist, logo_data)

            # Adjust layout and save
            plt.tight_layout()
            output_file = self.results_dir / "HistogramFrustration.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            # Save data to CSV
            self._save_data(logo_data)

            logger.info("Successfully generated visualization")

        except Exception as e:
            logger.error(f"Failed to generate visualization: {str(e)}")
            raise

    def _plot_sequence_logo(self, ax: plt.Axes) -> None:
        """Plot sequence logo on given axes."""
        try:
            # Read sequences
            sequences = []
            with open(self.msa_file) as f:
                for record in SeqIO.parse(f, "fasta"):
                    sequences.append(str(record.seq))

            # Calculate position-specific frequencies
            alphabet = list("ACDEFGHIKLMNPQRSTVWY")
            seq_length = len(sequences[0])
            freq_matrix = np.zeros((len(alphabet), seq_length))

            for pos in range(seq_length):
                counts = {aa: 0 for aa in alphabet}
                total = 0
                for seq in sequences:
                    if seq[pos] in counts:
                        counts[seq[pos]] += 1
                        total += 1
                if total > 0:
                    for i, aa in enumerate(alphabet):
                        freq_matrix[i, pos] = counts[aa] / total

            # Plot logo
            im = ax.imshow(freq_matrix, aspect="auto", cmap="viridis")
            ax.set_yticks(range(len(alphabet)))
            ax.set_yticklabels(alphabet)
            ax.set_xlabel("Position")
            ax.set_ylabel("Amino Acid")
            plt.colorbar(im, ax=ax, label="Frequency")

        except Exception as e:
            logger.error(f"Failed to plot sequence logo: {str(e)}")
            raise

    def _plot_frustration_histogram(
        self, ax: plt.Axes, logo_data: List[LogoData]
    ) -> None:
        """Plot frustration histogram on given axes."""
        try:
            positions = range(1, len(logo_data) + 1)

            # Extract data
            min_vals = [d.pct_min for d in logo_data]
            neu_vals = [d.pct_neu for d in logo_data]
            max_vals = [d.pct_max for d in logo_data]

            # Create stacked bar plot
            ax.bar(positions, min_vals, color="green", label="Minimally Frustrated")
            ax.bar(
                positions,
                neu_vals,
                bottom=min_vals,
                color="gray",
                label="Neutrally Frustrated",
            )
            ax.bar(
                positions,
                max_vals,
                bottom=[i + j for i, j in zip(min_vals, neu_vals)],
                color="red",
                label="Highly Frustrated",
            )

            # Customize plot
            ax.set_xlabel("Position")
            ax.set_ylabel("Fraction")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.set_xlim(0, len(logo_data) + 1)
            ax.set_ylim(0, 1)

            # Add reference sequence if available
            if self.reference_pdb:
                ref_seq = [d.aa_ref for d in logo_data]
                ax2 = ax.twiny()
                ax2.set_xlim(ax.get_xlim())
                ax2.set_xticks(positions)
                ax2.set_xticklabels(ref_seq, rotation=90)

        except Exception as e:
            logger.error(f"Failed to plot frustration histogram: {str(e)}")
            raise

    def _save_data(self, logo_data: List[LogoData]) -> None:
        """Save data to CSV file."""
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(
                [
                    {
                        "Position": d.position,
                        "AA_Ref": d.aa_ref,
                        "Num_Ref": d.num_ref,
                        "Prot_Ref": d.prot_ref,
                        "Pct_Min": d.pct_min,
                        "Pct_Neu": d.pct_neu,
                        "Pct_Max": d.pct_max,
                        "Count_Min": d.count_min,
                        "Count_Neu": d.count_neu,
                        "Count_Max": d.count_max,
                        "IC_Min": d.ic_min,
                        "IC_Neu": d.ic_neu,
                        "IC_Max": d.ic_max,
                        "IC_Total": d.ic_total,
                        "Frust_State": d.frust_state,
                    }
                    for d in logo_data
                ]
            )

            # Save to CSV
            output_file = self.results_dir / "FrustrationData.csv"
            df.to_csv(output_file, index=False)
            logger.debug(f"Saved data to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save data: {str(e)}")
            raise

    def generate_contact_maps(
        self, contacts: Dict[str, pd.DataFrame], ic_results: List[PositionInformation]
    ) -> None:
        """
        Generate contact map visualizations.

        Args:
            contacts: Dictionary of contact data by structure
            ic_results: List of position-specific information content results
        """
        try:
            logger.info("Generating contact maps")

            # Create figure
            plt.figure(figsize=(15, 15))

            # Create contact matrix
            n_pos = max(r.position for r in ic_results)
            contact_matrix = np.zeros((n_pos, n_pos))
            state_matrix = np.full((n_pos, n_pos), "UNK", dtype=str)

            # Fill matrices
            for struct_id, struct_data in contacts.items():
                for _, row in struct_data.iterrows():
                    i = int(row["Residue1"]) - 1
                    j = int(row["Residue2"]) - 1
                    contact_matrix[i, j] = row["IC_Total"]
                    state_matrix[i, j] = row["FrustState"]

            # Plot contact map
            colors = {
                "MIN": "green",
                "NEU": "grey",
                "MAX": "red",
                "UNK": "white",
                "minimally": "green",
                "neutral": "grey",
                "highly": "red",
            }

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

            # Add state markers
            for i in range(n_pos):
                for j in range(n_pos):
                    if state_matrix[i, j] != "UNK":
                        plt.plot(
                            j + 0.5,
                            i + 0.5,
                            "o",
                            color=colors.get(state_matrix[i, j], "white"),
                            markersize=3,
                        )

            plt.title("Contact Map with Frustration States")
            plt.xlabel("Residue Position")
            plt.ylabel("Residue Position")

            # Save plot
            output_file = self.results_dir / "plots" / "contact_map.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved contact map to {output_file}")

        except Exception as e:
            logger.error(f"Failed to generate contact maps: {str(e)}")
            raise

    def _get_frustration_value(self, contact: ContactInformation) -> float:
        """Convert frustration state to numerical value for visualization."""
        if contact.conserved_state == "MIN":
            return 1.0
        elif contact.conserved_state == "MAX":
            return -1.0
        return 0.0
