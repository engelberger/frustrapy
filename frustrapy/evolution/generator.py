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

    def generate_contact_maps(self, contacts: Dict[str, pd.DataFrame]) -> None:
        """
        Generate contact map visualizations.

        Args:
            contacts: Dictionary mapping structure IDs to contact DataFrames
        """
        try:
            # Convert DataFrame contacts to Contact objects
            contact_objects = []
            for struct_id, df in contacts.items():
                for _, row in df.iterrows():
                    contact = Contact(
                        residue1=int(row["Residue1"]),
                        residue2=int(row["Residue2"]),
                        frustration_index=float(row["FrustrationIndex"]),
                        frustration_state=str(row["FrustrationState"]),
                        structure=str(row["Structure"]),
                    )
                    contact_objects.append(contact)

            if not contact_objects:
                logger.warning("No contacts to plot")
                return

            # Get maximum position for plot dimensions
            positions = max(max(c.residue1, c.residue2) for c in contact_objects)

            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

            # Plot configurational contacts
            self._plot_contact_map(
                ax1, contact_objects, positions, "Configurational Frustration"
            )

            # Plot mutational contacts
            self._plot_contact_map(
                ax2, contact_objects, positions, "Mutational Frustration"
            )

            # Save figure
            plt.tight_layout()
            plt.savefig(
                self.results_dir / "ContactMaps.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        except Exception as e:
            logger.error(f"Failed to generate contact maps: {str(e)}")
            raise

    def _plot_contact_map(
        self, ax: plt.Axes, contacts: List[Contact], positions: int, title: str
    ) -> None:
        """Plot contact map on given axes."""
        try:
            # Create contact matrix
            matrix = np.zeros((positions, positions))

            # Fill matrices
            for contact in contacts:
                i, j = contact.residue1 - 1, contact.residue2 - 1
                matrix[i, j] = matrix[j, i] = contact.frustration_index

            # Plot matrix
            sns.heatmap(matrix, cmap="viridis", center=0, square=True, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Residue Position")
            ax.set_ylabel("Residue Position")

        except Exception as e:
            logger.error(f"Failed to plot contact map: {str(e)}")
            raise

    def _get_frustration_value(self, contact: ContactInformation) -> float:
        """Convert frustration state to numerical value for visualization."""
        if contact.conserved_state == "MIN":
            return 1.0
        elif contact.conserved_state == "MAX":
            return -1.0
        return 0.0
