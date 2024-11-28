from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from Bio import SeqIO
import logomaker

logger = logging.getLogger(__name__)


@dataclass
class LogoData:
    """Data for sequence logo visualization"""

    position: int
    aa_ref: str
    num_ref: str
    prot_ref: str
    pct_min: float
    pct_neu: float
    pct_max: float
    count_min: int
    count_neu: int
    count_max: int
    ic_min: float
    ic_neu: float
    ic_max: float
    ic_total: float
    frust_state: str


class LogoGenerator:
    """Generates sequence logo visualizations"""

    def __init__(
        self, msa_file: Path, results_dir: Path, reference_pdb: Optional[str] = None
    ):
        """Initialize logo generator"""
        self.msa_file = Path(msa_file)
        self.results_dir = Path(results_dir)
        self.reference_pdb = reference_pdb
        self.logo_data: List[LogoData] = []

    def generate_logo(self, contacts: Dict[str, pd.DataFrame]) -> None:
        """
        Generate sequence logo with frustration information.

        Args:
            contacts: Dictionary mapping structure IDs to contact DataFrames
        """
        try:
            # Process data
            self.logo_data = self._process_data(contacts)

            # Create visualization
            self._create_visualization()

            # Save data
            self._save_data()

        except Exception as e:
            logger.error(f"Failed to generate logo: {str(e)}")
            raise

    def _process_data(self, contacts: Dict[str, pd.DataFrame]) -> List[LogoData]:
        """Process contact data into logo format"""
        try:
            # Get sequence length from MSA
            with open(self.msa_file) as f:
                for record in SeqIO.parse(f, "fasta"):
                    seq_length = len(record.seq)
                    break

            # Process each position
            logo_data = []
            for pos in range(1, seq_length + 1):
                # Calculate frustration statistics
                frust_stats = self._calculate_frustration_stats(pos, contacts)

                # Create logo data
                data = LogoData(
                    position=pos,
                    aa_ref=self._get_reference_aa(pos),
                    num_ref=str(pos),
                    prot_ref=self.reference_pdb or "",
                    **frust_stats,
                )
                logo_data.append(data)

            return logo_data

        except Exception as e:
            logger.error(f"Failed to process data: {str(e)}")
            raise

    def _calculate_frustration_stats(
        self, position: int, contacts: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Calculate frustration statistics for a position"""
        try:
            # Collect contacts involving this position
            pos_contacts = []
            for struct_id, df in contacts.items():
                # Filter contacts involving this position
                pos_df = df[(df["Residue1"] == position) | (df["Residue2"] == position)]
                pos_contacts.extend(pos_df.to_dict("records"))

            if not pos_contacts:
                return self._get_default_stats()

            # Count frustration states
            count_min = sum(1 for c in pos_contacts if c["FrustrationState"] == "MIN")
            count_neu = sum(1 for c in pos_contacts if c["FrustrationState"] == "NEU")
            count_max = sum(1 for c in pos_contacts if c["FrustrationState"] == "MAX")
            total = len(pos_contacts)

            # Calculate percentages
            pct_min = count_min / total if total > 0 else 0
            pct_neu = count_neu / total if total > 0 else 0
            pct_max = count_max / total if total > 0 else 0

            # Calculate information content
            ic_min = self._calculate_ic(pct_min)
            ic_neu = self._calculate_ic(pct_neu)
            ic_max = self._calculate_ic(pct_max)
            ic_total = ic_min + ic_neu + ic_max

            # Determine dominant state
            if total > 0:
                max_count = max(count_min, count_neu, count_max)
                if max_count == count_min:
                    frust_state = "MIN"
                elif max_count == count_max:
                    frust_state = "MAX"
                else:
                    frust_state = "NEU"
            else:
                frust_state = "UNK"

            return {
                "pct_min": pct_min,
                "pct_neu": pct_neu,
                "pct_max": pct_max,
                "count_min": count_min,
                "count_neu": count_neu,
                "count_max": count_max,
                "ic_min": ic_min,
                "ic_neu": ic_neu,
                "ic_max": ic_max,
                "ic_total": ic_total,
                "frust_state": frust_state,
            }

        except Exception as e:
            logger.error(
                f"Failed to calculate frustration stats for position {position}: {str(e)}"
            )
            return self._get_default_stats()

    def _get_reference_aa(self, position: int) -> str:
        """Get reference amino acid for a position"""
        try:
            if not self.reference_pdb:
                return ""

            with open(self.msa_file) as f:
                for record in SeqIO.parse(f, "fasta"):
                    if record.id == self.reference_pdb:
                        return record.seq[position - 1]
            return ""

        except Exception as e:
            logger.error(
                f"Failed to get reference amino acid for position {position}: {str(e)}"
            )
            return ""

    def _get_default_stats(self) -> Dict:
        """Get default statistics for a position"""
        return {
            "pct_min": 0.0,
            "pct_neu": 0.0,
            "pct_max": 0.0,
            "count_min": 0,
            "count_neu": 0,
            "count_max": 0,
            "ic_min": 0.0,
            "ic_neu": 0.0,
            "ic_max": 0.0,
            "ic_total": 0.0,
            "frust_state": "UNK",
        }

    def _calculate_ic(self, pct: float) -> float:
        """Calculate information content for a percentage"""
        try:
            if pct == 0:
                return 0
            return -np.log2(pct)

        except Exception as e:
            logger.error(
                f"Failed to calculate information content for percentage {pct}: {str(e)}"
            )
            return 0

    def _create_visualization(self) -> None:
        """Create visualization from logo data"""
        try:
            # Create sequence logo
            seq_logo = self._create_sequence_logo()

            # Create frustration bars
            frust_bars = self._create_frustration_bars()

            # Combine plots
            self._combine_plots(seq_logo, frust_bars)

        except Exception as e:
            logger.error(f"Failed to create visualization: {str(e)}")
            raise

    def _save_data(self) -> None:
        """Save logo data to file"""
        try:
            # Save logo data to CSV
            pd.DataFrame(self.logo_data).to_csv(
                self.results_dir / "LogoData.csv", index=False
            )

        except Exception as e:
            logger.error(f"Failed to save logo data: {str(e)}")
            raise

    def _create_sequence_logo(self) -> plt.Figure:
        """Create sequence logo using logomaker."""
        try:
            # Read MSA and create counts matrix
            seqs = []
            with open(self.msa_file) as f:
                for record in SeqIO.parse(f, "fasta"):
                    seqs.append(str(record.seq))

            counts_mat = logomaker.alignment_to_matrix(seqs)

            # Create logo
            fig, ax = plt.subplots(figsize=(20, 3))
            logo = logomaker.Logo(
                counts_mat, ax=ax, color_scheme="chemistry", stack_order="small_on_top"
            )

            # Customize logo
            ax.set_xlabel("Position")
            ax.set_ylabel("Information (bits)")

            return fig

        except Exception as e:
            logger.error(f"Failed to create sequence logo: {str(e)}")
            raise

    def _create_frustration_bars(self) -> plt.Figure:
        """Create stacked bar plot of frustration data."""
        try:
            positions = [d.position for d in self.logo_data]
            min_vals = [d.pct_min for d in self.logo_data]
            neu_vals = [d.pct_neu for d in self.logo_data]
            max_vals = [d.pct_max for d in self.logo_data]

            fig, ax = plt.subplots(figsize=(20, 3))

            # Create stacked bars
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
            ax.legend()

            return fig

        except Exception as e:
            logger.error(f"Failed to create frustration bars: {str(e)}")
            raise

    def _combine_plots(self, seq_logo: plt.Figure, frust_bars: plt.Figure) -> None:
        """Combine sequence logo and frustration bars into final visualization."""
        try:
            # Create combined figure
            fig = plt.figure(figsize=(20, 7))

            # Add sequence logo
            ax1 = plt.subplot(211)
            seq_logo_ax = seq_logo.get_axes()[0]
            ax1.plot(seq_logo_ax.lines[0].get_xdata(), seq_logo_ax.lines[0].get_ydata())

            # Add frustration bars
            ax2 = plt.subplot(212)
            frust_bars_ax = frust_bars.get_axes()[0]
            for collection in frust_bars_ax.collections:
                ax2.add_collection(collection.copy())

            # Save combined figure
            plt.savefig(
                self.results_dir / "HistogramFrustration.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close("all")

        except Exception as e:
            logger.error(f"Failed to combine plots: {str(e)}")
            raise
