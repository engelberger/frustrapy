from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logomaker
import logging
from pathlib import Path
from .data_classes import MSAData, PositionInformation

logger = logging.getLogger(__name__)


@dataclass
class PositionData:
    """Data for a single position in the sequence logo"""

    position: int
    amino_acids: Dict[str, float]
    height: float


class SequenceLogoGenerator:
    """Handles sequence logo generation and information content calculation"""

    def __init__(
        self, msa_data: MSAData, results_dir: Path, reference_pdb: Optional[str] = None
    ):
        """Initialize sequence logo generator."""
        self.msa_data = msa_data
        self.results_dir = Path(results_dir)
        self.reference_pdb = reference_pdb
        self.aa_colors = self._get_aa_colors()

    def generate_logo(self, ic_results: List[PositionInformation]) -> None:
        """
        Generate sequence logo with frustration information.

        Args:
            ic_results: List of position-specific information content results
        """
        try:
            logger.info("Generating sequence logo")

            # Create logo data
            logo_data = []
            for result in ic_results:
                logo_data.append(
                    {
                        "position": result.position,
                        "residue": result.residue,
                        "conservation": result.conservation,
                        "ic_total": result.ic_total,
                        "frust_state": result.frust_state,
                    }
                )

            # Convert to DataFrame for easier plotting
            df = pd.DataFrame(logo_data)

            # Create logo plot
            plt.figure(figsize=(20, 10))

            # Plot sequence conservation
            plt.subplot(2, 1, 1)
            plt.bar(df["position"], df["conservation"], color="blue", alpha=0.5)
            plt.title("Sequence Conservation")
            plt.xlabel("Position")
            plt.ylabel("Conservation Score")

            # Plot frustration information
            plt.subplot(2, 1, 2)
            colors = {"MIN": "green", "NEU": "grey", "MAX": "red", "UNK": "white"}
            bar_colors = [colors.get(state, "white") for state in df["frust_state"]]
            plt.bar(df["position"], df["ic_total"], color=bar_colors)
            plt.title("Frustration Information Content")
            plt.xlabel("Position")
            plt.ylabel("Information Content")

            # Add legend
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor=color, label=state)
                for state, color in colors.items()
                if state != "UNK"
            ]
            plt.legend(handles=legend_elements)

            # Save plot
            plt.tight_layout()
            output_file = self.results_dir / "plots" / "sequence_logo.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved sequence logo to {output_file}")

        except Exception as e:
            logger.error(f"Failed to generate sequence logo: {str(e)}")
            raise

    def _calculate_logo_data(self) -> List[PositionData]:
        """Calculate sequence logo data."""
        try:
            # Get alignment length
            alignment_length = min(len(seq) for seq in self.msa_data.sequences)
            logger.debug(f"Using alignment length: {alignment_length}")

            # Calculate background frequencies
            bg_freqs = self._calculate_background_frequencies()

            # Calculate data for each position
            logo_data = []
            for pos in range(alignment_length):
                # Get amino acid frequencies
                aa_freqs = self._get_position_frequencies(pos)
                if aa_freqs:
                    # Calculate information content
                    height = self._calculate_position_ic(aa_freqs, bg_freqs)
                    # Create position data
                    pos_data = PositionData(
                        position=pos + 1, amino_acids=aa_freqs, height=height
                    )
                    logo_data.append(pos_data)

            if not logo_data:
                raise ValueError("No valid logo data generated")

            return logo_data

        except Exception as e:
            logger.error(f"Failed to calculate logo data: {str(e)}")
            raise

    def _plot_sequence_logo(self, ax: plt.Axes, logo_data: List[PositionData]) -> None:
        """Plot sequence logo using logomaker"""
        try:
            # Convert data to logomaker format
            matrix_data = []
            for pos_data in logo_data:
                matrix_data.append(pos_data.amino_acids)

            df = pd.DataFrame(matrix_data)
            df = df.fillna(0)  # Fill NaN values with 0

            # Create and customize logo
            logo = logomaker.Logo(
                df,
                ax=ax,
                color_scheme=self.aa_colors,
                width=0.9,
                vpad=0.1,
                fade_probabilities=True,
            )

            # Customize appearance
            ax.set_xlabel("Position")
            ax.set_ylabel("Information Content (bits)")
            ax.set_title("Sequence Conservation Logo")

        except Exception as e:
            logger.error(f"Failed to plot sequence logo: {str(e)}")
            raise

    def _plot_information_content(
        self, ax: plt.Axes, logo_data: List[PositionData]
    ) -> None:
        """Plot position-specific information content"""
        try:
            positions = [d.position for d in logo_data]
            heights = [d.height for d in logo_data]

            ax.bar(positions, heights, color="gray", alpha=0.5)
            ax.set_xlabel("Position")
            ax.set_ylabel("Information Content (bits)")
            ax.set_title("Position-Specific Information Content")

        except Exception as e:
            logger.error(f"Failed to plot information content: {str(e)}")
            raise

    def _calculate_background_frequencies(self) -> Dict[str, float]:
        """Calculate background amino acid frequencies"""
        aa_counts = {aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"}
        total = 0

        for seq in self.msa_data.sequences:
            for aa in seq:
                if aa in aa_counts:
                    aa_counts[aa] += 1
                    total += 1

        return {aa: count / total for aa, count in aa_counts.items()}

    def _get_position_frequencies(self, position: int) -> Dict[str, float]:
        """
        Calculate amino acid frequencies at a given position.

        Args:
            position: Position in sequence (0-based)

        Returns:
            Dictionary mapping amino acids to their frequencies
        """
        try:
            # Validate position
            if position < 0:
                raise ValueError(f"Position cannot be negative: {position}")

            # Get all sequences
            sequences = self.msa_data.sequences
            if not sequences:
                raise ValueError("No sequences found in MSA data")

            # Check if position is valid for all sequences
            max_length = min(len(seq) for seq in sequences)
            if position >= max_length:
                logger.warning(
                    f"Position {position} is out of range for some sequences"
                )
                return {}

            # Count amino acids at position
            aa_counts = {}
            total_count = 0
            for seq in sequences:
                try:
                    aa = seq[position]
                    if aa != "-":  # Skip gaps
                        aa_counts[aa] = aa_counts.get(aa, 0) + 1
                        total_count += 1
                except IndexError:
                    logger.warning(f"Sequence too short for position {position}")
                    continue

            # Calculate frequencies
            if total_count > 0:
                frequencies = {
                    aa: count / total_count for aa, count in aa_counts.items()
                }
            else:
                frequencies = {}

            return frequencies

        except Exception as e:
            logger.error(
                f"Error calculating frequencies for position {position}: {str(e)}"
            )
            return {}

    def _calculate_position_ic(
        self, pos_freqs: Dict[str, float], bg_freqs: Dict[str, float]
    ) -> float:
        """Calculate position-specific information content"""
        ic = 0
        for aa, freq in pos_freqs.items():
            if freq > 0:
                ic += freq * np.log2(freq / bg_freqs[aa])
        return ic

    def _get_aa_colors(self) -> Dict[str, str]:
        """Get amino acid coloring scheme"""
        return {
            "A": "#FF9966",
            "C": "#009999",
            "D": "#FF0000",
            "E": "#CC0033",
            "F": "#00FF00",
            "G": "#FF9999",
            "H": "#0066CC",
            "I": "#66CC00",
            "K": "#6600CC",
            "L": "#33CC00",
            "M": "#00CC00",
            "N": "#CC0066",
            "P": "#FFCC00",
            "Q": "#FF00CC",
            "R": "#0000FF",
            "S": "#FF3366",
            "T": "#FF6699",
            "V": "#99CC00",
            "W": "#00FFCC",
            "Y": "#33CCCC",
        }

    def _save_logo_data(
        self,
        logo_data: List[PositionData],
        frustration_data: Dict[str, pd.DataFrame],
    ) -> None:
        """
        Save logo data to CSV.

        Args:
            logo_data: List of position-specific logo data
            frustration_data: Dictionary mapping structure IDs to frustration DataFrames
        """
        try:
            data = []
            for pos_data in logo_data:
                row = {
                    "Position": pos_data.position,
                    "Information_Content": pos_data.height,
                }

                # Add amino acid frequencies
                for aa, freq in pos_data.amino_acids.items():
                    row[f"Freq_{aa}"] = freq

                # Add frustration scores if available
                for struct_id, frust_df in frustration_data.items():
                    # Use integer position for comparison
                    position = int(pos_data.position)
                    # Filter frustration data for this position
                    pos_frustration = frust_df[frust_df["Res1"] == position]

                    if not pos_frustration.empty:
                        row.update(
                            {
                                f"Frust_Index_{struct_id}": float(
                                    pos_frustration["FrstIndex"].iloc[0]
                                ),
                                f"Frust_State_{struct_id}": str(
                                    pos_frustration["FrstState"].iloc[0]
                                ),
                            }
                        )
                    else:
                        # Add NaN values if no frustration data for this position
                        row.update(
                            {
                                f"Frust_Index_{struct_id}": np.nan,
                                f"Frust_State_{struct_id}": "Unknown",
                            }
                        )

                data.append(row)

            # Save to CSV
            df = pd.DataFrame(data)
            df.to_csv(self.results_dir / "SequenceLogoData.csv", index=False)
            logger.debug(
                f"Saved logo data to {self.results_dir / 'SequenceLogoData.csv'}"
            )

        except Exception as e:
            logger.error(f"Failed to save logo data: {str(e)}")
            raise

    def _plot_frustration_bars(
        self, ax: plt.Axes, frustration_data: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Plot frustration bars for each position.

        Args:
            ax: Matplotlib axes to plot on
            frustration_data: Dictionary mapping structure IDs to frustration DataFrames
        """
        try:
            # Calculate average frustration index per position
            position_frustration = {}
            for struct_id, df in frustration_data.items():
                for _, row in df.iterrows():
                    pos = int(row["Res1"])
                    frust_index = float(row["FrstIndex"])

                    if pos not in position_frustration:
                        position_frustration[pos] = []
                    position_frustration[pos].append(frust_index)

            # Calculate averages and standard deviations
            positions = sorted(position_frustration.keys())
            averages = [np.mean(position_frustration[pos]) for pos in positions]
            stds = [np.std(position_frustration[pos]) for pos in positions]

            # Plot bars with error bars
            colors = ["red" if avg < 0 else "blue" for avg in averages]
            ax.bar(positions, averages, yerr=stds, color=colors, alpha=0.6)

            # Add horizontal line at y=0
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

            # Customize appearance
            ax.set_xlabel("Position")
            ax.set_ylabel("Average Frustration Index")
            ax.set_title("Position-Specific Frustration")

            # Add legend
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="red", alpha=0.6, label="Frustrated"),
                Patch(facecolor="blue", alpha=0.6, label="Minimally Frustrated"),
            ]
            ax.legend(handles=legend_elements, loc="upper right")

        except Exception as e:
            logger.error(f"Failed to plot frustration bars: {str(e)}")
            raise
