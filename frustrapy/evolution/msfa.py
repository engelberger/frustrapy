from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from Bio import SeqIO
from .data_classes import MSAData, EvolutionaryFrustrationData
from .sequence_logo import SequenceLogoData

logger = logging.getLogger(__name__)


@dataclass
class MSFAResult:
    """Stores results from Multiple Sequence Frustration Analysis"""

    position: int
    residue_name: str
    residue_number: int
    chain_id: str
    conservation_score: float
    frustration_scores: Dict[str, float]
    information_content: float
    density_scores: Dict[str, float]
    contacts: List[int]


class MSFAnalyzer:
    """Handles Multiple Sequence Frustration Analysis"""

    def __init__(
        self,
        results_dir: Path,
        msa_data: MSAData,
        reference_pdb: Optional[str] = None,
        mode: str = "configurational",
    ):
        """
        Initialize MSFA analyzer.

        Args:
            results_dir: Results directory
            msa_data: Multiple sequence alignment data
            reference_pdb: Reference PDB identifier
            mode: Analysis mode (configurational or singleresidue)
        """
        self.results_dir = Path(results_dir)
        self.msa_data = msa_data
        self.reference_pdb = reference_pdb
        self.mode = mode

    def analyze(self) -> Dict[int, MSFAResult]:
        """Perform Multiple Sequence Frustration Analysis"""
        try:
            # Load frustration data
            frustration_data = self._load_frustration_data()

            # Calculate conservation scores
            conservation_scores = self._calculate_conservation()

            # Calculate density scores if in configurational mode
            density_scores = {}
            if self.mode == "configurational":
                density_scores = self._calculate_density_scores()

            # Combine results
            results = {}
            for pos in range(1, self.msa_data.length + 1):
                if pos in frustration_data:
                    frust_data = frustration_data[pos]
                    results[pos] = MSFAResult(
                        position=pos,
                        residue_name=frust_data.residue_name,
                        residue_number=frust_data.residue_number,
                        chain_id=frust_data.chain_id,
                        conservation_score=conservation_scores.get(pos, 0.0),
                        frustration_scores=frust_data.frustration_scores,
                        information_content=frust_data.information_content,
                        density_scores=density_scores.get(pos, {}),
                        contacts=frust_data.contacts,
                    )

            # Save results
            self._save_results(results)

            return results

        except Exception as e:
            logger.error(f"Failed to perform MSFA: {str(e)}")
            raise

    def _load_frustration_data(self) -> Dict[int, EvolutionaryFrustrationData]:
        """Load frustration data from results"""
        data_file = self.results_dir / "FrustrationData.csv"
        if not data_file.exists():
            raise FileNotFoundError(f"Frustration data not found: {data_file}")

        df = pd.read_csv(data_file)
        data = {}

        for _, row in df.iterrows():
            pos = int(row["Position"])
            data[pos] = EvolutionaryFrustrationData(
                position=pos,
                residue_name=row["AA_Ref"],
                residue_number=int(row["Num_Ref"]),
                chain_id=row["Chain"] if "Chain" in row else "A",
                conservation_score=(
                    row["Conservation"] if "Conservation" in row else 0.0
                ),
                frustration_scores={
                    "MIN": row["Pct_Min"],
                    "NEU": row["Pct_Neu"],
                    "MAX": row["Pct_Max"],
                },
                information_content=row["IC_Total"],
                contacts=[],  # Will be filled from contact data
            )

        return data

    def _calculate_conservation(self) -> Dict[int, float]:
        """Calculate position-specific conservation scores"""
        conservation = {}

        # Get conservation matrix
        matrix = self.msa_data.conservation_matrix

        # Calculate Shannon entropy for each position
        for pos in range(matrix.shape[1]):
            freqs = matrix[:, pos]
            entropy = 0
            for freq in freqs:
                if freq > 0:
                    entropy -= freq * np.log2(freq)

            # Convert entropy to conservation score
            max_entropy = np.log2(20)  # Maximum possible entropy (20 amino acids)
            conservation[pos + 1] = 1 - (entropy / max_entropy)

        return conservation

    def _calculate_density_scores(self) -> Dict[int, Dict[str, float]]:
        """Calculate frustration density scores"""
        density_file = self.results_dir / f"{self.reference_pdb}.pdb_density.csv"
        if not density_file.exists():
            return {}

        df = pd.read_csv(density_file)
        density_scores = {}

        for _, row in df.iterrows():
            pos = int(row["Position"])
            density_scores[pos] = {
                "total_density": row["TotalDensity"],
                "min_density": row["MinDensity"],
                "neu_density": row["NeuDensity"],
                "max_density": row["MaxDensity"],
            }

        return density_scores

    def _save_results(self, results: Dict[int, MSFAResult]) -> None:
        """Save analysis results"""
        # Convert results to DataFrame
        data = []
        for pos, result in results.items():
            row = {
                "Position": pos,
                "AA_Ref": result.residue_name,
                "Num_Ref": result.residue_number,
                "Chain": result.chain_id,
                "Conservation": result.conservation_score,
                "IC_Total": result.information_content,
                "Num_Contacts": len(result.contacts),
            }

            # Add frustration scores
            row.update({f"Frust_{k}": v for k, v in result.frustration_scores.items()})

            # Add density scores
            row.update({f"Density_{k}": v for k, v in result.density_scores.items()})

            data.append(row)

        df = pd.DataFrame(data)
        output_file = self.results_dir / "MSFA_Results.csv"
        df.to_csv(output_file, index=False)

        # Generate summary plots
        self._generate_summary_plots(results)

    def _generate_summary_plots(self, results: Dict[int, MSFAResult]) -> None:
        """Generate summary visualization plots"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

        # Plot conservation scores
        positions = list(results.keys())
        conservation = [r.conservation_score for r in results.values()]
        ax1.plot(positions, conservation, "b-")
        ax1.set_ylabel("Conservation Score")
        ax1.set_title("Sequence Conservation")

        # Plot frustration distribution
        frustration_data = pd.DataFrame(
            [
                {
                    "Position": pos,
                    "MIN": r.frustration_scores["MIN"],
                    "NEU": r.frustration_scores["NEU"],
                    "MAX": r.frustration_scores["MAX"],
                }
                for pos, r in results.items()
            ]
        )
        frustration_data.plot(
            x="Position", y=["MIN", "NEU", "MAX"], kind="bar", stacked=True, ax=ax2
        )
        ax2.set_title("Frustration Distribution")

        # Plot density scores if available
        if any(r.density_scores for r in results.values()):
            density_data = pd.DataFrame(
                [
                    {
                        "Position": pos,
                        "Total": r.density_scores.get("total_density", 0),
                        "Minimal": r.density_scores.get("min_density", 0),
                        "Neutral": r.density_scores.get("neu_density", 0),
                        "Maximal": r.density_scores.get("max_density", 0),
                    }
                    for pos, r in results.items()
                ]
            )
            density_data.plot(x="Position", y=["Total"], ax=ax3)
            ax3.set_title("Contact Density")

        plt.tight_layout()
        plt.savefig(self.results_dir / "MSFA_Summary.png", dpi=300, bbox_inches="tight")
        plt.close()
