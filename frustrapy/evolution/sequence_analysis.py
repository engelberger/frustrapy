from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
from .data_classes import MSAData, EvolutionaryFrustrationData
from .single_residue import SingleResidueAnalyzer, SingleResidueResult

logger = logging.getLogger(__name__)


@dataclass
class SequenceAnalysisResult:
    """Results from sequence and frustration analysis"""

    position: int
    residue: str
    chain: str
    conservation: float
    frustration_scores: Dict[str, float]  # MIN, NEU, MAX proportions
    mutation_effects: Dict[str, float]  # AA -> effect score
    contacts: List[int]
    information_content: float


class SequenceAnalyzer:
    """Analyzes sequence conservation and frustration patterns"""

    def __init__(
        self, msa_data: MSAData, results_dir: Path, reference_pdb: Optional[str] = None
    ):
        """
        Initialize sequence analyzer.

        Args:
            msa_data: Multiple sequence alignment data
            results_dir: Output directory
            reference_pdb: Reference PDB identifier
        """
        self.msa_data = msa_data
        self.results_dir = Path(results_dir)
        self.reference_pdb = reference_pdb

    def analyze(self) -> List[SequenceAnalysisResult]:
        """Perform full sequence analysis"""
        try:
            # Calculate sequence conservation
            conservation = self._calculate_conservation()

            # Calculate frustration patterns
            frustration = self._analyze_frustration()

            # Analyze mutations if reference structure provided
            mutation_effects = {}
            if self.reference_pdb:
                mutation_effects = self._analyze_mutations()

            # Combine results
            results = []
            for pos in range(self.msa_data.length):
                results.append(
                    SequenceAnalysisResult(
                        position=pos + 1,
                        residue=self._get_reference_residue(pos),
                        chain=self._get_reference_chain(pos),
                        conservation=conservation[pos],
                        frustration_scores=frustration.get(pos + 1, {}),
                        mutation_effects=mutation_effects.get(pos + 1, {}),
                        contacts=self._get_contacts(pos + 1),
                        information_content=self._calculate_position_ic(pos),
                    )
                )

            # Generate visualizations
            self._generate_visualizations(results)

            return results

        except Exception as e:
            logger.error(f"Failed to analyze sequence: {str(e)}")
            raise

    def _calculate_conservation(self) -> np.ndarray:
        """Calculate position-specific conservation scores"""
        conservation = np.zeros(self.msa_data.length)

        # Get conservation matrix
        matrix = self.msa_data.conservation_matrix

        # Calculate Shannon entropy for each position
        for pos in range(matrix.shape[1]):
            freqs = matrix[:, pos]
            entropy = 0
            for freq in freqs[freqs > 0]:
                entropy -= freq * np.log2(freq)

            # Convert to conservation score
            max_entropy = np.log2(20)  # Maximum possible entropy
            conservation[pos] = 1 - (entropy / max_entropy)

        return conservation

    def _analyze_frustration(self) -> Dict[int, Dict[str, float]]:
        """Analyze frustration patterns"""
        frustration_file = self.results_dir / "FrustrationData.csv"
        if not frustration_file.exists():
            return {}

        df = pd.read_csv(frustration_file)
        frustration = {}

        for _, row in df.iterrows():
            pos = int(row["Position"])
            frustration[pos] = {
                "MIN": row["Pct_Min"],
                "NEU": row["Pct_Neu"],
                "MAX": row["Pct_Max"],
            }

        return frustration

    def _analyze_mutations(self) -> Dict[int, Dict[str, float]]:
        """Analyze mutation effects"""
        if not self.reference_pdb:
            return {}

        analyzer = SingleResidueAnalyzer(
            results_dir=self.results_dir,
            msa_data=self.msa_data,
            reference_pdb=self.reference_pdb,
        )

        mutation_effects = {}
        for pos in range(self.msa_data.length):
            result = analyzer.analyze_position(pos + 1)
            mutation_effects[pos + 1] = result.delta_scores

        return mutation_effects

    def _get_reference_residue(self, position: int) -> str:
        """Get reference residue at position"""
        if self.msa_data.reference_index is None:
            return "-"
        return self.msa_data.sequences[self.msa_data.reference_index][position]

    def _get_reference_chain(self, position: int) -> str:
        """Get reference chain at position"""
        if not self.reference_pdb:
            return "A"

        # Get chain from reference structure
        pdb_file = self.results_dir / f"{self.reference_pdb}.pdb"
        if not pdb_file.exists():
            return "A"

        with open(pdb_file) as f:
            for line in f:
                if line.startswith("ATOM"):
                    return line[21]

        return "A"

    def _get_contacts(self, position: int) -> List[int]:
        """Get contacting residues for a position"""
        contacts_file = self.results_dir / "ContactData.csv"
        if not contacts_file.exists():
            return []

        df = pd.read_csv(contacts_file)
        contacts = []

        # Get contacts for this position
        mask = (df["Res1"] == position) | (df["Res2"] == position)
        for _, row in df[mask].iterrows():
            if row["Res1"] == position:
                contacts.append(int(row["Res2"]))
            else:
                contacts.append(int(row["Res1"]))

        return sorted(contacts)

    def _calculate_position_ic(self, position: int) -> float:
        """Calculate information content for a position"""
        # Get amino acid frequencies
        freqs = self.msa_data.conservation_matrix[:, position]

        # Calculate Shannon entropy
        entropy = 0
        for freq in freqs[freqs > 0]:
            entropy -= freq * np.log2(freq)

        # Calculate information content
        max_entropy = np.log2(20)
        return max_entropy - entropy

    def _generate_visualizations(self, results: List[SequenceAnalysisResult]) -> None:
        """Generate analysis visualizations"""
        # Create figure
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.5])

        # Plot conservation
        ax1 = fig.add_subplot(gs[0])
        self._plot_conservation(ax1, results)

        # Plot frustration distribution
        ax2 = fig.add_subplot(gs[1])
        self._plot_frustration(ax2, results)

        # Plot information content
        ax3 = fig.add_subplot(gs[2])
        self._plot_information_content(ax3, results)

        # Plot mutation effects if available
        if any(r.mutation_effects for r in results):
            ax4 = fig.add_subplot(gs[3])
            self._plot_mutation_effects(ax4, results)

        plt.tight_layout()
        plt.savefig(
            self.results_dir / "SequenceAnalysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_conservation(
        self, ax: plt.Axes, results: List[SequenceAnalysisResult]
    ) -> None:
        """Plot sequence conservation"""
        positions = range(1, len(results) + 1)
        conservation = [r.conservation for r in results]

        ax.plot(positions, conservation, "b-")
        ax.set_ylabel("Conservation")
        ax.set_title("Sequence Conservation")

    def _plot_frustration(
        self, ax: plt.Axes, results: List[SequenceAnalysisResult]
    ) -> None:
        """Plot frustration distribution"""
        positions = range(1, len(results) + 1)
        min_vals = [r.frustration_scores.get("MIN", 0) for r in results]
        neu_vals = [r.frustration_scores.get("NEU", 0) for r in results]
        max_vals = [r.frustration_scores.get("MAX", 0) for r in results]

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

        ax.set_ylabel("Proportion")
        ax.set_title("Frustration Distribution")
        ax.legend()

    def _plot_information_content(
        self, ax: plt.Axes, results: List[SequenceAnalysisResult]
    ) -> None:
        """Plot information content"""
        positions = range(1, len(results) + 1)
        ic = [r.information_content for r in results]

        ax.bar(positions, ic, color="gray", alpha=0.5)
        ax.set_ylabel("Information Content (bits)")
        ax.set_title("Position-Specific Information Content")

    def _plot_mutation_effects(
        self, ax: plt.Axes, results: List[SequenceAnalysisResult]
    ) -> None:
        """Plot mutation effects heatmap"""
        # Create mutation matrix
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        matrix = np.zeros((len(amino_acids), len(results)))

        for i, result in enumerate(results):
            for j, aa in enumerate(amino_acids):
                matrix[j, i] = result.mutation_effects.get(aa, 0)

        # Plot heatmap
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            xticklabels=range(1, len(results) + 1),
            yticklabels=amino_acids,
        )
        ax.set_xlabel("Position")
        ax.set_ylabel("Mutation")
        ax.set_title("Mutation Effects (ΔFrustration)")
