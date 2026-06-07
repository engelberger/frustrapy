from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from Bio import SeqIO
from .data_classes import MSAData, EvolutionaryFrustrationData
from ..analysis.mutations import mutate_res_parallel
from ..analysis.frustration import calculate_frustration

logger = logging.getLogger(__name__)


@dataclass
class SingleResidueResult:
    """Stores results from single residue analysis"""

    position: int
    residue_name: str
    chain_id: str
    conservation_score: float
    mutation_scores: Dict[str, float]  # AA -> frustration score
    best_mutation: str
    worst_mutation: str
    native_score: float
    delta_scores: Dict[str, float]  # AA -> delta from native
    contacts: List[int]


class SingleResidueAnalyzer:
    """Handles single residue frustration analysis"""

    def __init__(
        self,
        results_dir: Path,
        msa_data: MSAData,
        reference_pdb: str,
        num_workers: int = 4,
    ):
        """
        Initialize single residue analyzer.

        Args:
            results_dir: Results directory
            msa_data: Multiple sequence alignment data
            reference_pdb: Reference PDB identifier
            num_workers: Number of parallel workers
        """
        self.results_dir = Path(results_dir)
        self.msa_data = msa_data
        self.reference_pdb = reference_pdb
        self.num_workers = num_workers

    def analyze_position(
        self, position: int, chain_id: str = "A"
    ) -> SingleResidueResult:
        """
        Analyze a single position in the protein.

        Args:
            position: Residue position to analyze
            chain_id: Chain identifier

        Returns:
            SingleResidueResult containing analysis results
        """
        try:
            # Get reference structure
            pdb_file = self.results_dir / f"{self.reference_pdb}.pdb"

            # Calculate conservation
            conservation = self._calculate_position_conservation(position)

            # Get native residue
            native_res = self._get_native_residue(position)

            # Perform mutations
            mutation_results = self._analyze_mutations(pdb_file, position, chain_id)

            # Calculate scores
            native_score = mutation_results[native_res]
            delta_scores = {
                aa: score - native_score for aa, score in mutation_results.items()
            }

            # Find best and worst mutations
            best_mutation = max(mutation_results.items(), key=lambda x: x[1])[0]
            worst_mutation = min(mutation_results.items(), key=lambda x: x[1])[0]

            # Get contacting residues
            contacts = self._get_contacts(position, chain_id)

            return SingleResidueResult(
                position=position,
                residue_name=native_res,
                chain_id=chain_id,
                conservation_score=conservation,
                mutation_scores=mutation_results,
                best_mutation=best_mutation,
                worst_mutation=worst_mutation,
                native_score=native_score,
                delta_scores=delta_scores,
                contacts=contacts,
            )

        except Exception as e:
            logger.error(f"Failed to analyze position {position}: {str(e)}")
            raise

    def _calculate_position_conservation(self, position: int) -> float:
        """Calculate conservation score for a position."""
        # Get amino acid frequencies
        aa_counts = {aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"}
        total = 0

        for seq in self.msa_data.sequences:
            aa = seq[position]
            if aa in aa_counts:
                aa_counts[aa] += 1
                total += 1

        if total == 0:
            return 0.0

        # Calculate Shannon entropy
        frequencies = [count / total for count in aa_counts.values() if count > 0]
        entropy = -sum(f * np.log2(f) for f in frequencies)

        # Convert to conservation score
        max_entropy = np.log2(20)  # Maximum possible entropy
        return 1 - (entropy / max_entropy)

    def _get_native_residue(self, position: int) -> str:
        """Get native residue at position from reference sequence."""
        if self.msa_data.reference_index is None:
            raise ValueError("No reference sequence specified")

        return self.msa_data.sequences[self.msa_data.reference_index][position]

    def _analyze_mutations(
        self, pdb_file: Path, position: int, chain_id: str
    ) -> Dict[str, float]:
        """Analyze all possible mutations at a position."""
        try:
            # Perform mutations in parallel
            results = {}
            amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

            for aa in amino_acids:
                # Calculate frustration for mutation
                pdb, _, _, single_res_data = calculate_frustration(
                    pdb_file=str(pdb_file),
                    mode="singleresidue",
                    residues={chain_id: [position]},
                    mutation=aa,
                    graphics=False,
                    visualization=False,
                )

                if single_res_data and chain_id in single_res_data:
                    results[aa] = single_res_data[chain_id][position].frustration_score

            return results

        except Exception as e:
            logger.error(f"Failed to analyze mutations: {str(e)}")
            raise

    def _get_contacts(self, position: int, chain_id: str) -> List[int]:
        """Get list of contacting residues."""
        contacts_file = self.results_dir / "ContactData.csv"
        if not contacts_file.exists():
            return []

        contacts = []
        df = pd.read_csv(contacts_file)

        # Find contacts where this residue is involved
        mask = ((df["Res1"] == position) & (df["Chain1"] == chain_id)) | (
            (df["Res2"] == position) & (df["Chain2"] == chain_id)
        )

        for _, row in df[mask].iterrows():
            if row["Res1"] == position:
                contacts.append(row["Res2"])
            else:
                contacts.append(row["Res1"])

        return sorted(contacts)

    def save_results(self, results: List[SingleResidueResult]) -> None:
        """Save analysis results."""
        # Convert results to DataFrame
        data = []
        for result in results:
            row = {
                "Position": result.position,
                "Residue": result.residue_name,
                "Chain": result.chain_id,
                "Conservation": result.conservation_score,
                "Native_Score": result.native_score,
                "Best_Mutation": result.best_mutation,
                "Worst_Mutation": result.worst_mutation,
                "Num_Contacts": len(result.contacts),
            }

            # Add mutation scores
            for aa, score in result.mutation_scores.items():
                row[f"Mut_{aa}"] = score
                row[f"Delta_{aa}"] = result.delta_scores[aa]

            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(self.results_dir / "SingleResidue_Results.csv", index=False)

        # Generate visualization
        self._generate_mutation_plot(results)

    def _generate_mutation_plot(self, results: List[SingleResidueResult]) -> None:
        """Generate mutation analysis visualization."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Plot conservation scores
        positions = [r.position for r in results]
        conservation = [r.conservation_score for r in results]
        ax1.plot(positions, conservation, "b-")
        ax1.set_ylabel("Conservation Score")
        ax1.set_title("Sequence Conservation")

        # Create mutation score matrix
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        mutation_matrix = np.zeros((len(results), len(amino_acids)))

        for i, result in enumerate(results):
            for j, aa in enumerate(amino_acids):
                mutation_matrix[i, j] = result.delta_scores.get(aa, 0)

        # Plot mutation heatmap
        sns.heatmap(
            mutation_matrix.T,
            ax=ax2,
            cmap="RdBu_r",
            center=0,
            xticklabels=positions,
            yticklabels=amino_acids,
        )
        ax2.set_xlabel("Position")
        ax2.set_ylabel("Mutation")
        ax2.set_title("Mutation Effects (ΔFrustration)")

        plt.tight_layout()
        plt.savefig(
            self.results_dir / "SingleResidue_Analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
