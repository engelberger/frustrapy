from typing import Dict, List, Optional, Set
from pathlib import Path
import logging
from Bio import SeqIO
from Bio.Seq import Seq
from .data_classes import MSAData
import numpy as np

logger = logging.getLogger(__name__)


class MSAValidator:
    """Validates multiple sequence alignment data"""

    def __init__(self, msa_data: MSAData):
        self.msa_data = msa_data
        self.valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY-")

    def validate(self) -> Dict[str, List[str]]:
        """
        Perform comprehensive MSA validation.

        Returns:
            Dict containing validation errors and warnings
        """
        results = {"errors": [], "warnings": []}

        # Basic validation
        self._validate_basic_requirements(results)

        # Sequence validation
        self._validate_sequences(results)

        # Conservation validation
        self._validate_conservation(results)

        # Reference validation
        if self.msa_data.reference_index is not None:
            self._validate_reference(results)

        return results

    def _validate_basic_requirements(self, results: Dict[str, List[str]]) -> None:
        """Validate basic MSA requirements."""
        if not self.msa_data.sequences:
            results["errors"].append("No sequences found in MSA")
            return

        if not self.msa_data.identifiers:
            results["errors"].append("No sequence identifiers found")
            return

        if len(self.msa_data.sequences) != len(self.msa_data.identifiers):
            results["errors"].append(
                f"Mismatch between number of sequences ({len(self.msa_data.sequences)}) "
                f"and identifiers ({len(self.msa_data.identifiers)})"
            )

        # Check for duplicate identifiers
        duplicates = self._find_duplicates(self.msa_data.identifiers)
        if duplicates:
            results["errors"].append(
                f"Duplicate sequence identifiers found: {', '.join(duplicates)}"
            )

    def _validate_sequences(self, results: Dict[str, List[str]]) -> None:
        """Validate sequence content and alignment."""
        # Check sequence lengths
        lengths = set(len(seq) for seq in self.msa_data.sequences)
        if len(lengths) > 1:
            results["errors"].append(
                f"Sequences have different lengths: {sorted(lengths)}"
            )

        # Check for invalid characters
        for i, seq in enumerate(self.msa_data.sequences):
            invalid_chars = set(seq) - self.valid_amino_acids
            if invalid_chars:
                results["errors"].append(
                    f"Invalid characters in sequence {i+1}: {invalid_chars}"
                )

        # Check for all-gap columns
        self._check_gap_columns(results)

    def _validate_conservation(self, results: Dict[str, List[str]]) -> None:
        """Validate sequence conservation patterns."""
        matrix = self.msa_data.conservation_matrix

        # Check for highly conserved positions
        highly_conserved = []
        for pos in range(matrix.shape[1]):
            max_freq = matrix[:, pos].max()
            if max_freq > 0.9:  # 90% conservation threshold
                highly_conserved.append(pos + 1)

        if highly_conserved:
            results["warnings"].append(
                f"Highly conserved positions (>90%): {highly_conserved}"
            )

        # Check for low complexity regions
        self._check_low_complexity(matrix, results)

    def _validate_reference(self, results: Dict[str, List[str]]) -> None:
        """Validate reference sequence if provided."""
        ref_seq = self.msa_data.sequences[self.msa_data.reference_index]

        # Check for gaps in reference
        gap_positions = [i + 1 for i, aa in enumerate(ref_seq) if aa == "-"]
        if gap_positions:
            results["warnings"].append(
                f"Gaps found in reference sequence at positions: {gap_positions}"
            )

        # Check reference sequence composition
        aa_counts = {aa: ref_seq.count(aa) for aa in self.valid_amino_acids - {"-"}}
        rare_aas = {aa: count for aa, count in aa_counts.items() if count == 1}
        if rare_aas:
            results["warnings"].append(
                f"Rare amino acids in reference sequence: {rare_aas}"
            )

    def _check_gap_columns(self, results: Dict[str, List[str]]) -> None:
        """Check for problematic gap patterns."""
        num_seqs = len(self.msa_data.sequences)
        seq_length = len(self.msa_data.sequences[0])

        # Check each column
        for pos in range(seq_length):
            gap_count = sum(1 for seq in self.msa_data.sequences if seq[pos] == "-")
            gap_fraction = gap_count / num_seqs

            if gap_fraction == 1.0:
                results["errors"].append(f"All-gap column found at position {pos+1}")
            elif gap_fraction > 0.5:
                results["warnings"].append(
                    f"High gap frequency ({gap_fraction:.2%}) at position {pos+1}"
                )

    def _check_low_complexity(
        self, matrix: np.ndarray, results: Dict[str, List[str]]
    ) -> None:
        """Check for low complexity regions."""
        from scipy.stats import entropy

        window_size = 10
        threshold = 1.5  # Minimum entropy threshold

        for i in range(matrix.shape[1] - window_size + 1):
            window = matrix[:, i : i + window_size]
            avg_entropy = np.mean([entropy(col) for col in window.T])

            if avg_entropy < threshold:
                results["warnings"].append(
                    f"Low complexity region detected at positions {i+1}-{i+window_size}"
                )

    def _find_duplicates(self, items: List[str]) -> Set[str]:
        """Find duplicate items in a list."""
        seen = set()
        duplicates = set()

        for item in items:
            if item in seen:
                duplicates.add(item)
            seen.add(item)

        return duplicates
