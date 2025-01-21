from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from ..core import Pdb
from .data_classes import PositionInformation

logger = logging.getLogger(__name__)


@dataclass
class ContactInformation:
    """Stores contact information for a set of structures"""

    residue1: int
    residue2: int
    total_contacts: int
    frequency: float
    p_neutral: float
    p_minimal: float
    p_maximal: float
    h_neutral: float
    h_minimal: float
    h_maximal: float
    h_total: float
    ic_neutral: float
    ic_minimal: float
    ic_maximal: float
    ic_total: float
    conserved_state: str


class ContactAnalyzer:
    """Analyzes contact information from frustration calculations"""

    def __init__(
        self,
        results_dir: Path,
        mode: str = "configurational",
        frustration_dir: Optional[Path] = None,
    ):
        """
        Initialize contact analyzer.

        Args:
            results_dir: Directory containing frustration results
            mode: Analysis mode (configurational or mutational)
            frustration_dir: Directory containing frustration results
        """
        self.results_dir = Path(results_dir)
        self.mode = mode
        self.frustration_dir = frustration_dir
        self.cutoff_min = 0.78
        self.cutoff_max = -1.0

    def analyze_contacts(
        self, alignment_length: int, ic_results: List[PositionInformation]
    ) -> Dict:
        """
        Analyze contacts across all structures.

        Args:
            alignment_length: Length of multiple sequence alignment
            ic_results: List of position-specific information content results

        Returns:
            Dictionary of PDB IDs and their corresponding frustration data
        """
        try:
            # Look for frustration results in frustration_dir if provided
            if self.frustration_dir and self.frustration_dir.exists():
                results_path = self.frustration_dir
                logger.debug(f"Using frustration directory: {results_path}")
            else:
                results_path = self.results_dir
                logger.debug(f"Using results directory: {results_path}")

            # Find all .done directories
            done_dirs = list(results_path.glob("*.done"))
            logger.debug(f"Found {len(done_dirs)} .done directories")

            if not done_dirs:
                logger.error(f"No .done directories found in {results_path}")
                logger.debug(f"Directory contents: {list(results_path.glob('*'))}")
                raise FileNotFoundError("No frustration results found")

            # Process each structure with IC results
            contacts = {}
            for done_dir in done_dirs:
                pdb_id = done_dir.name.replace(".done", "")
                frust_data = self._read_frustration_data(done_dir, pdb_id)
                if frust_data is not None:
                    # Enrich frustration data with IC results
                    frust_data = self._enrich_with_ic(frust_data, ic_results)
                    contacts[pdb_id] = frust_data
                    logger.debug(f"Processed contacts for {pdb_id}")

            if not contacts:
                raise FileNotFoundError("No valid contact data found in any structure")

            return contacts

        except Exception as e:
            logger.error(f"Failed to analyze contacts: {str(e)}")
            raise

    def _process_structure_contacts(
        self, pdb_dir: Path, contact_matrix: np.ndarray
    ) -> None:
        """Process contacts for a single structure."""
        try:
            # Read frustration data
            frust_file = pdb_dir / "FrustrationData" / f"{pdb_dir.stem}.pdb_{self.mode}"
            if not frust_file.exists():
                logger.warning(f"Frustration file not found: {frust_file}")
                return

            # Read equivalences
            equiv_file = pdb_dir / f"Equival_{pdb_dir.stem}.txt"
            equivalences = self._read_equivalences(equiv_file)

            # Process contacts
            frust_data = pd.read_csv(frust_file, sep="\s+")
            for _, row in frust_data.iterrows():
                pos1 = equivalences.get(row["Res1"])
                pos2 = equivalences.get(row["Res2"])
                if pos1 and pos2:
                    contact_matrix[pos1][pos2] = row["FrstIndex"]

        except Exception as e:
            logger.error(f"Failed to process contacts for {pdb_dir}: {str(e)}")
            raise

    def _calculate_information_content(
        self, contact_matrix: np.ndarray
    ) -> List[ContactInformation]:
        """Calculate information content from contact matrix."""
        contacts = []
        n = len(contact_matrix)

        for i in range(n):
            for j in range(i + 1, n):
                values = contact_matrix[i][j]
                if values.size > 0 and not np.all(values == -100):
                    contact_info = self._calculate_single_contact(values, i, j)
                    if contact_info:
                        contacts.append(contact_info)

        return contacts

    def _calculate_single_contact(
        self, values: np.ndarray, i: int, j: int
    ) -> Optional[ContactInformation]:
        """Calculate information content for a single contact."""
        try:
            total = np.sum(values != -100)
            if total <= 1:
                return None

            # Calculate frequencies
            minimal = np.sum(values >= self.cutoff_min)
            maximal = np.sum(values <= self.cutoff_max)
            neutral = total - minimal - maximal

            # Calculate probabilities
            p_min = minimal / total
            p_max = maximal / total
            p_neu = neutral / total

            # Calculate Shannon entropy
            h_min = self._calculate_entropy(p_min)
            h_max = self._calculate_entropy(p_max)
            h_neu = self._calculate_entropy(p_neu)
            h_total = -(h_min + h_max + h_neu)

            # Calculate information content
            ic_total = self._calculate_total_ic(h_total, total)
            ic_min = ic_total * p_min
            ic_max = ic_total * p_max
            ic_neu = ic_total * p_neu

            # Determine conserved state
            conserved = "NEU"
            max_count = max(minimal, maximal, neutral)
            if max_count == minimal:
                conserved = "MIN"
            elif max_count == maximal:
                conserved = "MAX"

            return ContactInformation(
                residue1=i,
                residue2=j,
                total_contacts=total,
                frequency=total / len(values),
                p_neutral=p_neu,
                p_minimal=p_min,
                p_maximal=p_max,
                h_neutral=h_neu,
                h_minimal=h_min,
                h_maximal=h_max,
                h_total=h_total,
                ic_neutral=ic_neu,
                ic_minimal=ic_min,
                ic_maximal=ic_max,
                ic_total=ic_total,
                conserved_state=conserved,
            )

        except Exception as e:
            logger.error(
                f"Failed to calculate contact information for {i}-{j}: {str(e)}"
            )
            return None

    def _calculate_entropy(self, p: float) -> float:
        """Calculate Shannon entropy term."""
        if p <= 0:
            return 0
        return -p * np.log2(p)

    def _calculate_total_ic(self, h_total: float, n: int) -> float:
        """Calculate total information content with small sample correction."""
        # Background entropy
        p_min_exp = 0.4
        p_max_exp = 0.1
        p_neu_exp = 0.5
        h_background = -(
            p_min_exp * np.log2(p_min_exp)
            + p_max_exp * np.log2(p_max_exp)
            + p_neu_exp * np.log2(p_neu_exp)
        )

        # Small sample correction
        correction = (3 - 1) / (2 * np.log(2) * n)

        return h_background - h_total - correction

    def _read_equivalences(self, equiv_file: Path) -> Dict[int, int]:
        """Read residue equivalences from file."""
        equivalences = {}
        with open(equiv_file) as f:
            next(f)  # Skip header
            for line in f:
                fields = line.strip().split()
                equivalences[int(fields[1])] = int(fields[0])
        return equivalences

    def _read_frustration_data(
        self, done_dir: Path, pdb_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Read frustration data for a single structure.

        Args:
            done_dir: Path to .done directory
            pdb_id: PDB identifier

        Returns:
            DataFrame with frustration data or None if data not found
        """
        try:
            # Look for frustration data file
            frust_file = done_dir / "FrustrationData" / f"{pdb_id}.pdb_configurational"
            if not frust_file.exists():
                logger.warning(
                    f"No frustration data found for {pdb_id} at {frust_file}"
                )
                return None

            # Read data
            df = pd.read_csv(frust_file, sep="\s+")

            # Process data
            contacts = []
            for _, row in df.iterrows():
                # Get residue pairs and their frustration
                res1 = int(row["Res1"])
                res2 = int(row["Res2"])
                frust_index = float(row["FrstIndex"])
                frust_state = str(row["FrstState"])

                # Add contact information
                contacts.append(
                    {
                        "Residue1": res1,
                        "Residue2": res2,
                        "FrustrationIndex": frust_index,
                        "FrustrationState": frust_state,
                        "Structure": pdb_id,
                    }
                )

            # Convert to DataFrame
            if contacts:
                return pd.DataFrame(contacts)
            else:
                logger.warning(f"No valid contacts found for {pdb_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to read frustration data for {pdb_id}: {str(e)}")
            return None

    def _enrich_with_ic(
        self, frust_data: pd.DataFrame, ic_results: List[PositionInformation]
    ) -> pd.DataFrame:
        """
        Enrich frustration data with information content results.

        Args:
            frust_data: DataFrame with frustration data
            ic_results: List of position-specific information content results

        Returns:
            Enriched DataFrame
        """
        # Create mapping of position to IC data
        ic_map = {r.position: r for r in ic_results}

        # Add IC columns
        frust_data["IC_Total"] = frust_data["Residue1"].map(
            lambda x: ic_map.get(x).ic_total if x in ic_map else 0.0
        )
        frust_data["FrustState"] = frust_data["Residue1"].map(
            lambda x: ic_map.get(x).frust_state if x in ic_map else "UNK"
        )

        return frust_data
