from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from .data_classes import MSAData

logger = logging.getLogger(__name__)


@dataclass
class ContactInformationResult:
    """Results from contact information calculation"""

    residue1: int
    residue2: int
    num_contacts: int
    freq_contacts: float
    p_neu: float
    p_min: float
    p_max: float
    h_neu: float
    h_min: float
    h_max: float
    h_total: float
    ic_neu: float
    ic_min: float
    ic_max: float
    ic_total: float
    frust_state: str


class ContactInformationCalculator:
    """Calculates contact information content"""

    def __init__(
        self, results_dir: Path, msa_data: MSAData, mode: str = "configurational"
    ):
        """Initialize calculator"""
        self.results_dir = Path(results_dir)
        self.msa_data = msa_data
        self.mode = mode
        self.cutoff_min = 0.78
        self.cutoff_max = -1.0

    def calculate(self) -> List[ContactInformationResult]:
        """Calculate contact information content"""
        try:
            # Get frustration data for all structures
            frustration_data = self._load_frustration_data()

            # Calculate contact frequencies and states
            contact_stats = self._calculate_contact_statistics(frustration_data)

            # Calculate information content
            results = self._calculate_information_content(contact_stats)

            # Save results
            self._save_results(results)

            return results

        except Exception as e:
            logger.error(f"Failed to calculate contact information: {str(e)}")
            raise

    def _load_frustration_data(self) -> Dict[str, pd.DataFrame]:
        """Load frustration data for all structures"""
        data = {}
        for pdb_dir in self.results_dir.glob("*.done"):
            frust_file = pdb_dir / "FrustrationData" / f"{pdb_dir.stem}.pdb_{self.mode}"
            if frust_file.exists():
                data[pdb_dir.stem] = pd.read_csv(frust_file, sep="\s+")
        return data

    def _calculate_contact_statistics(
        self, frustration_data: Dict[str, pd.DataFrame]
    ) -> Dict[Tuple[int, int], Dict]:
        """Calculate statistics for each contact"""
        contact_stats = {}

        for pdb_id, df in frustration_data.items():
            for _, row in df.iterrows():
                pos1, pos2 = int(row["Res1"]), int(row["Res2"])
                if pos1 > pos2:
                    pos1, pos2 = pos2, pos1

                key = (pos1, pos2)
                if key not in contact_stats:
                    contact_stats[key] = {
                        "total": 0,
                        "minimal": 0,
                        "neutral": 0,
                        "maximal": 0,
                        "indices": [],
                    }

                stats = contact_stats[key]
                stats["total"] += 1
                stats["indices"].append(row["FrstIndex"])

                # Classify frustration state
                if row["FrstIndex"] >= self.cutoff_min:
                    stats["minimal"] += 1
                elif row["FrstIndex"] <= self.cutoff_max:
                    stats["maximal"] += 1
                else:
                    stats["neutral"] += 1

        return contact_stats

    def _calculate_information_content(
        self, contact_stats: Dict[Tuple[int, int], Dict]
    ) -> List[ContactInformationResult]:
        """Calculate information content for contacts"""
        results = []
        total_structures = len(self._load_frustration_data())

        for (pos1, pos2), stats in contact_stats.items():
            if stats["total"] <= 1:
                continue

            # Calculate probabilities
            p_min = stats["minimal"] / stats["total"]
            p_max = stats["maximal"] / stats["total"]
            p_neu = stats["neutral"] / stats["total"]

            # Calculate Shannon entropy
            h_min = self._calculate_entropy(p_min)
            h_max = self._calculate_entropy(p_max)
            h_neu = self._calculate_entropy(p_neu)
            h_total = -(h_min + h_max + h_neu)

            # Calculate information content
            ic_total = self._calculate_total_ic(h_total, stats["total"])
            ic_min = ic_total * p_min
            ic_max = ic_total * p_max
            ic_neu = ic_total * p_neu

            # Determine conserved state
            max_count = max(stats["minimal"], stats["maximal"], stats["neutral"])
            if max_count == stats["minimal"]:
                conserved = "MIN"
            elif max_count == stats["maximal"]:
                conserved = "MAX"
            else:
                conserved = "NEU"

            results.append(
                ContactInformationResult(
                    residue1=pos1,
                    residue2=pos2,
                    num_contacts=stats["total"],
                    freq_contacts=stats["total"] / total_structures,
                    p_neu=p_neu,
                    p_min=p_min,
                    p_max=p_max,
                    h_neu=h_neu,
                    h_min=h_min,
                    h_max=h_max,
                    h_total=h_total,
                    ic_neu=ic_neu,
                    ic_min=ic_min,
                    ic_max=ic_max,
                    ic_total=ic_total,
                    frust_state=conserved,
                )
            )

        return results

    def _calculate_entropy(self, p: float) -> float:
        """Calculate Shannon entropy"""
        if p <= 0:
            return 0
        return -p * np.log2(p)

    def _calculate_total_ic(self, h_total: float, n: int) -> float:
        """Calculate total information content with correction"""
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

    def _save_results(self, results: List[ContactInformationResult]) -> None:
        """Save results to file"""
        data = []
        for result in results:
            data.append(
                {
                    "Res1": result.residue1,
                    "Res2": result.residue2,
                    "NumContacts": result.num_contacts,
                    "FreqConts": result.freq_contacts,
                    "pNEU": result.p_neu,
                    "pMIN": result.p_min,
                    "pMAX": result.p_max,
                    "hNEU": result.h_neu,
                    "hMIN": result.h_min,
                    "hMAX": result.h_max,
                    "hTotal": result.h_total,
                    "icNEU": result.ic_neu,
                    "icMIN": result.ic_min,
                    "icMAX": result.ic_max,
                    "icTotal": result.ic_total,
                    "FrstState": result.frust_state,
                }
            )

        df = pd.DataFrame(data)
        df.to_csv(self.results_dir / f"IC_{self.mode}.csv", index=False)
