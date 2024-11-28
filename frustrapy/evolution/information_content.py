from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from Bio import SeqIO
from .data_classes import MSAData, EvolutionaryFrustrationData

logger = logging.getLogger(__name__)


@dataclass
class PositionInformation:
    """Stores position-specific information content"""

    position: int
    residue: str
    chain: str
    conservation: float
    min_percent: float
    neu_percent: float
    max_percent: float
    min_count: int
    neu_count: int
    max_count: int
    h_min: float
    h_neu: float
    h_max: float
    h_total: float
    ic_min: float
    ic_neu: float
    ic_max: float
    ic_total: float
    frust_state: str
    conserved_state: str


class InformationContentCalculator:
    """Calculates sequence and frustration information content"""

    def __init__(
        self, msa_data: MSAData, results_dir: Path, reference_pdb: Optional[str] = None
    ):
        """
        Initialize calculator.

        Args:
            msa_data: Multiple sequence alignment data
            results_dir: Output directory
            reference_pdb: Reference PDB identifier
        """
        self.msa_data = msa_data
        self.results_dir = Path(results_dir)
        self.reference_pdb = reference_pdb

    def calculate(self) -> List[PositionInformation]:
        """Calculate position-specific information content"""
        try:
            # Load frustration data
            frustration_data = self._load_frustration_data()

            # Calculate sequence conservation
            conservation = self._calculate_conservation()

            # Calculate frustration information content
            results = []
            for pos in range(self.msa_data.length):
                # Get position data
                pos_data = self._get_position_data(pos, frustration_data)

                # Calculate information content
                ic_data = self._calculate_position_ic(pos_data)

                # Combine results
                results.append(
                    PositionInformation(
                        position=pos + 1,
                        residue=pos_data["residue"],
                        chain=pos_data["chain"],
                        conservation=conservation[pos],
                        min_percent=pos_data["frustration"]["min_percent"],
                        neu_percent=pos_data["frustration"]["neu_percent"],
                        max_percent=pos_data["frustration"]["max_percent"],
                        min_count=pos_data["frustration"]["min_count"],
                        neu_count=pos_data["frustration"]["neu_count"],
                        max_count=pos_data["frustration"]["max_count"],
                        h_min=ic_data["h_min"],
                        h_neu=ic_data["h_neu"],
                        h_max=ic_data["h_max"],
                        h_total=ic_data["h_total"],
                        ic_min=ic_data["ic_min"],
                        ic_neu=ic_data["ic_neu"],
                        ic_max=ic_data["ic_max"],
                        ic_total=ic_data["ic_total"],
                        frust_state=ic_data["frust_state"],
                        conserved_state=ic_data["conserved_state"],
                    )
                )

            # Save results
            self._save_results(results)

            return results

        except Exception as e:
            logger.error(f"Failed to calculate information content: {str(e)}")
            raise

    def _load_frustration_data(self) -> pd.DataFrame:
        """
        Load and process frustration data.

        Returns:
            DataFrame with processed frustration data

        Raises:
            FileNotFoundError: If frustration data file not found
        """
        try:
            # Get frustration data file
            data_file = self.results_dir / "data" / "FrustrationData.csv"
            logger.debug(f"Looking for frustration data at: {data_file}")

            if not data_file.exists():
                logger.error(f"Frustration data file not found at {data_file}")
                logger.debug(
                    f"Results directory contents: {list(self.results_dir.glob('**/*'))}"
                )
                raise FileNotFoundError(f"Frustration data not found: {data_file}")

            # Read data
            logger.debug("Reading frustration data file")
            df = pd.read_csv(data_file)
            logger.debug(f"Read {len(df)} rows of frustration data")
            logger.debug(f"Columns in data: {df.columns.tolist()}")

            # Process data using Res1 column instead of Position
            processed_data = []
            logger.debug("Processing frustration data rows")

            for _, row in df.iterrows():
                try:
                    pos = int(row["Res1"])  # Use Res1 instead of Position
                    structure = row["structure"]
                    frust_index = float(row["FrstIndex"])
                    frust_state = row["FrstState"]

                    processed_data.append(
                        {
                            "Position": pos,  # Keep Position in processed data for compatibility
                            "Structure": structure,
                            "FrustrationIndex": frust_index,
                            "FrustrationState": frust_state,
                        }
                    )
                except Exception as row_error:
                    logger.error(f"Error processing row: {row}")
                    logger.error(f"Row processing error: {str(row_error)}")
                    continue

            logger.debug(f"Processed {len(processed_data)} rows of data")
            result_df = pd.DataFrame(processed_data)
            logger.debug(
                f"Created DataFrame with columns: {result_df.columns.tolist()}"
            )

            return result_df

        except Exception as e:
            logger.error(f"Failed to load frustration data: {str(e)}")
            logger.debug(f"Current working directory: {Path.cwd()}")
            logger.debug(f"Results directory (absolute): {self.results_dir.absolute()}")
            raise

    def _calculate_conservation(self) -> np.ndarray:
        """Calculate position-specific conservation scores"""
        # Get conservation matrix
        matrix = self.msa_data.conservation_matrix

        # Calculate Shannon entropy for each position
        conservation = np.zeros(self.msa_data.length)
        for pos in range(self.msa_data.length):
            freqs = matrix[:, pos]
            entropy = 0
            for freq in freqs[freqs > 0]:
                entropy -= freq * np.log2(freq)

            # Convert to conservation score
            max_entropy = np.log2(20)  # Maximum possible entropy
            conservation[pos] = 1 - (entropy / max_entropy)

        return conservation

    def _get_position_data(
        self, position: int, frustration_data: Dict[int, Dict]
    ) -> Dict:
        """Get combined data for a position"""
        pos_data = frustration_data.get(
            position + 1,
            {
                "residue": "-",
                "chain": "A",
                "frustration": {
                    "min_percent": 0.0,
                    "neu_percent": 0.0,
                    "max_percent": 0.0,
                    "min_count": 0,
                    "neu_count": 0,
                    "max_count": 0,
                },
            },
        )

        return pos_data

    def _calculate_position_ic(self, pos_data: Dict) -> Dict:
        """
        Calculate information content for a position.

        Args:
            pos_data: Dictionary containing position data

        Returns:
            Dictionary with information content values
        """
        try:
            logger.debug(
                f"Calculating IC for position {pos_data.get('position', 'unknown')}"
            )
            logger.debug(f"Input data: {pos_data}")

            # Get frustration data
            frust_data = pos_data.get("frustration", {})
            logger.debug(f"Frustration data: {frust_data}")

            # Extract percentages
            p_min = frust_data.get("min_percent", 0.0)
            p_neu = frust_data.get("neu_percent", 0.0)
            p_max = frust_data.get("max_percent", 0.0)
            logger.debug(
                f"Percentages - MIN: {p_min:.3f}, NEU: {p_neu:.3f}, MAX: {p_max:.3f}"
            )

            # Calculate entropy terms
            h_min = self._calculate_entropy(p_min)
            h_neu = self._calculate_entropy(p_neu)
            h_max = self._calculate_entropy(p_max)
            h_total = h_min + h_neu + h_max
            logger.debug(
                f"Entropy terms - MIN: {h_min:.3f}, NEU: {h_neu:.3f}, MAX: {h_max:.3f}, Total: {h_total:.3f}"
            )

            # Get sample size
            n = sum(
                [
                    frust_data.get("min_count", 0),
                    frust_data.get("neu_count", 0),
                    frust_data.get("max_count", 0),
                ]
            )
            logger.debug(f"Sample size: {n}")

            # Calculate total information content
            ic_total = self._calculate_total_ic(h_total, n) if n > 0 else 0.0
            logger.debug(f"Total IC: {ic_total:.3f}")

            # Calculate individual contributions
            ic_min = ic_total * p_min if p_min > 0 else 0.0
            ic_neu = ic_total * p_neu if p_neu > 0 else 0.0
            ic_max = ic_total * p_max if p_max > 0 else 0.0
            logger.debug(
                f"IC contributions - MIN: {ic_min:.3f}, NEU: {ic_neu:.3f}, MAX: {ic_max:.3f}"
            )

            # Determine conserved state
            if n > 0:
                states = {"MIN": p_min, "NEU": p_neu, "MAX": p_max}
                conserved_state = max(states.items(), key=lambda x: x[1])[0]
                logger.debug(f"Conserved state: {conserved_state}")
            else:
                conserved_state = "UNK"
                logger.debug("No data available, using UNK state")

            result = {
                "ic_min": ic_min,
                "ic_neu": ic_neu,
                "ic_max": ic_max,
                "ic_total": ic_total,
                "conserved_state": conserved_state,
            }
            logger.debug(f"Final result: {result}")

            return result

        except Exception as e:
            logger.error(f"Error calculating IC: {str(e)}")
            logger.debug(f"Input data that caused error: {pos_data}")
            # Return default values on error
            return {
                "ic_min": 0.0,
                "ic_neu": 0.0,
                "ic_max": 0.0,
                "ic_total": 0.0,
                "conserved_state": "UNK",
            }

    def _save_results(self, results: List[PositionInformation]) -> None:
        """Save results to file"""
        data = []
        for result in results:
            row = {
                "Position": result.position,
                "Residue": result.residue,
                "Chain": result.chain,
                "Conservation": result.conservation,
                "Pct_Min": result.min_percent,
                "Pct_Neu": result.neu_percent,
                "Pct_Max": result.max_percent,
                "Count_Min": result.min_count,
                "Count_Neu": result.neu_count,
                "Count_Max": result.max_count,
                "H_Min": result.h_min,
                "H_Neu": result.h_neu,
                "H_Max": result.h_max,
                "H_Total": result.h_total,
                "IC_Min": result.ic_min,
                "IC_Neu": result.ic_neu,
                "IC_Max": result.ic_max,
                "IC_Total": result.ic_total,
                "FrustState": result.frust_state,
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(self.results_dir / "InformationContent.csv", index=False)

    def _calculate_entropy(self, p: float) -> float:
        """
        Calculate Shannon entropy term.

        Args:
            p: Probability value

        Returns:
            Shannon entropy value
        """
        if p <= 0:
            return 0
        return -p * np.log2(p)

    def _calculate_total_ic(self, h_total: float, n: int) -> float:
        """
        Calculate total information content with small sample correction.

        Args:
            h_total: Total entropy
            n: Sample size

        Returns:
            Corrected information content
        """
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
