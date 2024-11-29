from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union
from pathlib import Path
import logging
from Bio import SeqIO
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from .data_classes import MSAData, EvolutionaryFrustrationData
from .exceptions import FrustraEvoError

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

    REQUIRED_DIRECTORIES: Set[str] = {
        "data",
        "plots",
        "Equivalences",
        "Frustration",
        "pdbs",
        "msa",
        "logs",
    }

    def __init__(
        self,
        msa_data: MSAData,
        results_dir: Union[str, Path],
        reference_pdb: Optional[str] = None,
        pdb_dir: Optional[Union[str, Path]] = None,
        mode: str = "singleresidue",
    ) -> None:
        """
        Initialize calculator with expanded options.

        Args:
            msa_data: Multiple sequence alignment data
            results_dir: Output directory path
            reference_pdb: Reference PDB identifier
            mode: Analysis mode ("singleresidue", "mutational", or "configurational")

        Raises:
            FrustraEvoError: If initialization fails
        """
        try:
            self.msa_data = msa_data
            self.results_dir = Path(results_dir)
            self.reference_pdb = reference_pdb
            self.mode = mode.lower()
            # Validate mode
            valid_modes = {"singleresidue", "mutational", "configurational"}
            if self.mode not in valid_modes:
                raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

            # Initialize paths
            self.pdb_dir = pdb_dir
            self.msa_dir = self.results_dir / "msa"
            self.data_dir = self.results_dir / "data"
            self.logs_dir = self.results_dir / "logs"
            self.pdb_dest_dir = self.results_dir / "pdbs"
            # Create directory structure
            self._setup_directories()

            logger.info(f"Initialized calculator in {mode} mode")
            logger.debug(f"Results directory: {self.results_dir}")

        except Exception as e:
            logger.error(f"Failed to initialize calculator: {str(e)}")
            raise FrustraEvoError("Calculator initialization failed") from e

    def _setup_directories(self) -> None:
        """
        Create required directory structure for analysis.

        Raises:
            FrustraEvoError: If directory creation fails
        """
        try:
            for directory in self.REQUIRED_DIRECTORIES:
                dir_path = self.results_dir / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {dir_path}")

        except Exception as e:
            logger.error(f"Failed to create directory structure: {str(e)}")
            raise FrustraEvoError("Directory setup failed") from e

    def _copy_required_files(self) -> None:
        """
        Copy input files to working directories.

        Raises:
            FrustraEvoError: If file copying fails
        """
        try:
            # Create PDB directory if it doesn't exist

            self.pdb_dest_dir.mkdir(parents=True, exist_ok=True)

            # Copy PDB files from source directory
            pdb_source_dir = Path(self.pdb_dir) if self.pdb_dir else None
            if pdb_source_dir and pdb_source_dir.exists():
                pdb_files = list(pdb_source_dir.glob("*.pdb"))
                if not pdb_files:
                    raise FileNotFoundError(f"No PDB files found in {pdb_source_dir}")

                for pdb_file in pdb_files:
                    dest = self.pdb_dest_dir / pdb_file.name
                    shutil.copy2(pdb_file, dest)
                    logger.debug(f"Copied PDB file: {pdb_file.name} to {dest}")

                logger.info(f"Copied {len(pdb_files)} PDB files to {self.pdb_dest_dir}")

                # Verify files were copied
                copied_files = list(self.pdb_dest_dir.glob("*.pdb"))
                if not copied_files:
                    raise FrustraEvoError(
                        f"Failed to copy PDB files to {self.pdb_dest_dir}"
                    )
                logger.debug(
                    f"Verified {len(copied_files)} PDB files in destination directory"
                )

            # Copy MSA file
            if self.msa_data.fasta_file and self.msa_data.fasta_file.exists():
                msa_dest_dir = self.results_dir / "msa"
                msa_dest_dir.mkdir(parents=True, exist_ok=True)

                dest = msa_dest_dir / self.msa_data.fasta_file.name
                shutil.copy2(self.msa_data.fasta_file, dest)
                logger.debug(
                    f"Copied MSA file: {self.msa_data.fasta_file.name} to {dest}"
                )

                # Verify MSA file was copied
                if not dest.exists():
                    raise FrustraEvoError(f"Failed to copy MSA file to {dest}")

        except Exception as e:
            logger.error(f"Failed to copy required files: {str(e)}")
            logger.debug(
                f"Source PDB directory: {pdb_source_dir if pdb_source_dir else 'None'}"
            )
            logger.debug(f"Destination PDB directory: {pdb_dest_dir}")
            logger.debug(
                f"Source MSA file: {self.msa_data.fasta_file if self.msa_data.fasta_file else 'None'}"
            )
            raise FrustraEvoError("File copying failed") from e

    def _get_pdb_sequence(self, pdb_path: Path) -> str:
        """
        Extract amino acid sequence from PDB file.

        Args:
            pdb_path: Path to PDB file

        Returns:
            str: Amino acid sequence

        Raises:
            FrustraEvoError: If sequence extraction fails
        """
        try:
            AA_CODES: Dict[str, str] = {
                "ALA": "A",
                "ARG": "R",
                "ASN": "N",
                "ASP": "D",
                "CYS": "C",
                "GLN": "Q",
                "GLU": "E",
                "GLY": "G",
                "HIS": "H",
                "ILE": "I",
                "LEU": "L",
                "LYS": "K",
                "MET": "M",
                "PHE": "F",
                "PRO": "P",
                "SER": "S",
                "THR": "T",
                "TRP": "W",
                "TYR": "Y",
                "VAL": "V",
            }

            sequence = []
            prev_res_num = None

            with pdb_path.open() as f:
                for line in f:
                    if line.startswith("ATOM") and line[12:16].strip() == "CA":
                        res_name = line[17:20].strip()
                        res_num = int(line[22:26])

                        if res_num != prev_res_num and res_name in AA_CODES:
                            sequence.append(AA_CODES[res_name])
                            prev_res_num = res_num

            return "".join(sequence)

        except Exception as e:
            logger.error(f"Failed to extract sequence from PDB {pdb_path}: {str(e)}")
            raise FrustraEvoError("PDB sequence extraction failed") from e

    def _validate_sequences(self) -> None:
        """
        Validate sequences against PDB structures.

        Raises:
            FrustraEvoError: If sequence validation fails
        """
        try:
            logger.info("Validating sequences against PDB structures")

            # Ensure MSA directory exists
            self.msa_dir.mkdir(parents=True, exist_ok=True)

            clean_msa = self.msa_dir / "MSA_Clean.fasta"
            error_log = self.logs_dir / "ErrorSeq.log"

            # First copy the original MSA file to clean_msa if it doesn't exist
            if not clean_msa.exists() and self.msa_data.fasta_file:
                shutil.copy2(self.msa_data.fasta_file, clean_msa)
                logger.debug(f"Copied original MSA to: {clean_msa}")

            valid_sequences = []
            with error_log.open("w") as out_log:
                for record in SeqIO.parse(self.msa_data.fasta_file, "fasta"):
                    seq_id = record.id
                    sequence = str(record.seq).replace("-", "")

                    # Check PDB existence
                    pdb_file = self.pdb_dest_dir / f"{seq_id}.pdb"
                    if not pdb_file.exists():
                        logger.warning(f"PDB file not found: {pdb_file}")
                        out_log.write(f"Missing PDB file: {seq_id}\n")
                        continue

                    try:
                        # Compare sequences
                        pdb_sequence = self._get_pdb_sequence(pdb_file)
                        if sequence == pdb_sequence:
                            logger.debug(f"Sequence validated: {seq_id}")
                            valid_sequences.append(record)
                        else:
                            logger.warning(f"Sequence mismatch: {seq_id}")
                            out_log.write(
                                f"Sequence mismatch for {seq_id}:\n"
                                f"MSA:  {sequence}\n"
                                f"PDB:  {pdb_sequence}\n"
                                f"Diff: {self._get_sequence_diff(sequence, pdb_sequence)}\n\n"
                            )
                    except Exception as e:
                        logger.error(f"Error processing sequence {seq_id}: {str(e)}")
                        out_log.write(f"Error processing {seq_id}: {str(e)}\n")

            # Write validated sequences to clean MSA file
            if valid_sequences:
                with clean_msa.open("w") as out_msa:
                    SeqIO.write(valid_sequences, out_msa, "fasta")
                logger.info(
                    f"Wrote {len(valid_sequences)} validated sequences to {clean_msa}"
                )
            else:
                raise FrustraEvoError("No valid sequences found after validation")

            # Verify reference sequence is present
            if self.reference_pdb:
                ref_found = any(
                    record.id == self.reference_pdb for record in valid_sequences
                )
                if not ref_found:
                    raise ValueError(
                        f"Reference sequence {self.reference_pdb} not found in validated sequences"
                    )

        except Exception as e:
            logger.error(f"Failed to validate sequences: {str(e)}")
            raise FrustraEvoError("Sequence validation failed") from e

    def _get_sequence_diff(self, seq1: str, seq2: str) -> str:
        """
        Generate a visual difference between two sequences.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            str: Visual difference markers
        """
        return "".join(
            " " if (i >= len(seq1) or i >= len(seq2) or seq1[i] == seq2[i]) else "^"
            for i in range(max(len(seq1), len(seq2)))
        )

    def _prepare_reference_alignment(self) -> None:
        """
        Prepare reference sequence alignment.

        Raises:
            FrustraEvoError: If reference alignment preparation fails
            ValueError: If reference PDB is not specified or found
        """
        try:
            logger.info("Preparing reference alignment")

            if not self.reference_pdb:
                raise ValueError("Reference PDB must be specified")

            clean_msa = self.msa_dir / "MSA_Clean.fasta"
            ref_msa = self.msa_dir / "MSA_Clean_Ref.fasta"

            if not clean_msa.exists():
                raise FileNotFoundError(f"Clean MSA file not found: {clean_msa}")

            # Extract reference sequence
            ref_found = False
            records = []

            # First pass to find reference sequence
            for record in SeqIO.parse(clean_msa, "fasta"):
                if record.id == self.reference_pdb:
                    records.insert(0, record)  # Add reference sequence first
                    ref_found = True
                else:
                    records.append(record)

            if not ref_found:
                raise ValueError(
                    f"Reference sequence {self.reference_pdb} not found in MSA"
                )

            # Write aligned sequences
            with ref_msa.open("w") as out:
                SeqIO.write(records, out, "fasta")

            logger.debug(f"Created reference alignment with {len(records)} sequences")

        except Exception as e:
            logger.error(f"Failed to prepare reference alignment: {str(e)}")
            raise FrustraEvoError("Reference alignment preparation failed") from e

    def _create_final_alignment(self) -> None:
        """
        Create final alignment with position mapping.

        Raises:
            FrustraEvoError: If final alignment creation fails
        """
        try:
            logger.info("Creating final alignment")

            ref_msa = self.msa_dir / "MSA_Clean_Ref.fasta"
            final_msa = self.msa_dir / "MSA_Final.fasta"
            positions_file = self.data_dir / "Positions.txt"

            if not ref_msa.exists():
                raise FileNotFoundError(f"Reference MSA file not found: {ref_msa}")

            # Process alignment and create position mapping
            with final_msa.open("w") as out_msa, positions_file.open("w") as out_pos:
                for record in SeqIO.parse(ref_msa, "fasta"):
                    seq_id = record.id
                    sequence = str(record.seq)

                    # Write sequence to final MSA
                    out_msa.write(f">{seq_id}\n{sequence}\n")

                    # Create position mapping for reference sequence
                    if seq_id == self.reference_pdb:
                        out_pos.write(f">{seq_id}\n")
                        positions = []
                        pos = 1

                        for aa in sequence:
                            if aa != "-":
                                positions.append(str(pos))
                                pos += 1

                        out_pos.write(" ".join(positions) + "\n")

            logger.debug(f"Created final alignment and position mapping")

        except Exception as e:
            logger.error(f"Failed to create final alignment: {str(e)}")
            raise FrustraEvoError("Final alignment creation failed") from e

    def _prepare_logo_data(self) -> None:
        """
        Prepare sequence data for logo generation.

        Raises:
            FrustraEvoError: If logo data preparation fails
        """
        try:
            logger.info("Preparing logo data")

            final_msa = self.msa_dir / "MSA_Final.fasta"
            logo_data = self.data_dir / "Logo.fasta"

            if not final_msa.exists():
                raise FileNotFoundError(f"Final MSA file not found: {final_msa}")

            # Extract sequences without headers
            with logo_data.open("w") as out:
                for record in SeqIO.parse(final_msa, "fasta"):
                    out.write(f"{record.seq}\n")

            logger.debug(f"Created logo data file: {logo_data}")

        except Exception as e:
            logger.error(f"Failed to prepare logo data: {str(e)}")
            raise FrustraEvoError("Logo data preparation failed") from e

    def _validate_logo_data(self) -> None:
        """
        Validate logo data and combine equivalence files.

        Raises:
            FrustraEvoError: If logo data validation fails
        """
        try:
            logger.info("Validating logo data")

            equiv_dir = self.results_dir / "Equivalences"
            combined_file = equiv_dir / "AllEquivalences.txt"

            if not equiv_dir.exists() or not any(equiv_dir.iterdir()):
                raise FileNotFoundError(f"No equivalence files found in: {equiv_dir}")

            # Combine and validate equivalence files
            valid_entries = []
            header_written = False

            for equiv_file in equiv_dir.glob("Equival_*.txt"):
                with equiv_file.open() as f:
                    # Read and store header from first file
                    header = next(f, None)  # Skip header line
                    if not header_written and header:
                        valid_entries.append(header)
                        header_written = True

                    # Process data lines
                    for line in f:
                        fields = line.strip().split("\t")
                        if len(fields) > 4:  # Valid line with enough fields
                            try:
                                # Validate numeric fields
                                pos = int(
                                    fields[0]
                                )  # Now this won't try to parse "MSA_pos"
                                if pos <= 0:
                                    logger.warning(
                                        f"Invalid position in {equiv_file.name}: {pos}"
                                    )
                                    continue
                                valid_entries.append(line)
                            except ValueError:
                                logger.warning(
                                    f"Invalid data format in {equiv_file.name}: {line.strip()}"
                                )
                                continue

            # Write validated entries
            with combined_file.open("w") as out:
                out.writelines(valid_entries)

            logger.debug(
                f"Combined {len(valid_entries)-1} valid entries into {combined_file}"
            )  # Subtract 1 for header

        except Exception as e:
            logger.error(f"Failed to validate logo data: {str(e)}")
            raise FrustraEvoError("Logo data validation failed") from e

    def _calculate_conservation(self) -> Dict[int, float]:
        """
        Calculate position-specific conservation scores.

        Returns:
            Dict[int, float]: Position-specific conservation scores

        Raises:
            FrustraEvoError: If conservation calculation fails
        """
        try:
            logger.info("Calculating conservation scores")

            conservation_scores = {}
            matrix = self.msa_data.conservation_matrix

            # Calculate Shannon entropy for each position
            for pos in range(matrix.shape[1]):
                freqs = matrix[:, pos]
                entropy = 0.0

                # Calculate position-specific entropy
                for freq in freqs[freqs > 0]:  # Only consider non-zero frequencies
                    entropy -= freq * np.log2(freq)

                # Convert entropy to conservation score
                max_entropy = np.log2(20)  # Maximum possible entropy (20 amino acids)
                conservation_scores[pos + 1] = 1 - (entropy / max_entropy)

            logger.debug(
                f"Calculated conservation scores for {len(conservation_scores)} positions"
            )
            return conservation_scores

        except Exception as e:
            logger.error(f"Failed to calculate conservation scores: {str(e)}")
            raise FrustraEvoError("Conservation calculation failed") from e

    def _calculate_position_ic(self, pos_data: Dict) -> Dict[str, float]:
        """
        Calculate information content for a position.

        Args:
            pos_data: Dictionary containing position data

        Returns:
            Dict[str, float]: Information content values and states

        Raises:
            FrustraEvoError: If IC calculation fails
        """
        try:
            # Add position to debug message if available
            position = pos_data.get("position", "unknown")
            logger.debug(f"Calculating IC for position {position}")

            # Extract frustration data
            frust_data = pos_data.get("frustration", {})

            # Get percentages with validation
            p_min = max(0.0, min(1.0, frust_data.get("min_percent", 0.0)))
            p_neu = max(0.0, min(1.0, frust_data.get("neu_percent", 0.0)))
            p_max = max(0.0, min(1.0, frust_data.get("max_percent", 0.0)))

            # Normalize percentages if sum > 1
            total = p_min + p_neu + p_max
            if total > 1.0:
                p_min /= total
                p_neu /= total
                p_max /= total

            # Calculate entropy terms
            h_min = self._calculate_entropy_term(p_min)
            h_neu = self._calculate_entropy_term(p_neu)
            h_max = self._calculate_entropy_term(p_max)
            h_total = h_min + h_neu + h_max

            # Get sample counts
            counts = [
                frust_data.get("min_count", 0),
                frust_data.get("neu_count", 0),
                frust_data.get("max_count", 0),
            ]
            sample_size = sum(counts)

            # Calculate information content
            ic_total = (
                self._calculate_total_ic(h_total, sample_size)
                if sample_size > 0
                else 0.0
            )

            # Calculate individual contributions
            ic_min = ic_total * p_min if p_min > 0 else 0.0
            ic_neu = ic_total * p_neu if p_neu > 0 else 0.0
            ic_max = ic_total * p_max if p_max > 0 else 0.0

            # Determine states
            if sample_size > 0:
                states = {"MIN": p_min, "NEU": p_neu, "MAX": p_max}
                conserved_state = max(states.items(), key=lambda x: x[1])[0]
                frust_state = conserved_state
            else:
                conserved_state = frust_state = "UNK"

            return {
                "h_min": h_min,
                "h_neu": h_neu,
                "h_max": h_max,
                "h_total": h_total,
                "ic_min": ic_min,
                "ic_neu": ic_neu,
                "ic_max": ic_max,
                "ic_total": ic_total,
                "frust_state": frust_state,
                "conserved_state": conserved_state,
            }

        except Exception as e:
            logger.error(f"Failed to calculate position IC: {str(e)}")
            raise FrustraEvoError("Position IC calculation failed") from e

    def _calculate_entropy_term(self, probability: float) -> float:
        """
        Calculate Shannon entropy term.

        Args:
            probability: Probability value between 0 and 1

        Returns:
            float: Shannon entropy value

        Raises:
            ValueError: If probability is outside [0,1]
        """
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be between 0 and 1, got {probability}")

        if probability <= 0:
            return 0.0

        return -probability * np.log2(probability)

    def _calculate_total_ic(self, h_total: float, sample_size: int) -> float:
        """
        Calculate total information content with small sample correction.

        Args:
            h_total: Total entropy
            sample_size: Number of samples

        Returns:
            float: Corrected information content

        Raises:
            ValueError: If invalid input values
        """
        if h_total < 0:
            raise ValueError(f"Total entropy must be non-negative, got {h_total}")
        if sample_size <= 0:
            raise ValueError(f"Sample size must be positive, got {sample_size}")

        try:
            # Background entropy calculation
            p_min_exp = 0.4  # Expected minimal frustration probability
            p_max_exp = 0.1  # Expected maximal frustration probability
            p_neu_exp = 0.5  # Expected neutral frustration probability

            h_background = -(
                p_min_exp * np.log2(p_min_exp)
                + p_max_exp * np.log2(p_max_exp)
                + p_neu_exp * np.log2(p_neu_exp)
            )

            # Small sample correction
            correction = (3 - 1) / (2 * np.log(2) * sample_size)

            return max(0.0, h_background - h_total - correction)

        except Exception as e:
            logger.error(f"Failed to calculate total IC: {str(e)}")
            raise FrustraEvoError("Total IC calculation failed") from e

    def _get_position_data(
        self, position: int, frustration_data: Dict[int, Dict]
    ) -> Dict:
        """
        Get combined data for a position.

        Args:
            position: Sequence position
            frustration_data: Dictionary of frustration data

        Returns:
            Dict: Combined position data

        Raises:
            FrustraEvoError: If position data retrieval fails
        """
        try:
            # Add position to debug message
            logger.debug(
                f"Getting data for position {position + 1}"
            )  # +1 for 1-based indexing

            # Default data structure
            default_data = {
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
            }

            # Get position data with validation
            pos_data = frustration_data.get(position + 1, default_data)

            # Add position to warning message if needed
            frust_data = pos_data["frustration"]
            total = (
                frust_data["min_percent"]
                + frust_data["neu_percent"]
                + frust_data["max_percent"]
            )

            if total > 1.0:
                logger.warning(
                    f"Invalid percentages at position {position + 1}, normalizing"
                )
                frust_data["min_percent"] /= total
                frust_data["neu_percent"] /= total
                frust_data["max_percent"] /= total

            return pos_data

        except Exception as e:
            logger.error(
                f"Failed to get position data for position {position + 1}: {str(e)}"
            )
            raise FrustraEvoError(
                f"Position data retrieval failed for position {position + 1}"
            ) from e

    def _load_frustration_data(self) -> Dict[int, Dict]:
        """
        Load and process frustration data.

        Returns:
            Dict[int, Dict]: Processed frustration data by position

        Raises:
            FrustraEvoError: If frustration data loading fails
        """
        try:
            data_file = self.data_dir / "FrustrationData.csv"
            logger.debug(f"Looking for frustration data at: {data_file}")

            if not data_file.exists():
                raise FileNotFoundError(f"Frustration data not found: {data_file}")

            # Read and process data
            df = pd.read_csv(data_file)

            # Process data by position
            position_data = {}
            for pos in df["Res1"].unique():
                pos_df = df[df["Res1"] == pos]

                # Count frustration states
                state_counts = pos_df["FrstState"].value_counts()
                total_counts = len(pos_df)

                # Calculate percentages
                min_count = state_counts.get("minimally", 0)
                neu_count = state_counts.get("neutral", 0)
                max_count = state_counts.get("highly", 0)

                position_data[pos] = {
                    "position": pos,
                    "residue": (
                        pos_df["AA1"].iloc[0]
                        if not pos_df["AA1"].iloc[0] == "-"
                        else "-"
                    ),
                    "chain": pos_df["ChainRes1"].iloc[0],
                    "frustration": {
                        "min_percent": (
                            min_count / total_counts if total_counts > 0 else 0.0
                        ),
                        "neu_percent": (
                            neu_count / total_counts if total_counts > 0 else 0.0
                        ),
                        "max_percent": (
                            max_count / total_counts if total_counts > 0 else 0.0
                        ),
                        "min_count": min_count,
                        "neu_count": neu_count,
                        "max_count": max_count,
                    },
                }

            logger.debug(
                f"Processed frustration data for {len(position_data)} positions"
            )
            return position_data

        except Exception as e:
            logger.error(f"Failed to load frustration data: {str(e)}")
            raise FrustraEvoError("Frustration data loading failed") from e

    def _run_frustration_calculation(self, mode: str) -> pd.DataFrame:
        """
        Run frustration calculation in specified mode.

        Args:
            mode: Calculation mode ("mutational" or "configurational")

        Returns:
            pd.DataFrame: Frustration calculation results

        Raises:
            FrustraEvoError: If frustration calculation fails
        """
        try:
            logger.info(f"Running {mode} frustration calculation")

            # Prepare R script for frustration calculation
            r_script = self.results_dir / "frustration" / "FrustraR.R"
            with r_script.open("w") as f:
                f.write(
                    f"library(frustratometeR)\n"
                    f"PdbsDir <- '{self.results_dir}/frustration'\n"
                    f"ResultsDir <- '{self.results_dir}/frustration'\n"
                    f"dir_frustration(PdbsDir = PdbsDir, Mode = '{mode}', "
                    f"ResultsDir = ResultsDir, Graphics = FALSE)\n"
                )

            # Run R script
            import subprocess

            result = subprocess.run(
                ["Rscript", str(r_script)], capture_output=True, text=True, check=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"R script failed: {result.stderr}")

            # Load and return results
            return self._load_frustration_data()

        except Exception as e:
            logger.error(f"Failed to run {mode} frustration calculation: {str(e)}")
            raise FrustraEvoError("Frustration calculation failed") from e

    def _save_results(self, results: List[PositionInformation]) -> None:
        """
        Save analysis results to files.

        Args:
            results: List of position information results

        Raises:
            FrustraEvoError: If saving results fails
        """
        try:
            logger.info("Saving analysis results")

            # Convert results to DataFrame
            data = []
            for result in results:
                data.append(
                    {
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
                        "ConservedState": result.conserved_state,
                    }
                )

            # Save to CSV
            df = pd.DataFrame(data)
            output_file = self.data_dir / "InformationContent.csv"
            df.to_csv(output_file, index=False)
            logger.debug(f"Saved results to: {output_file}")

            # Generate summary plots
            self._generate_summary_plots(results)

        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise FrustraEvoError("Results saving failed") from e

    def _generate_summary_plots(self, results: List[PositionInformation]) -> None:
        """
        Generate summary visualization plots.

        Args:
            results: List of position information results

        Raises:
            FrustraEvoError: If plot generation fails
        """
        try:
            logger.info("Generating summary plots")

            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style
            plt.style.use("seaborn-v0_8-pastel")

            # Create figure with subplots
            fig = plt.figure(figsize=(15, 15))
            gs = fig.add_gridspec(3, 2, hspace=0.3)

            # 1. Conservation Profile
            ax1 = fig.add_subplot(gs[0, :])
            positions = [r.position for r in results]
            conservation = [r.conservation for r in results]
            ax1.plot(positions, conservation, "b-", linewidth=2)
            ax1.set_title("Sequence Conservation Profile", fontsize=12)
            ax1.set_xlabel("Position")
            ax1.set_ylabel("Conservation Score")

            # 2. Frustration Distribution
            ax2 = fig.add_subplot(gs[1, 0])
            frustration_data = pd.DataFrame(
                [
                    {
                        "Position": r.position,
                        "Minimal": r.min_percent,
                        "Neutral": r.neu_percent,
                        "Maximal": r.max_percent,
                    }
                    for r in results
                ]
            )

            frustration_data.plot(
                x="Position",
                y=["Minimal", "Neutral", "Maximal"],
                kind="bar",
                stacked=True,
                ax=ax2,
                color=["green", "grey", "red"],
            )
            ax2.set_title("Frustration Distribution", fontsize=12)
            ax2.legend(title="Frustration State")

            # 3. Information Content Profile
            ax3 = fig.add_subplot(gs[1, 1])
            ic_total = [r.ic_total for r in results]
            ax3.plot(positions, ic_total, "k-", linewidth=2)
            ax3.set_title("Information Content Profile", fontsize=12)
            ax3.set_xlabel("Position")
            ax3.set_ylabel("Total IC")

            # 4. State Distribution
            ax4 = fig.add_subplot(gs[2, :])
            states = [r.frust_state for r in results]
            state_counts = pd.Series(states).value_counts()
            colors = {"MIN": "green", "NEU": "grey", "MAX": "red", "UNK": "blue"}
            ax4.bar(
                state_counts.index,
                state_counts.values,
                color=[colors.get(s, "black") for s in state_counts.index],
            )
            ax4.set_title("Frustration State Distribution", fontsize=12)
            ax4.set_ylabel("Count")

            # Save plot
            plt.tight_layout()
            output_file = self.results_dir / "plots" / "summary_analysis.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            logger.debug(f"Saved summary plots to: {output_file}")

        except Exception as e:
            logger.error(f"Failed to generate summary plots: {str(e)}")
            raise FrustraEvoError("Plot generation failed") from e

    def _generate_contact_maps(
        self, results: List[PositionInformation], mode: str
    ) -> None:
        """
        Generate contact maps and histograms for frustration analysis.

        Args:
            results: List of position information results
            mode: Analysis mode ("mutational" or "configurational")

        Raises:
            FrustraEvoError: If contact map generation fails
        """
        try:
            logger.info(f"Generating {mode} contact maps and plots")

            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib.colors import LinearSegmentedColormap

            # Create figure with subplots
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])

            # 1. Contact Map
            ax_map = fig.add_subplot(gs[0, 0])
            self._plot_contact_map(ax_map, results, mode)

            # 2. Information Content Distribution
            ax_hist = fig.add_subplot(gs[0, 1])
            self._plot_ic_distribution(ax_hist, results, mode)

            # 3. Frequency Distribution
            ax_freq = fig.add_subplot(gs[1, 0])
            self._plot_frequency_distribution(ax_freq, results)

            # 4. State Distribution
            ax_states = fig.add_subplot(gs[1, 1])
            self._plot_state_distribution(ax_states, results)

            # Save plots
            plt.tight_layout()
            output_file = (
                self.results_dir / "plots" / f"{mode.lower()}_contact_analysis.png"
            )
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            logger.debug(f"Saved contact analysis plots to: {output_file}")

        except Exception as e:
            logger.error(f"Failed to generate contact maps: {str(e)}")
            raise FrustraEvoError("Contact map generation failed") from e

    def _plot_contact_map(
        self, ax: plt.Axes, results: List[PositionInformation], mode: str
    ) -> None:
        """
        Plot contact map with frustration states.
        """
        # Create contact matrix
        n_pos = max(r.position for r in results)
        contact_matrix = np.zeros((n_pos, n_pos))
        state_matrix = np.full((n_pos, n_pos), "UNK", dtype=str)

        for r in results:
            i, j = r.position - 1, r.position - 1
            contact_matrix[i, j] = r.ic_total
            state_matrix[i, j] = r.frust_state

        # Create custom colormap with all possible states
        colors = {
            "MIN": "green",
            "NEU": "grey",
            "MAX": "red",
            "UNK": "white",
            "N": "grey",  # Add this line
            "minimally": "green",  # Add these lines to handle legacy states
            "neutral": "grey",
            "highly": "red",
        }

        cmap = LinearSegmentedColormap.from_list(
            "frustration", ["white", "grey", "red", "green"]
        )

        # Plot heatmap
        sns.heatmap(
            contact_matrix,
            ax=ax,
            cmap=cmap,
            center=0,
            square=True,
            cbar_kws={"label": "Information Content"},
        )

        # Add state markers
        for i in range(n_pos):
            for j in range(n_pos):
                if state_matrix[i, j] != "UNK":
                    ax.plot(
                        j + 0.5,
                        i + 0.5,
                        "o",
                        color=colors.get(
                            state_matrix[i, j], "white"
                        ),  # Use get() with default
                        markersize=3,
                    )

        ax.set_title(f"{mode} Frustration Contact Map")
        ax.set_xlabel("Residue Position")
        ax.set_ylabel("Residue Position")

    def _plot_ic_distribution(
        self, ax: plt.Axes, results: List[PositionInformation], mode: str
    ) -> None:
        """
        Plot information content distribution.

        Args:
            ax: Matplotlib axes
            results: Position information results
            mode: Analysis mode
        """
        # Separate data by state
        data = {
            "MIN": [r.ic_total for r in results if r.frust_state == "MIN"],
            "NEU": [r.ic_total for r in results if r.frust_state == "NEU"],
            "MAX": [r.ic_total for r in results if r.frust_state == "MAX"],
        }

        colors = {"MIN": "green", "NEU": "grey", "MAX": "red"}

        # Plot distributions
        for state, values in data.items():
            if values:
                sns.kdeplot(
                    data=values,
                    ax=ax,
                    color=colors[state],
                    label=state,
                    fill=True,
                    alpha=0.3,
                )

        ax.set_title(f"{mode} Information Content Distribution")
        ax.set_xlabel("Information Content")
        ax.set_ylabel("Density")
        ax.legend(title="Frustration State")

    def _plot_frequency_distribution(
        self, ax: plt.Axes, results: List[PositionInformation]
    ) -> None:
        """
        Plot contact frequency distribution.

        Args:
            ax: Matplotlib axes
            results: Position information results
        """
        positions = [r.position for r in results]
        frequencies = {
            "Minimal": [r.min_percent for r in results],
            "Neutral": [r.neu_percent for r in results],
            "Maximal": [r.max_percent for r in results],
        }

        df = pd.DataFrame(frequencies, index=positions)
        df.plot(kind="bar", stacked=True, ax=ax, color=["green", "grey", "red"])

        ax.set_title("Contact Frequency Distribution")
        ax.set_xlabel("Position")
        ax.set_ylabel("Frequency")
        ax.legend(title="Frustration State")

    def _plot_state_distribution(
        self, ax: plt.Axes, results: List[PositionInformation]
    ) -> None:
        """
        Plot frustration state distribution.

        Args:
            ax: Matplotlib axes
            results: Position information results
        """
        states = [r.frust_state for r in results]
        state_counts = pd.Series(states).value_counts()

        colors = {"MIN": "green", "NEU": "grey", "MAX": "red", "UNK": "blue"}

        state_counts.plot(
            kind="bar",
            ax=ax,
            color=[colors.get(s, "black") for s in state_counts.index],
        )

        ax.set_title("Frustration State Distribution")
        ax.set_xlabel("State")
        ax.set_ylabel("Count")

        # Add percentage labels
        total = sum(state_counts)
        for i, v in enumerate(state_counts):
            percentage = v / total * 100
            ax.text(i, v, f"{percentage:.1f}%", ha="center", va="bottom")

    def _calculate_equivalences(self) -> None:
        """
        Calculate and save residue equivalences for all structures.

        Raises:
            FrustraEvoError: If equivalence calculation fails
        """
        try:
            logger.info("Calculating residue equivalences")

            # Ensure equivalences directory exists
            equiv_dir = self.results_dir / "Equivalences"
            equiv_dir.mkdir(exist_ok=True)

            # Process each sequence in MSA
            for seq_id in self.msa_data.identifiers:
                try:
                    # Get PDB file path
                    pdb_file = self.pdb_dest_dir / f"{seq_id}.pdb"
                    if not pdb_file.exists():
                        logger.warning(f"PDB file not found for {seq_id}")
                        continue

                    # Get sequence from MSA
                    msa_idx = self.msa_data.identifiers.index(seq_id)
                    msa_seq = self.msa_data.sequences[msa_idx]

                    # Calculate equivalences
                    equiv_file = equiv_dir / f"Equival_{seq_id}.txt"
                    self._save_structure_equivalences(
                        pdb_file=pdb_file, msa_seq=msa_seq, output_file=equiv_file
                    )
                    logger.debug(f"Saved equivalences for {seq_id}")

                except Exception as e:
                    logger.error(
                        f"Failed to process equivalences for {seq_id}: {str(e)}"
                    )
                    continue

            # Verify equivalences were created
            equiv_files = list(equiv_dir.glob("Equival_*.txt"))
            if not equiv_files:
                raise FrustraEvoError("No equivalence files were created")

            logger.info(f"Created {len(equiv_files)} equivalence files")

        except Exception as e:
            logger.error(f"Failed to calculate equivalences: {str(e)}")
            raise FrustraEvoError("Equivalence calculation failed") from e

    def _save_structure_equivalences(
        self, pdb_file: Path, msa_seq: str, output_file: Path
    ) -> None:
        """
        Save residue equivalences for a single structure.

        Args:
            pdb_file: Path to PDB file
            msa_seq: Sequence from MSA
            output_file: Output file path
        """
        try:
            # Get PDB sequence
            pdb_seq = self._get_pdb_sequence(pdb_file)

            # Map MSA positions to PDB positions
            equivalences = []
            pdb_pos = 1

            for msa_pos, aa in enumerate(msa_seq, start=1):
                if aa != "-":  # Skip gaps
                    if pdb_pos <= len(pdb_seq):
                        equivalences.append((msa_pos, pdb_pos, aa))
                        pdb_pos += 1

            # Write equivalences
            with output_file.open("w") as f:
                f.write("MSA_pos\tPDB_pos\tResidue\tChain\tStructure\n")
                for msa_pos, pdb_pos, aa in equivalences:
                    f.write(f"{msa_pos}\t{pdb_pos}\t{aa}\tA\t{pdb_file.stem}\n")

        except Exception as e:
            logger.error(f"Failed to save equivalences for {pdb_file.stem}: {str(e)}")
            raise

    def calculate(self) -> List[PositionInformation]:
        """
        Calculate position-specific information content.

        Returns:
            List[PositionInformation]: List of position-specific information content results

        Raises:
            FrustraEvoError: If calculation fails
        """
        try:
            logger.info(f"Starting information content calculation in {self.mode} mode")

            # Copy required files
            self._copy_required_files()

            # Validate sequences
            self._validate_sequences()

            # Prepare alignments
            self._prepare_reference_alignment()
            self._create_final_alignment()

            # Calculate equivalences
            self._calculate_equivalences()

            # Prepare logo data
            self._prepare_logo_data()
            self._validate_logo_data()

            # Load frustration data
            frustration_data = self._load_frustration_data()

            # Calculate conservation scores
            conservation_scores = self._calculate_conservation()

            # Calculate position-specific information content
            results = []
            for pos in range(self.msa_data.length):
                try:
                    # Get position data
                    pos_data = self._get_position_data(pos, frustration_data)

                    # Calculate information content
                    ic_data = self._calculate_position_ic(pos_data)

                    # Create position information object
                    results.append(
                        PositionInformation(
                            position=pos + 1,
                            residue=pos_data["residue"],
                            chain=pos_data["chain"],
                            conservation=conservation_scores.get(pos + 1, 0.0),
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

                except Exception as pos_error:
                    logger.error(
                        f"Error processing position {pos + 1}: {str(pos_error)}"
                    )
                    continue

            # Generate visualizations based on mode
            if self.mode in ["mutational", "configurational"]:
                self._generate_contact_maps(results, self.mode.capitalize())

            # Save results
            self._save_results(results)

            return results

        except Exception as e:
            logger.error(f"Failed to calculate information content: {str(e)}")
            raise FrustraEvoError("Information content calculation failed") from e

    def _organize_output_files(self) -> None:
        """
        Organize output files into appropriate directories.

        Raises:
            FrustraEvoError: If file organization fails
        """
        try:
            logger.info("Organizing output files")

            # Define file mappings
            file_mappings = {
                "data": ["*.csv", "*.txt"],
                "plots": ["*.png", "*.pdf"],
                "logs": ["*.log"],
                "msa": ["*.fasta", "*.aln"],
                "frustration": ["*.pdb", "*.dat"],
            }

            # Move files to appropriate directories
            for directory, patterns in file_mappings.items():
                dir_path = self.results_dir / directory
                dir_path.mkdir(exist_ok=True)

                for pattern in patterns:
                    for file_path in self.results_dir.glob(pattern):
                        if (
                            file_path.parent == self.results_dir
                        ):  # Only move files in root
                            dest = dir_path / file_path.name
                            file_path.rename(dest)
                            logger.debug(f"Moved {file_path.name} to {directory}/")

        except Exception as e:
            logger.error(f"Failed to organize output files: {str(e)}")
            raise FrustraEvoError("Output file organization failed") from e

    def _validate_results(self, results: List[PositionInformation]) -> bool:
        """
        Validate calculation results.

        Args:
            results: List of position information results

        Returns:
            bool: True if results are valid

        Raises:
            FrustraEvoError: If validation fails
        """
        try:
            logger.info("Validating calculation results")

            if not results:
                logger.error("No results to validate")
                return False

            # Check for required files
            required_files = {
                self.data_dir / "FrustrationData.csv",
                self.data_dir / "InformationContent.csv",
                self.plots_dir / "summary_analysis.png",
            }

            missing_files = [f for f in required_files if not f.exists()]
            if missing_files:
                logger.error(f"Missing required files: {missing_files}")
                return False

            # Validate result values
            for result in results:
                # Check probability values
                probs = [result.min_percent, result.neu_percent, result.max_percent]
                if not all(0 <= p <= 1 for p in probs):
                    logger.error(f"Invalid probabilities at position {result.position}")
                    return False

                # Check information content values
                if result.ic_total < 0:
                    logger.error(f"Negative IC at position {result.position}")
                    return False

                # Check state assignments
                if result.frust_state not in {"MIN", "NEU", "MAX", "UNK"}:
                    logger.error(f"Invalid state at position {result.position}")
                    return False

            logger.info("Results validation passed")
            return True

        except Exception as e:
            logger.error(f"Failed to validate results: {str(e)}")
            raise FrustraEvoError("Results validation failed") from e
