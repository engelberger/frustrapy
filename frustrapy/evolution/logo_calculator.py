from dataclasses import dataclass
from typing import Optional, Dict, List, Union, Set, Tuple
import os
import logging
from pathlib import Path
import subprocess
from ..core import Pdb
from ..utils.decorators import log_execution_time
from ..utils.helpers import pdb_equivalences
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np
import shutil
from ..analysis.frustration import calculate_frustration
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from .information_content import InformationContentCalculator
from .data_classes import MSAData, PositionInformation
from .sequence_logo import SequenceLogoGenerator
from .contacts import ContactAnalyzer
from .generator import HistogramGenerator
from .exceptions import MSAError, PDBError, CalculationError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class LogoCalculator:
    """Handles calculation of frustration logos for protein families"""

    job_id: str
    fasta_file: Path
    reference_pdb: Optional[str] = None
    pdb_dir: Optional[Path] = None
    contact_maps: bool = False
    results_dir: Optional[Path] = None
    mode: str = "configurational"

    def __post_init__(self):
        """Initialize paths and validate inputs"""
        # Convert all paths to absolute
        self.fasta_file = Path(self.fasta_file).absolute()
        if self.pdb_dir:
            self.pdb_dir = Path(self.pdb_dir).absolute()
        if self.results_dir:
            self.results_dir = Path(self.results_dir).absolute()
            self.job_dir = self.results_dir
        else:
            self.job_dir = Path(f"FrustraEvo_{self.job_id}").absolute()
            self.results_dir = self.job_dir / "OutPutFiles"

        # Initialize msa_data as None
        self.msa_data = None

        # Validate mode
        valid_modes = {"singleresidue", "mutational", "configurational"}
        if self.mode.lower() not in valid_modes:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of {valid_modes}")
        self.mode = self.mode.lower()

        # Create directories
        self.equivalences_dir = self.job_dir / "Equivalences"
        self.frustration_dir = self.job_dir / "Frustration"

        for directory in [
            self.job_dir,
            self.results_dir,
            self.equivalences_dir,
            self.frustration_dir,
        ]:
            directory.mkdir(exist_ok=True, parents=True)

        # Validate inputs
        if not self.fasta_file.exists():
            raise FileNotFoundError(f"FASTA file not found: {self.fasta_file}")

        if self.pdb_dir and not self.pdb_dir.exists():
            raise FileNotFoundError(f"PDB directory not found: {self.pdb_dir}")

        logger.debug(f"Initialized with absolute paths:")
        logger.debug(f"Job directory: {self.job_dir}")
        logger.debug(f"Results directory: {self.results_dir}")
        logger.debug(f"FASTA file: {self.fasta_file}")
        logger.debug(f"Mode: {self.mode}")
        if self.pdb_dir:
            logger.debug(f"PDB directory: {self.pdb_dir}")

    @log_execution_time
    def calculate(self) -> None:
        """Run the full logo calculation pipeline"""
        logger.info("Starting frustration logo calculation")

        try:
            # Generate PDB list from FASTA
            pdb_list = self._generate_pdb_list()

            # Process MSA and check sequences
            self._process_msa()
            self._check_sequences(pdb_list)

            # Create MSAData object for IC calculation
            self.msa_data = MSAData(
                sequences=[],
                identifiers=[],
                length=0,
                num_sequences=0,
                reference_index=None,
                fasta_file=self.job_dir / "MSA_Clean_final.fasta",
            )

            # Read MSA
            with open(self.msa_data.fasta_file) as f:
                for record in SeqIO.parse(f, "fasta"):
                    self.msa_data.sequences.append(str(record.seq))
                    self.msa_data.identifiers.append(record.id)

            self.msa_data.length = len(self.msa_data.sequences[0])
            self.msa_data.num_sequences = len(self.msa_data.sequences)

            if self.reference_pdb:
                try:
                    self.msa_data.reference_index = self.msa_data.identifiers.index(
                        self.reference_pdb
                    )
                except ValueError:
                    logger.warning(
                        f"Reference PDB {self.reference_pdb} not found in MSA"
                    )

            # Calculate frustration
            self._calculate_frustration(pdb_list)
            self._aggregate_frustration_data()

            # Initialize IC calculator
            ic_calculator = InformationContentCalculator(
                msa_data=self.msa_data,
                results_dir=self.results_dir,
                reference_pdb=self.reference_pdb,
                pdb_dir=self.pdb_dir,
                mode=self.mode,
            )

            # Calculate frustration and information content
            ic_results = ic_calculator.calculate()

            # Generate sequence logo using IC results
            self._generate_logo(ic_results=ic_results)

            # Optional: Generate contact maps
            if self.contact_maps:
                self._generate_contact_maps(ic_results=ic_results)

            # Clean up temporary files
            self._cleanup()

        except Exception as e:
            logger.error(f"Logo calculation failed: {str(e)}")
            raise

    def _generate_pdb_list(self) -> Path:
        """Generate list of PDBs from FASTA file"""
        logger.debug(f"Generating PDB list from {self.fasta_file}")

        try:
            # Create output file path
            list_file = self.job_dir / f"{self.fasta_file.stem}.list"
            pdb_ids: Set[str] = set()

            # Parse FASTA file and extract PDB IDs
            with open(self.fasta_file) as msa:
                for record in SeqIO.parse(msa, "fasta"):
                    pdb_id = record.id.lstrip(">")  # Remove '>' if present
                    pdb_ids.add(pdb_id)

            # Write PDB IDs to list file
            with open(list_file, "w") as out:
                for pdb_id in sorted(pdb_ids):
                    out.write(f"{pdb_id}\n")

            logger.info(f"Generated PDB list with {len(pdb_ids)} entries")
            return list_file

        except Exception as e:
            logger.error(f"Failed to generate PDB list: {str(e)}")
            raise

    def _process_msa(self) -> None:
        """Process multiple sequence alignment"""
        logger.debug("Processing multiple sequence alignment")

        try:
            # Output files
            clean_msa = self.job_dir / "MSA_Clean.fasta"
            aux_msa = self.job_dir / "MSA_Clean_aux.fasta"

            # Read and process sequences
            sequences: List[SeqRecord] = []
            with open(self.fasta_file) as f:
                for record in SeqIO.parse(f, "fasta"):
                    # Clean sequence
                    cleaned_seq = self._clean_sequence(str(record.seq))
                    new_record = SeqRecord(
                        Seq(cleaned_seq), id=record.id, description=""
                    )
                    sequences.append(new_record)

            # Write cleaned MSA files
            SeqIO.write(sequences, clean_msa, "fasta")
            SeqIO.write(sequences, aux_msa, "fasta")

            logger.info(f"Processed {len(sequences)} sequences")

        except Exception as e:
            logger.error(f"Failed to process MSA: {str(e)}")
            raise

    def _clean_sequence(self, sequence: str) -> str:
        """Clean and standardize a sequence"""
        # Replace unknown residues with gaps
        sequence = sequence.upper().replace("X", "-")

        # Validate amino acid characters
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY-")
        invalid_chars = set(sequence) - valid_aas
        if invalid_chars:
            logger.warning(f"Found invalid characters in sequence: {invalid_chars}")
            # Replace invalid characters with gaps
            for char in invalid_chars:
                sequence = sequence.replace(char, "-")

        return sequence

    def _get_alignment_length(self) -> int:
        """Get length of the multiple sequence alignment"""
        with open(self.job_dir / "MSA_Clean.fasta") as f:
            for record in SeqIO.parse(f, "fasta"):
                return len(record.seq)
        raise ValueError("No sequences found in MSA")

    def _check_sequences(self, pdb_list: Path) -> None:
        """
        Check sequence consistency between MSA and PDB structures.

        Args:
            pdb_list: Path to file containing list of PDB IDs

        Raises:
            MSAError: If sequence validation fails
            PDBError: If PDB sequence extraction fails
        """
        logger.debug("Checking sequence consistency")

        try:
            # Read PDB list
            with open(pdb_list) as f:
                pdb_ids = [line.strip() for line in f]

            # Read MSA
            msa_sequences: Dict[str, str] = {}
            clean_msa = self.job_dir / "MSA_Clean.fasta"
            with open(clean_msa) as f:
                for record in SeqIO.parse(f, "fasta"):
                    msa_sequences[record.id] = str(record.seq).replace("-", "")

            # Output files
            error_log = self.job_dir / "ErrorSeq.log"
            final_msa = self.job_dir / "MSA_Clean_final.fasta"

            valid_sequences: List[Tuple[str, str]] = []
            with open(error_log, "w") as log:
                for pdb_id in pdb_ids:
                    if pdb_id not in msa_sequences:
                        log.write(f"PDB {pdb_id} not found in MSA\n")
                        continue

                    try:
                        pdb_seq = self._get_pdb_sequence(pdb_id)
                        msa_seq = msa_sequences[pdb_id]

                        if pdb_seq == msa_seq:
                            valid_sequences.append((pdb_id, msa_sequences[pdb_id]))
                            logger.debug(f"Sequence {pdb_id} checked successfully")
                        else:
                            log.write(f"Sequence mismatch for {pdb_id}\n")
                            logger.warning(f"Sequence mismatch for {pdb_id}")
                    except Exception as e:
                        log.write(f"Error processing {pdb_id}: {str(e)}\n")
                        logger.error(f"Error processing {pdb_id}: {str(e)}")

            # Write final MSA with only valid sequences
            with open(final_msa, "w") as f:
                for pdb_id, seq in valid_sequences:
                    f.write(f">{pdb_id}\n{seq}\n")

            if not valid_sequences:
                raise MSAError("No valid sequences found after consistency check")

            logger.info(
                f"Validated {len(valid_sequences)} sequences out of {len(pdb_ids)}"
            )

        except Exception as e:
            logger.error(f"Failed to check sequences: {str(e)}")
            raise MSAError(f"Failed to check sequences: {str(e)}")

    def _get_pdb_sequence(self, pdb_id: str) -> str:
        """
        Extract sequence from PDB file.

        Args:
            pdb_id: PDB identifier

        Returns:
            str: Amino acid sequence from PDB

        Raises:
            PDBError: If PDB file is invalid or missing
        """
        # Define amino acid mapping
        aa_map = {
            "ALA": "A",
            "CYS": "C",
            "ASP": "D",
            "GLU": "E",
            "PHE": "F",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LYS": "K",
            "LEU": "L",
            "MET": "M",
            "ASN": "N",
            "PRO": "P",
            "GLN": "Q",
            "ARG": "R",
            "SER": "S",
            "THR": "T",
            "VAL": "V",
            "TRP": "W",
            "TYR": "Y",
            "MSE": "M",
            "HSE": "H",
            "HSD": "H",
            "HSP": "H",
            "HID": "H",
            "HIE": "H",
            "HIP": "H",
        }

        try:
            # Get PDB file path
            if self.pdb_dir:
                pdb_path = self.pdb_dir / f"{pdb_id}.pdb"
            else:
                pdb_path = self.frustration_dir / f"{pdb_id}.pdb"

            if not pdb_path.exists():
                raise PDBError(f"PDB file not found: {pdb_path}")

            sequence = []
            prev_res_num = None

            with open(pdb_path) as f:
                for line in f:
                    if line.startswith("ATOM"):
                        res_name = line[17:20].strip()
                        res_num = int(line[22:26])
                        chain_id = line[21]

                        # Only process if it's a new residue
                        if res_num != prev_res_num:
                            if res_name in aa_map:
                                sequence.append(aa_map[res_name])
                            else:
                                logger.warning(
                                    f"Unknown residue {res_name} in {pdb_id}"
                                )
                            prev_res_num = res_num

            if not sequence:
                raise PDBError(f"No valid residues found in {pdb_id}")

            return "".join(sequence)

        except Exception as e:
            logger.error(f"Failed to get sequence from PDB {pdb_id}: {str(e)}")
            raise PDBError(f"Failed to get sequence from PDB {pdb_id}: {str(e)}")

    def _calculate_frustration(
        self, pdb_list: Path, parallel: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate frustration for all PDBs.

        Args:
            pdb_list: Path to file containing list of PDB IDs
            parallel: Whether to run calculations in parallel (default: False for debugging)

        Returns:
            Dictionary mapping PDB IDs to their frustration data

        Raises:
            CalculationError: If frustration calculation fails
        """
        logger.info("Calculating frustration for PDB structures")
        all_frustration_data = {}

        try:
            # Read PDB list
            with open(pdb_list) as f:
                pdb_ids = [line.strip() for line in f]

            # Create progress bar
            pbar = tqdm(total=len(pdb_ids), desc="Calculating frustration")

            if parallel:
                # Parallel execution
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for pdb_id in pdb_ids:
                        future = executor.submit(
                            self._calculate_single_frustration, pdb_id
                        )
                        futures.append((pdb_id, future))

                    # Process results as they complete
                    for pdb_id, future in futures:
                        try:
                            frust_data, _ = future.result()
                            if frust_data is not None:
                                all_frustration_data[pdb_id] = frust_data
                            logger.debug(
                                f"Completed frustration calculation for {pdb_id}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to calculate frustration for {pdb_id}: {str(e)}"
                            )
                        finally:
                            pbar.update(1)
            else:
                # Serial execution for debugging
                for pdb_id in pdb_ids:
                    try:
                        frust_data, _ = self._calculate_single_frustration(pdb_id)
                        if frust_data is not None:
                            all_frustration_data[pdb_id] = frust_data
                        logger.debug(f"Completed frustration calculation for {pdb_id}")
                    except Exception as e:
                        logger.error(
                            f"Failed to calculate frustration for {pdb_id}: {str(e)}"
                        )
                    finally:
                        pbar.update(1)

            pbar.close()
            logger.info(
                f"Completed frustration calculations for {len(pdb_ids)} structures"
            )

            # Save combined frustration data
            if all_frustration_data:
                combined_data = pd.concat(
                    all_frustration_data.values(), ignore_index=True
                )
                output_file = self.results_dir / "FrustrationData.csv"
                combined_data.to_csv(output_file, index=False)
                logger.info(f"Saved combined frustration data to {output_file}")
            else:
                raise CalculationError(
                    "No frustration data was generated for any structure"
                )

            return all_frustration_data

        except Exception as e:
            logger.error(f"Failed to calculate frustration: {str(e)}")
            raise CalculationError(f"Failed to calculate frustration: {str(e)}")

    def _calculate_single_frustration(
        self, pdb_id: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        Calculate frustration for a single PDB structure.

        Args:
            pdb_id: PDB identifier

        Returns:
            Tuple containing:
            - DataFrame with frustration data
            - Dictionary with density results

        Raises:
            CalculationError: If calculation fails
            PDBError: If PDB file is invalid
        """
        try:
            # Get PDB file path
            if self.pdb_dir:
                pdb_file = self.pdb_dir / f"{pdb_id}.pdb"
            else:
                pdb_file = self.frustration_dir / f"{pdb_id}.pdb"

            if not pdb_file.exists():
                raise PDBError(f"PDB file not found: {pdb_file}")

            logger.debug(f"Calculating frustration for {pdb_id}")

            # Create results directory without duplication
            results_dir = self.frustration_dir / f"{pdb_id}.done"

            # Check if pre-calculated data exists
            frust_file = (
                results_dir / "FrustrationData" / f"{pdb_id}.pdb_configurational"
            )
            if frust_file.exists():
                logger.debug(f"Using pre-calculated data from {frust_file}")
                frust_data = pd.read_csv(frust_file, sep="\s+")
                frust_data["structure"] = pdb_id
                return frust_data, None

            # Calculate frustration
            pdb, plots, density_results, _ = calculate_frustration(
                pdb_file=str(pdb_file),
                mode="configurational",
                results_dir=str(results_dir),
                graphics=False,
                visualization=False,
            )

            # Fix duplicated directory if it was created
            duplicated_dir = results_dir / f"{pdb_id}.done"
            if duplicated_dir.exists():
                # Move contents up one level
                for item in duplicated_dir.iterdir():
                    target = results_dir / item.name
                    if target.exists():
                        if target.is_dir():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                    shutil.move(str(item), str(results_dir))
                # Remove duplicated directory
                shutil.rmtree(duplicated_dir)

            # Read frustration data from the correct path
            frust_file = (
                results_dir / "FrustrationData" / f"{pdb_id}.pdb_configurational"
            )
            if frust_file.exists():
                frust_data = pd.read_csv(frust_file, sep="\s+")
                frust_data["structure"] = pdb_id
                logger.debug(f"Successfully read frustration data from {frust_file}")
                return frust_data, density_results
            else:
                logger.warning(f"Frustration data file not found at {frust_file}")
                logger.debug(f"Directory contents: {list(results_dir.glob('**/*'))}")
                return None, density_results

        except Exception as e:
            logger.error(f"Failed to calculate frustration for {pdb_id}: {str(e)}")
            raise CalculationError(
                f"Failed to calculate frustration for {pdb_id}: {str(e)}"
            )

    def _generate_logo(self, ic_results: List[PositionInformation]) -> None:
        """
        Generate sequence logo with frustration information.

        Args:
            ic_results: List of position-specific information content results
        """
        try:
            logger.debug("Generating sequence logo")

            # Initialize logo generator with IC results
            logo_generator = SequenceLogoGenerator(
                msa_data=self.msa_data,
                results_dir=self.results_dir,
                reference_pdb=self.reference_pdb,
            )

            # Generate logo using IC results
            logo_generator.generate_logo(ic_results=ic_results)
            logger.info("Successfully generated sequence logo")

        except Exception as e:
            logger.error(f"Failed to generate logo: {str(e)}")
            raise CalculationError(f"Failed to generate logo: {str(e)}")

    def _generate_contact_maps(self, ic_results: List[PositionInformation]) -> None:
        """
        Generate contact map visualizations.

        Args:
            ic_results: List of position-specific information content results
        """
        try:
            logger.debug("Generating contact maps")

            # Initialize contact analyzer
            contact_analyzer = ContactAnalyzer(
                results_dir=self.results_dir,
                mode=self.mode,
                frustration_dir=self.frustration_dir,
            )

            # Analyze contacts using IC results
            contacts = contact_analyzer.analyze_contacts(
                alignment_length=self.msa_data.length, ic_results=ic_results
            )

            # Generate contact maps
            histogram_generator = HistogramGenerator(
                results_dir=self.results_dir,
                msa_file=self.job_dir / "MSA_Clean_final.fasta",
                reference_pdb=self.reference_pdb,
            )

            histogram_generator.generate_contact_maps(
                contacts=contacts, ic_results=ic_results
            )

            logger.info("Successfully generated contact maps")

        except Exception as e:
            logger.error(f"Failed to generate contact maps: {str(e)}")
            raise CalculationError(f"Failed to generate contact maps: {str(e)}")

    def _aggregate_frustration_data(self) -> None:
        """
        Aggregate frustration data from all structures into a single CSV file.

        This method:
        1. Collects frustration data from individual structures
        2. Combines them into a single DataFrame
        3. Saves the result as FrustrationData.csv

        Raises:
            CalculationError: If data aggregation fails
        """
        logger.debug("Aggregating frustration data")

        try:
            # Create data directory
            data_dir = self.results_dir / "data"
            data_dir.mkdir(exist_ok=True)

            # Output file
            output_file = data_dir / "FrustrationData.csv"

            # Collect data from all structures
            all_data = []

            # Read PDB list
            pdb_list = self.job_dir / f"{self.fasta_file.stem}.list"
            with open(pdb_list) as f:
                pdb_ids = [line.strip() for line in f]

            for pdb_id in pdb_ids:
                try:
                    # Corrected path to frustration data
                    frust_file = (
                        self.frustration_dir
                        / f"{pdb_id}.done/FrustrationData/{pdb_id}.pdb_configurational"
                    )

                    logger.debug(f"Looking for frustration data at: {frust_file}")

                    if frust_file.exists():
                        # Read data
                        df = pd.read_csv(frust_file, sep="\s+")
                        # Add structure identifier
                        df["structure"] = pdb_id
                        all_data.append(df)
                        logger.debug(f"Found and loaded frustration data for {pdb_id}")
                    else:
                        logger.warning(
                            f"No frustration data found for {pdb_id} at {frust_file}"
                        )

                except Exception as e:
                    logger.error(f"Failed to process data for {pdb_id}: {str(e)}")

            if not all_data:
                logger.error("No frustration data found for any structure")
                logger.debug(
                    f"Frustration directory contents: {list(self.frustration_dir.glob('**/*'))}"
                )
                raise CalculationError("No frustration data found for any structure")

            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)

            # Save combined data
            combined_data.to_csv(output_file, index=False)
            logger.info(f"Saved combined frustration data to {output_file}")

        except Exception as e:
            logger.error(f"Failed to aggregate frustration data: {str(e)}")
            raise CalculationError(f"Failed to aggregate frustration data: {str(e)}")

    def _cleanup(self) -> None:
        """
        Clean up temporary files and directories.

        This method:
        1. Removes temporary calculation files
        2. Organizes output files
        3. Preserves important results
        """
        try:
            logger.debug("Starting cleanup")

            # List of temporary files to remove
            temp_files = [
                "MSA_Clean_aux.fasta",
                "ErrorSeq.log",
            ]

            # Remove temporary files
            for temp_file in temp_files:
                file_path = self.job_dir / temp_file
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Removed temporary file: {temp_file}")

            # List of directories to clean if empty
            temp_dirs = [
                "tmp",
                "Equivalences",
                "Frustration",
            ]

            # Remove empty directories
            for temp_dir in temp_dirs:
                dir_path = self.job_dir / temp_dir
                if dir_path.exists() and not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    logger.debug(f"Removed empty directory: {temp_dir}")

            # Organize output files
            output_dirs = {
                "data": ["*.csv", "*.txt"],
                "plots": ["*.png", "*.pdf"],
                "logs": ["*.log"],
            }

            for dir_name, patterns in output_dirs.items():
                output_dir = self.results_dir / dir_name
                output_dir.mkdir(exist_ok=True)

                # Move files matching patterns
                for pattern in patterns:
                    for file_path in self.results_dir.glob(pattern):
                        if file_path.is_file():
                            target = output_dir / file_path.name
                            shutil.move(str(file_path), str(target))
                            logger.debug(f"Moved {file_path.name} to {dir_name}/")

            logger.info("Cleanup completed successfully")

        except Exception as e:
            logger.error(f"Failed to clean up: {str(e)}")
            # Don't raise error since cleanup is not critical
            pass
