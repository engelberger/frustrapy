from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union, Tuple
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from .exceptions import FrustraEvoError
import shutil
from Bio import SeqIO
from frustrapy import calculate_frustration  # Import here to avoid circular imports

logger = logging.getLogger(__name__)


@dataclass
class ResidueEquivalence:
    """Maps MSA positions to PDB residue numbers"""

    msa_pos: int
    pdb_pos: int
    residue: str
    chain: str
    structure: str


class ContactMatrix:
    """Stores frustration contacts for a single structure"""

    def __init__(self, size: int, structure_id: str):
        self.size = size
        self.structure_id = structure_id
        # Initialize with -100 as in legacy code
        self.matrix = np.full((size + 2, size + 2), -100.0)
        self.equivalences: Dict[int, ResidueEquivalence] = {}

    def add_contact(
        self, msa_pos1: int, msa_pos2: int, frustration_value: float
    ) -> None:
        """Add a contact between two MSA positions"""
        self.matrix[msa_pos1][msa_pos2] = frustration_value

    def get_contact(self, msa_pos1: int, msa_pos2: int) -> Optional[float]:
        """Get frustration value for a contact if it exists"""
        value = self.matrix[msa_pos1][msa_pos2]
        return None if value == -100.0 else value


class InformationContentCalculator:
    """Calculates sequence and frustration information content"""

    FRUSTRATION_CUTOFFS = {
        "MIN": 0.78,  # Minimally frustrated cutoff
        "MAX": -1.0,  # Maximally frustrated cutoff
    }

    REQUIRED_DIRECTORIES: Set[str] = {
        "data",
        "plots",
        "equivalences",
        "Frustration",
        "pdbs",
        "msa",
        "logs",
    }

    def __init__(
        self,
        msa_data: "MSAData",
        results_dir: Path,
        reference_pdb: str,
        pdb_dir: Optional[Path] = None,
        mode: str = "configurational",
    ):
        self.msa_data = msa_data
        self.results_dir = Path(results_dir)
        self.reference_pdb = reference_pdb
        self.mode = mode
        self.pdb_dir = pdb_dir

        # Initialize paths
        self.frustration_dir = self.results_dir / "Frustration"
        self.equivalences_dir = self.results_dir / "equivalences"
        self.msa_dir = self.results_dir / "msa"
        self.data_dir = self.results_dir / "data"
        self.logs_dir = self.results_dir / "logs"
        self.pdb_dest_dir = self.results_dir / "pdbs"

    def _setup_directories(self) -> None:
        """Create required directory structure"""
        try:
            for directory in self.REQUIRED_DIRECTORIES:
                dir_path = self.results_dir / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {dir_path}")
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            raise FrustraEvoError("Directory setup failed") from e

    def _copy_required_files(self) -> None:
        """Copy input files to working directories"""
        try:
            # Copy PDB files
            if self.pdb_dir and self.pdb_dir.exists():
                for pdb_file in self.pdb_dir.glob("*.pdb"):
                    dest = self.pdb_dest_dir / pdb_file.name
                    shutil.copy2(pdb_file, dest)
                    logger.debug(f"Copied PDB file: {pdb_file.name}")

            # Copy MSA file if available
            if self.msa_data.fasta_file:
                dest = self.msa_dir / self.msa_data.fasta_file.name
                shutil.copy2(self.msa_data.fasta_file, dest)
                logger.debug(f"Copied MSA file: {self.msa_data.fasta_file.name}")

        except Exception as e:
            logger.error(f"Failed to copy files: {e}")
            raise FrustraEvoError("File copying failed") from e

    def _validate_sequences(self) -> None:
        """Validate sequences against PDB structures"""
        try:
            clean_msa = self.msa_dir / "MSA_Clean.fasta"
            error_log = self.logs_dir / "ErrorSeq.log"

            valid_sequences = []
            with error_log.open("w") as out_log:
                for record in SeqIO.parse(self.msa_data.fasta_file, "fasta"):
                    seq_id = record.id
                    sequence = str(record.seq).replace("-", "")

                    # Check PDB existence
                    pdb_file = self.pdb_dest_dir / f"{seq_id}.pdb"
                    if not pdb_file.exists():
                        out_log.write(f"Missing PDB file: {seq_id}\n")
                        continue

                    # Compare sequences
                    pdb_sequence = self._get_pdb_sequence(pdb_file)
                    if sequence == pdb_sequence:
                        valid_sequences.append(record)
                    else:
                        out_log.write(f"Sequence mismatch for {seq_id}\n")

            # Write validated sequences
            with clean_msa.open("w") as out_msa:
                SeqIO.write(valid_sequences, out_msa, "fasta")

        except Exception as e:
            logger.error(f"Failed to validate sequences: {e}")
            raise FrustraEvoError("Sequence validation failed") from e

    def _get_pdb_sequence(self, pdb_file: Path) -> str:
        """Extract sequence from PDB file"""
        AA_CODES = {
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

        with pdb_file.open() as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    res_name = line[17:20].strip()
                    res_num = int(line[22:26])

                    if res_num != prev_res_num and res_name in AA_CODES:
                        sequence.append(AA_CODES[res_name])
                        prev_res_num = res_num

        return "".join(sequence)

    def _load_equivalences(self, structure_id: str) -> Dict[int, ResidueEquivalence]:
        """Load residue equivalences for a structure"""
        equiv_file = self.equivalences_dir / f"Equival_{structure_id}.txt"
        equivalences = {}

        try:
            with equiv_file.open() as f:
                next(f)  # Skip header
                for line in f:
                    fields = line.strip().split()
                    equiv = ResidueEquivalence(
                        msa_pos=int(fields[0]),
                        pdb_pos=int(fields[1]),
                        residue=fields[2],
                        chain=fields[3],
                        structure=fields[4],
                    )
                    equivalences[equiv.msa_pos] = equiv

            return equivalences

        except Exception as e:
            logger.error(f"Failed to load equivalences for {structure_id}: {e}")
            raise FrustraEvoError(f"Equivalence loading failed: {str(e)}")

    def _load_contact_matrices(self) -> List[ContactMatrix]:
        """Load frustration contact matrices for all structures using legacy approach"""
        matrices = []
        total_structures = len(self.msa_data.identifiers)
        logger.info(f"Loading {total_structures} contact matrices")

        for structure_id in self.msa_data.identifiers:
            try:
                # Debug structure processing
                logger.debug(f"Processing structure: {structure_id}")

                # 1. Load PDB-MSA position mapping exactly like legacy
                equiv_map = {}  # MSA -> PDB mapping
                rev_equiv_map = {}  # PDB -> MSA mapping
                with (
                    self.equivalences_dir / f"Equival_{structure_id}.txt"
                ).open() as f:
                    next(f)  # Skip header
                    for line in f:
                        fields = line.strip().split("\t")
                        msa_pos, pdb_pos = int(fields[0]), int(fields[1])
                        equiv_map[msa_pos] = pdb_pos
                        rev_equiv_map[pdb_pos] = msa_pos

                logger.debug(f"Loaded {len(equiv_map)} position mappings")

                # 2. Initialize matrix with exact legacy dimensions
                matrix = ContactMatrix(self.msa_data.length, structure_id)
                matrix.matrix.fill(-100.0)  # Legacy uses -100 as null value

                # 3. Calculate frustration using FrustraPy
                pdb_file = self.pdb_dest_dir / f"{structure_id}.pdb"
                pdb, plots, density_results, _ = calculate_frustration(
                    pdb_file=str(pdb_file),
                    mode=self.mode,
                    results_dir=str(self.frustration_dir),
                    debug=True,
                )

                # 4. Load frustration data using legacy column indices
                frust_file = (
                    self.frustration_dir
                    / f"{structure_id}.done/FrustrationData/{structure_id}.pdb_{self.mode}"
                )

                contact_count = 0
                with frust_file.open() as f:
                    next(f)  # Skip header
                    for line in f:
                        try:
                            fields = line.strip().split()
                            pdb_pos1, pdb_pos2 = int(fields[0]), int(fields[1])
                            frust_value = float(fields[9])  # FrstIndex column

                            # Only process if both positions are mapped
                            if pdb_pos1 in rev_equiv_map and pdb_pos2 in rev_equiv_map:
                                msa_pos1 = rev_equiv_map[pdb_pos1]
                                msa_pos2 = rev_equiv_map[pdb_pos2]

                                # Add contact both ways like legacy
                                matrix.add_contact(msa_pos1, msa_pos2, frust_value)
                                matrix.add_contact(msa_pos2, msa_pos1, frust_value)
                                contact_count += 1

                                logger.debug(
                                    f"Added symmetric contact {msa_pos1}-{msa_pos2} "
                                    f"(PDB: {pdb_pos1}-{pdb_pos2}) = {frust_value}"
                                )

                        except (ValueError, IndexError) as e:
                            logger.warning(
                                f"Invalid line in {structure_id} frustration file: "
                                f"{line.strip()} - {str(e)}"
                            )
                            continue

                # Store equivalences for later use
                matrix.equivalences = {
                    msa_pos: ResidueEquivalence(
                        msa_pos=msa_pos,
                        pdb_pos=pdb_pos,
                        residue=self._get_residue(structure_id, pdb_pos),
                        chain="A",  # Legacy assumes chain A
                        structure=structure_id,
                    )
                    for msa_pos, pdb_pos in equiv_map.items()
                }

                logger.info(
                    f"Processed {structure_id}: {contact_count} contacts, "
                    f"{len(matrix.equivalences)} positions"
                )
                matrices.append(matrix)

            except Exception as e:
                logger.error(
                    f"Failed to process {structure_id}: {str(e)}", exc_info=True
                )
                continue

        if not matrices:
            raise FrustraEvoError("No valid contact matrices could be loaded")

        logger.info(f"Successfully loaded {len(matrices)} contact matrices")
        return matrices

    def _get_residue(self, structure_id: str, pdb_pos: int) -> str:
        """Get residue type from PDB file at given position"""
        pdb_file = self.pdb_dest_dir / f"{structure_id}.pdb"
        with pdb_file.open() as f:
            for line in f:
                if line.startswith("ATOM") and line[22:26].strip() == str(pdb_pos):
                    return line[17:20].strip()
        return "UNK"

    def _calculate_frustration_state(self, value: float) -> str:
        """Determine frustration state based on cutoffs"""
        if value >= self.FRUSTRATION_CUTOFFS["MIN"]:
            return "MIN"
        elif value <= self.FRUSTRATION_CUTOFFS["MAX"]:
            return "MAX"
        return "NEU"

    def _calculate_contact_stats(self, values: List[float]) -> Dict:
        """Calculate statistics for a set of contact values"""
        if len(values) <= 1:
            return {}

        # Legacy uses fixed background entropy value
        h_background = 1.36096404744368

        # Count states
        states = [self._calculate_frustration_state(v) for v in values]
        counts = {"MIN": 0, "NEU": 0, "MAX": 0}
        for state in states:
            counts[state] += 1

        total = len(values)
        # Round probabilities to match legacy format
        probs = {state: round(count / total, 2) for state, count in counts.items()}

        # Calculate entropy terms exactly as legacy does
        h_terms = {
            state: round(self._calculate_entropy_term(prob), 15) if prob > 0 else 0.0
            for state, prob in probs.items()
        }
        h_total = round(sum(h_terms.values()), 15)

        # Calculate IC using legacy formula
        ic_total = round(max(0.0, h_background - h_total), 15)
        ic_terms = {
            state: round(ic_total * prob, 15) if prob > 0 else 0.0
            for state, prob in probs.items()
        }

        # Determine conserved state using legacy rules
        if counts["NEU"] >= max(counts["MIN"], counts["MAX"]):
            conserved_state = "NEU"
        elif counts["MIN"] >= counts["MAX"]:
            conserved_state = "MIN"
        else:
            conserved_state = "MAX"

        return {
            "counts": counts,
            "probabilities": probs,
            "entropy_terms": h_terms,
            "h_total": h_total,
            "ic_terms": ic_terms,
            "ic_total": ic_total,
            "conserved_state": conserved_state,
        }

    def _calculate_total_ic(self, h_total: float, sample_size: int) -> float:
        """Calculate total information content with background correction"""
        # Legacy uses fixed background entropy value
        h_background = 1.36096404744368

        # Small sample correction (not used in legacy)
        # correction = (3 - 1) / (2 * np.log(2) * sample_size)

        return round(max(0.0, h_background - h_total), 15)

    def calculate(self) -> pd.DataFrame:
        """Calculate information content for all contacts"""
        try:
            # Load matrices
            matrices = self._load_contact_matrices()
            if not matrices:
                raise FrustraEvoError("No valid contact matrices found")

            # Get reference PDB sequence
            ref_pdb_file = self.pdb_dest_dir / f"{self.reference_pdb}.pdb"
            ref_sequence = self._get_pdb_sequence(ref_pdb_file)

            # Process all contact pairs
            results = []
            for i in range(self.msa_data.length):
                for j in range(i + 1, self.msa_data.length):
                    # Collect values for this contact pair
                    values = []
                    for matrix in matrices:
                        if value := matrix.get_contact(i, j):
                            values.append(value)

                    # Only process if multiple contacts exist
                    if len(values) > 1:
                        stats = self._calculate_contact_stats(values)

                        # Get reference PDB information
                        ref_equiv = self._load_equivalences(self.reference_pdb)
                        ref_res1 = ref_equiv[i + 1]
                        ref_res2 = ref_equiv[j + 1]

                        # Format result row matching legacy output
                        results.append(
                            {
                                "Res1": ref_res1.pdb_pos,
                                "Res2": ref_res2.pdb_pos,
                                "AA1": ref_sequence[
                                    ref_res1.pdb_pos - 1
                                ],  # -1 for 0-based index
                                "AA2": ref_sequence[ref_res2.pdb_pos - 1],
                                "NumRes1_Ref": i + 1,
                                "Chain1_Ref": ref_res1.chain,
                                "NumRes2_Ref": j + 1,
                                "Chain2_Ref": ref_res2.chain,
                                "Prot_Ref": self.reference_pdb,
                                "NoContacts": len(values),
                                "FreqConts": len(values) / len(matrices),
                                "pNEU": stats["probabilities"]["NEU"],
                                "pMIN": stats["probabilities"]["MIN"],
                                "pMAX": stats["probabilities"]["MAX"],
                                "HNEU": stats["entropy_terms"]["NEU"],
                                "HMIN": stats["entropy_terms"]["MIN"],
                                "HMAX": stats["entropy_terms"]["MAX"],
                                "Htotal": stats["h_total"],
                                "ICNEU": stats["ic_terms"]["NEU"],
                                "ICMIN": stats["ic_terms"]["MIN"],
                                "ICMAX": stats["ic_terms"]["MAX"],
                                "ICtotal": stats["ic_total"],
                                "FstConserved": stats["conserved_state"],
                            }
                        )

            # Convert to DataFrame and save
            df = pd.DataFrame(results)

            # Ensure columns are in correct order
            column_order = [
                "Res1",
                "Res2",
                "AA1",
                "AA2",
                "NumRes1_Ref",
                "Chain1_Ref",
                "NumRes2_Ref",
                "Chain2_Ref",
                "Prot_Ref",
                "NoContacts",
                "FreqConts",
                "pNEU",
                "pMIN",
                "pMAX",
                "HNEU",
                "HMIN",
                "HMAX",
                "Htotal",
                "ICNEU",
                "ICMIN",
                "ICMAX",
                "ICtotal",
                "FstConserved",
            ]
            df = df[column_order]

            output_file = (
                self.results_dir
                / f"IC_{self.mode.capitalize()}_{self.reference_pdb}.csv"
            )
            df.to_csv(output_file, sep="\t", index=False)

            return df

        except Exception as e:
            logger.error(f"Calculation failed: {str(e)}")
            raise FrustraEvoError(f"Information content calculation failed: {str(e)}")

    def _calculate_entropy_term(self, probability: float) -> float:
        """Calculate Shannon entropy term"""
        if probability <= 0:
            return 0.0
        return -probability * np.log2(probability)

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

            equiv_dir = self.results_dir / "equivalences"
            combined_file = equiv_dir / "AllEquivalences.txt"

            if not equiv_dir.exists() or not any(equiv_dir.iterdir()):
                raise FileNotFoundError(f"No equivalence files found in: {equiv_dir}")

            # Combine and validate equivalence files
            valid_entries = []
            header_written = False

            for equiv_file in equiv_dir.glob("Equival_*.txt"):
                with equiv_file.open() as f:
                    header = next(f, None)  # Skip header line
                    if not header_written and header:
                        valid_entries.append(header)
                        header_written = True

                    for line in f:
                        fields = line.strip().split("\t")
                        if len(fields) > 4:  # Valid line with enough fields
                            try:
                                pos = int(fields[0])
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
            )

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

    def _calculate_equivalences(self) -> None:
        """
        Calculate and save residue equivalences for all structures.

        Raises:
            FrustraEvoError: If equivalence calculation fails
        """
        try:
            logger.info("Calculating residue equivalences")

            # Ensure equivalences directory exists
            equiv_dir = self.results_dir / "equivalences"
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
