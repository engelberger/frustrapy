from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path
import numpy as np
from ..core.data_classes import FrustrationDensity


@dataclass
class EvolutionaryFrustrationData:
    """Stores evolutionary frustration data for a position"""

    position: int
    residue_name: str
    residue_number: int
    chain_id: str
    conservation_score: float
    frustration_scores: Dict[str, float]  # MIN, MAX, NEU
    information_content: float
    contacts: List[int]  # List of contacting positions
    density: Optional[FrustrationDensity] = None


@dataclass
class MSAData:
    """Stores multiple sequence alignment data"""

    sequences: List[str]
    identifiers: List[str]
    length: int
    num_sequences: int
    reference_index: Optional[int] = None

    @property
    def conservation_matrix(self) -> np.ndarray:
        """Calculate position-specific conservation matrix"""
        matrix = np.zeros((20, self.length))  # 20 standard amino acids
        aa_map = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

        for seq in self.sequences:
            for pos, aa in enumerate(seq):
                if aa in aa_map:
                    matrix[aa_map[aa], pos] += 1

        # Normalize by number of sequences
        return matrix / self.num_sequences


@dataclass
class ContactData:
    """Stores contact information between positions"""

    position1: int
    position2: int
    distance: float
    frequency: float
    frustration_state: str
    information_content: float


@dataclass
class EvolutionaryAnalysisMetadata:
    """Stores metadata about the evolutionary analysis"""

    job_id: str
    reference_pdb: Optional[str]
    num_structures: int
    alignment_length: int
    analysis_date: str
    parameters: Dict[str, Union[str, float, bool]]


@dataclass
class PositionInformation:
    """Information content for a sequence position"""

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

    def __post_init__(self):
        """Validate fields after initialization"""
        # Ensure percentages sum to 1 (within floating point precision)
        total = self.min_percent + self.neu_percent + self.max_percent
        if not np.isclose(total, 1.0) and total > 0:
            logger.warning(
                f"Percentages don't sum to 1 for position {self.position}: {total}"
            )

        # Validate frustration state
        valid_states = {"MIN", "MAX", "NEU", "UNK"}
        if self.frust_state not in valid_states:
            logger.warning(
                f"Invalid frustration state for position {self.position}: {self.frust_state}"
            )
            self.frust_state = "UNK"

        if self.conserved_state not in valid_states:
            logger.warning(
                f"Invalid conserved state for position {self.position}: {self.conserved_state}"
            )
            self.conserved_state = "UNK"


class EvolutionaryDataManager:
    """Manages evolutionary analysis data"""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.metadata: Optional[EvolutionaryAnalysisMetadata] = None
        self.msa_data: Optional[MSAData] = None
        self.position_data: Dict[int, EvolutionaryFrustrationData] = {}
        self.contacts: List[ContactData] = []

    def load_data(self) -> None:
        """Load analysis data from results directory"""
        # Load metadata
        metadata_file = self.results_dir / "metadata.json"
        if metadata_file.exists():
            import json

            with open(metadata_file) as f:
                data = json.load(f)
                self.metadata = EvolutionaryAnalysisMetadata(**data)

        # Load MSA data
        msa_file = self.results_dir / "MSA_Clean_final.fasta"
        if msa_file.exists():
            from Bio import SeqIO

            sequences = []
            identifiers = []
            with open(msa_file) as f:
                for record in SeqIO.parse(f, "fasta"):
                    sequences.append(str(record.seq))
                    identifiers.append(record.id)

            self.msa_data = MSAData(
                sequences=sequences,
                identifiers=identifiers,
                length=len(sequences[0]),
                num_sequences=len(sequences),
                reference_index=(
                    identifiers.index(self.metadata.reference_pdb)
                    if self.metadata and self.metadata.reference_pdb in identifiers
                    else None
                ),
            )

        # Load position data
        data_file = self.results_dir / "FrustrationData.csv"
        if data_file.exists():
            import pandas as pd

            df = pd.read_csv(data_file)
            for _, row in df.iterrows():
                pos = int(row["Position"])
                self.position_data[pos] = EvolutionaryFrustrationData(
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

        # Load contact data
        contact_file = self.results_dir / "ContactData.csv"
        if contact_file.exists():
            import pandas as pd

            df = pd.read_csv(contact_file)
            for _, row in df.iterrows():
                contact = ContactData(
                    position1=int(row["Res1"]),
                    position2=int(row["Res2"]),
                    distance=float(row["Distance"]) if "Distance" in row else 0.0,
                    frequency=float(row["FreqConts"]),
                    frustration_state=row["FrstState"],
                    information_content=float(row["ICtotal"]),
                )
                self.contacts.append(contact)

                # Update position contacts
                if contact.position1 in self.position_data:
                    self.position_data[contact.position1].contacts.append(
                        contact.position2
                    )
                if contact.position2 in self.position_data:
                    self.position_data[contact.position2].contacts.append(
                        contact.position1
                    )

    def save_data(self) -> None:
        """Save analysis data to results directory"""
        # Save metadata
        if self.metadata:
            import json

            with open(self.results_dir / "metadata.json", "w") as f:
                json.dump(self.metadata.__dict__, f, indent=2)

        # Save position data
        if self.position_data:
            import pandas as pd

            data = []
            for pos, pos_data in self.position_data.items():
                data.append(
                    {
                        "Position": pos,
                        "AA_Ref": pos_data.residue_name,
                        "Num_Ref": pos_data.residue_number,
                        "Chain": pos_data.chain_id,
                        "Conservation": pos_data.conservation_score,
                        "Pct_Min": pos_data.frustration_scores["MIN"],
                        "Pct_Neu": pos_data.frustration_scores["NEU"],
                        "Pct_Max": pos_data.frustration_scores["MAX"],
                        "IC_Total": pos_data.information_content,
                        "Num_Contacts": len(pos_data.contacts),
                    }
                )
            df = pd.DataFrame(data)
            df.to_csv(self.results_dir / "FrustrationData.csv", index=False)

        # Save contact data
        if self.contacts:
            import pandas as pd

            data = [
                {
                    "Res1": c.position1,
                    "Res2": c.position2,
                    "Distance": c.distance,
                    "FreqConts": c.frequency,
                    "FrstState": c.frustration_state,
                    "ICtotal": c.information_content,
                }
                for c in self.contacts
            ]
            df = pd.DataFrame(data)
            df.to_csv(self.results_dir / "ContactData.csv", index=False)
