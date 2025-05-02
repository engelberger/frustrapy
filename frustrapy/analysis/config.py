from dataclasses import dataclass
from typing import Optional, Union, List, Dict
from .exceptions import ValidationError

@dataclass
class FrustrationConfig:
    """Configuration parameters for FrustrationCalculator."""
    pdb_file: Optional[str] = None
    pdb_id: Optional[str] = None
    chain: Optional[Union[str, List[str]]] = None
    residues: Optional[Dict[str, List[int]]] = None
    electrostatics_k: Optional[float] = None
    seq_dist: int = 12
    mode: str = "configurational"
    graphics: bool = True
    visualization: bool = True
    results_dir: Optional[str] = None
    debug: bool = False
    n_cpus: Optional[int] = None
    overwrite: bool = False  # If True, overwrite existing intermediate files; caution mode if False

    def __post_init__(self):
        # Validate mutually exclusive pdb_file/pdb_id
        if self.pdb_file is None and self.pdb_id is None:
            raise ValidationError("You must indicate PdbID or PdbFile!", param_name="pdb_file/pdb_id")
        # Validate electrostatics_k
        if self.electrostatics_k is not None and not isinstance(self.electrostatics_k, (int, float)):
            raise ValidationError("Electrostatic_K must be a numeric value!", param_name="electrostatics_k", value=self.electrostatics_k)
        # Validate seq_dist
        if self.seq_dist not in (3, 12):
            raise ValidationError("SeqDist must take the value 3 or 12!", param_name="seq_dist", value=self.seq_dist)
        # Validate mode
        if self.mode.lower() not in ("configurational", "mutational", "singleresidue"):
            raise ValidationError(f"{self.mode} frustration index doesn't exist!", param_name="mode", value=self.mode)
        # Validate boolean flags
        if not isinstance(self.graphics, bool):
            raise ValidationError("Graphics must be a boolean value!", param_name="graphics", value=self.graphics)
        if not isinstance(self.visualization, bool):
            raise ValidationError("Visualization must be a boolean value!", param_name="visualization", value=self.visualization)
        # Validate overwrite flag
        if not isinstance(self.overwrite, bool):
            raise ValidationError("Overwrite must be a boolean value!", param_name="overwrite", value=self.overwrite)
        # Validate n_cpus
        if self.n_cpus is not None and (not isinstance(self.n_cpus, int) or self.n_cpus <= 0):
            raise ValidationError("n_cpus must be a positive integer or None", param_name="n_cpus", value=self.n_cpus) 