from dataclasses import dataclass
from typing import Dict, Optional, Any


@dataclass
class SingleResidueData:
    """Data class to store single residue frustration analysis results"""

    residue_number: int
    chain_id: str
    residue_name: str
    mutations: Dict[str, float]  # Maps mutation (e.g. 'ALA') to frustration index
    native_energy: float
    decoy_energy: Optional[float]
    sd_energy: Optional[float]
    density: Optional[float]
    plots: Optional[Any] = None  # Store associated plot if available
