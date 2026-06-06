from dataclasses import dataclass
from typing import Dict, Optional, Any, List
import numpy as np


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


@dataclass
class FrustrationDensity:
    """Data class to store frustration density analysis results for a residue"""

    residue_number: int
    chain_id: str
    total_density: int
    highly_frustrated: int
    neutrally_frustrated: int
    minimally_frustrated: int
    rel_highly_frustrated: float
    rel_neutrally_frustrated: float
    rel_minimally_frustrated: float


@dataclass
class FrustrationDensityResults:
    """Container for all residue frustration density results"""

    densities: List[FrustrationDensity]
    contact_coordinates: np.ndarray  # Nx3 array of contact point coordinates
    frustration_values: np.ndarray  # N-length array of frustration values
