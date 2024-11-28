from typing import Optional


class FrustraEvoError(Exception):
    """Base exception for FrustraEvo module"""

    pass


class MSAError(FrustraEvoError):
    """Errors related to MSA processing"""

    pass


class PDBError(FrustraEvoError):
    """Errors related to PDB handling"""

    pass


class CalculationError(FrustraEvoError):
    """Errors in frustration calculations"""

    pass


class ValidationError(FrustraEvoError):
    """Data validation errors"""

    pass
