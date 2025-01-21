from typing import Optional, Dict, List, Union
from pathlib import Path
import logging
from .information_content import InformationContentCalculator
from .data_classes import MSAData
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_family(
    fasta_file: Union[str, Path],
    job_id: str,
    reference_pdb: Optional[str] = None,
    pdb_dir: Optional[Union[str, Path]] = None,
    contact_maps: bool = False,
    results_dir: Optional[Union[str, Path]] = None,
    debug: bool = False,
) -> Dict:
    """
    Analyze evolutionary frustration patterns in a protein family.

    Args:
        fasta_file: Path to MSA file in FASTA format
        job_id: Identifier for this analysis
        reference_pdb: Reference PDB identifier
        pdb_dir: Directory containing PDB files
        contact_maps: Whether to generate contact maps
        results_dir: Output directory
        debug: Enable debug mode

    Returns:
        Dict containing analysis results and paths to output files
    """
    try:
        # Convert paths
        fasta_file = Path(fasta_file)
        results_dir = Path(results_dir) if results_dir else Path(f"results_{job_id}")
        if pdb_dir:
            pdb_dir = Path(pdb_dir)

        # Load MSA data
        msa_data = MSAData.from_fasta(fasta_file)

        # Initialize calculator
        calculator = InformationContentCalculator(
            msa_data=msa_data,
            results_dir=results_dir,
            reference_pdb=reference_pdb,
            pdb_dir=pdb_dir,
        )

        # Setup required files and directories
        calculator._setup_directories()
        calculator._copy_required_files()
        calculator._validate_sequences()
        calculator._prepare_reference_alignment()
        calculator._create_final_alignment()
        calculator._calculate_equivalences()

        # Calculate information content
        ic_results = calculator.calculate()

        # Generate visualizations if requested
        if contact_maps:
            from .generator import HistogramGenerator

            generator = HistogramGenerator(
                results_dir=results_dir,
                msa_file=fasta_file,
                reference_pdb=reference_pdb,
            )
            generator.generate_contact_maps(ic_results)

        # Convert list of contacts to dictionary with indices as keys
        contact_data = ic_results.to_dict(orient="records")
        contact_dict = {i: contact for i, contact in enumerate(contact_data)}

        # Collect results with updated structure
        results = {
            "job_id": job_id,
            "output_dir": str(results_dir),
            "files": {
                "data": str(results_dir / f"IC_Configurational_{reference_pdb}.csv"),
                "contact_maps": (
                    str(results_dir / "plots" / "contact_map.png")
                    if contact_maps
                    else None
                ),
            },
            "contacts": {
                "information_content": contact_dict,  # Now a dictionary with indices as keys
                "summary": {
                    "total_contacts": len(ic_results),
                    "minimally_frustrated": len(
                        ic_results[ic_results["FstConserved"] == "MIN"]
                    ),
                    "neutrally_frustrated": len(
                        ic_results[ic_results["FstConserved"] == "NEU"]
                    ),
                    "maximally_frustrated": len(
                        ic_results[ic_results["FstConserved"] == "MAX"]
                    ),
                },
            },
        }

        return results

    except Exception as e:
        logger.error(f"Failed to analyze protein family: {str(e)}")
        raise


def get_version() -> str:
    """Get package version."""
    return "0.1.0"  # Update this with your version number


# Example usage in documentation
__doc__ = """
FrustraPy Evolution Module
-------------------------

This module provides functionality for analyzing evolutionary patterns of 
protein frustration in families of related proteins.

Example usage:

```python
from frustrapy.evolution import analyze_family

results = analyze_family(
    fasta_file="family.fasta",
    job_id="my_analysis",
    reference_pdb="1abc",
    contact_maps=True
)

# Access results
logo_path = results["files"]["logo"]
contact_maps = results["files"]["contact_maps"]
```
"""
