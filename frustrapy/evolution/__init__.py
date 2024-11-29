from typing import Optional, Dict, List, Union
from pathlib import Path
import logging
from .logo_calculator import LogoCalculator
from .contacts import ContactAnalyzer
from .generator import HistogramGenerator
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
        reference_pdb: Reference PDB identifier (optional)
        pdb_dir: Directory containing PDB files (optional)
        contact_maps: Whether to generate contact maps
        results_dir: Output directory (optional)
        debug: Enable debug mode

    Returns:
        Dict containing analysis results and paths to output files
    """
    try:
        # Convert paths to Path objects
        fasta_file = Path(fasta_file)
        if pdb_dir:
            pdb_dir = Path(pdb_dir)
        if results_dir:
            results_dir = Path(results_dir)

        # Initialize calculator with results_dir
        calculator = LogoCalculator(
            job_id=job_id,
            fasta_file=fasta_file,
            reference_pdb=reference_pdb,
            pdb_dir=pdb_dir,
            contact_maps=contact_maps,
            results_dir=results_dir,
        )

        # Run main calculation pipeline
        calculator.calculate()

        # Read information content results
        ic_file = calculator.results_dir / "data" / "InformationContent.csv"
        if ic_file.exists():
            ic_data = pd.read_csv(ic_file)
            ic_results = [
                PositionInformation(
                    position=int(row["Position"]),
                    residue=row["Residue"],
                    chain=row["Chain"],
                    conservation=float(row["Conservation"]),
                    min_percent=float(row["Pct_Min"]),
                    neu_percent=float(row["Pct_Neu"]),
                    max_percent=float(row["Pct_Max"]),
                    min_count=int(row["Count_Min"]),
                    neu_count=int(row["Count_Neu"]),
                    max_count=int(row["Count_Max"]),
                    h_min=float(row["H_Min"]),
                    h_neu=float(row["H_Neu"]),
                    h_max=float(row["H_Max"]),
                    h_total=float(row["H_Total"]),
                    ic_min=float(row["IC_Min"]),
                    ic_neu=float(row["IC_Neu"]),
                    ic_max=float(row["IC_Max"]),
                    ic_total=float(row["IC_Total"]),
                    frust_state=row["FrustState"],
                    conserved_state=row["ConservedState"],
                )
                for _, row in ic_data.iterrows()
            ]
        else:
            logger.warning(f"No IC data found at {ic_file}")
            ic_results = []

        # Initialize analyzers using calculator's paths
        contact_analyzer = ContactAnalyzer(
            results_dir=calculator.results_dir,
            mode="configurational",
            frustration_dir=calculator.frustration_dir,
        )

        # Get alignment length
        alignment_length = calculator._get_alignment_length()

        # Analyze contacts with IC results
        contacts = contact_analyzer.analyze_contacts(
            alignment_length=alignment_length, ic_results=ic_results
        )

        # Process contacts
        processed_contacts = {
            "data": contacts,
            "information_content": {
                str(r.position): {
                    "conserved_state": r.conserved_state,
                    "ic_min": float(r.ic_min),
                    "ic_max": float(r.ic_max),
                    "ic_neu": float(r.ic_neu),
                    "ic_total": float(r.ic_total),
                }
                for r in ic_results
            },
        }

        # Generate contact maps if requested
        if contact_maps:
            histogram_generator = HistogramGenerator(
                results_dir=calculator.results_dir,
                msa_file=calculator.job_dir / "MSA_Clean_final.fasta",
                reference_pdb=reference_pdb,
            )
            histogram_generator.generate_contact_maps(
                contacts=contacts, ic_results=ic_results
            )

        # Collect results
        results = {
            "job_id": job_id,
            "output_dir": str(calculator.results_dir),
            "files": {
                "logo": str(calculator.results_dir / "plots" / "sequence_logo.png"),
                "data": str(calculator.results_dir / "data" / "FrustrationData.csv"),
                "contact_maps": (
                    [str(calculator.results_dir / "plots" / "contact_map.png")]
                    if contact_maps
                    else None
                ),
            },
            "contacts": processed_contacts,
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
