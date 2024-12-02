import sys
import logging
from pathlib import Path
import frustrapy
from frustrapy.evolution import analyze_family
from frustrapy.evolution.exceptions import FrustraEvoError
import matplotlib.pyplot as plt

# Configure logging - Add these lines
logging.getLogger("matplotlib").setLevel(logging.WARNING)  # Silence matplotlib
logging.getLogger("PIL").setLevel(logging.WARNING)  # Silence PIL/Pillow
logging.getLogger("fontTools").setLevel(
    logging.WARNING
)  # Silence font related messages

# Main logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def run_test():
    """Run FrustraEvo analysis on example data."""
    try:
        logger.info("Starting FrustraEvo analysis")
        logger.info("FASTA file: frustraevo_example_inputs/alphas.fasta")
        logger.info("PDB directory: frustraevo_example_inputs/pdbs")
        logger.info("Results directory: frustraevo_results")

        # Run analysis
        results = analyze_family(
            fasta_file="frustraevo_example_inputs/alphas.fasta",
            job_id="test",
            reference_pdb="2dn1-A",
            pdb_dir="frustraevo_example_inputs/pdbs",
            contact_maps=True,
            results_dir="frustraevo_results",
            debug=True,
        )

        # Validate results
        total_contacts = len(results["contacts"]["information_content"])
        logger.info(f"Total contacts analyzed: {total_contacts}")

        # Count contacts by frustration state
        min_contacts = sum(
            1
            for c in results["contacts"]["information_content"].values()
            if c["FstConserved"]
            == "MIN"  # Changed from conserved_state to FstConserved
        )
        neu_contacts = sum(
            1
            for c in results["contacts"]["information_content"].values()
            if c["FstConserved"]
            == "NEU"  # Changed from conserved_state to FstConserved
        )
        max_contacts = sum(
            1
            for c in results["contacts"]["information_content"].values()
            if c["FstConserved"]
            == "MAX"  # Changed from conserved_state to FstConserved
        )

        logger.info(f"Minimally frustrated contacts: {min_contacts}")
        logger.info(f"Neutrally frustrated contacts: {neu_contacts}")
        logger.info(f"Maximally frustrated contacts: {max_contacts}")

        return results

    except Exception as e:
        logger.error(f"FrustraEvo analysis failed: {str(e)}")
        raise


def validate_results(results):
    """Validate analysis results"""
    try:
        # Check required files exist in data/ subdirectory
        data_files = [
            "FrustrationData.csv",
            "InformationContent.csv",
            "SequenceLogoData.csv",
        ]

        # Check required files exist in plots/ subdirectory
        plot_files = ["ContactMaps.png", "SequenceLogo.png"]

        output_dir = Path(results["output_dir"])
        missing_files = []

        # Check data files
        data_dir = output_dir / "data"
        for file in data_files:
            if not (data_dir / file).exists():
                missing_files.append(f"data/{file}")

        # Check plot files
        plots_dir = output_dir / "plots"
        for file in plot_files:
            if not (plots_dir / file).exists():
                missing_files.append(f"plots/{file}")

        if missing_files:
            logger.error(f"Missing output files: {', '.join(missing_files)}")
            return False

        # Validate data files contain expected content
        try:
            import pandas as pd

            # Check FrustrationData.csv
            frust_data = pd.read_csv(data_dir / "FrustrationData.csv")
            if len(frust_data) == 0:
                logger.error("FrustrationData.csv is empty")
                return False

            # Check InformationContent.csv
            ic_data = pd.read_csv(data_dir / "InformationContent.csv")
            if len(ic_data) == 0:
                logger.error("InformationContent.csv is empty")
                return False

        except Exception as e:
            logger.error(f"Failed to validate data files: {str(e)}")
            return False

        logger.info("Results validation passed")
        return True

    except Exception as e:
        logger.error(f"Results validation failed: {str(e)}")
        return False


if __name__ == "__main__":
    try:
        # Run analysis
        results = run_test()
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        sys.exit(1)
