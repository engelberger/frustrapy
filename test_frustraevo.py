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
    """Run test analysis on example data"""
    try:
        # Set up paths
        example_dir = Path("frustraevo_example_inputs")
        if not example_dir.exists():
            raise FileNotFoundError(f"Example directory not found: {example_dir}")

        # Input files
        fasta_file = example_dir / "alphas.fasta"  # Using the actual example file
        pdb_dir = example_dir / "pdbs"
        results_dir = Path("frustraevo_results")

        # Ensure results directory exists
        results_dir.mkdir(exist_ok=True)

        logger.info("Starting FrustraEvo analysis")
        logger.info(f"FASTA file: {fasta_file}")
        logger.info(f"PDB directory: {pdb_dir}")
        logger.info(f"Results directory: {results_dir}")

        # Run analysis with one of the example PDBs as reference
        results = analyze_family(
            fasta_file=fasta_file,
            job_id="test_run",
            reference_pdb="1fhj-A",  # Using an actual PDB from the example set
            pdb_dir=pdb_dir,
            contact_maps=True,
            results_dir=results_dir,
            debug=True,
        )

        # Print results summary
        logger.info("\nAnalysis Results:")
        logger.info(f"Job ID: {results['job_id']}")
        logger.info(f"Output directory: {results['output_dir']}")

        # Check output files
        logger.info("\nGenerated Files:")
        for category, files in results["files"].items():
            if isinstance(files, list):
                for f in files:
                    logger.info(f"{category}: {Path(f).name}")
            elif files:
                logger.info(f"{category}: {Path(files).name}")

        # Print contact analysis summary
        if results.get("contacts"):
            logger.info("\nContact Analysis Summary:")
            total_contacts = len(results["contacts"]["information_content"])
            min_contacts = sum(
                1
                for c in results["contacts"]["information_content"].values()
                if c["conserved_state"] == "MIN"
            )
            max_contacts = sum(
                1
                for c in results["contacts"]["information_content"].values()
                if c["conserved_state"] == "MAX"
            )
            neu_contacts = sum(
                1
                for c in results["contacts"]["information_content"].values()
                if c["conserved_state"] == "NEU"
            )

            logger.info(f"Total contacts analyzed: {total_contacts}")
            logger.info(
                f"Minimally frustrated contacts: {min_contacts} ({min_contacts/total_contacts*100:.1f}%)"
            )
            logger.info(
                f"Maximally frustrated contacts: {max_contacts} ({max_contacts/total_contacts*100:.1f}%)"
            )
            logger.info(
                f"Neutrally frustrated contacts: {neu_contacts} ({neu_contacts/total_contacts*100:.1f}%)"
            )

        return results

    except FrustraEvoError as e:
        logger.error(f"FrustraEvo analysis failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
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

        # Validate results
        if validate_results(results):
            logger.info("Test completed successfully")
            sys.exit(0)
        else:
            logger.error("Test failed validation")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        sys.exit(1)
