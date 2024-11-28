from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .logo_calculator import LogoCalculator
from .contacts import ContactAnalyzer, ContactInformation
from .logo import LogoGenerator
from .generator import HistogramGenerator
from ..analysis.frustration import calculate_frustration

logger = logging.getLogger(__name__)


@dataclass
class EvolutionaryAnalysisConfig:
    """Configuration for evolutionary analysis"""

    fasta_file: Path
    job_id: str
    scripts_path: Path
    reference_pdb: Optional[str] = None
    pdb_dir: Optional[Path] = None
    contact_maps: bool = False
    results_dir: Optional[Path] = None
    debug: bool = False
    num_workers: int = 4


@dataclass
class EvolutionaryAnalysisResults:
    """Results from evolutionary analysis"""

    job_id: str
    output_dir: Path
    logo_file: Path
    data_file: Path
    contact_maps: Optional[List[Path]]
    contacts: List[ContactInformation]
    statistics: Dict[str, float]


class EvolutionaryAnalysisPipeline:
    """Handles the full evolutionary analysis pipeline"""

    def __init__(self, config: EvolutionaryAnalysisConfig):
        """Initialize pipeline with configuration"""
        self.config = config
        self._validate_config()
        self.calculator = LogoCalculator(
            job_id=config.job_id,
            fasta_file=config.fasta_file,
            r_scripts_path=config.scripts_path,
            reference_pdb=config.reference_pdb,
            pdb_dir=config.pdb_dir,
            contact_maps=config.contact_maps,
        )

    def run(self) -> EvolutionaryAnalysisResults:
        """Run the full analysis pipeline"""
        try:
            logger.info(f"Starting evolutionary analysis for {self.config.job_id}")

            # Step 1: Initial setup and validation
            self._setup()

            # Step 2: Process sequences and calculate frustration
            pdb_list = self._process_sequences()
            self._calculate_frustration(pdb_list)

            # Step 3: Calculate contacts and generate visualizations
            contacts = self._analyze_contacts()
            self._generate_visualizations(contacts)

            # Step 4: Collect and organize results
            results = self._collect_results(contacts)

            logger.info("Evolutionary analysis completed successfully")
            return results

        except Exception as e:
            logger.error(f"Evolutionary analysis failed: {str(e)}")
            if not self.config.debug:
                self._cleanup()
            raise

    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if not self.config.fasta_file.exists():
            raise FileNotFoundError(f"FASTA file not found: {self.config.fasta_file}")

        if self.config.pdb_dir and not self.config.pdb_dir.exists():
            raise FileNotFoundError(f"PDB directory not found: {self.config.pdb_dir}")

        if self.config.scripts_path and not self.config.scripts_path.exists():
            raise FileNotFoundError(
                f"Scripts directory not found: {self.config.scripts_path}"
            )

    def _setup(self) -> None:
        """Set up working environment"""
        # Create directories
        self.calculator.job_dir.mkdir(exist_ok=True)
        self.calculator.results_dir.mkdir(exist_ok=True)

        # Copy necessary files
        self.calculator._copy_files()

    def _process_sequences(self) -> Path:
        """Process sequences and prepare for analysis"""
        # Generate PDB list
        pdb_list = self.calculator._generate_pdb_list()

        # Process MSA
        self.calculator._process_msa()

        # Check sequences
        self.calculator._check_sequences(pdb_list)

        return pdb_list

    def _calculate_frustration(self, pdb_list: Path) -> None:
        """Calculate frustration for all structures"""
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            with open(pdb_list) as f:
                pdb_ids = [line.strip() for line in f]

            with tqdm(total=len(pdb_ids), desc="Calculating frustration") as pbar:
                for pdb_id in pdb_ids:
                    future = executor.submit(self._calculate_single_frustration, pdb_id)
                    future.add_done_callback(lambda p: pbar.update(1))
                    futures.append(future)

                # Wait for all calculations to complete
                for future in futures:
                    future.result()  # This will raise any exceptions that occurred

    def _calculate_single_frustration(self, pdb_id: str) -> None:
        """Calculate frustration for a single structure"""
        try:
            pdb_file = (
                self.config.pdb_dir / f"{pdb_id}.pdb"
                if self.config.pdb_dir
                else self.calculator.frustration_dir / f"{pdb_id}.pdb"
            )

            calculate_frustration(
                pdb_file=str(pdb_file),
                mode="configurational",
                results_dir=str(self.calculator.frustration_dir),
                graphics=False,
                visualization=False,
            )

        except Exception as e:
            logger.error(f"Failed to calculate frustration for {pdb_id}: {str(e)}")
            raise

    def _analyze_contacts(self) -> List[ContactInformation]:
        """Analyze contacts across structures"""
        analyzer = ContactAnalyzer(self.calculator.results_dir)
        return analyzer.analyze_contacts(
            alignment_length=self.calculator._get_alignment_length()
        )

    def _generate_visualizations(self, contacts: List[ContactInformation]) -> None:
        """Generate all visualizations"""
        # Generate logo
        logo_generator = LogoGenerator(
            msa_file=self.calculator.job_dir / "MSA_Clean_final.fasta",
            results_dir=self.calculator.results_dir,
            reference_pdb=self.config.reference_pdb,
        )
        logo_generator.generate_logo(contacts)

        # Generate histograms
        histogram_generator = HistogramGenerator(
            results_dir=self.calculator.results_dir,
            msa_file=self.calculator.job_dir / "MSA_Clean_final.fasta",
            reference_pdb=self.config.reference_pdb,
        )
        histogram_generator.generate_visualization(logo_generator.logo_data)

        # Generate contact maps if requested
        if self.config.contact_maps:
            histogram_generator.generate_contact_maps(contacts)

    def _collect_results(
        self, contacts: List[ContactInformation]
    ) -> EvolutionaryAnalysisResults:
        """Collect and organize results"""
        results_dir = self.calculator.results_dir

        # Calculate basic statistics
        statistics = self._calculate_statistics(contacts)

        # Collect file paths
        logo_file = results_dir / "HistogramFrustration.png"
        data_file = results_dir / "FrustrationData.csv"
        contact_maps = None
        if self.config.contact_maps:
            contact_maps = [
                results_dir / f"CMaps_{self.config.job_id}_IC_Conf.png",
                results_dir / f"CMaps_{self.config.job_id}_IC_Mut.png",
            ]

        return EvolutionaryAnalysisResults(
            job_id=self.config.job_id,
            output_dir=results_dir,
            logo_file=logo_file,
            data_file=data_file,
            contact_maps=contact_maps,
            contacts=contacts,
            statistics=statistics,
        )

    def _calculate_statistics(
        self, contacts: List[ContactInformation]
    ) -> Dict[str, float]:
        """Calculate summary statistics"""
        if not contacts:
            return {}

        total_contacts = len(contacts)
        avg_ic = sum(c.ic_total for c in contacts) / total_contacts
        min_contacts = sum(1 for c in contacts if c.conserved_state == "MIN")
        max_contacts = sum(1 for c in contacts if c.conserved_state == "MAX")
        neu_contacts = sum(1 for c in contacts if c.conserved_state == "NEU")

        return {
            "total_contacts": total_contacts,
            "average_ic": avg_ic,
            "percent_minimal": min_contacts / total_contacts * 100,
            "percent_maximal": max_contacts / total_contacts * 100,
            "percent_neutral": neu_contacts / total_contacts * 100,
        }

    def _cleanup(self) -> None:
        """Clean up temporary files"""
        if not self.config.debug:
            self.calculator._cleanup()
