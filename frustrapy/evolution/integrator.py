from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from .logo_calculator import LogoCalculator
from .contacts import ContactAnalyzer
from .sequence_analysis import SequenceAnalyzer
from .single_residue import SingleResidueAnalyzer
from .data_classes import MSAData, EvolutionaryAnalysisMetadata
from ..analysis.frustration import calculate_frustration

logger = logging.getLogger(__name__)


@dataclass
class IntegratedAnalysisConfig:
    """Configuration for integrated evolutionary analysis"""

    fasta_file: Path
    job_id: str
    reference_pdb: Optional[str] = None
    pdb_dir: Optional[Path] = None
    results_dir: Optional[Path] = None
    contact_maps: bool = False
    debug: bool = False
    num_workers: int = 4
    modes: List[str] = ("configurational", "mutational")


@dataclass
class IntegratedAnalysisResults:
    """Results from integrated analysis"""

    job_id: str
    output_dir: Path
    sequence_analysis: Dict
    contact_analysis: Dict
    single_residue_analysis: Optional[Dict]
    visualizations: Dict[str, Path]
    metadata: EvolutionaryAnalysisMetadata


class EvolutionaryAnalysisIntegrator:
    """Integrates all evolutionary analysis components"""

    def __init__(self, config: IntegratedAnalysisConfig):
        """Initialize integrator with configuration"""
        self.config = config
        self.results_dir = self._setup_results_dir()
        self.msa_data = self._load_msa_data()

    def run_analysis(self) -> IntegratedAnalysisResults:
        """Run complete evolutionary analysis pipeline"""
        try:
            logger.info(f"Starting integrated analysis for {self.config.job_id}")

            # Step 1: Calculate frustration for all structures
            structures = self._calculate_frustration()

            # Step 2: Analyze sequences and conservation
            sequence_results = self._analyze_sequences()

            # Step 3: Analyze contacts
            contact_results = self._analyze_contacts()

            # Step 4: Single residue analysis if reference provided
            single_residue_results = None
            if self.config.reference_pdb:
                single_residue_results = self._analyze_single_residues()

            # Step 5: Generate visualizations
            visualizations = self._generate_visualizations(
                sequence_results, contact_results, single_residue_results
            )

            # Create metadata
            metadata = self._create_metadata()

            # Collect results
            results = IntegratedAnalysisResults(
                job_id=self.config.job_id,
                output_dir=self.results_dir,
                sequence_analysis=sequence_results,
                contact_analysis=contact_results,
                single_residue_analysis=single_residue_results,
                visualizations=visualizations,
                metadata=metadata,
            )

            # Save results
            self._save_results(results)

            return results

        except Exception as e:
            logger.error(f"Integrated analysis failed: {str(e)}")
            if not self.config.debug:
                self._cleanup()
            raise

    def _setup_results_dir(self) -> Path:
        """Setup results directory"""
        if self.config.results_dir:
            results_dir = Path(self.config.results_dir)
        else:
            results_dir = Path(f"FrustraEvo_{self.config.job_id}")

        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    def _load_msa_data(self) -> MSAData:
        """Load and validate MSA data"""
        from Bio import SeqIO

        sequences = []
        identifiers = []
        with open(self.config.fasta_file) as f:
            for record in SeqIO.parse(f, "fasta"):
                sequences.append(str(record.seq))
                identifiers.append(record.id)

        if not sequences:
            raise ValueError("No sequences found in MSA file")

        reference_index = None
        if self.config.reference_pdb:
            try:
                reference_index = identifiers.index(self.config.reference_pdb)
            except ValueError:
                logger.warning(
                    f"Reference PDB {self.config.reference_pdb} not found in MSA"
                )

        return MSAData(
            sequences=sequences,
            identifiers=identifiers,
            length=len(sequences[0]),
            num_sequences=len(sequences),
            reference_index=reference_index,
        )

    def _calculate_frustration(self) -> List[str]:
        """Calculate frustration for all structures"""
        structures = []
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            for pdb_id in self.msa_data.identifiers:
                for mode in self.config.modes:
                    future = executor.submit(
                        self._calculate_single_frustration, pdb_id, mode
                    )
                    futures.append((pdb_id, future))

            # Process results with progress bar
            with tqdm(total=len(futures), desc="Calculating frustration") as pbar:
                for pdb_id, future in futures:
                    try:
                        future.result()
                        structures.append(pdb_id)
                    except Exception as e:
                        logger.error(
                            f"Failed to calculate frustration for {pdb_id}: {str(e)}"
                        )
                    finally:
                        pbar.update(1)

        return structures

    def _calculate_single_frustration(self, pdb_id: str, mode: str) -> None:
        """Calculate frustration for a single structure"""
        pdb_file = self.config.pdb_dir / f"{pdb_id}.pdb"
        calculate_frustration(
            pdb_file=str(pdb_file),
            mode=mode,
            results_dir=str(self.results_dir / pdb_id),
            graphics=False,
            visualization=False,
        )

    def _analyze_sequences(self) -> Dict:
        """Analyze sequences and conservation"""
        analyzer = SequenceAnalyzer(
            msa_data=self.msa_data,
            results_dir=self.results_dir,
            reference_pdb=self.config.reference_pdb,
        )
        return analyzer.analyze()

    def _analyze_contacts(self) -> Dict:
        """Analyze contacts across structures"""
        results = {}
        for mode in self.config.modes:
            analyzer = ContactAnalyzer(results_dir=self.results_dir, mode=mode)
            results[mode] = analyzer.analyze_contacts(self.msa_data.length)
        return results

    def _analyze_single_residues(self) -> Optional[Dict]:
        """Analyze single residues if reference provided"""
        if not self.config.reference_pdb:
            return None

        analyzer = SingleResidueAnalyzer(
            results_dir=self.results_dir,
            msa_data=self.msa_data,
            reference_pdb=self.config.reference_pdb,
            num_workers=self.config.num_workers,
        )

        results = {}
        for pos in range(self.msa_data.length):
            results[pos + 1] = analyzer.analyze_position(pos + 1)

        return results

    def _generate_visualizations(
        self,
        sequence_results: Dict,
        contact_results: Dict,
        single_residue_results: Optional[Dict],
    ) -> Dict[str, Path]:
        """Generate all visualizations"""
        from .logo_calculator import LogoCalculator

        visualizations = {}

        # Initialize calculator
        calculator = LogoCalculator(
            job_id=self.config.job_id,
            fasta_file=self.config.fasta_file,
            results_dir=self.results_dir,
            reference_pdb=self.config.reference_pdb,
        )

        # Generate sequence logo
        logo_path = calculator.generate_logo(sequence_results)
        visualizations["logo"] = logo_path

        # Generate contact maps
        if self.config.contact_maps:
            for mode in self.config.modes:
                map_path = calculator.generate_contact_map(contact_results[mode], mode)
                visualizations[f"contact_map_{mode}"] = map_path

        # Generate single residue plots
        if single_residue_results:
            plot_path = calculator.generate_mutation_plot(single_residue_results)
            visualizations["mutations"] = plot_path

        return visualizations

    def _create_metadata(self) -> EvolutionaryAnalysisMetadata:
        """Create analysis metadata"""
        from datetime import datetime

        return EvolutionaryAnalysisMetadata(
            job_id=self.config.job_id,
            reference_pdb=self.config.reference_pdb,
            num_structures=len(self.msa_data.identifiers),
            alignment_length=self.msa_data.length,
            analysis_date=datetime.now().isoformat(),
            parameters={
                "modes": self.config.modes,
                "contact_maps": self.config.contact_maps,
                "num_workers": self.config.num_workers,
            },
        )

    def _save_results(self, results: IntegratedAnalysisResults) -> None:
        """Save analysis results"""
        import json

        # Save metadata
        with open(self.results_dir / "metadata.json", "w") as f:
            json.dump(results.metadata.__dict__, f, indent=2)

        # Save analysis results
        results_file = self.results_dir / "analysis_results.pkl"
        with open(results_file, "wb") as f:
            import pickle

            pickle.dump(results, f)

    def _cleanup(self) -> None:
        """Clean up temporary files"""
        import shutil

        if not self.config.debug:
            # List of temporary directories to clean
            temp_dirs = ["Frustration", "equivalences", "tmp"]

            for temp_dir in temp_dirs:
                path = self.results_dir / temp_dir
                if path.exists():
                    shutil.rmtree(path)
