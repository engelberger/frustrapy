from typing import Optional, Dict, List, Tuple, Any, Union
import logging
import os
import time
import tempfile
import subprocess
import shutil
import glob
from Bio.PDB import PDBParser, PDBIO
import pandas as pd
import numpy as np
from ..core import Pdb
from ..utils import get_os, replace_expr
from ..utils.helpers import pdb_equivalences, renum_files
from ..utils.decorators import log_execution_time
from .mutations import mutate_res_parallel, mutate_res_scan_parallel
from ..visualization import (
    plot_5andens,
    plot_5adens_proportions,
    plot_contact_map,
    plot_delta_frus,
)
from .chain_selector import NonHetSelect, ChainSelect
from ..core.data_classes import FrustrationDensity, FrustrationDensityResults
import sys
import pickle

# Added imports for configuration, custom exceptions, subprocess helper, and file utilities
from .config import FrustrationConfig
from .exceptions import ValidationError, FileOperationError, MissingBackboneAtomError
from ..utils.subprocess import run_subprocess
from ..utils.file_utils import safe_copy, safe_move, ensure_file_exists
from ..utils.ui import display_error, display_overwrite_warning, display_success

logger = logging.getLogger(__name__)


class FrustrationCalculator:
    """Handles protein frustration calculations and analysis."""

    def __init__(
        self,
        pdb_file: Optional[str] = None,
        pdb_id: Optional[str] = None,
        chain: Optional[Union[str, List[str]]] = None,
        residues: Optional[Dict[str, List[int]]] = None,
        electrostatics_k: Optional[float] = None,
        seq_dist: int = 12,
        mode: str = "configurational",
        graphics: bool = True,
        visualization: bool = True,
        results_dir: Optional[str] = None,
        debug: bool = False,
        overwrite: bool = False,
        n_cpus: Optional[int] = None,
        is_mutation_calculation: bool = False,
        backend=None,
    ):
        """Initialize frustration calculator with configuration parameters.

        Args:
            n_cpus (Optional[int]): Number of CPU cores to use for mutation analysis (None = all available).
            overwrite (bool): Whether to overwrite existing intermediate files (default False).
            backend: frustration energy backend -- a registered name (str), a
                :class:`~frustrapy.backends.FrustrationBackend` instance, or ``None``
                for the default (``lammps``). Selects the energy model; the on-disk
                output contract is identical across backends.
        """
        # Consolidate and validate configuration with dataclass
        try:
            self.config = FrustrationConfig(
                pdb_file=pdb_file,
                pdb_id=pdb_id,
                chain=chain,
                residues=residues,
                electrostatics_k=electrostatics_k,
                seq_dist=seq_dist,
                mode=mode,
                graphics=graphics,
                visualization=visualization,
                results_dir=results_dir,
                debug=debug,
                overwrite=overwrite,
                n_cpus=n_cpus,
            )
        except ValidationError as e:
            logger.error(f"Invalid configuration: {e}")
            raise
        # Assign validated parameters from config
        pdb_file = self.config.pdb_file
        pdb_id = self.config.pdb_id
        chain = self.config.chain
        residues = self.config.residues
        electrostatics_k = self.config.electrostatics_k
        seq_dist = self.config.seq_dist
        mode = self.config.mode
        graphics = self.config.graphics
        visualization = self.config.visualization
        results_dir = self.config.results_dir
        debug = self.config.debug
        overwrite = self.config.overwrite
        n_cpus = self.config.n_cpus
        
        # Store configuration
        self.pdb_file = pdb_file
        self.pdb_id = pdb_id
        self.chain = chain
        self.residues = residues
        self.electrostatics_k = electrostatics_k
        self.seq_dist = seq_dist
        self.mode = mode.lower()
        self.graphics = graphics
        self.visualization = visualization
        self.debug = debug
        self.overwrite = overwrite
        self.n_cpus = n_cpus
        self.results_dir = self._setup_results_dir(results_dir)
        self.temp_folder = tempfile.gettempdir()
        self.plots = {}
        self.is_mutation_calculation = is_mutation_calculation
        # Resolve the energy backend (default: lammps). The backend owns the energy
        # model; parsing, classification, and the density kernel stay shared.
        from ..backends import get_backend

        self.backend = get_backend(backend)

    @log_execution_time
    def calculate(self) -> Tuple[Pdb, Dict, Optional[FrustrationDensityResults]]:
        """Main method to calculate protein frustration."""
        try:
            # Per-phase wall-clock breakdown (Phase 5 / Axis 1 benchmark
            # instrumentation). Purely additive: keys are filled as each phase
            # runs and attached to the returned Pdb object as `pdb.phase_times`.
            # Never alters any output file.
            self.phase_times: Dict[str, float] = {}

            # Verify environment first
            self._verify_environment()

            # Make input PDB path absolute and verify it exists
            if self.pdb_file:
                self.pdb_file = os.path.abspath(self.pdb_file)
                logger.debug(f"Input PDB absolute path: {self.pdb_file}")
                if not os.path.exists(self.pdb_file):
                    raise FileNotFoundError(
                        f"Input PDB file not found: {self.pdb_file}"
                    )
                logger.debug(f"Input PDB file verified: {self.pdb_file}")
            else:
                # Download PDB if no file provided
                self.pdb_file = self._download_pdb()

            # Process structure
            _t_prep0 = time.perf_counter()
            structure = self._process_structure()

            # Setup working environment with absolute paths
            pdb_base = self._get_pdb_base()
            job_dir = os.path.abspath(self._setup_job_directory(pdb_base))
            logger.debug(f"Job directory (absolute): {job_dir}")

            # Copy PDB file to job directory with explicit absolute path handling
            job_pdb = os.path.join(job_dir, f"{pdb_base}.pdb")
            try:
                logger.debug(f"Copying PDB file from {self.pdb_file} to {job_pdb}")
                # Safely copy PDB file to job directory
                safe_copy(self.pdb_file, job_pdb, overwrite=self.overwrite)
                # Update pdb_file to use the copied version
                self.pdb_file = job_pdb
                logger.debug(f"Successfully copied PDB file to job directory: {job_pdb}")
            except FileOperationError as e:
                if "Destination file already exists" in str(e) and not self.overwrite:
                    # Display specific overwrite warning
                    display_overwrite_warning(e)
                else:
                    # Display general file operation error
                    display_error(e, is_debug=self.debug)
                # Library code must not call sys.exit(): propagate (CLAUDE.md §8).
                raise

            # Create PDB object with absolute paths
            pdb = self._create_pdb_object(job_dir, pdb_base)
            self._prepare_calculation_files(pdb)
            self.phase_times["prep"] = time.perf_counter() - _t_prep0

            # Run calculations through the selected backend. The backend owns the
            # energy model (compute_energies); parsing/classification and the density
            # kernel are shared post-processing (process_results / compute_density).
            _t = time.perf_counter()
            self.backend.compute_energies(self, pdb)
            self.phase_times["lammps"] = time.perf_counter() - _t

            _t = time.perf_counter()
            self.backend.process_results(self, pdb)
            self.phase_times["classify"] = time.perf_counter() - _t

            # Create FrustrationData directory
            frustration_dir = os.path.join(job_dir, "FrustrationData")
            os.makedirs(frustration_dir, exist_ok=True)

            # Calculate frustration density if mode is configurational or mutational
            frustration_density_results = None
            _t = time.perf_counter()
            if self.mode in ["configurational", "mutational"]:
                logger.debug("Calculating frustration density...")
                frustration_density_results = self.backend.compute_density(self, pdb)
            self.phase_times["density"] = time.perf_counter() - _t

            # Move/copy only the necessary files to FrustrationData directory
            _t = time.perf_counter()
            files_to_move = [
                (f"{pdb_base}.pdb", f"{pdb_base}.pdb"),
                (f"{pdb_base}.pdb_{self.mode}", f"{pdb_base}.pdb_{self.mode}"),
                ("tertiary_frustration.dat", "tertiary_frustration.dat"),
            ]

            # Copy files to FrustrationData directory
            for src_name, dest_name in files_to_move:
                src_path = os.path.join(job_dir, src_name)
                dest_path = os.path.join(frustration_dir, dest_name)
                # Skip if already moved
                if os.path.exists(dest_path):
                    logger.debug(f"File already exists in FrustrationData: {dest_name}")
                    continue
                if os.path.exists(src_path):
                    logger.debug(f"Moving file {src_name} to {dest_name}")
                    try:
                        safe_move(src_path, dest_path, overwrite=self.overwrite)
                        logger.debug(f"Moved file {src_name} to {dest_name}")
                    except FileOperationError as e:
                        logger.debug(f"Failed to move file {src_name}: {e}")
                        raise

            # Update pdb_file path
            self.pdb_file = os.path.join(frustration_dir, f"{pdb_base}.pdb")

            # Now verify all required files are in place
            self._verify_required_files(pdb)
            self.phase_times["io"] = time.perf_counter() - _t

            # Generate visualizations if requested
            _t = time.perf_counter()
            if self.graphics:
                self._generate_graphics(pdb)
            if self.visualization:
                self._generate_visualizations(pdb)
            self.phase_times["graphics"] = time.perf_counter() - _t

            # Cleanup only if not in debug mode
            if not self.debug:
                self._cleanup(job_dir)

            # Attach the per-phase breakdown to the returned object (Phase 5
            # Axis-1 instrumentation; additive, does not affect any output file).
            self.phase_times["total"] = sum(
                self.phase_times.get(k, 0.0)
                for k in ("prep", "lammps", "classify", "density", "io", "graphics")
            )
            pdb.phase_times = dict(self.phase_times)

            return pdb, self.plots, frustration_density_results

        except Exception as e:
            # Log detailed context only in debug mode
            if self.debug:
                logger.debug(f"Frustration calculation failed: {str(e)}", exc_info=True)
                logger.debug(f"Current working directory: {os.getcwd()}")
                if hasattr(self, "pdb_file"):
                    logger.debug(f"PDB file exists: {os.path.exists(self.pdb_file)}")
                if "job_dir" in locals():
                    logger.debug(f"Job directory contents: {os.listdir(job_dir)}")
                if "frustration_dir" in locals() and os.path.exists(frustration_dir):
                    logger.debug(f"FrustrationData contents: {os.listdir(frustration_dir)}")
            else:
                # Log simpler error message for INFO/WARN/ERROR levels
                logger.error(f"Frustration calculation failed: {str(e)}")
            raise

    def _setup_results_dir(self, results_dir: Optional[str]) -> str:
        """Setup and validate results directory."""
        if results_dir is None:
            results_dir = os.path.join(tempfile.gettempdir(), "")
        elif not os.path.exists(results_dir):
            os.makedirs(results_dir)
            logger.debug(f"Created results directory: {results_dir}")

        return results_dir if results_dir.endswith("/") else f"{results_dir}/"

    def _download_pdb(self) -> str:
        """Download PDB file if pdb_id is provided."""
        if self.pdb_file is None:
            logger.debug("Downloading PDB file...")
            pdb_url = f"https://files.rcsb.org/download/{self.pdb_id}.pdb"
            # P0-13: fail loudly (check=True), bound the network call (timeout=),
            # and keep TLS certificate verification enabled (wget verifies by default).
            run_subprocess(
                [
                    "wget",
                    "-P",
                    self.temp_folder,
                    pdb_url,
                    "-q",
                ],
                cwd=self.temp_folder,
                check=True,
                timeout=60,
            )
            downloaded = os.path.join(self.temp_folder, f"{self.pdb_id}.pdb")
            # Validate the download actually produced a usable file.
            if not os.path.isfile(downloaded) or os.path.getsize(downloaded) == 0:
                raise FileOperationError(
                    f"Failed to download a valid PDB for id {self.pdb_id} from {pdb_url}"
                )
            self.pdb_file = downloaded
        return self.pdb_file

    def _process_structure(self) -> Tuple[str, PDBParser]:
        """Process PDB structure and handle chain selection."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("structure", self.pdb_file)

        # Remove HET atoms
        pdb_io = PDBIO()
        pdb_io.set_structure(structure)
        pdb_io.save(self.pdb_file, NonHetSelect())

        if self.chain is not None:
            self._handle_chain_selection(structure, pdb_io)

        return structure

    def _setup_job_directory(self, pdb_base: str) -> str:
        """Setup job directory."""
        logger.debug(f"Setting up job directory for {pdb_base}")

        # Create job directory with absolute path
        job_dir = os.path.abspath(os.path.join(self.results_dir, f"{pdb_base}.done/"))
        os.makedirs(job_dir, exist_ok=True)
        logger.debug(f"Created/verified job directory: {job_dir}")

        # Verify directory is writable
        if not os.access(job_dir, os.W_OK):
            raise PermissionError(f"Job directory is not writable: {job_dir}")

        return job_dir

    def _get_pdb_base(self) -> str:
        """Get the base name for PDB files and directories."""
        base = os.path.splitext(os.path.basename(self.pdb_file))[0]

        # Handle chain suffix
        if self.chain is not None:
            if isinstance(self.chain, str):
                # Single chain
                return f"{base}_{self.chain}"
            else:
                # Multiple chains - use first chain for naming
                return f"{base}_{self.chain[0]}"
        return base

    def _handle_chain_selection(self, structure, pdb_io) -> None:
        """Handle chain selection and validation."""
        logger.debug(f"Checking chains: {self.chain}")

        # Get available chains
        available_chains = [chain.id for chain in structure.get_chains()]
        logger.debug(f"Available chains: {available_chains}")

        # Validate requested chains
        selected_chains = [self.chain] if isinstance(self.chain, str) else self.chain
        missing_chains = [c for c in selected_chains if c not in available_chains]

        if missing_chains:
            error_msg = f"The Chain(s) {' '.join(missing_chains)} don't exist! Available chains: {' '.join(available_chains)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"All requested chains {selected_chains} are valid")
        pdb_io.save(self.pdb_file, ChainSelect(selected_chains))
        logger.debug(f"Saved PDB with selected chains to {self.pdb_file}")

    def _create_pdb_object(self, job_dir: str, pdb_base: str) -> Pdb:
        """Create and initialize PDB object with filtered data."""
        logger.debug("Filtering PDB data...")
        df = self._read_and_filter_pdb()
        equivalences = pdb_equivalences(self.pdb_file, job_dir)
        return Pdb(job_dir, pdb_base, self.mode, df, equivalences)

    def _read_and_filter_pdb(self) -> pd.DataFrame:
        """Read and filter PDB data using Biopython for robust parsing."""
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("structure", self.pdb_file)
        records = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Skip hetero/water residues
                    if residue.get_id()[0] != ' ':
                        continue
                    for atom in residue:
                        records.append({
                            "ATOM": "ATOM",
                            "atom_num": atom.get_serial_number(),
                            "atom_name": atom.get_name(),
                            "res_name": residue.get_resname(),
                            "chain": chain.get_id(),
                            "res_num": residue.get_id()[1],
                            "x": atom.get_coord()[0],
                            "y": atom.get_coord()[1],
                            "z": atom.get_coord()[2],
                            "occupancy": atom.get_occupancy(),
                            "b_factor": atom.get_bfactor(),
                            "element": atom.element.strip(),
                        })
        df = pd.DataFrame(records)
        # Standardize residue names
        residue_mappings = {"MSE": "MET", "HIE": "HIS", "CYX": "CYS", "CY1": "CYS"}
        for old, new in residue_mappings.items():
            df.loc[df["res_name"] == old, "res_name"] = new
        # Filter for standard protein residues
        protein_res = [
            "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY",
            "HIS","ILE","LEU","LYS","MET","PHE","PRO","SER",
            "THR","TRP","TYR","VAL",
        ]
        df = df[df["res_name"].isin(protein_res)]
        # Set default chain if missing
        df.loc[df["chain"].isna(), "chain"] = "A"
        return df

    def _prepare_calculation_files(self, pdb: Pdb) -> None:
        """Prepare files needed for calculation."""
        logger.debug(f"Preparing calculation files in {pdb.job_dir}...")

        # Verify PDB object has required attributes
        required_attrs = ["scripts_dir", "job_dir", "pdb_base"]
        for attr in required_attrs:
            if not hasattr(pdb, attr):
                raise AttributeError(f"PDB object missing required attribute: {attr}")
            logger.debug(f"Found required attribute {attr}: {getattr(pdb, attr)}")

        # Verify input PDB file exists
        if not os.path.exists(self.pdb_file):
            raise FileNotFoundError(f"Input PDB file not found: {self.pdb_file}")
        logger.debug(f"Input PDB file exists: {self.pdb_file}")

        # Verify scripts directory exists and contains required files
        lammps_script = os.path.join(
            pdb.scripts_dir, "AWSEMFiles/AWSEMTools/PdbCoords2Lammps.sh"
        )
        if not os.path.exists(lammps_script):
            raise FileNotFoundError(
                f"LAMMPS conversion script not found: {lammps_script}"
            )
        logger.debug(f"Found LAMMPS conversion script: {lammps_script}")

        try:
            # Create necessary directories
            os.makedirs(pdb.job_dir, exist_ok=True)
            logger.debug(f"Created/verified job directory: {pdb.job_dir}")

            # Verify current working directory
            logger.debug(f"Current working directory: {os.getcwd()}")

            # Run conversion script with helper
            logger.debug("Running LAMMPS conversion script...")
            cmd = ["sh", lammps_script, pdb.pdb_base, pdb.pdb_base, pdb.scripts_dir]
            logger.debug(f"Running command: {' '.join(cmd)}")

            result = run_subprocess(cmd, cwd=pdb.job_dir)

            if result.stdout:
                logger.debug(f"Conversion script output:\n{result.stdout}")
            if result.stderr:
                logger.warning(f"Conversion script warnings:\n{result.stderr}")
            
            # Check for empty or missing output files immediately after conversion
            coord_file = os.path.join(pdb.job_dir, f"{pdb.pdb_base}.coord")
            data_file = os.path.join(pdb.job_dir, f"data.{pdb.pdb_base}")
            input_file = os.path.join(pdb.job_dir, f"{pdb.pdb_base}.in")
            
            # Parse warnings from stderr to extract information about missing atoms
            missing_atoms = []
            if result.warnings:
                for warning in result.warnings:
                    if "missing required backbone atom" in warning.lower():
                        # Try to extract residue, chain and atom information from warning
                        parts = warning.split()
                        for i, part in enumerate(parts):
                            if part == "Residue":
                                try:
                                    res_id = int(parts[i + 1])
                                    chain_id = parts[i + 3].strip("()")
                                    atom_name = parts[i + 7]
                                    missing_atoms.append((res_id, chain_id, atom_name))
                                except (IndexError, ValueError):
                                    # If we can't parse the exact format, just store the warning
                                    missing_atoms.append(warning)
            
            # Verify critical files exist and are not empty
            for file_path, desc in [
                (coord_file, ".coord file"),
                (data_file, "LAMMPS data file"),
                (input_file, "LAMMPS input file")
            ]:
                if not os.path.exists(file_path):
                    error_msg = f"Failed to generate {desc}: {file_path} not found"
                    logger.error(error_msg)
                    if missing_atoms:
                        raise MissingBackboneAtomError(error_msg, missing_atoms)
                    else:
                        raise FileNotFoundError(error_msg)
                    
                if os.path.getsize(file_path) == 0:
                    error_msg = f"Generated {desc} is empty: {file_path}"
                    logger.error(error_msg)
                    if missing_atoms:
                        raise MissingBackboneAtomError(error_msg, missing_atoms)
                    else:
                        raise FileOperationError(error_msg, dst=file_path)
                
                logger.debug(f"Verified {desc} exists and is not empty: {os.path.getsize(file_path)} bytes")

            # Log commands
            commands_file = os.path.join(pdb.job_dir, "commands.help")
            logger.debug(f"Writing commands to: {commands_file}")
            with open(commands_file, "w") as f:
                f.write(
                    f"sh {lammps_script} {pdb.pdb_base} {pdb.pdb_base} {pdb.scripts_dir}\n"
                )
                f.write(
                    f"cp {os.path.join(pdb.scripts_dir, 'AWSEMFiles/*.dat*')} {pdb.job_dir}\n"
                )

            # Copy data files with verification
            dat_files = glob.glob(os.path.join(pdb.scripts_dir, "AWSEMFiles/*.dat*"))
            logger.debug(f"Found {len(dat_files)} DAT files to copy")
            for dat_file in dat_files:
                dest_file = os.path.join(pdb.job_dir, os.path.basename(dat_file))
                logger.debug(f"Copying DAT file {dat_file} to {dest_file}")
                try:
                    safe_copy(dat_file, dest_file, overwrite=self.overwrite)
                    logger.debug(f"Copied DAT file to {dest_file}")
                except FileOperationError as e:
                    logger.debug(f"Failed to copy DAT file {dat_file}: {e}")
                    raise

            # List all files in job directory
            job_files = os.listdir(pdb.job_dir)
            logger.debug(f"Files in job directory after setup:\n{', '.join(job_files)}")

            # Verify required files exist
            required_files = [
                os.path.join(pdb.job_dir, f"{pdb.pdb_base}.in"),
                os.path.join(pdb.job_dir, "fix_backbone_coeff.data"),
            ]

            for req_file in required_files:
                if not os.path.exists(req_file):
                    logger.error(f"Required file not found: {req_file}")
                    logger.error(
                        f"Directory contents: {os.listdir(os.path.dirname(req_file))}"
                    )
                    raise FileNotFoundError(f"Required file not generated: {req_file}")
                logger.debug(f"Verified required file exists: {req_file}")

            # Configure options
            self._configure_options(pdb)

        except Exception as e:
            if self.debug:
                logger.debug(f"Failed to prepare calculation files: {str(e)}", exc_info=True)
                logger.debug(f"Current working directory: {os.getcwd()}")
                logger.debug(
                    f"Job directory contents: {os.listdir(pdb.job_dir) if os.path.exists(pdb.job_dir) else 'Directory not found'}"
                )
            else:
                logger.error(f"Failed to prepare calculation files: {str(e)}")
            raise

    def _configure_options(self, pdb: Pdb) -> None:
        """Configure calculation options and electrostatics."""
        logger.debug("Configuring calculation options...")

        input_file = os.path.join(pdb.job_dir, f"{pdb.pdb_base}.in")
        coeff_file = os.path.join(pdb.job_dir, "fix_backbone_coeff.data")

        # Log file paths and existence
        logger.debug(f"Input file path: {input_file}")
        logger.debug(f"Coefficient file path: {coeff_file}")
        logger.debug(f"Input file exists: {os.path.exists(input_file)}")
        logger.debug(f"Coefficient file exists: {os.path.exists(coeff_file)}")

        # Verify files exist
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            logger.error(
                f"Directory contents: {os.listdir(os.path.dirname(input_file))}"
            )
            raise FileNotFoundError(f"Input file not found: {input_file}")
        if not os.path.exists(coeff_file):
            logger.error(f"Coefficient file not found: {coeff_file}")
            logger.error(
                f"Directory contents: {os.listdir(os.path.dirname(coeff_file))}"
            )
            raise FileNotFoundError(f"Coefficient file not found: {coeff_file}")

        try:
            # Log file contents before modification
            with open(input_file, "r") as f:
                logger.debug(f"Input file contents before modification:\n{f.read()}")
            with open(coeff_file, "r") as f:
                logger.debug(
                    f"Coefficient file contents before modification:\n{f.read()}"
                )

            # Make modifications
            replace_expr("run\t\t10000", "run\t\t0", input_file)
            replace_expr("mutational", pdb.mode, coeff_file)

            # Verify modifications
            with open(input_file, "r") as f:
                logger.debug(f"Input file contents after modification:\n{f.read()}")
            with open(coeff_file, "r") as f:
                logger.debug(
                    f"Coefficient file contents after modification:\n{f.read()}"
                )

            if self.electrostatics_k is not None:
                self._configure_electrostatics(pdb)

        except Exception as e:
            if self.debug:
                logger.debug(f"Failed to configure options: {str(e)}", exc_info=True)
                logger.debug(f"Current working directory: {os.getcwd()}")
                logger.debug(f"Job directory contents: {os.listdir(pdb.job_dir)}")
            else:
                logger.error(f"Failed to configure options: {str(e)}")
            raise

    def _configure_electrostatics(self, pdb: Pdb) -> None:
        """Configure electrostatics settings."""
        logger.debug("Configuring electrostatics...")

        # Update configuration files
        replace_expr(
            "\\[DebyeHuckel\\]-",
            "\\[DebyeHuckel\\]",
            os.path.join(pdb.job_dir, "fix_backbone_coeff.data"),
        )
        replace_expr(
            "4.15 4.15 4.15",
            f"{self.electrostatics_k} {self.electrostatics_k} {self.electrostatics_k}",
            os.path.join(pdb.job_dir, "fix_backbone_coeff.data"),
        )

        # Generate GRO file
        run_subprocess([
            "python3",
            os.path.join(pdb.scripts_dir, "Pdb2Gro.py"),
            f"{pdb.pdb_base}.pdb",
            f"{pdb.pdb_base}.pdb.gro",
        ], cwd=pdb.job_dir)

        # Generate charge file
        run_subprocess([
            "perl",
            os.path.join(pdb.scripts_dir, "GenerateChargeFile.pl"),
            f"{pdb.pdb_base}.pdb.gro",
            "--output",
            os.path.join(pdb.job_dir, "charge_on_residues.dat"),
        ], cwd=pdb.job_dir)

    def _run_lammps_calculation(self, pdb: Pdb) -> None:
        """Run LAMMPS calculations."""
        from .lammps_runner import LammpsRunner

        logger.debug("Running LAMMPS calculations...")
        runner = LammpsRunner(
            job_dir=pdb.job_dir,
            pdb_base=pdb.pdb_base,
            seq_dist=self.seq_dist,
            scripts_dir=pdb.scripts_dir,  # Pass the scripts_dir from the PDB object
            debug=self.debug,
        )
        runner.run()

    def _process_results(self, pdb: Pdb) -> None:
        """Process calculation results."""
        logger.debug("Processing results...")

        # Verify job directory exists
        assert os.path.exists(pdb.job_dir), f"Job directory not found: {pdb.job_dir}"
        logger.debug(
            f"Job directory contents before processing: {os.listdir(pdb.job_dir)}"
        )

        # Create FrustrationData directory if it doesn't exist
        frustration_dir = os.path.join(pdb.job_dir, "FrustrationData")
        os.makedirs(frustration_dir, exist_ok=True)
        logger.debug(f"Created/verified FrustrationData directory: {frustration_dir}")

        # Get base name without chain suffix for equivalences file
        base_name = os.path.splitext(os.path.basename(self.pdb_file))[0]
        if self.chain is not None:
            # Remove any chain suffix
            if isinstance(self.chain, str):
                base_name = base_name.replace(f"_{self.chain}", "")
            else:
                for c in self.chain:
                    base_name = base_name.replace(f"_{c}", "")

        # Look for equivalences file with and without chain suffix
        possible_equivalences_files = [
            f"{base_name}.pdb_equivalences.txt",  # Without chain
            f"{pdb.pdb_base}.pdb_equivalences.txt",  # With chain
        ]

        # Find equivalences file
        equivalences_file = None
        for eq_file in possible_equivalences_files:
            eq_path = os.path.join(pdb.job_dir, eq_file)
            if os.path.exists(eq_path):
                equivalences_file = eq_file
                logger.debug(f"Found equivalences file: {eq_path}")
                break

        if equivalences_file is None:
            logger.error("No equivalences file found. Tried:")
            for eq_file in possible_equivalences_files:
                logger.error(f"  - {eq_file}")
            logger.error(f"Directory contents: {os.listdir(pdb.job_dir)}")
            raise FileNotFoundError(
                f"Required equivalences file not found in {pdb.job_dir}"
            )

        # Process renum files with correct path
        try:
            renum_files(
                job_id=pdb.pdb_base,
                job_dir=pdb.job_dir,
                mode=pdb.mode,
                equivalences_file=equivalences_file,
            )
            logger.debug(
                f"Job directory contents after renum: {os.listdir(pdb.job_dir)}"
            )

            # Move files to FrustrationData directory with correct chain handling
            files_to_move = {
                f"{pdb.pdb_base}.pdb": f"{pdb.pdb_base}.pdb",  # Keep chain in PDB filename
                f"{pdb.pdb_base}.pdb_{pdb.mode}": f"{pdb.pdb_base}.pdb_{pdb.mode}",
                "tertiary_frustration.dat": "tertiary_frustration.dat",
            }

            # Only include 5adens file if not in singleresidue mode
            if pdb.mode != "singleresidue":
                files_to_move[f"{pdb.pdb_base}.pdb_{pdb.mode}_5adens"] = (
                    f"{pdb.pdb_base}.pdb_{pdb.mode}_5adens"
                )

            for src_name, dest_name in files_to_move.items():
                src_path = os.path.join(pdb.job_dir, src_name)
                dest_path = os.path.join(frustration_dir, dest_name)
                # Skip if already moved
                if os.path.exists(dest_path):
                    logger.debug(f"File already exists in FrustrationData: {dest_name}")
                    continue
                if os.path.exists(src_path):
                    logger.debug(f"Moving file {src_name} to {dest_name}")
                    try:
                        safe_move(src_path, dest_path, overwrite=self.overwrite)
                        logger.debug(f"Moved file {src_name} to {dest_name}")
                    except FileOperationError as e:
                        logger.debug(f"Failed to move file {src_name}: {e}")
                        raise

            # Verify files were moved successfully
            logger.debug(
                f"FrustrationData directory contents: {os.listdir(frustration_dir)}"
            )

        except Exception as e:
            if self.debug:
                logger.debug(f"Failed to process results: {str(e)}", exc_info=True)
                logger.debug(f"Job directory contents: {os.listdir(pdb.job_dir)}")
                logger.debug(
                    f"FrustrationData contents: {os.listdir(frustration_dir) if os.path.exists(frustration_dir) else 'Directory not found'}"
                )
            else:
                logger.error(f"Failed to process results: {str(e)}")
            raise

    def _generate_graphics(self, pdb: Pdb) -> None:
        """Generate visualization graphics."""
        logger.debug("Generating graphics...")

        # Create Images directory
        images_dir = os.path.join(pdb.job_dir, "Images")
        os.makedirs(images_dir, exist_ok=True)
        logger.debug(f"Created/verified Images directory: {images_dir}")

        if pdb.mode != "singleresidue":
            logger.debug("Generating plots for configurational/mutational mode")

            # Update pdb object with correct paths
            pdb.frustration_dir = os.path.join(pdb.job_dir, "FrustrationData")

            # Verify required files exist
            required_files = {
                "density_file": os.path.join(
                    pdb.frustration_dir, f"{pdb.pdb_base}.pdb_{pdb.mode}_5adens"
                ),
                "contacts_file": os.path.join(
                    pdb.frustration_dir, f"{pdb.pdb_base}.pdb_{pdb.mode}"
                ),
                "tertiary_file": os.path.join(
                    pdb.frustration_dir, "tertiary_frustration.dat"
                ),
            }

            for name, path in required_files.items():
                if not os.path.exists(path):
                    logger.error(f"{name} not found: {path}")
                    logger.error(
                        f"FrustrationData contents: {os.listdir(pdb.frustration_dir)}"
                    )
                    raise FileNotFoundError(f"Required file not found: {path}")
                logger.debug(f"Found required {name}: {path}")

            try:
                logger.debug("Generating 5andens plot...")
                plot_5a = plot_5andens(pdb, save=True)

                logger.debug("Generating 5adens proportions plot...")
                plot_prop = plot_5adens_proportions(pdb, save=True)

                logger.debug("Generating contact map plot...")
                plot_contact = plot_contact_map(pdb, save=True)

                self.plots.update(
                    {
                        "plot_5andens": plot_5a,
                        "plot_5adens_proportions": plot_prop,
                        "plot_contact_map": plot_contact,
                    }
                )

                logger.debug(f"Successfully generated plots: {list(self.plots.keys())}")

            except Exception as e:
                if self.debug:
                    logger.debug(f"Failed to generate plots: {str(e)}", exc_info=True)
                    logger.debug(f"Job directory contents: {os.listdir(pdb.job_dir)}")
                    logger.debug(f"Images directory contents: {os.listdir(images_dir)}")
                    logger.debug(
                        f"FrustrationData contents: {os.listdir(pdb.frustration_dir)}"
                    )
                else:
                    logger.error(f"Failed to generate plots: {str(e)}")
                raise
        else:
            self._generate_singleresidue_analysis(pdb)

    def _generate_singleresidue_analysis(self, pdb: Pdb) -> None:
        """Generate analysis for single residue mode."""
        residues_analyzed = {}
        chains_to_analyze = (
            ([self.chain] if isinstance(self.chain, str) else self.chain)
            if self.chain
            else pdb.atom[pdb.atom["ATOM"] == "ATOM"]["chain"].unique()
        )

        # Lever 1 (flatten): collect every (residue, chain) to mutate FIRST, then
        # run the whole grid through ONE persistent pool. The previous code looped
        # serially over residues and re-forked a fresh pool inside each
        # mutate_res_parallel call (N fork/join cycles, hard 20-way ceiling per
        # cycle). Building the target list up front lets all N×20 LAMMPS evals share
        # a single pool — on > 20 cores the scan keeps > 20 tasks in flight.
        targets = []
        for chain_id in chains_to_analyze:
            residues_analyzed[chain_id] = []
            chain_residues = (
                self.residues.get(chain_id)
                if self.residues
                else pdb.atom[
                    (pdb.atom["ATOM"] == "ATOM") & (pdb.atom["chain"] == chain_id)
                ]["res_num"].unique()
            )
            for res in chain_residues:
                targets.append((res, chain_id))

        if not targets:
            return

        try:
            pdb = mutate_res_scan_parallel(
                pdb=pdb,
                targets=targets,
                split=True,
                method="threading",
                n_cpus=self.n_cpus,
            )
        except Exception as e:
            logger.error(f"Mutation scan failed: {str(e)}")
            return

        # Per-residue delta-frustration plots (cheap, serial; isolated so one
        # plotting failure does not abort the rest of the scan).
        for res, chain_id in targets:
            try:
                plot_key = f"delta_frus_res{res}_chain{chain_id}"
                self.plots[plot_key] = plot_delta_frus(
                    pdb=pdb,
                    res_num=res,
                    chain=chain_id,
                    method="threading",
                    save=True,
                )
            except Exception as e:
                logger.error(
                    f"Failed to plot residue {res} chain {chain_id}: {str(e)}"
                )

    def _generate_visualizations(self, pdb: Pdb) -> None:
        """Generate molecular visualizations."""
        if pdb.mode == "singleresidue":
            return

        logger.debug("Generating visualizations...")
        visualization_dir = os.path.join(pdb.job_dir, "VisualizationScripts")
        os.makedirs(visualization_dir, exist_ok=True)

        # Generate visualization scripts
        try:
            # Find the auxiliar file in the job directory
            auxiliar_file = os.path.join(pdb.job_dir, f"{pdb.pdb_base}_{pdb.mode}.pdb_auxiliar")
            if not os.path.exists(auxiliar_file):
                logger.warning(f"Auxiliar file not found at {auxiliar_file}, visualization may be incomplete")
                # Check if it exists with a different name
                for filename in os.listdir(pdb.job_dir):
                    if filename.endswith("_auxiliar"):
                        auxiliar_file = os.path.join(pdb.job_dir, filename)
                        logger.debug(f"Found auxiliar file with different name: {auxiliar_file}")
                        break
            
            if os.path.exists(auxiliar_file):
                logger.debug(f"Using auxiliar file: {auxiliar_file}")
                # First copy it to where the script expects it
                aux_basename = os.path.basename(auxiliar_file)
                parent_dir_aux_file = os.path.join(os.path.dirname(pdb.job_dir), aux_basename)
                safe_copy(auxiliar_file, parent_dir_aux_file, overwrite=True)
                
                # Now run the script with the correct path
                run_subprocess([
                    "perl",
                    os.path.join(pdb.scripts_dir, "GenerateVisualizations.pl"),
                    aux_basename,  # Use just the basename
                    pdb.pdb_base,
                    os.path.dirname(pdb.job_dir),
                    pdb.mode,
                ], cwd=os.path.dirname(pdb.job_dir))  # Run from the parent directory
                
                # Clean up the copied file
                if os.path.exists(parent_dir_aux_file) and parent_dir_aux_file != auxiliar_file:
                    os.remove(parent_dir_aux_file)
            else:
                logger.warning("No suitable auxiliar file found for visualization generation")
        except Exception as e:
            if self.debug:
                logger.debug(f"Failed to generate visualization scripts: {str(e)}", exc_info=True)
                logger.debug(f"Job directory contents: {os.listdir(pdb.job_dir)}")
                logger.debug(f"Images directory contents: {os.listdir(visualization_dir)}")
                logger.debug(
                    f"FrustrationData contents: {os.listdir(pdb.frustration_dir)}"
                )
            else:
                logger.error(f"Failed to generate visualization scripts: {str(e)}")
            raise

        # Move visualization files
        for ext in ["pml", "tcl", "jml"]:
            # Look in both job_dir and parent directory
            search_dirs = [pdb.job_dir, os.path.dirname(pdb.job_dir)]
            for search_dir in search_dirs:
                pattern = os.path.join(search_dir, f"*_{pdb.mode}.{ext}")
                for file_path in glob.glob(pattern):
                    dest_path = os.path.join(
                        visualization_dir, os.path.basename(file_path)
                    )
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    shutil.move(file_path, visualization_dir)

        # Copy required files
        files_to_copy = [
            (
                os.path.join(pdb.job_dir, "FrustrationData", f"{pdb.pdb_base}.pdb"),
                os.path.join(visualization_dir, f"{pdb.pdb_base}.pdb"),
            ),
            (
                os.path.join(pdb.scripts_dir, "draw_links.py"),
                os.path.join(visualization_dir, "draw_links.py"),
            ),
        ]

        for src, dest in files_to_copy:
            if os.path.exists(src):
                shutil.copy2(src, dest)

    def _cleanup(self, job_dir: str) -> None:
        """Clean up temporary files."""
        if not self.debug:
            # Remove temporary files
            for item in glob.glob(os.path.join(job_dir, "*")):
                if os.path.isfile(item) and not os.path.basename(item).startswith("."):
                    os.remove(item)

            # Clean up chain split directory if needed
            if self.chain is not None:
                split_chain_dir = os.path.join(self.temp_folder, "split_chain")
                if os.path.exists(split_chain_dir):
                    shutil.rmtree(split_chain_dir)

            # Remove downloaded PDB file
            if self.pdb_id:
                pdb_file_path = os.path.join(self.temp_folder, f"{self.pdb_id}.pdb")
                if os.path.exists(pdb_file_path):
                    os.remove(pdb_file_path)

    def _calculate_frustration_density(
        self, pdb: "Pdb", ratio: float = 5
    ) -> FrustrationDensityResults:
        """Calculate the density and proportion of each type of contact."""
        logger.debug("Starting frustration density calculation")
        logger.debug(f"Using sphere radius: {ratio}Å")

        # Get CA atom coordinates
        ca_mask = pdb.atom["atom_name"] == "CA"
        ca_xyz = pdb.atom.loc[ca_mask, ["x", "y", "z"]].values

        logger.debug(f"Found {len(ca_xyz)} CA atoms")

        # Read contact data from FrustrationData directory instead of job_dir
        contacts_file = os.path.join(
            pdb.job_dir, "FrustrationData", "tertiary_frustration.dat"
        )  # Changed path
        logger.debug(f"Reading contacts from {contacts_file}")

        try:
            contacts_df = pd.read_csv(
                contacts_file,
                sep=r"\s+",
                header=None,
                skiprows=2,
                usecols=[0, 1, 4, 5, 6, 7, 8, 9, 18],
                names=["i", "j", "xi", "yi", "zi", "xj", "yj", "zj", "f_ij"],
            )
        except Exception as e:
            if self.debug:
                logger.debug(f"Failed to read contacts file: {e}", exc_info=True)
                logger.debug(f"Working directory: {os.getcwd()}")
                logger.debug(f"Job directory contents: {os.listdir(pdb.job_dir)}")
                logger.debug(
                    f"FrustrationData contents: {os.listdir(os.path.join(pdb.job_dir, 'FrustrationData'))}"
                )
            else:
                logger.error(f"Failed to read contacts file: {e}")
            raise

        # Calculate contact point coordinates
        contact_coords = np.column_stack(
            [
                (contacts_df["xj"] + contacts_df["xi"]) / 2,
                (contacts_df["yj"] + contacts_df["yi"]) / 2,
                (contacts_df["zj"] + contacts_df["zi"]) / 2,
            ]
        )

        frustration_values = contacts_df["f_ij"].values

        # Get residue info
        positions = pdb.equivalences.iloc[:, 2].tolist()
        res_chains = pdb.equivalences.iloc[:, 0].tolist()

        # Calculate densities for each CA atom
        densities = []

        # Write results to file directly in FrustrationData directory
        output_file = os.path.join(
            pdb.job_dir, "FrustrationData", f"{pdb.pdb_base}.pdb_{pdb.mode}_5adens"
        )
        logger.debug(f"Writing density results to {output_file}")

        with open(output_file, "w") as f:
            f.write(
                "Res ChainRes Total HighlyFrst NeutrallyFrst MinimallyFrst "
                "relHighlyFrustrated relNeutralFrustrated relMinimallyFrustrated\n"
            )

            for i, (ca_point, res_num, chain_id) in enumerate(
                zip(ca_xyz, positions, res_chains)
            ):
                # Calculate distances to all contact points
                distances = np.sqrt(np.sum((ca_point - contact_coords) ** 2, axis=1))

                # Count contacts within radius
                contacts_mask = distances < ratio
                total_density = np.sum(contacts_mask)

                if total_density > 0:
                    local_frustration = frustration_values[contacts_mask]
                    highly = np.sum(local_frustration <= -1)
                    neutral = np.sum(
                        (local_frustration > -1) & (local_frustration < 0.78)
                    )
                    minimally = np.sum(local_frustration >= 0.78)

                    rel_highly = highly / total_density
                    rel_neutral = neutral / total_density
                    rel_minimally = minimally / total_density
                else:
                    highly = neutral = minimally = 0
                    rel_highly = rel_neutral = rel_minimally = 0

                # Write results immediately
                f.write(
                    f"{res_num} {chain_id} {total_density} {highly} {neutral} {minimally} "
                    f"{rel_highly} {rel_neutral} {rel_minimally}\n"
                )

                density = FrustrationDensity(
                    residue_number=res_num,
                    chain_id=chain_id,
                    total_density=total_density,
                    highly_frustrated=highly,
                    neutrally_frustrated=neutral,
                    minimally_frustrated=minimally,
                    rel_highly_frustrated=rel_highly,
                    rel_neutrally_frustrated=rel_neutral,
                    rel_minimally_frustrated=rel_minimally,
                )
                densities.append(density)

        logger.debug("Frustration density calculation completed")

        results = FrustrationDensityResults(
            densities=densities,
            contact_coordinates=contact_coords,
            frustration_values=frustration_values,
        )

        # Save results to pickle file in FrustrationData directory
        density_pkl = os.path.join(
            pdb.job_dir, "FrustrationData", f"{pdb.pdb_base}.pdb_{pdb.mode}_density.pkl"
        )

        logger.debug(f"Saving density results to pickle file: {density_pkl}")
        try:
            with open(density_pkl, "wb") as f:
                pickle.dump(results, f)
            logger.debug("Successfully saved density results")
        except Exception as e:
            if self.debug:
                logger.debug(f"Failed to save density results: {str(e)}", exc_info=True)
            else:
                logger.error(f"Failed to save density results: {str(e)}")
            raise

        return results

    def _verify_environment(self) -> None:
        """Verify the execution environment is properly set up."""
        logger.debug("Verifying execution environment...")

        # Check Python version and environment
        logger.debug(f"Python executable: {sys.executable}")
        logger.debug(f"Python version: {sys.version}")

        # Check working directory and permissions
        cwd = os.getcwd()
        logger.debug(f"Current working directory: {cwd}")
        logger.debug(f"Directory writable: {os.access(cwd, os.W_OK)}")

        # Check results directory
        logger.debug(f"Results directory: {self.results_dir}")
        logger.debug(f"Results directory exists: {os.path.exists(self.results_dir)}")
        if os.path.exists(self.results_dir):
            logger.debug(
                f"Results directory writable: {os.access(self.results_dir, os.W_OK)}"
            )
            logger.debug(f"Results directory contents: {os.listdir(self.results_dir)}")

        # Check input file
        if self.pdb_file:
            logger.debug(f"Input PDB file: {self.pdb_file}")
            logger.debug(f"Input file exists: {os.path.exists(self.pdb_file)}")
            if os.path.exists(self.pdb_file):
                logger.debug(f"Input file size: {os.path.getsize(self.pdb_file)} bytes")
                logger.debug(
                    f"Input file readable: {os.access(self.pdb_file, os.R_OK)}"
                )

    def _verify_required_files(self, pdb: Pdb) -> None:
        """Verify all required files exist before processing."""
        logger.debug("Verifying required files...")

        # Define paths relative to FrustrationData directory
        frustration_dir = os.path.join(pdb.job_dir, "FrustrationData")

        required_files = {
            "job_dir": pdb.job_dir,
            "pdb_file": os.path.join(
                frustration_dir, f"{pdb.pdb_base}.pdb"
            ),  # Changed path
            "frustration_dir": frustration_dir,
        }

        if pdb.mode in ["configurational", "mutational"]:
            required_files.update(
                {
                    "density_file": os.path.join(
                        frustration_dir,  # Changed path
                        f"{pdb.pdb_base}.pdb_{pdb.mode}_5adens",
                    ),
                    "tertiary_frustration": os.path.join(
                        frustration_dir, "tertiary_frustration.dat"  # Changed path
                    ),
                }
            )

        # Log all paths being checked
        logger.debug("Checking required files:")
        for name, path in required_files.items():
            logger.debug(f"  {name}: {path}")
            logger.debug(f"  Exists: {os.path.exists(path)}")

        # Verify each file exists
        for name, path in required_files.items():
            if not os.path.exists(path):
                logger.error(f"Required {name} not found: {path}")
                if os.path.isdir(os.path.dirname(path)):
                    logger.error(
                        f"Directory contents: {os.listdir(os.path.dirname(path))}"
                    )
                raise FileNotFoundError(f"Required {name} not found: {path}")
            logger.debug(f"Found required {name}: {path}")

        # Update pdb_file path in calculator to point to the file in FrustrationData
        self.pdb_file = required_files["pdb_file"]
