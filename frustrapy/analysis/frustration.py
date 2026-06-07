import os
import sys
import subprocess
import tempfile
import logging
import shutil
from typing import Optional, List, Dict, Union, Tuple, Any
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select
import numpy as np
import glob
import pickle
from ..core import Pdb, Dynamic
from ..utils import log_execution_time
from tqdm.auto import tqdm  # Make sure to use tqdm.auto for better compatibility
from .frustration_calculator import FrustrationCalculator, FrustrationDensityResults
from ..utils.helpers import organize_single_residue_data, pdb_equivalences, renum_files
from .exceptions import FileOperationError, MissingBackboneAtomError
from ..utils.ui import display_error, display_overwrite_warning, display_success, display_warning  # Import Rich display utils


logger = logging.getLogger(__name__)


def _dir_frustration_worker(kwargs: Dict) -> Tuple[str, Dict, Optional[FrustrationDensityResults]]:
    """Top-level (picklable) worker for the parallel PDB-batch path of
    :func:`dir_frustration` (Phase 6 Lever 2). Runs one structure and returns only
    the lightweight pieces the parent keeps — its ``pdb_base`` key, the plots dict,
    and the density results — so the heavy ``Pdb`` object is not shipped back.
    """
    pdb, plots, density_results, _single = calculate_frustration(**kwargs)
    return pdb.pdb_base, plots, density_results


@log_execution_time
def calculate_frustration(
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
    pbar: Optional[tqdm] = None,
    is_mutation_calculation: Optional[bool] = False,
) -> Tuple["Pdb", Dict, Optional[FrustrationDensityResults], Optional[Dict]]:
    """Calculate local energy frustration for a protein structure.

    Args:
        pdbs_dir (str): Directory containing all protein structures. The full path to the file is needed.
        order_list (Optional[List[str]]): Ordered list of PDB files to calculate frustration. If it is None, frustration is
        calculated for all PDBs.
        chain (Optional[Union[str, List[str]]]): Chain or Chains of the protein structure.
        residues (Optional[Dict[str, List[int]]]): Dictionary mapping chain IDs to lists of residue numbers to analyze.
        electrostatics_k (Optional[float]): K constant to use in the electrostatics Mode.
        seq_dist (int): Sequence at which contacts are considered to interact (3 or 12).
        mode (str): Local frustration index to be calculated (configurational, mutational, singleresidue).
        graphics (bool): The corresponding graphics are made.
        visualization (bool): Make visualizations, including pymol.
        results_dir (str): Path to the folder where results will be stored.
        debug (bool): Debug mode flag.
        n_cpus (Optional[int]): Number of CPU cores to use for mutation analysis (None = all available).
    """

    # Combine external flag for nested mutation calls and singleresidue detection
    is_mutation_calculation = is_mutation_calculation or (mode == "singleresidue" and residues is not None)
    
    # Also check for environment variable set by mutation processing.
    # FIXME (fix-on-merge flag): this FRUSTRAPY_MUTATION_CALCULATION env-var is a
    # process-global side channel for suppressing the success banner on nested
    # mutation calls. It is fragile under parallelism (shared across workers/runs);
    # prefer threading the `is_mutation_calculation` argument explicitly instead.
    if os.environ.get('FRUSTRAPY_MUTATION_CALCULATION') == 'True':
        is_mutation_calculation = True

    # Only log protocol for main calculations, not individual mutations
    if is_mutation_calculation:
        logger.debug(f"\nRunning Frustration Protocol:")
        logger.debug(f"- Analysis Mode: {mode}")
        if pdb_file:
            logger.debug(f"- Input Structure: {os.path.basename(pdb_file)}")
        if chain:
            logger.debug(f"- Analyzing Chain(s): {chain}")
        if residues:
            for chain_id, res_list in residues.items():
                logger.debug(f"- Residues for Chain {chain_id}: {res_list}")
        logger.info(f"- Sequence Distance: {seq_dist}")
        if electrostatics_k is not None:
            logger.debug(f"- Electrostatics K: {electrostatics_k}")
        logger.debug(f"- Graphics Generation: {'Enabled' if graphics else 'Disabled'}")
        logger.debug(
            f"- Structure Visualization: {'Enabled' if visualization else 'Disabled'}\n"
        )

    logger.debug("Starting frustration calculation")

    # Validate PDB file existence if provided
    if pdb_file is not None:
        pdb_file = os.path.abspath(pdb_file)
        logger.debug(f"Using PDB file: {pdb_file}")
        if not os.path.exists(pdb_file):
            logger.error(f"PDB file not found: {pdb_file}")
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")

    # Make results_dir absolute path if provided
    if results_dir is not None:
        results_dir = os.path.abspath(results_dir)
        logger.debug(f"Using results directory: {results_dir}")

    logger.debug(f"Initializing FrustrationCalculator with mode: {mode}")
    calculator = FrustrationCalculator(
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
        is_mutation_calculation=is_mutation_calculation,
    )

    logger.debug("Starting calculation")
    try:
        pdb, plots, density_results = calculator.calculate()
        logger.debug("Calculation completed")
        # Display success message
        success_msg = f"Frustration calculation completed successfully for {pdb.pdb_base}.\nResults stored in: {pdb.job_dir}"
        # Only show the rich success banner for top-level calculations
        if not is_mutation_calculation:
            display_success(success_msg)
    except MissingBackboneAtomError as e:
        # For missing backbone atoms, display a warning
        warning_message = str(e)
        suggestions = [
            "Try to repair your PDB file by adding missing atoms with software like PyMOL or MODELLER.",
            "You can also remove the problematic residues from your PDB file if they're not critical.",
            "For automated repairs, tools like PDB-tools (https://github.com/haddocking/pdb-tools) can help."
        ]
        
        # Show the warning message
        display_warning(warning_message, title="Missing Backbone Atoms", suggestions=suggestions)
        
        # Log the error with full context if in debug mode
        if debug:
            logger.debug(f"Frustration calculation failed due to missing backbone atoms: {e}", exc_info=True)
        else:
            logger.error(f"Frustration calculation failed due to missing backbone atoms: {e}")

        # Library code must not call sys.exit(): propagate so the caller decides
        # how to handle it (CLAUDE.md §8). The warning has already been displayed.
        raise
    except FileOperationError as e:
        if "Destination file already exists" in e.message and not overwrite:
            # Display specific overwrite warning
            display_overwrite_warning(e)
        else:
            # Display general file operation error
            display_error(e, is_debug=debug)
        # Library code must not call sys.exit(): propagate the error (CLAUDE.md §8).
        raise
    except Exception as e:
        # Display any other error
        display_error(e, is_debug=debug)
        logger.error(f"An unexpected error occurred during frustration calculation: {e}", exc_info=True)
        raise

    single_residue_data = None
    # Save single residue data if in singleresidue mode
    if mode == "singleresidue" and residues:
        try:
            # Organize and save data
            residues_analyzed = {}
            for chain_id in residues:
                residues_analyzed[chain_id] = [
                    {"res_num": res} for res in residues[chain_id]
                ]

            single_residue_data = organize_single_residue_data(pdb, residues_analyzed)

            # Save to pickle file
            output_dir = os.path.join(pdb.job_dir, "SingleResidueData")
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(
                output_dir, f"{pdb.pdb_base}_single_residue_data.pkl"
            )
            with open(output_file, "wb") as f:
                pickle.dump(single_residue_data, f)

            logger.info(f"Saved single residue analysis data to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save single residue data: {str(e)}")
            single_residue_data = None
            # Continue execution since this is not critical

    # Clean up flag after calculation
    if hasattr(calculate_frustration, "in_mutation_calculation"):
        delattr(calculate_frustration, "in_mutation_calculation")

    return pdb, plots, density_results, single_residue_data


@log_execution_time
def dir_frustration(
    pdbs_dir: str,
    order_list: Optional[List[str]] = None,
    chain: Optional[Union[str, List[str]]] = None,
    residues: Optional[Dict[str, List[int]]] = None,
    electrostatics_k: Optional[float] = None,
    seq_dist: int = 12,
    mode: str = "configurational",
    graphics: bool = True,
    visualization: bool = True,
    results_dir: str = None,
    debug: bool = False,
    n_cpus: Optional[int] = None,
    n_procs: Optional[int] = None,
) -> Tuple[Dict, Optional[FrustrationDensityResults]]:
    """Calculate local energy frustration for all protein structures in one directory.

    Args:
        n_cpus (Optional[int]): inner CPU budget per structure (mutation pool).
        n_procs (Optional[int]): number of structures to process concurrently
            (Phase 6 Lever 2, outer batch axis). ``None``/``1`` keeps the historic
            serial loop. When ``> 1`` the structures run in a
            ``ProcessPoolExecutor`` and each inner mutation pool is throttled to a
            **shared** core budget — ``inner = max(1, cpu_count // n_procs)`` — so
            the two nested pools never oversubscribe to ``cpu_count²`` workers.
    """

    # Add protocol information logging for directory analysis
    logger.info(f"\nRunning Directory Frustration Analysis:")
    logger.info(f"- Analysis Mode: {mode}")
    logger.info(f"- Input Directory: {os.path.basename(pdbs_dir)}")
    if order_list:
        logger.info(f"- Number of structures to analyze: {len(order_list)}")
    if chain:
        logger.info(f"- Analyzing Chain(s): {chain}")
    if residues:
        for chain_id, res_list in residues.items():
            logger.info(f"- Residues for Chain {chain_id}: {res_list}")
    logger.info(f"- Sequence Distance: {seq_dist}")
    if electrostatics_k is not None:
        logger.info(f"- Electrostatics K: {electrostatics_k}")
    logger.info(f"- Graphics Generation: {'Enabled' if graphics else 'Disabled'}")
    logger.info(
        f"- Structure Visualization: {'Enabled' if visualization else 'Disabled'}"
    )
    logger.info(f"- Results will be saved to: {results_dir}\n")

    if results_dir is None:
        results_dir = os.path.join(tempfile.gettempdir(), "")
    elif not os.path.exists(results_dir):
        # Make the results directory and absolute path
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir)
        logger.debug(f"The results directory {results_dir} has been created.")

    if results_dir[-1] != "/":
        results_dir += "/"

    if pdbs_dir[-1] != "/":
        pdbs_dir += "/"

    # Make the pdbs_dir absolute path
    pdbs_dir = os.path.abspath(pdbs_dir)

    if electrostatics_k is not None and not isinstance(electrostatics_k, (int, float)):
        raise ValueError("Electrostatic_K must be a numeric value!")

    if seq_dist != 3 and seq_dist != 12:
        raise ValueError("SeqDist must take the value 3 or 12!")

    mode = mode.lower()
    if mode not in ["configurational", "mutational", "singleresidue"]:
        raise ValueError(
            f"{mode} frustration index doesn't exist. The frustration indexes are: configurational, mutational or singleresidue!"
        )

    if graphics not in [True, False]:
        raise ValueError("Graphics must be a boolean value!")

    if visualization not in [True, False]:
        raise ValueError("Visualization must be a boolean value!")

    calculation_enabled = True
    modes_log_file = os.path.join(results_dir, "Modes.log")
    if os.path.exists(modes_log_file):
        logger.debug(f"The modes log file {modes_log_file} exists.")
        with open(modes_log_file, "r") as f:
            modes = f.read().splitlines()
        if mode in modes:
            calculation_enabled = False

    # P0-7: initialize before the loop so an empty order_list (or a skipped
    # calculation) cannot raise UnboundLocalError on the return below.
    plots_dir_dict = {}
    density_results = None

    if calculation_enabled:
        if order_list is None:
            order_list = [
                f for f in os.listdir(pdbs_dir) if f.endswith((".pdb", ".PDB"))
            ]

        # Build one kwargs bundle per structure (identical to the serial call).
        common_kwargs = dict(
            chain=chain,
            residues=residues,
            electrostatics_k=electrostatics_k,
            seq_dist=seq_dist,
            mode=mode,
            graphics=graphics,
            visualization=visualization,
            results_dir=results_dir,
            debug=debug,
        )

        from ..utils.concurrency import (
            cpu_budget,
            get_pool_context,
            pool_worker_initializer,
            resolve_concurrency,
        )

        cores = cpu_budget()
        # Default stays serial (n_procs falsy => one structure at a time, byte-for-
        # byte the historical path). When batch parallelism is requested, the
        # single shared budget splits `cores` into outer = concurrent structures
        # and inner = CPUs each structure's inner mutation pool may use, with
        # outer * inner <= cores.
        if not n_procs:
            n_procs_eff, inner_cpus = 1, max(1, cores)
        else:
            n_procs_eff, inner_cpus = resolve_concurrency(
                n_procs, len(order_list), cores
            )

        if n_procs_eff > 1:
            # Lever 2: outer batch parallelism. The shared budget above keeps the
            # outer pool and each inner mutation pool together within `cores`
            # workers (never `cores²`). cf. CLAUDE.md / ROADMAP "never stack two
            # cpu_count() pools".
            logger.info(
                f"- Parallel batch: {n_procs_eff} structures concurrent x "
                f"{inner_cpus} inner CPU(s) (budget {cores} cores)"
            )
            from concurrent.futures import ProcessPoolExecutor

            jobs = [
                {
                    **common_kwargs,
                    "pdb_file": os.path.join(pdbs_dir, pf),
                    "n_cpus": inner_cpus,
                }
                for pf in order_list
            ]
            with ProcessPoolExecutor(
                max_workers=n_procs_eff,
                mp_context=get_pool_context(),
                initializer=pool_worker_initializer,
            ) as ex:
                for pdb_base, plots, density_results in ex.map(
                    _dir_frustration_worker, jobs
                ):
                    plots_dir_dict[pdb_base] = plots
        else:
            for pdb_file in order_list:
                pdb_path = os.path.join(pdbs_dir, pdb_file)
                # Update unpacking to handle 4 return values
                pdb, plots, density_results, single_res_data = calculate_frustration(
                    pdb_file=pdb_path,
                    n_cpus=n_cpus,
                    **common_kwargs,
                )
                # Add the plots to the dictionary
                plots_dir_dict[pdb.pdb_base] = plots

        with open(modes_log_file, "a") as f:
            f.write(mode + "\n")

        logger.debug("\n\n****Storage information****")
        logger.debug(
            f"Frustration data for all Pdb's directory {pdbs_dir} are stored in {results_dir}"
        )

    # P0-6: always return a 2-tuple, including when the mode was already logged
    # (calculation skipped) — the pre-fix code fell through and returned None.
    return plots_dir_dict, density_results


@log_execution_time
def dynamic_frustration(
    pdbs_dir: str,
    order_list: Optional[List[str]] = None,
    chain: Optional[str] = None,
    electrostatics_k: Optional[float] = None,
    seq_dist: int = 12,
    mode: str = "configurational",
    gifs: bool = False,
    results_dir: Optional[str] = None,
    n_cpus: Optional[int] = None,
    n_procs: Optional[int] = None,
) -> "Dynamic":
    """
    Calculates local energetic frustration for a trajectory.

    Args:
        pdbs_dir (str): Directory containing all protein structures. The full path to the file is needed.
        order_list (Optional[List[str]]): Ordered list of PDB files to calculate frustration. If it is None, frustration is calculated for all PDBs. Default: None.
        chain (Optional[str]): Chain of the protein structure. Default: None.
        electrostatics_k (Optional[float]): K constant to use in the electrostatics Mode. Default: None (no electrostatics is considered).
        seq_dist (int): Sequence at which contacts are considered to interact (3 or 12). Default: 12.
        mode (str): Local frustration index to be calculated (configurational, mutational, singleresidue). Default: configurational.
        gifs (bool): If it is True, the contact map gifs and 5 adens proportion of all the frames of the dynamic will be stored, otherwise they will not be stored. Default: False.
        results_dir (Optional[str]): Path to the folder where results will be stored. If not specified, it will be stored in the directory returned by tempdir(). Default: None.

    Returns:
        Dynamic: Dynamic frustration object.
    """
    if results_dir is None:
        results_dir = os.path.join(tempfile.gettempdir(), "")
    elif not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logger.debug(f"The results directory {results_dir} has been created.")

    if results_dir[-1] != "/":
        results_dir += "/"

    if pdbs_dir[-1] != "/":
        pdbs_dir += "/"

    if electrostatics_k is not None and not isinstance(electrostatics_k, (int, float)):
        raise ValueError("Electrostatic_K must be a numeric value!")

    if seq_dist != 3 and seq_dist != 12:
        raise ValueError("SeqDist must take the value 3 or 12!")

    mode = mode.lower()
    if mode not in ["configurational", "mutational", "singleresidue"]:
        raise ValueError(
            f"{mode} frustration index doesn't exist. The frustration indexes are: configurational, mutational or singleresidue!"
        )

    if gifs not in [True, False]:
        raise ValueError("Graphics must be a boolean value!")

    if order_list is None:
        order_list = [f for f in os.listdir(pdbs_dir) if f.endswith(".pdb")]

    logger.debug(
        "-----------------------------Object Dynamic Frustration-----------------------------"
    )
    dynamic = Dynamic(
        pdbs_dir=pdbs_dir,
        order_list=order_list,
        chain=chain,
        electrostatics_k=electrostatics_k,
        seq_dist=seq_dist,
        mode=mode,
        results_dir=results_dir,
    )

    logger.debug(
        "-----------------------------Calculating Dynamic Frustration-----------------------------"
    )
    # P1-16: capture the per-frame return instead of discarding it. Frames are an
    # embarrassingly-parallel axis — `n_procs` runs them concurrently under the
    # same shared core budget as the batch path (Lever 2).
    frames_plots, frames_density = dir_frustration(
        pdbs_dir=pdbs_dir,
        order_list=order_list,
        chain=chain,
        electrostatics_k=electrostatics_k,
        seq_dist=seq_dist,
        mode=mode,
        results_dir=results_dir,
        n_cpus=n_cpus,
        n_procs=n_procs,
    )
    # Expose per-frame results on the Dynamic object (one plots entry per frame).
    dynamic.frames_plots = frames_plots
    dynamic.frames_density = frames_density

    logger.debug("\n\n****Storage information****")
    logger.debug(f"The frustration of the full dynamic is stored in {results_dir}")

    if gifs:
        if mode == "configurational" or mode == "mutational":
            # This visualization functions are not re implemented in the new version
            # raise NotImplementedError("Visualization functions for dynamics are not implemented in the new version.")
            raise NotImplementedError(
                "Visualization functions for dynamics are not implemented in the new version."
            )
            # gif_5adens_proportions(dynamic)
            # gif_contact_map(dynamic)

    return dynamic


def get_frustration(
    pdb: Pdb, res_num: Optional[List[int]] = None, chain: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Returns the frustration of all Pdb residues, of a specific Chain or residue (Resno).
    By default, the complete Pdb frustration table is obtained and returned.

    Args:
        pdb (Pdb): Pdb frustration object obtained by calculate_frustration().
        res_num (Optional[List[int]]): Specific residues in Pdb. Default: None.
        chain (Optional[List[str]]): Specific chains in Pdb. Default: None.

    Returns:
        pd.DataFrame: Frustration table.
    """
    frustration_data_path = (
        f"{pdb.job_dir}/FrustrationData/{pdb.pdb_base}.pdb_{pdb.mode}"
    )
    # The written table already carries a header row with the canonical column names
    # (14 cols for configurational/mutational, 8 for singleresidue). Read those names
    # directly rather than re-asserting a hand-maintained list that drifted out of sync
    # with the on-disk format (it omitted DensityRes1/DensityRes2/Welltype).
    frustration_table = pd.read_csv(f"{frustration_data_path}", sep=r"\s+")

    # Column names differ by mode: singleresidue has Res/ChainRes; configurational
    # and mutational have Res1/Res2/ChainRes1/ChainRes2. Filtering the wrong set of
    # columns raises KeyError, so branch on pdb.mode (P0-5).
    if chain is not None:
        if pdb.mode == "singleresidue":
            frustration_table = frustration_table[
                frustration_table["ChainRes"].isin(chain)
            ]
        else:
            frustration_table = frustration_table[
                frustration_table["ChainRes1"].isin(chain)
                | frustration_table["ChainRes2"].isin(chain)
            ]
    if res_num is not None:
        if pdb.mode == "singleresidue":
            frustration_table = frustration_table[
                frustration_table["Res"].isin(res_num)
            ]
        else:
            frustration_table = frustration_table[
                frustration_table["Res1"].isin(res_num)
                | frustration_table["Res2"].isin(res_num)
            ]
    return frustration_table
