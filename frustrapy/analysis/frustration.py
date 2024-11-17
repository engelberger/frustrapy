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


logger = logging.getLogger(__name__)


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
    pbar: Optional[tqdm] = None,
    is_mutation_calculation: Optional[bool] = False,
) -> Tuple["Pdb", Dict, Optional[FrustrationDensityResults]]:
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
    """

    # Set flag for mutation calculations to suppress logging
    is_mutation_calculation = mode == "singleresidue" and residues is not None

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
        is_mutation_calculation=is_mutation_calculation,
    )

    logger.debug("Starting calculation")
    pdb, plots, density_results = calculator.calculate()
    logger.debug("Calculation completed")

    # Save single residue data if in singleresidue mode
    if mode == "singleresidue" and residues:
        try:
            # Organize and save data
            residues_analyzed = {}
            for chain_id in residues:
                residues_analyzed[chain_id] = [
                    {"res_num": res} for res in residues[chain_id]
                ]

            organized_data = organize_single_residue_data(pdb, residues_analyzed)

            # Save to pickle file
            output_dir = os.path.join(pdb.job_dir, "SingleResidueData")
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(
                output_dir, f"{pdb.pdb_base}_single_residue_data.pkl"
            )
            with open(output_file, "wb") as f:
                pickle.dump(organized_data, f)

            logger.info(f"Saved single residue analysis data to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save single residue data: {str(e)}")
            # Continue execution since this is not critical

    # Clean up flag after calculation
    if hasattr(calculate_frustration, "in_mutation_calculation"):
        delattr(calculate_frustration, "in_mutation_calculation")

    return pdb, plots, density_results


@log_execution_time
def dir_frustration(
    pdbs_dir: str,
    order_list: Optional[List[str]] = None,
    chain: Optional[Union[str, List[str]]] = None,
    residues: Optional[Dict[str, List[int]]] = None,  # Add residues parameter
    electrostatics_k: Optional[float] = None,
    seq_dist: int = 12,
    mode: str = "configurational",
    graphics: bool = True,
    visualization: bool = True,
    results_dir: str = None,
    debug: bool = False,
) -> None:
    """Calculate local energy frustration for all protein structures in one directory."""

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

    if calculation_enabled:
        if order_list is None:
            order_list = [
                f for f in os.listdir(pdbs_dir) if f.endswith((".pdb", ".PDB"))
            ]

        # Create a dictionary to store the plots for each PDB
        plots_dir_dict = {}

        for pdb_file in order_list:
            pdb_path = os.path.join(pdbs_dir, pdb_file)
            pdb, plots, density_results = calculate_frustration(
                pdb_file=pdb_path,
                chain=chain,
                residues=residues,  # Pass residues parameter
                electrostatics_k=electrostatics_k,
                seq_dist=seq_dist,
                mode=mode,
                graphics=graphics,
                visualization=visualization,
                results_dir=results_dir,
                debug=debug,
            )
            # Add the plots to the dictionary
            plots_dir_dict[pdb.pdb_base] = plots

        with open(modes_log_file, "a") as f:
            f.write(mode + "\n")

        logger.debug("\n\n****Storage information****")
        logger.debug(
            f"Frustration data for all Pdb's directory {pdbs_dir} are stored in {results_dir}"
        )
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
) -> "Dynamic":
    """
    Calculates local energetic frustration for a dynamic.

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
    dir_frustration(
        pdbs_dir=pdbs_dir,
        order_list=order_list,
        chain=chain,
        electrostatics_k=electrostatics_k,
        seq_dist=seq_dist,
        mode=mode,
        results_dir=results_dir,
    )

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
    frustration_table = pd.read_csv(f"{frustration_data_path}", sep="\s+", header=None)

    # Define column names based on the mode
    if pdb.mode == "singleresidue":
        frustration_table.columns = [
            "Res",
            "ChainRes",
            "DensityRes",
            "AA",
            "NativeEnergy",
            "DecoyEnergy",
            "SDEnergy",
            "FrstIndex",
        ]
    else:  # For configurational or mutational
        frustration_table.columns = [
            "Res1",
            "Res2",
            "ChainRes1",
            "ChainRes2",
            "AA1",
            "AA2",
            "NativeEnergy",
            "DecoyEnergy",
            "SDEnergy",
            "FrstIndex",
            "FrstState",
        ]

    if chain is not None:
        frustration_table = frustration_table[frustration_table["ChainRes"].isin(chain)]
    if res_num is not None:
        frustration_table = frustration_table[
            (frustration_table["Res"].isin(res_num))
            | (frustration_table["Res1"].isin(res_num))
            | (frustration_table["Res2"].isin(res_num))
        ]
    return frustration_table
