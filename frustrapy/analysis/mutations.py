import multiprocessing
import logging
import os
import shutil
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import time
from tqdm import tqdm
from ..core import Pdb, SingleResidueData
from ..utils import log_execution_time
import sys

logger = logging.getLogger(__name__)


def _process_amino_acid(
    args: Tuple[str, "Pdb", int, str, bool, bool, bool, str]
) -> Dict[str, Any]:
    """
    Helper function to process a single amino acid mutation.

    Args:
        args: Tuple containing:
            aa (str): Amino acid code
            pdb (Pdb): Pdb frustration object
            res_num (int): Residue number to mutate
            chain (str): Chain identifier
            split (bool): Whether to split chains
            debug (bool): Debug mode flag
            is_glycine (bool): Whether the residue is glycine
            method (str): Mutation method ('threading' or 'modeller')

    Returns:
        Dict[str, Any]: Results of processing the amino acid mutation
    """
    aa, pdb, res_num, chain, split, debug, is_glycine, method = args
    logger = logging.getLogger(__name__)

    # Only log detailed mutation info if in debug mode
    if logger.getEffectiveLevel() <= logging.DEBUG:
        logger.debug(
            f"Processing variant {aa} for residue {res_num} in chain '{chain}'"
        )

    # Get indices of atoms for the residue to mutate
    residue_mask = (pdb.atom["res_num"] == res_num) & (pdb.atom["chain"] == chain)
    residue_indices = pdb.atom[residue_mask].index

    # Create new PDB with all atoms
    mutated_pdb = pdb.atom.copy()

    # Get current residue name before mutation
    current_res_name = mutated_pdb.loc[residue_indices[0], "res_name"]

    # Only modify if we're actually changing the residue type
    if current_res_name != aa:
        # Special handling for GLY -> X mutations
        if current_res_name == "GLY" and aa != "GLY":
            # Get coordinates of N, CA, C atoms
            n_coords = pdb.atom.loc[
                (pdb.atom["res_num"] == res_num)
                & (pdb.atom["chain"] == chain)
                & (pdb.atom["atom_name"] == "N"),
                ["x", "y", "z"],
            ].iloc[0]

            ca_coords = pdb.atom.loc[
                (pdb.atom["res_num"] == res_num)
                & (pdb.atom["chain"] == chain)
                & (pdb.atom["atom_name"] == "CA"),
                ["x", "y", "z"],
            ].iloc[0]

            c_coords = pdb.atom.loc[
                (pdb.atom["res_num"] == res_num)
                & (pdb.atom["chain"] == chain)
                & (pdb.atom["atom_name"] == "C"),
                ["x", "y", "z"],
            ].iloc[0]

            # Calculate CB coordinates using standard geometry
            # CB is placed 1.521 Å from CA at tetrahedral angle
            import numpy as np

            # Create vectors
            ca_n = np.array(
                [
                    n_coords.x - ca_coords.x,
                    n_coords.y - ca_coords.y,
                    n_coords.z - ca_coords.z,
                ]
            )
            ca_c = np.array(
                [
                    c_coords.x - ca_coords.x,
                    c_coords.y - ca_coords.y,
                    c_coords.z - ca_coords.z,
                ]
            )

            # Normalize vectors
            ca_n = ca_n / np.linalg.norm(ca_n)
            ca_c = ca_c / np.linalg.norm(ca_c)

            # Calculate CB direction (perpendicular to N-CA-C plane)
            cb_direction = np.cross(ca_n, ca_c)
            cb_direction = cb_direction / np.linalg.norm(cb_direction)

            # Adjust to tetrahedral angle (109.5 degrees)
            theta = np.radians(109.5)
            rot_axis = np.cross(ca_n, cb_direction)
            rot_axis = rot_axis / np.linalg.norm(rot_axis)

            # Rotation matrix
            c = np.cos(theta)
            s = np.sin(theta)
            t = 1 - c
            x, y, z = rot_axis

            rot_mat = np.array(
                [
                    [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
                    [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
                    [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
                ]
            )

            # Apply rotation
            cb_direction = np.dot(rot_mat, ca_n)

            # Calculate CB coordinates (1.521 Å from CA)
            cb_coords = (
                np.array([ca_coords.x, ca_coords.y, ca_coords.z]) + 1.521 * cb_direction
            )

            # Get backbone atoms
            backbone_mask = residue_mask & pdb.atom["atom_name"].isin(
                ["N", "CA", "C", "O"]
            )
            backbone_coords = pdb.atom[backbone_mask].copy()

            # Remove all atoms of the residue
            mutated_pdb = mutated_pdb.drop(index=residue_indices)

            # Add back backbone atoms with new residue name
            backbone_coords["res_name"] = aa
            mutated_pdb = pd.concat([mutated_pdb, backbone_coords])

            # Add CB atom
            cb_row = backbone_coords.iloc[0].copy()
            cb_row["atom_name"] = "CB"
            cb_row["x"] = cb_coords[0]
            cb_row["y"] = cb_coords[1]
            cb_row["z"] = cb_coords[2]
            cb_row["element"] = "C"

            mutated_pdb = pd.concat([mutated_pdb, pd.DataFrame([cb_row])])

        # X -> GLY mutations
        elif current_res_name != "GLY" and aa == "GLY":
            # Keep only N, CA, C, O for GLY
            backbone_atoms = ["N", "CA", "C", "O"]
            backbone_mask = residue_mask & pdb.atom["atom_name"].isin(backbone_atoms)
            backbone_indices = pdb.atom[backbone_mask].index

            # Remove all atoms except the basic backbone
            non_backbone_indices = residue_indices.difference(backbone_indices)
            if not non_backbone_indices.empty:
                mutated_pdb = mutated_pdb.drop(index=non_backbone_indices)

            # Update residue name
            mutated_pdb.loc[backbone_indices, "res_name"] = aa

        # Standard residue mutations (non-GLY to non-GLY)
        else:
            # Keep N, CA, C, O, CB
            backbone_atoms = ["N", "CA", "C", "O", "CB"]
            backbone_mask = residue_mask & pdb.atom["atom_name"].isin(backbone_atoms)
            backbone_indices = pdb.atom[backbone_mask].index

            # Remove all atoms except backbone
            non_backbone_indices = residue_indices.difference(backbone_indices)
            if not non_backbone_indices.empty:
                mutated_pdb = mutated_pdb.drop(index=non_backbone_indices)

            # Update residue name
            mutated_pdb.loc[backbone_indices, "res_name"] = aa

    # Sort by atom number to maintain proper order
    mutated_pdb = mutated_pdb.sort_index()

    # Save the mutated PDB
    if split:
        output_pdb_path = os.path.join(
            pdb.job_dir, f"{pdb.pdb_base}_{int(res_num)}_{aa}.pdb"
        )
    else:
        output_pdb_path = os.path.join(
            pdb.job_dir, f"{pdb.pdb_base}_{int(res_num)}_{aa}_{chain}.pdb"
        )

    # Ensure correct data types
    mutated_pdb["res_name"] = mutated_pdb["res_name"].astype(str)
    mutated_pdb["atom_name"] = mutated_pdb["atom_name"].astype(str)
    mutated_pdb["chain"] = mutated_pdb["chain"].astype(str)
    mutated_pdb["element"] = mutated_pdb["element"].astype(str)

    # Handle 'alt_loc' and 'insertion_code' columns
    for col in ["alt_loc", "insertion_code"]:
        if col in mutated_pdb.columns:
            mutated_pdb[col] = mutated_pdb[col].astype(str)
        else:
            mutated_pdb[col] = " "

    # Write PDB file
    with open(output_pdb_path, "w") as pdb_file:
        for _, row in mutated_pdb.iterrows():
            pdb_line = (
                f"{row['ATOM']:<6}"  # Record name, columns 1-6
                f"{int(row['atom_num']):>5}"  # Atom serial number, columns 7-11
                f" {row['atom_name']:<4}"  # Atom name, columns 13-16
                f"{row['alt_loc']:<1}"  # Alternate location indicator, column 17
                f"{row['res_name']:<3}"  # Residue name, columns 18-20
                f" {row['chain']:<1}"  # Chain ID, column 22
                f"{int(row['res_num']):>4}"  # Residue sequence number, columns 23-26
                f"{row['insertion_code']:<1}"  # Insertion code, column 27
                f"   "  # Empty columns 28-30
                f"{float(row['x']):>8.3f}"  # X coordinate, columns 31-38
                f"{float(row['y']):>8.3f}"  # Y coordinate, columns 39-46
                f"{float(row['z']):>8.3f}"  # Z coordinate, columns 47-54
                f"{float(row['occupancy']):>6.2f}"  # Occupancy, columns 55-60
                f"{float(row['b_factor']):>6.2f}"  # Temperature factor, columns 61-66
                f"          "  # Empty columns 67-76
                f"{row['element']:<2}"  # Element symbol, columns 77-78
                f"\n"
            )
            pdb_file.write(pdb_line)

    logger.debug(f"Saved mutated PDB to {output_pdb_path}")

    # Construct the output PDB base name including chain
    output_pdb_base = f"{pdb.pdb_base}_{int(res_num)}_{aa}_{chain}"

    # Calculate frustration for the mutated PDB
    logger.debug("Calculating frustration...")
    from .frustration import calculate_frustration

    # Pass is_mutation_calculation=True to suppress protocol logging
    calculate_frustration(
        pdb_file=output_pdb_path,
        mode=pdb.mode,
        results_dir=pdb.job_dir,
        graphics=False,
        visualization=False,
        chain=chain,
        debug=debug,
        is_mutation_calculation=True,  # Add this flag
    )

    # Store the frustration data
    logger.debug("Storing frustration data...")
    frustration_data_dir = os.path.join(
        pdb.job_dir,
        f"{output_pdb_base}.done",
        "FrustrationData",
    )

    mutations_dir = os.path.join(pdb.job_dir, "MutationsData")
    frustra_mut_file = os.path.join(
        mutations_dir, f"{pdb.mode}_Res{int(res_num)}_{method}_{chain}.txt"
    )

    if pdb.mode == "singleresidue":
        src_frusta_file = os.path.join(
            frustration_data_dir,
            f"{output_pdb_base}.pdb_singleresidue",
        )
        dst_frusta_file = os.path.join(
            mutations_dir,
            f"{output_pdb_base}.pdb_singleresidue",
        )

        shutil.move(src_frusta_file, dst_frusta_file)

        frustra_table = pd.read_csv(
            dst_frusta_file,
            sep="\s+",
            header=0,
            usecols=["Res", "ChainRes", "AA", "FrstIndex"],
        )
        # Ensure 'Res' column is integer for comparison
        frustra_table["Res"] = frustra_table["Res"].astype(int)
        frustra_table = frustra_table[
            (frustra_table["ChainRes"] == chain) & (frustra_table["Res"] == res_num)
        ]
        frustra_table.to_csv(
            frustra_mut_file, sep="\t", header=False, index=False, mode="a"
        )

    elif pdb.mode in ["configurational", "mutational"]:
        src_frusta_file = os.path.join(
            frustration_data_dir,
            f"{os.path.basename(output_pdb_path).split('.')[0]}.pdb_{pdb.mode}",
        )
        dst_frusta_file = os.path.join(
            mutations_dir,
            f"{os.path.basename(output_pdb_path).split('.')[0]}.pdb_{pdb.mode}",
        )

        shutil.move(src_frusta_file, dst_frusta_file)

        frustra_table = pd.read_csv(
            dst_frusta_file,
            sep="\s+",
            header=0,
            usecols=[
                "Res1",
                "Res2",
                "ChainRes1",
                "ChainRes2",
                "AA1",
                "AA2",
                "FrstIndex",
                "FrstState",
            ],
        )
        frustra_table = frustra_table[
            ((frustra_table["ChainRes1"] == chain) & (frustra_table["Res1"] == res_num))
            | (
                (frustra_table["ChainRes2"] == chain)
                & (frustra_table["Res2"] == res_num)
            )
        ]
        frustra_table.to_csv(
            frustra_mut_file, sep="\t", header=False, index=False, mode="a"
        )

    # Clean up temporary files
    logger.debug("Cleaning up temporary files...")
    temp_done_dir = os.path.join(
        pdb.job_dir, f"{os.path.basename(output_pdb_path).split('.')[0]}.done"
    )
    if os.path.exists(temp_done_dir):
        shutil.rmtree(temp_done_dir)

    if not debug:
        os.remove(output_pdb_path)

    # After processing the mutation and calculating frustration, add the data to pdb.Mutations
    mutation_key = f"Res_{res_num}_{chain}"

    # Initialize Mutations dict if it doesn't exist
    if not hasattr(pdb, "Mutations"):
        pdb.Mutations = {}
    if method not in pdb.Mutations:
        pdb.Mutations[method] = {}

    # Store mutation data
    pdb.Mutations[method][mutation_key] = {
        "Method": method,
        "Res": res_num,
        "Chain": chain,
        "File": frustra_mut_file,
    }

    # Return both the mutation data and updated pdb object
    return {
        "aa": aa,
        "frustra_mut_file": frustra_mut_file,
        "pdb": pdb,  # Add this to return the updated pdb object
    }


def mutate_res_parallel(
    pdb: "Pdb",
    res_num: int,
    chain: str,
    split: bool = True,
    method: str = "threading",
    pbar: Optional[tqdm] = None,
) -> "Pdb":
    """Parallel version of mutate_res that processes amino acid mutations concurrently."""
    start_time = time.time()
    logger.info(f"\nAnalyzing mutations for residue {res_num} in chain {chain}")

    # Validate inputs (same as mutate_res)
    if not isinstance(split, bool):
        logger.error("Split must be a boolean value!")
        raise ValueError("Split must be a boolean value!")

    if method not in ["threading", "modeller"]:
        logger.error(
            f"Invalid method '{method}'. Available methods: 'threading', 'modeller'."
        )
        raise ValueError("Method must be 'threading' or 'modeller'.")

    if not ((pdb.atom["res_num"] == res_num) & (pdb.atom["chain"] == chain)).any():
        logger.error(f"Residue number {res_num} in chain '{chain}' does not exist!")
        raise ValueError(f"Residue number {res_num} in chain '{chain}' does not exist!")

    if method == "modeller" and not split:
        logger.error("Complex modeling with Modeller is not available!")
        raise ValueError("Complex modeling with Modeller is not available!")

    # Setup output directory and file
    mutations_dir = os.path.join(pdb.job_dir, "MutationsData")
    os.makedirs(mutations_dir, exist_ok=True)
    frustra_mut_file = os.path.join(
        mutations_dir, f"{pdb.mode}_Res{int(res_num)}_{method}_{chain}.txt"
    )

    if os.path.exists(frustra_mut_file):
        os.remove(frustra_mut_file)

    # Write header to the output file
    with open(frustra_mut_file, "w") as f:
        if pdb.mode in ["configurational", "mutational"]:
            f.write("Res1 Res2 ChainRes1 ChainRes2 AA1 AA2 FrstIndex FrstState\n")
        elif pdb.mode == "singleresidue":
            f.write("Res ChainRes AA FrstIndex\n")

    # Define amino acid codes
    amino_acids = [
        "LEU",
        "ASP",
        "ILE",
        "ASN",
        "THR",
        "VAL",
        "ALA",
        "GLY",
        "GLU",
        "ARG",
        "LYS",
        "HIS",
        "GLN",
        "SER",
        "PRO",
        "PHE",
        "TYR",
        "MET",
        "TRP",
        "CYS",
    ]

    # Create our own progress bar if none was provided
    if pbar is None:
        pbar = tqdm(
            total=len(amino_acids),
            desc=f"Processing mutations for residue {res_num}",
            position=0,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )

    # Check if the residue is glycine
    is_glycine = (
        pdb.atom.loc[
            (pdb.atom["res_num"] == res_num)
            & (pdb.atom["chain"] == chain)
            & (pdb.atom["atom_name"] == "CA"),
            "res_name",
        ].iloc[0]
        == "GLY"
    )

    logger.debug("Starting parallel mutation processing")
    process_start = time.time()

    # Create process pool
    n_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_processes)

    # Create arguments list
    args_list = [
        (aa, pdb, res_num, chain, split, False, is_glycine, method)
        for aa in amino_acids
    ]

    # Process mutations in parallel with progress bar
    results = []
    try:
        for result in pool.imap_unordered(_process_amino_acid, args_list):
            results.append(result)
            # Update pdb object with mutation data from each result
            if "pdb" in result:
                # Update mutation data
                if not hasattr(pdb, "Mutations"):
                    pdb.Mutations = {}
                if method not in pdb.Mutations:
                    pdb.Mutations[method] = {}

                mutation_key = f"Res_{res_num}_{chain}"
                if mutation_key not in pdb.Mutations[method]:
                    pdb.Mutations[method][mutation_key] = result["pdb"].Mutations[
                        method
                    ][mutation_key]

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"residue": f"{res_num}"}, refresh=False)
    finally:
        if pbar is not None and pbar.disable is False:
            pbar.close()

    pool.close()
    pool.join()

    total_time = time.time() - start_time

    # Log only the final storage location
    logger.info(
        f"The frustration data for residue {res_num} is stored in {frustra_mut_file}"
    )

    return pdb


def mutate_res(
    pdb: "Pdb",
    res_num: int,
    chain: str,
    split: bool = True,
    method: str = "threading",
    debug: bool = False,
) -> "Pdb":
    """Serial version of amino acid mutation processing."""
    logger = logging.getLogger(__name__)
    start_time = time.time()
    method = method.lower()

    # Validate inputs (same validation as before)
    if not isinstance(split, bool):
        logger.error("Split must be a boolean value!")
        raise ValueError("Split must be a boolean value!")

    if method not in ["threading", "modeller"]:
        logger.error(
            f"Invalid method '{method}'. Available methods: 'threading', 'modeller'."
        )
        raise ValueError("Method must be 'threading' or 'modeller'.")

    if not ((pdb.atom["res_num"] == res_num) & (pdb.atom["chain"] == chain)).any():
        logger.error(f"Residue number {res_num} in chain '{chain}' does not exist!")
        raise ValueError(f"Residue number {res_num} in chain '{chain}' does not exist!")

    if method == "modeller" and not split:
        logger.error("Complex modeling with Modeller is not available!")
        raise ValueError("Complex modeling with Modeller is not available!")

    # Setup output directory and file
    mutations_dir = os.path.join(pdb.job_dir, "MutationsData")
    os.makedirs(mutations_dir, exist_ok=True)
    frustra_mut_file = os.path.join(
        mutations_dir, f"{pdb.mode}_Res{int(res_num)}_{method}_{chain}.txt"
    )

    if os.path.exists(frustra_mut_file):
        os.remove(frustra_mut_file)

    # Write header to the output file
    with open(frustra_mut_file, "w") as f:
        if pdb.mode in ["configurational", "mutational"]:
            f.write("Res1 Res2 ChainRes1 ChainRes2 AA1 AA2 FrstIndex FrstState\n")
        elif pdb.mode == "singleresidue":
            f.write("Res ChainRes AA FrstIndex\n")

    # Define amino acid codes
    amino_acids = [
        "LEU",
        "ASP",
        "ILE",
        "ASN",
        "THR",
        "VAL",
        "ALA",
        "GLY",
        "GLU",
        "ARG",
        "LYS",
        "HIS",
        "GLN",
        "SER",
        "PRO",
        "PHE",
        "TYR",
        "MET",
        "TRP",
        "CYS",
    ]

    # Check if the residue is glycine
    is_glycine = (
        pdb.atom.loc[
            (pdb.atom["res_num"] == res_num)
            & (pdb.atom["chain"] == chain)
            & (pdb.atom["atom_name"] == "CA"),
            "res_name",
        ].iloc[0]
        == "GLY"
    )

    logger.debug("Starting serial mutation processing")
    process_start = time.time()

    # Add progress bar for serial processing
    with tqdm(
        total=len(amino_acids), desc=f"Processing mutations for residue {res_num}"
    ) as pbar:
        for aa in amino_acids:
            if debug:
                logger.debug(f"Processing mutation to {aa}")
            aa_start = time.time()
            result = _process_amino_acid(
                (aa, pdb, res_num, chain, split, debug, is_glycine, method)
            )
            # Update pdb object with mutation data
            if "pdb" in result:
                if not hasattr(pdb, "Mutations"):
                    pdb.Mutations = {}
                if method not in pdb.Mutations:
                    pdb.Mutations[method] = {}

                mutation_key = f"Res_{res_num}_{chain}"
                if mutation_key not in pdb.Mutations[method]:
                    pdb.Mutations[method][mutation_key] = result["pdb"].Mutations[
                        method
                    ][mutation_key]

            if debug:
                logger.debug(f"Processed {aa} in {time.time() - aa_start:.2f} seconds")
            pbar.update(1)

    process_time = time.time() - process_start

    # Update the pdb object with mutation information
    if not hasattr(pdb, "Mutations"):
        pdb.Mutations = {}
    if method not in pdb.Mutations:
        pdb.Mutations[method] = {}

    mutation_key = f"Res_{res_num}_{chain}"
    pdb.Mutations[method][mutation_key] = {
        "Method": method,
        "Res": res_num,
        "Chain": chain,
        "File": frustra_mut_file,
    }

    logger.debug(
        f"The frustration data for residue {res_num} is stored in {frustra_mut_file}"
    )

    total_time = time.time() - start_time
    logger.debug(f"Serial processing completed in {process_time:.2f} seconds")
    logger.debug(f"Total mutation time (including setup): {total_time:.2f} seconds")
    logger.debug(
        f"Average time per amino acid: {process_time/len(amino_acids):.2f} seconds"
    )

    return pdb
