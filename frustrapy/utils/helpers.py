import os
import sys
import subprocess
import logging
from typing import Optional, Dict, Any, List, Tuple
from Bio.PDB import PDBParser
import pandas as pd
from ..core import Pdb
from ..core.data_classes import SingleResidueData


logger = logging.getLogger(__name__)


def get_os() -> str:
    """
    Get the operating system on which the package is running.
    """
    sysinf = os.uname()
    if sysinf:
        os_name = sysinf.sysname
        if os_name == "Darwin":
            os_name = "osx"
    else:
        os_name = sys.platform
        if os_name.startswith("darwin"):
            os_name = "osx"
        elif os_name.startswith("linux"):
            os_name = "linux"
    return os_name.lower()


def check_backbone_complete(pdb: "Pdb") -> bool:
    """
    Checks the backbone of a given protein structure to be processed by the package pipeline.

    Args:
        pdb (Pdb): Pdb frustration object.

    Returns:
        bool: Flag indicating if the backbone should be completed.
    """
    backbone_atoms = ["N", "CA", "C", "O", "CB"]
    atom_df = pdb.atom[pdb.atom["type"] == "ATOM"]
    atom_res = atom_df[atom_df["atom_name"].isin(backbone_atoms)]

    ca_count = len(atom_res[atom_res["atom_name"] == "CA"])
    o_count = len(atom_res[atom_res["atom_name"] == "O"])
    cb_count = len(atom_res[atom_res["atom_name"] == "CB"])
    gly_count = len(pdb.equivalences[pdb.equivalences["res_name"] == "GLY"])
    n_count = len(atom_res[atom_res["atom_name"] == "N"])

    complete = (
        ca_count == len(pdb.equivalences)
        and o_count == len(pdb.equivalences)
        and (cb_count + gly_count) == len(pdb.equivalences)
        and n_count == len(pdb.equivalences)
    )

    return complete


def complete_backbone(pdb: "Pdb") -> bool:
    """
    Completes the backbone of a given protein structure to be processed by the package pipeline.

    Args:
        pdb (Pdb): Pdb Frustration object.

    Returns:
        bool: Flag indicating if the backbone was completed.
    """
    completed = False
    if not check_backbone_complete(pdb):
        missing_atoms_script = os.path.join(pdb.scripts_dir, "MissingAtoms.py")
        pdb_file = os.path.join(pdb.job_dir, f"{pdb.pdb_base}.pdb")
        subprocess.run(["python3", missing_atoms_script, pdb.job_dir, pdb_file])

        completed_pdb_file = os.path.join(pdb.job_dir, f"{pdb.pdb_base}.pdb_completed")
        os.rename(completed_pdb_file, pdb_file)
        completed = True

    return completed


def pdb_equivalences(pdb_file: str, output_dir: str) -> pd.DataFrame:
    """
    Generates auxiliary files in the execution of the script,
    numerical equivalences between the PDB and its sequence.

    Args:
        pdb_file (str): Path to the PDB file.
        output_dir (str): Directory where the output file will be saved.

    Returns:
        pd.DataFrame: DataFrame containing PDB equivalences.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    equivalences = []

    for model in structure:
        for chain in model:
            seq_id = 1
            for residue in chain:
                if residue.id[0] == " ":
                    res_num = residue.id[1]
                    res_name = residue.resname
                    chain_id = chain.id
                    equivalences.append([chain_id, seq_id, res_num, res_name])
                    seq_id += 1

    equivalences_df = pd.DataFrame(
        equivalences, columns=["Chain", "SeqID", "ResNum", "ResName"]
    )

    logger.debug(f"Processing PDB file: {pdb_file}")
    pdb_file_base = os.path.basename(pdb_file)
    pdb_filename = pdb_file_base
    logger.debug(f"PDB filename: {pdb_filename}")
    output_path = os.path.join(output_dir, f"{pdb_filename}_equivalences.txt")
    logger.debug(f"Output path: {output_path}")

    equivalences_df.to_csv(output_path, sep="\t", index=False, header=False)
    logger.debug(f"Saved equivalences to {output_path}")

    with open(os.path.join(output_dir, "commands.help"), "a") as f:
        f.write(f"\n{output_path} equivalences saved")
    return equivalences_df


def replace_expr(pattern: str, replacement: str, file: str) -> None:
    """
    Search and replace pattern by replacement in the file lines.

    Args:
        pattern (str): Pattern string to replace.
        replacement (str): The character string to be replaced.
        file (str): Full path of the file where the pattern will be replaced.
    """
    with open(file, "r") as f:
        document = f.readlines()
    document = [line.replace(pattern, replacement) for line in document]
    with open(file, "w") as f:
        f.writelines(document)


def renum_files(
    job_id: str, job_dir: str, mode: str, equivalences_file: Optional[str] = None
) -> None:
    """Process and renumber files."""
    logger.debug("Processing renum files...")

    # Use provided equivalences file name or construct default
    if equivalences_file is None:
        equivalences_file = f"{job_id}.pdb_equivalences.txt"

    equivalence_file_path = os.path.join(job_dir, equivalences_file)
    logger.debug(f"Using equivalences file: {equivalence_file_path}")

    if not os.path.exists(equivalence_file_path):
        logger.error(f"Equivalences file not found: {equivalence_file_path}")
        logger.error(f"Directory contents: {os.listdir(job_dir)}")
        raise FileNotFoundError(
            f"Required equivalences file not found: {equivalence_file_path}"
        )

    # Read equivalences from file
    with open(equivalence_file_path, "r") as file:
        equivalences = file.readlines()
    equivalences = [line.strip() for line in equivalences]

    # Use a list of tuples to store equivalences
    equiv_list = []

    # Process equivalences
    for line in equivalences:
        splitted = line.split()
        equiv_list.append(
            (splitted[1], splitted[0], splitted[2])
        )  # (chain_number, chain_letter, equiv_res)

    # Read tertiary frustration data
    tertiary_frustration_path = os.path.join(job_dir, "tertiary_frustration.dat")
    with open(tertiary_frustration_path, "r") as file:
        tertiary_frustration = file.readlines()

    # Open output files
    frust_file_path = os.path.join(job_dir, f"{job_id}.pdb_{mode}")
    frust_aux_file_path = os.path.join(job_dir, f"{job_id}_{mode}.pdb_auxiliar")
    frust_renum_file_path = os.path.join(job_dir, f"{job_id}_{mode}_renumbered")

    def find_equiv(chain_number_idx: int) -> Tuple[str, str]:
        """Find and return the chain letter and equiv_res for a given chain number."""
        # Fix zero indexing
        # substract 1
        chain_number_idx -= 1
        if chain_number_idx in range(len(equiv_list)):
            # Return chain_letter and equiv_res
            return equiv_list[chain_number_idx][1], equiv_list[chain_number_idx][2]
        return None, None  # Return None if not found

    with open(frust_file_path, "w") as frust, open(
        frust_aux_file_path, "w"
    ) as frust_aux, open(frust_renum_file_path, "w") as frust_renum:
        if mode in ["configurational", "mutational"]:
            frust.write(
                "Res1 Res2 ChainRes1 ChainRes2 DensityRes1 DensityRes2 AA1 AA2 NativeEnergy DecoyEnergy SDEnergy FrstIndex Welltype FrstState\n"
            )

            for line in tertiary_frustration[2:]:
                splitted = line.split()
                res1, res2 = splitted[0], splitted[1]
                density1, density2 = splitted[11], splitted[12]
                aa1, aa2 = splitted[13], splitted[14]
                native_energy, decoy_energy, sd_energy = (
                    splitted[15],
                    splitted[16],
                    splitted[17],
                )
                frst_index = splitted[18]
                res_res_distance = ""
                frst_type = ""
                frst_type_aux = ""

                # Assign well-type
                if float(splitted[10]) < 6.5:
                    res_res_distance = "short"
                elif float(splitted[10]) >= 6.5:
                    if float(density1) < 2.6 and float(density2) < 2.6:
                        res_res_distance = "water-mediated"
                    else:
                        res_res_distance = "long"

                if float(frst_index) <= -1:
                    frst_type = "highly"
                    frst_type_aux = "red"
                elif -1 < float(frst_index) < 0.78:
                    frst_type = "neutral"
                    frst_type_aux = "gray"
                elif float(frst_index) >= 0.78:
                    frst_type = "minimally"
                    frst_type_aux = "green"

                chain_letter_res1, equiv_res1 = find_equiv(int(res1))
                chain_letter_res2, equiv_res2 = find_equiv(int(res2))

                frust.write(
                    f"{equiv_res1} {equiv_res2} {chain_letter_res1} {chain_letter_res2} {density1} {density2} {aa1} {aa2} {native_energy} {decoy_energy} {sd_energy} {frst_index} {res_res_distance} {frst_type}\n"
                )
                frust_renum.write(
                    f"{res1} {res2} {chain_letter_res1} {chain_letter_res2} {density1} {density2} {aa1} {aa2} {native_energy} {decoy_energy} {sd_energy} {frst_index} {res_res_distance} {frst_type}\n"
                )

                if frst_type_aux in ["green", "red"]:
                    frust_aux.write(
                        f"{equiv_res1} {equiv_res2} {chain_letter_res1} {chain_letter_res2} {res_res_distance} {frst_type_aux}\n"
                    )

        elif mode == "singleresidue":
            frust.write(
                "Res ChainRes DensityRes AA NativeEnergy DecoyEnergy SDEnergy FrstIndex\n"
            )

            for line in tertiary_frustration[2:]:
                splitted = line.split()
                res = splitted[0]
                density = splitted[5]
                aa = splitted[6]
                native_energy, decoy_energy, sd_energy = (
                    splitted[7],
                    splitted[8],
                    splitted[9],
                )
                frst_index = splitted[10]

                chain_letter, equiv_res = find_equiv(int(res))

                frust.write(
                    f"{equiv_res} {chain_letter} {density} {aa} {native_energy} {decoy_energy} {sd_energy} {frst_index}\n"
                )

    # Remove the renumbered file
    os.remove(frust_renum_file_path)


def organize_single_residue_data(
    pdb: "Pdb", residues_analyzed: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[int, SingleResidueData]]:
    """
    Organize single residue data into a structured format.

    Args:
        pdb: Pdb object
        residues_analyzed: Dictionary of analysis results per chain/residue

    Returns:
        Dict mapping chain_id -> {residue_number -> SingleResidueData}
    """
    organized_data = {}
    logger = logging.getLogger(__name__)

    for chain_id, residues in residues_analyzed.items():
        organized_data[chain_id] = {}
        logger.debug(f"Processing chain {chain_id}")

        for res_data in residues:
            res_num = res_data["res_num"]
            logger.debug(f"Processing residue {res_num}")

            # Read the mutation data file
            mutation_file = os.path.join(
                pdb.job_dir,
                "MutationsData",
                f"{pdb.mode}_Res{res_num}_threading_{chain_id}.txt",
            )

            logger.debug(f"Reading mutation file: {mutation_file}")

            # Read the file with proper column separation
            try:
                mutation_df = pd.read_csv(
                    mutation_file,
                    sep="\s+",  # Split on whitespace
                    names=[
                        "Res",
                        "ChainRes",
                        "AA",
                        "FrstIndex",
                    ],  # Specify column names
                    skiprows=1,  # Skip header row
                )
                logger.debug(
                    f"Successfully read mutation file with columns: {mutation_df.columns.tolist()}"
                )
            except Exception as e:
                logger.error(f"Error reading mutation file: {str(e)}")
                logger.debug("Attempting to read file contents directly:")
                with open(mutation_file, "r") as f:
                    logger.debug(f"File contents:\n{f.read()}")
                raise

            # Create mutations dictionary
            mutations = {}
            for _, row in mutation_df.iterrows():
                mutations[row["AA"]] = row["FrstIndex"]

            # Get original residue name
            original_res = pdb.atom[
                (pdb.atom["res_num"] == res_num) & (pdb.atom["chain"] == chain_id)
            ]["res_name"].iloc[0]

            logger.debug(f"Original residue: {original_res}")

            # Create SingleResidueData object
            res_data = SingleResidueData(
                residue_number=res_num,
                chain_id=chain_id,
                residue_name=original_res,
                mutations=mutations,
                native_energy=mutation_df["FrstIndex"].iloc[
                    0
                ],  # Use first row's frustration as native
                decoy_energy=None,  # These values aren't available in single residue mode
                sd_energy=None,
                density=None,
                plots=res_data.get("plot", None),
            )

            logger.debug(f"Created SingleResidueData object for residue {res_num}")
            organized_data[chain_id][res_num] = res_data

    return organized_data
