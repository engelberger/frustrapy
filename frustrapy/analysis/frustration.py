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
from .mutations import mutate_res, mutate_res_parallel
from ..utils import log_execution_time, get_os, replace_expr
from ..utils.helpers import (
    check_backbone_complete,
    complete_backbone,
    pdb_equivalences,
    renum_files,
    organize_single_residue_data,
)

from ..visualization import (
    plot_5andens,
    plot_5adens_proportions,
    plot_contact_map,
    plot_delta_frus,
)

logger = logging.getLogger(__name__)


def xadens(pdb: "Pdb", ratio: float = 5) -> None:
    """
        Calculate the proportion of each type of contact (neutral, highly minimally frustrated)
    elety    Is performed by obtaining information from the "tertiary_frustration.dat" file (intermediate processing).

        Args:
            pdb (Pdb): Pdb frustration object.
            ratio (float): Sphere radius. Default: 5.
    """
    # pdb.atom.keys()
    # Index(['ATOM', 'atom_num', 'atom_name', 'res_name', 'chain', 'res_num', 'x',
    #   'y', 'z', 'occupancy', 'b_factor', 'element'],
    #  dtype='object')
    xyz = pdb.atom.loc[:, ["x", "y", "z"]].values
    ca_xyz = pdb.atom.loc[pdb.atom["atom_name"] == "CA", ["x", "y", "z"]].values
    # ca_xyz = pdb.atom.loc[pdb.atom["atom_name"] == "CA", "xyz"].tolist()
    ca_x, ca_y, ca_z = [], [], []
    ca_x = ca_xyz[:, 0]
    ca_y = ca_xyz[:, 1]
    ca_z = ca_xyz[:, 2]

    conts_coords = pd.read_csv(
        os.path.join(pdb.job_dir, "tertiary_frustration.dat"),
        sep="\s+",
        header=None,
        usecols=[0, 1, 4, 5, 6, 7, 8, 9, 18],
    )
    positions = pdb.equivalences.iloc[:, 2].tolist()
    res_chain = pdb.equivalences.iloc[:, 0].tolist()

    # Convert columns to numeric type
    conts_coords.iloc[:, 2] = pd.to_numeric(conts_coords.iloc[:, 2], errors="coerce")
    conts_coords.iloc[:, 3] = pd.to_numeric(conts_coords.iloc[:, 3], errors="coerce")
    conts_coords.iloc[:, 4] = pd.to_numeric(conts_coords.iloc[:, 4], errors="coerce")
    conts_coords.iloc[:, 5] = pd.to_numeric(conts_coords.iloc[:, 5], errors="coerce")
    conts_coords.iloc[:, 6] = pd.to_numeric(conts_coords.iloc[:, 6], errors="coerce")
    conts_coords.iloc[:, 7] = pd.to_numeric(conts_coords.iloc[:, 7], errors="coerce")
    conts_coords.iloc[:, 8] = pd.to_numeric(conts_coords.iloc[:, 8], errors="coerce")

    # Now perform your operations
    vps = pd.DataFrame(
        {
            "col1": conts_coords.iloc[:, 0],
            "col2": conts_coords.iloc[:, 1],
            "col3": (
                conts_coords.iloc[:, 5].astype(float)
                + conts_coords.iloc[:, 2].astype(float)
            )
            / 2.0,
            "col4": (
                conts_coords.iloc[:, 6].astype(float)
                + conts_coords.iloc[:, 3].astype(float)
            )
            / 2.0,
            "col5": (
                conts_coords.iloc[:, 7].astype(float)
                + conts_coords.iloc[:, 4].astype(float)
            )
            / 2.0,
            "col6": conts_coords.iloc[:, 8].astype(float),
        }
    )

    vps.to_csv(
        os.path.join(pdb.job_dir, f"{pdb.pdb_base}.pdb.vps"),
        sep="\t",
        header=False,
        index=False,
    )

    output_file = os.path.join(pdb.job_dir, f"{pdb.pdb_base}.pdb_{pdb.mode}_5adens")
    if not os.path.exists(output_file):
        open(output_file, "w").close()

    with open(output_file, "a") as f:
        f.write(
            "Res ChainRes Total HighlyFrst NeutrallyFrst MinimallyFrst "
            "relHighlyFrustrated relNeutralFrustrated relMinimallyFrustrated\n"
        )

    # Convert columns to numeric type
    vps.iloc[:, 3] = pd.to_numeric(vps.iloc[:, 3], errors="coerce")
    vps.iloc[:, 4] = pd.to_numeric(vps.iloc[:, 4], errors="coerce")
    vps.iloc[:, 5] = pd.to_numeric(vps.iloc[:, 5], errors="coerce")

    for i, ca_point in enumerate(ca_xyz):
        distances = np.sqrt(
            (ca_point[0] - vps["col3"]) ** 2
            + (ca_point[1] - vps["col4"]) ** 2
            + (ca_point[2] - vps["col5"]) ** 2
        )

        total_density = len(vps[distances < ratio])
        highly_frustrated = len(vps[(distances < ratio) & (vps["col6"] <= -1)])
        neutral_frustrated = len(
            vps[(distances < ratio) & (vps["col6"] > -1) & (vps["col6"] < 0.78)]
        )
        minimally_frustrated = len(vps[(distances < ratio) & (vps["col6"] >= 0.78)])

        rel_highly_frustrated_density = 0
        rel_neutral_frustrated_density = 0
        rel_minimally_frustrated_density = 0

        if total_density > 0:
            rel_highly_frustrated_density = highly_frustrated / total_density
            rel_neutral_frustrated_density = neutral_frustrated / total_density
            rel_minimally_frustrated_density = minimally_frustrated / total_density

        with open(output_file, "a") as f:
            f.write(
                f"{positions[i]} {res_chain[i]} {total_density} {highly_frustrated} "
                f"{neutral_frustrated} {minimally_frustrated} {rel_highly_frustrated_density} "
                f"{rel_neutral_frustrated_density} {rel_minimally_frustrated_density}\n"
            )


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
) -> Tuple["Pdb", Dict]:
    """
    Calculate local energy frustration for a protein structure.

    Args:
        ... existing args ...
        residues (Optional[Dict[str, List[int]]]): Dictionary mapping chain IDs to lists of residue numbers
            to analyze. Only used for singleresidue mode. Default: None (analyze all residues).
    """
    # Initialize plots dictionary at the start
    plots = {}
    logger = logging.getLogger(__name__)

    if results_dir is None:
        results_dir = os.path.join(tempfile.gettempdir(), "")
    elif not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logger.info(f"Created results directory: {results_dir}")

    if results_dir[-1] != "/":
        results_dir += "/"

    if pdb_file is None and pdb_id is None:
        raise ValueError("You must indicate PdbID or PdbFile!")

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

    temp_folder = tempfile.gettempdir()

    if chain is None:
        boolsplit = False
    else:
        boolsplit = True

    if pdb_file is None:
        logger.info(
            "-----------------------------Download files-----------------------------"
        )
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        logger.info(f"Downloading PDB from {pdb_url}")
        subprocess.run(
            [
                "wget",
                "--no-check-certificate",
                "-P",
                temp_folder,
                pdb_url,
                "-q",
                "--progress=bar:force:noscroll",
                "--show-progress",
            ]
        )
        pdb_file = os.path.join(temp_folder, f"{pdb_id}.pdb")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)

    class NonHetSelect(Select):
        def accept_residue(self, residue):
            return 1 if residue.id[0] == " " else 0

    pdb_io = PDBIO()
    pdb_io.set_structure(structure)
    pdb_io.save(pdb_file, NonHetSelect())

    if chain is not None:
        logger.debug(f"Checking chains: {chain}")

        # Get list of available chain IDs
        available_chains = []
        for chain_obj in structure.get_chains():
            available_chains.append(chain_obj.id)
        logger.debug(f"Available chains: {available_chains}")

        # Check if all requested chains exist
        valid_chains = True
        missing_chains = []
        for c in chain:
            if c not in available_chains:
                valid_chains = False
                missing_chains.append(c)

        if valid_chains:
            logger.info(f"All requested chains {chain} are valid")

            class ChainSelect(Select):
                def __init__(self, selected_chains):
                    self.selected_chains = selected_chains

                def accept_chain(self, chain_obj):
                    return chain_obj.id in self.selected_chains

            # Convert chain to list if it's a string
            selected_chains = [chain] if isinstance(chain, str) else chain
            pdb_io.save(pdb_file, ChainSelect(selected_chains))
            logger.debug(f"Saved PDB with selected chains to {pdb_file}")
        else:
            logger.error(f"Missing chains: {missing_chains}")
            raise ValueError(
                f"The Chain {' '.join(missing_chains)} doesn't exist! The Chains are: {' '.join(available_chains)}"
            )
        pdb_base = f"{os.path.splitext(os.path.basename(pdb_file))[0]}_{chain}"
        logger.info(f"Set pdb_base to: {pdb_base}")
    else:
        pdb_base = os.path.splitext(os.path.basename(pdb_file))[0]
        logger.info(f"No chain specified. Set pdb_base to: {pdb_base}")

    job_dir = os.path.join(results_dir, f"{pdb_base}.done/")
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    shutil.copy(pdb_file, os.path.join(job_dir, f"{pdb_base}.pdb"))
    # Assert that the pdb file is in the job directory
    pdb_file = os.path.join(job_dir, f"{pdb_base}.pdb")
    assert os.path.exists(pdb_file)
    os.chdir(job_dir)

    logger.info("-----------------------------Filtering-----------------------------")
    df = pd.read_csv(
        pdb_file,
        sep="\s+",
        header=None,
        skiprows=0,
        names=[
            "ATOM",
            "atom_num",
            "atom_name",
            "res_name",
            "chain",
            "res_num",
            "x",
            "y",
            "z",
            "occupancy",
            "b_factor",
            "element",
        ],
    )

    df.loc[df["res_name"] == "MSE", "res_name"] = "MET"
    df.loc[df["res_name"] == "HIE", "res_name"] = "HIS"
    df.loc[df["res_name"] == "CYX", "res_name"] = "CYS"
    df.loc[df["res_name"] == "CY1", "res_name"] = "CYS"

    protein_res = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    ]
    df = df[df["res_name"].isin(protein_res)]

    df.loc[df["chain"].isna(), "chain"] = "A"

    pdb = Pdb(job_dir, pdb_base, mode, df, pdb_equivalences(pdb_file, job_dir))
    logger.info("HI")
    pdb_equivalences(pdb_file, job_dir)
    logger.info(
        "-----------------------------Preparing files-----------------------------"
    )
    subprocess.run(
        [
            "sh",
            os.path.join(pdb.scripts_dir, "AWSEMFiles/AWSEMTools/PdbCoords2Lammps.sh"),
            pdb.pdb_base,
            pdb.pdb_base,
            pdb.scripts_dir,
        ]
    )
    # Log the command used to generate the lammps file
    with open(os.path.join(job_dir, "commands.help"), "w") as f:
        f.write(
            f"sh {os.path.join(pdb.scripts_dir, 'AWSEMFiles/AWSEMTools/PdbCoords2Lammps.sh')} {pdb.pdb_base} {pdb.pdb_base} {pdb.scripts_dir}"
        )
    # log the command used to copy the dat files
    with open(os.path.join(job_dir, "commands.help"), "a") as f:
        f.write(
            f"\ncp {os.path.join(pdb.scripts_dir, 'AWSEMFiles/*.dat*')} {pdb.job_dir}"
        )
    os.system(f"cp {os.path.join(pdb.scripts_dir, 'AWSEMFiles/*.dat*')} {pdb.job_dir}")

    logger.info(
        "-----------------------------Setting options-----------------------------"
    )
    replace_expr(
        "run\t\t10000", "run\t\t0", os.path.join(pdb.job_dir, f"{pdb.pdb_base}.in")
    )
    replace_expr(
        "mutational", pdb.mode, os.path.join(pdb.job_dir, "fix_backbone_coeff.data")
    )

    if electrostatics_k is not None:
        logger.info(
            "-----------------------------Setting electrostatics-----------------------------"
        )
        replace_expr(
            "\\[DebyeHuckel\\]-",
            "\\[DebyeHuckel\\]",
            os.path.join(pdb.job_dir, "fix_backbone_coeff.data"),
        )
        replace_expr(
            "4.15 4.15 4.15",
            f"{electrostatics_k} {electrostatics_k} {electrostatics_k}",
            os.path.join(pdb.job_dir, "fix_backbone_coeff.data"),
        )
        logger.info("Setting electrostatics...")
        subprocess.run(
            [
                "python3",
                os.path.join(pdb.scripts_dir, "Pdb2Gro.py"),
                f"{pdb.pdb_base}.pdb",
                f"{pdb.pdb_base}.pdb.gro",
            ]
        )
        # Log the command used to generate the gro file
        with open(os.path.join(job_dir, "commands.help"), "a") as f:
            f.write(
                f"\npython3 {os.path.join(pdb.scripts_dir, 'Pdb2Gro.py')} {pdb.pdb_base}.pdb {pdb.pdb_base}.pdb.gro"
            )
        subprocess.run(
            [
                "perl",
                os.path.join(pdb.scripts_dir, "GenerateChargeFile.pl"),
                f"{pdb.pdb_base}.pdb.gro",
                ">",
                os.path.join(job_dir, "charge_on_residues.dat"),
            ]
        )
        # TODO : Test reimplementing this with a python reimplementation of the perl script
        # from .generate_charge_file import generate_charge_file
        # generate_charge_file(
        #     f"{pdb.pdb_base}.pdb.gro",
        #     os.path.join(job_dir, "charge_on_residues.dat"),
        # )
        with open(os.path.join(job_dir, "commands.help"), "a") as f:
            f.write(
                f"\nperl {os.path.join(pdb.scripts_dir, 'GenerateChargeFile.pl')} {pdb.pdb_base}.pdb.gro > charge_on_residues.dat"
            )

    logger.info(
        "-----------------------------Calculating Frustration-----------------------------"
    )
    operative_system = get_os()
    if operative_system == "linux":
        logger.info(f"Processing PDB: {os.path.basename(pdb_file)}")
        if mode == "singleresidue":
            # Fix the residues.get() error by checking if residues exists
            if residues and "A" in residues:
                logger.info(
                    f"Analyzing mutations for residue(s): {', '.join(str(r) for r in residues['A'])}"
                )
            else:
                logger.info("Analyzing all residues (no specific residues specified)")

        subprocess.run(
            [
                "cp",
                os.path.join(pdb.scripts_dir, f"lmp_serial_{seq_dist}_Linux"),
                pdb.job_dir,
            ]
        )

        logger.info("Running LAMMPS energy calculations...")
        # Capture LAMMPS output and only log it when debugging
        lammps_cmd = (
            f"{pdb.job_dir}lmp_serial_{seq_dist}_Linux < {pdb.job_dir}{pdb.pdb_base}.in"
        )
        try:
            result = subprocess.run(
                lammps_cmd,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                cwd=pdb.job_dir,
            )
            if debug:
                logger.debug("LAMMPS Calculation Details:")
                logger.debug(result.stdout)
                if result.stderr:
                    logger.debug("LAMMPS Errors:")
                    logger.debug(result.stderr)
        except subprocess.CalledProcessError as e:
            logger.error(
                f"LAMMPS energy calculation failed with return code {e.returncode}"
            )
            logger.error("Error details:")
            logger.error(e.stderr)
            raise

    elif operative_system == "osx":
        subprocess.run(
            [
                "cp",
                os.path.join(pdb.scripts_dir, f"lmp_serial_{seq_dist}_MacOS"),
                pdb.job_dir,
            ]
        )
        subprocess.run(["chmod", "+x", f"lmp_serial_{seq_dist}_MacOS"])

        # Capture LAMMPS output for MacOS
        try:
            result = subprocess.run(
                [f"./lmp_serial_{seq_dist}_MacOS"],
                input=open(f"{pdb.pdb_base}.in").read(),
                capture_output=True,
                text=True,
                check=True,
                cwd=pdb.job_dir,
            )
            if debug:
                logger.debug("LAMMPS Output:")
                logger.debug(result.stdout)
                if result.stderr:
                    logger.debug("LAMMPS Errors:")
                    logger.debug(result.stderr)
        except subprocess.CalledProcessError as e:
            logger.error(f"LAMMPS execution failed with return code {e.returncode}")
            logger.error("LAMMPS Error Output:")
            logger.error(e.stderr)
            raise

        with open(os.path.join(job_dir, "commands.help"), "a") as f:
            f.write(
                f"cp {os.path.join(pdb.scripts_dir, f'lmp_serial_{seq_dist}_MacOS')} {pdb.job_dir}"
                f"\nchmod +x lmp_serial_{seq_dist}_MacOS"
                f"\n./lmp_serial_{seq_dist}_MacOS < {pdb.pdb_base}.in"
            )

    # The legacy script called a perl script to post process the output of lamps out of the tertiary frustration
    # I refactored this perl script to python and it is now called from here and it can be debugged more easily
    # TODO : Currently this methods is defined in the same file, later it should be moved to a separate file as utils.py
    renum_files(
        job_id=pdb.pdb_base,
        job_dir=pdb.job_dir,
        mode=pdb.mode,
    )
    # subprocess.run(
    #    [
    #        "python",
    #        os.path.join(pdb.scripts_dir, "RenumFiles.py"),
    #        pdb.pdb_base,
    #        pdb.job_dir,
    #        pdb.mode,
    #    ]
    # )

    with open(os.path.join(job_dir, "commands.help"), "a") as f:
        f.write(
            f"\npython {os.path.join(pdb.scripts_dir, 'RenumFiles.py')} {pdb.pdb_base} {pdb.job_dir} {pdb.mode}"
        )

    if pdb.mode == "configurational" or pdb.mode == "mutational":
        xadens(pdb)
        # Stop execution if the mode is 'configurational' or 'mutational'
        # exit()
    logger.info(
        "-----------------------------Reorganization-----------------------------"
    )
    frustration_dir = os.path.join(pdb.job_dir, "FrustrationData")
    if not os.path.exists(frustration_dir):
        os.makedirs(frustration_dir)

    pdb_output_folder = os.path.join(results_dir, f"{pdb_base}.done")
    os.system(f"mv {pdb_output_folder}/*.pdb_{pdb.mode} {frustration_dir}")

    # os.system(f"mv {results_dir}*.pdb_{pdb.mode} {frustration_dir}")
    if pdb.mode == "configurational" or pdb.mode == "mutational":
        # subprocess.run(["mv", f"{results_dir}*_{pdb.mode}_5adens", frustration_dir])
        os.system(f"mv {pdb_output_folder}/*_{pdb.mode}_5adens {frustration_dir}")

    # subprocess.run(["mv", f"{results_dir}*.pdb", frustration_dir])
    os.system(f"mv {pdb_output_folder}/*.pdb {frustration_dir}")

    if graphics:  # and pdb.mode != "singleresidue"
        logger.info("-----------------------------Images-----------------------------")
        images_dir = os.path.join(pdb.job_dir, "Images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        if pdb.mode != "singleresidue":
            # Generate plots for configurational/mutational modes
            plotly_5andens = plot_5andens(pdb, save=True)
            plotly_5adens_proportions = plot_5adens_proportions(pdb, save=True)
            plotly_contact_map = plot_contact_map(pdb, save=True)

            plots = {
                "plot_5andens": plotly_5andens,
                "plot_5adens_proportions": plotly_5adens_proportions,
                "plot_contact_map": plotly_contact_map,
            }
        if pdb.mode == "singleresidue":
            # For singleresidue mode
            try:
                # Track analyzed residues and their data
                residues_analyzed = {}

                if chain:
                    chains_to_analyze = [chain] if isinstance(chain, str) else chain
                else:
                    chains_to_analyze = pdb.atom[pdb.atom["ATOM"] == "ATOM"][
                        "chain"
                    ].unique()

                for chain_id in chains_to_analyze:
                    residues_analyzed[chain_id] = []

                    if residues and chain_id in residues:
                        chain_residues = residues[chain_id]
                    else:
                        chain_residues = pdb.atom[
                            (pdb.atom["ATOM"] == "ATOM")
                            & (pdb.atom["chain"] == chain_id)
                        ]["res_num"].unique()

                    for res in chain_residues:
                        try:
                            # Perform mutation analysis
                            pdb = mutate_res_parallel(
                                pdb=pdb,
                                res_num=res,
                                chain=chain_id,
                                split=True,
                                method="threading",
                            )

                            res_data = {"res_num": res}

                            try:
                                delta_plot = plot_delta_frus(
                                    pdb=pdb,
                                    res_num=res,
                                    chain=chain_id,
                                    method="threading",
                                    save=True,
                                )
                                plots[f"delta_frus_res{res}_chain{chain_id}"] = (
                                    delta_plot
                                )
                                res_data["plot"] = delta_plot

                            except Exception as plot_error:
                                logger.warning(
                                    f"Failed to generate plot for residue {res} chain {chain_id}: {str(plot_error)}"
                                )
                                res_data["plot"] = None

                            residues_analyzed[chain_id].append(res_data)

                        except Exception as e:
                            logger.error(
                                f"Failed to analyze residue {res} chain {chain_id}: {str(e)}"
                            )
                            continue

                # Organize and save data
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
                error_msg = (
                    f"Failed to generate singleresidue analysis. "
                    f"This could be due to:\n"
                    f"1. Problems with the PDB structure\n"
                    f"2. Missing or corrupted data files\n"
                    f"3. Invalid configuration\n"
                    f"Original error: {str(e)}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

    if visualization and pdb.mode != "singleresidue":
        logger.info(
            "-----------------------------Visualizations-----------------------------"
        )
        subprocess.run(
            [
                "perl",
                os.path.join(pdb.scripts_dir, "GenerateVisualizations.pl"),
                f"{pdb.pdb_base}_{pdb.mode}.pdb_auxiliar",
                pdb.pdb_base,
                os.path.dirname(pdb.job_dir),
                pdb.mode,
            ]
        )

        visualization_dir = os.path.join(pdb.job_dir, "VisualizationScripts")
        if not os.path.exists(visualization_dir):
            os.makedirs(visualization_dir)

        # Copy the PDB file to the visualization directory
        shutil.copy(
            os.path.join(frustration_dir, f"{pdb.pdb_base}.pdb"),
            os.path.join(visualization_dir, f"{pdb.pdb_base}.pdb"),
        )

        # For moving files matching patterns (*_{pdb.mode}.pml, *_{pdb.mode}.tcl, *_{pdb.mode}.jml), use glob and shutil
        for extension in ["pml", "tcl", "jml"]:
            files_to_move = glob.glob(
                os.path.join(pdb.job_dir, f"*_{pdb.mode}.{extension}")
            )
            for file_path in files_to_move:
                # Try to move the file and if it exists, print that it already exists but it will be overwritten
                try:
                    # If the file exists, print that it already exists but it will be overwritten
                    if os.path.exists(
                        os.path.join(visualization_dir, os.path.basename(file_path))
                    ):
                        logger.info(
                            f"The file {os.path.basename(file_path)} already exists in the visualization directory. It will be overwritten."
                        )
                        # Remove the file
                        os.remove(
                            os.path.join(visualization_dir, os.path.basename(file_path))
                        )
                        # Move the file
                        shutil.move(file_path, visualization_dir)
                except shutil.Error:
                    logger.info(
                        f"The file {os.path.basename(file_path)} already exists in the visualization directory. It will be overwritten."
                    )

                # shutil.move(file_path, visualization_dir)

        # Copy the 'draw_links.py' script to the visualization directory
        shutil.copy(
            os.path.join(pdb.scripts_dir, "draw_links.py"),
            visualization_dir,
        )

    logger.info("\n\n****Storage information****")
    logger.info(f"The frustration data was stored in {frustration_dir}")
    if graphics and pdb.mode != "singleresidue":
        logger.info(f"Graphics are stored in {images_dir}")
    if visualization and pdb.mode != "singleresidue":
        logger.info(f"Visualizations are stored in {visualization_dir}")

    # List all files and directories in the job directory
    all_items = glob.glob(os.path.join(job_dir, "*"))

    # Filter out directories and hidden files (starting with '.')
    files_to_remove = [
        item
        for item in all_items
        if os.path.isfile(item) and not os.path.basename(item).startswith(".")
    ]

    # Remove the files if debug is false
    if not debug:
        for file_path in files_to_remove:
            os.remove(file_path)

    # Remove the 'split_chain' directory if 'chain' is not None, using an absolute path
    if chain is not None:
        split_chain_dir = os.path.join(temp_folder, "split_chain")
        if os.path.exists(split_chain_dir):
            shutil.rmtree(split_chain_dir)

    # Remove the PDB file downloaded to the temp folder, using an absolute path
    pdb_file_path = os.path.join(temp_folder, f"{pdb_id}.pdb")
    if os.path.exists(pdb_file_path):
        os.remove(pdb_file_path)

    return pdb, plots


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
    """
    Calculate local energy frustration for all protein structures in one directory.

    Args:
        pdbs_dir (str): Directory containing all protein structures. The full path to the file is needed.
        order_list (Optional[List[str]]): Ordered list of PDB files to calculate frustration. If it is None, frustration is calculated for all PDBs.
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
    if results_dir is None:
        results_dir = os.path.join(tempfile.gettempdir(), "")
    elif not os.path.exists(results_dir):
        # Make the results directory and absolute path
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir)
        logger.info(f"The results directory {results_dir} has been created.")

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
        logger.info(f"The modes log file {modes_log_file} exists.")
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
            pdb, plots = calculate_frustration(
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

        logger.info("\n\n****Storage information****")
        logger.info(
            f"Frustration data for all Pdb's directory {pdbs_dir} are stored in {results_dir}"
        )
        return plots_dir_dict


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
        logger.info(f"The results directory {results_dir} has been created.")

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

    logger.info(
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

    logger.info(
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

    logger.info("\n\n****Storage information****")
    logger.info(f"The frustration of the full dynamic is stored in {results_dir}")

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
