import os
import sys
import subprocess
import tempfile
from typing import Optional, List, Dict, Union, Tuple
from typing import TYPE_CHECKING
from Bio.PDB import PDBParser
import pandas as pd
import tempfile
import shutil
from Bio.PDB import PDBParser, PDBIO, Select
from Bio import SeqIO
import glob
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr
import igraph as ig
import leidenalg as la

from visualization import *

# if TYPE_CHECKING:
#    from .pdb import Pdb


class Pdb:
    def __init__(
        self,
        job_dir: str,
        pdb_base: str,
        mode: str,
        atom: pd.DataFrame,
        equivalences: pd.DataFrame,
        scripts_dir: str = os.path.join(os.path.dirname(__file__), "scripts"),
    ):
        self.job_dir = job_dir
        self.pdb_base = pdb_base
        self.mode = mode
        self.atom = atom
        self.equivalences = equivalences
        self.scripts_dir = scripts_dir


class Dynamic:
    def __init__(
        self,
        pdbs_dir: str,
        order_list: Optional[List[str]] = None,
        chain: Optional[str] = None,
        electrostatics_k: Optional[float] = None,
        seq_dist: int = 12,
        mode: str = "configurational",
        results_dir: Optional[str] = None,
        clusters: Optional[Dict] = None,
    ):
        self.pdbs_dir = pdbs_dir
        self.order_list = order_list or []
        self.chain = chain
        self.electrostatics_k = electrostatics_k
        self.seq_dist = seq_dist
        self.mode = mode
        self.results_dir = results_dir
        self.clusters = clusters or {}

        # Additional attributes for dynamic analysis
        self.residues_dynamic = (
            {}
        )  # Stores dynamic frustration data for specific residues

    def load_order_list(self):
        """
        Loads the order list of PDB files if not provided.
        """
        if not self.order_list:
            import os

            self.order_list = [
                f for f in os.listdir(self.pdbs_dir) if f.endswith(".pdb")
            ]

    def add_residue_dynamic(self, resno: int, chain: str, data: pd.DataFrame):
        """
        Adds dynamic frustration data for a specific residue.

        Args:
            resno (int): Residue number.
            chain (str): Chain identifier.
            data (pd.DataFrame): Frustration data for the residue.
        """
        if chain not in self.residues_dynamic:
            self.residues_dynamic[chain] = {}
        self.residues_dynamic[chain][f"Res_{resno}"] = data

    def get_residue_dynamic(self, resno: int, chain: str) -> pd.DataFrame:
        """
        Retrieves dynamic frustration data for a specific residue.

        Args:
            resno (int): Residue number.
            chain (str): Chain identifier.

        Returns:
            pd.DataFrame: Frustration data for the residue.
        """
        try:
            return self.residues_dynamic[chain][f"Res_{resno}"]
        except KeyError:
            raise ValueError(
                f"No dynamic data found for residue {resno} in chain {chain}."
            )


def renum_files(job_id, job_dir, mode):
    # Read equivalences from file
    equivalence_file_path = os.path.join(job_dir, f"{job_id}.pdb_equivalences.txt")
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


def has_modeller():
    try:
        import modeller

        return True
    except ImportError:
        return False


def get_os() -> str:
    """
    Get the operating system on which the package is running.

    Returns:
        str: Character string indicating the operating system.
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
            "col3": (conts_coords.iloc[:, 5] + conts_coords.iloc[:, 2]) / 2.0,
            "col4": (conts_coords.iloc[:, 6] + conts_coords.iloc[:, 3]) / 2.0,
            "col5": (conts_coords.iloc[:, 7] + conts_coords.iloc[:, 4]) / 2.0,
            "col6": conts_coords.iloc[:, 8],
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
            "Res ChainRes Total nHighlyFrst nNeutrallyFrst nMinimallyFrst "
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


def get_frustration(
    pdb: Pdb, resno: Optional[List[int]] = None, chain: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Returns the frustration of all Pdb residues, of a specific Chain or residue (Resno).
    By default, the complete Pdb frustration table is obtained and returned.

    Args:
        pdb (Pdb): Pdb frustration object obtained by calculate_frustration().
        resno (Optional[List[int]]): Specific residues in Pdb. Default: None.
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

    if resno is not None:
        frustration_table = frustration_table[
            (frustration_table["Res"].isin(resno))
            | (frustration_table["Res1"].isin(resno))
            | (frustration_table["Res2"].isin(resno))
        ]

    return frustration_table


def get_frustration_dynamic(
    dynamic: "Dynamic", resno: int, chain: str, frames: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Obtains and returns the table of frustration of a specific chain and residue in a complete dynamic
    or in the indicated frames. The frustration of a specific residue must have previously been calculated
    using dynamic_res().

    Args:
        dynamic (Dynamic): Dynamic frustration object obtained by dynamic_frustration().
        resno (int): Specific residue.
        chain (str): Specific chain.
        frames (Optional[List[int]]): Specific frames. Default: None.

    Returns:
        pd.DataFrame: Frustration table.
    """
    # Assuming Dynamic object has an attribute 'ResiduesDynamic' which is a dictionary
    # where keys are chain identifiers and values are another dictionary.
    # This inner dictionary has keys formatted as "Res_{resno}" pointing to the file paths
    # containing the frustration data for each residue.

    if chain not in dynamic.ResiduesDynamic:
        raise ValueError(f"No analysis for chain {chain}.")

    res_key = f"Res_{resno}"
    if res_key not in dynamic.ResiduesDynamic[chain]:
        raise ValueError(f"No analysis for residue {resno} in chain {chain}.")

    # Load the frustration data for the specific residue and chain
    frustration_data_path = dynamic.ResiduesDynamic[chain][res_key]
    frustration_df = pd.read_csv(frustration_data_path, sep="\s+", header=0)

    # If specific frames are requested, filter the dataframe
    if frames is not None:
        frustration_df = frustration_df[frustration_df["Frame"].isin(frames)]

    return frustration_df


def get_clusters(
    dynamic: Dynamic, clusters: Union[str, List[int]] = "all"
) -> pd.DataFrame:
    """
    Obtain information about the clusters obtained from detect_dynamic_clusters(),
    the name and number of the residues belonging to each cluster indicated in Clusters.

    Args:
        dynamic (Dynamic): Dynamic Frustration Object.
        clusters (Union[str, List[int]]): Indicates the clusters, for example, [1, 2, 3], clusters 1, 2 and 3. Default: "all".

    Returns:
        pd.DataFrame: Data frame containing name, number and cluster of each residue belonging to Clusters.
    """
    if "Graph" not in dynamic.clusters or dynamic.clusters["Graph"] is None:
        raise ValueError("Cluster detection failed, run detect_dynamic_clusters()")

    # Extract cluster data
    cluster_data = pd.DataFrame(
        {
            "AA": [name.split("_")[0] for name in dynamic.clusters["Graph"].vs["name"]],
            "Res": [
                int(name.split("_")[1]) for name in dynamic.clusters["Graph"].vs["name"]
            ],
            "Cluster": dynamic.clusters["LeidenClusters"]["cluster"],
            "Mean": [
                dynamic.clusters["Means"][res]
                for res in dynamic.clusters["LeidenClusters"].index
            ],
            "Sd": [
                dynamic.clusters["Sd"][res]
                for res in dynamic.clusters["LeidenClusters"].index
            ],
            "FrstRange": [
                dynamic.clusters["FrstRange"][res]
                for res in dynamic.clusters["LeidenClusters"].index
            ],
        }
    )

    # Filter by specified clusters if not "all"
    if clusters != "all":
        cluster_data = cluster_data[cluster_data["Cluster"].isin(clusters)]

    return cluster_data


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
    # print(equivalences_df)
    # Save to file
    print(pdb_file)
    # Get the base name of the pdb file
    pdb_file_base = os.path.basename(pdb_file)
    pdb_filename = pdb_file_base
    print(pdb_filename)
    output_path = os.path.join(output_dir, f"{pdb_filename}_equivalences.txt")
    print(output_path)
    equivalences_df.to_csv(output_path, sep="\t", index=False, header=False)
    # Print a log that the csv was saved  to the x file
    with open(os.path.join(output_dir, "commands.log"), "a") as f:
        f.write(f"\n{output_path} equivalences saved")
    return equivalences_df


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


def calculate_frustration(
    pdb_file: Optional[str] = None,
    pdb_id: Optional[str] = None,
    chain: Optional[Union[str, List[str]]] = None,
    electrostatics_k: Optional[float] = None,
    seq_dist: int = 12,
    mode: str = "configurational",
    graphics: bool = True,
    visualization: bool = True,
    results_dir: Optional[str] = None,
) -> "Pdb":
    if results_dir is None:
        results_dir = os.path.join(tempfile.gettempdir(), "")
    elif not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"The results directory {results_dir} has been created.")

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
        print(
            "-----------------------------Download files-----------------------------"
        )
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
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
        if all(c in [chain.id for chain in structure.get_chains()] for c in chain):

            class ChainSelect(Select):
                def accept_chain(self, chain):
                    return 1 if chain.id in chain else 0

            pdb_io.save(pdb_file, ChainSelect())
        else:
            missing_chains = [
                c
                for c in chain
                if c not in [chain.id for chain in structure.get_chains()]
            ]
            raise ValueError(
                f"The Chain {' '.join(missing_chains)} doesn't exist! The Chains are: {' '.join([chain.id for chain in structure.get_chains()])}"
            )
        pdb_base = f"{os.path.splitext(os.path.basename(pdb_file))[0]}_{chain}"
    else:
        pdb_base = os.path.splitext(os.path.basename(pdb_file))[0]

    job_dir = os.path.join(results_dir, f"{pdb_base}.done/")
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    shutil.copy(pdb_file, os.path.join(job_dir, f"{pdb_base}.pdb"))
    # Assert that the pdb file is in the job directory
    pdb_file = os.path.join(job_dir, f"{pdb_base}.pdb")
    assert os.path.exists(pdb_file)
    os.chdir(job_dir)

    print("-----------------------------Filtering-----------------------------")
    df = pd.read_csv(
        pdb_file,
        sep="\s+",
        header=None,
        skiprows=1,
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
    print("HI")
    pdb_equivalences(pdb_file, job_dir)
    print("-----------------------------Preparing files-----------------------------")
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
    with open(os.path.join(job_dir, "commands.log"), "w") as f:
        f.write(
            f"sh {os.path.join(pdb.scripts_dir, 'AWSEMFiles/AWSEMTools/PdbCoords2Lammps.sh')} {pdb.pdb_base} {pdb.pdb_base} {pdb.scripts_dir}"
        )
    # log the command used to copy the dat files
    with open(os.path.join(job_dir, "commands.log"), "a") as f:
        f.write(
            f"\ncp {os.path.join(pdb.scripts_dir, 'AWSEMFiles/*.dat*')} {pdb.job_dir}"
        )
    os.system(f"cp {os.path.join(pdb.scripts_dir, 'AWSEMFiles/*.dat*')} {pdb.job_dir}")

    print("-----------------------------Setting options-----------------------------")
    replace_expr(
        "run\t\t10000", "run\t\t0", os.path.join(pdb.job_dir, f"{pdb.pdb_base}.in")
    )
    replace_expr(
        "mutational", pdb.mode, os.path.join(pdb.job_dir, "fix_backbone_coeff.data")
    )

    if electrostatics_k is not None:
        print(
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
        print("Setting electrostatics...")
        subprocess.run(
            [
                "python3",
                os.path.join(pdb.scripts_dir, "Pdb2Gro.py"),
                f"{pdb.pdb_base}.pdb",
                f"{pdb.pdb_base}.pdb.gro",
            ]
        )
        # Log the command used to generate the gro file
        with open(os.path.join(job_dir, "commands.log"), "a") as f:
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
        with open(os.path.join(job_dir, "commands.log"), "a") as f:
            f.write(
                f"\nperl {os.path.join(pdb.scripts_dir, 'GenerateChargeFile.pl')} {pdb.pdb_base}.pdb.gro > charge_on_residues.dat"
            )

    print("-----------------------------Calculating-----------------------------")
    operative_system = get_os()
    if operative_system == "linux":
        subprocess.run(
            [
                "cp",
                os.path.join(pdb.scripts_dir, f"lmp_serial_{seq_dist}_Linux"),
                pdb.job_dir,
            ]
        )
        with open(os.path.join(job_dir, "commands.log"), "a") as f:
            f.write(
                f"\ncp {os.path.join(pdb.scripts_dir, f'lmp_serial_{seq_dist}_Linux')} {pdb.job_dir}"
            )
        subprocess.run(["chmod", "+x", f"lmp_serial_{seq_dist}_Linux"])
        # TODO THIS IS NOT WORKING
        # subprocess.run(
        #    [
        #        f"{pdb.job_dir}lmp_serial_{seq_dist}_Linux",
        #        "<",
        #        f"{pdb.job_dir}{pdb.pdb_base}.in",
        #    ]
        # )
        # TODO THIS IS WORKING
        os.system(
            f"cd {pdb.job_dir} && {pdb.job_dir}lmp_serial_{seq_dist}_Linux < {pdb.job_dir}{pdb.pdb_base}.in"
        )
        with open(os.path.join(job_dir, "commands.log"), "a") as f:
            f.write(
                f"\nchmod +x lmp_serial_{seq_dist}_Linux"
                f"\n{pdb.job_dir}lmp_serial_{seq_dist}_Linux < {pdb.job_dir}{pdb.pdb_base}.in"
            )
    elif operative_system == "osx":
        subprocess.run(
            [
                "cp",
                os.path.join(pdb.scripts_dir, f"lmp_serial_{seq_dist}_MacOS"),
                pdb.job_dir,
            ]
        )
        subprocess.run(["chmod", "+x", f"lmp_serial_{seq_dist}_MacOS"])
        subprocess.run([f"./lmp_serial_{seq_dist}_MacOS", "<", f"{pdb.pdb_base}.in"])
        with open(os.path.join(job_dir, "commands.log"), "a") as f:
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
    #subprocess.run(
    #    [
    #        "python",
    #        os.path.join(pdb.scripts_dir, "RenumFiles.py"),
    #        pdb.pdb_base,
    #        pdb.job_dir,
    #        pdb.mode,
    #    ]
    #)

    with open(os.path.join(job_dir, "commands.log"), "a") as f:
        f.write(
            f"\npython {os.path.join(pdb.scripts_dir, 'RenumFiles.py')} {pdb.pdb_base} {pdb.job_dir} {pdb.mode}"
        )

    if pdb.mode == "configurational" or pdb.mode == "mutational":
        xadens(pdb)
        # Stop execution if the mode is 'configurational' or 'mutational'
        # exit()
    print("-----------------------------Reorganization-----------------------------")
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

    if graphics and pdb.mode != "singleresidue":
        print("-----------------------------Images-----------------------------")
        images_dir = os.path.join(pdb.job_dir, "Images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        plot_5andens(pdb, save=True)
        plot_5adens_proportions(pdb, save=True)
        plot_contact_map(pdb, save=True)

    if visualization and pdb.mode != "singleresidue":
        print(
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
        # visualization_dir = os.path.join(pdb.job_dir, "VisualizationScripts")
        # if not os.path.exists(visualization_dir):
        #    os.makedirs(visualization_dir)
        # subprocess.run(
        #    [
        #        "cp",
        #        os.path.join(frustration_dir, f"{pdb.pdb_base}.pdb"),
        #        os.path.join(visualization_dir, f"{pdb.pdb_base}.pdb"),
        #    ]
        # )
        # subprocess.run(["mv", f"*_{pdb.mode}.pml", visualization_dir])
        # subprocess.run(["mv", f"*_{pdb.mode}.tcl", visualization_dir])
        # subprocess.run(["mv", f"*_{pdb.mode}.jml", visualization_dir])
        # subprocess.run(
        #    ["cp", os.path.join(pdb.scripts_dir, "draw_links.py"), visualization_dir]
        # )
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
                        print(
                            f"The file {os.path.basename(file_path)} already exists in the visualization directory. It will be overwritten."
                        )
                        # Remove the file
                        os.remove(
                            os.path.join(visualization_dir, os.path.basename(file_path))
                        )
                        # Move the file
                        shutil.move(file_path, visualization_dir)
                except shutil.Error:
                    print(
                        f"The file {os.path.basename(file_path)} already exists in the visualization directory. It will be overwritten."
                    )

                # shutil.move(file_path, visualization_dir)

        # Copy the 'draw_links.py' script to the visualization directory
        shutil.copy(
            os.path.join(pdb.scripts_dir, "draw_links.py"),
            visualization_dir,
        )

    print("\n\n****Storage information****")
    print(f"The frustration data was stored in {frustration_dir}")
    if graphics and pdb.mode != "singleresidue":
        print(f"Graphics are stored in {images_dir}")
    if visualization and pdb.mode != "singleresidue":
        print(f"Visualizations are stored in {visualization_dir}")

    # List all files and directories in the job directory
    all_items = glob.glob(os.path.join(job_dir, "*"))

    # Filter out directories and hidden files (starting with '.')
    files_to_remove = [
        item
        for item in all_items
        if os.path.isfile(item) and not os.path.basename(item).startswith(".")
    ]

    # Remove the files
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

    return pdb


def dir_frustration(
    pdbs_dir: str,
    order_list: Optional[List[str]] = None,
    chain: Optional[Union[str, List[str]]] = None,
    electrostatics_k: Optional[float] = None,
    seq_dist: int = 12,
    mode: str = "configurational",
    graphics: bool = True,
    visualization: bool = True,
    results_dir: str = None,
) -> None:
    """
    Calculate local energy frustration for all protein structures in one directory.

    Args:
        pdbs_dir (str): Directory containing all protein structures. The full path to the file is needed.
        order_list (Optional[List[str]]): Ordered list of PDB files to calculate frustration. If it is None, frustration is calculated for all PDBs. Default: None.
        chain (Optional[Union[str, List[str]]]): Chain or Chains of the protein structure. Default: None.
        electrostatics_k (Optional[float]): K constant to use in the electrostatics Mode. Default: None (no electrostatics is considered).
        seq_dist (int): Sequence at which contacts are considered to interact (3 or 12). Default: 12.
        mode (str): Local frustration index to be calculated (configurational, mutational, singleresidue). Default: configurational.
        graphics (bool): The corresponding graphics are made. Default: True.
        visualization (bool): Make visualizations, including pymol. Default: True.
        results_dir (str): Path to the folder where results will be stored.
    """
    if results_dir is None:
        results_dir = os.path.join(tempfile.gettempdir(), "")
    elif not os.path.exists(results_dir):
        # Make the results directory and absolute path
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir)
        print(f"The results directory {results_dir} has been created.")

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
        with open(modes_log_file, "r") as f:
            modes = f.read().splitlines()
        if mode in modes:
            calculation_enabled = False

    if calculation_enabled:
        if order_list is None:
            order_list = [
                f for f in os.listdir(pdbs_dir) if f.endswith((".pdb", ".PDB"))
            ]

        for pdb_file in order_list:
            pdb_path = os.path.join(pdbs_dir, pdb_file)
            calculate_frustration(
                pdb_file=pdb_path,
                chain=chain,
                electrostatics_k=electrostatics_k,
                seq_dist=seq_dist,
                mode=mode,
                graphics=graphics,
                visualization=visualization,
                results_dir=results_dir,
            )

        with open(modes_log_file, "a") as f:
            f.write(mode + "\n")

        print("\n\n****Storage information****")
        print(
            f"Frustration data for all Pdb's directory {pdbs_dir} are stored in {results_dir}"
        )


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
        print(f"The results directory {results_dir} has been created.")

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

    print(
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

    print(
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

    print("\n\n****Storage information****")
    print(f"The frustration of the full dynamic is stored in {results_dir}")

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


def dynamic_res(
    dynamic: "Dynamic", resno: int, chain: str, graphics: bool = True
) -> "Dynamic":
    """
    Obtain the local frustration of a specific residue in the dynamics.

    Args:
        dynamic (Dynamic): Dynamic frustration object.
        resno (int): Residue specific analyzed.
        chain (str): Chain of specific residue.
        graphics (bool): If it is True, the graphs corresponding to the residual frustration in the dynamics are made and stored according to the frustration index used. Default: True.

    Returns:
        Dynamic: Dynamic frustration object adding Mutations attribute for the residue of the indicated chain.
    """
    if graphics not in [True, False]:
        raise ValueError("Graphics must be a boolean value!")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(
        "structure",
        os.path.join(
            dynamic.results_dir,
            f"{dynamic.order_list[0]}.done/FrustrationData/{dynamic.order_list[0]}",
        ),
    )

    if len(structure[0][chain][(" ", resno, " ")].child_list) == 0:
        if chain not in [chain.id for chain in structure.get_chains()]:
            raise ValueError(
                f"Chain {chain} doesn't exist. The chains found are: {', '.join([chain.id for chain in structure.get_chains()])}"
            )
        else:
            raise ValueError(f"Resno {resno} of chain {chain} doesn't exist.")

    plots_dir = os.path.join(dynamic.results_dir, f"Dynamic_plots_res_{resno}_{chain}")
    result_file = os.path.join(plots_dir, f"{dynamic.mode}_Res_{resno}_{chain}")

    os.makedirs(plots_dir, exist_ok=True)

    if os.path.exists(result_file):
        os.remove(result_file)

    print(
        "-----------------------------Getting frustrated data-----------------------------"
    )
    if dynamic.mode == "configurational" or dynamic.mode == "mutational":
        with open(result_file, "w") as f:
            f.write(
                "Res ChainRes Total nHighlyFrst nNeutrallyFrst nMinimallyFrst relHighlyFrustrated relNeutralFrustrated relMinimallyFrustrated\n"
            )

        for pdb_file in dynamic.order_list:
            data_res = pd.read_csv(
                os.path.join(
                    dynamic.results_dir,
                    f"{pdb_file}.done/FrustrationData/{pdb_file}.pdb_{dynamic.mode}_5adens",
                ),
                sep="\s+",
                header=0,
            )
            data_res = data_res[
                (data_res["Res"] == resno) & (data_res["ChainRes"] == chain)
            ]

            with open(result_file, "a") as f:
                data_res.to_csv(f, sep="\t", header=False, index=False)

    else:
        with open(result_file, "w") as f:
            f.write(
                "Res ChainRes DensityRes AA NativeEnergy DecoyEnergy SDEnergy FrstIndex\n"
            )

        for pdb_file in dynamic.order_list:
            data_res = pd.read_csv(
                os.path.join(
                    dynamic.results_dir,
                    f"{pdb_file}.done/FrustrationData/{pdb_file}.pdb_singleresidue",
                ),
                sep="\s+",
                header=0,
            )
            data_res = data_res[
                (data_res["Res"] == resno) & (data_res["ChainRes"] == chain)
            ]

            with open(result_file, "a") as f:
                data_res.to_csv(f, sep="\t", header=False, index=False)

    if chain not in dynamic.residues_dynamic:
        dynamic.residues_dynamic[chain] = {}

    dynamic.residues_dynamic[chain][f"Res_{resno}"] = result_file

    print("\n\n****Storage information****")
    print(f"The frustration of the residue is stored in {result_file}")

    if graphics:
        # Raise NotImplementedError("Visualization functions for dynamics are not implemented in the new version.")
        raise NotImplementedError(
            "Visualization functions for dynamics are not implemented in the new version."
        )
        # plot_res_dynamics(dynamic, resno, chain, save=True)
        if dynamic.mode != "singleresidue":
            plot_dynamic_res_5adens_proportion(dynamic, resno, chain, save=True)

    return dynamic


def mutate_res(
    pdb: "Pdb", resno: int, chain: str, split: bool = True, method: str = "threading"
) -> "Pdb":
    """
    Calculate the local energy frustration for each of the 20 residual variants in the Resno position and Chain chain.
    Use the frustration index indicated in the Pdb object.

    Args:
        pdb (Pdb): Pdb frustration object.
        resno (int): Resno of the residue to be mutated.
        chain (str): Chain of the residue to be mutated.
        split (bool): Split that you are going to calculate frustration. If it is True specific string, if it is False full complex. Default: True.
        method (str): Method indicates the method to use to perform the mutation (Threading or Modeller). Default: Threading.

    Returns:
        Pdb: Returns Pdb frustration object with corresponding Mutation attribute.
    """
    # Check if the residue exists in the specified chain
    if (
        len(
            pdb.atom[
                (pdb.atom["resno"] == resno)
                & (pdb.atom["chain"] == chain)
                & (pdb.atom["atom_name"] == "CA")
            ]
        )
        == 0
    ):
        raise ValueError("Resno of chain doesn't exist!")
    elif pd.isna(
        pdb.atom[(pdb.atom["resno"] == resno) & (pdb.atom["atom_name"] == "CA")][
            "chain"
        ].iloc[0]
    ):
        chain = "A"

    if split not in [True, False]:
        raise ValueError("Split must be a boolean value!")

    method = method.lower()
    if method not in ["threading", "modeller"]:
        raise ValueError(
            f"{method} isn't a method of mutation. The methods are: threading or modeller!"
        )

    if split == False and method == "modeller":
        raise ValueError("Complex modeling with Modeller isn't available!")

    # Output file
    mutations_dir = os.path.join(pdb.job_dir, "MutationsData")
    os.makedirs(mutations_dir, exist_ok=True)
    frustra_mut_file = os.path.join(
        mutations_dir, f"{pdb.mode}_Res{resno}_{method}_{chain}.txt"
    )

    if os.path.exists(frustra_mut_file):
        os.remove(frustra_mut_file)

    if pdb.mode == "configurational" or pdb.mode == "mutational":
        with open(frustra_mut_file, "w") as f:
            f.write("Res1 Res2 ChainRes1 ChainRes2 AA1 AA2 FrstIndex FrstState\n")
    elif pdb.mode == "singleresidue":
        with open(frustra_mut_file, "w") as f:
            f.write("Res ChainRes AA FrstIndex\n")

    # Data types, files
    col_classes = []
    if pdb.mode == "singleresidue":
        col_classes = ["int", "str", "float", "str", "float", "float", "float", "float"]
    elif pdb.mode == "configurational" or pdb.mode == "mutational":
        col_classes = [
            "int",
            "int",
            "str",
            "str",
            "float",
            "float",
            "str",
            "str",
            "float",
            "float",
            "float",
            "float",
            "str",
            "str",
        ]

    if method == "threading":
        aa_vector = [
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
        glycine = False

        # Indices of all the atoms of the residue to mutate
        index_totales = pdb.atom[
            (pdb.atom["chain"] == chain) & (pdb.atom["resno"] == resno)
        ].index

        # If it were glycine
        index_backbone_gly = pdb.atom[
            (pdb.atom["chain"] == chain)
            & (pdb.atom["resno"] == resno)
            & (pdb.atom["atom_name"].isin(["N", "CA", "C", "O"]))
        ].index

        # Check if it is glycine
        if pdb.atom.loc[index_totales[0], "resid"] == "GLY":
            glycine = True
        else:
            glycine = False

        # Backbone indices
        if glycine:
            index_backbone = pdb.atom[
                (pdb.atom["chain"] == chain)
                & (pdb.atom["resno"] == resno)
                & (pdb.atom["atom_name"].isin(["N", "CA", "C", "O"]))
            ].index
        else:
            index_backbone = pdb.atom[
                (pdb.atom["chain"] == chain)
                & (pdb.atom["resno"] == resno)
                & (pdb.atom["atom_name"].isin(["N", "CA", "C", "O", "CB"]))
            ].index

        # It is mutated by 20 AA
        for aa in aa_vector:
            print(
                f"\n-----------------------------Getting variant {aa}-----------------------------"
            )
            pdb_mut = pdb.copy()

            # If AA is equal to the residue it is not necessary to mutate and neither is it glycine
            if aa != pdb_mut.atom.loc[index_totales[0], "resid"] and not glycine:
                # if the residue to be inserted is not glycine, insert backbone with CB
                if aa != "GLY":
                    diff_atom = index_totales.difference(index_backbone)
                    diff_xyz = (
                        pdb_mut.atom.loc[index_totales, ["x", "y", "z"]]
                        .values.flatten()
                        .tolist()
                    )
                    diff_xyz = [
                        coord
                        for coord in diff_xyz
                        if coord
                        not in pdb_mut.atom.loc[index_backbone, ["x", "y", "z"]]
                        .values.flatten()
                        .tolist()
                    ]
                # if the residue to be inserted is glycine, a backbone without CB is inserted
                else:
                    diff_atom = index_totales.difference(index_backbone_gly)
                    diff_xyz = (
                        pdb_mut.atom.loc[index_totales, ["x", "y", "z"]]
                        .values.flatten()
                        .tolist()
                    )
                    diff_xyz = [
                        coord
                        for coord in diff_xyz
                        if coord
                        not in pdb_mut.atom.loc[index_backbone_gly, ["x", "y", "z"]]
                        .values.flatten()
                        .tolist()
                    ]

                # If the previous subtraction is not empty, the corresponding atoms and coordinates are removed
                if len(diff_atom) > 0:
                    pdb_mut.atom = pdb_mut.atom.drop(diff_atom)
                if len(diff_xyz) > 0:
                    pdb_mut.atom = pdb_mut.atom[
                        ~pdb_mut.atom[["x", "y", "z"]]
                        .apply(tuple, axis=1)
                        .isin([tuple(coord) for coord in diff_xyz])
                    ]

            # Residues are renamed
            rename = pdb_mut.atom[
                (pdb_mut.atom["chain"] == chain) & (pdb_mut.atom["resno"] == resno)
            ].index
            pdb_mut.atom.loc[rename, "resid"] = aa

            # Mutated PDB is saved
            if split:
                pdb_mut.to_pdb(
                    os.path.join(pdb.job_dir, f"{pdb.pdb_base}_{resno}_{aa}.pdb")
                )
            else:
                pdb_mut.to_pdb(
                    os.path.join(
                        pdb.job_dir, f"{pdb.pdb_base}_{resno}_{aa}_{chain}.pdb"
                    )
                )

            print(
                "----------------------------Calculating frustration-----------------------------"
            )
            if split:
                calculate_frustration(
                    pdb_file=os.path.join(
                        pdb.job_dir, f"{pdb.pdb_base}_{resno}_{aa}.pdb"
                    ),
                    mode=pdb.mode,
                    results_dir=pdb.job_dir,
                    graphics=False,
                    visualization=False,
                    chain=chain,
                )
            else:
                calculate_frustration(
                    pdb_file=os.path.join(
                        pdb.job_dir, f"{pdb.pdb_base}_{resno}_{aa}_{chain}.pdb"
                    ),
                    mode=pdb.mode,
                    results_dir=pdb.job_dir,
                    graphics=False,
                    visualization=False,
                )

            print("----------------------------Storing-----------------------------")
            if pdb.mode == "singleresidue":
                os.rename(
                    os.path.join(
                        pdb.job_dir,
                        f"{pdb.pdb_base}_{resno}_{aa}_{chain}.done/FrustrationData/{pdb.pdb_base}_{resno}_{aa}_{chain}.pdb_singleresidue",
                    ),
                    os.path.join(
                        mutations_dir,
                        f"{pdb.pdb_base}_{resno}_{aa}_{chain}.pdb_singleresidue",
                    ),
                )
                frustra_table = pd.read_csv(
                    os.path.join(
                        mutations_dir,
                        f"{pdb.pdb_base}_{resno}_{aa}_{chain}.pdb_singleresidue",
                    ),
                    sep="\s+",
                    header=0,
                    dtype=dict(
                        zip(["Res", "ChainRes", "AA", "FrstIndex"], col_classes)
                    ),
                )
                frustra_table = frustra_table[
                    (frustra_table["ChainRes"] == chain)
                    & (frustra_table["Res"] == resno)
                ][["Res", "ChainRes", "AA", "FrstIndex"]]
                frustra_table.to_csv(
                    frustra_mut_file, sep="\t", header=False, index=False, mode="a"
                )

            elif pdb.mode == "configurational" or pdb.mode == "mutational":
                os.rename(
                    os.path.join(
                        pdb.job_dir,
                        f"{pdb.pdb_base}_{resno}_{aa}_{chain}.done/FrustrationData/{pdb.pdb_base}_{resno}_{aa}_{chain}.pdb_{pdb.mode}",
                    ),
                    os.path.join(
                        mutations_dir,
                        f"{pdb.pdb_base}_{resno}_{aa}_{chain}.pdb_{pdb.mode}",
                    ),
                )
                frustra_table = pd.read_csv(
                    os.path.join(
                        mutations_dir,
                        f"{pdb.pdb_base}_{resno}_{aa}_{chain}.pdb_{pdb.mode}",
                    ),
                    sep="\s+",
                    header=0,
                    dtype=dict(
                        zip(
                            [
                                "Res1",
                                "Res2",
                                "ChainRes1",
                                "ChainRes2",
                                "AA1",
                                "AA2",
                                "FrstIndex",
                                "FrstState",
                            ],
                            col_classes,
                        )
                    ),
                )
                frustra_table = frustra_table[
                    (
                        (frustra_table["ChainRes1"] == chain)
                        & (frustra_table["Res1"] == resno)
                    )
                    | (
                        (frustra_table["ChainRes2"] == chain)
                        & (frustra_table["Res2"] == resno)
                    )
                ][
                    [
                        "Res1",
                        "Res2",
                        "ChainRes1",
                        "ChainRes2",
                        "AA1",
                        "AA2",
                        "FrstIndex",
                        "FrstState",
                    ]
                ]
                frustra_table.to_csv(
                    frustra_mut_file, sep="\t", header=False, index=False, mode="a"
                )

            # Unnecessary files are removed
            os.remove(
                os.path.join(
                    mutations_dir, f"{pdb.pdb_base}_{resno}_{aa}_{chain}.pdb_{pdb.mode}"
                )
            )
            os.system(
                f"rm -R {os.path.join(pdb.job_dir, f'{pdb.pdb_base}_{resno}_{aa}_{chain}.done/')}"
            )
            os.system(f"cd {pdb.job_dir} ; rm *pdb")

    elif method == "modeller":
        # Raise NotImplementedError("Modeling with Modeller isn't available yet!")
        raise NotImplementedError("Modeling with Modeller isn't available yet!")
        if not has_modeller():
            raise ImportError("Please install modeller package to continue!")

        aa_vector = [
            "L",
            "D",
            "I",
            "N",
            "T",
            "V",
            "A",
            "G",
            "E",
            "R",
            "K",
            "H",
            "Q",
            "S",
            "P",
            "F",
            "Y",
            "M",
            "W",
            "C",
        ]

        print(
            "-----------------------------Getting sequence-----------------------------"
        )
        pos = 0
        pdb_id = pdb.pdb_base
        for i in range(len(pdb.pdb_base)):
            if pdb.pdb_base[i] == "_":
                pos = i
                break

        if pos != 0:
            pdb_id = pdb.pdb_base[:pos]

        pdb_file = os.path.join(
            pdb.job_dir, "FrustrationData", f"{pdb.pdb_base}_{chain}.pdb"
        )
        print(pdb_file)
        pdb_structure = PDBParser(QUIET=True).get_structure("structure", pdb_file)
        seq = "".join(
            [
                residue.resname
                for residue in pdb_structure[0][chain].get_residues()
                if residue.id[0] == " "
            ]
        )
        seq = list(seq)
        seq[resno - 1] = "-"
        seq = "".join(seq)

        fasta_id = f">{pdb_id}_{chain}.pdb"
        with open(os.path.join(pdb.job_dir, "seqs.fasta"), "w") as f:
            f.write(fasta_id + "\n")
            f.write(seq + "\n")

        fasta = list(SeqIO.parse(os.path.join(pdb.job_dir, "seqs.fasta"), "fasta"))
        seq_list = []
        rowname = []
        for record in fasta:
            seq_list.append(list(str(record.seq)))
            rowname.append(record.id[-1].upper())

        print("-----------------------------Equivalences-----------------------------")
        seq_pdb = pd.DataFrame(
            {
                "AA": [
                    residue.resname
                    for residue in pdb_structure[0][chain].get_residues()
                    if residue.id[0] == " "
                ],
                "resno": [
                    residue.id[1]
                    for residue in pdb_structure[0][chain].get_residues()
                    if residue.id[0] == " "
                ],
                "index": list(
                    range(
                        1,
                        len(
                            [
                                residue
                                for residue in pdb_structure[0][chain].get_residues()
                                if residue.id[0] == " "
                            ]
                        )
                        + 1,
                    )
                ),
            }
        )

        aln = pd.DataFrame(
            seq_list[: seq.count("-")], columns=list(range(len(seq_list[0])))
        )
        aln = pd.concat([aln, pd.DataFrame(seq_pdb["AA"])], ignore_index=True)
        aln.to_csv(
            os.path.join(pdb.job_dir, "alignment.fasta"),
            sep="\t",
            header=False,
            index=False,
        )

        # from modeller import *

        env = environ()
        aln = alignment(
            env, file=os.path.join(pdb.job_dir, "alignment.fasta"), format="FASTA"
        )
        aln.write(
            file=os.path.join(pdb.job_dir, "alignment.ali"), alignment_format="PIR"
        )

        seq_gap = pd.DataFrame(
            {
                "AA": list(aln[1].residues.sequence),
                "resno": [0] * len(aln[1].residues.sequence),
                "index": list(range(1, len(aln[1].residues.sequence) + 1)),
            }
        )

        j = 0
        for i in range(len(seq_gap)):
            if (
                seq_gap.loc[i, "AA"] != "-"
                and seq_gap.loc[i, "AA"] == seq_pdb.loc[j, "AA"]
            ):
                seq_gap.loc[i, "resno"] = seq_pdb.loc[j, "resno"]
                j += 1

        pos = seq_gap[seq_gap["resno"] == resno].index[0]

        os.system(f"cp {os.path.join(pdb.scripts_dir, 'align2d.py')} {pdb.job_dir}")
        os.system(f"cp {os.path.join(pdb.scripts_dir, 'make_ali.py')} {pdb.job_dir}")
        os.system(
            f"cp {os.path.join(pdb.scripts_dir, 'model-single.py')} {pdb.job_dir}"
        )
        os.system(f"cp {pdb.pdb_path} {pdb.job_dir}")

        for aa in aa_vector:
            print(
                f"\n-----------------------------Getting variant {aa}-----------------------------"
            )
            if split:
                with open(os.path.join(pdb.job_dir, "Modelo.fa"), "w") as f:
                    f.write(">Modelo\n")
                    seq_mut = seq_list[: seq.count("-")]
                    seq_mut[pos] = aa
                    f.write("".join(seq_mut) + "\n")

            print("-----------------------------Aligning-----------------------------")
            os.system(f"cd {pdb.job_dir} ;python3 make_ali.py Modelo")
            if split:
                os.system(
                    f"cd {pdb.job_dir} ;python3 align2d.py {pdb.pdb_base} Modelo {chain}"
                )

            print("-----------------------------Modeling-----------------------------")
            os.system(
                f"cd {pdb.job_dir} ;python3 model-single.py {pdb.pdb_base} Modelo"
            )
            os.rename(
                os.path.join(pdb.job_dir, "Modelo.B99990001.pdb"),
                os.path.join(pdb.job_dir, f"{pdb.pdb_base}_{resno}_{aa}_{chain}.pdb"),
            )
            os.system(
                f"cd {pdb.job_dir} ;rm *D00000001 *ini *rsr *sch *V99990001 *ali *pap *fa"
            )

            print(
                "----------------------------Calculating frustration-----------------------------"
            )
            calculate_frustration(
                pdb_file=os.path.join(
                    pdb.job_dir, f"{pdb.pdb_base}_{resno}_{aa}_{chain}.pdb"
                ),
                mode=pdb.mode,
                results_dir=pdb.job_dir,
                graphics=False,
                visualization=False,
            )

            print("----------------------------Storing-----------------------------")
            if pdb.mode == "singleresidue":
                os.rename(
                    os.path.join(
                        pdb.job_dir,
                        f"{pdb.pdb_base}_{resno}_{aa}_{chain}.done/FrustrationData/{pdb.pdb_base}_{resno}_{aa}_{chain}.pdb_singleresidue",
                    ),
                    os.path.join(
                        mutations_dir,
                        f"{pdb.pdb_base}_{resno}_{aa}_{chain}.pdb_singleresidue",
                    ),
                )
                frustra_table = pd.read_csv(
                    os.path.join(
                        mutations_dir,
                        f"{pdb.pdb_base}_{resno}_{aa}_{chain}.pdb_singleresidue",
                    ),
                    sep="\s+",
                    header=0,
                    dtype=dict(
                        zip(["Res", "ChainRes", "AA", "FrstIndex"], col_classes)
                    ),
                )
                frustra_table = frustra_table[frustra_table["Res"] == pos][
                    ["Res", "ChainRes", "AA", "FrstIndex"]
                ]
                frustra_table.to_csv(
                    frustra_mut_file, sep="\t", header=False, index=False, mode="a"
                )

            elif pdb.mode == "configurational" or pdb.mode == "mutational":
                os.rename(
                    os.path.join(
                        pdb.job_dir,
                        f"{pdb.pdb_base}_{resno}_{aa}_{chain}.done/FrustrationData/{pdb.pdb_base}_{resno}_{aa}_{chain}.pdb_{pdb.mode}",
                    ),
                    os.path.join(
                        mutations_dir,
                        f"{pdb.pdb_base}_{resno}_{aa}_{chain}.pdb_{pdb.mode}",
                    ),
                )
                frustra_table = pd.read_csv(
                    os.path.join(
                        mutations_dir,
                        f"{pdb.pdb_base}_{resno}_{aa}_{chain}.pdb_{pdb.mode}",
                    ),
                    sep="\s+",
                    header=0,
                    dtype=dict(
                        zip(
                            [
                                "Res1",
                                "Res2",
                                "ChainRes1",
                                "ChainRes2",
                                "AA1",
                                "AA2",
                                "FrstIndex",
                                "FrstState",
                            ],
                            col_classes,
                        )
                    ),
                )
                frustra_table = frustra_table[
                    (frustra_table["Res1"] == pos) | (frustra_table["Res2"] == pos)
                ][
                    [
                        "Res1",
                        "Res2",
                        "ChainRes1",
                        "ChainRes2",
                        "AA1",
                        "AA2",
                        "FrstIndex",
                        "FrstState",
                    ]
                ]
                frustra_table.to_csv(
                    frustra_mut_file, sep="\t", header=False, index=False, mode="a"
                )

            # Unnecessary files are removed
            os.remove(
                os.path.join(
                    mutations_dir, f"{pdb.pdb_base}_{resno}_{aa}_{chain}.pdb_{pdb.mode}"
                )
            )
            os.system(
                f"rm -R {os.path.join(pdb.job_dir, f'{pdb.pdb_base}_{resno}_{aa}_{chain}.done/')}"
            )
            os.system(f"cd {pdb.job_dir} ; rm {pdb.pdb_base}_*")

        os.system(f"cd {pdb.job_dir} ; rm *pdb seqs.fasta *py")

        print("----------------------------Renumbering-----------------------------")
        if pdb.mode == "singleresidue":
            data = pd.read_csv(
                os.path.join(
                    mutations_dir, f"singleresidue_Res{resno}_{method}_{chain}.txt"
                ),
                sep="\s+",
                header=0,
            )
            data["ChainRes"] = chain
            for i in range(len(data)):
                data.loc[i, "Res"] = seq_gap.loc[data.loc[i, "Res"], "resno"]
            data.to_csv(
                os.path.join(
                    mutations_dir, f"singleresidue_Res{resno}_{method}_{chain}.txt"
                ),
                sep="\t",
                header=True,
                index=False,
            )

        elif pdb.mode == "configurational":
            data = pd.read_csv(
                os.path.join(
                    mutations_dir, f"configurational_Res{resno}_{method}_{chain}.txt"
                ),
                sep="\s+",
                header=0,
            )
            data[["ChainRes1", "ChainRes2"]] = chain
            for i in range(len(data)):
                data.loc[i, "Res1"] = seq_gap.loc[data.loc[i, "Res1"], "resno"]
                data.loc[i, "Res2"] = seq_gap.loc[data.loc[i, "Res2"], "resno"]
            data.to_csv(
                os.path.join(
                    mutations_dir, f"configurational_Res{resno}_{method}_{chain}.txt"
                ),
                sep="\t",
                header=True,
                index=False,
            )

        elif pdb.mode == "mutational":
            data = pd.read_csv(
                os.path.join(
                    mutations_dir, f"mutational_Res{resno}_{method}_{chain}.txt"
                ),
                sep="\s+",
                header=0,
            )
            data[["ChainRes1", "ChainRes2"]] = chain
            for i in range(len(data)):
                data.loc[i, "Res1"] = seq_gap.loc[data.loc[i, "Res1"], "resno"]
                data.loc[i, "Res2"] = seq_gap.loc[data.loc[i, "Res2"], "resno"]
            data.to_csv(
                os.path.join(
                    mutations_dir, f"mutational_Res{resno}_{method}_{chain}.txt"
                ),
                sep="\t",
                header=True,
                index=False,
            )

    if "Mutations" not in pdb.__dict__:
        pdb.Mutations = {}
    if method not in pdb.Mutations:
        pdb.Mutations[method] = {}
    pdb.Mutations[method][f"Res_{resno}_{chain}"] = {
        "Method": method,
        "Res": resno,
        "Chain": chain,
        "File": os.path.join(
            mutations_dir, f"{pdb.mode}_Res{resno}_{method}_{chain}.txt"
        ),
    }

    print("\n\n****Storage information****")
    print(
        f"The frustration of the residue is stored in {os.path.join(mutations_dir, f'{pdb.mode}_Res{resno}_{method}_{chain}.txt')}"
    )

    return pdb


def detect_dynamic_clusters(
    dynamic: "Dynamic",
    loess_span: float = 0.05,
    min_frst_range: float = 0.7,
    filt_mean: float = 0.15,
    ncp: int = 10,
    min_corr: float = 0.95,
    leiden_resol: float = 1,
    corr_type: str = "spearman",
) -> "Dynamic":
    """
    Detects residue modules with similar single-residue frustration dynamics.
    It filters out the residuals with variable dynamics, for this, it adjusts a loess
    model with span = LoessSpan and calculates the dynamic range of frustration and the mean of single-residue frustration.
    It is left with the residuals with a dynamic frustration range greater than the quantile defined by MinFrstRange and with a mean Mean <(-FiltMean) or Mean> FiltMean.
    Performs an analysis of main components and keeps Ncp components, to compute the correlation(CorrType) between them and keep the residues that have greater correlation MinCorr and p-value> 0.05.
    An undirected graph is generated and Leiden clustering is applied with LeidenResol resolution.

    Args:
        dynamic (Dynamic): Dynamic Frustration Object.
        loess_span (float): Parameter α > 0 that controls the degree of smoothing of the loess() function of model fit. Default: 0.05.
        min_frst_range (float): Frustration dynamic range filter threshold. 0 <= MinFrstRange <= 1. Default: 0.7.
        filt_mean (float): Frustration Mean Filter Threshold. FiltMean >= 0. Default: 0.15.
        ncp (int): Number of principal components to be used in PCA(). Ncp >= 1. Default: 10.
        min_corr (float): Correlation filter threshold. 0 <= MinCorr <= 1. Default: 0.95.
        leiden_resol (float): Parameter that defines the coarseness of the cluster. LeidenResol > 0. Default: 1.
        corr_type (str): Type of correlation index to compute. Values: "pearson" or "spearman". Default: "spearman".

    Returns:
        Dynamic: Dynamic Frustration Object and its Clusters attribute.
    """
    if dynamic.mode != "singleresidue":
        raise ValueError(
            "This functionality is only available for the singleresidue index, run dynamic_frustration() with Mode = 'singleresidue'"
        )

    corr_type = corr_type.lower()
    if corr_type not in ["pearson", "spearman"]:
        raise ValueError(
            "Correlation type(CorrType) indicated isn't available or doesn't exist, indicate 'pearson' or 'spearman'"
        )

    required_libraries = ["leidenalg", "igraph", "sklearn", "scipy", "numpy", "pandas"]
    missing_libraries = [
        library for library in required_libraries if library not in globals()
    ]
    if missing_libraries:
        raise ImportError(
            f"Please install the following libraries to continue: {', '.join(missing_libraries)}"
        )

    # Loading residues and resno
    ini = pd.read_csv(
        os.path.join(
            dynamic.results_dir,
            f"{os.path.splitext(dynamic.order_list[0])[0]}.done/FrustrationData/{os.path.splitext(dynamic.order_list[0])[0]}.pdb_singleresidue",
        ),
        sep="\s+",
        header=0,
    )
    residues = ini["AA"].tolist()
    resnos = ini["Res"].tolist()

    # Loading data
    print("-----------------------------Loading data-----------------------------")
    frustra_data = pd.DataFrame()
    for pdb_file in dynamic.order_list:
        read = pd.read_csv(
            os.path.join(
                dynamic.results_dir,
                f"{os.path.splitext(pdb_file)[0]}.done/FrustrationData/{os.path.splitext(pdb_file)[0]}.pdb_singleresidue",
            ),
            sep="\s+",
            header=0,
        )
        frustra_data[f"frame_{len(frustra_data.columns)}"] = read["FrstIndex"]

    frustra_data.index = [
        f"{residue}_{resno}" for residue, resno in zip(residues, resnos)
    ]

    # Model fitting and filter by difference and mean
    print(
        "-----------------------------Model fitting and filtering by dynamic range and frustration mean-----------------------------"
    )
    frstrange = []
    means = []
    sds = []
    fitted = pd.DataFrame()
    for i in range(len(residues)):
        res = pd.DataFrame(
            {"Frustration": frustra_data.iloc[i], "Frames": range(len(frustra_data))}
        )
        modelo = lowess(
            res["Frustration"],
            res["Frames"],
            frac=loess_span,
            it=0,
            delta=0.0,
            is_sorted=False,
        )
        fitted[f"res_{i}"] = modelo[:, 1]
        frstrange.append(modelo[:, 1].max() - modelo[:, 1].min())
        means.append(modelo[:, 1].mean())
        sds.append(modelo[:, 1].std())

    estadistics = pd.DataFrame({"Diferences": frstrange, "Means": means})
    frustra_data = frustra_data[
        (
            estadistics["Diferences"]
            > np.quantile(estadistics["Diferences"], min_frst_range)
        )
        & ((estadistics["Means"] < -filt_mean) | (estadistics["Means"] > filt_mean))
    ]

    # Principal component analysis
    print(
        "-----------------------------Principal component analysis-----------------------------"
    )
    pca = PCA(n_components=ncp)
    pca_result = pca.fit_transform(frustra_data.T)

    if corr_type == "spearman":
        corr_func = spearmanr
    else:
        corr_func = pearsonr

    corr_matrix = np.zeros((pca_result.shape[1], pca_result.shape[1]))
    p_values = np.zeros((pca_result.shape[1], pca_result.shape[1]))
    for i in range(pca_result.shape[1]):
        for j in range(i, pca_result.shape[1]):
            corr, p_value = corr_func(pca_result[:, i], pca_result[:, j])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
            p_values[i, j] = p_value
            p_values[j, i] = p_value

    np.fill_diagonal(corr_matrix, 0)
    corr_matrix[
        (corr_matrix < min_corr) & (corr_matrix > -min_corr) | (p_values > 0.05)
    ] = 0

    print("-----------------------------Undirected graph-----------------------------")
    net = ig.Graph.Adjacency((corr_matrix > 0).tolist(), mode="undirected")

    print("-----------------------------Leiden Clustering-----------------------------")
    leiden_clusters = la.find_partition(
        net, la.RBConfigurationVertexPartition, resolution_parameter=leiden_resol
    )
    cluster_data = pd.DataFrame({"cluster": leiden_clusters.membership})
    cluster_data = cluster_data.loc[net.degree() > 0]

    net.delete_vertices(net.vs.select(_degree=0))

    dynamic.clusters["Graph"] = net
    dynamic.clusters["LeidenClusters"] = cluster_data
    dynamic.clusters["LoessSpan"] = loess_span
    dynamic.clusters["MinFrstRange"] = min_frst_range
    dynamic.clusters["FiltMean"] = filt_mean
    dynamic.clusters["Ncp"] = ncp
    dynamic.clusters["MinCorr"] = min_corr
    dynamic.clusters["LeidenResol"] = leiden_resol
    dynamic.clusters["Fitted"] = fitted
    dynamic.clusters["Means"] = means
    dynamic.clusters["FrstRange"] = frstrange
    dynamic.clusters["Sd"] = sds
    dynamic.clusters["CorrType"] = corr_type

    if "Graph" not in dynamic.clusters or dynamic.clusters["Graph"] is None:
        print("The process was not completed successfully!")
    else:
        print("The process has finished successfully!")

    return dynamic
