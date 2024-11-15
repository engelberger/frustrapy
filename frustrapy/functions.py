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
import multiprocessing
from typing import Dict, Any, Tuple
import time
from functools import wraps
import pickle
from dataclasses import dataclass

try:
    from .visualization import *
except ImportError:
    from visualization import *
try:
    from .renum_files import renum_files
except ImportError:
    from renum_files import renum_files


# Add to the imports section at the top
try:
    from .scripts.pdb_to_lammps import PDBToLAMMPS
except ImportError:
    from scripts.pdb_to_lammps import PDBToLAMMPS


import plotly.graph_objects as go


import os
import logging
import shutil
import pandas as pd
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pdb_class import Pdb  # Replace with the actual import path of your Pdb class

# Setup the logging
import logging

# Create a logger for this module
logger = logging.getLogger(__name__)

# Configure the logger if it hasn't been configured
if not logger.handlers:
    # Create handlers
    file_handler = logging.FileHandler("frustrapy.log")
    console_handler = logging.StreamHandler()

    # Create formatters and add it to handlers
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Set level
    logger.setLevel(logging.INFO)


# Add performance monitoring decorator
def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.debug(f"Starting {func.__name__} with args: {args}, kwargs: {kwargs}")

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds")
            logger.exception(f"Exception in {func.__name__}: {str(e)}")
            raise

    return wrapper


# Add memory usage monitoring
def log_memory_usage():
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        logger.debug(
            f"Memory usage - RSS: {memory_info.rss / 1024 / 1024:.2f} MB, VMS: {memory_info.vms / 1024 / 1024:.2f} MB"
        )
    except ImportError:
        logger.debug("psutil not installed - memory usage logging disabled")


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

    def copy(self) -> "Pdb":
        """
        Creates a deep copy of the Pdb object.

        Returns:
            Pdb: A new Pdb object with copied attributes
        """
        return Pdb(
            job_dir=self.job_dir,
            pdb_base=self.pdb_base,
            mode=self.mode,
            atom=self.atom.copy(),  # Create a deep copy of the DataFrame
            equivalences=self.equivalences.copy(),  # Create a deep copy of the DataFrame
            scripts_dir=self.scripts_dir,
        )


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

    def add_residue_dynamic(self, res_num: int, chain: str, data: pd.DataFrame):
        """
        Adds dynamic frustration data for a specific residue.

        Args:
            res_num (int): Residue number.
            chain (str): Chain identifier.
            data (pd.DataFrame): Frustration data for the residue.
        """
        if chain not in self.residues_dynamic:
            self.residues_dynamic[chain] = {}
        self.residues_dynamic[chain][f"Res_{res_num}"] = data

    def get_residue_dynamic(self, res_num: int, chain: str) -> pd.DataFrame:
        """
        Retrieves dynamic frustration data for a specific residue.

        Args:
            res_num (int): Residue number.
            chain (str): Chain identifier.

        Returns:
            pd.DataFrame: Frustration data for the residue.
        """
        try:
            return self.residues_dynamic[chain][f"Res_{res_num}"]
        except KeyError:
            raise ValueError(
                f"No dynamic data found for residue {res_num} in chain {chain}."
            )


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


def get_frustration_dynamic(
    dynamic: "Dynamic", res_num: int, chain: str, frames: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Obtains and returns the table of frustration of a specific chain and residue in a complete dynamic
    or in the indicated frames. The frustration of a specific residue must have previously been calculated
    using dynamic_res().

    Args:
        dynamic (Dynamic): Dynamic frustration object obtained by dynamic_frustration().
        res_num (int): Specific residue.
        chain (str): Specific chain.
        frames (Optional[List[int]]): Specific frames. Default: None.

    Returns:
        pd.DataFrame: Frustration table.
    """
    # Assuming Dynamic object has an attribute 'ResiduesDynamic' which is a dictionary
    # where keys are chain identifiers and values are another dictionary.
    # This inner dictionary has keys formatted as "Res_{res_num}" pointing to the file paths
    # containing the frustration data for each residue.

    if chain not in dynamic.ResiduesDynamic:
        raise ValueError(f"No analysis for chain {chain}.")

    res_key = f"Res_{res_num}"
    if res_key not in dynamic.ResiduesDynamic[chain]:
        raise ValueError(f"No analysis for residue {res_num} in chain {chain}.")

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

    logger.debug(f"Processing PDB file: {pdb_file}")
    pdb_file_base = os.path.basename(pdb_file)
    pdb_filename = pdb_file_base
    logger.debug(f"PDB filename: {pdb_filename}")
    output_path = os.path.join(output_dir, f"{pdb_filename}_equivalences.txt")
    logger.debug(f"Output path: {output_path}")

    equivalences_df.to_csv(output_path, sep="\t", index=False, header=False)
    logger.info(f"Saved equivalences to {output_path}")

    with open(os.path.join(output_dir, "commands.help"), "a") as f:
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


@dataclass
class SingleResidueData:
    """Data class to store single residue frustration analysis results"""

    residue_number: int
    chain_id: str
    residue_name: str
    mutations: Dict[str, float]  # Maps mutation (e.g. 'ALA') to frustration index
    native_energy: float
    decoy_energy: float
    sd_energy: float
    density: float
    plots: Optional[Any] = None  # Store associated plot if available


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


@log_execution_time
def dynamic_res(
    dynamic: "Dynamic", res_num: int, chain: str, graphics: bool = True
) -> "Dynamic":
    """
    Obtain the local frustration of a specific residue in the dynamics.

    Args:
        dynamic (Dynamic): Dynamic frustration object.
        res_num (int): Residue specific analyzed.
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
            f"{os.path.splitext(dynamic.order_list[0])[0]}.done/FrustrationData/{os.path.splitext(dynamic.order_list[0])[0]}.pdb_singleresidue",
        ),
    )

    if len(structure[0][chain][(" ", res_num, " ")].child_list) == 0:
        if chain not in [chain.id for chain in structure.get_chains()]:
            raise ValueError(
                f"Chain {chain} doesn't exist. The chains found are: {', '.join([chain.id for chain in structure.get_chains()])}"
            )
        else:
            raise ValueError(f"Resno {res_num} of chain {chain} doesn't exist.")

    plots_dir = os.path.join(
        dynamic.results_dir, f"Dynamic_plots_res_{res_num}_{chain}"
    )
    result_file = os.path.join(plots_dir, f"{dynamic.mode}_Res_{res_num}_{chain}")

    os.makedirs(plots_dir, exist_ok=True)

    if os.path.exists(result_file):
        os.remove(result_file)

    logger.info(
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
                (data_res["Res"] == res_num) & (data_res["ChainRes"] == chain)
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
                (data_res["Res"] == res_num) & (data_res["ChainRes"] == chain)
            ]

            with open(result_file, "a") as f:
                data_res.to_csv(f, sep="\t", header=False, index=False)

    if chain not in dynamic.residues_dynamic:
        dynamic.residues_dynamic[chain] = {}

    dynamic.residues_dynamic[chain][f"Res_{res_num}"] = result_file

    logger.info("\n\n****Storage information****")
    logger.info(f"The frustration of the residue is stored in {result_file}")

    if graphics:
        # Raise NotImplementedError("Visualization functions for dynamics are not implemented in the new version.")
        raise NotImplementedError(
            "Visualization functions for dynamics are not implemented in the new version."
        )
        # plot_res_dynamics(dynamic, res_num, chain, save=True)
        if dynamic.mode != "singleresidue":
            plot_dynamic_res_5adens_proportion(dynamic, res_num, chain, save=True)

    return dynamic


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
    logger.info(f"Processing variant {aa} for residue {res_num} in chain '{chain}'")

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
    logger.info("Calculating frustration...")
    calculate_frustration(
        pdb_file=output_pdb_path,
        mode=pdb.mode,
        results_dir=pdb.job_dir,
        graphics=False,
        visualization=False,
        chain=chain,
        debug=debug,
    )

    # Store the frustration data
    logger.info("Storing frustration data...")
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

    return {
        "aa": aa,
        "frustra_mut_file": frustra_mut_file,
    }


def mutate_res_parallel(
    pdb: "Pdb",
    res_num: int,
    chain: str,
    split: bool = True,
    method: str = "threading",
    debug: bool = False,
    n_processes: Optional[int] = None,
) -> "Pdb":
    """
    Parallel version of mutate_res that processes amino acid mutations concurrently.

    Args:
        pdb (Pdb): Pdb frustration object
        res_num (int): Residue number to mutate
        chain (str): Chain identifier
        split (bool): Whether to split chains
        method (str): Mutation method ('threading' or 'modeller')
        debug (bool): Debug mode flag
        n_processes (Optional[int]): Number of processes to use. Defaults to CPU count.

    Returns:
        Pdb: Updated Pdb object with mutation results
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    method = method.lower()

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

    # Prepare arguments for parallel processing
    process_args = [
        (aa, pdb, res_num, chain, split, debug, is_glycine, method)
        for aa in amino_acids
    ]

    # Use all available cores if n_processes is not specified
    if n_processes is None:
        n_processes = multiprocessing.cpu_count()

    # Log start of parallel processing
    logger.info(f"Starting parallel mutation processing with {n_processes} processes")

    # Process amino acids in parallel with timing
    process_start = time.time()
    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.map(_process_amino_acid, process_args)
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

    logger.info(
        f"The frustration data for residue {res_num} is stored in {frustra_mut_file}"
    )

    total_time = time.time() - start_time
    logger.info(f"Parallel processing completed in {process_time:.2f} seconds")
    logger.info(f"Total mutation time (including setup): {total_time:.2f} seconds")
    logger.info(
        f"Average time per amino acid: {process_time/len(amino_acids):.2f} seconds"
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
    """
    Serial version of amino acid mutation processing.
    Uses the same helper function as the parallel version for consistency.

    Args:
        pdb (Pdb): Pdb frustration object
        res_num (int): Residue number to mutate
        chain (str): Chain identifier
        split (bool): Whether to split chains
        method (str): Mutation method ('threading' or 'modeller')
        debug (bool): Debug mode flag

    Returns:
        Pdb: Updated Pdb object with mutation results
    """
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

    logger.info("Starting serial mutation processing")

    # Process each amino acid sequentially with timing
    process_start = time.time()
    for aa in amino_acids:
        aa_start = time.time()
        _process_amino_acid((aa, pdb, res_num, chain, split, debug, is_glycine, method))
        logger.debug(f"Processed {aa} in {time.time() - aa_start:.2f} seconds")
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

    logger.info(
        f"The frustration data for residue {res_num} is stored in {frustra_mut_file}"
    )

    total_time = time.time() - start_time
    logger.info(f"Serial processing completed in {process_time:.2f} seconds")
    logger.info(f"Total mutation time (including setup): {total_time:.2f} seconds")
    logger.info(
        f"Average time per amino acid: {process_time/len(amino_acids):.2f} seconds"
    )

    return pdb


@log_execution_time
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

    # Loading residues and res_num
    ini = pd.read_csv(
        os.path.join(
            dynamic.results_dir,
            f"{os.path.splitext(dynamic.order_list[0])[0]}.done/FrustrationData/{os.path.splitext(dynamic.order_list[0])[0]}.pdb_singleresidue",
        ),
        sep="\s+",
        header=0,
    )
    residues = ini["AA"].tolist()
    res_nums = ini["Res"].tolist()

    # Loading data
    logger.info(
        "-----------------------------Loading data-----------------------------"
    )
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
        f"{residue}_{res_num}" for residue, res_num in zip(residues, res_nums)
    ]

    # Model fitting and filter by difference and mean
    logger.info(
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
    logger.info(
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
    logger.info(
        "-----------------------------Undirected graph-----------------------------"
    )
    net = ig.Graph.Adjacency((corr_matrix > 0).tolist(), mode="undirected")

    logger.info(
        "-----------------------------Leiden Clustering-----------------------------"
    )
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
        logger.error("The process was not completed successfully!")
    else:
        logger.info("The process has finished successfully!")

    return dynamic


@log_execution_time
def plot_delta_frus_summary(
    pdb: "Pdb",
    chain: Optional[str] = None,
    method: str = "threading",
    save: bool = True,
) -> go.Figure:
    """
    Creates a summary plot of delta frustration for all residues in specified chain(s).

    Args:
        pdb (Pdb): Pdb frustration object
        chain (Optional[str]): If provided, only plot residues from this chain
        method (str): Mutation method used ('threading' or 'modeller')
        save (bool): Whether to save the plot

    Returns:
        go.Figure: Plotly figure object
    """
    logger = logging.getLogger(__name__)

    if not hasattr(pdb, "Mutations") or method not in pdb.Mutations:
        raise ValueError(f"No mutations found for method '{method}'")

    # Collect data for all residues
    data = []
    for mutation_key, mutation_info in pdb.Mutations[method].items():
        if chain and not mutation_key.endswith(f"_{chain}"):
            continue

        res_num = mutation_info["Res"]
        res_chain = mutation_info["Chain"]

        # Read mutation data
        mutation_data = pd.read_csv(mutation_info["File"], sep="\t", header=0)

        # Add to data collection
        data.append({"res_num": res_num, "chain": res_chain, "data": mutation_data})

    if not data:
        raise ValueError("No mutation data found")

    # Create figure
    fig = go.Figure()

    # Sort data by residue number
    data.sort(key=lambda x: x["res_num"])

    for residue_data in data:
        res_num = residue_data["res_num"]
        res_chain = residue_data["chain"]
        mutation_data = residue_data["data"]

        # Calculate frustration states for each mutation
        if pdb.mode == "singleresidue":
            highly_frustrated = mutation_data[mutation_data["FrstIndex"] <= -1]
            neutral_frustrated = mutation_data[
                (mutation_data["FrstIndex"] > -1) & (mutation_data["FrstIndex"] < 0.78)
            ]
            minimally_frustrated = mutation_data[mutation_data["FrstIndex"] >= 0.78]
        else:
            highly_frustrated = mutation_data[mutation_data["FrstState"] == -1]
            neutral_frustrated = mutation_data[mutation_data["FrstState"] == 0]
            minimally_frustrated = mutation_data[mutation_data["FrstState"] == 1]

        # Add scatter points for each frustration state
        for aa in highly_frustrated["AA"]:
            fig.add_scatter(
                x=[res_num],
                y=[mutation_data[mutation_data["AA"] == aa]["FrstIndex"].iloc[0]],
                mode="text",
                text=[aa],
                textfont=dict(color="red"),
                name=f"Highly frustrated",
                showlegend=False,
            )

        for aa in neutral_frustrated["AA"]:
            fig.add_scatter(
                x=[res_num],
                y=[mutation_data[mutation_data["AA"] == aa]["FrstIndex"].iloc[0]],
                mode="text",
                text=[aa],
                textfont=dict(color="gray"),
                name=f"Neutral frustrated",
                showlegend=False,
            )

        for aa in minimally_frustrated["AA"]:
            fig.add_scatter(
                x=[res_num],
                y=[mutation_data[mutation_data["AA"] == aa]["FrstIndex"].iloc[0]],
                mode="text",
                text=[aa],
                textfont=dict(color="green"),
                name=f"Minimally frustrated",
                showlegend=False,
            )

    # Add reference lines
    fig.add_hline(y=0.78, line_dash="dash", line_color="gray")
    fig.add_hline(y=-1, line_dash="dash", line_color="gray")
    fig.add_hline(y=0, line_color="gray")

    # Update layout
    title = "Delta Frustration Summary"
    if chain:
        title += f" for Chain {chain}"

    fig.update_layout(
        title=title,
        xaxis_title="Residue Position",
        yaxis_title="ΔFrustration",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
    )

    # Add legend items
    fig.add_scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(color="red"),
        name="Highly frustrated",
        showlegend=True,
    )
    fig.add_scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(color="gray"),
        name="Neutral frustrated",
        showlegend=True,
    )
    fig.add_scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(color="green"),
        name="Minimally frustrated",
        showlegend=True,
    )

    if save:
        if chain:
            filename = f"Delta_frus_summary_chain_{chain}.html"
        else:
            filename = "Delta_frus_summary_all_chains.html"

        output_dir = os.path.join(pdb.job_dir, "MutationsData", "Images")
        os.makedirs(output_dir, exist_ok=True)
        fig.write_html(os.path.join(output_dir, filename))
        fig.write_image(os.path.join(output_dir, filename.replace(".html", ".png")))
        logger.info(f"Saved summary plot to {output_dir}/{filename}")

    return fig


@log_execution_time
def plot_delta_frus_heatmap(
    pdb: "Pdb",
    method: str = "threading",
    save: bool = True,
) -> go.Figure:
    """
    Creates a heatmap of delta frustration for all residues across all chains.

    Args:
        pdb (Pdb): Pdb frustration object
        method (str): Mutation method used ('threading' or 'modeller')
        save (bool): Whether to save the plot

    Returns:
        go.Figure: Plotly figure object
    """
    logger = logging.getLogger(__name__)

    if not hasattr(pdb, "Mutations") or method not in pdb.Mutations:
        raise ValueError(f"No mutations found for method '{method}'")

    # Collect data for all residues
    data_dict = {}
    chains = set()
    residues = set()

    for mutation_key, mutation_info in pdb.Mutations[method].items():
        res_num = mutation_info["Res"]
        chain = mutation_info["Chain"]

        # Read mutation data
        mutation_data = pd.read_csv(mutation_info["File"], sep="\t", header=0)

        # Store average frustration for this residue
        data_dict[(res_num, chain)] = mutation_data["FrstIndex"].mean()
        chains.add(chain)
        residues.add(res_num)

    if not data_dict:
        raise ValueError("No mutation data found")

    # Create matrix for heatmap
    chains = sorted(list(chains))
    residues = sorted(list(residues))

    z_data = []
    for chain in chains:
        row = []
        for res in residues:
            value = data_dict.get((res, chain), None)
            row.append(value)
        z_data.append(row)

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=residues,
            y=chains,
            colorscale=[
                [0, "red"],  # Highly frustrated
                [0.4, "white"],  # Neutral
                [1, "green"],  # Minimally frustrated
            ],
            colorbar=dict(
                title="Average ΔFrustration",
                ticktext=["Highly<br>frustrated", "Neutral", "Minimally<br>frustrated"],
                tickvals=[-1, 0, 0.78],
            ),
        )
    )

    # Update layout
    fig.update_layout(
        title="Delta Frustration Heatmap",
        xaxis_title="Residue Position",
        yaxis_title="Chain",
        xaxis=dict(tickmode="linear"),
        yaxis=dict(tickmode="linear"),
    )

    if save:
        output_dir = os.path.join(pdb.job_dir, "MutationsData", "Images")
        os.makedirs(output_dir, exist_ok=True)
        filename = "Delta_frus_heatmap.html"
        fig.write_html(os.path.join(output_dir, filename))
        fig.write_image(os.path.join(output_dir, filename.replace(".html", ".png")))
        logger.info(f"Saved heatmap to {output_dir}/{filename}")

    return fig


# Add this function near other experimental features
@log_execution_time
def experimental_pdb_to_lammps(
    pdb_file: str,
    output_name: str,
    awsem_path: str,
    use_cg_bonds: bool = False,
    use_go_model: bool = False,
    enable_experimental: bool = False,
) -> None:
    """
    Experimental feature: Convert PDB files to LAMMPS format using the new implementation.
    This is currently disabled by default as it needs more testing.

    Args:
        pdb_file (str): Path to input PDB file
        output_name (str): Base name for output files
        awsem_path (str): Path to AWSEM installation
        use_cg_bonds (bool): Enable coarse-grained bonds
        use_go_model (bool): Enable GO model
        enable_experimental (bool): Flag to enable this experimental feature

    Raises:
        RuntimeError: If the experimental feature is not enabled
        ValueError: If input parameters are invalid
    """
    if not enable_experimental:
        raise RuntimeError(
            "This is an experimental feature and is currently disabled by default. "
            "To enable it, set enable_experimental=True. "
            "Note that this feature needs more testing and validation."
        )

    logger.warning(
        "Using experimental PDB to LAMMPS conversion. "
        "This feature is under development and may not work as expected."
    )

    try:
        # Convert paths to Path objects
        pdb_path = Path(pdb_file)
        output_path = Path(output_name)
        awsem_path = Path(awsem_path)

        # Validate inputs
        if not pdb_path.exists():
            raise ValueError(f"PDB file not found: {pdb_file}")

        if not awsem_path.exists():
            raise ValueError(f"AWSEM path not found: {awsem_path}")

        # Create converter and process PDB
        converter = PDBToLAMMPS(
            pdb_file=pdb_path,
            output_prefix=output_path,
            awsem_path=awsem_path,
            cg_bonds=use_cg_bonds,
            go_model=use_go_model,
        )

        logger.info("Processing PDB file...")
        converter.process_pdb()

        logger.info("Writing coordinate file...")
        converter.write_coord_file()

        logger.info("Writing LAMMPS files...")
        converter.write_lammps_files()

        logger.info("Conversion completed successfully")

        # Log output files
        logger.info(f"Generated files:")
        logger.info(f"- Coordinate file: {output_path.with_suffix('.coord')}")
        logger.info(f"- LAMMPS data file: {output_path.with_suffix('.data')}")
        logger.info(f"- LAMMPS input script: {output_path.with_suffix('.in')}")

    except Exception as e:
        logger.error(f"PDB to LAMMPS conversion failed: {str(e)}")
        raise

    logger.warning(
        "This is an experimental feature. Please validate the output files "
        "before using them in production."
    )
