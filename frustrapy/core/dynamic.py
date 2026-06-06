from typing import Optional, List, Dict
import pandas as pd


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
        self.residues_dynamic = {}

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
