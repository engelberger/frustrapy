import os
import pandas as pd
from typing import List, Dict, Any


class Pdb:
    def __init__(self, pdb_file: str, job_dir: str, mode: str, chain: str = None):
        self.pdb_file = pdb_file
        self.job_dir = job_dir
        self.mode = mode
        self.chain = chain
        self.pdb_base = self._get_pdb_base()
        self.atom = self._read_pdb()
        self.equivalences = self._get_equivalences()
        self.scripts_dir = self._get_scripts_dir()
        self.pdb_path = self._get_pdb_path()

    def _get_pdb_base(self) -> str:
        return os.path.splitext(os.path.basename(self.pdb_file))[0]

    def _read_pdb(self) -> "pd.DataFrame":
        atom_data = []
        with open(self.pdb_file, "r") as file:
            for line in file:
                if line.startswith("ATOM"):
                    atom_data.append(line.strip().split())
        atom_df = pd.DataFrame(
            atom_data,
            columns=[
                "type",
                "atom_num",
                "atom_name",
                "alt_loc",
                "res_name",
                "chain",
                "res_num",
                "icode",
                "x",
                "y",
                "z",
                "occupancy",
                "temp_factor",
                "element",
                "charge",
            ],
        )
        atom_df["xyz"] = list(zip(atom_df["x"], atom_df["y"], atom_df["z"]))
        return atom_df

    def _get_equivalences(self) -> "pd.DataFrame":
        atom_df = self.atom[self.atom["type"] == "ATOM"]
        equivalences = atom_df[["chain", "res_num", "res_name"]].drop_duplicates()
        equivalences["eq_index"] = range(1, len(equivalences) + 1)
        return equivalences

    def _get_scripts_dir(self) -> str:
        return os.path.join(os.path.dirname(__file__), "scripts")

    def _get_pdb_path(self) -> str:
        return os.path.join(self.job_dir, "FrustrationData", f"{self.pdb_base}.pdb")

    def write_pdb(self, output_file: str) -> None:
        with open(output_file, "w") as file:
            for _, row in self.atom.iterrows():
                file.write(
                    f"{row['type']:<6}{row['atom_num']:>5} {row['atom_name']:^4}{row['alt_loc']:1}"
                    f"{row['res_name']:>3} {row['chain']:1}{row['res_num']:>4}{row['icode']:1}   "
                    f"{row['x']:>8.3f}{row['y']:>8.3f}{row['z']:>8.3f}{row['occupancy']:>6.2f}"
                    f"{row['temp_factor']:>6.2f}          {row['element']:>2}{row['charge']:>2}\n"
                )

    def select_atoms(self, selection: Dict[str, Any]) -> "pd.DataFrame":
        selected_atoms = self.atom
        for key, value in selection.items():
            selected_atoms = selected_atoms[selected_atoms[key].isin(value)]
        return selected_atoms

    def trim(self, chain: str = None) -> "Pdb":
        if chain is None:
            chain = self.chain
        trimmed_pdb = Pdb(self.pdb_file, self.job_dir, self.mode, chain)
        trimmed_pdb.atom = self.atom[self.atom["chain"] == chain]
        trimmed_pdb.equivalences = self._get_equivalences()
        return trimmed_pdb
