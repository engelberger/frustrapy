import os
import pandas as pd
from typing import Optional


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
        self.frustration_dir = os.path.join(job_dir, "FrustrationData")

        # Initialize Mutations dictionary
        self.Mutations = {}

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
