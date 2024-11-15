#!/usr/bin/env python3

import sys
import logging
from pathlib import Path
from typing import List, Dict, TextIO, Optional, Tuple
from dataclasses import dataclass
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Atom:
    """Class representing an atom in the system"""

    no: int
    chain_no: str
    residue_no: int
    atom_type: int
    charge: float
    x: float
    y: float
    z: float
    description: str = ""

    def write_coords(self, f: TextIO) -> None:
        """Write atom in coordinates format"""
        f.write(
            f"{self.no}\t{self.chain_no}\t{self.atom_type}\t"
            f"{self.x:15.8f}\t{self.y:15.8f}\t{self.z:15.8f}\t{self.description}\n"
        )

    def write_lammps(self, f: TextIO) -> None:
        """Write atom in LAMMPS data file format"""
        space11 = " " * 11
        f.write(f"{(space11+str(self.no))[-12:]}\t")
        f.write(
            f"{self.chain_no}\t{self.residue_no}\t{self.atom_type}\t"
            f"{self.charge}\t{self.x}\t{self.y}\t{self.z}\n"
        )


@dataclass
class Bond:
    """Class representing a bond between atoms"""

    no: int
    bond_type: int
    atom1: int
    atom2: int

    def write(self, f: TextIO) -> None:
        space11 = " " * 11
        f.write(f"{(space11+str(self.no))[-12:]}\t")
        f.write(f"{self.bond_type}\t{self.atom1}\t{self.atom2}\n")


class PDBToLAMMPS:
    """Converts PDB files to LAMMPS data format"""

    MASSES = {
        "standard": [12.0, 14.0, 16.0, 12.0, 1.0],
        "cg": [27.0, 14.0, 28.0, 60.0, 60.0],
        "go": [118.0],
    }

    def __init__(
        self,
        pdb_file: Path,
        output_prefix: Path,
        awsem_path: Path,
        cg_bonds: bool = False,
        go_model: bool = False,
    ):
        self.pdb_file = pdb_file
        self.output_prefix = output_prefix
        self.awsem_path = awsem_path
        self.cg_bonds = cg_bonds
        self.go_model = go_model

        self.atoms: List[Atom] = []
        self.bonds: List[Bond] = []
        self.n_residues = 0
        self.box_dimensions = {
            "xlo": -20000.0,
            "xhi": 20000.0,
            "ylo": -20000.0,
            "yhi": 20000.0,
            "zlo": -20000.0,
            "zhi": 20000.0,
        }

    def process_pdb(self) -> None:
        """Process PDB file and extract atomic coordinates"""
        try:
            from Bio.PDB import PDBParser, Structure

            parser = PDBParser(PERMISSIVE=1)
            structure = parser.get_structure("protein", self.pdb_file)

            atom_no = 0
            for model in structure:
                for chain in model:
                    chain_id = chain.get_id()
                    for residue in chain:
                        if residue.get_id()[0] == " ":  # Regular amino acid
                            self.n_residues += 1
                            self._process_residue(residue, chain_id, atom_no)
                            atom_no = len(self.atoms)

        except Exception as e:
            logger.error(f"Error processing PDB file: {e}")
            raise

    def _process_residue(self, residue, chain_id: str, start_atom_no: int) -> None:
        """Process a single residue and create corresponding atoms"""
        atom_map = {"N": 2, "CA": 1, "C": 1, "O": 3, "CB": 4}

        if self.go_model:
            if "CA" in residue:
                self._add_atom(residue["CA"], chain_id, 1)
            return

        for atom_name, atom_type in atom_map.items():
            if atom_name in residue:
                atom = residue[atom_name]
                self._add_atom(atom, chain_id, atom_type)

        # Create bonds if needed
        if self.cg_bonds:
            self._create_residue_bonds(start_atom_no)

    def _add_atom(self, atom, chain_id: str, atom_type: int) -> None:
        """Add a new atom to the system"""
        coord = atom.get_coord()
        self.atoms.append(
            Atom(
                no=len(self.atoms) + 1,
                chain_no=chain_id,
                residue_no=self.n_residues,
                atom_type=atom_type,
                charge=0.0,
                x=coord[0],
                y=coord[1],
                z=coord[2],
                description=atom.get_name(),
            )
        )

    def _create_residue_bonds(self, start_atom_no: int) -> None:
        """Create bonds for a residue"""
        if len(self.atoms) - start_atom_no >= 4:  # Has all necessary atoms
            n_bonds = len(self.bonds)
            # CA-CA bond
            self.bonds.append(
                Bond(n_bonds + 1, 1, start_atom_no + 1, start_atom_no + 2)
            )
            # CA-O bond
            self.bonds.append(
                Bond(n_bonds + 2, 2, start_atom_no + 2, start_atom_no + 3)
            )
            if len(self.atoms) - start_atom_no >= 5:  # Has CB
                # CA-CB bond
                self.bonds.append(
                    Bond(n_bonds + 3, 4, start_atom_no + 2, start_atom_no + 5)
                )

    def write_coord_file(self) -> None:
        """Write coordinate file"""
        coord_file = self.output_prefix.with_suffix(".coord")
        with open(coord_file, "w") as f:
            for atom in self.atoms:
                atom.write_coords(f)
        logger.info(f"Written coordinate file: {coord_file}")

    def write_lammps_files(self) -> None:
        """Write LAMMPS data file and input script"""
        self._write_data_file()
        self._write_input_script()

    def _write_data_file(self) -> None:
        """Write LAMMPS data file"""
        data_file = self.output_prefix.with_suffix(".data")

        masses = (
            self.MASSES["go"]
            if self.go_model
            else (self.MASSES["cg"] if self.cg_bonds else self.MASSES["standard"])
        )

        with open(data_file, "w") as f:
            f.write("LAMMPS protein data file\n\n")

            # System dimensions
            f.write(f"{len(self.atoms):12d}  atoms\n")
            f.write(f"{len(self.bonds):12d}  bonds\n")
            f.write("           0  angles\n")
            f.write("           0  dihedrals\n")
            f.write("           0  impropers\n\n")

            # Types
            n_atom_types = 1 if self.go_model else 5
            n_bond_types = 5 if self.cg_bonds else 0
            f.write(f"{n_atom_types:12d}  atom types\n")
            f.write(f"{n_bond_types:12d}  bond types\n\n")

            # Box dimensions
            for dim in ["x", "y", "z"]:
                f.write(
                    f"{self.box_dimensions[dim+'lo']:8.1f} {self.box_dimensions[dim+'hi']:8.1f} {dim}lo {dim}hi\n"
                )

            # Masses
            f.write("\nMasses\n\n")
            for i, mass in enumerate(masses, 1):
                f.write(f"{i:12d}  {mass}\n")

            # Atoms
            f.write("\nAtoms\n\n")
            for atom in self.atoms:
                atom.write_lammps(f)

            # Bonds
            if self.bonds:
                f.write("\nBonds\n\n")
                for bond in self.bonds:
                    bond.write(f)

        logger.info(f"Written LAMMPS data file: {data_file}")

    def _write_input_script(self) -> None:
        """Write LAMMPS input script"""
        # Read template
        template_path = self.awsem_path / "AWSEMFiles/AWSEMTools/inFilePattern.data"
        try:
            with open(template_path) as f:
                template = f.read()
        except FileNotFoundError:
            logger.error(f"Template file not found: {template_path}")
            return

        # Prepare replacements
        replacements = {
            "``read_data_file": f"read_data {self.output_prefix.with_suffix('.data')}",
            "``groups": self._generate_groups_string(),
            "``bonds": "bond_style harmonic" if self.cg_bonds else "",
            "``pair_interactions": self._generate_pair_interactions(),
            "``pair_coeff": self._generate_pair_coefficients(),
        }

        # Apply replacements
        for key, value in replacements.items():
            template = template.replace(key, value)

        # Write output
        input_file = self.output_prefix.with_suffix(".in")
        with open(input_file, "w") as f:
            f.write(template)

        logger.info(f"Written LAMMPS input script: {input_file}")

    def _generate_groups_string(self) -> str:
        """Generate groups definition string"""
        groups = []
        if self.go_model:
            groups.append(
                ["alpha_carbons", "id"] + [str(i + 1) for i in range(len(self.atoms))]
            )
        else:
            # Add specific atom type groups
            for group_name, atom_type in [
                ("alpha_carbons", 1),
                ("beta_atoms", 4),
                ("oxygens", 3),
            ]:
                atoms_in_group = [
                    str(a.no) for a in self.atoms if a.atom_type == atom_type
                ]
                if atoms_in_group:
                    groups.append([group_name, "id"] + atoms_in_group)

        return "\n".join(f"group {' '.join(group)}" for group in groups)

    def _generate_pair_interactions(self) -> str:
        """Generate pair interactions string"""
        if self.cg_bonds and not self.go_model:
            return "pair_style vexcluded 2 3.5 3.5"
        return ""

    def _generate_pair_coefficients(self) -> str:
        """Generate pair coefficients string"""
        if self.cg_bonds and not self.go_model:
            return (
                "pair_coeff * * 0.0\n"
                "pair_coeff 1 1 20.0 3.5 4.5\n"
                "pair_coeff 1 4 20.0 3.5 4.5\n"
                "pair_coeff 4 4 20.0 3.5 4.5\n"
                "pair_coeff 3 3 20.0 3.5 3.5"
            )
        return ""


def main() -> None:
    """Main function to run the PDB to LAMMPS conversion"""
    if len(sys.argv) < 4:
        logger.error(
            "Usage: python pdb_to_lammps.py <pdb_file> <output_prefix> <awsem_path> [-b] [-go]"
        )
        sys.exit(1)

    pdb_file = Path(sys.argv[1])
    output_prefix = Path(sys.argv[2])
    awsem_path = Path(sys.argv[3])

    cg_bonds = "-b" in sys.argv
    go_model = "-go" in sys.argv

    logger.info(f"Processing PDB file: {pdb_file}")
    logger.info(f"Output prefix: {output_prefix}")
    logger.info(f"AWSEM path: {awsem_path}")
    logger.info(f"CG bonds: {cg_bonds}")
    logger.info(f"GO model: {go_model}")

    try:
        converter = PDBToLAMMPS(pdb_file, output_prefix, awsem_path, cg_bonds, go_model)
        converter.process_pdb()
        converter.write_coord_file()
        converter.write_lammps_files()
        logger.info("Conversion completed successfully")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
