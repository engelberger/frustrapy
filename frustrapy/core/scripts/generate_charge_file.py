#!/usr/bin/env python3

import sys
import logging
from pathlib import Path
from typing import List, Tuple, TextIO
from dataclasses import dataclass

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ResidueCharge:
    """Class to store residue charge information"""

    residue_id: int
    charge: float


class ChargeFileGenerator:
    """Generates charge files for protein residues based on GRO file input"""

    CHARGED_RESIDUES = {"ARG": 1.0, "LYS": 1.0, "ASP": -1.0, "GLU": -1.0}

    def __init__(self, input_file: Path):
        self.input_file = input_file
        self.charges: List[ResidueCharge] = []
        self.residue_count = 0

    def process_gro_file(self) -> None:
        """Process the GRO file and extract charge information"""
        try:
            with open(self.input_file, "r") as f:
                # Skip first two header lines
                next(f)
                next(f)
                self._process_residues(f)
        except FileNotFoundError:
            logger.error(f"Input file not found: {self.input_file}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            sys.exit(1)

    def _process_residues(self, file_handle: TextIO) -> None:
        """Process residues from the GRO file"""
        for line in file_handle:
            line = line.strip()
            if not line:
                break

            try:
                residue_id = int(line[:5])
                residue_name = line[5:8].strip()
                atom_name = line[12:15].strip()

                if atom_name == "CA":
                    self.residue_count += 1
                elif atom_name == "CB" and residue_name in self.CHARGED_RESIDUES:
                    charge = self.CHARGED_RESIDUES[residue_name]
                    self.charges.append(
                        ResidueCharge(residue_id=self.residue_count, charge=charge)
                    )
                    logger.debug(
                        f"Added charge {charge} for residue {residue_name} "
                        f"at position {self.residue_count}"
                    )

            except ValueError as e:
                logger.warning(f"Skipping malformed line: {line}. Error: {e}")
                continue

    def write_output(self) -> None:
        """Write the charge information to stdout"""
        print(len(self.charges))
        for charge_info in self.charges:
            print(f"{charge_info.residue_id:6d}   {charge_info.charge:8.4f}")


def main() -> None:
    """Main function to run the charge file generator"""
    if len(sys.argv) < 2:
        logger.error("Usage: python generate_charge_file.py protein.gro")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    logger.info(f"Processing file: {input_path}")

    generator = ChargeFileGenerator(input_path)
    generator.process_gro_file()
    generator.write_output()


if __name__ == "__main__":
    main()
