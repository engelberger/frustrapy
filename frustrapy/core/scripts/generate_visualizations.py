#!/usr/bin/env python3

import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, TextIO, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class VisualizationLine:
    """Class to store parsed visualization data from aux file"""

    resid1: str
    resid2: str
    chain1: str
    chain2: str
    interaction_type: str
    color: str


class VisualizationGenerator:
    """Generates visualization files in JML, TCL, and PyMOL formats"""

    def __init__(self, aux_file: Path, pdb_name: str, output_dir: Path, suffix: str):
        self.aux_file = aux_file
        self.pdb_name = pdb_name
        self.output_dir = output_dir
        self.suffix = suffix
        self.sanitized_pdb_name = self._sanitize_name(pdb_name)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize PDB name by replacing dots and hyphens with underscores"""
        return name.replace(".", "_").replace("-", "_")

    def _write_jml_header(self, file: TextIO) -> None:
        """Write header for JML file"""
        file.write(
            "select protein; cartoons;\n"
            "connect delete;\n"
            "spacefill off;\n"
            "color background white;\n"
        )

    def _write_pml_header(self, file: TextIO) -> None:
        """Write header for PyMOL file"""
        file.write(
            f"load {self.pdb_name}.pdb, {self.sanitized_pdb_name}\n"
            f"hide line,{self.sanitized_pdb_name}\n"
            "unset dynamic_measures\n"
            f"show cartoon,{self.sanitized_pdb_name}\n"
            f"color grey,{self.sanitized_pdb_name}\n"
            "run draw_links.py\n"
        )

    def _parse_aux_file(self) -> List[VisualizationLine]:
        """Parse the auxiliary file and return structured data"""
        try:
            with open(self.aux_file, "r") as f:
                lines = f.readlines()

            parsed_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 6:
                    parsed_lines.append(
                        VisualizationLine(
                            resid1=parts[0],
                            resid2=parts[1],
                            chain1=parts[2],
                            chain2=parts[3],
                            interaction_type=parts[4],
                            color=parts[5],
                        )
                    )
            return parsed_lines
        except Exception as e:
            logger.error(f"Error parsing aux file: {e}")
            sys.exit(1)

    def generate_files(self) -> None:
        """Generate all visualization files"""
        parsed_data = self._parse_aux_file()

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Open all files
        with (
            open(
                self.output_dir / f"{self.pdb_name}_{self.suffix}.jml", "w"
            ) as jml_file,
            open(
                self.output_dir / f"{self.pdb_name}_{self.suffix}.tcl", "w"
            ) as tcl_file,
            open(
                self.output_dir / f"{self.pdb_name}.pdb_{self.suffix}.pml", "w"
            ) as pml_file,
        ):

            # Write headers
            self._write_jml_header(jml_file)
            self._write_pml_header(pml_file)

            # Process each line
            for idx, line in enumerate(parsed_data):
                self._write_visualization_line(jml_file, tcl_file, pml_file, line, idx)

            # Write footers
            self._write_file_footers(pml_file, tcl_file)

    def _write_visualization_line(
        self,
        jml_file: TextIO,
        tcl_file: TextIO,
        pml_file: TextIO,
        line: VisualizationLine,
        idx: int,
    ) -> None:
        """Write a single visualization line to all output files"""
        # JML format
        jml_file.write(
            f"select {line.resid1}:{line.chain1}.CA, {line.resid2}:{line.chain2}.CA;\n"
            f"CONNECT single; CONNECT {line.color} "
        )

        # TCL format
        tcl_file.write(
            f'set sel{line.resid1} [atomselect top "resid {line.resid1} and name CA and chain {line.chain1}"]\n'
            f'set sel{line.resid2} [atomselect top "resid {line.resid2} and name CA and chain {line.chain2}"]\n'
            f"lassign [atomselect{idx*2} get {{x y z}}] pos1\n"
            f"lassign [atomselect{idx*2+1} get {{x y z}}] pos2\n"
            f"draw color {line.color}\n"
            "draw line $pos1 $pos2 style solid width 2\n"
        )

        # PyMOL format
        if line.interaction_type == "water-mediated":
            distance_type = "min_frst_wm" if line.color == "green" else "max_frst_wm"
            pml_file.write(
                f"distance {distance_type}_{self.sanitized_pdb_name}= "
                f"({self.sanitized_pdb_name}//{line.chain1}/{line.resid1}/CA),"
                f"({self.sanitized_pdb_name}//{line.chain2}/{line.resid2}/CA)\n"
            )
            jml_file.write("partial radius 0.1\n")
        else:
            pml_file.write(
                f"draw_links resi {line.resid1} and name CA and Chain {line.chain1} and {self.sanitized_pdb_name}, "
                f"resi {line.resid2} and name CA and Chain {line.chain2} and {self.sanitized_pdb_name}, "
                f"color={line.color}, color2={line.color}, radius=0.05, "
                f"object_name={line.resid1}:{line.resid2}_{line.color}_{self.sanitized_pdb_name}\n"
            )
            jml_file.write("single radius 0.1\n")

    def _write_file_footers(self, pml_file: TextIO, tcl_file: TextIO) -> None:
        """Write footer content for PyMOL and TCL files"""
        pml_file.write(
            f"zoom all\n"
            f"hide labels\n"
            f"color red, max_frst_wm_{self.sanitized_pdb_name}\n"
            f"color green, min_frst_wm_{self.sanitized_pdb_name}"
        )

        tcl_file.write(
            "\nmol modselect 0 top all\n"
            "mol modstyle 0 top newcartoon\n"
            "mol modcolor 0 top colorid 15\n"
        )


def main() -> None:
    """Main function to run the visualization generator"""
    if len(sys.argv) != 5:
        logger.error(
            "Usage: python generate_visualizations.py aux_file pdb_name output_dir suffix"
        )
        sys.exit(1)

    aux_file = Path(sys.argv[1])
    pdb_name = sys.argv[2]
    output_dir = Path(sys.argv[3])
    suffix = sys.argv[4]

    logger.info(f"Generating visualization files for {pdb_name}")

    generator = VisualizationGenerator(aux_file, pdb_name, output_dir, suffix)
    generator.generate_files()

    logger.info("Visualization files generated successfully")


if __name__ == "__main__":
    main()
