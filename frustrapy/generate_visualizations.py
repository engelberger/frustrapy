import os
from typing import Tuple
import sys
import os

def create_files_and_print_headers(jml_file: str, tcl_file: str, pml_file: str, pdb_base: str, mode: str) -> None:
    """
    Create the JML, TCL, and PML files and print the headers if needed.

    :param jml_file: Path to the JML file.
    :param tcl_file: Path to the TCL file.
    :param pml_file: Path to the PML file.
    :param pdb_base: Base name of the PDB file.
    :param mode: Mode of the visualization.
    """
    with open(jml_file, "w") as jml:
        jml.write("select protein; cartoons;\nconnect delete;\nspacefill off;\ncolor background white;\n")

    with open(tcl_file, "w") as tcl:
        pass

    name_pdb = pdb_base.replace('"', '_').replace('.', '_').replace('-', '_')

    with open(pml_file, "w") as pml:
        pml.write(f"load {pdb_base}.pdb, {name_pdb}\nhide line,{name_pdb}\nunset dynamic_measures\nshow cartoon,{name_pdb}\ncolor grey,{name_pdb}\nrun draw_links.py\n")

def process_aux_file(aux_file: str, jml_file: str, tcl_file: str, pml_file: str, name_pdb: str) -> None:
    """
    Read the auxiliary file and write the corresponding data to the JML, TCL, and PML files.

    :param aux_file: Path to the auxiliary file.
    :param jml_file: Path to the JML file.
    :param tcl_file: Path to the TCL file.
    :param pml_file: Path to the PML file.
    :param name_pdb: Name of the PDB file.
    """
    with open(aux_file, "r") as aux:
        aux_lines = aux.readlines()

    z = 0
    y = z + 1

    with open(jml_file, "a") as jml, open(tcl_file, "a") as tcl, open(pml_file, "a") as pml:
        for line in aux_lines:
            splitted = line.strip().split()
            jml.write(f"select {splitted[0]}:{splitted[2]}.CA, {splitted[1]}:{splitted[3]}.CA;\nCONNECT single; CONNECT {splitted[5]} ")

            tcl.write(f"set sel{splitted[0]} [atomselect top \"resid {splitted[0]} and name CA and chain {splitted[2]}\"]\nset sel{splitted[1]} [atomselect top \"resid {splitted[1]} and name CA and chain {splitted[3]}\"]\n# get the coordinates\nlassign [atomselect{z} get {{x y z}}] pos1\nlassign [atomselect{y} get {{x y z}}] pos2\n# draw a green line between the two atoms\ndraw color {splitted[5]}\ndraw line $pos1 $pos2 style solid width 2\n")

            if splitted[4] == "water-mediated":
                if splitted[5] == "green":
                    pml.write(f"distance min_frst_wm_{name_pdb}= ({name_pdb}//{splitted[2]}/{splitted[0]}/CA),({name_pdb}//{splitted[3]}/{splitted[1]}/CA)\n")
                else:
                    pml.write(f"distance max_frst_wm_{name_pdb}= ({name_pdb}//{splitted[2]}/{splitted[0]}/CA),({name_pdb}//{splitted[3]}/{splitted[1]}/CA)\n")
                jml.write("partial radius 0.1\n")
            else:
                pml.write(f"draw_links resi {splitted[0]} and name CA and Chain {splitted[2]} and {name_pdb}, resi {splitted[1]} and name CA and Chain {splitted[3]} and {name_pdb}, color={splitted[5]}, color2={splitted[5]}, radius=0.05, object_name={splitted[0]}:{splitted[1]}_{splitted[5]}_{name_pdb}\n")
                jml.write("single radius 0.1\n")

            z += 2
            y += 2

def print_tails(pml_file: str, tcl_file: str, name_pdb: str) -> None:
    """
    Print the tails in the PML and TCL files if needed.

    :param pml_file: Path to the PML file.
    :param tcl_file: Path to the TCL file.
    :param name_pdb: Name of the PDB file.
    """
    with open(pml_file, "a") as pml:
        pml.write(f"zoom all\nhide labels\ncolor red, max_frst_wm_{name_pdb}\ncolor green, min_frst_wm_{name_pdb}")

    with open(tcl_file, "a") as tcl:
        tcl.write("mol modselect 0 top all\nmol modstyle 0 top newcartoon\nmol modcolor 0 top colorid 15\n")

def generate_visualizations(aux_file: str, pdb_base: str, job_dir: str, mode: str) -> None:
    """
    Generate visualizations based on the provided parameters.

    :param aux_file: Path to the auxiliary file.
    :param pdb_base: Base name of the PDB file.
    :param job_dir: Directory where the job files are located.
    :param mode: Mode of the visualization.
    """
    jml_file = os.path.join(job_dir, f"{pdb_base}_{mode}.jml")
    tcl_file = os.path.join(job_dir, f"{pdb_base}_{mode}.tcl")
    pml_file = os.path.join(job_dir, f"{pdb_base}.pdb_{mode}.pml")

    name_pdb = pdb_base.replace('"', '_').replace('.', '_').replace('-', '_')

    create_files_and_print_headers(jml_file, tcl_file, pml_file, pdb_base, mode)
    process_aux_file(aux_file, jml_file, tcl_file, pml_file, name_pdb)
    print_tails(pml_file, tcl_file, name_pdb)
