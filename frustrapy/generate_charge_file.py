import os
from typing import List, Tuple

def parse_gro_file(gro_file: str) -> List[Tuple[int, str, str]]:
    """
    Parse the GRO file and extract residue and atom information.

    :param gro_file: Path to the GRO file.
    :return: List of tuples containing residue ID, residue name, and atom name.
    """
    with open(gro_file, "r") as file:
        lines = file.readlines()[2:]  # Skip the first two lines

    residue_data = []
    res_count = 0

    for line in lines:
        if not line.strip():
            break

        split_line = line.split()
        residue_id = int(split_line[0])
        residue = split_line[1]
        atom = split_line[2]

        if atom == "CA":
            res_count += 1
        elif (residue == "ARG" and atom == "CB") or (residue == "LYS" and atom == "CB") or \
             (residue == "ASP" and atom == "CB") or (residue == "GLU" and atom == "CB"):
            residue_data.append((res_count, residue, atom))

    return residue_data

def assign_charges(residue_data: List[Tuple[int, str, str]]) -> List[Tuple[int, float]]:
    """
    Assign charges to residues based on their names.

    :param residue_data: List of tuples containing residue ID, residue name, and atom name.
    :return: List of tuples containing residue ID and assigned charge.
    """
    charge_data = []

    for res_id, residue, _ in residue_data:
        if residue in ["ARG", "LYS"]:
            charge = 1.0
        elif residue in ["ASP", "GLU"]:
            charge = -1.0
        else:
            continue

        charge_data.append((res_id, charge))

    return charge_data

def write_charge_file(charge_data: List[Tuple[int, float]], output_file: str) -> None:
    """
    Write the charge data to an output file.

    :param charge_data: List of tuples containing residue ID and assigned charge.
    :param output_file: Path to the output file.
    """
    with open(output_file, "w") as file:
        file.write(f"{len(charge_data)}\n")
        for res_id, charge in charge_data:
            file.write(f"{res_id:6d}   {charge:8.4f}\n")

def generate_charge_file(gro_file: str, output_file: str) -> None:
    """
    Generate the charge file based on the provided GRO file.

    :param gro_file: Path to the GRO file.
    :param output_file: Path to the output charge file.
    """
    residue_data = parse_gro_file(gro_file)
    charge_data = assign_charges(residue_data)
    write_charge_file(charge_data, output_file)
