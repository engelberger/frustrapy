# New file: utility functions for py3Dmol-based visualizations
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser


def load_config_contacts(frust_file: str, central_chain: str, central_res: int):
    """Load and filter configurational contacts for a specific residue."""
    df = pd.read_csv(frust_file, sep=r"\s+")
    contacts = []
    contact_residues = set()
    res_names = {}

    for _, row in df.iterrows():
        if row.ChainRes1 == central_chain and row.Res1 == central_res:
            contacts.append((row.ChainRes2, int(row.Res2), row.FrstState, abs(row.FrstIndex), row.AA2))
            contact_residues.add((row.ChainRes2, int(row.Res2)))
            res_names[(row.ChainRes2, int(row.Res2))] = row.AA2
        elif row.ChainRes2 == central_chain and row.Res2 == central_res:
            contacts.append((row.ChainRes1, int(row.Res1), row.FrstState, abs(row.FrstIndex), row.AA1))
            contact_residues.add((row.ChainRes1, int(row.Res1)))
            res_names[(row.ChainRes1, int(row.Res1))] = row.AA1

    # include central residue in the set for styling
    contact_residues.add((central_chain, central_res))
    return contacts, contact_residues, res_names


def build_ca_map(pdb_file: str):
    """Parse PDB and return a map of Cα coordinates."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_file)[0]
    ca_map = {}
    for chain in structure:
        for res in chain:
            if "CA" in res:
                coord = res["CA"].get_coord()
                ca_map[(chain.id, res.get_id()[1])] = tuple(map(float, coord))
    return ca_map


def create_summary_df(contacts):
    """Create a pandas DataFrame summary from a contacts list."""
    data = []
    for ch, resnum, state, magnitude, resname in contacts:
        data.append({
            "Chain": ch,
            "Residue": f"{resname}_{resnum}",
            "State": state,
            "Magnitude": magnitude,
        })
    df = pd.DataFrame(data)
    return df.sort_values("Magnitude", ascending=False)


def style_frustration(val: str) -> str:
    """Return a CSS style string for a given frustration state."""
    v = val.lower()
    if "highly" in v:
        return 'background-color: #ffcccc'
    if "minimally" in v:
        return 'background-color: #ccffcc'
    if "neutral" in v:
        return 'background-color: #f2f2f2'
    if "native" in v:
        return 'background-color: #cce5ff'
    return ''


def scale_radius(magnitude: float, min_magnitude: float, max_magnitude: float, min_rad: float = 0.1, max_rad: float = 0.4) -> float:
    """Scale a magnitude to a cylinder radius between min_rad and max_rad."""
    range_mag = max_magnitude - min_magnitude
    if range_mag == 0:
        return (min_rad + max_rad) / 2
    return min_rad + (magnitude - min_magnitude) / range_mag * (max_rad - min_rad) 