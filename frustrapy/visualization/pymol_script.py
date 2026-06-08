"""PyMOL ``.pml`` script generator (pure Python).

Reimplements the PyMOL half of ``core/scripts/GenerateVisualizations.pl`` without
shelling out to Perl. Driven by the parsed FrustrationData tables.

Configurational / mutational: one ``draw_links`` cylinder per direct red/green
contact between the two CA atoms, plus a ``distance`` object per water-mediated
red/green contact -- the exact geometry and command set the Perl emitted. The
output is byte-identical to ``GenerateVisualizations.pl`` (verified on 1crn
configurational + mutational; see ``tests/test_visualization_parity.py``).

Single-residue: the Perl never produced a PyMOL script for this mode. This
module adds one -- per-residue cartoon coloring by single-residue class (red /
gray / green at the -1 / 0.58 cutoffs) so the three modes are consistent across
PyMOL, ChimeraX and the Plotly plots.
"""

import os
from typing import List, Optional

import pandas as pd

from .frustration_data import (
    CLASS_COLOR,
    ContactLink,
    ResidueColor,
    contact_links_from_df,
    read_table,
    residue_colors_from_df,
)

# PyMOL's grey for neutral cartoon/background (British spelling, as in the Perl).
_PYMOL_CLASS_COLOR = {"highly": "red", "minimally": "green", "neutral": "grey"}


def sanitize_object_name(pdb_base: str) -> str:
    """Replicate the Perl ``tr/".\\-"/"__"/`` object-name sanitization.

    Replaces ``.`` and ``-`` with ``_`` (PyMOL object names cannot contain them);
    leaves everything else unchanged.
    """
    return pdb_base.replace(".", "_").replace("-", "_")


def _pml_header(pdb_base: str, name: str) -> str:
    return (
        f"load {pdb_base}.pdb, {name}\n"
        f"hide line,{name}\n"
        "unset dynamic_measures\n"
        f"show cartoon,{name}\n"
        f"color grey,{name}\n"
        "run draw_links.py\n"
    )


def _contact_line(link: ContactLink, name: str) -> str:
    if link.welltype == "water-mediated":
        prefix = "min" if link.color == "green" else "max"
        return (
            f"distance {prefix}_frst_wm_{name}= "
            f"({name}//{link.chain1}/{link.res1}/CA),"
            f"({name}//{link.chain2}/{link.res2}/CA)\n"
        )
    return (
        f"draw_links resi {link.res1} and name CA and Chain {link.chain1} and {name}, "
        f"resi {link.res2} and name CA and Chain {link.chain2} and {name}, "
        f"color={link.color}, color2={link.color}, radius=0.05, "
        f"object_name={link.res1}:{link.res2}_{link.color}_{name}\n"
    )


def generate_contact_pml(links: List[ContactLink], pdb_base: str) -> str:
    """Build the configurational/mutational ``.pml`` content from contact links.

    Byte-identical to ``GenerateVisualizations.pl``: header, one line per
    red/green contact (in table order), then the tail (no trailing newline).
    """
    name = sanitize_object_name(pdb_base)
    parts = [_pml_header(pdb_base, name)]
    for link in links:
        parts.append(_contact_line(link, name))
    parts.append(
        "zoom all\n"
        "hide labels\n"
        f"color red, max_frst_wm_{name}\n"
        f"color green, min_frst_wm_{name}"
    )
    return "".join(parts)


def generate_singleresidue_pml(residues: List[ResidueColor], pdb_base: str) -> str:
    """Build a per-residue-coloring ``.pml`` for single-residue mode (new).

    Colors each residue's cartoon by its single-residue frustration class
    (red / gray / green at -1 / 0.58). No contact cylinders are drawn.
    """
    name = sanitize_object_name(pdb_base)
    parts = [
        f"load {pdb_base}.pdb, {name}\n"
        f"hide line,{name}\n"
        f"show cartoon,{name}\n"
        f"color grey,{name}\n"
    ]
    for res in residues:
        col = _PYMOL_CLASS_COLOR[res.state]
        parts.append(
            f"color {col}, {name} and chain {res.chain} and resi {res.res}\n"
        )
    parts.append("zoom all\nhide labels")
    return "".join(parts)


def write_pml(
    table_path: str,
    out_path: str,
    pdb_base: str,
    mode: str,
) -> str:
    """Generate a ``.pml`` for ``mode`` from a parsed FrustrationData table.

    Returns the path written.
    """
    df = read_table(table_path)
    if mode == "singleresidue":
        content = generate_singleresidue_pml(residue_colors_from_df(df), pdb_base)
    else:
        content = generate_contact_pml(contact_links_from_df(df), pdb_base)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(content)
    return out_path
