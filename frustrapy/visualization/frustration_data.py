"""Parsed-table -> visualization records.

Single source of truth for turning the FrustrationData text tables into the
records the molecular-visualization generators (PyMOL ``.pml``, ChimeraX
``.cxc``) draw. Both generators consume the same records so the contact set,
the class->color mapping, and the per-residue coloring are identical across
surfaces.

Classification follows the audited cutoffs (``core/constants.py``):

* configurational / mutational *contacts*: ``FrstIndex <= -1`` highly (red),
  ``>= 0.78`` minimally (green), otherwise neutral (gray). The PyMOL/ChimeraX
  contact views draw only the red/green contacts -- exactly the rows the legacy
  ``RenumFiles.pl`` writes to the ``*_auxiliar`` file that
  ``GenerateVisualizations.pl`` consumed.
* single-residue *plot* coloring: ``FrstIndex <= -1`` highly (red),
  ``>= 0.58`` minimally (green), otherwise neutral (gray). The 0.58 single-
  residue cutoff is intentionally distinct from the 0.78 contact cutoff -- do
  not collapse them.
"""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from ..core.constants import (
    FRST_HIGHLY_MAX,
    FRST_MINIMALLY_MIN_CONTACT,
    FRST_MINIMALLY_MIN_SINGLERES,
)

# class -> display color, shared by every molecular-visualization surface.
CLASS_COLOR = {"highly": "red", "minimally": "green", "neutral": "gray"}


@dataclass(frozen=True)
class ContactLink:
    """A configurational/mutational contact drawn between two CA atoms."""

    res1: str
    res2: str
    chain1: str
    chain2: str
    welltype: str  # "short" | "long" | "water-mediated"
    color: str  # "red" (highly) | "green" (minimally)
    state: str  # "highly" | "minimally"


@dataclass(frozen=True)
class ResidueColor:
    """A single-residue site colored by its frustration class."""

    res: str
    chain: str
    aa: str
    frst_index: float
    state: str  # "highly" | "neutral" | "minimally"
    color: str  # red | gray | green


def classify_contact(frst_index: float) -> str:
    """Contact class using the configurational/mutational cutoffs (-1 / 0.78)."""
    if frst_index <= FRST_HIGHLY_MAX:
        return "highly"
    if frst_index >= FRST_MINIMALLY_MIN_CONTACT:
        return "minimally"
    return "neutral"


def classify_singleresidue(frst_index: float) -> str:
    """Single-residue plot class using the plot cutoff (-1 / 0.58)."""
    if frst_index <= FRST_HIGHLY_MAX:
        return "highly"
    if frst_index >= FRST_MINIMALLY_MIN_SINGLERES:
        return "minimally"
    return "neutral"


def contact_links_from_df(df: pd.DataFrame) -> List[ContactLink]:
    """Extract the red/green contact links from a parsed config/mutational table.

    Preserves table row order (== ``tertiary_frustration.dat`` order, == the
    legacy ``*_auxiliar`` order) and keeps only the highly/minimally contacts,
    matching ``RenumFiles.pl`` line 83.
    """
    links: List[ContactLink] = []
    for row in df.itertuples(index=False):
        state = str(row.FrstState)
        if state == "highly":
            color = "red"
        elif state == "minimally":
            color = "green"
        else:
            continue
        links.append(
            ContactLink(
                res1=str(row.Res1),
                res2=str(row.Res2),
                chain1=str(row.ChainRes1),
                chain2=str(row.ChainRes2),
                welltype=str(row.Welltype),
                color=color,
                state=state,
            )
        )
    return links


def residue_colors_from_df(df: pd.DataFrame) -> List[ResidueColor]:
    """Color every residue of a parsed single-residue table by its class."""
    out: List[ResidueColor] = []
    for row in df.itertuples(index=False):
        fi = float(row.FrstIndex)
        state = classify_singleresidue(fi)
        out.append(
            ResidueColor(
                res=str(row.Res),
                chain=str(row.ChainRes),
                aa=str(row.AA),
                frst_index=fi,
                state=state,
                color=CLASS_COLOR[state],
            )
        )
    return out


def read_table(table_path: str) -> pd.DataFrame:
    """Read a FrustrationData text table (whitespace separated)."""
    return pd.read_csv(table_path, sep=r"\s+")


def contact_links(table_path: str) -> List[ContactLink]:
    return contact_links_from_df(read_table(table_path))


def residue_colors(table_path: str) -> List[ResidueColor]:
    return residue_colors_from_df(read_table(table_path))


def class_counts_contacts(links: List[ContactLink]) -> dict:
    counts = {"highly": 0, "minimally": 0}
    for link in links:
        counts[link.state] += 1
    return counts


def class_counts_residues(residues: List[ResidueColor]) -> dict:
    counts = {"highly": 0, "neutral": 0, "minimally": 0}
    for res in residues:
        counts[res.state] += 1
    return counts
