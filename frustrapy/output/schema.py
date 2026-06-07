"""Single source of truth for the FrustraPy output contract.

Every artifact FrustraPy writes is described here as a :class:`TableSchema`:
the filename template, the field separator, whether a header row is present, and
the ordered list of columns with their logical dtype and a one-line row rule. The
helpers at the bottom (:func:`read_table`, :func:`validate_table`) let tests and
downstream code assert that a produced file matches its schema without repeating
the column lists.

The column lists here are kept verbatim with the writers:

* contact / single-residue tables: ``frustrapy/utils/helpers.py``
  (headers at ``helpers.py:230`` and ``helpers.py:287``);
* ``*_5adens`` density table: ``frustrapy/analysis/frustration_calculator.py:1086``;
* FrustraEvo ``IC_*`` / ``SeqIC_*`` tables:
  ``frustrapy/evolution/information_content.py`` (column orders at ``:711``,
  ``:881``, ``:997``).

Logical dtypes are intentionally coarse — ``int``, ``float``, ``str`` — because the
text tables store everything as strings; validation checks that each column *parses*
as its declared dtype, not that pandas inferred it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

# Field separators used on disk.
WHITESPACE = r"\s+"  # space-delimited tables (read with sep=r"\s+")
TAB = "\t"  # tab-delimited evolution tables


@dataclass(frozen=True)
class ColumnSpec:
    """One column of a table: its name and its logical dtype.

    ``dtype`` is one of ``"int"``, ``"float"``, ``"str"``. The value is what the
    column is *parseable* as, not necessarily how pandas infers it from a single
    file (e.g. an all-integer float column).
    """

    name: str
    dtype: str

    def __post_init__(self) -> None:
        if self.dtype not in ("int", "float", "str"):
            raise ValueError(f"unknown dtype {self.dtype!r} for column {self.name!r}")


@dataclass(frozen=True)
class TableSchema:
    """Structural contract for one tabular artifact."""

    key: str
    filename_template: str
    separator: str
    has_header: bool
    columns: Tuple[ColumnSpec, ...]
    row_rule: str
    description: str

    @property
    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    @property
    def ncols(self) -> int:
        return len(self.columns)

    def filename(self, **kwargs) -> str:
        """Render the on-disk filename, e.g. ``filename(base="1crn", mode="configurational")``."""
        return self.filename_template.format(**kwargs)


# --------------------------------------------------------------------------- #
# Per-structure frustration tables (the parity reference)
# --------------------------------------------------------------------------- #

#: ``{base}.pdb_configurational`` / ``{base}.pdb_mutational`` -- 14 columns, one row
#: per scored native contact. Configurational and mutational are parsed identically
#: and share this schema.
CONTACT_TABLE = TableSchema(
    key="contact",
    filename_template="{base}.pdb_{mode}",
    separator=WHITESPACE,
    has_header=True,
    columns=(
        ColumnSpec("Res1", "int"),
        ColumnSpec("Res2", "int"),
        ColumnSpec("ChainRes1", "str"),
        ColumnSpec("ChainRes2", "str"),
        ColumnSpec("DensityRes1", "float"),
        ColumnSpec("DensityRes2", "float"),
        ColumnSpec("AA1", "str"),
        ColumnSpec("AA2", "str"),
        ColumnSpec("NativeEnergy", "float"),
        ColumnSpec("DecoyEnergy", "float"),
        ColumnSpec("SDEnergy", "float"),
        ColumnSpec("FrstIndex", "float"),
        ColumnSpec("Welltype", "str"),
        ColumnSpec("FrstState", "str"),
    ),
    row_rule="one row per native contact i-j within the sequence-distance cutoff",
    description="Configurational/mutational per-contact frustration index and energies.",
)

#: ``{base}.pdb_singleresidue`` -- 8 columns (no FrstState), one row per residue.
SINGLERESIDUE_TABLE = TableSchema(
    key="singleresidue",
    filename_template="{base}.pdb_singleresidue",
    separator=WHITESPACE,
    has_header=True,
    columns=(
        ColumnSpec("Res", "int"),
        ColumnSpec("ChainRes", "str"),
        ColumnSpec("DensityRes", "float"),
        ColumnSpec("AA", "str"),
        ColumnSpec("NativeEnergy", "float"),
        ColumnSpec("DecoyEnergy", "float"),
        ColumnSpec("SDEnergy", "float"),
        ColumnSpec("FrstIndex", "float"),
    ),
    row_rule="one row per residue in the structure",
    description="Single-residue frustration index and energies (no frustration-state column).",
)

#: ``{base}.pdb_{mode}_5adens`` -- 9 columns, one row per residue, the 5 A spatial
#: frustration-density / proportion table.
DENSITY_5ADENS_TABLE = TableSchema(
    key="density_5adens",
    filename_template="{base}.pdb_{mode}_5adens",
    separator=WHITESPACE,
    has_header=True,
    columns=(
        ColumnSpec("Res", "int"),
        ColumnSpec("ChainRes", "str"),
        ColumnSpec("Total", "int"),
        ColumnSpec("HighlyFrst", "int"),
        ColumnSpec("NeutrallyFrst", "int"),
        ColumnSpec("MinimallyFrst", "int"),
        ColumnSpec("relHighlyFrustrated", "float"),
        ColumnSpec("relNeutralFrustrated", "float"),
        ColumnSpec("relMinimallyFrustrated", "float"),
    ),
    row_rule="one row per residue (count and proportion of contacts within 5 A by class)",
    description="5 A frustration density: per-residue contact counts and proportions by class.",
)


# --------------------------------------------------------------------------- #
# FrustraEvo (evolution) information-content tables
# --------------------------------------------------------------------------- #

_IC_CONTACT_COLUMNS = (
    ColumnSpec("Res1", "int"),
    ColumnSpec("Res2", "int"),
    ColumnSpec("AA1", "str"),
    ColumnSpec("AA2", "str"),
    ColumnSpec("NumRes1_Ref", "int"),
    ColumnSpec("Chain1_Ref", "str"),
    ColumnSpec("NumRes2_Ref", "int"),
    ColumnSpec("Chain2_Ref", "str"),
    ColumnSpec("Prot_Ref", "str"),
    ColumnSpec("NoContacts", "int"),
    ColumnSpec("FreqConts", "float"),
    ColumnSpec("pNEU", "float"),
    ColumnSpec("pMIN", "float"),
    ColumnSpec("pMAX", "float"),
    ColumnSpec("HNEU", "float"),
    ColumnSpec("HMIN", "float"),
    ColumnSpec("HMAX", "float"),
    ColumnSpec("Htotal", "float"),
    ColumnSpec("ICNEU", "float"),
    ColumnSpec("ICMIN", "float"),
    ColumnSpec("ICMAX", "float"),
    ColumnSpec("ICtotal", "float"),
    ColumnSpec("FstConserved", "str"),
)

#: ``IC_Configurational_{reference}.csv`` -- 23 columns, tab-separated, one row per
#: shared contact across the family.
IC_CONFIGURATIONAL_TABLE = TableSchema(
    key="ic_configurational",
    filename_template="IC_Configurational_{reference}.csv",
    separator=TAB,
    has_header=True,
    columns=_IC_CONTACT_COLUMNS,
    row_rule="one row per contact shared across the aligned family (configurational mode)",
    description="FrustraEvo per-contact configurational information content.",
)

#: ``IC_Mutational_{reference}.csv`` -- identical 23-column layout, mutational mode.
IC_MUTATIONAL_TABLE = TableSchema(
    key="ic_mutational",
    filename_template="IC_Mutational_{reference}.csv",
    separator=TAB,
    has_header=True,
    columns=_IC_CONTACT_COLUMNS,
    row_rule="one row per contact shared across the aligned family (mutational mode)",
    description="FrustraEvo per-contact mutational information content.",
)

#: ``IC_SingleRes_{reference}.csv`` -- 15 columns, tab-separated, one row per
#: reference residue.
IC_SINGLERES_TABLE = TableSchema(
    key="ic_singleres",
    filename_template="IC_SingleRes_{reference}.csv",
    separator=TAB,
    has_header=True,
    columns=(
        ColumnSpec("Res", "int"),
        ColumnSpec("AA_Ref", "str"),
        ColumnSpec("Num_Ref", "int"),
        ColumnSpec("Prot_Ref", "str"),
        ColumnSpec("%Min", "float"),
        ColumnSpec("%Neu", "float"),
        ColumnSpec("%Max", "float"),
        ColumnSpec("CantMin", "int"),
        ColumnSpec("CantNeu", "int"),
        ColumnSpec("CantMax", "int"),
        ColumnSpec("ICMin", "float"),
        ColumnSpec("ICNeu", "float"),
        ColumnSpec("ICMax", "float"),
        ColumnSpec("ICTot", "float"),
        # Despite the name, the last column holds the conserved single-residue state
        # label (MIN/NEU/MAX), not a number -- kept verbatim from FrustraEvo.
        ColumnSpec("FrustIC", "str"),
    ),
    row_rule="one row per reference residue (1..N over reference-non-gap columns)",
    description="FrustraEvo per-residue single-residue frustration information content.",
)

#: ``SeqIC_{reference}.tab`` -- 2 columns, per-alignment-column sequence Shannon
#: information content.
SEQIC_TABLE = TableSchema(
    key="seqic",
    filename_template="SeqIC_{reference}.tab",
    separator=TAB,
    has_header=True,
    columns=(
        ColumnSpec("Position", "int"),
        ColumnSpec("Entropy", "float"),
    ),
    row_rule="one row per reference-non-gap alignment column (1..N)",
    description="FrustraEvo per-column sequence Shannon information content.",
)


#: Registry of every tabular schema, keyed by ``schema.key``.
TABLE_SCHEMAS = {
    s.key: s
    for s in (
        CONTACT_TABLE,
        SINGLERESIDUE_TABLE,
        DENSITY_5ADENS_TABLE,
        IC_CONFIGURATIONAL_TABLE,
        IC_MUTATIONAL_TABLE,
        IC_SINGLERES_TABLE,
        SEQIC_TABLE,
    )
}


#: Public return shapes (documented, not enforced here).
RETURN_SHAPES = {
    "calculate_frustration": (
        "4-tuple (Pdb, dict_of_plots, Optional[FrustrationDensityResults], "
        "Optional[dict_single_residue_data]); slots 3-4 are None outside singleresidue mode"
    ),
    "dir_frustration": "2-tuple (dict_of_per_structure_results, Optional[dict_of_plots])",
    "analyze_family": (
        "dict with keys including 'contacts' (per-contact IC summary) and the paths "
        "to the written IC_*/SeqIC_* tables"
    ),
}


# Non-tabular artifacts, documented for completeness (not validated structurally):
#  * ``{base}.pdb_{mode}_density.pkl`` -- pickle of FrustrationDensityResults
#    (densities list + contact_coordinates + frustration_values). Read the text
#    ``*_5adens`` table instead of unpickling untrusted files.
#  * ``tertiary_frustration.dat`` -- raw column output of the LAMMPS/AWSEM binary;
#    the source the contact/single-residue tables are parsed from.
NON_TABULAR_ARTIFACTS = {
    "density_pkl": "{base}.pdb_{mode}_density.pkl",
    "tertiary_frustration": "tertiary_frustration.dat",
}


def schema_for_mode(mode: str) -> TableSchema:
    """Return the per-contact / single-residue table schema for a calculation mode."""
    if mode in ("configurational", "mutational"):
        return CONTACT_TABLE
    if mode == "singleresidue":
        return SINGLERESIDUE_TABLE
    raise ValueError(f"unknown mode {mode!r}")


def read_table(path: str, schema: TableSchema) -> pd.DataFrame:
    """Read a produced table as strings, using the schema's separator/header.

    Columns are read as ``str`` (no dtype inference) so validation controls the
    parsing and round-trip comparisons stay byte-faithful to the text on disk.
    """
    header = 0 if schema.has_header else None
    return pd.read_csv(path, sep=schema.separator, header=header, dtype=str, engine="python")


def _column_parses_as(series: pd.Series, dtype: str) -> bool:
    """Whether every non-empty value in ``series`` parses as ``dtype``."""
    values = [v for v in series.tolist() if v is not None and str(v) != ""]
    if dtype == "str":
        return True
    try:
        if dtype == "float":
            for v in values:
                float(v)
            return True
        if dtype == "int":
            for v in values:
                # Accept integer-valued floats ("0", "12", and also "12.0").
                f = float(v)
                if f != int(f):
                    return False
            return True
    except (ValueError, TypeError):
        return False
    return False


def validate_table(
    path: str, schema: TableSchema, expected_rows: Optional[int] = None
) -> List[str]:
    """Validate a produced file against a schema; return a list of problems.

    An empty list means the file conforms: header matches the schema column names in
    order, every column parses as its declared dtype, and (if given) the row count
    equals ``expected_rows``.
    """
    problems: List[str] = []
    df = read_table(path, schema)

    if list(df.columns) != schema.column_names:
        problems.append(
            f"{schema.key}: columns {list(df.columns)} != expected {schema.column_names}"
        )
        # Column mismatch makes per-column dtype checks meaningless.
        return problems

    for col in schema.columns:
        if not _column_parses_as(df[col.name], col.dtype):
            problems.append(
                f"{schema.key}: column {col.name!r} has values that do not parse as {col.dtype}"
            )

    if expected_rows is not None and len(df) != expected_rows:
        problems.append(f"{schema.key}: {len(df)} rows != expected {expected_rows}")

    return problems


def render_markdown() -> str:
    """Render the output contract as Markdown, straight from the schema registry.

    ``python -m frustrapy.output.schema`` prints this; ``docs/OUTPUT_CONTRACT.md`` is the
    committed result, so the doc cannot drift from the schema.
    """
    out: List[str] = []
    out.append("# FrustraPy output contract")
    out.append("")
    out.append(
        "Generated from `frustrapy/output/schema.py`, the single machine-readable source "
        "of truth. Do not edit by hand: change the schema and regenerate with "
        "`python -m frustrapy.output.schema > docs/OUTPUT_CONTRACT.md`."
    )
    out.append("")
    out.append(
        "The text tables under `{results_dir}/{base}.done/FrustrationData/` are the default "
        "output and the numerical reference. Any alternative store must round-trip "
        "value-identical to them."
    )
    out.append("")
    out.append("## Tabular artifacts")
    out.append("")
    for s in TABLE_SCHEMAS.values():
        sep = "tab" if s.separator == TAB else "whitespace"
        out.append(f"### `{s.filename_template}` ({s.ncols} columns, {sep}-separated)")
        out.append("")
        out.append(s.description)
        out.append("")
        out.append(f"Row rule: {s.row_rule}.")
        out.append("")
        out.append("| # | Column | dtype |")
        out.append("|---|--------|-------|")
        for i, c in enumerate(s.columns, 1):
            out.append(f"| {i} | `{c.name}` | {c.dtype} |")
        out.append("")
    out.append("## Non-tabular artifacts")
    out.append("")
    for key, name in NON_TABULAR_ARTIFACTS.items():
        out.append(f"- `{name}` ({key})")
    out.append("")
    out.append("## Public return shapes")
    out.append("")
    for fn, shape in RETURN_SHAPES.items():
        out.append(f"- `{fn}`: {shape}")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    print(render_markdown())
