"""Validation of the output contract against the single schema source of truth.

``frustrapy.output.schema`` is the one machine-readable description of every artifact
FrustraPy writes. These tests assert two things:

* the schema and the writers agree (fast, no engine): every column list the writers
  emit is exactly the schema's column list, so the two cannot drift;
* the produced files conform to the schema (slow, runs the engine): on the verified
  1CRN anchor across all three modes and on the 3-member FrustraEvo family, every
  table has the schema columns, parseable dtypes, and the anchored row counts.

The slow tests reuse the shared ``crn_*`` fixtures (conftest) and the ``globin_family``
fixture (test_frustraevo); both run the real LAMMPS/AWSEM toolchain, so run pytest from
an activated venv (see tests/conftest.py).
"""

import inspect
import os
import re

import pytest

from frustrapy.output import schema as S
from frustrapy.output import (
    CONTACT_TABLE,
    SINGLERESIDUE_TABLE,
    DENSITY_5ADENS_TABLE,
    IC_CONFIGURATIONAL_TABLE,
    IC_MUTATIONAL_TABLE,
    IC_SINGLERES_TABLE,
    SEQIC_TABLE,
    TABLE_SCHEMAS,
    RETURN_SHAPES,
    schema_for_mode,
    validate_table,
)


# --------------------------------------------------------------------------- #
# Schema self-consistency (fast)
# --------------------------------------------------------------------------- #


def test_column_counts_match_contract():
    """The documented column counts: 14 contact, 8 single-residue, 9 density, 23 IC
    contact, 15 IC single-residue, 2 SeqIC."""
    assert CONTACT_TABLE.ncols == 14
    assert SINGLERESIDUE_TABLE.ncols == 8
    assert DENSITY_5ADENS_TABLE.ncols == 9
    assert IC_CONFIGURATIONAL_TABLE.ncols == 23
    assert IC_MUTATIONAL_TABLE.ncols == 23
    assert IC_SINGLERES_TABLE.ncols == 15
    assert SEQIC_TABLE.ncols == 2


def test_no_duplicate_columns_in_any_schema():
    for s in TABLE_SCHEMAS.values():
        names = s.column_names
        assert len(names) == len(set(names)), f"duplicate column in {s.key}"


def test_registry_keyed_by_key():
    for key, s in TABLE_SCHEMAS.items():
        assert s.key == key


def test_schema_for_mode_maps_modes():
    assert schema_for_mode("configurational") is CONTACT_TABLE
    assert schema_for_mode("mutational") is CONTACT_TABLE
    assert schema_for_mode("singleresidue") is SINGLERESIDUE_TABLE
    with pytest.raises(ValueError):
        schema_for_mode("bogus")


def test_ic_contact_modes_share_columns():
    """IC_Configurational and IC_Mutational have identical column layouts."""
    assert IC_CONFIGURATIONAL_TABLE.column_names == IC_MUTATIONAL_TABLE.column_names


def test_return_shapes_documented():
    for fn in ("calculate_frustration", "dir_frustration", "analyze_family"):
        assert fn in RETURN_SHAPES and RETURN_SHAPES[fn]


# --------------------------------------------------------------------------- #
# Schema <-> writer coupling (fast): the writers emit exactly the schema columns
# --------------------------------------------------------------------------- #


def _joined_literals(module):
    """Source of ``module`` with Python adjacent-string-literal boundaries removed, so a
    header split across two quoted strings reads as one contiguous string."""
    src = inspect.getsource(module)
    # Collapse a closing quote + whitespace/newline + opening quote into nothing,
    # joining ``"a " "b"`` -> ``"a b"`` as the interpreter would.
    return re.sub(r'"\s*"', "", src)


def test_contact_and_singleresidue_headers_match_writer_source():
    """helpers.py writes the contact/single-residue headers; they must equal the schema
    column lists joined by a space."""
    from frustrapy.utils import helpers

    src = _joined_literals(helpers)
    assert " ".join(CONTACT_TABLE.column_names) in src
    assert " ".join(SINGLERESIDUE_TABLE.column_names) in src


def test_density_header_matches_writer_source():
    """The 5adens header in frustration_calculator.py equals the schema."""
    from frustrapy.analysis import frustration_calculator

    src = _joined_literals(frustration_calculator)
    assert " ".join(DENSITY_5ADENS_TABLE.column_names) in src


def test_ic_columns_match_writer_source():
    """The evolution IC writer references each schema column name as a literal, and the
    single-residue / SeqIC headers are tab-joined literals."""
    from frustrapy.evolution import information_content

    src = _joined_literals(information_content)
    for col in IC_CONFIGURATIONAL_TABLE.column_names:
        assert f'"{col}"' in src, f"IC contact column {col} missing from writer source"
    # In source the separators are the escape sequence ``\t`` (backslash-t), not a real tab.
    assert "\\t".join(IC_SINGLERES_TABLE.column_names) in src
    assert "\\t".join(SEQIC_TABLE.column_names) in src


# --------------------------------------------------------------------------- #
# Validator behaviour on synthetic data (fast)
# --------------------------------------------------------------------------- #


def test_validate_table_accepts_conforming_file(tmp_path):
    f = tmp_path / "1crn.pdb_singleresidue"
    f.write_text(
        "Res ChainRes DensityRes AA NativeEnergy DecoyEnergy SDEnergy FrstIndex\n"
        "1 A 0.0 T -1.5 -0.2 0.8 1.62\n"
        "2 A 0.0 T -1.1 -0.3 0.7 1.10\n"
    )
    assert validate_table(str(f), SINGLERESIDUE_TABLE, expected_rows=2) == []


def test_validate_table_flags_wrong_columns(tmp_path):
    f = tmp_path / "bad"
    f.write_text("Res Chain\n1 A\n")
    problems = validate_table(str(f), SINGLERESIDUE_TABLE)
    assert problems and "columns" in problems[0]


def test_validate_table_flags_bad_dtype(tmp_path):
    f = tmp_path / "1crn.pdb_singleresidue"
    f.write_text(
        "Res ChainRes DensityRes AA NativeEnergy DecoyEnergy SDEnergy FrstIndex\n"
        "notanint A 0.0 T -1.5 -0.2 0.8 1.62\n"
    )
    problems = validate_table(str(f), SINGLERESIDUE_TABLE)
    assert any("Res" in p and "int" in p for p in problems)


def test_validate_table_flags_wrong_row_count(tmp_path):
    f = tmp_path / "1crn.pdb_singleresidue"
    f.write_text(
        "Res ChainRes DensityRes AA NativeEnergy DecoyEnergy SDEnergy FrstIndex\n"
        "1 A 0.0 T -1.5 -0.2 0.8 1.62\n"
    )
    problems = validate_table(str(f), SINGLERESIDUE_TABLE, expected_rows=46)
    assert any("rows" in p for p in problems)


# --------------------------------------------------------------------------- #
# Produced files conform to the schema (slow / e2e)
# --------------------------------------------------------------------------- #


def _frustration_data_dir(job_dir):
    return os.path.join(job_dir, "FrustrationData")


def test_configurational_files_conform(crn_configurational):
    """1CRN configurational: the contact table (232 rows) and its 5adens table
    conform to the schema."""
    fd = _frustration_data_dir(crn_configurational["job_dir"])
    contact = os.path.join(fd, "1crn.pdb_configurational")
    assert validate_table(contact, CONTACT_TABLE, expected_rows=232) == []

    dens = os.path.join(fd, "1crn.pdb_configurational_5adens")
    assert validate_table(dens, DENSITY_5ADENS_TABLE, expected_rows=46) == []


def test_mutational_table_conforms(crn_mutational):
    fd = _frustration_data_dir(crn_mutational["job_dir"])
    contact = os.path.join(fd, "1crn.pdb_mutational")
    assert validate_table(contact, CONTACT_TABLE) == []


def test_singleresidue_table_conforms(crn_singleresidue):
    """1CRN single-residue: 46 rows, 8 columns, schema-conforming."""
    fd = _frustration_data_dir(crn_singleresidue["job_dir"])
    table = os.path.join(fd, "1crn.pdb_singleresidue")
    assert validate_table(table, SINGLERESIDUE_TABLE, expected_rows=46) == []


def test_evolution_ic_files_conform(globin_family):
    """FrustraEvo on the 3-member family: IC_Configurational, IC_SingleRes, and SeqIC
    conform to their schemas."""
    rd = globin_family["results_dir"]
    ref = "1fsx-A"

    ic_conf = os.path.join(rd, IC_CONFIGURATIONAL_TABLE.filename(reference=ref))
    assert validate_table(ic_conf, IC_CONFIGURATIONAL_TABLE) == []

    ic_sr = os.path.join(rd, IC_SINGLERES_TABLE.filename(reference=ref))
    assert validate_table(ic_sr, IC_SINGLERES_TABLE) == []

    seqic = os.path.join(rd, SEQIC_TABLE.filename(reference=ref))
    assert validate_table(seqic, SEQIC_TABLE) == []
