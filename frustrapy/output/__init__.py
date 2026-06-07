"""Output contract for FrustraPy artifacts.

This subpackage holds the single, machine-readable description of every file and
return value FrustraPy produces (``schema``), so that the code that writes the
tables, the tests that check them, and the documentation all reference one source
of truth instead of repeating column lists.

The text tables under ``{results_dir}/{base}.done/FrustrationData/`` remain the
default output and the numerical reference. Any alternative store (e.g. the HDF5
store added for high-throughput runs) must round-trip value-identical to them.
"""

from .schema import (
    ColumnSpec,
    TableSchema,
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
    read_table,
    validate_table,
)
from .store import (
    FORMAT_VERSION,
    GZIP_LEVEL,
    FrustrationStore,
    FrustrationCorpus,
    TrainingDataset,
    build_training_dataset,
    read_corpus_from,
    compound_dtype_for_schema,
    dataframe_to_records,
    records_to_dataframe,
    structure_group_path,
    dataset_path,
    family_group_path,
    require_h5py,
)

__all__ = [
    "ColumnSpec",
    "TableSchema",
    "CONTACT_TABLE",
    "SINGLERESIDUE_TABLE",
    "DENSITY_5ADENS_TABLE",
    "IC_CONFIGURATIONAL_TABLE",
    "IC_MUTATIONAL_TABLE",
    "IC_SINGLERES_TABLE",
    "SEQIC_TABLE",
    "TABLE_SCHEMAS",
    "RETURN_SHAPES",
    "schema_for_mode",
    "read_table",
    "validate_table",
    "FORMAT_VERSION",
    "GZIP_LEVEL",
    "FrustrationStore",
    "FrustrationCorpus",
    "TrainingDataset",
    "build_training_dataset",
    "read_corpus_from",
    "compound_dtype_for_schema",
    "dataframe_to_records",
    "records_to_dataframe",
    "structure_group_path",
    "dataset_path",
    "family_group_path",
    "require_h5py",
]
