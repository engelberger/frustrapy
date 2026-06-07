# FrustraPy output contract

Generated from `frustrapy/output/schema.py`, the single machine-readable source of truth. Do not edit by hand: change the schema and regenerate with `python -m frustrapy.output.schema > docs/OUTPUT_CONTRACT.md`.

The text tables under `{results_dir}/{base}.done/FrustrationData/` are the default output and the numerical reference. Any alternative store must round-trip value-identical to them.

## Tabular artifacts

### `{base}.pdb_{mode}` (14 columns, whitespace-separated)

Configurational/mutational per-contact frustration index and energies.

Row rule: one row per native contact i-j within the sequence-distance cutoff.

| # | Column | dtype |
|---|--------|-------|
| 1 | `Res1` | int |
| 2 | `Res2` | int |
| 3 | `ChainRes1` | str |
| 4 | `ChainRes2` | str |
| 5 | `DensityRes1` | float |
| 6 | `DensityRes2` | float |
| 7 | `AA1` | str |
| 8 | `AA2` | str |
| 9 | `NativeEnergy` | float |
| 10 | `DecoyEnergy` | float |
| 11 | `SDEnergy` | float |
| 12 | `FrstIndex` | float |
| 13 | `Welltype` | str |
| 14 | `FrstState` | str |

### `{base}.pdb_singleresidue` (8 columns, whitespace-separated)

Single-residue frustration index and energies (no frustration-state column).

Row rule: one row per residue in the structure.

| # | Column | dtype |
|---|--------|-------|
| 1 | `Res` | int |
| 2 | `ChainRes` | str |
| 3 | `DensityRes` | float |
| 4 | `AA` | str |
| 5 | `NativeEnergy` | float |
| 6 | `DecoyEnergy` | float |
| 7 | `SDEnergy` | float |
| 8 | `FrstIndex` | float |

### `{base}.pdb_{mode}_5adens` (9 columns, whitespace-separated)

5 A frustration density: per-residue contact counts and proportions by class.

Row rule: one row per residue (count and proportion of contacts within 5 A by class).

| # | Column | dtype |
|---|--------|-------|
| 1 | `Res` | int |
| 2 | `ChainRes` | str |
| 3 | `Total` | int |
| 4 | `HighlyFrst` | int |
| 5 | `NeutrallyFrst` | int |
| 6 | `MinimallyFrst` | int |
| 7 | `relHighlyFrustrated` | float |
| 8 | `relNeutralFrustrated` | float |
| 9 | `relMinimallyFrustrated` | float |

### `IC_Configurational_{reference}.csv` (23 columns, tab-separated)

FrustraEvo per-contact configurational information content.

Row rule: one row per contact shared across the aligned family (configurational mode).

| # | Column | dtype |
|---|--------|-------|
| 1 | `Res1` | int |
| 2 | `Res2` | int |
| 3 | `AA1` | str |
| 4 | `AA2` | str |
| 5 | `NumRes1_Ref` | int |
| 6 | `Chain1_Ref` | str |
| 7 | `NumRes2_Ref` | int |
| 8 | `Chain2_Ref` | str |
| 9 | `Prot_Ref` | str |
| 10 | `NoContacts` | int |
| 11 | `FreqConts` | float |
| 12 | `pNEU` | float |
| 13 | `pMIN` | float |
| 14 | `pMAX` | float |
| 15 | `HNEU` | float |
| 16 | `HMIN` | float |
| 17 | `HMAX` | float |
| 18 | `Htotal` | float |
| 19 | `ICNEU` | float |
| 20 | `ICMIN` | float |
| 21 | `ICMAX` | float |
| 22 | `ICtotal` | float |
| 23 | `FstConserved` | str |

### `IC_Mutational_{reference}.csv` (23 columns, tab-separated)

FrustraEvo per-contact mutational information content.

Row rule: one row per contact shared across the aligned family (mutational mode).

| # | Column | dtype |
|---|--------|-------|
| 1 | `Res1` | int |
| 2 | `Res2` | int |
| 3 | `AA1` | str |
| 4 | `AA2` | str |
| 5 | `NumRes1_Ref` | int |
| 6 | `Chain1_Ref` | str |
| 7 | `NumRes2_Ref` | int |
| 8 | `Chain2_Ref` | str |
| 9 | `Prot_Ref` | str |
| 10 | `NoContacts` | int |
| 11 | `FreqConts` | float |
| 12 | `pNEU` | float |
| 13 | `pMIN` | float |
| 14 | `pMAX` | float |
| 15 | `HNEU` | float |
| 16 | `HMIN` | float |
| 17 | `HMAX` | float |
| 18 | `Htotal` | float |
| 19 | `ICNEU` | float |
| 20 | `ICMIN` | float |
| 21 | `ICMAX` | float |
| 22 | `ICtotal` | float |
| 23 | `FstConserved` | str |

### `IC_SingleRes_{reference}.csv` (15 columns, tab-separated)

FrustraEvo per-residue single-residue frustration information content.

Row rule: one row per reference residue (1..N over reference-non-gap columns).

| # | Column | dtype |
|---|--------|-------|
| 1 | `Res` | int |
| 2 | `AA_Ref` | str |
| 3 | `Num_Ref` | int |
| 4 | `Prot_Ref` | str |
| 5 | `%Min` | float |
| 6 | `%Neu` | float |
| 7 | `%Max` | float |
| 8 | `CantMin` | int |
| 9 | `CantNeu` | int |
| 10 | `CantMax` | int |
| 11 | `ICMin` | float |
| 12 | `ICNeu` | float |
| 13 | `ICMax` | float |
| 14 | `ICTot` | float |
| 15 | `FrustIC` | str |

### `SeqIC_{reference}.tab` (2 columns, tab-separated)

FrustraEvo per-column sequence Shannon information content.

Row rule: one row per reference-non-gap alignment column (1..N).

| # | Column | dtype |
|---|--------|-------|
| 1 | `Position` | int |
| 2 | `Entropy` | float |

## Non-tabular artifacts

- `{base}.pdb_{mode}_density.pkl` (density_pkl)
- `tertiary_frustration.dat` (tertiary_frustration)

## Public return shapes

- `calculate_frustration`: 4-tuple (Pdb, dict_of_plots, Optional[FrustrationDensityResults], Optional[dict_single_residue_data]); slots 3-4 are None outside singleresidue mode
- `dir_frustration`: 2-tuple (dict_of_per_structure_results, Optional[dict_of_plots])
- `analyze_family`: dict with keys including 'contacts' (per-contact IC summary) and the paths to the written IC_*/SeqIC_* tables

