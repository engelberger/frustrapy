# Prep-amortization golden baseline

Frozen pre-optimization FrustraPy outputs. The prep-amortization work computes the
per-structure data prep once and reuses it across a saturation scan's variants;
these fixtures are the oracle that proves it changes no numbers.

## Contents

- `{base}.pdb_{configurational,mutational,singleresidue}`: full FrustrationData
  tables (14 columns for the contact modes, 8 for singleresidue), default
  `lammps` backend, seq_dist 12.
- `{base}.pdb_{configurational,mutational}_5adens`: the 5 Angstrom density tables.
- `scan_{base}_Res{res}_{chain}.txt`: per-variant FrstIndex from a saturation
  scan (20 amino acids per target site).
- `manifest.json`: per-fixture structure, mode, row count, sha256, plus a
  native-vs-lammps parity summary.
- `generate_baseline.py`: regenerates everything in this directory.

Structures: `1crn` (crambin, single chain) and `1zni` (insulin, 4 chains with
cross-chain contacts; cleaned to ATOM records, altloc A or blank normalized to
blank, HETATM/water/ions dropped).

## Regenerate

From an activated venv with the native core built (`pip install ./native`):

    python tests/data/prep_baseline/generate_baseline.py

## Verify

`tests/test_prep_amortization_parity.py` re-runs the current code and asserts it
reproduces these fixtures (lammps token-for-token with zero numeric difference;
native within the CPU parity tolerance with Spearman 1.0 and 100 percent sign and
class agreement; the scans reproduce the per-variant FrstIndex and select the
correct (res_num, chain) per target).

See `docs/PREP_AMORTIZATION_PLAN.md` for the design.
