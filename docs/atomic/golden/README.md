# Atomic Frustratometer golden fixture (parity oracle for the post-processing half)

This directory holds a `tertiary_frustration.dat` regenerated, **without Rosetta**, from
the shipped reference logs in `/workspace/atomic_frustratometer_ref/example_output/`. It is
the parity oracle for the AA backend's post-processing half (the Z-score and the contact
table), exactly as the LAMMPS `tertiary_frustration.dat` is the oracle for the AWSEM path.

## Why this is regenerable here (and the engine half is not)

The expensive Rosetta FastRelax/repack is already baked into the shipped `.log` files
(native.log + 1.log..50.log; each holds the per-residue-pair `ResResE` energy table). The
reference post-processor `Frust_Post_public.py` is pure Python (numpy + Biopython); given the
logs and the input structure it is fully deterministic. So porting that script to Python 3 and
re-running it over the shipped logs reproduces the frustration output with no Rosetta and no
PyRosetta in the container.

## Exact command used

```
# scratch dir contains: 3gso.pdb (copy of native.pdb), native.log, 1.log .. 50.log
python3 Frust_Post_public_py3.py 100 -2.5 0.5 50 9 0 Function1 -1.5 0.5
```

Argument meaning (positional, matching the reference CLI):
`reslen=100` (unused for sizing; true length comes from the structure), `minvalue=-2.5`
(minimally-frustrated cutoff), `maxvalue=0.5` (highly-frustrated cutoff), `decoy_num=50`,
`sep=9` (sequence separation), `enable=0` (no neutral lines in the VMD/PyMOL scripts),
`scheme=Function1` (drop the `fa_rep` term only), `minvalue_l=-1.5`, `maxvalue_l=0.5`
(ligand cutoffs; no ligand here). This mirrors `job.sh` line 32 with `$2 = 50`.

`3gso.pdb` is a byte-for-byte copy of `example_output/native.pdb`, which `job.sh` (lines 11-12)
makes a copy of the input `1QYS.pdb`; verified identical (`diff -q` clean). The post-processor
hardcodes the structure name `3gso`.

## Decoy count: the truth (reconciling the mission note)

The mission brief said "README says 50 but only 20 logs are present". In **this** container
that is not what shipped: `example_output/` contains all of `1.log .. 50.log` and
`1.pdb .. 50.pdb` (51 logs counting `native.log`), `peptide` has exactly 50 sequence lines,
and `README.md` documents the demo as `./job.sh 1QYS.pdb 50` with "using 50 decoys only". So
the demo used **N=50**, all 50 decoy logs are present, and the fixture is generated with
`decoy_num=50`. The earlier "20 logs" note is stale for this copy; the fixture uses the full 50.

(Run-to-run determinism confirmed: regenerating gives a byte-identical file. Reducing to N=20
changes the decoy statistics, e.g. protein decoy mean -5.659/std 3.360 at N=20 vs
mean -5.554/std 3.395 at N=50, so the decoy count is load-bearing and is pinned at 50.)

## What the fixture contains

`tertiary_frustration.dat`: 328 protein-protein contact rows, 92 residues, 0 bad sequences /
50 good sequences. Reference column layout (16 fields, space-separated):

```
i  j  chain_i  chain_j  x_i y_i z_i  x_j y_j z_j  r_ij  AA_i  AA_j  E_native  decoy_mean  decoy_std
```

where `i,j` are 0-based residue indices, the coordinates are the CB (CA for GLY) representative
atom, `r_ij` is the CB-CB representative-atom distance, and the contact set is `r_ij <= 10.0 A`
with `|i-j| > 9` (intra-chain). The per-contact Z-score is `(E_native - decoy_mean)/decoy_std`,
**not** written as a column by the reference (it is computed downstream / used only for the
viz cutoffs). See `AA_DESIGN_DECISION.md` for how this layout maps to FrustraPy's 19-column
`tertiary_frustration.dat` and the **sign-convention** difference vs the AWSEM path.

## Files

- `Frust_Post_public_py3.py` - faithful Python 3 port of the reference post-processor (only
  mechanical 2->3 changes plus stripping the matplotlib/scipy plotting imports and the
  per-residue debug prints; numeric path unchanged).
- `tertiary_frustration.dat` - the regenerated golden output (sha256 recorded in the design doc).
