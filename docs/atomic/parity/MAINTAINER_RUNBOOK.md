# Maintainer runbook: tier-2 atomic parity + cost benchmark

This runbook covers the two PyRosetta-gated measurements that cannot run in the
FrustraPy container: the end-to-end statistical parity of the atomic backend vs the
published method (tier 2), and the atomic cost benchmark vs the coarse-grain path.
Everything here needs PyRosetta installed (see `../README.md`). The in-container
tier-1 post-processor parity gate needs none of this and is described in
`../README.md` and run by `run_atomic_parity.py --golden`.

The harnesses are:

- `run_atomic_parity.py` - the statistical parity metric table (reuses
  `native/bench/metrics.py`).
- `run_atomic_cost.py` - the cost model + the coarse-grain baseline.

## 0. Prerequisites

- PyRosetta installed and the RosettaCommons license accepted (`../README.md`).
- The published method's reference output for the same structure, i.e. its
  `tertiary_frustration.dat` (16-column reference layout) for 1QYS / TOP7 at a known
  decoy count. The trimmed reference copy used for the audit lives at
  `/workspace/atomic_frustratometer_ref/`; its `example_output/` is the N=50 demo.
- The FrustraPy package importable (`import frustrapy` works).

## 1. Seed the Rosetta RNG (reproducibility)

The reference is unseeded, so its repack is stochastic. For a reproducible FrustraPy
run, pin the Rosetta RNG before the first calculation in the process:

```python
from frustrapy.analysis.mutation_backends import set_pyrosetta_seed
set_pyrosetta_seed(1)   # appends "-run:constant_seed -run:jran 1" at PyRosetta init
```

This must be called before any `backend="atomic"` calculation in the process, because
PyRosetta is initialised once per process and reads the flags only at init time.
`set_pyrosetta_seed(None)` restores the default unseeded behavior.

## 2. Triplicate atomic runs

Run the atomic backend three times with distinct seeds, into separate output
directories. Use `D >= 200` decoys for converged statistics (the cutoff the paper
recommends; the demo's 50 is under-converged).

```python
import os
import frustrapy
from frustrapy.analysis.mutation_backends import set_pyrosetta_seed

PDB = "1qys.pdb"          # the structure the reference output is for
N_DECOYS = 200

for seed in (1, 2, 3):
    set_pyrosetta_seed(seed)              # only effective in a fresh process; see note
    os.environ["FRUSTRAPY_ATOMIC_N_DECOYS"] = str(N_DECOYS)
    os.environ["FRUSTRAPY_ATOMIC_SEED"] = str(seed)   # seeds the decoy generator too
    frustrapy.calculate_frustration(
        pdb_file=PDB, mode="configurational", backend="atomic",
        results_dir=f"atomic_seed{seed}",
    )
```

Note: because PyRosetta initialises once per process, run each seed in a **fresh
process** (separate `python` invocation or a subprocess) so each gets its own
`-run:jran`. A simple shell loop over three `python -c "..."` calls is the most robust
way to get three independent seeded repacks.

Each run writes `atomic_seed{seed}/1qys.done/.../tertiary_frustration.dat` (19-column
AWSEM layout).

## 3. The parity metric table

Point the parity harness at the reference output and the three replicates:

```
python docs/atomic/parity/run_atomic_parity.py \
    --reference /workspace/atomic_frustratometer_ref/example_output/tertiary_frustration.dat \
    --test atomic_seed1/.../tertiary_frustration.dat \
           atomic_seed2/.../tertiary_frustration.dat \
           atomic_seed3/.../tertiary_frustration.dat \
    --mode configurational
```

(If the reference ships only the per-contact native/decoy logs, regenerate its
`tertiary_frustration.dat` first with the golden post-processor, exactly as
`golden/README.md` documents, at the matching decoy count.)

The harness prints, per replicate, the Spearman / Pearson R^2 / max|delta| / RMSE /
class-agreement on `FrstIndex` over the shared contact set, then a
mean +/- sample-std summary across the triplicate. Expectations and how to read them:

- **Contact set**: should be identical across replicates and vs the reference (the
  contact set is geometry-only, independent of the stochastic repack). Any
  only-in-ref / only-in-test count is a structure-preparation discrepancy to fix
  first, not stochastic noise; the harness warns on it.
- **Spearman / class-agreement**: the load-bearing parity metrics. The repack is
  stochastic so do not expect bit-exact, but rank order and class assignment should be
  high and stable across seeds. Report the triplicate mean +/- std.
- **max|delta| / RMSE**: the spread of the stochastic repack on the index itself;
  these are the honest "how close" numbers and will be larger than the tier-1 print-
  precision floor.

Record the full table in the benchmark report alongside the coarse-grain numbers.

## 4. The cost benchmark

First measure the coarse-grain baseline in-container (real number, no Rosetta):

```
python docs/atomic/parity/run_atomic_cost.py --lammps-baseline \
    --pdb 1qys.pdb --mode configurational --repeats 3
```

Then measure one atomic decoy repack wall time `t_repack` on your hardware (time a
single decoy thread+repack+extract; `frustrapy.backends.atomic_engine.compute_decoy_pair_energies`
on one decoy sequence is the unit), and project the full atomic cost and the
atomic-vs-LAMMPS ratio:

```
python docs/atomic/parity/run_atomic_cost.py \
    --t-repack <measured_seconds> --n-decoys 200 \
    --n-res 92 --n-contacts 328 --lammps-wall <measured_lammps_seconds> \
    --mode configurational
```

The harness prints the projected wall time for all three modes (so the
single-residue-scan worst case is visible), the native-pose amortization effect, and
the atomic / LAMMPS ratio. The projection is a transparent function of the measured
`t_repack`; report `t_repack` itself so the number is reproducible.

For a fuller picture, sweep `--n-decoys` (e.g. 50, 100, 200, 500) and a couple of
protein sizes (`--n-res` / `--n-contacts`) to show the scaling, and fold the table
into the existing cross-backend benchmark narrative (`native/bench/`).

## 5. What to report

- Tier-1 numbers (already green in-container): copy from
  `run_atomic_parity.py --golden`.
- Tier-2 triplicate parity table: Spearman / R^2 / max|delta| / RMSE / class-agreement,
  mean +/- std over three seeds, with the contact-set-identical check noted.
- Cost: measured LAMMPS baseline, measured `t_repack`, projected atomic wall times per
  mode, and the atomic / LAMMPS ratio, with `D` and hardware stated.
