# AA-ENGINE maintainer runbook: validating the PyRosetta relax/repack half

This runbook covers the one half of the atomic (Rosetta) Frustratometer engine that
cannot be validated in the FrustraPy container: the PyRosetta-driven FastRelax /
repack / threading that produces the per-residue-pair (`ResResE`) energies. PyRosetta
is license-gated and not installed here, so this is a **maintainer step**. Everything
downstream of the energies (the decoy generator, the `ResResE` parser, the
per-residue / per-contact aggregation) is already validated in-container against the
shipped reference logs and the golden fixture; see `AA_DESIGN_DECISION.md` section 4
and `tests/test_atomic_engine.py`.

The code under test lives in `frustrapy/backends/atomic_engine.py`:

- `compute_native_pair_energies(pdb_path, ...)` -- E1, reproduces `native.xml`.
- `compute_decoy_pair_energies(pdb_path, decoy_seq, ...)` -- E2, reproduces
  `test.xml` (threading + relax).

Both return `List[ResPairEnergy]`, the exact structure `parse_resrese_log` yields
from a shipped `*.log`, so once the maintainer confirms the energies match, the rest
of the pipeline (aggregation, Z-score, `.dat`) is the already-validated path.

## What the reference does (the parity target)

The reference pipeline is `/workspace/atomic_frustratometer_ref/example_input/job.sh`,
which drives `rosetta_scripts` over `native.xml` and `test.xml`:

- empty `<SCOREFXNS>` block, so Rosetta's default full-atom score function (**ref2015**,
  confirmed on the shipped `native.log` weights line) is used;
- `FastRelax` with `RestrictToRepacking`, a `MoveMap` fixing the backbone and repacking
  side chains only (`bb=0 chi=1`), `relaxscript="rosettacon2018"`, `repeats=2`;
- a `Neighborhood` selector over the whole chain with `PreventRepackingRLT` outside it
  (a no-op for a single-chain input, but it matters for multi-chain inputs);
- per-pair energies emitted by `ScoreCutoffFilter report_residue_pair_energies="1"`
  (the `ResResE` lines);
- for decoys, a `SimpleThreadingMover` (`start_position="1A"`, `pack_neighbors=1`,
  `neighbor_dis=10`, `pack_rounds=5`) threads the permuted sequence onto the fixed
  native backbone before relax.

The shipped reference example is `1QYS.pdb` (92 residues, chain A), `N=50` permutation
decoys.

## Step 0 -- install PyRosetta (one time, under license)

PyRosetta is free for academic / non-commercial use under the RosettaCommons license
but must be obtained and installed separately. FrustraPy already declares the optional
`pyrosetta` extra and reuses the same install recipe as the mutation backend:

```
pip install frustrapy[pyrosetta]      # pulls the small pyrosetta-installer helper
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
```

Confirm it imports:

```
python -c "from frustrapy.backends.atomic_engine import pyrosetta_available; print(pyrosetta_available())"
# -> True
```

PyRosetta is initialised once per process with `-mute all -ignore_unrecognized_res
true -ignore_zero_occupancy false` (reused verbatim from
`frustrapy/analysis/mutation_backends.py`, matching the reference's
`-ignore_zero_occupancy false`).

## Step 1 -- regenerate the native energies and diff against `native.log`

Run the native fixed-backbone repack on the same input the reference used and compare
the per-pair energies to the shipped `native.log`.

```python
from frustrapy.backends import atomic_engine as ae

REF = "/workspace/atomic_frustratometer_ref/example_output"

# Our PyRosetta path (E1):
ours = ae.compute_native_pair_energies(f"{REF}/native.pdb")

# The shipped reference log, parsed the same way:
theirs = ae.parse_resrese_log(f"{REF}/native.log")

# Index both by the residue-pair key and compare the per-pair scheme energy.
def by_pair(recs):
    return {(r.res1_key, r.res2_key): r for r in recs}

ours_by, theirs_by = by_pair(ours), by_pair(theirs)
shared = set(ours_by) & set(theirs_by)
import statistics
diffs = [
    ours_by[k].scheme_energy("Function1") - theirs_by[k].scheme_energy("Function1")
    for k in shared
]
print("shared pairs:", len(shared), "of ours", len(ours_by), "theirs", len(theirs_by))
print("max |Function1 diff|:", max(abs(d) for d in diffs))
```

What to expect and how to read it:

- **Repacking is stochastic.** FastRelax explores rotamers, so a fresh run will not be
  bit-identical to the shipped log; the reference itself is unseeded. Judge agreement
  the way the parity spine judges the AWSEM decoys: by distribution, not by last bit.
  Expect the per-pair energies to track closely (the backbone is fixed; only side-chain
  rotamers move), the set of interacting pairs to overlap almost completely, and the
  per-residue / per-contact aggregates (and ultimately `FrstIndex`) to agree to within
  the repack noise. To shrink the noise, average several native repacks or fix the
  Rosetta RNG (`-run:constant_seed -run:jran <N>`).
- **If a whole term is systematically off** (e.g. `fa_rep` or `pro_close` shifted by a
  constant, or the pair set very different), that points at a weighting or score-type
  mismatch in `_extract_pair_energies`, not repack noise. Check that the score function
  is ref2015 (`pyrosetta.get_score_function()` default) and that every term name in
  `atomic_engine.TERM_NAMES` maps to a live `ScoreType` on your PyRosetta build (terms
  unknown to a build are reported as 0; the header of your `native.log` is the source
  of truth for the term set).
- **`_extract_pair_energies` reads the energy graph** (`edge.fill_energy_map()` weighted
  by `scorefxn.weights()`). This is the PyRosetta equivalent of `ScoreCutoffFilter
  report_residue_pair_energies="1"`. If your PyRosetta build names a method differently,
  adjust there; the public function signatures and the downstream aggregation do not
  change.

## Step 2 -- regenerate one decoy and diff against `j.log`

Each shipped decoy log `j.log` corresponds to line `j` of `example_output/peptide`
(the permuted sequences). Thread that exact sequence and compare:

```python
peptide = open(f"{REF}/peptide").read().splitlines()
decoy_seq = peptide[0]                       # decoy 1 == 1.log

ours = ae.compute_decoy_pair_energies(f"{REF}/native.pdb", decoy_seq)
theirs = ae.parse_resrese_log(f"{REF}/1.log")
# ... same by-pair comparison as Step 1 ...
```

The same stochastic caveat applies. The threaded sequence is deterministic (it is read
from `peptide`), so the only variability is the repack.

To reproduce a full reference run from scratch, use FrustraPy's seedable generator
instead of the unseeded `RandSeq.py`:

```python
native_seq = "".join(r.aa1 for r in ...)     # or read native.seq
decoys = ae.generate_decoy_sequences(native_seq, n_decoys=50, seed=0)
```

`seed` is an improvement over the reference (which is unseeded); pin it for
reproducible parity runs.

## Step 3 -- end-to-end aggregation (no Rosetta needed once the logs exist)

Once the native and decoy energies are in hand (from a real Rosetta run or the shipped
logs), the aggregation is the in-container-validated path:

```python
result = ae.load_engine_result_from_logs(
    f"{REF}/native.log", [f"{REF}/{i}.log" for i in range(1, 51)], scheme="Function1"
)
# contacts come from the geometry / AA-OUTPUT; per-contact native + decoy mean/sd:
summaries = result.summarize_contacts(contacts)
```

`tests/test_atomic_engine.py::test_engine_reproduces_golden_columns` already asserts
this reproduces the golden `tertiary_frustration.dat` `E_native` / `decoy_mean` /
`decoy_std` columns to 1e-6 over all 328 contacts, so a passing Step 1/2 means the full
pipeline matches the reference.

## Notes for AA-INTEGRATE (parallelism)

`compute_decoy_pair_energies` is a pure function of `(pdb_path, decoy_seq)` with no
shared mutable state beyond the per-process PyRosetta singleton. Run decoys
concurrently under FrustraPy's shared core budget (one pool, `inner = cores // outer`);
do not nest pools. See the package parallelism conventions
(`frustrapy/utils/concurrency.py`, `_threadlimits.py`).

## What is NOT settled here

The sign flip (`FrstIndex = (DecoyEnergy - NativeEnergy) / SDEnergy`), the 19-column
`tertiary_frustration.dat` adapter, the density / `Welltype` columns, and the atomic
frustration cutoffs (`>= 2.5` minimal / `<= -0.5` highly on the AWSEM-sign Z, the
paper's own values, distinct from AWSEM's `0.78` / `-1`) are AA-OUTPUT decisions, not
engine decisions. The engine emits raw native energies and decoy mean/sd only.
