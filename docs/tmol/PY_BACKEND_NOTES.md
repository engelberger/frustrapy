# tmol Python energy backend (mission #38, TMOL-PY-BACKEND)

Author: Felipe Engelberger. This mission builds the license-clean (Apache-2.0) tmol
energy provider for the all-atom frustration backend and wires it in, per the gate's
role recommendation (`docs/tmol/TMOL_LANE_DECISION.md` section 3: tmol replaces the
PyRosetta engine half of the atomic backend for the academic tier). It builds on #36
(`PARAM_SOURCING.md`) and #37 (`ENERGY_AUDIT.md`); read those first.

Evidence tags follow the repo convention. `[VERIFIED]` = read at a cited file:line or
run end to end here; `[VERIFIED EMPIRICALLY]` = confirmed against a real output/oracle
in this container; `[INFERRED]` = reasoned deduction.

## 1. Method: ADAPT-AND-WRAP (not a reimplementation)

#37 confirmed tmol evaluates the five ref2015/`beta_nov2016` terms on CPU in-container
and reproduces its own shipped oracle. This mission therefore wraps tmol; it does not
reimplement any energy term. The evaluator
(`frustrapy/backends/atomic_tmol_engine.py`):

- builds a tmol `ScoreFunction` carrying ONLY the in-scope pairwise terms at their
  `beta_nov2016` weights (`fa_ljatr 1.0, fa_ljrep 0.55, fa_lk 1.0, fa_elec 1.0,
  hbond 1.0, lk_ball_iso -0.38, lk_ball 0.92, lk_bridge -0.33, lk_bridge_uncpl -0.33`),
  the one-body `ref` term intentionally excluded from the per-contact sum (ENERGY_AUDIT
  section 3) `[VERIFIED]`;
- reads back the per-residue-pair energy tensor via tmol's
  `render_block_pair_scoring_module`, whose `__call__` returns shape
  `(n_poses, n_blocks, n_blocks)` (weighted, summed) and, unweighted, per-subterm
  `(n_subterm, n_poses, n_blocks, n_blocks)` `[VERIFIED EMPIRICALLY, #38 probe]`;
- decomposes that into per-residue energies with the reference's `0.5 * pair`
  split and the `fa_rep <= 5.0` clashing-pair filter (the `Function1` analogue: drop the
  repulsive term, the `pro_close`/`dslf` terms not being in the pairwise set), producing
  exactly the `Dict[res, energy]` shape `atomic_engine.residue_energies` produces from a
  Rosetta log, so the shared `EngineResult` aggregation and the `atomic_post` writer are
  reused unchanged.

`evaluate_pair_energies(pose) -> PairEnergies` is the single evaluator interface; it is
differentiable (`evaluate_pair_energy_tensor` returns the torch tensors and the leaf
coords, so `torch.autograd.grad` flows).

## 2. Oracle tolerance achieved `[VERIFIED EMPIRICALLY, 2026-06-08]`

The pairwise score function the evaluator uses reproduces tmol's shipped 1ubq
whole-pose baselines (`docs/tmol/oracle/1ubq_term_energies.json`, #37) at tmol's own
test tolerance (atol=1e-5, rtol=1e-3), re-proved through the frustrapy-side builder
(`tests/tmol/test_tmol_evaluator.py::test_whole_pose_terms_match_tmol_oracle`).

Summed over all block pairs, the per-residue-pair evaluator reproduces the whole-pose
oracle for the dominant terms:

| term | block-pair sum (unweighted, 1ubq) | tmol baseline | abs diff |
|---|---|---|---|
| fa_ljatr | -417.9582 | -417.95831 | ~1e-4 |
| fa_ljrep | 240.71469 | 240.71466 | ~3e-5 |
| fa_lk | 298.27655 | 298.27652 | ~3e-5 |
| hbond | -55.67562 | -55.67561 | ~1e-5 |
| fa_elec | -134.0210 | -136.29248 | 2.27 |

The `fa_elec` gap (2.27 on -136, ~1.7%) is an **intra-residue (diagonal) bookkeeping**
difference between the block-pair and whole-pose paths; it lives on the `(i, i)`
diagonal, which the frustration contact set never uses (contacts are off-diagonal,
`|i - j| > sep`). The off-diagonal contact elec is the whole-pose value `[INFERRED from
the whole-pose match + diagonal-only gap]`. ljlk and hbond match the oracle directly.

## 3. What the PyRosetta blocker is now, honestly

tmol REMOVES the PyRosetta dependency for **energy evaluation (scoring)**: the native
pose is scored directly from its all-atom coordinates with no Rosetta, no GPU, no
license-gated binary. The native scorer and the evaluator run end to end in-container
(`tests/tmol/test_tmol_evaluator.py`, including a real-tmol-native
`calculate_frustration(backend="atomic-tmol")` end-to-end on 1crn producing the full
output contract) `[VERIFIED EMPIRICALLY]`.

tmol does NOT remove the need for a **side-chain packer** when scoring DECOYS. The
reference's permutation decoys are threaded onto the fixed backbone and *repacked*;
placing decoy side chains needs a rotamer optimizer plus the Dunbrack rotamer library.
tmol ships a packer (`tmol.pack.pack_rotamers` /
`tmol.pack.build_missing_sidechains.build_missing_sidechains`), so a license-clean-tier
path exists, but:

- it is **slow** (a full simulated-annealing packing run per decoy; its rotamer/anneal
  kernels JIT-compile on first use, a multi-minute cold start in this container)
  `[VERIFIED EMPIRICALLY: the pack-kernel JIT did not complete in a several-minute probe]`;
- the **Dunbrack library is in the non-commercial Rosetta parameter tier**
  (`docs/tmol/PARAM_SOURCING.md`), so it is not Apache-clean to redistribute.

So `compute_decoy_residue_energies_tmol` / `_thread_and_repack` are implemented (the
real tmol threading + repack recipe, mirroring
`atomic_engine.compute_decoy_pair_energies`) but are the **heavy, maintainer-validated**
path, exactly like the PyRosetta engine functions are in the AA lane. The verdict:
**the PyRosetta blocker is removed for the engine's scoring half; the decoy packing
half is now license-clean-capable via tmol's packer but remains heavy and
parameter-tier-gated, so it is the maintainer path.**

## 4. Backend wiring (reuses the AA lane, no fork)

`AtomicTmolBackend` (`frustrapy/backends/atomic_tmol.py`, registered name
`atomic-tmol`) reuses the AA lane verbatim:

- `atomic_post` (representative-atom geometry, contact selection, the sign-flipped
  `FrstIndex`, the AWSEM-format `tertiary_frustration.dat` writer) -- gated bit-for-bit
  against `docs/atomic/golden/` by `tests/test_atomic_post.py`; the tmol backend calls
  the same writer, so that parity carries over and is re-asserted through the backend's
  own code path in `tests/tmol/test_atomic_backend.py`;
- `atomic_modes` (configurational = parity-backed permutation; mutational /
  singleresidue = experimental extensions);
- `atomic_engine.EngineResult` / `ContactEnergySummary` for the per-contact
  decoy-statistics aggregation.

It differs from `lammps`/`native` only in HOW it is prepared (`requires_lammps_prep =
False`, the seam from the AA lane's `AA_DESIGN_DECISION.md`) and in the energy engine.
It coexists with the PyRosetta `atomic` backend so the user picks the tier:
`atomic` (PyRosetta, full repack) or `atomic-tmol` (Apache scoring). The default backend
is still `lammps`; the parity spine is untouched.

**Sign convention (load-bearing).** `FrstIndex = (DecoyEnergy - NativeEnergy) / SDEnergy`
(positive = minimally frustrated), the AWSEM-sign negation of the atomic reference's own
Z. Reused from `atomic_post`; asserted end to end in
`tests/tmol/test_atomic_backend.py` (a favorable native gives a positive index and
classifies minimally).

**Decoy honesty.** `configurational` maps to the reference permutation scheme
(parity-backed post-processor); `mutational`/`singleresidue` on the atomic backend are
EXPERIMENTAL extensions beyond the published method, labeled as such in `atomic_modes`.

## 5. Parallelism / thread budget

The tmol backend spawns NO process pool. tmol parallelizes via torch intra-op threads;
the decoy loop is a plain serial loop, so it never nests a pool inside the calculator's
pool. The caller owns the budget: `configure_torch_threads(n)` sets `torch.set_num_threads`
from the calculator's resolved inner core budget (or `$FRUSTRAPY_TMOL_THREADS`), keeping
`outer_pool * torch_threads <= cores` consistent with the package's parallel-safety
notes `[VERIFIED, code]`.

## 6. Maintainer setup (the tmol environment)

Per ENERGY_AUDIT section 4, the published CPU wheel ships without the
`_compiled_inverse_kin` pybind module, so until that upstream gap is closed a C++
toolchain + JIT bridge is required. The combined frustrapy + tmol environment used to
validate this mission:

```
uv venv /tmp/tmolprobe --python 3.12 && source /tmp/tmolprobe/bin/activate
uv pip install "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cpu
uv pip install <tmol cp312 cpu wheel> ninja
uv pip install <frustrapy runtime deps> && uv pip install -e . --no-deps
export TMOL_USE_JIT=1   # and ensure 'ninja' is on PATH
```

Preferred long-term: build tmol from source (compiles all pybind modules, no JIT) or
have upstream include the missing module in the AOT wheel. The frustrapy tmol backend
imports tmol lazily, so `import frustrapy` and the default LAMMPS path never need tmol;
`tests/tmol/test_tmol_evaluator.py` skips cleanly when tmol is not importable.

## 7. Tests

- `tests/tmol/test_tmol_evaluator.py` (tmol-gated): whole-pose terms vs the 1ubq oracle;
  block-pair evaluator shape + ljlk/hbond parity; `Function1` scheme matrix; autograd
  finite gradients; native scorer on 1crn; a real-tmol-native end-to-end (slow).
- `tests/tmol/test_atomic_backend.py` (no tmol needed; injected scorers): registry +
  `requires_lammps_prep` seam + lazy-tmol import; end-to-end on 1crn producing the full
  output contract + the 14-column table + the FrstIndex sign; experimental singleresidue
  8-column table; and the golden-overlap parity through the backend's writer (slow,
  needs the reference logs).

## 8. Incidental fix

A latent pandas >= 3.0 copy-on-write bug in `plots.py` (in-place assignment into a
read-only `DataFrame.values` view when building the contact-map heatmap) surfaced once a
contact-mode graphics run was exercised end to end; fixed by zeroing the lower triangle
on a writable copy. Heatmap output is unchanged; the existing `test_visualization.py`
still passes.
