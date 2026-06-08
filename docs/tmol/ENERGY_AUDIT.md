# tmol ENERGY-AUDIT (mission #37): ref2015 term-to-param-to-oracle map

Author: Felipe Engelberger. Read-only mapping mission. Produces the term map that
PARAM-SOURCING (#36) and PY-BACKEND (#38) consume. No kernel or backend code is written here.

Source: the maintainer fork engelberger/tmol cloned read-only at `/workspace/tmol_src`
(branch `master`, HEAD `f4d0916`, version 0.1.12 per `pyproject.toml:25`). tmol is an
Apache-2.0 PyTorch reimplementation of the Rosetta `beta_nov2016_cart` energy function with
AOT-compiled C++/CUDA kernels (gate doc `docs/tmol/TMOL_LANE_DECISION.md`). Every file:line
below was read in that clone; the CPU-runnability section reports a run, not a reading.

Evidence tags follow the repo convention. `[VERIFIED]` = read at the cited file:line in
/workspace/tmol_src, or run end-to-end. `[VERIFIED EMPIRICALLY]` = confirmed against a real
output file or a reproduced number. `[INFERRED]` = reasoned deduction.

---

## 0. The scoring entry point and the score graph

- `tmol/score/score_function.py:16` `class ScoreFunction` is the assembler. It holds a weight
  per `ScoreType` and lazily instantiates one term object per requested score type through
  `ScoreTermFactory` (`retrieve_term_for_score_type`, `:61`). `[VERIFIED]`
- A term declares `n_bodies()` (1 or 2); the ScoreFunction sorts terms into one-body and
  two-body lists by that value (`score_function.py:72-80`). `[VERIFIED]`
- `render_whole_pose_scoring_module(pose_stack)` (`score_function.py:140`) calls each term's
  own `render_whole_pose_scoring_module` and wraps them in `WholePoseScoringModule`
  (`:237`), whose `__call__(coords, sum_terms=True, apply_weights=True)` (`:246`) concatenates
  every term's per-subterm energy rows, multiplies by the weight vector, and sums.
  `unweighted_scores` (`:253`) returns the raw per-subterm rows; the per-term baselines are
  these raw rows (apply_weights does not change a single-term, weight-1 row). `[VERIFIED]`
- The base-class `render_whole_pose_scoring_module` that every in-scope term inherits is
  `tmol/score/energy_term.py:106-114`; it fetches the term's pose-score function via
  `get_pose_score_term_function()` and builds a `TermWholePoseScoringModule`. `[VERIFIED]`
- The canonical Rosetta weight set tmol ships is `beta_nov2016`, assembled by
  `tmol/score/__init__.py:12-48` (`_non_memoized_beta2016`): the per-`ScoreType` `set_weight`
  calls are the production weights (for example `fa_ljatr 1.0`, `fa_ljrep 0.55`, `fa_lk 1.0`,
  `fa_elec 1.0`, `hbond 1.0`, `lk_ball_iso -0.38`, `lk_ball 0.92`, `lk_bridge -0.33`,
  `lk_bridge_uncpl -0.33`, `ref 1.0`; plus the total-only terms `omega 0.48`, `rama 0.50`,
  `disulfide 1.25`, `cart_* 0.5`, `dunbrack_rot 0.76`, `dunbrack_rotdev 0.69`,
  `dunbrack_semirot 0.78`). `[VERIFIED]` tmol reimplements `beta_nov2016`; canonical
  production frustration work is ref2015, so the tmol-vs-Rosetta weight/term gap must be
  quantified, not assumed zero (this is mission #39, flagged in the gate doc section 3).

All `ScoreType` enum members live in `tmol/score/score_types.py` (`AutoNumber`); the in-scope
ones are `fa_ljatr:5`, `fa_ljrep:6`, `fa_lk:7`, `fa_elec:8`, `hbond:9`, `lk_ball_iso:22`,
`lk_ball:23`, `lk_bridge:24`, `lk_bridge_uncpl:25`, `ref:26`. `[VERIFIED]`

---

## 1. Per-term map

The parameter files are catalogued in detail by #36 (`docs/tmol/PARAM_SOURCING.md`,
`param_inventory.json`); this audit names the yaml each term reads and the loader code, and
cross-references #36 rather than duplicating the value tables. All yamls live under
`tmol/database/default/scoring/`; `ParameterDatabase.get_default()`
(`tmol/database/__init__.py:19-25`) -> `from_file` (`:31-35`) reads them. A permissive build
can repoint `from_file` at a user-supplied directory (gate doc section 1.3).

### 1.1 ljlk -> fa_atr, fa_rep, fa_sol  (Rosetta fa_atr/fa_rep/fa_sol)

| Field | Value |
|---|---|
| Term class | `LJLKEnergyTerm` `tmol/score/ljlk/ljlk_energy_term.py:16` `[VERIFIED]` |
| ScoreTypes | `[fa_ljatr, fa_ljrep, fa_lk]` via `tmol/score/terms/ljlk_creator.py:9`; `score_types()` `ljlk_energy_term.py:35-38` `[VERIFIED]` |
| n_bodies | `2` (`ljlk_energy_term.py:40-41`) -> two-body, **pairwise-decomposable per residue pair** `[VERIFIED]` |
| Params (yaml) | `ljlk.yaml`; loader `LJLKParamResolver.from_database` `tmol/score/ljlk/params.py:82-92`; globals `LJLKGlobalParams` `params.py:20-37`; per-atom-type `LJLKTypeParams` `params.py:40-57` (`lj_radius`, `lj_wdepth`, `lk_dgfree`, `lk_lambda`, `lk_volume`) `[VERIFIED]`. Rosetta-DB-sourced (gate doc 1.2). |
| Kernel | torch ops loaded `tmol/score/ljlk/potentials/compiled.py:16-17` (`ljlk_pose_scores`, `ljlk_rotamer_scores`); C++/CUDA sources `compiled.ops.cpp`, `ljlk_pose_score.cpu.cpp`, `ljlk_pose_score.cuda.cu`; dispatched from `get_pose_score_term_function` `ljlk_energy_term.py:88-91` `[VERIFIED]` |
| Render | inherits `energy_term.py:106-114` `[VERIFIED]` |
| Oracle | `LJLKEnergyTerm/test_whole_pose_scoring_10.yaml` term0(fa_ljatr)=`-417.95831298828125`, term1(fa_ljrep)=`240.71466064453125`, term2(fa_lk)=`298.2765197753906` on 1ubq `[VERIFIED]` |

Note: tmol `fa_lk` is the Rosetta `fa_sol` (LK implicit-solvation) term; a positive value is
the expected desolvation penalty. `[INFERRED]`

### 1.2 lk_ball -> lk_ball_iso, lk_ball, lk_bridge, lk_bridge_uncpl

| Field | Value |
|---|---|
| Term class | `LKBallEnergyTerm` `tmol/score/lk_ball/lk_ball_energy_term.py:18` `[VERIFIED]` |
| ScoreTypes | `[lk_ball_iso, lk_ball, lk_bridge, lk_bridge_uncpl]` via `tmol/score/terms/lk_ball_creator.py:9`; `score_types()` `lk_ball_energy_term.py:66-69` `[VERIFIED]` |
| n_bodies | `2` (`lk_ball_energy_term.py:71-72`) -> **pairwise**, but evaluates explicit attached "waters" per atom first `[VERIFIED]` |
| Params (yaml) | `ljlk.yaml` (shares the LJLK resolver, `lk_ball_energy_term.py:26-28` -> `LJLKParamResolver.from_database`); extra lk_ball globals in `ljlk.yaml` (`lkb_water_dist`, `lkb_water_angle_sp2/sp3/ring`, `lkb_water_tors_*`); lk_ball block params `tmol/score/lk_ball/params.py:10-23` `[VERIFIED]` |
| Kernel | `tmol/score/lk_ball/potentials/compiled.py` ops; C++/CUDA `compiled.ops.cpp`, `lk_ball_pose_score.cpu.cpp`/`.cuda.cu`, plus water generation `gen_pose_waters.cpu.cpp`/`.cuda.cu`; invoked `lk_ball_energy_term.py:225-283` (`gen_pose_waters` `:261`, `lk_ball_pose_score` `:283`) `[VERIFIED]` |
| Render | inherits `energy_term.py:106-114` `[VERIFIED]` |
| Oracle | `LKBallEnergyTerm/test_whole_pose_scoring_10.yaml` term0=`422.03961181640625`, term1=`172.19644165039062`, term2=`1.5785889625549316`, (+term3) on 1ubq `[VERIFIED]` |

### 1.3 elec -> fa_elec

| Field | Value |
|---|---|
| Term class | `ElecEnergyTerm` `tmol/score/elec/elec_energy_term.py:13` `[VERIFIED]` |
| ScoreTypes | `[fa_elec]` via `tmol/score/terms/elec_creator.py:9`; `score_types()` `elec_energy_term.py:30-33` `[VERIFIED]` |
| n_bodies | `2` (`elec_energy_term.py:35-36`) -> **pairwise** `[VERIFIED]` |
| Params (yaml) | `elec.yaml`; globals `ElecGlobalParams` `tmol/score/elec/params.py:17-23` (`elec_min_dis 1.6`, `elec_max_dis 5.5`, `elec_sigmoidal_die_D 79.931`, `elec_sigmoidal_die_D0 6.648`, `elec_sigmoidal_die_S 0.441546`, `elec.yaml:1-6`); per-atom partial charges `PartialCharges` and count-pair reps `CountPairReps` `tmol/database/scoring/elec.py:17-28`, resolved in `tmol/score/elec/params.py:144-168` `[VERIFIED]`. Rosetta sigmoidal-dielectric constants (gate doc 1.2). |
| Kernel | `tmol/score/elec/potentials/compiled.py` ops; C++/CUDA `compiled.ops.cpp`, `elec_pose_score.cpu.cpp`/`.cuda.cu`; invoked `elec_energy_term.py:127-129` `[VERIFIED]` |
| Render | inherits `energy_term.py:106-114` `[VERIFIED]` |
| Oracle | `ElecEnergyTerm/test_whole_pose_scoring_10.yaml` term0=`-136.29248` on 1ubq `[VERIFIED]` |

### 1.4 hbond

| Field | Value |
|---|---|
| Term class | `HBondEnergyTerm` `tmol/score/hbond/hbond_energy_term.py:14` `[VERIFIED]` |
| ScoreTypes | `[hbond]` via `tmol/score/terms/hbond_creator.py:9`; `score_types()` `hbond_energy_term.py:47-50` `[VERIFIED]`. One aggregated `hbond` score type only; tmol does NOT split into Rosetta's `hbond_sr_bb`/`hbond_lr_bb`/`hbond_bb_sc`/`hbond_sc`. |
| n_bodies | `2` (`hbond_energy_term.py:52-53`) -> **pairwise**, with the subtlety that it needs donor/acceptor base geometry generated across residues (`gen_hbond_bases`) `[VERIFIED]` |
| Params (yaml) | `hbond.yaml`; polynomial parameters carry the explicit provenance comment `hbond.yaml:58` `# Parameters imported from rosetta sp2_elec_params @v2017.48-dev59886`; assembled by `CompactedHBondDatabase.from_database` `tmol/score/hbond/params.py:193-285`; loader `HBondDatabase.from_file` `tmol/database/scoring/hbond.py:131-133` `[VERIFIED]` |
| Kernel | `tmol/score/hbond/potentials/compiled.py` ops; C++/CUDA `compiled.ops.cpp`, `hbond_pose_score.cpu.cpp`/`.cuda.cu`, base generation `gen_hbond_bases.cpu.cpp`/`.cuda.cu`; invoked `hbond_energy_term.py:105-131` `[VERIFIED]` |
| Render | inherits `energy_term.py:106-114` `[VERIFIED]` |
| Oracle | `HBondEnergyTerm/test_whole_pose_scoring_10.yaml` term0=`-55.67562484741211` on 1ubq `[VERIFIED]` |

### 1.5 ref (reference weights)

| Field | Value |
|---|---|
| Term class | `RefEnergyTerm` `tmol/score/ref/ref_energy_term.py:12` `[VERIFIED]` |
| ScoreTypes | `[ref]` via `tmol/score/terms/ref_creator.py:9`; `score_types()` `ref_energy_term.py:26-29` `[VERIFIED]` |
| n_bodies | `1` (`ref_energy_term.py:31-32`) -> **ONE-BODY per-residue reference energy, NOT pairwise** `[VERIFIED]` |
| Params (yaml) | `ref.yaml` per-residue weights (`ALA 2.3386`, `CYS 3.2718`, `ASP -2.2837`, ...); read into `self.ref_weights` `ref_energy_term.py:18`; loader `RefDatabase.from_file` `tmol/database/scoring/ref.py:12-15` `[VERIFIED]` |
| Kernel | **none** - pure torch (`eval_ref_energy_for_pose` `ref_energy_term.py:79-124`, an index-select into the weights tensor) `[VERIFIED]` |
| Render | inherits `energy_term.py:106-114`; pose-score fn returns `eval_ref_energy_for_pose` (`ref_energy_term.py:66-67`) `[VERIFIED]` |
| Oracle | `RefEnergyTerm/test_whole_pose_scoring_10.yaml` term0=`-41.275001525878906` on 1ubq `[VERIFIED]` |

---

## 2. The parity oracle (one fixture set, shared by #38/#39/#40)

- Baselines live under `tmol/tests/data/term_baselines/<TermClass>/test_whole_pose_scoring_10.yaml`,
  one directory per term (`LJLKEnergyTerm`, `LKBallEnergyTerm`, `ElecEnergyTerm`,
  `HBondEnergyTerm`, `RefEnergyTerm`). Structure is a nested map `term{i}: {pose{j}: float}`;
  the 1ubq fixture is replicated 10 times so every `pose0..pose9` holds the same value
  (`PoseStackBuilder.from_poses([p1]*10)`, `test_energy_term.py:259`). `[VERIFIED]`
- Compare helper: `tmol/tests/score/common/test_energy_term.py:84-90` `assert_allclose` wraps
  `numpy.testing.assert_allclose`. The whole-pose test
  `EnergyTermTestBase.test_whole_pose_scoring_10` (`:244-275`) loads the baseline
  (`get_test_baseline_data`, `:191`), evaluates `pose_scorer(coords)` (`:267`), and asserts
  with tolerances **atol=1e-5, rtol=1e-3** (defaults at `:252-253`). `[VERIFIED]`
- `update_baseline` is a Python parameter defaulting to `False` (`:251`); when `True` it
  overwrites the yaml (`save_test_baseline_data`, `:155`). There is no pytest CLI flag for it
  (no `--update-baseline` option in any `conftest.py`); regeneration means editing the call
  site. So the shipped tests run against the shipped baselines by default. `[VERIFIED]`
- Per-term test files and the whole-pose test method:
  - `tmol/tests/score/ljlk/test_ljlk_energy_term.py` `TestLJLKEnergyTerm.test_whole_pose_scoring_10`
  - `tmol/tests/score/lk_ball/test_lk_ball_energy_term.py` `TestLKBallEnergyTerm.test_whole_pose_scoring_10`
  - `tmol/tests/score/elec/test_elec_energy_term.py` `TestElecEnergyTerm.test_whole_pose_scoring_10`
  - `tmol/tests/score/hbond/test_hbond_energy_term.py` `TestHBondEnergyTerm.test_whole_pose_scoring_10`
  - `tmol/tests/score/ref/test_ref_energy_term.py` `TestRefEnergyTerm.test_whole_pose_scoring_10`
  each subclassing `EnergyTermTestBase` `[VERIFIED]`
- Fixtures: the pose is built from `tmol/tests/data/pdb/1ubq.pdb`, exposed as the `ubq_pdb`
  fixture (`tmol/tests/data/__init__.py:30-31` -> `pdb.data["1ubq"]`, loaded by
  `tmol/tests/data/pdb/__init__.py:6-9`). Other PDB fixtures exist (`1qys.pdb`, `1BL8.pdb`,
  size-series `bysize_*`) but the baselines above use 1ubq. `[VERIFIED]`

### Exact reproduce command (per term)

```
pytest tmol/tests/score/ljlk/test_ljlk_energy_term.py::TestLJLKEnergyTerm::test_whole_pose_scoring_10 -v
# substitute lk_ball / elec / hbond / ref and the matching Test*EnergyTerm class
```

run with the default `update_baseline=False`, tolerance atol=1e-5 / rtol=1e-3. This requires
tmol's compiled extensions to load (see section 4 for the in-container caveat).

---

## 3. Pairwise scope decision (load-bearing for #38 and #40)

The atomic-frustration contact energy is an **inter-residue contact** quantity
(`FrstIndex = (DecoyEnergy - NativeEnergy) / SDEnergy`, the parity-spine sign convention).
It therefore uses the **pairwise (two-body) set** plus the one-body reference baseline, and
**excludes** the total-only terms. Recorded explicitly so #38 (Python provider) and #40
(WebGPU kernels) implement the identical term set.

**IN - pairwise contact terms (`n_bodies()==2`, decomposable per residue pair i-j):**

| Term | ScoreTypes | n_bodies |
|---|---|---|
| ljlk | fa_ljatr, fa_ljrep, fa_lk | 2 |
| lk_ball | lk_ball_iso, lk_ball, lk_bridge, lk_bridge_uncpl | 2 |
| elec | fa_elec | 2 |
| hbond | hbond | 2 |

**IN as a one-body baseline (`n_bodies()==1`, per residue, NOT per contact):**

| Term | ScoreType | n_bodies |
|---|---|---|
| ref | ref | 1 |

ref is a per-residue constant; it does not contribute to an i-j contact energy and must NOT
be added into a pairwise contact sum. It is relevant only to a per-residue total (single-
residue index) or a whole-structure total. `[INFERRED from n_bodies==1; VERIFIED that ref is
one-body]`

**OUT - total-only terms (excluded from contact frustration energy):** dunbrack
(`dunbrack_rot/rotdev/semirot`), rama, omega, backbone_torsion, cartbonded
(`cart_lengths/angles/torsions/impropers/hxltorsions`), disulfide, constraint. These are
present in tmol (`tmol/score/{dunbrack,backbone_torsion,cartbonded,disulfide,constraint}/`)
and matter for a full single-structure total energy, but not for an inter-residue contact.
`disulfide` is technically a special pairwise cross-link, not a generic distance contact, so
it is excluded from the generic contact set. `[VERIFIED present; INFERRED scope]`

This is the same "orchestrate, do not reimplement the energy model" discipline as the lammps
and atomic backends: #38 swaps only the per-pair energy source under the existing
`compute_energies` seam (gate doc section 3).

---

## 4. In-container CPU-runnability verdict

**Verdict: YES, tmol evaluates the five ref2015 terms on CPU in this container and reproduces
its own shipped oracle - BUT NOT from the AOT wheel out of the box. The v0.1.14 cp312 CPU
wheel is missing one pybind extension module; a C++ toolchain plus JIT mode is required.**
`[VERIFIED EMPIRICALLY, 2026-06-08]`

What was run (full commands in `oracle/README.md`):

1. Network to GitHub releases works (the gate's firewall allowance held). The closest
   published cp312 CPU wheel to the 0.1.12 clone is `tmol-0.1.14+cpu-cp312-cp312-linux_x86_64.whl`
   (0.1.13/0.1.14 are cp312; 0.1.29+ are cp314). Downloaded (53 MB). `[VERIFIED]`
2. Clean venv, Python 3.12.13. `uv pip install "torch>=2.5,<3" --index-url .../cpu` pulled
   `torch 2.12.0+cpu`; then `uv pip install <wheel>` resolved all runtime deps cleanly. `[VERIFIED]`
3. **The AOT wheel does NOT import out of the box.** `import tmol` (and any score-term import)
   fails with `ModuleNotFoundError: No module named 'tmol.kinematics.compiled._compiled_inverse_kin'`.
   Root cause: the wheel ships `tmol/_C.cpython-312...so` (which registers every energy-term
   `torch.ops.tmol_*` op) plus two pybind `.so` modules (`_cubic_hermite_polynomial`,
   `bspline _compiled`), but NOT the third pybind module `_compiled_inverse_kin`, which
   `tmol/kinematics/compiled/compiled_inverse_kin.py:3` imports via `load_module` and which
   `tmol/score/energy_term.py:4` pulls in eagerly (through `tmol.pack.rotamer.build_rotamers`).
   This is a packaging gap in the published CPU wheel, not a torch-ABI incompatibility:
   `_C` loads fine against torch 2.12 (no `TmolExtensionIncompatibleError`). `[VERIFIED EMPIRICALLY]`
4. **Bridged with JIT.** `g++` 13.3 is present; `ninja` was pip-installed. Setting
   `TMOL_USE_JIT=1` makes `tmol/_load_ext.py:36-37` take the JIT branch for every op/module
   (and skip `_C` entirely, so no double op-registration). The one missing kinematics module
   JIT-compiled in ~99 s and cached; the four term kernels (ljlk, lk_ball, elec, hbond)
   JIT-compiled CPU-only (`cuda_if_available` drops the `.cu` files) in ~27-54 s each on first
   run. `[VERIFIED EMPIRICALLY]`
5. **Whole-pose scoring of all five terms on 1ubq reproduced the shipped baselines** at
   tmol's own tolerance (atol=1e-5, rtol=1e-3): LJLK max|d|=0.0, LKBall 3.05e-5, Elec 4.69e-7,
   HBond 1.14e-5, Ref 0.0; all PASS. Pose build (10x1ubq) ~72 s, total run ~239 s including
   first-run JIT compiles (negligible thereafter, cached). Artifacts in `docs/tmol/oracle/`.
   `[VERIFIED EMPIRICALLY]`

**Consequence for #38.** CPU evaluation works and matches the oracle, so #38 is fundamentally
**adapt-and-wrap, not reimplement**. Two operational caveats the downstream brief must carry:
- the published CPU wheel needs the `_compiled_inverse_kin` gap closed. Three clean options,
  in order of preference: (a) **build tmol from source** in-container (`pip install -e .`,
  CMake/scikit-build-core, CPU-only since no nvcc), which compiles all pybind modules; (b)
  ask the maintainer / file upstream to include `_compiled_inverse_kin` in the AOT wheel; (c)
  ship the JIT recipe used here (`TMOL_USE_JIT=1` + g++ + ninja). The frustrapy `atomic`
  backend should not depend on a one-off JIT step in production - prefer (a) or (b).
- the run used wheel 0.1.14 against the 0.1.12 source clone; the numbers match the clone's
  baselines, so the skew is immaterial for these terms, but #38/#39 should pin one tmol
  version and regenerate the oracle against it.

## 5. Summary

- The five in-scope terms (ljlk -> fa_atr/fa_rep/fa_sol, lk_ball, elec -> fa_elec, hbond, and
  the one-body ref) are each first-class tmol score terms with a clear class, `score_types()`,
  `n_bodies()`, a named yaml + loader, a compiled kernel (except ref, pure torch), and a
  shipped baseline. All file:line citations above are `[VERIFIED]` against `/workspace/tmol_src`.
- Pairwise contact set for the atomic frustration energy = ljlk + lk_ball + elec + hbond
  (all `n_bodies()==2`); ref is one-body (per residue, not per contact); the rest (dunbrack,
  rama, omega, cartbonded, disulfide, constraint) are total-only and excluded (section 3).
- The oracle is one shipped fixture set (`tmol/tests/data/term_baselines/`, 1ubq, atol=1e-5/
  rtol=1e-3) reused by #38/#39/#40.
- CPU runs in-container and reproduces the oracle; the published AOT CPU wheel needs the
  `_compiled_inverse_kin` packaging gap closed (build-from-source or upstream fix). #38 is
  adapt-and-wrap.
