# TMOL lane decision (gate TMOL-GATE, mission #35)

Decision document for the 8-mission tmol lane (Lane D, tasks #35 to #42). This is the gate
deliverable. The seven downstream missions stay PENDING until a human reads this document and
approves. No kernel or backend code is written by this mission.

Author: Felipe Engelberger. Status: DECISION GATE, awaiting human approval.

## 0. What was verified, and how

The maintainer fork engelberger/tmol was cloned read-only to `/workspace/tmol_src`
(default branch `master`, HEAD `f4d0916a5a952e8fb048950019c2ee8df8c99558`,
commit date 2026-06-08, top commit "Key on hash of cart-bonded params in PBT caching (#383)").
The container firewall allowed the clone and allowed fetching the public Rosetta core
LICENSE.md over https. Every license and provenance claim below cites a file path in that clone,
a URL fetched in-container, or an artifact already present in the container. Nothing here is
paraphrased from memory; where a fact could not be fetched it is marked as a maintainer step.

## 1. License-tier matrix (the heart of the gate)

### 1.1 What each component is licensed under (verified)

- tmol CODE is Apache-2.0.
  - `/workspace/tmol_src/LICENSE` is the standard Apache License 2.0 text.
  - `/workspace/tmol_src/pyproject.toml:5` declares `license = {text = "Apache-2.0"}`; the
    project author is the "Institute for Protein Design" (`contact@ipd.uw.edu`),
    version 0.1.12.
  - Consequence: the algorithm and the source can be reused and adapted directly with
    attribution and a NOTICE, with no clean-room rewrite required for the code itself.

- Rosetta CORE is NOT OSI open source. It is the Rosetta Software Non-Commercial License
  Agreement.
  - Fetched in-container from `https://raw.githubusercontent.com/RosettaCommons/rosetta/main/LICENSE.md`:
    the preamble states plainly "While the Rosetta source code is published on GitHub, it is not
    'Open Source' (according to the OSI definition)", that commercial use requires a license from
    University of Washington CoMotion (`license@uw.edu`), that "All forks of the Rosetta code
    must maintain the current licensing restrictions", and that PyRosetta is covered by a
    separate license (`LICENSE.PyRosetta.md`).
  - In-container corroboration: the atomic reference pipeline at
    `/workspace/atomic_frustratometer_ref/BINARY_LOCATION.txt` records that the Rosetta
    `rosetta_scripts` binary is "license-gated, NOT copied". The wrapper code there is Apache-2.0
    (`/workspace/atomic_frustratometer_ref/LICENSE`), but the engine it drives is license-gated.

- PyRosetta has its own separate license (`LICENSE.PyRosetta.md`, referenced in the Rosetta
  preamble). It is not installed in the container (consistent with the AA lane finding,
  memory `aa-audit-done`). Free for academic use, commercial via CoMotion.

- "Open Rosetta" / OMSF is an in-progress transition, not done. The current canonical Rosetta
  LICENSE.md fetched above is still the Non-Commercial Agreement, so the premise "Rosetta is now
  open source" is REFUTED as of this gate (2026-06-08). Re-verify when downstream missions run.

### 1.2 The parameter provenance verdict (the crux)

tmol's repository is Apache, but the numeric parameter tables it ships in
`tmol/database/default/scoring/` descend from the Rosetta database. This is verified, not assumed:

- `tmol/database/default/scoring/hbond.yaml:58` carries an explicit provenance comment:
  `polynomial_parameters: # Parameters imported from rosetta sp2_elec_params @v2017.48-dev59886`.
- `tmol/database/default/scoring/ljlk.yaml` holds the Rosetta etable values verbatim
  (for example `CNH2 lj_radius: 1.968297, lj_wdepth: 0.094638, lk_dgfree: 3.70334`).
- `tmol/database/default/scoring/ref.yaml` holds the ref2015 per-residue reference weights
  (for example `ALA: 2.3386`, `PRO: -5.1227`, `TRP: 3.035`).
- `tmol/database/default/scoring/elec.yaml` holds the Rosetta sigmoidal-dielectric fa_elec
  constants (`elec_sigmoidal_die_D: 79.931`, `D0: 6.648`, `S: 0.441546`).

So the answer to the gate's three-way question (re-derived vs vendored vs inconsistent) is (b):
tmol VENDORS Rosetta-DB values. They are not re-fit Apache numbers. There is NO per-file license
header on these YAMLs and NO `NOTICE` file anywhere in the tmol clone (`grep` for
`NOTICE`/`copyright`/`rosetta` across `tmol/database/default/` returns only the one hbond comment
above). This is the tension the lane must manage, not wish away: Apache code, Rosetta-sourced
numbers, no NOTICE.

### 1.3 The matrix

The decision is a tier matrix, not one verdict. "Reuse the wheel where the license allows;
reserve original work for the genuinely novel parts."

| Tier | Audience | Code | Rosetta-DB params | Action |
|---|---|---|---|---|
| Academic / FrustraPy default | not-for-profit research, government, universities | Apache, reuse directly | free under the Rosetta Non-Commercial Agreement for this audience | REUSE the tmol params directly; do NOT re-derive. Add the Apache NOTICE for the code. |
| Permissive / redistributable (tmol-webgpu, any commercial path) | anyone, including for-profit | Apache, reuse directly | shipping them verbatim risks imposing the non-commercial restriction on the whole package | PARAMETER-LOADER SEAM: ship Apache code plus a loader; the build vendors NO non-commercial param file; params are user-supplied (under the user's own Rosetta license) or from genuinely open published values. |
| Re-derive | only where a tier is blocked AND a public source exists | n/a | re-derive ONLY the specific blocking values that have a public origin (cite the paper/table) | targeted, not blanket. Where a value is a fitted Rosetta number with no clean public origin, mark NO-PUBLIC-ORIGIN and keep it behind the loader; do not invent a derivation. |

The loader seam already exists structurally in tmol: `tmol/database/__init__.py:19-25`
(`ParameterDatabase.get_default`) calls `ParameterDatabase.from_file(<package>/default)`, and
`from_file` (`:31-35`) reads the chemical and scoring databases from a directory path. A
permissive build points `from_file` at a user-supplied directory and ships none of the
non-commercial YAMLs. This is concrete, not aspirational.

Attribution obligation for the academic tier: because no NOTICE ships in tmol, reusing tmol code
under Apache-2.0 requires us to author the NOTICE ourselves (Apache-2.0 section 4). That is a
deliverable of mission #36 (`docs/tmol/NOTICE_tmol.md`).

## 2. tmol feasibility (read-confirmed; runtime confirmation is downstream)

The gate audited the source. Whether tmol evaluates energies on CPU in-container is an empirical
question reserved for #37/#38 (it requires installing the wheel); the gate states what is known
from reading, and flags the open runtime question honestly.

- In-scope energy terms are all present as first-class score terms under `tmol/score/`:
  `ljlk` (fa_atr/fa_rep/fa_sol), `lk_ball`, `elec` (fa_elec), `hbond`, and `ref`. The full
  ref2015 set also includes `dunbrack`, `rama`, `cartbonded`, `disulfide`, `backbone_torsion`,
  `omega`, `constraint`, which are not needed for inter-residue contact frustration but matter
  for a total-energy match.
- A scoring entry point exists: `tmol/score/score_function.py:16` (`class ScoreFunction`),
  `:140` (`render_whole_pose_scoring_module`), `:246` (`__call__(coords, sum_terms, apply_weights)`).
- tmol ships a numerical ORACLE usable for parity: `tmol/tests/data/term_baselines/` has a
  per-term directory for each in-scope term (`LJLKEnergyTerm`, `LKBallEnergyTerm`,
  `ElecEnergyTerm`, `HBondEnergyTerm`, `RefEnergyTerm`) containing reference energy YAMLs
  (for example `LJLKEnergyTerm/test_whole_pose_scoring_10.yaml` records term0 = -417.95831298828125
  on ubiquitin). The compare helper is `tmol/tests/score/common/test_energy_term.py`
  (`assert_allclose`). The fixtures are real PDBs at `tmol/tests/data/pdb/` (`1ubq.pdb`,
  `1qys.pdb`, `1BL8.pdb`, others). The per-term tests
  (`tmol/tests/score/{ljlk,lk_ball,elec,hbond}/test_*_energy_term.py`) run with
  `update_baseline=False`, that is, against the shipped baselines. This one fixture set becomes
  the SHARED oracle for both the Python provider (#38/#39) and the WebGPU port (#40), one oracle
  two consumers, no per-branch drift.
- tmol covers the full atomic pipeline, not just energy: `tmol/relax/fast_relax.py` (FastRelax)
  and `tmol/pack/{pack_rotamers.py, packer_task.py, simulated_annealing.py, rotamer/}` (rotamer
  repacking). So tmol can plausibly do the entire atomic engine half (fixed-backbone relax,
  RestrictToRepacking-equivalent repack, then per-residue-pair energy) that the AA plan needed
  PyRosetta for.
- Compute lives in AOT-compiled C++/CUDA kernels (`tmol/_cpp_lib.py`, `tmol/_load_ext.py`,
  `tmol/_cuda_env.py`, `CMakeLists.txt`, scikit-build-core). The README advertises a CPU-only
  wheel (`+cpu`, PyTorch 2.10, cp312). OPEN QUESTION, deferred to #37/#38: does loading a pose
  and evaluating per-pair energies work out of the box on CPU with no GPU and no local nvcc?
  This was NOT confirmed at the gate (the wheel was not installed). If CPU eval does not work,
  #38 becomes a focused term reimplementation validated against the shipped oracle rather than a
  thin adapter. The downstream brief must answer this by RUNNING, not reading.

## 3. tmol's recommended role and the AA-lane connection

Recommendation: BOTH roles, that is option (c).

1. tmol becomes a license-clean Python per-residue-pair energy provider for the `atomic`
   backend, an in-container alternative to PyRosetta (mission #38). For the academic tier it can
   REPLACE the PyRosetta engine half of the atomic backend, removing the blocker that left
   AA-ENGINE stalled (memory `aa-audit-done`: the engine half was blocked because PyRosetta is
   license-gated and not installable here). This replacement is conditional on the section 2 CPU
   runtime question resolving in favor; if it does not, tmol still provides the term reference
   and the shared oracle.

2. tmol's terms are also the basis for the standalone, community-reusable `tmol-webgpu` repo
   (missions #40/#41), kept OUT of frustrapy and OUT of frustra-webgpu so the Rosetta Commons
   community can reuse browser-native ref2015 kernels. frustra-webgpu then consumes tmol-webgpu
   as a dependency.

### How tmol plugs into the AtomicBackend without forking the AA design

The AA lane already chose the seam (memory `aa-audit-done`):
`FrustrationBackend.compute_energies(calculator, pdb)` is the single abstract method, and the
only calculator misfit is the LAMMPS-specific prep that runs at
`frustration_calculator.py:186` before the backend, fixed by a backend-gated
`requires_lammps_prep` flag. tmol replaces ONLY the "per-pair energy" source inside
`compute_energies`. Everything downstream is unchanged and reused as-is:

- decoy generation (composition-preserving permutations, the paper's one scheme),
- the Z-score with the load-bearing sign convention `FrstIndex = (DecoyEnergy - NativeEnergy) / SDEnergy`,
- the AA post-processor that writes `tertiary_frustration.dat` in AWSEM format (the Python3 port
  in `docs/atomic/golden/Frust_Post_public_py3.py`, regenerable without Rosetta, already a golden
  fixture from AA-AUDIT),
- `process_results`, `compute_density`, the output-contract tables, and the visualization.

So tmol does NOT reimplement the energy model in the frustration sense; it swaps the per-pair
energy source under a fixed seam, exactly the "orchestrate, do not reimplement" discipline the
lammps and atomic backends already follow.

### Dependency on the AA lane

The AA post-processor (AA-OUTPUT) and golden fixture already exist (memory `aa-audit-done`), so
#38 does not block on them. The AtomicBackend seam itself (the `requires_lammps_prep` flag plus
`compute_energies`) is defined by the AA lane's AA-INTEGRATE. Recommendation: tmol #38 should NOT
wait for the PyRosetta-based AA-ENGINE to finish, since the whole point is to provide an
in-container engine that AA-ENGINE could not. If the AtomicBackend seam has landed when #38
starts, reuse it verbatim; if it has not, #38 may define the same minimal seam itself and the AA
lane adopts it. Either way there is one seam, not two atomic engines that diverge.

Honesty constraints carried forward unchanged: the permutation decoy scheme is the paper; any
atomic config/mutational/singleresidue split is an extension beyond the paper and must be
labeled parity-backed vs experimental. tmol reimplements `beta_nov2016_cart`; canonical Rosetta
production is ref2015, so the tmol-vs-Rosetta gap must be quantified, not assumed zero (mission #39).

## 4. In-container vs maintainer validation split

In-container, no license needed (tmol is Apache, ships its own params and oracle, CPU wheel
advertised):

- read and map the tmol source, terms, params, and oracle (this gate; #36 and #37),
- install the CPU wheel and evaluate ref2015 per-pair energies on CPU (#38, pending the section 2
  runtime confirmation),
- validate the evaluator and the backend against tmol's shipped `term_baselines` oracle (#38, #39
  in-container half),
- regenerate the atomic golden post-processor output without Rosetta (already done, AA-AUDIT),
- numeric parity of the WebGPU kernels against the PyTorch evaluator export (#40),
- the browser atomic pipeline parity against the Python atomic backend (#41),
- CPU and WebGPU timings and structural-pattern agreement for the benchmark (#42 in-container rows).

Maintainer, license-gated (needs Rosetta or PyRosetta on a licensed machine):

- cross-check tmol energies against canonical Rosetta/PyRosetta energies, the scientific claim
  that lets the atomic configurational index be called parity-backed rather than experimental
  (#39 maintainer half),
- any GPU rows and the PyRosetta benchmark rows (#42),
- any decision to redistribute Rosetta-DB params commercially (legal, not engineering).

## 5. Confirmed 8-mission breakdown

The gate confirms the GATE file's #35 to #42 structure. One structural note for the record: an
earlier draft set split the energy work into a standalone evaluator mission plus a backend-wiring
mission, and folded the term map into the gate. This gate instead keeps the GATE file's framing:
a read-only ENERGY-AUDIT (#37) that PARAM-SOURCING (#36) and PY-BACKEND (#38) both consume, and a
single PY-BACKEND (#38) that both builds the provider and wires it into the AtomicBackend. The two
framings cover the same work; the GATE framing is adopted.

| # | Mission | In-container vs maintainer | Depends on | Deliverable |
|---|---|---|---|---|
| 35 | TMOL-GATE (this) | in-container | none | this document + 7 briefs |
| 36 | TMOL-PARAM-SOURCING | in-container | #35 approved | `docs/tmol/PARAM_SOURCING.md`, `docs/tmol/NOTICE_tmol.md`, `param_inventory.json`, boundary test |
| 37 | TMOL-ENERGY-AUDIT | in-container | #35 approved (parallel to #36) | `docs/tmol/ENERGY_AUDIT.md` (term to param to oracle map) |
| 38 | TMOL-PY-BACKEND | in-container (if CPU eval works) | #36, #37, AA seam (AA-OUTPUT golden done; AtomicBackend seam reused or self-defined) | runnable `atomic` backend via tmol, evaluator + wiring + tests |
| 39 | TMOL-PARITY | in-container half + maintainer half | #38 | `docs/tmol/parity/` vs tmol oracle and atomic golden (in-container) and vs PyRosetta (maintainer); parity-backed vs experimental label |
| 40 | TMOL-WEBGPU-KERNELS | in-container | #36, #37, #38 (numeric reference) | standalone `/workspace/tmol-webgpu` repo, WGSL kernels, headless parity |
| 41 | TMOL-WEBGPU-FRUSTRA | in-container | #40, #38 (parity target) | atomic-frustration mode in `/workspace/frustra-webgpu`, parity vs Python atomic backend |
| 42 | TMOL-BENCH-DOCS | in-container rows + maintainer rows | #38, #41 | cross-backend benchmark + docs + citations |

Critical-path note: #36 and #37 can run in parallel after approval; #38 is the hinge that
unblocks #39, #40, #41; #42 closes the lane. The single biggest risk to the whole lane is the
section 2 CPU-runtime question; #37 and #38 must resolve it early and, if CPU eval fails, the
maintainer escalates (GPU machine or term reimplementation) before #40/#41 invest in WebGPU.

## 6. Summary verdict

- License: tmol code is Apache-2.0; Rosetta core and PyRosetta are Non-Commercial (not OSI open,
  verified from the live LICENSE.md); tmol vendors Rosetta-DB params with no NOTICE. The strategy
  is a tier matrix: reuse params directly for academic FrustraPy, a loader seam with no vendored
  non-commercial file for the redistributable tmol-webgpu, targeted re-derivation only where a
  tier is blocked and a public source exists.
- Role: tmol serves both as the in-container, license-clean Python energy provider that can
  replace the PyRosetta engine half of the atomic backend (academic tier, pending the CPU-runtime
  confirmation), and as the basis for the standalone tmol-webgpu repo. It plugs into the existing
  `compute_energies` seam without forking the AA design.
- Gate: the 8-mission breakdown is confirmed; the seven downstream briefs are authored and remain
  PENDING until this document is approved.
