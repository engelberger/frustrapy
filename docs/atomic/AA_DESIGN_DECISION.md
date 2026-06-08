# AA-AUDIT: All-atom (Rosetta) Frustratometer integration - design decision gate

Status: DECISION GATE. This document is the deliverable of mission AA-AUDIT, the first of a
seven-mission set. The other six (AA-INTERFACE-FIT, AA-ENGINE, AA-OUTPUT, AA-MODES,
AA-INTEGRATE, AA-PARITY-BENCH) stay PENDING until a human reads this and approves. No backend
code is written here; the only code produced is a Python 3 port of the reference post-processor
used to regenerate the golden parity fixture (`golden/`).

Reference audited: the atomic packing Frustratometer (Chen et al., Rosetta-based "all-atom"),
trimmed copy at `/workspace/atomic_frustratometer_ref/` (the 1 GB Rosetta DB and the 140 MB
`rosetta_scripts.static.linuxgccrelease` binary were intentionally excluded; see
`BINARY_LOCATION.txt`).

Evidence convention follows CLAUDE.md: `[VERIFIED]` = read in a primary source with file:line or
run end-to-end here; `[VERIFIED EMPIRICALLY]` = confirmed on a real output file produced in this
container; `[INFERRED]` = reasoned deduction; `[UNTESTED]` = depends on Rosetta/PyRosetta, which
are not installed here.

---

## 1. Method summary and mapping to FrustraPy's output contract

### 1.1 The pipeline (from `example_input/job.sh`)

`[VERIFIED, job.sh:1-32]` One run of the reference is a shell script `job.sh <pdb> <N>`:

1. Extract the native CA sequence from the input PDB by `awk`-ing CA `ATOM` records and
   translating three-letter to one-letter codes (`job.sh:5`), written to `native.seq`
   (here 91 residues: `DIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQL`).
2. `python RandSeq.py N native.seq > peptide` generates N decoy sequences (`job.sh:8`).
3. `cp <pdb> 3gso.pdb` and `cp <pdb> native.pdb` (`job.sh:11-12`): the structure is scored under
   the dummy name `3gso`. (`diff -q 1QYS.pdb native.pdb` is clean here `[VERIFIED EMPIRICALLY]`.)
4. Score the native pose: `rosetta_scripts ... -parser:protocol native.xml -s 3gso.pdb ... >
   native.log`, then `grep "ResResE" > native.log` (`job.sh:15-17`).
5. For each decoy line, `sed` the sequence into `test.xml` (replacing the placeholder
   `GKRSNTTGK`), run `rosetta_scripts ... test.xml -s 3gso.pdb > $j.log`, `grep ResResE`, and
   revert the `sed` (`job.sh:20-29`).
6. `python Frust_Post_public.py 100 -2.5 0.5 N 9 0 Function1 -1.5 0.5` (`job.sh:32`).

### 1.2 The decoy model (from `RandSeq.py`)

`[VERIFIED, RandSeq.py:14-15]` Each decoy is `''.join(random.sample(seq, len(seq)))` - a
**random permutation of the native sequence**. This is **composition-preserving** (the decoy
has exactly the native amino-acid counts, reordered), which is a fundamentally different decoy
model from AWSEM's three modes (AWSEM randomizes residue *identities*, optionally with geometry;
see CLAUDE.md section 3). This distinction drives the decoy-model honesty discussion in section 3.

### 1.3 The Rosetta protocol (from `native.xml` / `test.xml`)

`[VERIFIED, native.xml, test.xml]` Both protocols are:
- Empty `<SCOREFXNS>` block, so the **default Rosetta full-atom score function is used**. The
  `native.log` weights line confirms **ref2015** (not talaris): the term set is
  `fa_atr fa_rep fa_sol fa_intra_r fa_intra_s lk_ball_wt fa_elec pro_close hbond_sr_b hbond_lr_b
  hbond_bb_s hbond_sc dslf_fa13 omega fa_dun p_aa_pp yhh_planar ref rama_prepr total` with
  weights `fa_atr 1.0, fa_rep 0.55, fa_sol 1.0, ...` (`native.log:1-2`) - the canonical ref2015
  weight set `[VERIFIED EMPIRICALLY]`.
- `FastRelax` with `RestrictToRepacking` (task op `rtrp`) and a `MoveMap` that **fixes the
  backbone and repacks side chains only** (`bb=0 chi=1`), `relaxscript="rosettacon2018"`,
  `repeats=2`. `native.xml` uses `<Chain number="1" chi="1" bb="0"/>`; `test.xml` uses
  `<Span begin="1" end="92" chi="1" bb="0"/>` (equivalent for this single 92-residue chain).
- A `Neighborhood` selector over `resnums="1A-92A"` with `PreventRepackingRLT` on everything
  outside it (`turn_off_others`) - here the whole chain, so this is a no-op for the monomer but
  matters for multi-chain inputs.
- `test.xml` additionally has a `SimpleThreadingMover` (`name="threader" start_position="1A"
  thread_sequence="GKRSNTTGK" pack_neighbors="1" neighbor_dis="10" pack_rounds="5"`,
  `test.xml:21,31`) that threads the decoy sequence onto the fixed native backbone before relax.
  `job.sh` swaps the decoy sequence into this placeholder for each decoy.
- Energies are emitted by `ScoreCutoffFilter report_residue_pair_energies="1"` (`*.xml:15`),
  which writes the per-residue-pair `ResResE` lines. The native run yields 2120 `ResResE` pair
  lines (`grep -c ResResE native.log` = 2122 including the 2 header lines) `[VERIFIED EMPIRICALLY]`.

So: **fixed backbone, side-chain repack only; the native pose is scored as-is and each decoy is
the native sequence permuted, threaded onto the same backbone, and repacked.** The relax/repack
is what makes the engine half expensive (~15 min for 50 decoys per the README) and is the part
that needs Rosetta.

### 1.4 The post-processor (from `Frust_Post_public.py`)

`[VERIFIED, Frust_Post_public.py]` Pure Python (numpy + Biopython; matplotlib/scipy only for
optional plots). It:
- builds the residue index and a CB-CB (CA for GLY) distance matrix from `3gso.pdb`
  (`get_index`, `calc_dist_matrix`, atom map at lines 52-53,81-82) `[VERIFIED]`;
- for each `.log`, sums a per-residue energy from the `ResResE` pair lines, **keeping only pairs
  with `fa_rep <= 5.0`** (`strs[4]`, lines 161,187), and per `Function1` defines the per-pair
  energy as `total - fa_rep - pro_close - dslf_fa13`
  (`ene = float(strs[-1]) - float(strs[4]) - float(strs[10]) - float(strs[15])`, line 167); each
  pair contributes `0.5*ene` to each of its two residues (`read_log`, lines 172-173) `[VERIFIED]`;
- the per-contact "energy" is the symmetric sum `mat[i,j] = ene_res[i] + ene_res[j]`
  (lines 176-177) - a many-body, per-residue decomposition, not a literal pair energy `[VERIFIED]`;
- pools all decoy contact energies into **one protein-wide** mean and sd (`pro_mean`, `pro_std`,
  lines 235-236) and assigns that single mean/sd to every protein residue (lines 239-241)
  `[VERIFIED]`; ligands (none here) get a per-ligand local statistic;
- writes `tertiary_frustration.dat` per accepted contact (`|i-j| > sep` and `CBdist <= 10.0`),
  with the per-contact Z-score `frust[i,j] = (mat_nat[i,j] - res_mean[i]) / res_std[i]`
  (lines 278,280) `[VERIFIED]`.

### 1.5 Mapping to FrustraPy's output contract

FrustraPy's parser (`frustrapy/utils/helpers.py:233-268`) expects a **19-column**
`tertiary_frustration.dat` indexed as: `[0]Res1 [1]Res2 ... [10]r_ij [11]DensityRes1
[12]DensityRes2 [13]AA1 [14]AA2 [15]NativeEnergy [16]DecoyEnergy [17]SDEnergy [18]FrstIndex`
(helpers.py:235-243), and it copies `[15][16][17][18]` verbatim as strings, then classifies
`FrstState` from `FrstIndex` `[VERIFIED, helpers.py:236-268]`.

The atomic post-processor writes a **16-column** line (`Frust_Post_public.py:280`):
`[0]i [1]j [2]chain_i [3]chain_j [4..6]xyz_i [7..9]xyz_j [10]CBdist [11]AA_i [12]AA_j
[13]E_native [14]decoy_mean [15]decoy_std` - and crucially **no density columns and no FrstIndex
column** `[VERIFIED EMPIRICALLY, golden/tertiary_frustration.dat]`.

So the atomic `.dat` is "AWSEM-format-like" but **not** column-aligned with FrustraPy's parser.
Mapping requires a thin **adapter** that, per contact, emits the 19-column line:
`Res1=i(+offset) Res2=j ChainRes1 ChainRes2 r_ij DensityRes1 DensityRes2 AA1 AA2 NativeEnergy
DecoyEnergy SDEnergy FrstIndex Welltype` where:
- `NativeEnergy <- E_native`, `DecoyEnergy <- decoy_mean`, `SDEnergy <- decoy_std` (direct);
- `FrstIndex` must be **computed** (the atomic `.dat` omits it). To match FrustraPy's sign
  convention, set `FrstIndex = (DecoyEnergy - NativeEnergy)/SDEnergy`, i.e. the **negation** of
  the atomic script's own `frust` (see the sign note below);
- `DensityRes1/2` (the AWSEM 5 A burial density) are absent in the atomic model. The shared
  `compute_density` kernel can fill the `_5adens` table, but the two per-row density columns the
  parser reads at `[11][12]` only feed the `Welltype` short/long/water-mediated label
  (helpers.py:249-258); for the atomic path that label is not defined, so the adapter must either
  emit a sentinel density or the AA output must drop `Welltype`. **This is a real output-contract
  decision deferred to AA-OUTPUT.**

**Sign convention - the #1 hazard, restated for the atomic path.** The atomic script defines
`frust = (E_native - decoy_mean)/sd` and colors `frust <= -2.5` green = **minimally** frustrated,
`frust >= 0.5` red = **highly** frustrated (`Frust_Post_public.py:278,308,320`). Rosetta energies
are "lower = more favorable", so a favorable native contact has very negative `E_native`, giving a
very **negative** Z = minimally frustrated. This is the **published-paper sign**
(`(E_native - <E_decoy>)`), which is the **opposite** of FrustraPy/AWSEM's implemented
`FrstIndex = (DecoyEnergy - NativeEnergy)/SDEnergy` where **positive = minimally frustrated**
(CLAUDE.md section 2, helpers.py:260-268). Therefore the adapter must **negate** the atomic Z so
that the existing FrstState classifier (`>= 0.78` minimal, `<= -1` highly) keeps its meaning. The
adapter inverts the sign; it does **not** invert the cutoffs - and the cutoff *values* are a
separate non-parity issue (section 3).

---

## 2. Interface-fit verdict

**Question:** does the all-atom method slot cleanly into `FrustrationBackend.compute_energies`
as-is, or does it need an earlier seam because the calculator runs LAMMPS-specific prep the
atomic path does not want?

**Finding.** The backend seam is invoked at `frustration_calculator.py:193`
(`self.backend.compute_energies(self, pdb)`), but **before** it, at line 186, the calculator
unconditionally calls `self._prepare_calculation_files(pdb)`, which runs
`PdbCoords2Lammps.sh` via subprocess (`frustration_calculator.py:452-474`) and **fails hard if
that script is missing** (`raise FileNotFoundError`, lines 455-458). That step builds the LAMMPS
data deck, `fix_backbone_coeff.data`, `gamma.dat`, and `burial_gamma.dat` - all of which the
**atomic Rosetta path does not use**. By the time `compute_energies` runs, the job dir holds the
cleaned `{base}.pdb` (which the atomic path *does* want) **and** the LAMMPS artifacts (which it
does not) `[VERIFIED, frustration_calculator.py:184-194,435-474]`.

So the abstract method `compute_energies(calculator, pdb)` is shaped right (write
`tertiary_frustration.dat` into `pdb.job_dir`), and the `NativeBackend` already demonstrates a
backend that ignores the LAMMPS *deck* and only reads the cleaned PDB + coeff/gamma files
(`backends/native.py:559-594`). The atomic backend would read only `pdb.job_dir/{base}.pdb`. The
**only** misfit is that the calculator always pays for `PdbCoords2Lammps.sh` even for a backend
that will not use its output, and that prep requires `perl`/the AWSEM scripts to be present.

**Verdict.** `compute_energies` is the **correct seam**; no new abstract method is needed. But the
mandatory LAMMPS prep should become **backend-gated**. Recommended minimal interface change
(smallest of the three options the mission lists):

- Add a class attribute `requires_lammps_prep: ClassVar[bool] = True` on `FrustrationBackend`
  (default True, so `LammpsBackend`/`NativeBackend` are unchanged), and guard the call at
  `frustration_calculator.py:186` with `if self.backend.requires_lammps_prep:`. The atomic backend
  sets it `False` and does its own prep (clean PDB, write its Rosetta XMLs) inside, or via a new
  optional hook `backend.prepare(self, pdb)` called in its place.

This is a ~5-line change, strictly additive, and keeps the parity spine untouched (the LAMMPS and
native backends still run `PdbCoords2Lammps.sh`). It is small enough that **AA-INTERFACE-FIT can be
folded into AA-INTEGRATE** rather than run as a standalone mission - the only interface work is
this flag plus the optional `prepare` hook. Recommendation: **keep AA-INTERFACE-FIT as a thin
sub-step of AA-INTEGRATE, not a separate gated mission.** (One caveat: the calculator also moves a
fixed `files_to_move` list including `{base}.pdb_{mode}` and `tertiary_frustration.dat`
(frustration_calculator.py:214-218); the atomic backend must produce those same names, which the
adapter in section 1.5 already guarantees.)

---

## 3. Decoy-model honesty: one atomic scheme vs three AWSEM modes

The reference implements **exactly one** decoy scheme: composition-preserving **permutation** of
the native sequence, threaded onto the fixed native backbone and repacked
(`RandSeq.py:14-15`, `test.xml:21`) `[VERIFIED]`. FrustraPy ships **three** AWSEM modes
(configurational, mutational, singleresidue) that differ in what the decoy randomizes
(CLAUDE.md section 3). These do **not** correspond one-to-one. Honest mapping:

- **Atomic permutation decoy ~ closest to AWSEM "mutational"** in spirit (it perturbs sequence
  identity while holding geometry fixed - the backbone never moves, only side chains repack), but
  it is **not** the same: AWSEM mutational independently randomizes the identities of the two
  contacting residues from the full alphabet, whereas the atomic decoy is a **global permutation**
  (composition-locked, all positions changed at once, with explicit side-chain repacking). Label
  this **"atomic-configurational/permutation"**, the paper's single defensible mode.
- **An atomic "singleresidue" mode** (mutate one site, keep the rest native, score per-site) is a
  natural EXTENSION but is **beyond the reference paper** - it would require a different decoy
  generation (single-position scan, not whole-sequence permutation) and a per-site energy
  decomposition. Defensible later, must be labeled original.
- **An atomic "configurational" mode in the AWSEM sense** (randomize identities AND distance AND
  density) has **no atomic analogue**, because the atomic method holds geometry fixed by
  construction (fixed backbone). Claiming it would be misleading.

**Recommendation.** Ship **one** atomic mode first - the permutation decoy as the reference
defines it - and label it explicitly as a **distinct decoy model**, not as a parity match to any
AWSEM mode. Any configurational/mutational/singleresidue split on the atomic backend is an
**EXTENSION BEYOND THE PAPER** and every such mode must carry a `non-parity / original` label in
the output and docs. The reference's two tuning knobs are also non-parity and must be surfaced as
atomic-specific, not silently reusing the AWSEM cutoffs:
- the **energy scheme** `Function1` / `Function2` / `Packing` (which Rosetta terms are dropped;
  `Frust_Post_public.py:166-171`), and
- the **frustration cutoffs** `-2.5` (minimal) / `0.5` (highly) on the atomic-sign Z
  (`job.sh:32`), which are protein/scheme-tuned and bear no relation to the AWSEM `0.78`/`-1`
  contact cutoffs. Reusing AWSEM's `0.78`/`-1` on a Rosetta-energy Z is **not validated** and must
  be flagged; the atomic cutoffs (after the sign flip in section 1.5: `>= 2.5` minimal,
  `<= -0.5` highly) are the paper's own and should be the atomic default, labeled original.

---

## 4. Validation reality: what can be checked here vs what is a maintainer step

**Installed here:** Python 3 + numpy + Biopython (the post-processor's deps). **NOT installed:**
Rosetta (140 MB binary excluded, `BINARY_LOCATION.txt`) and **PyRosetta** (license-gated;
`mutation_backends.pyrosetta_available()` returns False here). So the **engine half**
(FastRelax/repack/threading that produces the `.log` files) **cannot be run-validated in this
container**.

**What WAS validated here `[VERIFIED EMPIRICALLY]`:**
- The reference post-processor is **regenerable without Rosetta**. The Python 3 port
  (`golden/Frust_Post_public_py3.py`) run over the shipped logs reproduces
  `tertiary_frustration.dat`: 328 contacts, 92 residues, 0 bad / 50 good decoy sequences,
  protein decoy mean -5.5540 / std 3.3952, Z in [-3.832, 2.662]
  (golden/`tertiary_frustration.dat`, sha256 `3bc9ef30a42b37cecbc01713acea4c409ecb7633f20d9702bf7c09f44ad89524`).
- It is **deterministic**: two runs produce a byte-identical file (the expensive randomness is
  already frozen into the shipped `.log` files).
- The **ResResE extraction and the `Function1` energy decomposition** are exercised end-to-end
  against the shipped native + 50 decoy logs (the `fa_rep <= 5.0` filter, the
  `total - fa_rep - pro_close - dslf_fa13` term selection, the `0.5*ene` per-residue split, the
  protein-wide decoy pooling). This is the **post-processing half** and it now has a golden oracle.

**What is a MAINTAINER step (license/binary-gated, `[UNTESTED]` here):**
- Install Rosetta (academic license), re-run `native.xml` + `test.xml` over `1QYS.pdb`, and
  confirm our regenerated per-pair energies match the shipped `native.log` / `*.log` (i.e. that the
  shipped logs are reproducible). We can only assert the post-processor is faithful **given** the
  logs.
- Install **PyRosetta** under license and drive the relax/repack/threading from Python (replacing
  the `rosetta_scripts` subprocess) for the AA-ENGINE mission.

**PyRosetta surface FrustraPy already uses (mission asks to read
`frustrapy/analysis/mutation_backends.py`, task 21).** The existing `pyrosetta` *mutation* backend
already exercises a usable subset `[VERIFIED, mutation_backends.py]`:
- `pyrosetta.init(extra_options="-mute all -ignore_unrecognized_res true
  -ignore_zero_occupancy false", silent=True)`, once per process, guarded by a module singleton
  (`_ensure_pyrosetta`, lines 99-109);
- `pyrosetta.pose_from_pdb(path)` to load a full-atom pose (line 198);
- `pose.pdb_info().pdb2pose(chain, res_num)` to map a PDB residue to a pose index (line 199);
- `from pyrosetta.toolbox import mutate_residue; mutate_residue(pose, resi, aa1,
  pack_radius=...)` which mutates and **repacks side chains within a radius using the default
  scorefxn** (line 207) - this is precisely the repack primitive the atomic engine needs;
- `pose.dump_pdb(path)` to write the result (line 208);
- `pyrosetta_available()` / the `pyrosetta` extra in `pyproject.toml`, the install recipe
  (`pip install pyrosetta-installer; ...install_pyrosetta()`), and the lazy-import + actionable
  error pattern (lines 70-97) - all reusable verbatim.

**Reusable for the atomic engine:** the init/lazy-import/availability scaffolding, `pose_from_pdb`,
`pdb2pose`, `dump_pdb`, and the repack-with-scorefxn idea. **What the atomic engine adds beyond the
mutation backend:** a `FastRelax` mover with a fixed-backbone `MoveMap` and `RestrictToRepacking`
(not just `toolbox.mutate_residue`), the `SimpleThreadingMover` for whole-sequence threading, the
ref2015 scorefxn handle, and per-residue-pair energy extraction (the `ResResE` equivalent via
`pose.energies()` / a `ScoreCutoffFilter`-equivalent). So **"we already support PyRosetta"** is
true only for the **mutation** path; the atomic engine reuses the plumbing but needs new Rosetta
movers and the per-pair energy readout, all of which are **maintainer-validated** once PyRosetta is
installed.

---

## 5. Mission plan confirmation

The seven-mission set, confirmed/revised with dependencies and the in-container-vs-maintainer split:

| Mission | Scope | Can land + test in-container now? | Blocked on maintainer PyRosetta/Rosetta? |
|---|---|---|---|
| **AA-AUDIT** (this) | Audit + decision gate + golden fixture | **Done** | No |
| **AA-INTERFACE-FIT** | `requires_lammps_prep` flag + optional `prepare` hook | Yes (pure Python, ~5 lines) | No |
| **AA-OUTPUT** | 16-col atomic `.dat` -> 19-col FrustraPy adapter; sign flip; density/Welltype decision; FrstState | **Yes** - the golden fixture in `golden/` is the oracle; the adapter + classifier are pure post-processing and fully testable here | No |
| **AA-ENGINE** | PyRosetta FastRelax/repack/threading -> per-pair energies -> `.log`-equivalent | Skeleton/structure only; **cannot run-validate** | **Yes** (PyRosetta + Rosetta DB) |
| **AA-MODES** | Atomic permutation decoy generator; optional original single-residue mode | Decoy *generator* (the permutation sampler) testable; end-to-end needs the engine | Partial (generator no; scoring yes-blocked) |
| **AA-INTEGRATE** | Register `AtomicBackend`; wire into `calculate_frustration(backend="atomic")`; public API/docs | Wiring + the post-processing path testable against the golden; full run blocked | Partial |
| **AA-PARITY-BENCH** | Diff our `.dat` vs reference; bench vs LAMMPS/native | Post-processing parity vs golden **now**; engine parity needs Rosetta | Partial |

**Revisions to the plan:**
1. **Fold AA-INTERFACE-FIT into AA-INTEGRATE.** The interface change is a single flag plus an
   optional hook (section 2); it does not warrant a standalone gated mission.
2. **Reorder: do AA-OUTPUT before AA-ENGINE.** The post-processing half is fully validatable here
   against the golden fixture, de-risks the column mapping and the sign flip, and unblocks
   AA-PARITY-BENCH's post-processing lane - none of which need Rosetta. The engine half is the only
   hard-blocked piece, so push it as late as possible.
3. **Split AA-ENGINE and AA-PARITY-BENCH each into an in-container part and a maintainer part**, so
   the in-container parts (engine skeleton + lazy PyRosetta scaffolding reused from
   `mutation_backends.py`; post-processing parity vs golden) can land and be tested now, and only
   the Rosetta-dependent validation waits on the maintainer.

**Net dependency order:** AA-AUDIT (done) -> AA-OUTPUT -> AA-MODES (generator) -> AA-INTEGRATE
(incl. the interface flag) -> AA-ENGINE (maintainer-gated) -> AA-PARITY-BENCH (engine lane
maintainer-gated). AA-INTERFACE-FIT is absorbed into AA-INTEGRATE.

---

## Appendix: golden fixture command and provenance

See `golden/README.md`. Command (in a dir holding `3gso.pdb`=copy of `native.pdb`, `native.log`,
`1.log..50.log`):

```
python3 golden/Frust_Post_public_py3.py 100 -2.5 0.5 50 9 0 Function1 -1.5 0.5
```

Decoy count **N=50** (README demo `./job.sh 1QYS.pdb 50`; `peptide` has 50 lines; all 50 decoy
logs shipped). The mission note "only 20 logs are present" is stale for this trimmed copy - `1.log`
through `50.log` are all present here `[VERIFIED EMPIRICALLY]`. Output:
`golden/tertiary_frustration.dat` (328 contacts; sha256
`3bc9ef30a42b37cecbc01713acea4c409ecb7633f20d9702bf7c09f44ad89524`).
