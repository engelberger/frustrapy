# AA-MODES: the three frustration modes on the atomic backend (one parity-backed, two extensions)

This documents the AA-MODES deliverable: `frustrapy/backends/atomic_modes.py`, which
maps FrustraPy's three AWSEM modes (configurational, mutational, singleresidue) onto
atomic decoy-generation strategies. It parameterizes **only the decoy-generation
step**; the per-pair energy extraction (`atomic_engine`) and the post-processor
(`atomic_post`) are shared and never forked per mode, exactly as the LAMMPS backend
swaps a single keyword in the coefficient file and reuses everything downstream.

## Read this first: parity vs extension

The Rosetta reference (Chen et al.) implements **exactly one** decoy scheme:
composition-preserving **permutation** of the whole native sequence on a fixed
backbone (`RandSeq.py`; see `AA_DESIGN_DECISION.md` section 3). FrustraPy's three
AWSEM modes come from **three different** decoy ensembles, which do **not** correspond
one-to-one to that single scheme. Therefore:

| Atomic mode | Status | What the decoy randomizes | Parity oracle |
|---|---|---|---|
| **configurational** | **PARITY-BACKED** | the whole sequence at once (composition-preserving permutation), threaded onto the fixed backbone and repacked | yes - the post-processing half is gated bit-for-bit against `docs/atomic/golden/` (`tests/test_atomic_post.py`) |
| **mutational** | **EXPERIMENTAL** (extension beyond the paper) | only the identities of the two contacting residues i,j (per contact); geometry frozen | none |
| **singleresidue** | **EXPERIMENTAL** (extension beyond the paper) | only the identity at site i (per site); all neighbors native | none |

`configurational` is the reference's own decoy model and is the only atomic mode with
a parity oracle. `mutational` and `singleresidue` are designed **by analogy** with the
AWSEM definitions (CLAUDE.md section 3); there is no atomic reference for them, no
published method to reproduce, and no parity oracle. They are labeled `experimental`
in the code (`ModeInfo.parity_status`), in every relevant docstring, and here. **Do
not let a user mistake either extension for the published method.**

Why the AWSEM split maps the way it does: the three AWSEM indices share one energy
model and one Z-score and differ only in what the decoy ensemble randomizes. The
atomic backend holds geometry **fixed by construction** (fixed backbone, side-chain
repack only), so "geometry" never varies on the atomic path. The atomic modes
therefore differ only in **which sequence positions are re-identified**, which is
exactly what the three decoy strategies encode (whole sequence / contact pair /
single site).

## What ships now

- **configurational**: fully wired and parity-backed for the post-processing half.
  `ConfigurationalDecoyStrategy` reuses `atomic_engine.generate_decoy_sequences`
  verbatim (the same generator the golden fixture validates), produces one
  protein-wide decoy group, and routes through `atomic_post.write_tertiary_frustration`
  -> the shared `process_results` -> the canonical 14-column contact table. The only
  maintainer-gated piece is the Rosetta scoring of the decoys (AA-ENGINE); the decoy
  generation, the aggregation math, and the output are all in-container-tested.
- **mutational** and **singleresidue**: shipped as **clearly-experimental** modes.
  Their decoy-generation strategies (the part that genuinely differs per mode) are
  complete and unit-tested in-container (`tests/test_atomic_modes.py`): correct
  randomization domain, decoy counts, seed reproducibility, and the invariant that no
  position outside the mode's domain is ever perturbed. Their routing to the shared
  post-processor is demonstrated end-to-end with **mocked** energies (the Rosetta
  scoring is injected as a callable and stubbed): mutational reuses the SAME contact
  writer via per-contact summaries (14-column table), singleresidue uses the
  single-residue `.dat` layout the shared `process_results` already parses (8-column
  table, no `FrstState`). What is **not** validated for these two: any numerical
  result. There is no oracle, and the maintainer Rosetta run is required to produce
  real energies.

It is honest and intended that **one solid mode (configurational) ships parity-backed
and the other two ship as labeled extensions**, rather than three half-validated
modes.

## How a mode is selected (the one seam)

`atomic_modes.get_decoy_strategy(mode)` resolves a mode name to its `DecoyStrategy`.
A strategy's `generate(native_seq, ...)` returns a list of `DecoyGroup`s, each holding
`DecoySpec`s (a full threaded sequence plus the randomization domain it is allowed to
touch). The grouping is the only mode-specific downstream signal:

- configurational -> one group, `key=None` (protein-wide pooling, the reference
  `decoy_stat`);
- singleresidue -> one group per site, `key=site_index` (per-site pooling);
- mutational -> one group per contact, `key=(i, j)` (per-contact pooling).

`aggregate_groups(groups, score_fn)` then scores every spec with the **same**
`score_fn` (the engine's Rosetta thread+repack+extract on the live path, mockable in
tests) and pools each group into one `(mean, std, n)` via `pooled_statistics` (drop
zeros, population std `ddof=0`) - the exact pooling the reference uses. The per-group
statistics become the post-processor's summaries: protein-wide for configurational
(`EngineResult`), per-contact for mutational (`ContactEnergySummary`), per-site for
singleresidue (`SiteEnergySummary`).

## Cutoffs are unchanged and never collapsed

The atomic modes do not own the classification cutoffs and do not change them. The
shared `process_results` applies the same thresholds as every other backend:

- contacts (configurational/mutational): `FrstIndex <= -1.0` highly, `>= 0.78`
  minimally (`FRST_HIGHLY_MAX` / `FRST_MINIMALLY_MIN_CONTACT`);
- single-residue: the table carries **no** `FrstState` column; classification is
  **plot-only** at `0.58` (`FRST_MINIMALLY_MIN_SINGLERES`).

The `0.78` (contact) and `0.58` (single-residue plot) splits are intentional and are
**never** collapsed; `tests/test_atomic_modes.py::test_cutoffs_unchanged_and_not_collapsed`
guards this.

A separate, documented atomic concern is that the reference's own frustration cutoffs
(`-2.5` / `0.5` on the atomic-sign Z; `AA_DESIGN_DECISION.md` section 3) differ from
the AWSEM `0.78` / `-1`. Because AA-OUTPUT flips the sign so the atomic `FrstIndex`
follows the AWSEM convention (`(decoy_mean - native)/sd`, positive = minimally), the
shared AWSEM cutoffs are applied after the flip. Whether the AWSEM contact cutoffs are
the right tuning for a Rosetta-energy Z is itself **not validated** (a maintainer /
AA-PARITY-BENCH question); the configurational sign flip is validated, the cutoff
*values* on atomic energies are not.

## Test coverage (in-container, no Rosetta)

`tests/test_atomic_modes.py` (21 tests):

- parity-status contract: only configurational is `parity-backed`; mutational and
  singleresidue are `experimental`;
- decoy domain per mode: whole sequence / single site / contact pair, with the
  out-of-domain invariant asserted for every generated spec;
- decoy counts (exhaustive scans + sampled), seed reproducibility, and edge/error
  cases;
- the shared aggregation: one `score_fn` pools all three mode shapes; `pooled_statistics`
  drops zeros and uses population std;
- routing each mode through the shared `process_results` to the right table shape
  (14-column contacts for configurational and mutational, 8-column single-residue),
  with mocked energies;
- cutoffs unchanged and not collapsed.
