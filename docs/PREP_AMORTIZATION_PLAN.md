# Prep amortization: audit and design

Status: audit and plan only. This document changes no behavior. It maps the
frustration compute trees, fixes the boundary between per-structure prep and
per-variant compute, records the multi-chain invariants, documents the golden
regression baseline, and proposes the amortized API that later work will
implement. The regression baseline and `tests/test_prep_amortization_parity.py`
are the gate every later change is diffed against.

## 1. Why

A per-variant frustration calculation splits into data prep and the energy
kernel. The section-6 bottleneck measurement (host arm64, p53, 393 residues)
attributed about 1.33 s to prep (the `PdbCoords2Lammps` subprocess plus the
structure parse and the gamma read) against about 0.21 s for the kernel, so prep
is roughly 6x the actual compute. A saturation-mutagenesis scan of N positions
x 20 amino acids currently redoes prep for every one of the N x 20 mutants, so
prep dominates the scan wall time.

The structure geometry, the contact map, the 5 Angstrom density, and the gamma
tables are invariant under a sequence mutation: only the residue identity
(`res_type`) changes. So prep can be computed once per structure and reused
across all N x 20 variants. This is the single highest-value optimization and it
helps every backend equally, because all backends consume the same prepared job
directory.

The bottleneck attribution and the per-phase timing it relies on live in
`frustrapy/analysis/frustration_calculator.py` (`calculate()` records
`pdb.phase_times` with keys `prep`, `lammps`, `classify`, `density`, `io`) and in
`native/bench/scaling_bottleneck.py` (`prep_s` vs `*_kernel_*_s`).

## 2. The three compute trees and the prep boundary

All three indices share one energy model (AWSEM) and one Z-score equation; they
differ only in how the decoy ensemble is generated. The energy model is computed
either by the precompiled LAMMPS binary (the `lammps` reference backend) or by the
native C++ core (`native/src/core.cpp`, `compute_frustration`). The mapping below
is read from the native core, which is parity-gated bit-for-bit against the binary,
so it is an exact description of the reference model too.

The native `Engine` (core.cpp) is constructed from a `StructureView` (coords,
`res_type`, `chain_id`, `res_seqid`) plus a `ParamsView` (gamma tables and well
scalars). Its constructor computes the per-residue density `rho` once. Everything
that follows is a reduction over that fixed geometry.

### What each mode randomizes (the decoy ensemble)

| Mode | Probed unit | Decoys randomize | Geometry / density |
|---|---|---|---|
| configurational | contact i-j | decoy identities `it,jt`, a random in-cutoff pair distance `r`, and the density slots `rho[bi],rho[bj]` | native, fixed |
| mutational | contact i-j | only decoy identities `it,jt` (the `(i,k)`,`(j,k)` terms keep native `k` identities) | native, fixed |
| singleresidue | site i | only the decoy identity at site i | native, fixed; all neighbors native |

In all three, the decoy ensemble draws from the native structure's own
`res_type` values and `rho` array. The random-index stream is the deterministic
glibc generator seeded at 1 (`GlibcRand`), so the decoys are reproducible.

### What is invariant under a mutation, per mode

A mutation changes the residue identity at one or more sites. For every mode the
geometry-derived quantities are invariant; only `res_type` enters the energy
reductions. Concretely:

| Quantity | configurational | mutational | singleresidue | Source |
|---|---|---|---|---|
| `coord` (CB/CA) | invariant | invariant | invariant | `_parse_structure`, `StructureView.coord` |
| `chain_id` | invariant | invariant | invariant | `StructureView.chain_id` |
| `res_seqid` | invariant | invariant | invariant | `StructureView.res_seqid` |
| contact map | invariant | invariant | invariant | `contact_map` / contact-list loop (geometry only) |
| 5 Angstrom density `rho` | invariant | invariant | invariant | `Engine::compute_density` (geometry only) |
| gamma tables | invariant | invariant | invariant | `_read_gammas` (per-structure constant) |
| `res_type` | changes | changes | changes (one site) | `StructureView.res_type` |

This is verified directly in the code: `contact_map` and `local_density`
(core.hpp / core.cpp) take only `coord`, `chain_id`, `res_seqid` and the well
geometry; they never read `res_type`. The density loop in
`Engine::compute_density` likewise depends only on geometry. `res_type` is read
only inside `water_energy` / `burial_energy` and the `native_*` and decoy
reductions. So the geometry work (parse, contact list, density) is independent of
identity and can be done once per structure.

A subtlety in the configurational decoy term: the decoy density slots are
`rho[bi]`, `rho[bj]` for random residue indices `bi`, `bj` drawn from the same
native `rho` array. Because `rho` is geometry-only and invariant under mutation,
the configurational decoy ensemble is also fully reusable across mutants (it does
not even depend on the mutated identity, since it samples random identities and
random density slots from the native structure). The only per-variant work in
configurational mode is the native contact energy `native_config(i,j)`.

### The prep boundary

Per structure (do once):

- parse the cleaned job PDB into `coord`, `res_type`, `chain_id`, `res_seqid`
  (`frustrapy/backends/native.py:_parse_structure`);
- read the AWSEM coefficient file and gamma tables (`_read_coeff`, `_read_gammas`);
- the geometry-derived contact list and per-residue density `rho`
  (`Engine` construction in `compute_frustration`);
- upstream of all of the above, the `PdbCoords2Lammps.sh` subprocess that
  materializes `fix_backbone_coeff.data`, `gamma.dat`, `burial_gamma.dat` and the
  cleaned PDB (`frustration_calculator.py:_prepare_calculation_files`).

Per variant (must redo):

- swap `res_type` (the whole vector for an arbitrary mutant; for a single-residue
  scan only the entry at the scanned site i);
- recompute the native energy of the probed unit and its decoy statistics.

## 3. The saturation-scan and multi-chain paths

### Saturation-mutagenesis workflow

Entry points are in `frustrapy/analysis/mutations.py`:

- `mutate_res_scan_parallel(pdb, targets, ...)` is the flattened scan. It builds
  the whole `(residue x amino-acid)` grid as one flat task list spanning all
  target residues and dispatches it through a single process pool (Phase 6
  Lever 1). The pool uses the fork-safe context from
  `frustrapy/utils/concurrency.py` (`get_pool_context`, forkserver or spawn) and
  `imap_unordered`; results are merged deterministically in the fixed
  `AMINO_ACIDS` order by `_concat_parts`, so the row order is reproducible
  regardless of completion order.
- `mutate_res_parallel(pdb, res_num, chain, ...)` is a thin single-target wrapper
  over `mutate_res_scan_parallel`.
- `mutate_res(...)` is the deprecated serial path, kept for back-compat.
- The engine reaches the scan from
  `FrustrationCalculator._generate_singleresidue_analysis`, which collects every
  `(res_num, chain)` to mutate and calls `mutate_res_scan_parallel` once.

Where prep is redone per mutant: each grid task runs `_process_amino_acid`, which
builds a mutant PDB file (the geometric `threading` backend keeps the native
backbone plus CB and relabels the residue; the optional `pyrosetta` backend
repacks side chains), then calls `_score_mutant`, which runs a full
`calculate_frustration` on that mutant PDB. That full call re-enters the
calculator: it re-parses the structure, re-runs `PdbCoords2Lammps.sh`, re-reads
the gammas, and re-runs the backend. So every one of the N x 20 variants pays the
entire per-structure prep again. The pool forks at the single
`ctx.Pool(...).imap_unordered(_process_amino_acid, args_list)` call in
`mutate_res_scan_parallel`.

The geometry-invariance caveat for the scan: with the `threading` backend, a
non-glycine to non-glycine mutation keeps the native backbone and CB unchanged,
so the cleaned mutant PDB has exactly the native CB coordinate and only the
residue name (hence `res_type`) differs. The "only `res_type` changes"
invariant holds exactly for those variants. The exceptions are the glycine
transitions: GLY to X adds a CB (computed from N, CA, C, see
`_process_amino_acid`), and X to GLY removes the CB so the interaction coordinate
falls back to CA. In those cases the interaction coordinate at the single mutated
site changes, which perturbs that site's density and the contacts touching it.
The amortized path must therefore treat a glycine-involving variant as a
single-site geometry patch (recompute `rho` contributions and contacts that
involve the mutated site), not a pure `res_type` swap. All other variants are
pure identity swaps. This caveat is per-site and local; it never invalidates the
rest of the structure's prep.

### Multi-chain audit

Multi-chain correctness is carried end to end by `chain_id` and `res_seqid`:

- `_parse_structure` assigns a 1-based `chain_index` in first-seen order and
  records both a per-residue `chain_id` (used by the energy/density separation
  tests) and the same value as `chain_num` (used to print the chain columns).
- The contact map and density separation test is "different chain, or same chain
  with residue-number separation past the threshold"
  (`Engine::separated_contact` and `compute_density`:
  `chain_id[i] != chain_id[j] || abs(res_seqid[i]-res_seqid[j]) >= contact_min_sep`,
  and `> seq_dist` for the density). Cross-chain pairs are always treated as
  separated, so inter-chain contacts are included. This is exercised by the 1zni
  fixture (insulin, 4 chains), which produces cross-chain contacts in the
  configurational table.
- The output tables carry `ChainRes1`/`ChainRes2` (contact modes) and `ChainRes`
  (singleresidue), written from `chain_num`, so the chain of each residue is
  explicit in the table.
- Mutation site selection is by `(res_num, chain)` throughout the scan:
  `_residue_is_glycine`, the mutant build mask
  `(res_num == r) & (chain == c)`, the per-variant `.part` filtering in
  `_score_mutant` (filters the table to `ChainRes == chain` and `Res == res_num`,
  or both endpoints for contact modes), and the output file name
  `singleresidue_Res{res}_{method}_{chain}.txt`. The same residue number in two
  different chains is handled as two distinct targets.

What the amortized path must preserve (the multi-chain invariants):

- the first-seen chain indexing, so `chain_id`/`chain_num` and therefore the
  `ChainRes*` columns are unchanged;
- cross-chain pairs always separated (contacts and density), so inter-chain
  contacts remain in the contact list and in the density;
- mutation targets keyed by `(res_num, chain)`, so a residue number shared across
  chains selects the correct site per chain.

The golden baseline includes a multi-chain structure for all three modes and a
multi-chain scan with target sites in two different chains, so a regression in
any of these invariants is caught.

Latent multi-chain limitation found during this audit (not fixed here, behavior
unchanged): in `_process_amino_acid`, the per-variant mutant PDB filename is
`{base}_{res_num}_{aa}.pdb` when `split=True` (the default the engine uses) and
`{base}_{res_num}_{aa}_{chain}.pdb` only when `split=False`. So with `split=True`
two scan targets that share a residue number in different chains (for example
residue 5 in chain A and residue 5 in chain B, common because many complexes
restart numbering per chain) write, read, and delete the same mutant file. Run
concurrently in the flattened pool, those workers race and the scan fails with a
`FileNotFoundError`. The engine reaches this through
`_generate_singleresidue_analysis`, which collects every `(res_num, chain)` across
all chains and runs them with `split=True`, so a multi-chain saturation scan over
shared residue numbers is currently affected. The amortized scan path should key
the per-variant mutant file by `(res_num, chain)` regardless of `split` so the
files are unique per target; the regression test pins the working
distinct-residue-number multi-chain scan, and this limitation is recorded so the
fix lands with the amortization rather than silently changing the baseline.

## 4. The golden regression baseline

Captured by `tests/data/prep_baseline/generate_baseline.py`, committed under
`tests/data/prep_baseline/`, and asserted by
`tests/test_prep_amortization_parity.py`.

Panel:

- single-chain: 1crn (crambin, 46 residues, one chain);
- multi-chain: 1zni (insulin, 4 chains, with cross-chain contacts), cleaned to
  ATOM records (altloc A or blank normalized to blank, HETATM/water/ions
  dropped);
- all three modes for each structure (configurational, mutational,
  singleresidue), seq_dist 12, default `lammps` backend;
- the 5 Angstrom density table for the two contact modes;
- a small saturation scan: 1crn at two single-chain positions, and 1zni at one
  position in chain A and the same residue number in chain B, each x 20 amino
  acids, capturing the per-variant FrstIndex.

The `lammps` backend output is the stored oracle; `manifest.json` records the row
counts and sha256 of every fixture plus a native-vs-lammps parity summary. The
fixtures are the regression oracle: later work must reproduce them to full parity.
The test asserts:

- `lammps` reproduces every stored table with zero numeric difference and
  identical `FrstState`, and reproduces every scan with zero FrstIndex
  difference (the Linux binary is empirically deterministic);
- the native backend reproduces every stored table within the CPU parity
  tolerance (max abs FrstIndex difference <= 1.5e-3, Spearman 1.0, 100 percent
  sign and class agreement), skipping cleanly if the native core is not built;
- the multi-chain scan rows select the correct `(res_num, chain)` per target.

Regenerate with:

    python tests/data/prep_baseline/generate_baseline.py

run from an activated venv with the native core built.

## 5. Proposed amortized API (not implemented here)

The goal is to compute prep once per structure and reuse it across all variants,
keeping the existing one-shot API parity-identical for current callers.

### Prepared structure context (Python)

Introduce a per-structure handle that holds the prep outputs:

- the parsed arrays `coord`, `res_type`, `chain_id`, `res_seqid` and the
  `chain_num`/`letters` for output;
- the coefficient scalars and the gamma tables;
- optionally the geometry-derived contact list and the per-residue density `rho`.

The handle is produced once (one `PdbCoords2Lammps.sh` run plus one parse plus one
gamma read), then reused. A scan swaps only `res_type` per variant (for a
single-residue scan, only the entry at the scanned site), patches the single CB
coordinate for a glycine-involving variant, and asks the backend to recompute the
energy and decoy statistics.

This sits above the backends and benefits all of them, because it removes the
repeated subprocess prep that every backend currently pays per mutant. The
existing `native/bench/cross_backend.py:_prepare_core_inputs` already demonstrates
the split: it runs prep once and then calls `compute_frustration` repeatedly with
the same kwargs; the amortized API generalizes that into the scan path.

### Native core boundary

Two options, to be decided when implementing:

1. Give `compute_frustration` an optional precomputed input: a precomputed
   density `rho` and/or a precomputed contact list, so the `Engine` skips the
   geometry work when they are supplied. The current signature stays valid (the
   precomputed inputs are optional), so existing callers are unaffected.
2. Introduce an explicit prepared-structure handle in the core (build geometry
   once, then call a cheaper per-variant reduction that takes a new `res_type`
   vector or a single-site identity override). This is cleaner for a scan but a
   larger surface change.

Either way the existing one-shot `compute_frustration(StructureView, ParamsView,
mode)` must stay bit-for-bit identical for current callers; the prepared path is
additive. The native core already separates geometry (the `Engine` constructor
and `compute_density`) from the identity-dependent reductions, so option 1 is a
small, low-risk change: accept an optional `rho` (and optionally a contact list)
and skip recomputing them.

The single-site override needed for a single-residue scan maps directly to
`Engine::native_single(i, it)`, which already takes the identity `it` as an
argument and holds geometry/density fixed. A scan over the 20 identities at site i
is 20 calls to `native_single(i, it)` plus the decoy reduction, all on the same
prepared `Engine`. The glycine caveat is handled by patching the CB coordinate of
site i and recomputing only `rho` and the contacts that involve site i before the
20 calls.

### Multi-chain invariants the new path must hold

- first-seen chain indexing preserved, so `ChainRes*` columns are unchanged;
- cross-chain pairs always separated in both the contact list and the density;
- mutation targets keyed by `(res_num, chain)`;
- the prepared geometry (contact list, density) is shared across all variants of a
  structure, with the per-site glycine patch as the only exception.

### Win metric

Re-run the section-6 bottleneck measurement
(`native/bench/scaling_bottleneck.py`) on a scan. Success is prep amortized to
about once per structure instead of once per variant: the per-structure prep cost
is paid one time, and the per-variant cost collapses to the kernel, so the kernel
dominates the scan wall time. Concretely, total scan time should fall from about
`(N x 20) x (prep + kernel)` toward `prep + (N x 20) x kernel`, and the
FrustrationData tables and per-variant FrstIndex arrays must be unchanged against
the golden baseline (Spearman = R2 = class-agreement = 1.0, max abs FrstIndex
difference 0 for the CPU path, about 1e-6 for any float32 or GPU path).
