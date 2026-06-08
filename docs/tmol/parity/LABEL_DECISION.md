# Label decision: the tmol atomic configurational index (TMOL-PARITY #39, scope item 4)

This records the parity label for the all-atom frustration index computed with the tmol
energy engine (`backend="atomic-tmol"`), feeding the publication benchmark (#42). It
rests on the in-container parity runs in this directory, all regenerable with the two
scripts under `frustrapy/backends/tmol_eval/`.

## The question

The `atomic-tmol` backend computes the same all-atom frustration index as the PyRosetta
`atomic` backend, with the same post-processor, the same decoy scheme, and the same sign
convention; it differs only in the energy engine. The energy engine is the load-bearing
part: tmol reimplements Rosetta `beta_nov2016_cart`, while the published Rosetta-packing
Frustratometer (the parity oracle) runs production `ref2015`. So the index can be called
**parity-backed** only to the extent that tmol energies reproduce the Rosetta energies the
reference used. The gap must be measured, not assumed zero.

## What was proven in-container (no license), for real

| check | scope | result | source |
|---|---|---|---|
| evaluator vs tmol's own 1ubq oracle | item 1 | 9/9 in-scope subterms within atol=1e-3,rtol=1e-3; worst dev **3.05e-5** | `tmol_oracle_parity.md` |
| backend post-processor vs AA golden | item 2 | FrstIndex Spearman **1.000000**, class **100%**, max\|d\| **4.99e-4** (bit-for-bit at print precision, identical contact set) | `tmol_golden_parity.md` |
| tmol vs frozen Rosetta logs, per-residue Function1 | item 3 (in-container) | Spearman **0.905**, R^2 0.70, RMSE 1.67 (pooled over 50 poses) | `tmol_vs_rosetta_frozen.md` |
| tmol vs frozen Rosetta, engine-swap FrstIndex | item 3 (in-container) | Spearman **0.917**, R^2 0.80, class-agreement **84.0%**, max\|d\| 2.27 | `tmol_vs_rosetta_frozen.md` |

Per-term breakdown (where the energy-function gap lives): the dominant
Lennard-Jones/solvation/electrostatic terms track Rosetta tightly (per-pair Spearman
fa_atr 0.97, fa_rep 0.99, fa_sol 0.98, fa_elec 0.98), while the gap is **term-local** in
hbond (Spearman 0.34, signed sum -9.3 vs -77.8) and lk_ball (Spearman 0.65, sign-flipped
sum -10.6 vs +11.5). The divergence is therefore NOT a uniform weight/version offset; it
is concentrated in the hydrogen-bond and anisotropic-solvation kernels, where tmol's
`beta_nov2016` per-pair decomposition differs from Rosetta `ref2015`'s `ResResE`
assignment. `[VERIFIED EMPIRICALLY, 2026-06-08, in-container]`

## What is NOT yet proven (maintainer, license-gated)

The live-PyRosetta cross-check (`tmol_vs_pyrosetta.md`, currently SKIPPED) is required to:

1. confirm the shipped frozen logs reproduce a fresh Rosetta `ref2015` single-point score
   (the frozen logs are the only Rosetta numbers available in-container; their fidelity to
   a fresh run is assumed, not yet re-verified here);
2. cross-check tmol against Rosetta on the **native** pose, which the shipped fixture
   provides only as a representative-atom structure tmol cannot build, so the in-container
   cross-check used complete decoy poses as the comparison set.

`[UNTESTED -- awaits the maintainer's licensed PyRosetta run]`

## Decision

The tmol `atomic-tmol` **configurational** index is labeled:

> **PARITY-BACKED IN RANK AND CLASS, NOT BIT-EXACT vs the Rosetta-energy index.**

Concretely, for the publication benchmark (#42):

* The **engine wrapping is exact** (oracle 3e-5) and the **post-processing is bit-for-bit**
  against the published-method golden (item 2). Those two halves carry the same `[VERIFIED]`
  weight as the LAMMPS parity spine.
* The **energy engine introduces a real, quantified gap**: swapping Rosetta `ref2015`
  energies for tmol `beta_nov2016` energies preserves the per-contact frustration ranking
  (Spearman ~0.92) and the highly/neutral/minimally class for ~84% of contacts, but is not
  bit-identical, and the gap is concentrated in the hbond and lk_ball terms. So the tmol
  configurational index must be reported as a **license-clean approximation of the
  Rosetta-packing index, parity-backed in rank/class**, with the energy-function gap stated,
  rather than as bit-for-bit parity.
* `mutational` and `singleresidue` on the atomic backend remain **experimental** (extensions
  beyond the published single-decoy-scheme method; no parity oracle), unchanged from #38.

The label is upgradeable to full parity-backed only after the maintainer's live PyRosetta
cross-check (item 3 full) confirms the frozen logs reproduce and quantifies tmol-vs-live
agreement on the native pose. Until then, "parity-backed in rank and class" is the honest
ceiling the in-container evidence supports.
