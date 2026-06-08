# tmol energies vs frozen Rosetta ResResE logs (in-container cross-check)

tmol evaluates `beta_nov2016_cart`; the shipped logs are Rosetta production `ref2015`. This diffs them on the IDENTICAL shipped all-atom poses (`{i}.pdb` scored by tmol vs `{i}.log` Rosetta per-pair energies), so it quantifies the energy-function gap with no live Rosetta. `[VERIFIED EMPIRICALLY]`

- reference: `/workspace/atomic_frustratometer_ref/example_output`
- poses scored by tmol: **50** (each complete all-atom)
- pseudo-native pose for the engine-swap index: **1.pdb** (the shipped `native.pdb` is representative-atom-only and cannot be built by tmol)

## (a) per-residue Function1 energy (the quantity that feeds FrstIndex)

Pooled over every residue of every pose:

| metric | value |
|---|---|
| Spearman | 0.9049 |
| Pearson R^2 | 0.6959 |
| max\|delta\| | 10.148 |
| RMSE | 1.670 |

Per-pose residue Spearman ranges 0.851..0.947.

## (b) per-term breakdown (pseudo-native pose) -- uniform vs term-local?

Weighted per-pair energy, Rosetta column(s) vs the matching tmol subterm(s):

| term | n pairs | max\|d\| | mean\|d\| | Rosetta sum | tmol sum | Spearman |
|---|---|---|---|---|---|---|
| fa_atr | 2094 | 0.122 | 0.0048 | -476.38 | -467.27 | 0.9664 |
| fa_rep | 2094 | 4.345 | 0.0058 | 360.95 | 349.37 | 0.9873 |
| fa_sol | 2094 | 0.875 | 0.0295 | 339.87 | 387.84 | 0.9753 |
| fa_elec | 2094 | 0.965 | 0.0223 | -132.41 | -151.80 | 0.9804 |
| lk_ball | 2094 | 1.668 | 0.0503 | -10.55 | 11.46 | 0.6467 |
| hbond | 2094 | 3.597 | 0.0343 | -9.31 | -77.81 | 0.3447 |

## (c) engine-swap FrstIndex parity (the label number)

Same post-processor, same pose set, two energy engines: the per-contact `FrstIndex` computed on tmol energies vs on the frozen Rosetta-log energies. This isolates the effect of the energy engine on the index a user acts on.

- contacts compared: **324** (pseudo-native 1.pdb, 49 decoy poses)

| metric | value |
|---|---|
| Spearman | 0.9168 |
| Pearson R^2 | 0.8026 |
| max\|delta\| | 2.270 |
| RMSE | 0.420 |
| class-agreement (-1/0.78) | 84.0% |

**In-container verdict: PASS** against the rank/class gate (not bit-exact; the energy functions differ by construction). See `LABEL_DECISION.md` for what this means for the label.
