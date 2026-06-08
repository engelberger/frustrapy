# tmol backend post-processor vs the AA golden fixture (scope item 2)

The `atomic-tmol` backend reuses `frustrapy.backends.atomic_post.write_tertiary_frustration` verbatim. This check drives that exact writer from an `EngineResult` built from the shipped reference `ResResE` logs and diffs the per-contact `FrstIndex` against the committed golden `docs/atomic/golden/tertiary_frustration.dat`. It isolates the POST-PROCESSING half the tmol backend shares with the AA lane (the energy-engine gap is the separate `tmol_vs_rosetta_frozen.md` cross-check). `[VERIFIED EMPIRICALLY]`

- reference logs: `/workspace/atomic_frustratometer_ref/example_output` (N=50 decoys)
- contacts compared: **328** (only-in-golden=0, only-in-regenerated=0)

| metric | value |
|---|---|
| Spearman | 0.999999 |
| Pearson R^2 | 1.000000 |
| max\|delta\| | 4.988e-04 |
| RMSE | 2.924e-04 |
| class-agreement | 100.00% |

**Verdict: PASS** (the shared post-processor reproduces the golden FrstIndex bit-for-bit at print precision over the identical contact set; the tmol backend calls the same writer).
