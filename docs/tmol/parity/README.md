# tmol energy parity (TMOL-PARITY, #39)

Numerical-parity evidence for the tmol energy provider
(`frustrapy/backends/atomic_tmol_engine.py`) and the `atomic-tmol` backend, against the
three reference points the mission defines. tmol reimplements Rosetta `beta_nov2016_cart`;
Rosetta production is `ref2015`, so the energy-function gap is quantified here, not assumed
zero.

## Reports (regenerable)

| file | what | runs |
|---|---|---|
| `tmol_oracle_parity.md` | evaluator vs tmol's own shipped 1ubq oracle (scope item 1) | in-container, needs tmol |
| `tmol_golden_parity.md` | backend post-processor vs the AA golden fixture (scope item 2) | in-container, no tmol |
| `tmol_vs_rosetta_frozen.md` | tmol energies vs the frozen Rosetta `ResResE` logs on identical poses (scope item 3, in-container half): per-residue, per-term breakdown, engine-swap FrstIndex | in-container, needs tmol |
| `tmol_vs_pyrosetta.md` | live-PyRosetta cross-check (scope item 3, full) | maintainer, needs PyRosetta (SKIPPED here) |
| `LABEL_DECISION.md` | the parity label for the atomic configurational index (scope item 4) | decision record |

## Regenerate

The two scripts live in `frustrapy/backends/tmol_eval/`. The tmol checks need tmol
importable in the running interpreter (the published CPU wheel needs the JIT bridge
`TMOL_USE_JIT=1` + ninja; see `docs/tmol/PY_BACKEND_NOTES.md` section 6). The golden check
needs only the shipped reference logs.

```
# in-container parity (oracle + golden)
python -m frustrapy.backends.tmol_eval.validate_in_container --write-report --gate

# tmol-vs-Rosetta cross-check (frozen logs in-container; live PyRosetta SKIPPED here)
python -m frustrapy.backends.tmol_eval.validate_vs_pyrosetta --write-report --gate
```

The same gate thresholds are asserted by `tests/tmol/test_pyrosetta_parity.py` (the fast
lane runs the golden assertion and the clean-skip checks; the tmol and PyRosetta layers
skip unless their prerequisite is present). The frozen cross-check scores 50 poses and is
marked `slow`.

## The reference data

The shipped atomic reference (`/workspace/atomic_frustratometer_ref/example_output/`, not
committed; override with `ATOMIC_FRUST_REF`) provides, for the 1QYS/TOP7 demo, 50 complete
all-atom Rosetta-repacked decoy poses (`1.pdb` ..) each with a matching `ResResE` log
(`1.log` ..). tmol can score the complete poses; the logs hold Rosetta's `ref2015` per-pair
energies for the same coordinates. That pairing is what makes the in-container
tmol-vs-Rosetta cross-check possible with no live Rosetta. The shipped `native.pdb` is a
representative-atom-only structure (the reference scores it from the logs, not by
re-scoring), so tmol cannot build it; the engine-swap FrstIndex check uses a complete decoy
pose as its pseudo-native, which keeps the comparison a clean energy-engine swap.
