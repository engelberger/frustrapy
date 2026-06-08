# tmol energies vs live PyRosetta (maintainer cross-check)

**SKIPPED**: PyRosetta is not installed in this container (license-gated). This is the maintainer half; run it on a licensed Rosetta host.

What the maintainer run does when PyRosetta is present:

1. Re-score the shipped complete poses (`1.pdb` ..) with a fresh Rosetta `ref2015` score function (`atomic_engine._extract_pair_energies`).
2. Diff live-PyRosetta vs the frozen `*.log` per-residue energies -- this closes the reproducibility loop (do the shipped logs match a fresh run?).
3. Diff tmol vs live-PyRosetta per-residue energies -- the direct beta_nov2016-vs-ref2015 cross-check on the native pose too (which the frozen fixture lacks complete atoms for).

Reported per pose: Spearman, R^2, max|delta|, RMSE, class-agreement, and the per-term breakdown. Run on a licensed host with::

    python -m frustrapy.backends.tmol_eval.validate_vs_pyrosetta --write-report

and commit the regenerated table. The in-container frozen-log cross-check (`tmol_vs_rosetta_frozen.md`) already quantifies the same energy-function gap from the shipped Rosetta outputs; the live run confirms those outputs reproduce and extends the comparison to the native pose. `[UNTESTED -- awaits the maintainer's licensed PyRosetta run]`
