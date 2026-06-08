"""tmol energy-parity validation harnesses (TMOL-PARITY, #39).

This subpackage holds the reproducible scripts that validate the numerical parity of
the tmol energy provider (:mod:`frustrapy.backends.atomic_tmol_engine`) and the tmol
atomic backend (:class:`frustrapy.backends.atomic_tmol.AtomicTmolBackend`) against the
three reference points the mission defines:

* ``validate_in_container`` -- the two pure in-container parity checks that need no
  license: the evaluator vs tmol's own shipped 1ubq oracle (``docs/tmol/oracle/``,
  scope item 1), and the atomic backend's post-processor output vs the AA golden
  fixture (``docs/atomic/golden/``, scope item 2). Both run here, today, for real.

* ``validate_vs_pyrosetta`` -- the Rosetta/PyRosetta cross-check (scope items 3-4).
  IN-CONTAINER it runs a real cross-check of tmol energies against the FROZEN Rosetta
  ``ResResE`` logs shipped with the atomic reference (the shipped decoy ``*.pdb`` are
  complete all-atom poses tmol can score, and the matching ``*.log`` hold Rosetta's
  per-pair ref2015 energies for the same poses), which quantifies the
  beta_nov2016-vs-ref2015 energy-function gap. The LIVE PyRosetta path (re-scoring the
  poses with a fresh Rosetta) is the maintainer half; with no PyRosetta installed it
  writes a clear SKIPPED marker rather than failing.

Neither script imports tmol or PyRosetta at module import; both are lazy and the
checks that need them skip cleanly when they are absent.
"""
