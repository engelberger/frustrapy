# AA-INTEGRATE: registering backend="atomic" and flowing it through the public API

This documents the AA-INTEGRATE deliverable: `frustrapy/backends/atomic.py`
(`AtomicBackend`), which makes the all-atom Rosetta engine a first-class, selectable
backend that flows through the existing public API exactly like `lammps` and
`native`, with the same on-disk output contract, so the canonical tables, the 5 A
density, and the Plotly/py3Dmol visualization all work unchanged.

The engine half (AA-ENGINE), the AWSEM-format writer (AA-OUTPUT, parity-gated vs the
golden), and the mode -> decoy-strategy map (AA-MODES) were the prior missions; this
one wires them behind the `FrustrationBackend` seam.

## API surface added

- `frustrapy.backends.AtomicBackend` (`name="atomic"`, `requires_lammps_prep=False`),
  registered in `frustrapy.backends._REGISTRY`. `get_backend("atomic")` resolves it;
  `available_backends()` lists `['atomic', 'lammps', 'native']`. `DEFAULT_BACKEND`
  stays `"lammps"`.
- `backend="atomic"` is accepted by `calculate_frustration`, `dir_frustration`, and
  `dynamic_frustration` (the last gained a `backend` parameter here; the first two
  already had it). The 4-tuple return contract is preserved.
- `frustrapy.backends.atomic.score_sequences(...)`: the parallel decoy loop, with an
  injectable scorer (for testing without Rosetta).
- `frustrapy.backends.atomic.AtomicOptions`: per-run options resolved from optional
  `atomic_*` calculator attributes / `FRUSTRAPY_ATOMIC_*` env vars (decoy count, seed,
  seq_sep, scheme, repeats, n_procs, distance cutoff, per-site/per-contact counts).

## The `requires_lammps_prep` seam (folded #29)

`FrustrationBackend.requires_lammps_prep` (new `ClassVar`, default `True`) gates
whether `FrustrationCalculator._prepare_calculation_files` runs the
`PdbCoords2Lammps.sh` deck prep before `compute_energies`. `lammps` and `native`
consume that deck, so they keep `True` and run byte-identically (verified: 1CRN all
three modes via `tests/test_anchor.py` and `tests/test_native_parity.py`). The atomic
backend reads only the cleaned `{base}.pdb` and the `{base}.pdb_equivalences.txt` map
(both written by the calculator regardless of backend, in `_create_pdb_object`), so it
sets `False` and the calculator skips the subprocess prep entirely. The shared
`process_results` / `compute_density` post-processing is unaffected.

## Parallel safety (G3)

The decoy loop is the expensive axis. `score_sequences` runs it under the package's
SHARED core budget (`frustrapy/utils/concurrency.py`): one pool, sized
`min(requested, cores, n_tasks)` via `resolve_pool_size`, with a fork-safe start
method. When the backend already runs inside an outer pool worker
(`multiprocessing.parent_process()` is not `None`, e.g. `dir_frustration(n_procs=K)`
or the mutation scan), `_resolve_decoy_workers` returns 1 so the nested pools never
multiply past the budget (`outer * 1 <= cores`), mirroring `native._resolve_native_threads`.
The native pose energy is computed ONCE per structure and reused across every contact
(amortized, mirroring the prep-amortization lever).

## Multi-chain (G2)

The atomic geometry (`atomic_post.load_contact_geometry`) walks residues in PDB order
per chain, producing `cid_list` keys `chain+resnum` in the same global order the
calculator-level `helpers.pdb_equivalences` writes the equivalences file. The shared
`renum_files` maps the `.dat` residue indices back to `ChainRes` by global position,
so `ChainRes1`/`ChainRes2`, cross-chain contacts, and per-chain numbering come out
correct with no atomic special-casing. `tests/test_atomic_backend.py::test_multichain_chainres_through_renum`
proves this on a synthetic two-chain structure (both chains present in the ChainRes
columns, a genuine cross-chain contact, the canonical 14-column table). The shipped
parity fixture (1QYS) is a monomer.

## What is maintainer-gated (license / PyRosetta, not runnable in this container)

- The actual energies: `compute_native_pair_energies` and `compute_decoy_pair_energies`
  drive PyRosetta (ref2015 FastRelax + RestrictToRepacking, SimpleThreadingMover for
  decoys). PyRosetta is lazily imported and not installed here; a real atomic run
  raises a clear `ImportError` at compute time, never at import time. `import frustrapy`
  and the `lammps` default never import PyRosetta.
- The end-to-end multi-chain Rosetta run: `compute_decoy_pair_energies` threads from
  `start_position="1A"` by default; multi-chain threading / per-chain Neighborhood
  repacking is a maintainer adjustment (see `AA_MAINTAINER_RUNBOOK.md`). The chain
  PLUMBING and column mapping are validated here without Rosetta.
- Experimental modes: only `configurational` is parity-backed (the reference's
  permutation scheme, gated bit-for-bit against `docs/atomic/golden/` in
  `tests/test_atomic_post.py`). `mutational` and `singleresidue` are extensions beyond
  the published method (no atomic reference, no parity oracle); their decoy generation
  and per-group aggregation are implemented and unit-tested, but their Rosetta scoring
  is unvalidated (see `AA_MODES.md`).

## Tests added

`tests/test_atomic_backend.py` (fast lane, no Rosetta, no external reference data):
registry + lazy import + `requires_lammps_prep` flags; the clear PyRosetta error at
compute time; `_resolve_decoy_workers` budget logic; `score_sequences` serial==parallel
with an injected picklable scorer; multi-chain ChainRes through `renum_files`; a full
`calculate_frustration(backend="atomic")` / `dir_frustration(backend="atomic")` run
with Rosetta mocked (4-tuple, tables, density, and the plots consuming the atomic
output dir); and the absence of LAMMPS deck artifacts on the atomic path.
