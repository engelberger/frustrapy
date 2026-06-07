# Parallelism inventory (P1)

Engineering map of every place FrustraPy spawns processes, threads, or
subprocesses, how each one is sized, and every way they nest. This is the
audit baseline for the parallelization-safety work; the user-facing summary of
the resulting limits lives separately (see "Parallelism and resource limits").

Evidence is `file:line` against the package tree at the time of writing.

## 1. Process pools (the things that fan out)

| # | Construct | Location | Pool / executor | Worker count (sizing) |
|---|-----------|----------|-----------------|-----------------------|
| 1 | `mutate_res_scan_parallel` (flattened residue x amino-acid scan) | `analysis/mutations.py:512`, pool at `:613` | `multiprocessing.Pool` (default `fork` on Linux), `imap_unordered(_process_amino_acid, ...)` | `_resolve_pool_size(n_cpus, len(args_list), cpu_count())` = `max(1, min(n_cpus or cores, cores, n_tasks))`, where `n_tasks = n_residues * 20` (`:593`) |
| 2 | `mutate_res_parallel` (legacy single-residue 20-task pool) | `analysis/mutations.py` | `multiprocessing.Pool` | same `_resolve_pool_size` helper; superseded by #1 for the live singleresidue path but still part of the public API |
| 3 | `dir_frustration(n_procs=K)` (batch over structures) | `analysis/frustration.py:360` | `concurrent.futures.ProcessPoolExecutor(max_workers=n_procs_eff)` | `n_procs_eff = max(1, min(n_procs, len(order_list), cores))` (`:335-338`); each structure gets `inner_cpus = max(1, cores // n_procs_eff)` passed as `n_cpus` (`:345-356`) |
| 4 | `dynamic_frustration(n_procs=K)` (trajectory frames) | `analysis/frustration.py:468` | none of its own — delegates to `dir_frustration` | inherits #3 sizing |
| 5 | evolution per-structure precompute | `evolution/information_content.py:373` | `ProcessPoolExecutor(max_workers=n_workers)` | `n_workers = max(1, min(n_procs or cores, cores, len(jobs)))` (`:360-362`); each job is built with `n_cpus=1` (`:337,353`) so its inner pool degenerates to 1 |

`_resolve_pool_size` (`analysis/mutations.py:461`) is the only named sizing
helper. The other two pools (#3, #5) compute their own bound inline. There is
**no single shared budget object** today — each site independently calls
`multiprocessing.cpu_count()`. Process-level safety currently relies on the
outer pool correctly threading a reduced `n_cpus` down to the inner pool (see
nesting paths below); nothing structurally prevents a future caller from
sizing a pool against full `cores` while already inside another pool.

## 2. Subprocesses spawned inside each worker (all single-threaded)

Every unit of frustration work shells out to the AWSEM/LAMMPS toolchain. These
do not fan out further; they are one child process each:

- `LammpsRunner.run` -> `cp` the binary (`analysis/lammps_runner.py:79`) then
  run `lmp_serial_{seq_dist}_{OS}` (`:98`), argv list + stdin + `timeout=3600`,
  no `shell=True`.
- `PdbCoords2Lammps.sh` via `run_subprocess` (`analysis/frustration_calculator.py:459`,
  default `timeout=300`); the shell itself spawns `python3 PDBToCoordinates.py`
  then `python3 CoordinatesToWorkLammpsDataFile.py`.
- `GenerateChargeFile.pl` (`:655`, electrostatics path only).
- `GenerateVisualizations.pl` (`:944`, graphics path only).
- `RenumFiles.pl` / backbone completion via `subprocess.run` (`utils/helpers.py:86`,
  `timeout=300`).
- structure-fetch network call (`analysis/frustration_calculator.py:284`, `timeout=60`).

## 3. Backends that run inside the pool workers

- **threading** (default mutation backend): pure Python, no extra threads of
  its own.
- **pyrosetta** (`analysis/mutation_backends.py`): one PyRosetta init per worker
  process. PyRosetta and numpy/OpenBLAS may each spin native threads inside the
  worker (see thread caveat below).
- **gpu** backend: planned only (vision G1), not implemented. When added it must
  draw from the same concurrency budget as everything above; flagged here so the
  hook is not missed.

## 4. Native-thread caveat (BLAS / OpenMP)

No code pins BLAS/OpenMP thread counts (`grep -E
'OMP_NUM_THREADS|OPENBLAS_NUM_THREADS|MKL_NUM_THREADS|NUMEXPR'` over `frustrapy/`
returns nothing). numpy/pandas (and PyRosetta) can each start up to `cores`
native threads **per process**. So even when the process count is correctly
bounded to `cores`, the live *thread* count can reach `cores^2`. This is the
primary residual risk for the bounding work (P2).

## 5. Start method

No `set_start_method` / `get_context` anywhere — pools use the platform default
(`fork` on Linux). Forking a process that has already started BLAS/OpenMP
threads is unsafe; a `spawn`/`forkserver` context is the standard fix where
fork-after-threads can occur (P2).

## 6. Nesting paths (exact call chains)

```
PATH A  batch x singleresidue  (the classic fork-bomb shape)
  dir_frustration(n_procs=K)                      frustration.py:360  [ProcessPoolExecutor K]
    -> _dir_frustration_worker
      -> calculate_frustration(n_cpus=inner_cpus) frustration.py:356  (inner_cpus = cores // K)
        -> FrustrationCalculator(n_cpus)
          -> _generate_singleresidue_analysis      frustration_calculator.py:854
            -> mutate_res_scan_parallel(n_cpus)    mutations.py:512   [Pool min(inner_cpus, cores, tasks)]
  processes ~= K * inner_cpus = K * (cores // K) <= cores   -> BOUNDED today (via inner_cpus threading)
  threads   ~= (K * inner_cpus) * up-to-cores BLAS          -> UNBOUNDED (cores^2)  <- P2

PATH B  evolution
  analyze_family(n_procs=K)
    -> information_content precompute              information_content.py:373  [ProcessPoolExecutor n_workers<=cores]
      -> _evo_frustration_worker
        -> calculate_frustration(n_cpus=1)         information_content.py:337,353
          -> singleresidue scan pool = 1
  processes <= n_workers <= cores  -> BOUNDED.  threads: same cores^2 caveat.

PATH C  direct singleresidue
  calculate_frustration(mode='singleresidue', n_cpus=None)
    -> mutate_res_scan_parallel(n_cpus=None)       [Pool min(cores, cores, tasks) = up to cores]
  single level -> BOUNDED in processes.  threads: same caveat.

PATH D  trajectory
  dynamic_frustration(n_procs=K) -> dir_frustration(n_procs=K)  -> identical to PATH A.
```

## 7. Summary of what is and is not bounded today

- Process counts on the live paths (A-D) are bounded to `cores` **only because**
  the outer pool threads a reduced `n_cpus`/`inner_cpus` into the inner pool.
  There is no shared budget enforcing this; it is convention per call site.
- Native thread counts are **not** bounded — `cores^2` threads are reachable on
  every path (no BLAS/OpenMP pinning).
- Start method is `fork`, unsafe after threads are started.

P2 closes these by introducing one shared concurrency budget that every pool
draws from, pinning per-worker BLAS/OpenMP threads to 1, and using a safe start
method; P3 verifies the live peak process/thread count stays at or below
`cores + slack` under PATH A and PATH B.
