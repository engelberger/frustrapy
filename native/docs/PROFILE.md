# Native CPU core: profiling and multicore scaling

Empirical profile of the native C++ frustration core (`native/src/core.cpp`) and the
before/after of parallelizing it across all cores. All numbers below are measured in
this dev container (14 logical cores, GCC 13.3, `-O3`, OpenMP via libgomp); none are
projected.

## Method and tooling

`perf`, `valgrind`, `gdb`, `py-spy`, and `/usr/bin/time` are not installed in this
container, so profiling used what is available and reproducible:

- **Per-phase wall timers compiled into the core**, gated on the environment variable
  `FRUSTRAPY_NATIVE_PROFILE=1` (a `PhaseTimer` RAII guard around `density`,
  `contact-list`, `rng-precompute`, and `energy`). This attributes time to the actual
  reduction phases rather than guessing.
- **`resource.getrusage`** for peak RSS (the `/usr/bin/time -v` substitute).
- **The cross-backend harness** (`native/bench/cross_backend.py`) for wall-clock and
  speedup, both end-to-end (`run_cross_backend_benchmark`) and isolated to the C++
  reduction (`benchmark_core_scaling`).

Representative proteins: 1CRN (46 residues), 1UBQ (76), and 3PGK (415, single chain,
fetched from the PDB for the large-protein measurements). seq_dist = 12, 1000 decoys.

## Where the time goes (1 thread, 3PGK, 415 residues)

Per-phase wall time from the compiled timers, serial:

| mode | density | contact-list | rng-precompute | energy | energy share |
|------|--------:|-------------:|---------------:|-------:|-------------:|
| configurational | 1.2 ms | 0.4 ms | 0.5 ms | 0.8 ms | ~28% (all sub-ms) |
| mutational | 1.2 ms | 0.4 ms | 26 ms | 25177 ms | 99.8% |
| singleresidue | 1.1 ms | -- | 1.1 ms | 1680 ms | 99.8% |

The **energy reduction is the bottleneck** in the two heavy modes: the per-contact
mutational decoy loop (`nc * nd` decoys, each O(n) over neighbours) and the
per-residue singleresidue decoy loop (`n * nd` decoys, each O(n)). It is over 99% of
the compute on a 415-residue protein. Density is O(n^2) but only ~1 ms here (a few ms
even at 500+ residues); configurational is dominated by nothing in particular and is
sub-millisecond end to end.

The RNG (`rng-precompute`) is cheap (tens of ms at most) because it only materializes
random indices; it stays serial to preserve the exact glibc decoy stream (see the
parity note below).

## What was parallelized

Energy reduction, density, and `local_density` are parallelized with OpenMP over the
independent work units (per residue for singleresidue, per contact for the contact
modes, per residue for density). The decoy random draws are materialized serially in
the exact reference loop order first, so the shared glibc RNG stream is untouched;
the expensive per-unit reductions then run in parallel writing to disjoint output
slots. Each per-unit decoy array is summed in fixed order, so the result is
**bit-identical for any thread count** (verified: `unit_i/j`, `native_energy`,
`decoy_energy`, `sd_energy`, `frst_index`, `rho` arrays equal at 1/2/4/7/14 threads,
all three modes, on 1UBQ and 3PGK).

## A regression found by profiling, and fixed

The first cut parallelized every region unconditionally. Profiling showed two cases
where that is *slower* than serial:

- **The first OpenMP region in a process pays a one-time thread-team startup of
  ~60 ms** (measured as a `density` phase of 61 ms on the very first parallel call vs
  ~1.4 ms on subsequent calls).
- **Configurational energy is O(1) per contact**, so the whole loop is sub-millisecond
  even for thousands of contacts; spawning a thread team to do it made it slower
  (3PGK configurational: 3.2 ms serial -> 8.3 ms at 14 threads before the fix).

Fix: **work-aware guards** on each parallel region (`kDensityMinPairs = 250000`,
`kEnergyMinOps = 5e6` in `core.cpp`). A region only goes parallel when its estimated
work clearly exceeds the team-startup overhead; otherwise it runs serial and never
pays the 60 ms startup. After the fix, configurational at 14 threads matches serial
(2.8 ms == 2.8 ms) and the heavy modes are unaffected. Small proteins stay serial
where parallelism would not pay (1CRN singleresidue, `n*nd*n = 2.1e6 < 5e6`, stays
serial: 72.8 ms vs 65.7 ms, no regression).

## Strong scaling (isolated C++ core, 3PGK, median of 3, warm pool)

`benchmark_core_scaling`, timing `compute_frustration` only:

| threads | mutational wall | speedup | singleresidue wall | speedup |
|--------:|----------------:|--------:|-------------------:|--------:|
| 1 | 17.04 s | 1.00x | 1.121 s | 1.00x |
| 2 | 8.72 s | 1.95x | 0.567 s | 1.98x |
| 4 | 4.30 s | 3.96x | 0.299 s | 3.75x |
| 7 | 2.54 s | 6.70x | 0.174 s | 6.46x |
| 14 | 2.17 s | **7.85x** | 0.149 s | **7.54x** |

Near-linear to ~7 threads; the 7->14 step is sub-linear because these are 14 *logical*
cores (SMT) and the reduction is memory-bandwidth-bound past the physical-core count.
Peak RSS ~250 MB (the per-contact decoy-index buffers, `nc*nd*2` int32, ~23 MB on
3PGK; the rest is numpy/import).

## End-to-end, all backends (3PGK, mutational, 1 repeat)

`run_cross_backend_benchmark` (full `calculate_frustration` per backend), FrstIndex
Spearman vs the LAMMPS reference:

| backend | threads | wall(s) | speedup vs serial | Spearman vs lammps |
|---------|--------:|--------:|------------------:|-------------------:|
| lammps (reference) | 1 | 16.73 | -- | 1.0000 |
| native CPU x1 (serial) | 1 | 17.99 | 1.00x | 1.0000 |
| native CPU x14 | 14 | 2.59 | 6.93x | 1.0000 |

The native multicore core is **6.45x faster than the LAMMPS reference** on this
protein in mutational mode, with bit-identical FrstIndex ranking. On a small protein
the per-run prep dominates and the win shrinks (1UBQ, mutational: lammps 1.69 s,
native serial 1.67 s, native x14 0.79 s = 2.1x end to end; the isolated core is 3.7x).

## IO note

The "per-run 27 MB binary copy" suspect applies to the **LAMMPS path only**: the
native backend does not run `lmp_serial`, so it does not copy the binary into the job
directory (confirmed: a native job dir contains `gamma.dat`, `burial_gamma.dat`,
`fix_backbone_coeff.data`, and the prepared structure, no `lmp_serial_*`). The
remaining per-run constant for native is the shared `PdbCoords2Lammps.sh` prep
(~0.4 s), which is outside the C++ core and is the dominant cost only for small
proteins; it is not addressed here.

## Thread-safety

No data races by construction: in every parallel region each iteration writes only its
own output slot (distinct `unit_i[i]`/contact index `c`), the per-iteration `decoys`
vector is loop-local, and all reads (`Engine` const methods, the `rho_` density, the
precomputed `dec_it`/`dec_jt`) are read-only and fully populated before the region.
The output vectors are pre-sized, so no thread reallocates.

The ASan/UBSan build (`-C cmake.define.FRUSTRAPY_NATIVE_SANITIZE=ON`) runs clean on the
14-thread compute for all three modes (preload `libasan.so` + `libstdc++.so.6` so the
`__cxa_throw` interceptor resolves; `ASAN_OPTIONS=detect_leaks=0` to ignore the
interpreter's own allocations). ASan does not detect data races; TSan was not run
because GCC's `libgomp` is not TSan-annotated and would report false positives at the
OpenMP barrier (the LLVM `archer` runtime would be needed). The race-freedom argument
above plus the bit-identical-across-thread-counts result are the evidence relied on.

## Composition with outer process parallelism

The inner thread count is resolved in `frustrapy/backends/native.py`
(`_resolve_native_threads`) so it composes with FrustraPy's outer process pools
without a `cores^2` blow-up: `FRUSTRAPY_NATIVE_THREADS` overrides everything; inside a
pool worker (`multiprocessing.parent_process()` is not `None`) the core uses 1 thread,
because the outer pool -- capped at `<= cores` by the shared concurrency budget --
already saturates the machine; in the main process (single structure) it uses the full
core budget. So `outer * inner <= cores` on every path.
