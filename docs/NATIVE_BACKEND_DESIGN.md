# G1.2 / G1.3 — Native C++ CPU core + CUDA backend: design

Status: N1 (design + buildable skeleton). N2 (CPU reference core) and N3 (CUDA) follow.
Branch: `dev_native_backend`. Parity-before-speed: every step gates on numbers vs the
`lammps` reference, not on speed.

This document is the design for a native compute core that plugs in behind the
`FrustrationBackend` interface (G1.0, `frustrapy/backends/base.py`). The reference
backend (`lammps`) shells out to the precompiled AWSEM/LAMMPS binary; the native
backend reimplements the same energy reductions in memory-safe C++ (CPU now, CUDA
later) and is parity-gated against that reference.

## 1. Toolchain versions (fetched 2026-06-07)

Pinning the docs the design is built against so kernels are not coded from memory.

| Component | Version | Source |
|---|---|---|
| CUDA C++ Programming Guide | v13.3 (docs updated 2026-05-27) | docs.nvidia.com/cuda/cuda-c-programming-guide |
| CUDA C++ Best Practices Guide | v13.3 | docs.nvidia.com/cuda/cuda-c-best-practices-guide |
| nanobind | 2.12.0 | nanobind.readthedocs.io |
| scikit-build-core | 0.12.x | scikit-build-core.readthedocs.io |
| CMake (min) | 3.15; 3.28 in container | cmake.org |
| C++ Core Guidelines | living doc, 2025-07-08 (v0.8) | isocpp.github.io/CppCoreGuidelines |
| Compiler in container | g++ 13.3 (C++23-capable), Python 3.12 | local |

CUDA Best Practices priority order, applied in section 5: (1) coalesce global memory
access, (2) minimize host/device transfer, (3) profile to find hotspots, (4) use
effective bandwidth as the metric, then shared-memory / bank-conflict tuning.

## 2. What the native core must reproduce

The native energy, the ~1000-decoy ensemble mean and standard deviation, and the index
are computed today inside the AWSEM/LAMMPS binary, which writes
`tertiary_frustration.dat`. The Python layer parses that file
(`frustrapy/utils/helpers.py:206-258`) and derives `FrstState` from `FrstIndex` via the
contact cutoffs. The native core replaces only the energy/decoy computation — the part
that produces `tertiary_frustration.dat` — and reuses the existing parser, threshold
classification, and 5 A density post-processing unchanged (they are concrete methods on
`FrustrationBackend`).

The sign convention is load-bearing and is the #1 reimplementation hazard:

```
FrstIndex = (mean(E_decoy) - E_native) / sd(E_decoy)
```

A favorable (more negative) native energy gives a positive index (minimally frustrated).
This is the negation of the paper's published numerator; copying the paper verbatim
inverts every index. The native core must match the binary's sign, not the paper's.

### 2.1 The AWSEM contact energy model

Parameters ship in `frustrapy/core/scripts/AWSEMFiles/`:

- `fix_backbone_coeff.data` — the term blocks and their constants:
  - `[Water]`: epsilon, two radial wells (direct 4.5-6.5 A, water/protein-mediated
    6.5-9.5 A), density switching parameters `rho_0`, well width `eta`.
  - `[Burial]`: `kappa = 4.0`, three density bins (0-3, 3-6, 6-9 contacts).
  - `[Tertiary_Frustratometer]`: contact cutoff 9.5 A, `1000` decoys, seed `1`, mode
    keyword (`configurational` | `mutational` | `singleresidue`, swapped in by
    `frustration_calculator.py`).
- `gamma.dat` — the residue-pair gamma tables: `gamma_direct[20][20]`,
  `gamma_water[20][20]`, `gamma_protein[20][20]` (the file lays out direct then the two
  mediated tables; exact row ordering is a parity risk, section 6).
- `burial_gamma.dat` — `burial_gamma[20][3]`, one row per residue type, one column per
  density bin.

The per-contact energy of a residue pair (i, j) with separation |i - j| >= seq_dist is
the sum of three terms (AWSEM, Davtyan et al. 2012; frustratometer fix in LAMMPS):

1. **V_direct** — short-range contact. A radial well `theta(r_ij; 4.5, 6.5)` between the
   CB atoms (CA for glycine) weighted by `gamma_direct[a_i][a_j]`.
2. **V_water / V_mediated** — medium-range (6.5-9.5 A) contact whose weight interpolates
   between `gamma_water` and `gamma_protein` by a local-density sigmoid
   `sigma(rho_i) sigma(rho_j)`; high local density -> protein-mediated, low -> water.
3. **V_burial** — per-residue burial: for each residue, `kappa`-weighted sum over the
   three density bins of `burial_gamma[a_i][bin]` against a smooth bin membership
   function of the residue's local density `rho_i`.

Local density `rho_i = sum_{k != i} theta(r_ik)` is the smoothed neighbor count used by
both the mediated switch and the burial term. The 5 A spatial density that the Python
layer reports separately (`_calculate_frustration_density`) is a different quantity and
stays in Python.

### 2.2 The three modes = three decoy ensembles

One energy model, one Z-score, three ways to build the decoy ensemble (selected by the
mode keyword). The native core computes the same per-mode decoy statistics:

| Mode | Probed unit | Decoys randomize | Geometry |
|---|---|---|---|
| configurational | contact i-j | identities of i,j + distance r_ij + densities | varied |
| mutational | contact i-j | identities of i,j only | frozen native |
| singleresidue | site i | identity at site i only | neighbors native |

Mutational is exhaustive over the 20x20 identity pairs at fixed geometry, so it is the
most reproducible and the largest GPU win (section 5). Configurational additionally
resamples distance and density from the structure's own distribution.

## 3. Data layout — structure of arrays (SoA)

All hot data is SoA so the CPU vectorizes and the GPU coalesces. One structure per
chain-complex, built once on the Python side from the prepared PDB and passed zero-copy:

```
struct Structure {            // all arrays length N_res, parallel
  span<const double> ca_x, ca_y, ca_z;   // CA coordinates
  span<const double> cb_x, cb_y, cb_z;   // CB (CA for GLY)
  span<const int32>  res_type;           // 0..19, gamma.dat ordering
  span<const int32>  chain_id;
  span<const int32>  res_seqid;          // residue number for seq-separation test
};

struct AwsemParams {          // immutable, shared across all decoys
  mdspan<const double, extents<20,20>> gamma_direct, gamma_water, gamma_protein;
  mdspan<const double, extents<20,3>>  burial_gamma;
  WellParams direct, mediated;           // r_min, r_max, eta
  BurialParams burial;                   // kappa, three bins
  int seq_dist; int n_decoys; uint64 seed;
};
```

Per-contact outputs are a parallel set of arrays (SoA again), one entry per contact that
passes the 9.5 A / seq_dist filter, matching `tertiary_frustration.dat` columns:

```
struct ContactResult {        // length N_contacts
  vector<int32>  res_i, res_j;
  vector<double> native_energy, decoy_mean, decoy_sd, frst_index;
  vector<double> rho_i, rho_j;           // local densities (for welltype)
  vector<double> r_ij;                   // CB-CB distance (welltype: short/long/water)
};
```

`std::mdspan` (C++23) is used for the fixed 20x20 / 20x3 parameter tables where it is
available; `std::span` (C++20, available in g++ 13) is the portable fallback for the
ragged per-residue arrays. No raw pointer arithmetic crosses a function boundary
(C++ Core Guidelines I.13, F.24).

## 4. Kernel decomposition

The computation factors into stages that map cleanly onto both a vectorized CPU loop and
a CUDA grid:

1. **K_density** — `rho_i` for all i. O(N^2) pairwise `theta(r_ik)`; precomputed once,
   reused by mediated and burial. On GPU: one thread per residue, reduction over k; or a
   tiled N x N pass with shared-memory blocking.
2. **K_contacts** — build the contact list (pairs within 9.5 A and seq_dist). On GPU:
   pair-per-thread predicate + stream compaction.
3. **K_native** — per contact, `V_direct + V_mediated`; per residue, `V_burial`. This is
   the native energy. One thread per contact (direct/mediated) and one per residue
   (burial), reduced into the contact totals.
4. **K_decoy** — per contact, build the decoy ensemble and reduce to mean and sd.
   - mutational: loop the 20x20 identity grid at frozen geometry -> 400 evaluations per
     contact, fully deterministic, embarrassingly parallel (the (contact x 400) axis is
     the GPU batch axis).
   - configurational: `n_decoys` samples, each resampling identities + r_ij + densities
     from the structure's empirical distributions (RNG per decoy).
   - singleresidue: per site, vary only the identity at i (20 evaluations) with native
     neighbors.
   Mean and sd are an online (Welford) reduction to avoid a second pass and catastrophic
   cancellation.
5. **K_index** — `frst_index = (decoy_mean - native_energy) / decoy_sd` (the sign above),
   elementwise.

The 5 A density / proportion summary is NOT in the native core — it stays in
`FrustrationCalculator._calculate_frustration_density` and is shared via the base-class
`compute_density`.

### GPU batch axes (N3)

The decoy axis is the parallel win. For mutational mode the work is
`(N_contacts x 400)` independent energy evaluations sharing the frozen geometry and the
gamma tables (which live in constant/shared memory); for configurational it is
`(N_contacts x n_decoys)`. K_decoy is the kernel to fuse and tile: load a contact's
geometry once into registers/shared memory, sweep its decoy batch, Welford-reduce in
shared memory, write one (mean, sd) pair to global memory — coalesced, minimal transfer
(Best Practices priorities 1 and 2).

## 5. Python binding surface (nanobind, zero-copy)

The compiled module is a self-contained top-level extension `frustrapy_native` built by
the `native/` subproject. nanobind exchanges NumPy arrays zero-copy via the buffer /
DLPack protocol; inputs are constrained to contiguous CPU arrays of known dtype:

```cpp
// input: contiguous CPU arrays, validated at the boundary
using Coords = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using ResType = nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

nb::dict compute_frustration(
    Coords ca, Coords cb, ResType res_type, ResType chain_id, ResType res_seqid,
    GammaTables gammas, WellParams wells, int seq_dist, int n_decoys,
    const std::string& mode, uint64_t seed);
```

The result is a `dict` of NumPy arrays (`native_energy`, `decoy_energy`, `sd_energy`,
`frst_index`, `res_i`, `res_j`, `rho_i`, `rho_j`, `r_ij`). Output arrays own
C++-allocated memory through an `nb::capsule` whose deleter frees it when the NumPy array
expires (the ownership rule from the nanobind ndarray docs: never return a view of
freed storage; always attach an owner). No array is returned with
`rv_policy::reference` to transient storage.

The Python wrapper (`frustrapy/backends/native.py`, `NativeBackend`) is responsible for:
extracting SoA arrays from the prepared `Pdb`, loading the gamma/burial parameters,
calling `frustrapy_native.compute_frustration`, and writing `tertiary_frustration.dat`
in the binary's exact column layout so the existing parser
(`process_results`) and density (`compute_density`) run unchanged. This keeps the seam
narrow: the native core produces the same intermediate file; everything downstream is
the shared, parity-gated post-processing.

## 6. Memory-safety conventions (C++ Core Guidelines, 2025-07-08)

- **No naked `new`/`delete`; no owning raw pointers** (R.11). Buffers are
  `std::vector`; ownership is `std::unique_ptr` where a single owner is needed (R.12),
  `shared_ptr` only if genuinely shared (R.13). RAII for every resource (P.8).
- **No array-as-pointer interfaces** (I.13): pass `std::span` / `std::mdspan`, never
  `(T*, size_t)`. No pointer arithmetic across a function boundary (F.24).
- **Bounds-checked in debug**: index through `span`/`mdspan` with `at()`-style checks
  under a debug build; rely on `-D_GLIBCXX_ASSERTIONS` and the sanitizers in CI.
- **Static over runtime checks** (P.4-P.6): fixed-extent `mdspan<...,20,20>` for the
  gamma tables so the 20x20 shape is a compile-time guarantee.
- **No UB**: no uninitialized reads, no signed overflow in indices (use `std::ptrdiff_t`
  / `int64_t` for large pair counts), no aliasing violations. Decoy RNG is an explicit
  seeded `std::mt19937_64` (seed from `[Tertiary_Frustratometer]`), never a global.
- **Sanitizers gate N2**: the test build compiles `-fsanitize=address,undefined` and the
  parity suite must run ASan/UBSan-clean before any speed work.
- **The gamma-table row ordering is the top parity risk** — the 20-letter index order in
  `gamma.dat` / `burial_gamma.dat` must match the order the LAMMPS fix assigns to residue
  types. N2 verifies this by reproducing a single known contact's native energy before
  trusting the full reduction. This is checked first, on one contact, by hand.

## 7. Build system — scikit-build-core + nanobind, conditional CUDA

The `native/` subproject is self-contained so the main `frustrapy` package keeps its
hatchling pure-Python build untouched. `pip install ./native` produces the extension.

- `native/pyproject.toml`: `build-backend = "scikit_build_core.build"`,
  `requires = ["scikit-build-core>=0.10", "nanobind>=2.0"]`.
- `native/CMakeLists.txt`:
  - `find_package(Python ... Development.Module)`, `find_package(nanobind CONFIG)`
    (located via `python -m nanobind --cmake_dir`).
  - `nanobind_add_module(_core STABLE_ABI NB_STATIC ...)`.
  - CUDA is optional and off by default: `option(FRUSTRAPY_NATIVE_CUDA "..." OFF)`. When
    on, `include(CheckLanguage); check_language(CUDA)`; if present,
    `enable_language(CUDA)` and compile `src/cuda/kernels.cu` with the CPU core. When
    absent (the container has no nvcc), the build proceeds CPU-only and the runtime
    `has_cuda()` returns `False` — graceful fallback, never a hard failure.
  - C++23 requested (`CMAKE_CXX_STANDARD 23`), `_GLIBCXX_ASSERTIONS` on; a
    `FRUSTRAPY_NATIVE_SANITIZE` option adds `-fsanitize=address,undefined` for the N2
    parity build.

The runtime backend import is lazy and graceful: `NativeBackend.compute_energies` imports
`frustrapy_native` only when invoked, and raises a clear, actionable error if the
extension was never built — so installing `frustrapy` without a compiler still works and
the `lammps` default is unaffected.

## 8. Parity plan (gates N2)

1. Reproduce one known contact's native energy by hand (resolves the gamma ordering).
2. Native energy per contact vs `tertiary_frustration.dat` column: max abs error below an
   energy tolerance on a PDB panel (1crn + a small multi-chain case) x 3 modes.
3. `FrstIndex` Spearman correlation >= 0.99 vs the `lammps` reference, all modes; sign
   agreement 100% (no inverted classes); contact-class agreement reported.
4. Sanitizers (ASan/UBSan) clean on the parity build.
5. The native backend is wired as a selectable `backend="native"`; the on-disk output
   contract (the 14/8-column tables) is identical to the `lammps` path because it shares
   the parser and density code.

Only after these pass does N3 port K_density / K_native / K_decoy to CUDA and measure
speed. The maintainer runs the CUDA harness on Colab / the cluster; no GPU timing is
fabricated here.

## 9. Status of this skeleton (N1)

`native/` builds on CPU and imports as `frustrapy_native`. The reduction entry points are
declared with the final binding signatures but are **stubs** (they raise
`std::runtime_error("native energy core not implemented (N2)")`). One real, testable
kernel — `contact_map`, a pure-geometry CB-CB neighbor count within a cutoff — is
implemented end to end to exercise the SoA + zero-copy ndarray path without claiming any
energy parity. `has_cuda()` reports whether the optional CUDA path was compiled (False in
this container). The `native` backend is registered and selectable but raises the
not-implemented error until N2 lands the reductions.
