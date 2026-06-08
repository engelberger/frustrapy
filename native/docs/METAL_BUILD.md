# Building and testing the Metal (Apple GPU) backend

This is the host-Mac runbook for the optional Metal path of `frustrapy_native`
(mission G1.4). The Metal kernels are the Apple-GPU counterpart of the CUDA kernels
(`src/cuda/kernels.cu`): the same AWSEM math, the same decoy generation, the same
`FrstIndex = (DecoyEnergy - NativeEnergy) / SDEnergy` sign convention. They were
**authored in a Linux container with no Apple GPU and have never been compiled or run**;
the maintainer builds, runs, and fixes them on a real Mac. Everything below is the
recipe plus the known-unknowns to resolve on hardware.

The default build is CPU-only and unaffected: `pip install ./native` produces a core
where `has_metal()` is `False`, the CPU parity gate passes, and the Metal parity lane
(`tests/test_native_parity.py::test_metal_matches_lammps_1crn`) skips cleanly.

## Requirements (host Mac)

- Apple Silicon Mac (the Metal GPU), macOS 13+ (15.5 SDK is what this targets).
- Xcode command line tools: `clang++`, `cmake`, and `xcrun metal` / `xcrun metallib`.
- **metal-cpp** — Apple's single-header C++ binding for Metal (pure C++, *not* MLX, not
  Objective-C). Download the archive matching your macOS SDK from
  <https://developer.apple.com/metal/cpp/> and unzip it; the include root is the
  directory that contains `Metal/Metal.hpp`, `Foundation/Foundation.hpp`, and
  `QuartzCore/QuartzCore.hpp`.
- nanobind >= 2.0, scikit-build-core >= 0.10, NumPy (same as the CPU build).

## Build

```
pip install ./native \
  -C cmake.define.FRUSTRAPY_NATIVE_METAL=ON \
  -C cmake.define.FRUSTRAPY_METAL_CPP_DIR=/path/to/metal-cpp
```

`FRUSTRAPY_METAL_CPP_DIR` must point at the metal-cpp include root (the dir with
`Metal/Metal.hpp`); CMake fails fast with a clear message if it is unset or wrong. The
build:

1. compiles `src/metal/kernels.metal` to `kernels.metallib` via
   `xcrun -sdk macosx metal` then `xcrun -sdk macosx metallib` (a CMake custom command);
2. compiles `src/metal/dispatch.cpp` (pure C++ metal-cpp; no OBJCXX needed) into the
   `_core` extension with `FRUSTRAPY_NATIVE_METAL` defined and links `-framework Metal
   -framework Foundation -framework QuartzCore`;
3. installs `kernels.metallib` next to the extension in the `frustrapy_native` package.

At runtime `dispatch.cpp` locates the metallib in this order: `$FRUSTRAPY_METALLIB`
(explicit override), the `kernels.metallib` sitting next to the loaded extension (the
installed default, found via `dladdr`), the compile-time `FRUSTRAPY_METALLIB_PATH` if
set, then the process default library. If none load it raises an actionable error.

CUDA and Metal are independent options; the CPU core is always present. `use_cuda` and
`use_metal` are mutually exclusive at call time.

## Run the parity test

```
# from the repo root, after building frustrapy + frustrapy_native (Metal)
pip install -e .
pytest tests/test_native_parity.py -m slow -k metal -v
```

`test_metal_matches_lammps_1crn` runs 1CRN through the `lammps` reference and the native
backend with `FRUSTRAPY_NATIVE_USE_METAL=1`, for all three modes, and asserts:

- row counts match,
- max per-column energy diff `<= 5e-2` (a float32 tolerance, looser than the bit-for-bit
  CPU gate — see precision notes below),
- `FrstIndex` Spearman `>= 0.99`,
- `FrstIndex` sign agreement on every non-near-zero contact (guards the #1
  reimplementation hazard, the Z-score sign inversion).

You can also drive the backend directly:

```python
import os
os.environ["FRUSTRAPY_NATIVE_USE_METAL"] = "1"
import frustrapy
frustrapy.calculate_frustration(pdb_file="1crn.pdb", mode="configurational",
                                backend="native", graphics=False)
```

or call the extension straight: `frustrapy_native.compute_frustration(..., use_metal=True)`.

## Precision: the float32 known-unknown (resolve on hardware)

**Apple GPUs have no IEEE `double` in MSL.** The CUDA path does every reduction in
`double`; the Metal kernels do the on-device arithmetic in `float`. The driver keeps the
*discrete* decisions exact (host-side, in `double`): the contact list, the glibc decoy
random-index stream, and the entire **configurational** decoy ensemble (computed
host-side in double exactly as the CUDA driver does — so configurational mode is the
least affected). Only these run on the GPU in float32:

- `k_density` — per-residue local density. **Highest-risk**: `rho` feeds the `tanh`
  water-sigma, so a float density shift propagates into every contact energy. Flagged
  `TODO(metal-precision)` in `kernels.metal`.
- `k_native_contacts` / `k_native_single` — native energies in float.
- `k_decoy_mut` / `k_decoy_single` — the decoy mean/sd threadgroup reductions sum ~1000
  float energies; the float accumulation loses low-order bits vs the double CUDA sum.

Expected impact: `FrstIndex` ranking (Spearman) and sign should hold — the test gates on
those — but the absolute energy columns will differ from the bit-for-bit CPU/LAMMPS
numbers by more than the `1.5e-3` CPU tolerance. The Metal lane therefore uses `5e-2`.
**Re-tighten or loosen this on real hardware** once measured; if it is wildly off, the
sign convention or a buffer-layout bug is the more likely cause than float rounding.

If float32 parity is insufficient, in increasing order of effort:

1. **Compensated (Kahan) summation in the threadgroup reductions.** The reduction loop
   in `k_decoy_mut` / `k_decoy_single` is the only place to change: carry a running
   compensation term per thread and through the tree reduction. Cheap, keeps the work on
   the GPU.
2. **Double-on-CPU reduction of the decoy mean/sd.** Have the decoy kernels write the
   per-decoy energies to a buffer (size `n_units * n_decoys` floats) instead of reducing
   in-kernel, read them back, and reduce mean/sd on the host in `double`. The expensive
   per-decoy energy eval stays on the GPU; only the cheap final reduction moves to the
   CPU. This is the recommended fallback for tight parity.
3. **Double-on-CPU density.** `rho` is already read back to the host
   (`b_rho->contents()`), so recomputing density in `double` on the host (reuse
   `Engine::rho()` / `local_density` from the CPU core) is a drop-in replacement for the
   `k_density` pass if the float density proves to be the dominant error source.

## Other known-unknowns for the host build

- **Threadgroup sizing.** The block-per-unit reductions (`k_decoy_mut`,
  `k_decoy_single`) dispatch exactly `kRTPB = 256` threads per threadgroup and the tree
  reduction assumes a power of two. `dispatch.cpp` checks
  `maxThreadsPerThreadgroup >= 256` (Apple GPUs allow 1024, so this is safe) and throws
  otherwise. If you change `kRTPB`, change `RTPB` in `kernels.metal` to match and keep it
  a power of two.
- **Struct layout coupling.** `MParams` is duplicated in `kernels.metal` and
  `dispatch.cpp` and must stay byte-identical (POD floats/ints, no arrays). It is passed
  via `setBytes`. If you add a field, add it to both in the same position.
- **Buffer indices.** The `setBuffer`/`setBytes` indices in `dispatch.cpp` must match the
  `[[buffer(k)]]` attributes in each kernel. They are listed per kernel in both files.
- **Storage mode.** Buffers use `MTL::ResourceStorageModeShared` (unified memory on
  Apple Silicon — `contents()` is valid after `waitUntilCompleted`). On an Intel Mac
  with a discrete GPU you would need `Managed` + an explicit synchronize; Apple Silicon
  is the target.
- **metallib discovery in a wheel.** The `dladdr`-relative lookup assumes
  `kernels.metallib` is installed next to `_core*.so`. If you repackage, set
  `FRUSTRAPY_METALLIB` to be safe.
- **`.cpp` vs `.mm`.** `dispatch.cpp` is pure C++ (metal-cpp is header-only C++), so no
  Objective-C toolchain is required. If you prefer ObjC++ for other reasons, rename to
  `dispatch.mm` and `enable_language(OBJCXX)` in CMake — the code is unchanged.
