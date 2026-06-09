// Optional CUDA forward path for the torch-free energy kernels.
//
// Declares the GPU twins of the CPU drivers in driver.hpp. They take the SAME resolved
// host inputs (AtomInput / PairInput / HBondPairInput / EnergyParams) the CPU path
// consumes, upload them once, run the forward kernels (single-point only), and return the
// identical PairEnergyResult shape, so a caller can pick CPU or CUDA from one set of
// prepared inputs. No torch at runtime: the device math is plain CUDA C++ that mirrors the
// parity-gated CPU kernel headers term-for-term (src/cuda/kernels.cu).
//
// hasCuda() is always available (header-only, macro-guarded) so the binding and harness
// can branch without a CUDA toolchain. The compute*CUDA symbols are defined only in the
// CUDA translation unit (kernels.cu), built when FRUSTRAMOL_TMOL_CUDA is set; the binding
// references them only after hasCuda() is true, so a CPU-only build never needs them.
// See native_tmol/NOTICE for attribution.

#pragma once

#include <vector>

#include "types.hpp"

namespace frustramol_tmol {

// True iff this build compiled the CUDA forward path. False in the default CPU-only
// build (and in the dev container, which has no nvcc), exactly like the AWSEM native
// core's has_cuda().
inline bool hasCuda() noexcept {
#ifdef FRUSTRAMOL_TMOL_CUDA
    return true;
#else
    return false;
#endif
}

// fa_ljatr / fa_ljrep / fa_lk / fa_elec per-residue-pair matrices on the GPU. Mirrors
// computePairEnergiesCPU; the result matches it to floating-point reduction order (the
// atomicAdd accumulation order differs, never the formula). Defined in kernels.cu.
PairEnergyResult computePairEnergiesCUDA(const std::vector<AtomInput>& atoms,
                                         const std::vector<PairInput>& pairs,
                                         const EnergyParams& params, int nBlocks);

// lk_ball_iso / lk_ball / lk_bridge / lk_bridge_uncpl per-residue-pair matrices on the
// GPU. Mirrors computeLkBallCPU (both polar/occluder directions per heavy-atom pair).
PairEnergyResult computeLkBallCUDA(const std::vector<AtomInput>& atoms,
                                   const std::vector<PairInput>& pairs,
                                   const EnergyParams& params, int nBlocks);

// hbond per-residue-pair matrix on the GPU. Mirrors computeHbondCPU (one donor-H /
// acceptor pair per thread, the < 5 count-pair exclusion applied on device).
PairEnergyResult computeHbondCUDA(const std::vector<AtomInput>& atoms,
                                  const std::vector<HBondPairInput>& hbondPairs,
                                  const EnergyParams& params, int nBlocks);

}  // namespace frustramol_tmol
