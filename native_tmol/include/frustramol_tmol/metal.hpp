// Declarations for the optional Metal (Apple GPU) energy path (M7).
//
// These mirror the CPU drivers in driver.hpp one-to-one (same inputs, same
// PairEnergyResult), but evaluate each pair's per-subterm energy on the GPU in float32 and
// scatter-add into the block-pair matrices host-side in double. The implementations live in
// src/metal/dispatch.cpp and are compiled ONLY when FRUSTRAMOL_TMOL_METAL is defined (a
// Mac with metal-cpp + `xcrun metal`). On any other build these symbols are absent, so
// callers must guard references with `#ifdef FRUSTRAMOL_TMOL_METAL`. See
// docs/tmol/M7_NATIVE_METAL.md and native_tmol/NOTICE.

#pragma once

#include <vector>

#include "types.hpp"

namespace frustramol_tmol {

// True iff this build has the Metal path AND a Metal-capable system default device is
// present. Always false when compiled without FRUSTRAMOL_TMOL_METAL.
bool metalAvailable();

// fa_ljatr / fa_ljrep / fa_lk / fa_elec per-residue-pair matrices, computed on the GPU.
PairEnergyResult computePairEnergiesMetal(const std::vector<AtomInput>& atoms,
                                          const std::vector<PairInput>& pairs,
                                          const EnergyParams& params, int nBlocks);

// lk_ball_iso / lk_ball / lk_bridge / lk_bridge_uncpl per-residue-pair matrices.
PairEnergyResult computeLkBallMetal(const std::vector<AtomInput>& atoms,
                                    const std::vector<PairInput>& pairs,
                                    const EnergyParams& params, int nBlocks);

// hbond per-residue-pair matrix.
PairEnergyResult computeHbondMetal(const std::vector<AtomInput>& atoms,
                                   const std::vector<HBondPairInput>& hbondPairs,
                                   const EnergyParams& params, int nBlocks);

}  // namespace frustramol_tmol
