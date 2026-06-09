// Input/output data types for the torch-free CPU energy kernels.
//
// Mirrors the Apache-2.0 tmol-webgpu public data shape (src/types.ts). Atom typing,
// charge assignment, and the lk_ball waters / hbond bases are RESOLVED upstream (a
// Rosetta/tmol pose); the kernels consume those resolved values. Single-point forward
// only. See native_tmol/NOTICE for attribution.

#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "elec.hpp"
#include "hbond.hpp"
#include "lk_ball.hpp"
#include "ljlk.hpp"

namespace frustramol_tmol {

// One atom's resolved kernel inputs.
struct AtomInput {
    double x = 0, y = 0, z = 0;
    int block = 0;       // residue (block) index; energies are summed per block pair
    int ljlkType = 0;    // index into EnergyParams.ljlkTypeParams
    double charge = 0;   // partial charge (fa_elec)
    bool isHeavy = false;// ljlk fa_lk and lk_ball score heavy atoms only
    bool hasWaters = false;  // false == tmol "undefined" (no waters resolved)
    Waters waters{};     // lk_ball: tmol-resolved water coordinates
    Vec3 xyz() const { return {x, y, z}; }
};

// One scored donor-H / acceptor pair for hbond, with tmol-resolved geometry folded in.
struct HBondPairInput {
    int h = 0;  // index into the atom array of the donor hydrogen
    int a = 0;  // index into the atom array of the acceptor
    Vec3 D{};   // resolved donor heavy parent
    Vec3 B{};   // resolved acceptor base
    Vec3 B0{};  // resolved acceptor base2
    HBondPairParams pair{};
    int sep = 0;  // H-A bonded path length; pairs with separation < 5 are excluded
};

// One scored atom pair from the neighbor list, with its two bonded count-pair
// separations (ljlk and elec use different count-pair conventions).
struct PairInput {
    int i = 0;
    int j = 0;
    int sepLjlk = 0;
    int sepElec = 0;
};

// The runtime-loaded parameter tables the kernels need.
struct EnergyParams {
    std::vector<LjlkTypeParams> ljlkTypeParams;
    LjlkGlobalParams ljlkGlobal{};
    ElecGlobalParams elecGlobal{};
    LkBallGlobalParams lkBallGlobal{};
    HBondGlobalParams hbondGlobal{};
};

// Per-residue-pair and whole-pose energies, per subterm. blockPair[t] is the flattened
// nBlocks x nBlocks matrix (row-major, symmetric placement); wholePose[t] its sum.
struct PairEnergyResult {
    int nBlocks = 0;
    std::vector<std::string> names;          // subterm name per row of blockPair
    std::vector<std::vector<double>> blockPair;
    std::vector<double> wholePose;

    const std::vector<double>& mat(const std::string& name) const {
        for (std::size_t t = 0; t < names.size(); ++t)
            if (names[t] == name) return blockPair[t];
        throw std::out_of_range("unknown subterm: " + name);
    }
    double pose(const std::string& name) const {
        for (std::size_t t = 0; t < names.size(); ++t)
            if (names[t] == name) return wholePose[t];
        throw std::out_of_range("unknown subterm: " + name);
    }
};

// The subterm name lists, in tmol's all_terms() order (matches tmol-webgpu).
inline const std::vector<std::string>& ljlkElecSubterms() {
    static const std::vector<std::string> v = {"fa_ljatr", "fa_ljrep", "fa_lk",
                                               "fa_elec"};
    return v;
}
inline const std::vector<std::string>& lkBallSubterms() {
    static const std::vector<std::string> v = {"lk_ball_iso", "lk_ball",
                                               "lk_bridge", "lk_bridge_uncpl"};
    return v;
}
inline const std::vector<std::string>& hbondSubterms() {
    static const std::vector<std::string> v = {"hbond"};
    return v;
}

}  // namespace frustramol_tmol
