// CPU reference drivers: loop the neighbor / hbond lists and accumulate per-residue-pair
// energies for the in-scope ref2015 pairwise terms.
//
// Ports the Apache-2.0 tmol-webgpu drivers (src/cpu.ts = ljlk + fa_elec; src/cpu_terms.ts
// = lk_ball + hbond) to torch-free C++. Single-point forward only (no autograd). The
// per-pair scoring math is in the kernel headers; this file is the contact-pair loop and
// the block-pair accumulation, with optional OpenMP over the pair list.
//
// Parallelism / fork safety. The only concurrency here is OpenMP worker threads, bounded
// by the explicit `n_threads` argument the caller passes (never read from the machine
// core count internally), exactly like the AWSEM native core's effective_threads. There
// is no process fork: a caller running structures concurrently (dir_frustration with an
// outer process pool) passes the SHARED-BUDGET inner thread count (cores // outer), so
// outer*inner <= cores and the nested process+thread fan-out can never oversubscribe.
//
// Determinism. Each worker thread accumulates into a private set of block-pair matrices;
// the matrices are reduced across threads in thread-index order after the parallel
// region. Single-thread accumulation visits pairs in list order, matching the reference
// oracle bit-for-bit (to float precision); multi-thread results may differ only in the
// last ULPs of floating-point summation order (a stated fp-reduction-order tolerance, not
// a formula difference). See native_tmol/NOTICE for attribution.

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "types.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace frustramol_tmol {

// Mirrors the AWSEM native core's effective_threads: an explicit positive request wins;
// 0 means "use all available cores"; without OpenMP the core is serial (1).
inline int effectiveThreads(int n_threads) {
#ifdef _OPENMP
    if (n_threads > 0) return n_threads;
    const int procs = omp_get_num_procs();
    return procs > 0 ? procs : 1;
#else
    (void)n_threads;
    return 1;
#endif
}

namespace detail {

// Per-thread accumulator over `nSub` flattened nBlocks*nBlocks matrices, reduced in
// thread-index order into the final result. `pairBody(p, add)` scores one pair and calls
// `add(subIndex, b1, b2, energy)`; placement matches the tmol block-pair convention
// (min(b1,b2), max(b1,b2)). `nItems` is the loop length.
template <typename Body>
PairEnergyResult accumulate(const std::vector<std::string>& names, int nBlocks,
                            std::size_t nItems, int n_threads, Body pairBody) {
    const int nSub = static_cast<int>(names.size());
    const std::size_t NN = static_cast<std::size_t>(nBlocks) * nBlocks;
    int threads = effectiveThreads(n_threads);
    if (threads < 1) threads = 1;
    if (static_cast<std::size_t>(threads) > nItems && nItems > 0)
        threads = static_cast<int>(nItems);
    if (threads < 1) threads = 1;

    std::vector<std::vector<double>> acc(
        static_cast<std::size_t>(threads),
        std::vector<double>(static_cast<std::size_t>(nSub) * NN, 0.0));

#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        std::vector<double>& A = acc[static_cast<std::size_t>(tid)];
        auto add = [&](int sub, int b1, int b2, double e) {
            if (e == 0.0) return;
            const int lo = b1 < b2 ? b1 : b2;
            const int hi = b1 < b2 ? b2 : b1;
            A[static_cast<std::size_t>(sub) * NN +
              static_cast<std::size_t>(lo) * nBlocks + hi] += e;
        };
#pragma omp for schedule(static)
        for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(nItems); ++p)
            pairBody(static_cast<std::size_t>(p), add);
    }
#else
    {
        std::vector<double>& A = acc[0];
        auto add = [&](int sub, int b1, int b2, double e) {
            if (e == 0.0) return;
            const int lo = b1 < b2 ? b1 : b2;
            const int hi = b1 < b2 ? b2 : b1;
            A[static_cast<std::size_t>(sub) * NN +
              static_cast<std::size_t>(lo) * nBlocks + hi] += e;
        };
        for (std::size_t p = 0; p < nItems; ++p) pairBody(p, add);
    }
#endif

    PairEnergyResult res;
    res.nBlocks = nBlocks;
    res.names = names;
    res.blockPair.assign(static_cast<std::size_t>(nSub),
                         std::vector<double>(NN, 0.0));
    res.wholePose.assign(static_cast<std::size_t>(nSub), 0.0);
    for (int sub = 0; sub < nSub; ++sub) {
        std::vector<double>& out = res.blockPair[static_cast<std::size_t>(sub)];
        double s = 0.0;
        for (std::size_t k = 0; k < NN; ++k) {
            double cell = 0.0;
            for (int t = 0; t < threads; ++t)
                cell += acc[static_cast<std::size_t>(t)]
                          [static_cast<std::size_t>(sub) * NN + k];
            out[k] = cell;
            s += cell;
        }
        res.wholePose[static_cast<std::size_t>(sub)] = s;
    }
    return res;
}

}  // namespace detail

// Evaluate per-residue-pair ljlk (fa_ljatr/fa_ljrep/fa_lk) + fa_elec for one structure.
inline PairEnergyResult computePairEnergiesCPU(const std::vector<AtomInput>& atoms,
                                               const std::vector<PairInput>& pairs,
                                               const EnergyParams& params,
                                               int nBlocks, int n_threads = 0) {
    return detail::accumulate(
        ljlkElecSubterms(), nBlocks, pairs.size(), n_threads,
        [&](std::size_t p, auto add) {
            const PairInput& pr = pairs[p];
            const AtomInput& ai = atoms[static_cast<std::size_t>(pr.i)];
            const AtomInput& aj = atoms[static_cast<std::size_t>(pr.j)];
            const double dx = ai.x - aj.x, dy = ai.y - aj.y, dz = ai.z - aj.z;
            const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (d <= 0) return;
            const int b1 = ai.block, b2 = aj.block;

            // fa_elec: all atoms with a charge.
            const double e =
                elec(d, ai.charge, aj.charge, pr.sepElec, params.elecGlobal);
            if (e != 0) add(3, b1, b2, e);  // fa_elec

            const LjlkTypeParams& ti =
                params.ljlkTypeParams[static_cast<std::size_t>(ai.ljlkType)];
            const LjlkTypeParams& tj =
                params.ljlkTypeParams[static_cast<std::size_t>(aj.ljlkType)];
            // LJ (fa_atr / fa_rep) scores ALL atoms, hydrogens included.
            const auto lj = ljScore(d, pr.sepLjlk, ti, tj, params.ljlkGlobal);
            if (lj.first != 0) add(0, b1, b2, lj.first);    // fa_ljatr
            if (lj.second != 0) add(1, b1, b2, lj.second);  // fa_ljrep
            // LK solvation (fa_lk) scores HEAVY atoms only.
            if (ai.isHeavy && aj.isHeavy) {
                const double lk =
                    lkIsotropicScore(d, pr.sepLjlk, ti, tj, params.ljlkGlobal);
                if (lk != 0) add(2, b1, b2, lk);  // fa_lk
            }
        });
}

// Evaluate per-residue-pair lk_ball (four subterms) for one structure. Each heavy-atom
// neighbor pair is scored in BOTH directions (each atom in turn as the polar atom whose
// waters are occluded), exactly as tmol enumerates polar/occluder pairs.
inline PairEnergyResult computeLkBallCPU(const std::vector<AtomInput>& atoms,
                                         const std::vector<PairInput>& pairs,
                                         const EnergyParams& params, int nBlocks,
                                         int n_threads = 0) {
    static const Waters noWater{};
    return detail::accumulate(
        lkBallSubterms(), nBlocks, pairs.size(), n_threads,
        [&](std::size_t p, auto add) {
            const PairInput& pr = pairs[p];
            const AtomInput& ai = atoms[static_cast<std::size_t>(pr.i)];
            const AtomInput& aj = atoms[static_cast<std::size_t>(pr.j)];
            if (!ai.isHeavy || !aj.isHeavy) return;  // polar + occluder are heavy
            const double dx = ai.x - aj.x, dy = ai.y - aj.y, dz = ai.z - aj.z;
            const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (d <= 0) return;

            auto scoreDir = [&](const AtomInput& pol, const AtomInput& occ) {
                const Waters& wi = pol.hasWaters ? pol.waters : noWater;
                if (!wi.present[0]) return;
                const Waters& wj = occ.hasWaters ? occ.waters : noWater;
                const LkBallSubterms s = lkBallScore(
                    pol.xyz(), occ.xyz(), wi, wj, pr.sepLjlk, d,
                    params.ljlkTypeParams[static_cast<std::size_t>(pol.ljlkType)],
                    params.ljlkTypeParams[static_cast<std::size_t>(occ.ljlkType)],
                    params.lkBallGlobal);
                add(0, pol.block, occ.block, s.lk_ball_iso);
                add(1, pol.block, occ.block, s.lk_ball);
                add(2, pol.block, occ.block, s.lk_bridge);
                add(3, pol.block, occ.block, s.lk_bridge_uncpl);
            };
            scoreDir(ai, aj);
            scoreDir(aj, ai);
        });
}

// Evaluate per-residue-pair hbond for one structure. Each donor-H / acceptor pair carries
// its resolved geometry and per-pair polynomials; pairs with bonded separation < 5 are
// excluded (tmol count-pair).
inline PairEnergyResult computeHbondCPU(const std::vector<AtomInput>& atoms,
                                        const std::vector<HBondPairInput>& hbondPairs,
                                        const EnergyParams& params, int nBlocks,
                                        int n_threads = 0) {
    return detail::accumulate(
        hbondSubterms(), nBlocks, hbondPairs.size(), n_threads,
        [&](std::size_t p, auto add) {
            const HBondPairInput& hp = hbondPairs[p];
            if (hp.sep < 5) return;  // tmol count-pair exclusion
            const AtomInput& H = atoms[static_cast<std::size_t>(hp.h)];
            const AtomInput& A = atoms[static_cast<std::size_t>(hp.a)];
            const double e = hbondScore(hp.D, H.xyz(), A.xyz(), hp.B, hp.B0,
                                        hp.pair, params.hbondGlobal);
            add(0, H.block, A.block, e);  // hbond
        });
}

}  // namespace frustramol_tmol
