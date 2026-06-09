// lk_ball per-atom-pair energy (Rosetta lk_ball_iso / lk_ball / lk_bridge /
// lk_bridge_uncpl).
//
// Ported verbatim from the Apache-2.0 tmol-webgpu reference (src/kernels/lk_ball.ts),
// itself a verbatim port of tmol/score/lk_ball/potentials/lk_ball.hh (Apache-2.0).
// Single-point forward only. See native_tmol/NOTICE for attribution.
//
// Unlike ljlk/elec, lk_ball is not a bare function of the inter-atom distance: each
// polar heavy atom carries up to MAX_WATER explicit "water" positions that tmol's
// gen_pose_waters builds. This kernel consumes those RESOLVED waters (the geometry
// generation is orchestration, like atom typing) and reproduces the four subterms.

#pragma once

#include <array>
#include <cmath>

#include "geom.hpp"
#include "ljlk.hpp"  // connectivityWeight, ljSigma, lkIsotropicPair, LjlkTypeParams

namespace frustramol_tmol {

inline constexpr int MAX_WATER = 4;

// Up to MAX_WATER resolved water positions for one atom. `present[w]` mirrors the TS
// "null water" guard (an absent water is skipped, never scored).
struct Waters {
    std::array<Vec3, MAX_WATER> pos{};
    std::array<bool, MAX_WATER> present{{false, false, false, false}};
};

// lk_ball global parameters (the three hydrogen-bond sigma overrides plus the water
// distance and the heavy-atom interaction cutoff).
struct LkBallGlobalParams {
    double lj_hbond_dis;
    double lj_hbond_OH_donor_dis;
    double lj_hbond_hdis;
    double lkb_water_dist;
    double distance_threshold;
};

// tmol lkball_globals (lk_ball.hh): fixed ramp constants.
inline constexpr double OVERLAP_GAP_A2 = 0.5;
inline constexpr double OVERLAP_WIDTH_A2 = 2.6;
inline constexpr double ANGLE_OVERLAP_A2 = 2.8 * OVERLAP_WIDTH_A2;
inline constexpr double RAMP_WIDTH_A2 = 3.709;

inline double lkb_sq(double v) { return v * v; }

// lk_fraction::V - the directional desolvation ramp of occluder J against polar I's
// waters.
inline double lkFraction(const Waters& waters, const Vec3& j, double ljRadiusJ) {
    double d2Low = lkb_sq(1.4 + ljRadiusJ) - RAMP_WIDTH_A2;
    if (d2Low < 0.0) d2Low = 0.0;

    double wtedD2Delta = 0;
    for (int w = 0; w < MAX_WATER; w++) {
        if (!waters.present[w]) continue;  // NaN water -> skipped (tmol isnan guard)
        const Vec3 d = sub(j, waters.pos[w]);
        const double d2Delta = dot(d, d) - d2Low;
        wtedD2Delta += std::exp(-d2Delta);
    }
    wtedD2Delta = -std::log(wtedD2Delta);

    if (wtedD2Delta < 0) return 1;
    if (wtedD2Delta < RAMP_WIDTH_A2)
        return lkb_sq(1 - lkb_sq(wtedD2Delta / RAMP_WIDTH_A2));
    return 0;
}

// lk_bridge_fraction::V - the water-overlap (bridging) term between two polar atoms.
inline double lkBridgeFraction(const Vec3& i, const Vec3& j, const Waters& wi,
                               const Waters& wj, double lkbWaterDist) {
    // water overlap
    double wtedD2Delta = 0.0;
    for (int a = 0; a < MAX_WATER; a++) {
        if (!wi.present[a]) continue;
        for (int b = 0; b < MAX_WATER; b++) {
            if (!wj.present[b]) continue;
            const Vec3 d = sub(wi.pos[a], wj.pos[b]);
            const double d2Delta = dot(d, d) - OVERLAP_GAP_A2;
            wtedD2Delta += std::exp(-d2Delta);
        }
    }
    wtedD2Delta = -std::log(wtedD2Delta);

    double overlapfrac;
    if (wtedD2Delta > OVERLAP_WIDTH_A2)
        overlapfrac = 0;
    else
        overlapfrac = lkb_sq(1 - lkb_sq(wtedD2Delta / OVERLAP_WIDTH_A2));

    // base angle
    const double overlapTargetLen2 = (8.0 / 3.0) * lkb_sq(lkbWaterDist);
    const Vec3 dij = sub(i, j);
    const double overlapLen2 = dot(dij, dij);
    const double baseDelta = std::abs(overlapLen2 - overlapTargetLen2);

    double anglefrac;
    if (baseDelta > ANGLE_OVERLAP_A2)
        anglefrac = 0;
    else
        anglefrac = lkb_sq(1 - lkb_sq(baseDelta / ANGLE_OVERLAP_A2));

    return overlapfrac * anglefrac;
}

// One subterm bundle from lk_ball_score::V.
struct LkBallSubterms {
    double lk_ball_iso;
    double lk_ball;
    double lk_bridge;
    double lk_bridge_uncpl;
};

// lk_ball_score::V for one ordered (polar I, occluder J) pair.
//
// `bondedPathLength <= 3` returns all zero (tmol lk_ball_atom_energy_full early-out);
// note lk_bridge_uncpl does NOT carry the connectivity weight, so this guard is
// load-bearing and cannot be folded into connectivityWeight alone. The polar atom I
// must have at least one water, or there is no I-against-J score.
inline LkBallSubterms lkBallScore(const Vec3& i, const Vec3& j, const Waters& wi,
                                  const Waters& wj, double bondedPathLength,
                                  double dist, const LjlkTypeParams& ti,
                                  const LjlkTypeParams& tj,
                                  const LkBallGlobalParams& g) {
    const LkBallSubterms zero{0, 0, 0, 0};
    if (bondedPathLength <= 3) return zero;
    if (!wi.present[0]) return zero;
    if (dist >= g.distance_threshold) return zero;

    LjlkGlobalParams lg{g.lj_hbond_dis, g.lj_hbond_OH_donor_dis, g.lj_hbond_hdis};
    const double sigma = ljSigma(ti, tj, lg);
    const double lkIso =
        connectivityWeight(bondedPathLength) *
        lkIsotropicPair(dist, sigma, ti.lj_radius, ti.lk_dgfree, ti.lk_lambda,
                        tj.lk_volume);
    const double fracDesolv = lkFraction(wi, j, tj.lj_radius);
    double fracOverlap = 0.0;
    if (tj.is_donor || tj.is_acceptor) {
        fracOverlap = lkBridgeFraction(i, j, wi, wj, g.lkb_water_dist);
    }
    return {
        lkIso,                // lk_ball_iso
        lkIso * fracDesolv,   // lk_ball
        lkIso * fracOverlap,  // lk_bridge
        fracOverlap / 2,      // lk_bridge_uncpl
    };
}

}  // namespace frustramol_tmol
