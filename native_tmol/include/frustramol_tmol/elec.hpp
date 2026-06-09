// fa_elec per-atom-pair energy (Coulomb with the Rosetta sigmoidal distance-dependent
// dielectric).
//
// Ported verbatim from the Apache-2.0 tmol-webgpu reference (src/kernels/elec.ts),
// itself a verbatim port of tmol/score/elec/potentials/{elec,potentials}.hh
// (Apache-2.0). Pure scalar function of distance, the two partial charges, and the
// bonded count-pair separation. Single-point forward only. See native_tmol/NOTICE.

#pragma once

#include <cmath>

#include "interpolate.hpp"
#include "ljlk.hpp"  // connectivityWeight

namespace frustramol_tmol {

// fa_elec global parameters (the sigmoidal-dielectric constants and the distance fades).
struct ElecGlobalParams {
    double D;
    double D0;
    double S;
    double min_dis;
    double max_dis;
};

inline constexpr double ELEC_C1 = 322.0637;  // electrostatic energy constant

inline double elecEps(double dist, double D, double D0, double S) {
    return D - 0.5 * (D - D0) *
                   (2 + 2 * dist * S + dist * dist * S * S) *
                   std::exp(-dist * S);
}
inline double elecDepsDdist(double dist, double D, double D0, double S) {
    return 0.5 * (D - D0) * dist * dist * S * S * S * std::exp(-dist * S);
}

// fa_elec energy for one atom pair.
inline double elec(double dist, double eI, double eJ, double bondedPathLength,
                   const ElecGlobalParams& g) {
    const double D = g.D;
    const double D0 = g.D0;
    const double S = g.S;
    const double minDis = g.min_dis;
    const double maxDis = g.max_dis;
    const double lowPolyStart = minDis - 0.25;
    const double lowPolyEnd = minDis + 0.25;
    const double hiPolyStart = maxDis - 1.0;
    const double hiPolyEnd = maxDis;

    const double weight = connectivityWeight(bondedPathLength);
    const double C2 = ELEC_C1 / (maxDis * elecEps(maxDis, D, D0, S));
    const double eiej = eI * eJ;
    if (eiej == 0) return 0;

    double elecE = 0;
    if (dist < lowPolyStart) {
        const double minDisScore = ELEC_C1 / (minDis * elecEps(minDis, D, D0, S)) - C2;
        elecE = eiej * minDisScore;
    } else if (dist < lowPolyEnd) {
        const double minDisScore = ELEC_C1 / (minDis * elecEps(minDis, D, D0, S)) - C2;
        const double epsElec = elecEps(lowPolyEnd, D, D0, S);
        const double depsElec = elecDepsDdist(lowPolyEnd, D, D0, S);
        const double dmaxElec = eiej * (ELEC_C1 / (lowPolyEnd * epsElec) - C2);
        // Note: tmol passes deps_elec_d_dist (the dielectric derivative) as the
        // endpoint slope here, not the energy derivative. Ported verbatim.
        elecE = interpolate(dist, lowPolyStart, eiej * minDisScore, 0.0,
                            lowPolyEnd, dmaxElec, depsElec);
    } else if (dist < hiPolyStart) {
        const double epsElec = elecEps(dist, D, D0, S);
        elecE = eiej * (ELEC_C1 / (dist * epsElec) - C2);
    } else if (dist < hiPolyEnd) {
        const double epsElec = elecEps(hiPolyStart, D, D0, S);
        const double depsElec = elecDepsDdist(hiPolyStart, D, D0, S);
        const double dminElec = eiej * (ELEC_C1 / (hiPolyStart * epsElec) - C2);
        const double dminElecDDist =
            (-ELEC_C1 * eiej * (epsElec + hiPolyStart * depsElec)) /
            (hiPolyStart * hiPolyStart * epsElec * epsElec);
        elecE = interpolateToZero(dist, hiPolyStart, dminElec, dminElecDDist,
                                  hiPolyEnd);
    }
    return weight * elecE;
}

}  // namespace frustramol_tmol
