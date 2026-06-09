// ljlk per-atom-pair energy functions (Rosetta fa_atr / fa_rep / fa_lk).
//
// Ported verbatim from the Apache-2.0 tmol-webgpu reference (src/kernels/ljlk.ts),
// itself a verbatim port of tmol/score/ljlk/potentials/{lj,lk_isotropic,common}.hh
// (Apache-2.0). Pure scalar functions of the inter-atom distance, the bonded
// count-pair separation, and the two atoms' resolved type parameters. Single-point
// forward only. See native_tmol/NOTICE for attribution.

#pragma once

#include <cmath>
#include <utility>

#include "interpolate.hpp"

namespace frustramol_tmol {

// Per-atom-type ljlk parameters (one row of the loaded ljlk atom-type table).
struct LjlkTypeParams {
    double lj_radius;
    double lj_wdepth;
    double lk_dgfree;
    double lk_lambda;
    double lk_volume;
    bool is_donor;
    bool is_hydroxyl;
    bool is_polarh;
    bool is_acceptor;
};

// ljlk global parameters (the three hydrogen-bond sigma overrides).
struct LjlkGlobalParams {
    double lj_hbond_dis;
    double lj_hbond_OH_donor_dis;
    double lj_hbond_hdis;
};

inline constexpr double PI_POW_1P5 = 5.56832799683;

// Count-pair connectivity weight: 0 for <4 bonds, 0.2 at exactly 4, 1 beyond.
inline double connectivityWeight(double bondedPathLength) {
    if (bondedPathLength > 4) return 1.0;
    if (bondedPathLength == 4) return 0.2;
    return 0.0;
}

// The interaction sigma for a pair, with the donor/acceptor hydrogen-bond overrides.
inline double ljSigma(const LjlkTypeParams& i, const LjlkTypeParams& j,
                      const LjlkGlobalParams& g) {
    if ((i.is_donor && !i.is_hydroxyl && j.is_acceptor) ||
        (j.is_donor && !j.is_hydroxyl && i.is_acceptor)) {
        return g.lj_hbond_dis;
    } else if ((i.is_donor && i.is_hydroxyl && j.is_acceptor) ||
               (j.is_donor && j.is_hydroxyl && i.is_acceptor)) {
        return g.lj_hbond_OH_donor_dis;
    } else if ((i.is_polarh && j.is_acceptor) ||
               (j.is_polarh && i.is_acceptor)) {
        return g.lj_hbond_hdis;
    }
    return i.lj_radius + j.lj_radius;
}

// Base van der Waals well: value and d/ddist.
inline double vdwV(double dist, double sigma, double epsilon) {
    const double sd = sigma / dist;
    const double sd2 = sd * sd;
    const double sd6 = sd2 * sd2 * sd2;
    const double sd12 = sd6 * sd6;
    return epsilon * (sd12 - 2 * sd6);
}
inline double vdwDV(double dist, double sigma, double epsilon) {
    const double sd = sigma / dist;
    const double sd2 = sd * sd;
    const double sd6 = sd2 * sd2 * sd2;
    const double sd12 = sd6 * sd6;
    return epsilon * ((-12.0 * sd12 / dist) - (-12.0 * sd6 / dist));
}

// LJ attractive and repulsive subterms (fa_ljatr, fa_ljrep) for one pair.
inline std::pair<double, double> ljScore(double dist, double bondedPathLength,
                                         const LjlkTypeParams& i,
                                         const LjlkTypeParams& j,
                                         const LjlkGlobalParams& g) {
    const double cpolyDmax = 6.0;
    if (dist > cpolyDmax) return {0.0, 0.0};

    const double sigma = ljSigma(i, j, g);
    const double epsilon = std::sqrt(i.lj_wdepth * j.lj_wdepth);
    const double dLin = sigma * 0.6;
    double cpolyDmin;
    if (sigma > 4.5)
        cpolyDmin = sigma > cpolyDmax - 0.1 ? cpolyDmax - 0.1 : sigma;
    else
        cpolyDmin = 4.5;

    const double weight = connectivityWeight(bondedPathLength);

    double vatr;
    if (dist > cpolyDmin) {
        const double v = vdwV(cpolyDmin, sigma, epsilon);
        const double dv = vdwDV(cpolyDmin, sigma, epsilon);
        vatr = interpolateToZero(dist, cpolyDmin, v, dv, cpolyDmax);
    } else if (dist > dLin) {
        vatr = vdwV(dist, sigma, epsilon);
    } else {
        const double v = vdwV(dLin, sigma, epsilon);
        const double dv = vdwDV(dLin, sigma, epsilon);
        vatr = v + dv * (dist - dLin);
    }

    double vrep;
    if (dist < sigma) {
        vrep = vatr + epsilon;
        vatr = -epsilon;
    } else {
        vrep = 0.0;
    }
    return {weight * vatr, weight * vrep};
}

// Lazaridis-Karplus desolvation of i by the volume of j.
inline double fDesolvV(double dist, double ljRadiusI, double lkDgfreeI,
                       double lkLambdaI, double lkVolumeJ) {
    const double dr = dist - ljRadiusI;
    return (-lkVolumeJ * lkDgfreeI / (2 * PI_POW_1P5 * lkLambdaI) /
            (dist * dist) * std::exp(-(dr * dr) / (lkLambdaI * lkLambdaI)));
}
inline double fDesolvDV(double dist, double ljRadiusI, double lkDgfreeI,
                        double lkLambdaI, double lkVolumeJ) {
    const double dr = dist - ljRadiusI;
    const double expVal = std::exp(-(dr * dr) / (lkLambdaI * lkLambdaI));
    return (-lkVolumeJ * lkDgfreeI / (2 * PI_POW_1P5 * lkLambdaI) * expVal *
            ((-2 / (dist * dist * dist)) +
             (1 / (dist * dist) * -(2 * dist - 2 * ljRadiusI) /
              (lkLambdaI * lkLambdaI))));
}

// One direction of the LK pair desolvation (i desolvated by j), with the smoothing
// splines. Exported (unweighted) because lk_ball reuses this exact one-directional
// spline as the isotropic core of its score (tmol lk_isotropic_pair::V).
inline double lkIsotropicPair(double dist, double ljSigmaIj, double ljRadiusI,
                              double lkDgfreeI, double lkLambdaI,
                              double lkVolumeJ) {
    const double dMin = ljSigmaIj * 0.89;
    double cpolyCloseDmin = dMin * dMin - 1.45;
    if (cpolyCloseDmin < 0.01) cpolyCloseDmin = 0.01;
    cpolyCloseDmin = std::sqrt(cpolyCloseDmin);
    const double cpolyCloseDmax = std::sqrt(dMin * dMin + 1.05);
    const double cpolyFarDmin = 4.5;
    const double cpolyFarDmax = 6.0;

    if (dist > cpolyFarDmax) {
        return 0.0;
    } else if (dist > cpolyFarDmin) {
        const double v =
            fDesolvV(cpolyFarDmin, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
        const double dv =
            fDesolvDV(cpolyFarDmin, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
        return interpolateToZero(dist, cpolyFarDmin, v, dv, cpolyFarDmax);
    } else if (dist > cpolyCloseDmax) {
        return fDesolvV(dist, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
    } else if (dist > cpolyCloseDmin) {
        const double vMin =
            fDesolvV(dMin, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
        const double vMax =
            fDesolvV(cpolyCloseDmax, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
        const double dvMax =
            fDesolvDV(cpolyCloseDmax, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
        return interpolate(dist, cpolyCloseDmin, vMin, 0.0, cpolyCloseDmax, vMax,
                           dvMax);
    }
    return fDesolvV(dMin, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
}

// LK isotropic solvation subterm (fa_lk) for one pair: both desolvation directions.
inline double lkIsotropicScore(double dist, double bondedPathLength,
                               const LjlkTypeParams& i, const LjlkTypeParams& j,
                               const LjlkGlobalParams& g) {
    const double sigma = ljSigma(i, j, g);
    const double weight = connectivityWeight(bondedPathLength);
    const double ij = lkIsotropicPair(dist, sigma, i.lj_radius, i.lk_dgfree,
                                      i.lk_lambda, j.lk_volume);
    const double ji = lkIsotropicPair(dist, sigma, j.lj_radius, j.lk_dgfree,
                                      j.lk_lambda, i.lk_volume);
    return weight * (ij + ji);
}

}  // namespace frustramol_tmol
