// hbond per-donor/acceptor energy (Rosetta hbond).
//
// Ported verbatim from the Apache-2.0 tmol-webgpu reference (src/kernels/hbond.ts),
// itself a verbatim port of tmol/score/hbond/potentials/{potentials,hbond}.hh
// (Apache-2.0). Single-point forward only. See native_tmol/NOTICE for attribution.
//
// Like lk_ball, hbond consumes GENERATED GEOMETRY: the donor heavy parent D and the
// acceptor base atoms B/B0 that tmol's gen_hbond_bases resolves per donor/acceptor.
// This kernel scores one (donor H, acceptor A) pair given the resolved D/H/A/B/B0
// coordinates, the acceptor hybridization, the donor*acceptor weight, and the three
// per-pair polynomials.

#pragma once

#include <array>
#include <cmath>

#include "geom.hpp"

namespace frustramol_tmol {

inline constexpr double HB_PI = 3.14159265358979323846;

// One bounded polynomial: 11 Horner coefficients with an out-of-range clamp.
struct HBondPoly {
    std::array<double, 11> coeffs;
    std::array<double, 2> range;  // [xmin, xmax]
    std::array<double, 2> bound;  // [value below xmin, value above xmax]
};

// Per-(donor,acceptor)-pair hbond parameters, resolved from the tmol database.
struct HBondPairParams {
    int hyb;            // acceptor hybridization: 1=sp2, 2=sp3, 3=ring
    double ad_weight;   // acceptor_weight * donor_weight
    HBondPoly AHdist;
    HBondPoly cosBAH;
    HBondPoly cosAHD;
};

// hbond global parameters (the sp2 chi shape constants, the sp3 softmax fade, and the
// maximum H-A distance scored).
struct HBondGlobalParams {
    double hb_sp2_range_span;
    double hb_sp2_BAH180_rise;
    double hb_sp2_outer_width;
    double hb_sp3_softmax_fade;
    double threshold_distance;
    double max_ha_dis;
};

inline constexpr int HB_SP2 = 1;
inline constexpr int HB_SP3 = 2;
inline constexpr int HB_RING = 3;

// bound_poly::V - clamp outside [xmin, xmax], else Horner-evaluate the degree-10 poly.
inline double boundPoly(double x, const HBondPoly& p) {
    if (x < p.range[0]) return p.bound[0];
    if (x > p.range[1]) return p.bound[1];
    double v = p.coeffs[0];
    for (int i = 1; i < 11; i++) v = v * x + p.coeffs[i];
    return v;
}

inline double bahAngleBaseForm(const Vec3& b, const Vec3& a, const Vec3& h,
                               const HBondPoly& poly) {
    const Vec3 ah = sub(h, a);
    const Vec3 ba = sub(a, b);
    return boundPoly(cosInteriorAngle(ah, ba), poly);
}

// BAH_angle_V: the base-acceptor-H angle term, blended by acceptor hybridization.
inline double bahAngle(const Vec3& b, const Vec3& b0, const Vec3& a, const Vec3& h,
                       int hyb, const HBondPoly& poly, double sp3SoftmaxFade) {
    if (hyb == HB_SP2) return bahAngleBaseForm(b, a, h, poly);
    if (hyb == HB_RING)
        return bahAngleBaseForm(scale(add(b, b0), 0.5), a, h, poly);
    // sp3: softmax over the two bases
    const double pxH = bahAngleBaseForm(b, a, h, poly);
    const double pxH0 = bahAngleBaseForm(b0, a, h, poly);
    return std::log(std::exp(pxH * sp3SoftmaxFade) +
                    std::exp(pxH0 * sp3SoftmaxFade)) /
           sp3SoftmaxFade;
}

// sp2chi_energy_V: the sp2 dihedral (chi) energy.
inline double sp2chiEnergy(double ang, double chi, double d, double m, double l) {
    const double pi = HB_PI;
    const double H = 0.5 * (std::cos(2 * chi) + 1);
    if (ang > (pi * 2.0) / 3.0) {
        const double F = (d / 2) * std::cos(3 * (pi - ang)) + d / 2 - 0.5;
        const double G = d - 0.5;
        return H * F + (1 - H) * G;
    } else if (ang >= pi * (2.0 / 3.0 - l)) {
        const double outerRise = std::cos(pi - ((pi * 2) / 3 - ang) / l);
        const double F = (m / 2) * outerRise + m / 2 - 0.5;
        const double G = ((m - d) / 2) * outerRise + (m - d) / 2 + d - 0.5;
        return H * F + (1 - H) * G;
    }
    return m - 0.5;
}

// B0BAH_chi_V: the sp2 chi contribution (zero for sp3/ring acceptors).
inline double b0bahChi(const Vec3& b0, const Vec3& b, const Vec3& a, const Vec3& h,
                       int hyb, const HBondGlobalParams& g) {
    if (hyb != HB_SP2) return 0;
    const double bah = ptInteriorAngle(b, a, h);
    const double b0bah = dihedralAngle(b0, b, a, h);
    return sp2chiEnergy(bah, b0bah, g.hb_sp2_BAH180_rise, g.hb_sp2_range_span,
                        g.hb_sp2_outer_width);
}

// hbond_score::V for one (donor H, acceptor A) pair. Returns 0 outside max_ha_dis or
// for the count-pair exclusion (bonded separation < 5, applied by the caller).
inline double hbondScore(const Vec3& d, const Vec3& h, const Vec3& a, const Vec3& b,
                         const Vec3& b0, const HBondPairParams& pair,
                         const HBondGlobalParams& g) {
    if (dist(h, a) >= g.max_ha_dis) return 0;

    double e = 0.0;
    e += boundPoly(dist(a, h), pair.AHdist);              // A-H distance
    e += boundPoly(ptInteriorAngle(a, h, d), pair.cosAHD); // A-H-D angle (radians)
    e += bahAngle(b, b0, a, h, pair.hyb, pair.cosBAH, g.hb_sp3_softmax_fade);
    e += b0bahChi(b0, b, a, h, pair.hyb, g);

    e *= pair.ad_weight;

    // Truncate and fade [-0.1, 0.1] to [-0.1, 0.0].
    if (e > 0.1) return 0;
    if (e > -0.1) return -0.025 + 0.5 * e - 2.5 * e * e;
    return e;
}

}  // namespace frustramol_tmol
