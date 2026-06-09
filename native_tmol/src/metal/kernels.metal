// Metal (MSL) compute kernels for the torch-free all-atom ref2015 energy backend (M7).
//
// Compiled only when FRUSTRAMOL_TMOL_METAL is ON and `xcrun metal` is available (Apple,
// macOS). The dev container has no Apple GPU and no metal compiler, so this file is NOT
// built or run there: it is authored to mirror the validated CPU kernels
// (native_tmol/include/frustramol_tmol/{interpolate,geom,ljlk,elec,lk_ball,hbond}.hpp)
// scalar-for-scalar, which themselves are verbatim ports of the Apache-2.0 tmol-webgpu
// reference. Single-point forward only: no autograd, no torch.
//
// STRUCTURE. The CPU drivers (driver.hpp) loop a pair list and scatter each pair's
// per-subterm energy into a per-residue-pair (block-pair) matrix. Here, exactly one GPU
// thread evaluates one pair and writes its per-subterm energy to a flat output array; the
// block-pair scatter-add (the cheap, discrete, order-defined accumulation) is done
// host-side in double in dispatch.cpp, identical to driver.hpp's `add` lambda. This keeps
// the accumulation bit-identical to the CPU reference and confines float32 to the per-pair
// energy arithmetic, mirroring the AWSEM Metal lane (native/src/metal/kernels.metal),
// where the discrete decisions stay host-side in double and only the energy math runs on
// the GPU in float32.
//
// PRECISION -- READ THIS. Apple GPUs are float32-only (no IEEE double in MSL), while the
// CPU reference does every term in double. Every line below that the CPU kernel performs
// in `double` is performed here in `float`. The spots most exposed to the float32 gap
// (flagged inline with TODO(metal-precision)):
//   * lk_ball's exp/log water sums (m_lk_fraction / m_lk_bridge_fraction) -- a sum of a
//     handful of exponentials, then a log; small but nonlinear;
//   * hbond's degree-10 Horner polynomials (m_bound_poly) and the sp3 log-sum-exp
//     (m_bah_angle) -- polynomial evaluation near the clamp boundaries is the worst case.
// If parity exceeds the G2 tolerance (1e-3) on real hardware, the maintainer can promote
// the most sensitive helper(s) to a compensated (Kahan) accumulation, or fold the
// affected term back onto the host CPU path (the per-pair energy is the dominant cost and
// is what runs on the GPU here). See docs/tmol/M7_NATIVE_METAL.md.
//
// LAYOUT COUPLING. The MAtom / MPoly structs and the MLjlkGlobal / MElecGlobal /
// MLkBallGlobal / MHbondGlobal scalar bundles below MUST stay byte-identical to the
// matching structs in dispatch.cpp (POD floats/ints only) so setBytes()/device buffers
// land the exact layout the kernels read. Buffer indices are per-kernel and must match the
// setBuffer/setBytes order in dispatch.cpp.

#include <metal_stdlib>
using namespace metal;

constant int MAX_WATER = 4;
constant float PI_POW_1P5 = 5.56832799683f;
constant float ELEC_C1 = 322.0637f;
constant float HB_PI = 3.14159265358979323846f;

// lkball_globals (lk_ball.hh): fixed ramp constants.
constant float OVERLAP_GAP_A2 = 0.5f;
constant float OVERLAP_WIDTH_A2 = 2.6f;
constant float ANGLE_OVERLAP_A2 = 2.8f * 2.6f;
constant float RAMP_WIDTH_A2 = 3.709f;

// hbond hybridization codes.
constant int HB_SP2 = 1;
constant int HB_SP3 = 2;
constant int HB_RING = 3;

// Per-atom flag bits packed into MAtom.flags (must match dispatch.cpp).
constant uint F_DONOR = 1u;
constant uint F_HYDROXYL = 2u;
constant uint F_POLARH = 4u;
constant uint F_ACCEPTOR = 8u;
constant uint F_HEAVY = 16u;

// One atom's resolved ljlk/elec type params + flags (mirrors LjlkTypeParams + charge +
// is_heavy). 7 x 4 bytes = 28 bytes, 4-byte aligned.
struct MAtom {
    float lj_radius;
    float lj_wdepth;
    float lk_dgfree;
    float lk_lambda;
    float lk_volume;
    float charge;
    uint flags;
};

// One bounded hbond polynomial (mirrors HBondPoly): 11 Horner coeffs + [xmin,xmax] +
// [below,above]. 15 x 4 bytes = 60 bytes.
struct MPoly {
    float coeffs[11];
    float range[2];
    float bound[2];
};

struct MLjlkGlobal {
    float lj_hbond_dis;
    float lj_hbond_OH_donor_dis;
    float lj_hbond_hdis;
};
struct MElecGlobal {
    float D;
    float D0;
    float S;
    float min_dis;
    float max_dis;
};
struct MLkBallGlobal {
    float lj_hbond_dis;
    float lj_hbond_OH_donor_dis;
    float lj_hbond_hdis;
    float lkb_water_dist;
    float distance_threshold;
};
struct MHbondGlobal {
    float hb_sp2_range_span;
    float hb_sp2_BAH180_rise;
    float hb_sp2_outer_width;
    float hb_sp3_softmax_fade;
    float threshold_distance;
    float max_ha_dis;
};

inline bool has_flag(uint flags, uint bit) { return (flags & bit) != 0u; }

inline float3 load_xyz(device const float* coord, int i) {
    return float3(coord[3 * i + 0], coord[3 * i + 1], coord[3 * i + 2]);
}

// ---------------------------------------------------------------------------
// interpolate.hpp -- cubic Hermite helpers
// ---------------------------------------------------------------------------
inline float m_interpolate(float x, float x0, float p0, float dpdx0, float x1, float p1,
                           float dpdx1) {
    const float t = (x - x0) / (x1 - x0);
    const float dp0 = dpdx0 * (x1 - x0);
    const float dp1 = dpdx1 * (x1 - x0);
    return p0 + t * (dp0 + t * (-2.0f * dp0 - dp1 - 3.0f * p0 + 3.0f * p1 +
                                t * (dp0 + dp1 + 2.0f * p0 - 2.0f * p1)));
}
inline float m_interpolate_to_zero(float x, float x0, float p0, float dpdx0, float x1) {
    const float t = (x - x0) / (x1 - x0);
    const float dp0 = dpdx0 * (x1 - x0);
    return p0 + t * (dp0 + t * (-2.0f * dp0 - 3.0f * p0 + t * (dp0 + 2.0f * p0)));
}

// ---------------------------------------------------------------------------
// geom.hpp -- 3-vector geometry for lk_ball and hbond
// ---------------------------------------------------------------------------
inline float m_interior_angle(float3 a, float3 b) {
    return 2.0f * atan2(length(cross(a, b)), length(a) * length(b) + dot(a, b));
}
inline float m_pt_interior_angle(float3 a, float3 b, float3 c) {
    return m_interior_angle(a - b, c - b);
}
inline float m_cos_interior_angle(float3 a, float3 b) {
    return dot(a, b) / (length(a) * length(b));
}
inline float m_dihedral_angle(float3 i, float3 j, float3 k, float3 l) {
    const float3 f = i - j;
    const float3 g = j - k;
    const float3 h = l - k;
    const float3 a = cross(f, g);
    const float3 b = cross(h, g);
    const float sgn = dot(g, cross(a, b)) >= 0.0f ? -1.0f : 1.0f;
    const float c = clamp(dot(a, b) / (length(a) * length(b)), -1.0f, 1.0f);
    return sgn * acos(c);
}

// ---------------------------------------------------------------------------
// ljlk.hpp -- fa_atr / fa_rep / fa_lk
// ---------------------------------------------------------------------------
inline float m_connectivity_weight(float bondedPathLength) {
    if (bondedPathLength > 4.0f) return 1.0f;
    if (bondedPathLength == 4.0f) return 0.2f;
    return 0.0f;
}

inline float m_lj_sigma(MAtom i, MAtom j, MLjlkGlobal g) {
    const bool i_don = has_flag(i.flags, F_DONOR);
    const bool i_hyd = has_flag(i.flags, F_HYDROXYL);
    const bool i_pol = has_flag(i.flags, F_POLARH);
    const bool i_acc = has_flag(i.flags, F_ACCEPTOR);
    const bool j_don = has_flag(j.flags, F_DONOR);
    const bool j_hyd = has_flag(j.flags, F_HYDROXYL);
    const bool j_pol = has_flag(j.flags, F_POLARH);
    const bool j_acc = has_flag(j.flags, F_ACCEPTOR);
    if ((i_don && !i_hyd && j_acc) || (j_don && !j_hyd && i_acc)) {
        return g.lj_hbond_dis;
    } else if ((i_don && i_hyd && j_acc) || (j_don && j_hyd && i_acc)) {
        return g.lj_hbond_OH_donor_dis;
    } else if ((i_pol && j_acc) || (j_pol && i_acc)) {
        return g.lj_hbond_hdis;
    }
    return i.lj_radius + j.lj_radius;
}

inline float m_vdwV(float dist, float sigma, float epsilon) {
    const float sd = sigma / dist;
    const float sd2 = sd * sd;
    const float sd6 = sd2 * sd2 * sd2;
    const float sd12 = sd6 * sd6;
    return epsilon * (sd12 - 2.0f * sd6);
}
inline float m_vdwDV(float dist, float sigma, float epsilon) {
    const float sd = sigma / dist;
    const float sd2 = sd * sd;
    const float sd6 = sd2 * sd2 * sd2;
    const float sd12 = sd6 * sd6;
    return epsilon * ((-12.0f * sd12 / dist) - (-12.0f * sd6 / dist));
}

// Returns float2(fa_ljatr, fa_ljrep).
inline float2 m_lj_score(float dist, float bondedPathLength, MAtom i, MAtom j,
                         MLjlkGlobal g) {
    const float cpolyDmax = 6.0f;
    if (dist > cpolyDmax) return float2(0.0f, 0.0f);

    const float sigma = m_lj_sigma(i, j, g);
    const float epsilon = sqrt(i.lj_wdepth * j.lj_wdepth);
    const float dLin = sigma * 0.6f;
    float cpolyDmin;
    if (sigma > 4.5f)
        cpolyDmin = sigma > cpolyDmax - 0.1f ? cpolyDmax - 0.1f : sigma;
    else
        cpolyDmin = 4.5f;

    const float weight = m_connectivity_weight(bondedPathLength);

    float vatr;
    if (dist > cpolyDmin) {
        const float v = m_vdwV(cpolyDmin, sigma, epsilon);
        const float dv = m_vdwDV(cpolyDmin, sigma, epsilon);
        vatr = m_interpolate_to_zero(dist, cpolyDmin, v, dv, cpolyDmax);
    } else if (dist > dLin) {
        vatr = m_vdwV(dist, sigma, epsilon);
    } else {
        const float v = m_vdwV(dLin, sigma, epsilon);
        const float dv = m_vdwDV(dLin, sigma, epsilon);
        vatr = v + dv * (dist - dLin);
    }

    float vrep;
    if (dist < sigma) {
        vrep = vatr + epsilon;
        vatr = -epsilon;
    } else {
        vrep = 0.0f;
    }
    return float2(weight * vatr, weight * vrep);
}

inline float m_fdesolvV(float dist, float ljRadiusI, float lkDgfreeI, float lkLambdaI,
                        float lkVolumeJ) {
    const float dr = dist - ljRadiusI;
    return (-lkVolumeJ * lkDgfreeI / (2.0f * PI_POW_1P5 * lkLambdaI) / (dist * dist) *
            exp(-(dr * dr) / (lkLambdaI * lkLambdaI)));
}
inline float m_fdesolvDV(float dist, float ljRadiusI, float lkDgfreeI, float lkLambdaI,
                         float lkVolumeJ) {
    const float dr = dist - ljRadiusI;
    const float expVal = exp(-(dr * dr) / (lkLambdaI * lkLambdaI));
    return (-lkVolumeJ * lkDgfreeI / (2.0f * PI_POW_1P5 * lkLambdaI) * expVal *
            ((-2.0f / (dist * dist * dist)) +
             (1.0f / (dist * dist) * -(2.0f * dist - 2.0f * ljRadiusI) /
              (lkLambdaI * lkLambdaI))));
}

// One direction of LK desolvation (i desolvated by j) with the smoothing splines. Reused
// by lk_ball as its isotropic core (tmol lk_isotropic_pair::V).
inline float m_lk_isotropic_pair(float dist, float ljSigmaIj, float ljRadiusI,
                                 float lkDgfreeI, float lkLambdaI, float lkVolumeJ) {
    const float dMin = ljSigmaIj * 0.89f;
    float cpolyCloseDmin = dMin * dMin - 1.45f;
    if (cpolyCloseDmin < 0.01f) cpolyCloseDmin = 0.01f;
    cpolyCloseDmin = sqrt(cpolyCloseDmin);
    const float cpolyCloseDmax = sqrt(dMin * dMin + 1.05f);
    const float cpolyFarDmin = 4.5f;
    const float cpolyFarDmax = 6.0f;

    if (dist > cpolyFarDmax) {
        return 0.0f;
    } else if (dist > cpolyFarDmin) {
        const float v = m_fdesolvV(cpolyFarDmin, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
        const float dv =
            m_fdesolvDV(cpolyFarDmin, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
        return m_interpolate_to_zero(dist, cpolyFarDmin, v, dv, cpolyFarDmax);
    } else if (dist > cpolyCloseDmax) {
        return m_fdesolvV(dist, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
    } else if (dist > cpolyCloseDmin) {
        const float vMin = m_fdesolvV(dMin, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
        const float vMax =
            m_fdesolvV(cpolyCloseDmax, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
        const float dvMax =
            m_fdesolvDV(cpolyCloseDmax, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
        return m_interpolate(dist, cpolyCloseDmin, vMin, 0.0f, cpolyCloseDmax, vMax, dvMax);
    }
    return m_fdesolvV(dMin, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
}

// fa_lk for one pair: both desolvation directions.
inline float m_lk_isotropic_score(float dist, float bondedPathLength, MAtom i, MAtom j,
                                  MLjlkGlobal g) {
    const float sigma = m_lj_sigma(i, j, g);
    const float weight = m_connectivity_weight(bondedPathLength);
    const float ij =
        m_lk_isotropic_pair(dist, sigma, i.lj_radius, i.lk_dgfree, i.lk_lambda, j.lk_volume);
    const float ji =
        m_lk_isotropic_pair(dist, sigma, j.lj_radius, j.lk_dgfree, j.lk_lambda, i.lk_volume);
    return weight * (ij + ji);
}

// ---------------------------------------------------------------------------
// elec.hpp -- fa_elec
// ---------------------------------------------------------------------------
inline float m_elec_eps(float dist, float D, float D0, float S) {
    return D - 0.5f * (D - D0) * (2.0f + 2.0f * dist * S + dist * dist * S * S) *
                   exp(-dist * S);
}
inline float m_elec_deps(float dist, float D, float D0, float S) {
    return 0.5f * (D - D0) * dist * dist * S * S * S * exp(-dist * S);
}

inline float m_elec(float dist, float eI, float eJ, float bondedPathLength,
                    MElecGlobal g) {
    const float D = g.D;
    const float D0 = g.D0;
    const float S = g.S;
    const float minDis = g.min_dis;
    const float maxDis = g.max_dis;
    const float lowPolyStart = minDis - 0.25f;
    const float lowPolyEnd = minDis + 0.25f;
    const float hiPolyStart = maxDis - 1.0f;
    const float hiPolyEnd = maxDis;

    const float weight = m_connectivity_weight(bondedPathLength);
    const float C2 = ELEC_C1 / (maxDis * m_elec_eps(maxDis, D, D0, S));
    const float eiej = eI * eJ;
    if (eiej == 0.0f) return 0.0f;

    float elecE = 0.0f;
    if (dist < lowPolyStart) {
        const float minDisScore = ELEC_C1 / (minDis * m_elec_eps(minDis, D, D0, S)) - C2;
        elecE = eiej * minDisScore;
    } else if (dist < lowPolyEnd) {
        const float minDisScore = ELEC_C1 / (minDis * m_elec_eps(minDis, D, D0, S)) - C2;
        const float epsElec = m_elec_eps(lowPolyEnd, D, D0, S);
        const float depsElec = m_elec_deps(lowPolyEnd, D, D0, S);
        const float dmaxElec = eiej * (ELEC_C1 / (lowPolyEnd * epsElec) - C2);
        // Note: tmol passes deps_elec_d_dist (the dielectric derivative) as the endpoint
        // slope here, not the energy derivative. Ported verbatim.
        elecE = m_interpolate(dist, lowPolyStart, eiej * minDisScore, 0.0f, lowPolyEnd,
                              dmaxElec, depsElec);
    } else if (dist < hiPolyStart) {
        const float epsElec = m_elec_eps(dist, D, D0, S);
        elecE = eiej * (ELEC_C1 / (dist * epsElec) - C2);
    } else if (dist < hiPolyEnd) {
        const float epsElec = m_elec_eps(hiPolyStart, D, D0, S);
        const float depsElec = m_elec_deps(hiPolyStart, D, D0, S);
        const float dminElec = eiej * (ELEC_C1 / (hiPolyStart * epsElec) - C2);
        const float dminElecDDist =
            (-ELEC_C1 * eiej * (epsElec + hiPolyStart * depsElec)) /
            (hiPolyStart * hiPolyStart * epsElec * epsElec);
        elecE = m_interpolate_to_zero(dist, hiPolyStart, dminElec, dminElecDDist, hiPolyEnd);
    }
    return weight * elecE;
}

// ---------------------------------------------------------------------------
// lk_ball.hpp -- lk_ball_iso / lk_ball / lk_bridge / lk_bridge_uncpl
// ---------------------------------------------------------------------------
inline float m_lkb_sq(float v) { return v * v; }

// lk_fraction::V -- directional desolvation ramp of occluder J against polar I's waters.
inline float m_lk_fraction(device const float* waters, uint wmask, int polar,
                           float3 j, float ljRadiusJ) {
    float d2Low = m_lkb_sq(1.4f + ljRadiusJ) - RAMP_WIDTH_A2;
    if (d2Low < 0.0f) d2Low = 0.0f;

    // TODO(metal-precision): float exp/log here; CPU sums these in double.
    float wtedD2Delta = 0.0f;
    for (int w = 0; w < MAX_WATER; ++w) {
        if (!has_flag(wmask, 1u << uint(w))) continue;
        const int base = (polar * MAX_WATER + w) * 3;
        const float3 wp = float3(waters[base + 0], waters[base + 1], waters[base + 2]);
        const float3 d = j - wp;
        const float d2Delta = dot(d, d) - d2Low;
        wtedD2Delta += exp(-d2Delta);
    }
    wtedD2Delta = -log(wtedD2Delta);

    if (wtedD2Delta < 0.0f) return 1.0f;
    if (wtedD2Delta < RAMP_WIDTH_A2)
        return m_lkb_sq(1.0f - m_lkb_sq(wtedD2Delta / RAMP_WIDTH_A2));
    return 0.0f;
}

// lk_bridge_fraction::V -- water-overlap (bridging) term between two polar atoms.
inline float m_lk_bridge_fraction(float3 i, float3 j, device const float* waters,
                                  uint wmaskI, int polarI, uint wmaskJ, int polarJ,
                                  float lkbWaterDist) {
    // TODO(metal-precision): float exp/log here; CPU sums these in double.
    float wtedD2Delta = 0.0f;
    for (int a = 0; a < MAX_WATER; ++a) {
        if (!has_flag(wmaskI, 1u << uint(a))) continue;
        const int baseA = (polarI * MAX_WATER + a) * 3;
        const float3 wa = float3(waters[baseA + 0], waters[baseA + 1], waters[baseA + 2]);
        for (int b = 0; b < MAX_WATER; ++b) {
            if (!has_flag(wmaskJ, 1u << uint(b))) continue;
            const int baseB = (polarJ * MAX_WATER + b) * 3;
            const float3 wb =
                float3(waters[baseB + 0], waters[baseB + 1], waters[baseB + 2]);
            const float3 d = wa - wb;
            const float d2Delta = dot(d, d) - OVERLAP_GAP_A2;
            wtedD2Delta += exp(-d2Delta);
        }
    }
    wtedD2Delta = -log(wtedD2Delta);

    float overlapfrac;
    if (wtedD2Delta > OVERLAP_WIDTH_A2)
        overlapfrac = 0.0f;
    else
        overlapfrac = m_lkb_sq(1.0f - m_lkb_sq(wtedD2Delta / OVERLAP_WIDTH_A2));

    const float overlapTargetLen2 = (8.0f / 3.0f) * m_lkb_sq(lkbWaterDist);
    const float3 dij = i - j;
    const float overlapLen2 = dot(dij, dij);
    const float baseDelta = fabs(overlapLen2 - overlapTargetLen2);

    float anglefrac;
    if (baseDelta > ANGLE_OVERLAP_A2)
        anglefrac = 0.0f;
    else
        anglefrac = m_lkb_sq(1.0f - m_lkb_sq(baseDelta / ANGLE_OVERLAP_A2));

    return overlapfrac * anglefrac;
}

// lk_ball_score::V for one ordered (polar I, occluder J) pair. Returns float4
// (lk_ball_iso, lk_ball, lk_bridge, lk_bridge_uncpl). Mirrors lkBallScore + scoreDir's
// `if (!wi.present[0]) return;` guard exactly.
inline float4 m_lk_ball_score(int polarI, int occJ, float3 xi, float3 xj,
                              device const float* waters, uint wmaskI, uint wmaskJ,
                              float bondedPathLength, float dist, MAtom ti, MAtom tj,
                              MLkBallGlobal g) {
    const float4 zero = float4(0.0f);
    if (bondedPathLength <= 3.0f) return zero;
    if (!has_flag(wmaskI, 1u)) return zero;  // polar I needs water[0]
    if (dist >= g.distance_threshold) return zero;

    MLjlkGlobal lg;
    lg.lj_hbond_dis = g.lj_hbond_dis;
    lg.lj_hbond_OH_donor_dis = g.lj_hbond_OH_donor_dis;
    lg.lj_hbond_hdis = g.lj_hbond_hdis;
    const float sigma = m_lj_sigma(ti, tj, lg);
    const float lkIso =
        m_connectivity_weight(bondedPathLength) *
        m_lk_isotropic_pair(dist, sigma, ti.lj_radius, ti.lk_dgfree, ti.lk_lambda,
                            tj.lk_volume);
    const float fracDesolv = m_lk_fraction(waters, wmaskI, polarI, xj, tj.lj_radius);
    float fracOverlap = 0.0f;
    if (has_flag(tj.flags, F_DONOR) || has_flag(tj.flags, F_ACCEPTOR)) {
        fracOverlap = m_lk_bridge_fraction(xi, xj, waters, wmaskI, polarI, wmaskJ, occJ,
                                           g.lkb_water_dist);
    }
    return float4(lkIso, lkIso * fracDesolv, lkIso * fracOverlap, fracOverlap / 2.0f);
}

// ---------------------------------------------------------------------------
// hbond.hpp -- hbond
// ---------------------------------------------------------------------------
inline float m_bound_poly(float x, MPoly p) {
    if (x < p.range[0]) return p.bound[0];
    if (x > p.range[1]) return p.bound[1];
    float v = p.coeffs[0];
    for (int i = 1; i < 11; ++i) v = v * x + p.coeffs[i];  // TODO(metal-precision)
    return v;
}

inline float m_bah_angle_base_form(float3 b, float3 a, float3 h, MPoly poly) {
    const float3 ah = h - a;
    const float3 ba = a - b;
    return m_bound_poly(m_cos_interior_angle(ah, ba), poly);
}

inline float m_bah_angle(float3 b, float3 b0, float3 a, float3 h, int hyb, MPoly poly,
                         float sp3SoftmaxFade) {
    if (hyb == HB_SP2) return m_bah_angle_base_form(b, a, h, poly);
    if (hyb == HB_RING) return m_bah_angle_base_form((b + b0) * 0.5f, a, h, poly);
    // sp3: softmax over the two bases. TODO(metal-precision): float log-sum-exp.
    const float pxH = m_bah_angle_base_form(b, a, h, poly);
    const float pxH0 = m_bah_angle_base_form(b0, a, h, poly);
    return log(exp(pxH * sp3SoftmaxFade) + exp(pxH0 * sp3SoftmaxFade)) / sp3SoftmaxFade;
}

inline float m_sp2chi_energy(float ang, float chi, float d, float m, float l) {
    const float pi = HB_PI;
    const float H = 0.5f * (cos(2.0f * chi) + 1.0f);
    if (ang > (pi * 2.0f) / 3.0f) {
        const float F = (d / 2.0f) * cos(3.0f * (pi - ang)) + d / 2.0f - 0.5f;
        const float G = d - 0.5f;
        return H * F + (1.0f - H) * G;
    } else if (ang >= pi * (2.0f / 3.0f - l)) {
        const float outerRise = cos(pi - ((pi * 2.0f) / 3.0f - ang) / l);
        const float F = (m / 2.0f) * outerRise + m / 2.0f - 0.5f;
        const float G = ((m - d) / 2.0f) * outerRise + (m - d) / 2.0f + d - 0.5f;
        return H * F + (1.0f - H) * G;
    }
    return m - 0.5f;
}

inline float m_b0bah_chi(float3 b0, float3 b, float3 a, float3 h, int hyb,
                         MHbondGlobal g) {
    if (hyb != HB_SP2) return 0.0f;
    const float bah = m_pt_interior_angle(b, a, h);
    const float b0bah = m_dihedral_angle(b0, b, a, h);
    return m_sp2chi_energy(bah, b0bah, g.hb_sp2_BAH180_rise, g.hb_sp2_range_span,
                           g.hb_sp2_outer_width);
}

inline float m_hbond_score(float3 d, float3 h, float3 a, float3 b, float3 b0, int hyb,
                           float adWeight, MPoly AHdist, MPoly cosBAH, MPoly cosAHD,
                           MHbondGlobal g) {
    if (distance(h, a) >= g.max_ha_dis) return 0.0f;

    float e = 0.0f;
    e += m_bound_poly(distance(a, h), AHdist);
    e += m_bound_poly(m_pt_interior_angle(a, h, d), cosAHD);
    e += m_bah_angle(b, b0, a, h, hyb, cosBAH, g.hb_sp3_softmax_fade);
    e += m_b0bah_chi(b0, b, a, h, hyb, g);

    e *= adWeight;

    // Truncate and fade [-0.1, 0.1] to [-0.1, 0.0].
    if (e > 0.1f) return 0.0f;
    if (e > -0.1f) return -0.025f + 0.5f * e - 2.5f * e * e;
    return e;
}

// ===========================================================================
// Kernels: one thread per pair. Output is per-pair, per-subterm; dispatch.cpp does the
// block-pair scatter-add host-side in double (identical to driver.hpp's `add` lambda).
// ===========================================================================

// ljlk + fa_elec. out[4*p + {0:fa_ljatr, 1:fa_ljrep, 2:fa_lk, 3:fa_elec}].
kernel void k_ljlk_elec(device const float* coord [[buffer(0)]],
                        device const MAtom* atom [[buffer(1)]],
                        device const int* pair_i [[buffer(2)]],
                        device const int* pair_j [[buffer(3)]],
                        device const int* sep_ljlk [[buffer(4)]],
                        device const int* sep_elec [[buffer(5)]],
                        constant int& n_pairs [[buffer(6)]],
                        constant MLjlkGlobal& ljlk_g [[buffer(7)]],
                        constant MElecGlobal& elec_g [[buffer(8)]],
                        device float* out [[buffer(9)]],
                        uint gid [[thread_position_in_grid]]) {
    const int p = (int)gid;
    if (p >= n_pairs) return;
    const int i = pair_i[p], j = pair_j[p];
    const float3 xi = load_xyz(coord, i);
    const float3 xj = load_xyz(coord, j);
    const float d = distance(xi, xj);

    float ljatr = 0.0f, ljrep = 0.0f, lk = 0.0f, el = 0.0f;
    if (d > 0.0f) {
        const MAtom ai = atom[i];
        const MAtom aj = atom[j];
        el = m_elec(d, ai.charge, aj.charge, (float)sep_elec[p], elec_g);
        const float2 lj = m_lj_score(d, (float)sep_ljlk[p], ai, aj, ljlk_g);
        ljatr = lj.x;
        ljrep = lj.y;
        if (has_flag(ai.flags, F_HEAVY) && has_flag(aj.flags, F_HEAVY))
            lk = m_lk_isotropic_score(d, (float)sep_ljlk[p], ai, aj, ljlk_g);
    }
    out[4 * p + 0] = ljatr;
    out[4 * p + 1] = ljrep;
    out[4 * p + 2] = lk;
    out[4 * p + 3] = el;
}

// lk_ball (both directions). out[8*p + {0..3}] = dir(i polar, j occ) subterms;
// out[8*p + {4..7}] = dir(j polar, i occ) subterms. Each block is
// (lk_ball_iso, lk_ball, lk_bridge, lk_bridge_uncpl).
kernel void k_lk_ball(device const float* coord [[buffer(0)]],
                      device const MAtom* atom [[buffer(1)]],
                      device const float* waters [[buffer(2)]],
                      device const uint* water_mask [[buffer(3)]],
                      device const int* pair_i [[buffer(4)]],
                      device const int* pair_j [[buffer(5)]],
                      device const int* sep_ljlk [[buffer(6)]],
                      constant int& n_pairs [[buffer(7)]],
                      constant MLjlkGlobal& ljlk_g [[buffer(8)]],
                      constant MLkBallGlobal& lkb_g [[buffer(9)]],
                      device float* out [[buffer(10)]],
                      uint gid [[thread_position_in_grid]]) {
    const int p = (int)gid;
    if (p >= n_pairs) return;
    for (int k = 0; k < 8; ++k) out[8 * p + k] = 0.0f;

    const int i = pair_i[p], j = pair_j[p];
    const MAtom ai = atom[i];
    const MAtom aj = atom[j];
    if (!has_flag(ai.flags, F_HEAVY) || !has_flag(aj.flags, F_HEAVY)) return;
    const float3 xi = load_xyz(coord, i);
    const float3 xj = load_xyz(coord, j);
    const float d = distance(xi, xj);
    if (d <= 0.0f) return;
    const float bpl = (float)sep_ljlk[p];

    // scoreDir(ai polar, aj occ): the CPU early-outs the whole direction if !wi.present[0].
    if (has_flag(water_mask[i], 1u)) {
        const float4 s = m_lk_ball_score(i, j, xi, xj, waters, water_mask[i],
                                         water_mask[j], bpl, d, ai, aj, lkb_g);
        out[8 * p + 0] = s.x;
        out[8 * p + 1] = s.y;
        out[8 * p + 2] = s.z;
        out[8 * p + 3] = s.w;
    }
    // scoreDir(aj polar, ai occ).
    if (has_flag(water_mask[j], 1u)) {
        const float4 s = m_lk_ball_score(j, i, xj, xi, waters, water_mask[j],
                                         water_mask[i], bpl, d, aj, ai, lkb_g);
        out[8 * p + 4] = s.x;
        out[8 * p + 5] = s.y;
        out[8 * p + 6] = s.z;
        out[8 * p + 7] = s.w;
    }
}

// hbond. One thread per donor-H / acceptor pair. out[p] = hbond energy (0 for the
// count-pair exclusion sep < 5, applied host-side, and outside max_ha_dis).
kernel void k_hbond(device const float* hp_H [[buffer(0)]],
                    device const float* hp_A [[buffer(1)]],
                    device const float* hp_D [[buffer(2)]],
                    device const float* hp_B [[buffer(3)]],
                    device const float* hp_B0 [[buffer(4)]],
                    device const int* hyb [[buffer(5)]],
                    device const float* ad_weight [[buffer(6)]],
                    device const MPoly* AHdist [[buffer(7)]],
                    device const MPoly* cosBAH [[buffer(8)]],
                    device const MPoly* cosAHD [[buffer(9)]],
                    constant int& n_pairs [[buffer(10)]],
                    constant MHbondGlobal& g [[buffer(11)]],
                    device float* out [[buffer(12)]],
                    uint gid [[thread_position_in_grid]]) {
    const int p = (int)gid;
    if (p >= n_pairs) return;
    const float3 H = load_xyz(hp_H, p);
    const float3 A = load_xyz(hp_A, p);
    const float3 D = load_xyz(hp_D, p);
    const float3 B = load_xyz(hp_B, p);
    const float3 B0 = load_xyz(hp_B0, p);
    out[p] = m_hbond_score(D, H, A, B, B0, hyb[p], ad_weight[p], AHdist[p], cosBAH[p],
                           cosAHD[p], g);
}
