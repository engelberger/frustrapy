// CUDA forward kernels for the torch-free all-atom (ref2015) energy path (M6).
//
// Compiled only when FRUSTRAMOL_TMOL_CUDA is ON and nvcc is available (the dev container
// has no GPU, so it is OFF by default and the CPU lane stays the in-container path). The
// device math MIRRORS the parity-gated CPU kernel headers term-for-term:
//
//     include/frustramol_tmol/interpolate.hpp  -> d_interpolate / d_interpolateToZero
//     include/frustramol_tmol/geom.hpp         -> v_* / interiorAngle / dihedralAngle ...
//     include/frustramol_tmol/ljlk.hpp         -> ljSigma / ljScore / lkIsotropic* ...
//     include/frustramol_tmol/elec.hpp         -> elecEps / elec
//     include/frustramol_tmol/lk_ball.hpp      -> lkFraction / lkBridgeFraction / lkBallScore
//     include/frustramol_tmol/hbond.hpp        -> boundPoly / bahAngle / hbondScore
//
// Each device function sits directly under the header + function it reproduces, so the
// correspondence is auditable line-for-line. There is no torch and no autograd: single-
// point forward only, exactly like the CPU drivers. The accumulation matches the CPU
// driver's block-pair convention (energy placed at [min(b1,b2), max(b1,b2)]); the only
// numerical difference from the CPU result is the floating-point order of the atomicAdd
// reduction (last ULPs), never a formula difference, well within the tmol gate tolerance.
//
// This file has not been compiled or run in the dev container (no nvcc/GPU). It is written
// to mirror the CPU headers exactly; the maintainer builds and validates parity on a CUDA
// host (Colab / cluster) against the CPU driver AND the tmol oracle, as the existing AWSEM
// native CUDA lane (native/src/cuda/kernels.cu) and the Metal lane do. No GPU timing is
// asserted here. Conventions follow the CUDA C++ Best Practices Guide: structure-of-arrays
// global access, upload once, one thread per scored pair. See native_tmol/NOTICE.

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "frustramol_tmol/cuda.hpp"
#include "frustramol_tmol/types.hpp"

namespace frustramol_tmol {

namespace {

constexpr int MW = MAX_WATER;  // 4, from lk_ball.hpp

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                                 cudaGetErrorString(e));
    }
}

// RAII device buffer (no naked cudaMalloc/cudaFree), mirroring native/src/cuda/kernels.cu.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t n) : n_(n) {
        cuda_check(cudaMalloc(&ptr_, (n ? n : 1) * sizeof(T)), "cudaMalloc");
    }
    explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) {
        if (!host.empty())
            cuda_check(cudaMemcpy(ptr_, host.data(), n_ * sizeof(T),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy H2D");
    }
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }
    std::size_t size() const { return n_; }
    void zero() { cuda_check(cudaMemset(ptr_, 0, (n_ ? n_ : 1) * sizeof(T)), "cudaMemset"); }
    void to_host(std::vector<T>& host) const {
        host.resize(n_);
        if (n_)
            cuda_check(cudaMemcpy(host.data(), ptr_, n_ * sizeof(T),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy D2H");
    }

private:
    T* ptr_ = nullptr;
    std::size_t n_ = 0;
};

// ---------------------------------------------------------------------------
// Device-side parameter bundles (POD copies of the host *GlobalParams structs).
// ---------------------------------------------------------------------------
struct DLjlkGlobal {  // ljlk.hpp LjlkGlobalParams
    double lj_hbond_dis, lj_hbond_OH_donor_dis, lj_hbond_hdis;
};
struct DElecGlobal {  // elec.hpp ElecGlobalParams
    double D, D0, S, min_dis, max_dis;
};
struct DLkBallGlobal {  // lk_ball.hpp LkBallGlobalParams
    double lj_hbond_dis, lj_hbond_OH_donor_dis, lj_hbond_hdis, lkb_water_dist,
        distance_threshold;
};
struct DHbondGlobal {  // hbond.hpp HBondGlobalParams
    double hb_sp2_range_span, hb_sp2_BAH180_rise, hb_sp2_outer_width,
        hb_sp3_softmax_fade, threshold_distance, max_ha_dis;
};

// Per-atom ljlk type row (ljlk.hpp LjlkTypeParams), loaded from the SoA inside a kernel.
struct DLjlk {
    double lj_radius, lj_wdepth, lk_dgfree, lk_lambda, lk_volume;
    bool is_donor, is_hydroxyl, is_polarh, is_acceptor;
};

// One bounded hbond polynomial (hbond.hpp HBondPoly), loaded from the SoA per pair.
struct DHBPoly {
    double coeffs[11];
    double range[2];
    double bound[2];
};

// ---------------------------------------------------------------------------
// Double atomicAdd (intrinsic on sm_60+, CAS fallback below; standard idiom).
// ---------------------------------------------------------------------------
__device__ inline double atomicAddD(double* addr, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(addr, val);
#else
    unsigned long long int* a = reinterpret_cast<unsigned long long int*>(addr);
    unsigned long long int old = *a, assumed;
    do {
        assumed = old;
        old = atomicCAS(a, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

// Block-pair placement, mirroring driver.hpp detail::accumulate's `add` lambda: skip
// exact zero, place at [min(b1,b2), max(b1,b2)] of a flattened nBlocks x nBlocks matrix.
__device__ inline void addBP(double* mat, int nBlocks, int b1, int b2, double e) {
    if (e == 0.0) return;
    const int lo = b1 < b2 ? b1 : b2;
    const int hi = b1 < b2 ? b2 : b1;
    atomicAddD(&mat[static_cast<std::size_t>(lo) * nBlocks + hi], e);
}

// ===========================================================================
// geom.hpp  (single-point forward geometry; Vec3 -> double3)
// ===========================================================================
__device__ inline double3 v_sub(double3 a, double3 b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ inline double3 v_add(double3 a, double3 b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ inline double v_dot(double3 a, double3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ inline double3 v_cross(double3 a, double3 b) {
    return make_double3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                        a.x * b.y - a.y * b.x);
}
__device__ inline double v_norm(double3 a) { return sqrt(v_dot(a, a)); }
__device__ inline double v_dist(double3 a, double3 b) { return v_norm(v_sub(a, b)); }
__device__ inline double3 v_scale(double3 a, double s) {
    return make_double3(a.x * s, a.y * s, a.z * s);
}
// interior_angle::V(A, B): 2*atan2(|AxB|, |A||B| + A.B).
__device__ inline double interiorAngle(double3 a, double3 b) {
    const double3 cr = v_cross(a, b);
    return 2.0 * atan2(v_norm(cr), v_norm(a) * v_norm(b) + v_dot(a, b));
}
// pt_interior_angle::V(A, B, C): interior angle at vertex B.
__device__ inline double ptInteriorAngle(double3 a, double3 b, double3 c) {
    return interiorAngle(v_sub(a, b), v_sub(c, b));
}
// cos_interior_angle::V(A, B).
__device__ inline double cosInteriorAngle(double3 a, double3 b) {
    return v_dot(a, b) / (v_norm(a) * v_norm(b));
}
// dihedral_angle::V(I, J, K, L) via Blondel-Karplus.
__device__ inline double dihedralAngle(double3 i, double3 j, double3 k, double3 l) {
    const double3 f = v_sub(i, j);
    const double3 g = v_sub(j, k);
    const double3 h = v_sub(l, k);
    const double3 a = v_cross(f, g);
    const double3 b = v_cross(h, g);
    const double sign = v_dot(g, v_cross(a, b)) >= 0 ? -1.0 : 1.0;
    const double c = fmax(-1.0, fmin(v_dot(a, b) / (v_norm(a) * v_norm(b)), 1.0));
    return sign * acos(c);
}

// ===========================================================================
// interpolate.hpp
// ===========================================================================
__device__ inline double d_interpolate(double x, double x0, double p0, double dpdx0,
                                        double x1, double p1, double dpdx1) {
    const double t = (x - x0) / (x1 - x0);
    const double dp0 = dpdx0 * (x1 - x0);
    const double dp1 = dpdx1 * (x1 - x0);
    return p0 + t * (dp0 + t * (-2 * dp0 - dp1 - 3 * p0 + 3 * p1 +
                                t * (dp0 + dp1 + 2 * p0 - 2 * p1)));
}
__device__ inline double d_interpolateToZero(double x, double x0, double p0,
                                             double dpdx0, double x1) {
    const double t = (x - x0) / (x1 - x0);
    const double dp0 = dpdx0 * (x1 - x0);
    return p0 + t * (dp0 + t * (-2 * dp0 - 3 * p0 + t * (dp0 + 2 * p0)));
}

// ===========================================================================
// ljlk.hpp
// ===========================================================================
__device__ const double PI_POW_1P5 = 5.56832799683;

// connectivity_weight: 0 for <4 bonds, 0.2 at exactly 4, 1 beyond.
__device__ inline double connectivityWeight(double bondedPathLength) {
    if (bondedPathLength > 4) return 1.0;
    if (bondedPathLength == 4) return 0.2;
    return 0.0;
}
// ljSigma with the donor/acceptor hydrogen-bond overrides.
__device__ inline double ljSigma(const DLjlk& i, const DLjlk& j,
                                 const DLjlkGlobal& g) {
    if ((i.is_donor && !i.is_hydroxyl && j.is_acceptor) ||
        (j.is_donor && !j.is_hydroxyl && i.is_acceptor)) {
        return g.lj_hbond_dis;
    } else if ((i.is_donor && i.is_hydroxyl && j.is_acceptor) ||
               (j.is_donor && j.is_hydroxyl && i.is_acceptor)) {
        return g.lj_hbond_OH_donor_dis;
    } else if ((i.is_polarh && j.is_acceptor) || (j.is_polarh && i.is_acceptor)) {
        return g.lj_hbond_hdis;
    }
    return i.lj_radius + j.lj_radius;
}
__device__ inline double vdwV(double dist, double sigma, double epsilon) {
    const double sd = sigma / dist;
    const double sd2 = sd * sd;
    const double sd6 = sd2 * sd2 * sd2;
    const double sd12 = sd6 * sd6;
    return epsilon * (sd12 - 2 * sd6);
}
__device__ inline double vdwDV(double dist, double sigma, double epsilon) {
    const double sd = sigma / dist;
    const double sd2 = sd * sd;
    const double sd6 = sd2 * sd2 * sd2;
    const double sd12 = sd6 * sd6;
    return epsilon * ((-12.0 * sd12 / dist) - (-12.0 * sd6 / dist));
}
// fa_ljatr / fa_ljrep for one pair, returned through atr/rep out-params.
__device__ inline void ljScore(double dist, double bondedPathLength, const DLjlk& i,
                               const DLjlk& j, const DLjlkGlobal& g, double& outAtr,
                               double& outRep) {
    const double cpolyDmax = 6.0;
    if (dist > cpolyDmax) {
        outAtr = 0.0;
        outRep = 0.0;
        return;
    }
    const double sigma = ljSigma(i, j, g);
    const double epsilon = sqrt(i.lj_wdepth * j.lj_wdepth);
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
        vatr = d_interpolateToZero(dist, cpolyDmin, v, dv, cpolyDmax);
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
    outAtr = weight * vatr;
    outRep = weight * vrep;
}
// Lazaridis-Karplus desolvation of i by the volume of j: value and d/ddist.
__device__ inline double fDesolvV(double dist, double ljRadiusI, double lkDgfreeI,
                                  double lkLambdaI, double lkVolumeJ) {
    const double dr = dist - ljRadiusI;
    return (-lkVolumeJ * lkDgfreeI / (2 * PI_POW_1P5 * lkLambdaI) / (dist * dist) *
            exp(-(dr * dr) / (lkLambdaI * lkLambdaI)));
}
__device__ inline double fDesolvDV(double dist, double ljRadiusI, double lkDgfreeI,
                                   double lkLambdaI, double lkVolumeJ) {
    const double dr = dist - ljRadiusI;
    const double expVal = exp(-(dr * dr) / (lkLambdaI * lkLambdaI));
    return (-lkVolumeJ * lkDgfreeI / (2 * PI_POW_1P5 * lkLambdaI) * expVal *
            ((-2 / (dist * dist * dist)) +
             (1 / (dist * dist) * -(2 * dist - 2 * ljRadiusI) /
              (lkLambdaI * lkLambdaI))));
}
// One direction of the LK desolvation spline (lk_isotropic_pair::V).
__device__ inline double lkIsotropicPair(double dist, double ljSigmaIj,
                                         double ljRadiusI, double lkDgfreeI,
                                         double lkLambdaI, double lkVolumeJ) {
    const double dMin = ljSigmaIj * 0.89;
    double cpolyCloseDmin = dMin * dMin - 1.45;
    if (cpolyCloseDmin < 0.01) cpolyCloseDmin = 0.01;
    cpolyCloseDmin = sqrt(cpolyCloseDmin);
    const double cpolyCloseDmax = sqrt(dMin * dMin + 1.05);
    const double cpolyFarDmin = 4.5;
    const double cpolyFarDmax = 6.0;

    if (dist > cpolyFarDmax) {
        return 0.0;
    } else if (dist > cpolyFarDmin) {
        const double v =
            fDesolvV(cpolyFarDmin, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
        const double dv =
            fDesolvDV(cpolyFarDmin, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
        return d_interpolateToZero(dist, cpolyFarDmin, v, dv, cpolyFarDmax);
    } else if (dist > cpolyCloseDmax) {
        return fDesolvV(dist, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
    } else if (dist > cpolyCloseDmin) {
        const double vMin = fDesolvV(dMin, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
        const double vMax =
            fDesolvV(cpolyCloseDmax, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
        const double dvMax =
            fDesolvDV(cpolyCloseDmax, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
        return d_interpolate(dist, cpolyCloseDmin, vMin, 0.0, cpolyCloseDmax, vMax,
                             dvMax);
    }
    return fDesolvV(dMin, ljRadiusI, lkDgfreeI, lkLambdaI, lkVolumeJ);
}
// fa_lk for one pair: both desolvation directions, with the count-pair weight.
__device__ inline double lkIsotropicScore(double dist, double bondedPathLength,
                                          const DLjlk& i, const DLjlk& j,
                                          const DLjlkGlobal& g) {
    const double sigma = ljSigma(i, j, g);
    const double weight = connectivityWeight(bondedPathLength);
    const double ij = lkIsotropicPair(dist, sigma, i.lj_radius, i.lk_dgfree,
                                      i.lk_lambda, j.lk_volume);
    const double ji = lkIsotropicPair(dist, sigma, j.lj_radius, j.lk_dgfree,
                                      j.lk_lambda, i.lk_volume);
    return weight * (ij + ji);
}

// ===========================================================================
// elec.hpp
// ===========================================================================
__device__ const double ELEC_C1 = 322.0637;

__device__ inline double elecEps(double dist, double D, double D0, double S) {
    return D - 0.5 * (D - D0) * (2 + 2 * dist * S + dist * dist * S * S) *
                   exp(-dist * S);
}
__device__ inline double elecDepsDdist(double dist, double D, double D0, double S) {
    return 0.5 * (D - D0) * dist * dist * S * S * S * exp(-dist * S);
}
// fa_elec energy for one atom pair.
__device__ inline double elec(double dist, double eI, double eJ,
                              double bondedPathLength, const DElecGlobal& g) {
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
        // tmol passes deps_elec_d_dist (the dielectric derivative) as the endpoint
        // slope here, not the energy derivative. Ported verbatim from elec.hpp.
        elecE = d_interpolate(dist, lowPolyStart, eiej * minDisScore, 0.0, lowPolyEnd,
                              dmaxElec, depsElec);
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
        elecE =
            d_interpolateToZero(dist, hiPolyStart, dminElec, dminElecDDist, hiPolyEnd);
    }
    return weight * elecE;
}

// ===========================================================================
// lk_ball.hpp
// ===========================================================================
__device__ const double OVERLAP_GAP_A2 = 0.5;
__device__ const double OVERLAP_WIDTH_A2 = 2.6;
__device__ const double ANGLE_OVERLAP_A2 = 2.8 * 2.6;  // 2.8 * OVERLAP_WIDTH_A2
__device__ const double RAMP_WIDTH_A2 = 3.709;

__device__ inline double lkb_sq(double v) { return v * v; }

// lk_fraction::V - directional desolvation ramp of occluder J against polar I's waters.
// `wpos`/`wpres` are the polar atom's MW water slots (flattened: wpos[w*3+{0,1,2}]).
__device__ inline double lkFraction(const double* wpos, const bool* wpres, double3 j,
                                    double ljRadiusJ) {
    double d2Low = lkb_sq(1.4 + ljRadiusJ) - RAMP_WIDTH_A2;
    if (d2Low < 0.0) d2Low = 0.0;

    double wtedD2Delta = 0;
    for (int w = 0; w < MW; w++) {
        if (!wpres[w]) continue;
        const double3 wp = make_double3(wpos[w * 3 + 0], wpos[w * 3 + 1], wpos[w * 3 + 2]);
        const double3 d = v_sub(j, wp);
        const double d2Delta = v_dot(d, d) - d2Low;
        wtedD2Delta += exp(-d2Delta);
    }
    wtedD2Delta = -log(wtedD2Delta);

    if (wtedD2Delta < 0) return 1;
    if (wtedD2Delta < RAMP_WIDTH_A2)
        return lkb_sq(1 - lkb_sq(wtedD2Delta / RAMP_WIDTH_A2));
    return 0;
}

// lk_bridge_fraction::V - water-overlap (bridging) term between two polar atoms.
__device__ inline double lkBridgeFraction(double3 i, double3 j, const double* wiPos,
                                          const bool* wiPres, const double* wjPos,
                                          const bool* wjPres, double lkbWaterDist) {
    double wtedD2Delta = 0.0;
    for (int a = 0; a < MW; a++) {
        if (!wiPres[a]) continue;
        const double3 wa =
            make_double3(wiPos[a * 3 + 0], wiPos[a * 3 + 1], wiPos[a * 3 + 2]);
        for (int b = 0; b < MW; b++) {
            if (!wjPres[b]) continue;
            const double3 wb =
                make_double3(wjPos[b * 3 + 0], wjPos[b * 3 + 1], wjPos[b * 3 + 2]);
            const double3 d = v_sub(wa, wb);
            const double d2Delta = v_dot(d, d) - OVERLAP_GAP_A2;
            wtedD2Delta += exp(-d2Delta);
        }
    }
    wtedD2Delta = -log(wtedD2Delta);

    double overlapfrac;
    if (wtedD2Delta > OVERLAP_WIDTH_A2)
        overlapfrac = 0;
    else
        overlapfrac = lkb_sq(1 - lkb_sq(wtedD2Delta / OVERLAP_WIDTH_A2));

    const double overlapTargetLen2 = (8.0 / 3.0) * lkb_sq(lkbWaterDist);
    const double3 dij = v_sub(i, j);
    const double overlapLen2 = v_dot(dij, dij);
    const double baseDelta = fabs(overlapLen2 - overlapTargetLen2);

    double anglefrac;
    if (baseDelta > ANGLE_OVERLAP_A2)
        anglefrac = 0;
    else
        anglefrac = lkb_sq(1 - lkb_sq(baseDelta / ANGLE_OVERLAP_A2));

    return overlapfrac * anglefrac;
}

struct DLkBallSubterms {
    double lk_ball_iso, lk_ball, lk_bridge, lk_bridge_uncpl;
};

// lk_ball_score::V for one ordered (polar I, occluder J) pair. wi*/wj* are the I and J
// water slots; `wiPres[0]` must be true (the polar atom needs at least one water).
__device__ inline DLkBallSubterms lkBallScore(double3 i, double3 j, const double* wiPos,
                                              const bool* wiPres, const double* wjPos,
                                              const bool* wjPres,
                                              double bondedPathLength, double dist,
                                              const DLjlk& ti, const DLjlk& tj,
                                              const DLkBallGlobal& g) {
    DLkBallSubterms zero{0, 0, 0, 0};
    if (bondedPathLength <= 3) return zero;
    if (!wiPres[0]) return zero;
    if (dist >= g.distance_threshold) return zero;

    DLjlkGlobal lg{g.lj_hbond_dis, g.lj_hbond_OH_donor_dis, g.lj_hbond_hdis};
    const double sigma = ljSigma(ti, tj, lg);
    const double lkIso =
        connectivityWeight(bondedPathLength) *
        lkIsotropicPair(dist, sigma, ti.lj_radius, ti.lk_dgfree, ti.lk_lambda,
                        tj.lk_volume);
    const double fracDesolv = lkFraction(wiPos, wiPres, j, tj.lj_radius);
    double fracOverlap = 0.0;
    if (tj.is_donor || tj.is_acceptor) {
        fracOverlap =
            lkBridgeFraction(i, j, wiPos, wiPres, wjPos, wjPres, g.lkb_water_dist);
    }
    DLkBallSubterms out;
    out.lk_ball_iso = lkIso;
    out.lk_ball = lkIso * fracDesolv;
    out.lk_bridge = lkIso * fracOverlap;
    out.lk_bridge_uncpl = fracOverlap / 2;
    return out;
}

// ===========================================================================
// hbond.hpp
// ===========================================================================
__device__ const double HB_PI = 3.14159265358979323846;
__device__ const int HB_SP2 = 1;
__device__ const int HB_SP3 = 2;
__device__ const int HB_RING = 3;

// bound_poly::V - clamp outside [xmin, xmax], else Horner-evaluate the degree-10 poly.
__device__ inline double boundPoly(double x, const DHBPoly& p) {
    if (x < p.range[0]) return p.bound[0];
    if (x > p.range[1]) return p.bound[1];
    double v = p.coeffs[0];
    for (int i = 1; i < 11; i++) v = v * x + p.coeffs[i];
    return v;
}
__device__ inline double bahAngleBaseForm(double3 b, double3 a, double3 h,
                                          const DHBPoly& poly) {
    const double3 ah = v_sub(h, a);
    const double3 ba = v_sub(a, b);
    return boundPoly(cosInteriorAngle(ah, ba), poly);
}
// BAH_angle_V: base-acceptor-H angle, blended by acceptor hybridization.
__device__ inline double bahAngle(double3 b, double3 b0, double3 a, double3 h, int hyb,
                                  const DHBPoly& poly, double sp3SoftmaxFade) {
    if (hyb == HB_SP2) return bahAngleBaseForm(b, a, h, poly);
    if (hyb == HB_RING)
        return bahAngleBaseForm(v_scale(v_add(b, b0), 0.5), a, h, poly);
    const double pxH = bahAngleBaseForm(b, a, h, poly);
    const double pxH0 = bahAngleBaseForm(b0, a, h, poly);
    return log(exp(pxH * sp3SoftmaxFade) + exp(pxH0 * sp3SoftmaxFade)) / sp3SoftmaxFade;
}
// sp2chi_energy_V: the sp2 dihedral (chi) energy.
__device__ inline double sp2chiEnergy(double ang, double chi, double d, double m,
                                      double l) {
    const double pi = HB_PI;
    const double H = 0.5 * (cos(2 * chi) + 1);
    if (ang > (pi * 2.0) / 3.0) {
        const double F = (d / 2) * cos(3 * (pi - ang)) + d / 2 - 0.5;
        const double G = d - 0.5;
        return H * F + (1 - H) * G;
    } else if (ang >= pi * (2.0 / 3.0 - l)) {
        const double outerRise = cos(pi - ((pi * 2) / 3 - ang) / l);
        const double F = (m / 2) * outerRise + m / 2 - 0.5;
        const double G = ((m - d) / 2) * outerRise + (m - d) / 2 + d - 0.5;
        return H * F + (1 - H) * G;
    }
    return m - 0.5;
}
// B0BAH_chi_V: sp2 chi contribution (zero for sp3/ring acceptors).
__device__ inline double b0bahChi(double3 b0, double3 b, double3 a, double3 h, int hyb,
                                  const DHbondGlobal& g) {
    if (hyb != HB_SP2) return 0;
    const double bah = ptInteriorAngle(b, a, h);
    const double b0bah = dihedralAngle(b0, b, a, h);
    return sp2chiEnergy(bah, b0bah, g.hb_sp2_BAH180_rise, g.hb_sp2_range_span,
                        g.hb_sp2_outer_width);
}
// hbond_score::V for one (donor H, acceptor A) pair.
__device__ inline double hbondScore(double3 d, double3 h, double3 a, double3 b,
                                    double3 b0, int hyb, double ad_weight,
                                    const DHBPoly& AHdist, const DHBPoly& cosBAH,
                                    const DHBPoly& cosAHD, const DHbondGlobal& g) {
    if (v_dist(h, a) >= g.max_ha_dis) return 0;

    double e = 0.0;
    e += boundPoly(v_dist(a, h), AHdist);               // A-H distance
    e += boundPoly(ptInteriorAngle(a, h, d), cosAHD);   // A-H-D angle (radians)
    e += bahAngle(b, b0, a, h, hyb, cosBAH, g.hb_sp3_softmax_fade);
    e += b0bahChi(b0, b, a, h, hyb, g);

    e *= ad_weight;

    if (e > 0.1) return 0;
    if (e > -0.1) return -0.025 + 0.5 * e - 2.5 * e * e;
    return e;
}

// ===========================================================================
// Kernels (one thread per scored pair). SoA layout, mirroring the CPU drivers.
// ===========================================================================
__device__ inline double3 loadc(const double* coords, int i) {
    return make_double3(coords[3 * i + 0], coords[3 * i + 1], coords[3 * i + 2]);
}
__device__ inline DLjlk loadLjlk(int i, const double* ljr, const double* ljw,
                                 const double* lkd, const double* lkl, const double* lkv,
                                 const char* is_donor, const char* is_hyd,
                                 const char* is_polh, const char* is_acc) {
    DLjlk t;
    t.lj_radius = ljr[i];
    t.lj_wdepth = ljw[i];
    t.lk_dgfree = lkd[i];
    t.lk_lambda = lkl[i];
    t.lk_volume = lkv[i];
    t.is_donor = is_donor[i] != 0;
    t.is_hydroxyl = is_hyd[i] != 0;
    t.is_polarh = is_polh[i] != 0;
    t.is_acceptor = is_acc[i] != 0;
    return t;
}

// ljlk (fa_ljatr/fa_ljrep/fa_lk) + fa_elec, mirroring computePairEnergiesCPU's pairBody.
__global__ void k_pair(const double* coords, const int* block, const double* charge,
                       const char* is_heavy, const double* ljr, const double* ljw,
                       const double* lkd, const double* lkl, const double* lkv,
                       const char* is_donor, const char* is_hyd, const char* is_polh,
                       const char* is_acc, const int* pi, const int* pj,
                       const int* sep_ljlk, const int* sep_elec, int nPairs,
                       DLjlkGlobal lg, DElecGlobal eg, int nBlocks, double* mats) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nPairs) return;
    const int i = pi[p], j = pj[p];
    const double3 ci = loadc(coords, i), cj = loadc(coords, j);
    const double d = v_dist(ci, cj);
    if (d <= 0) return;
    const int b1 = block[i], b2 = block[j];
    const std::size_t NN = static_cast<std::size_t>(nBlocks) * nBlocks;

    const double e = elec(d, charge[i], charge[j], static_cast<double>(sep_elec[p]), eg);
    if (e != 0) addBP(mats + 3 * NN, nBlocks, b1, b2, e);  // fa_elec

    const DLjlk ti = loadLjlk(i, ljr, ljw, lkd, lkl, lkv, is_donor, is_hyd, is_polh, is_acc);
    const DLjlk tj = loadLjlk(j, ljr, ljw, lkd, lkl, lkv, is_donor, is_hyd, is_polh, is_acc);
    double atr, rep;
    ljScore(d, static_cast<double>(sep_ljlk[p]), ti, tj, lg, atr, rep);
    if (atr != 0) addBP(mats + 0 * NN, nBlocks, b1, b2, atr);  // fa_ljatr
    if (rep != 0) addBP(mats + 1 * NN, nBlocks, b1, b2, rep);  // fa_ljrep
    if (is_heavy[i] && is_heavy[j]) {
        const double lk = lkIsotropicScore(d, static_cast<double>(sep_ljlk[p]), ti, tj, lg);
        if (lk != 0) addBP(mats + 2 * NN, nBlocks, b1, b2, lk);  // fa_lk
    }
}

// lk_ball (four subterms), mirroring computeLkBallCPU: each heavy-atom pair scored in
// BOTH directions (each atom in turn as the polar atom whose waters are occluded).
__global__ void k_lkball(const double* coords, const int* block, const char* is_heavy,
                         const double* ljr, const double* ljw, const double* lkd,
                         const double* lkl, const double* lkv, const char* is_donor,
                         const char* is_hyd, const char* is_polh, const char* is_acc,
                         const double* wpos, const char* wpres, const int* pi,
                         const int* pj, const int* sep_ljlk, int nPairs,
                         DLkBallGlobal g, int nBlocks, double* mats) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nPairs) return;
    const int i = pi[p], j = pj[p];
    if (!is_heavy[i] || !is_heavy[j]) return;
    const double3 ci = loadc(coords, i), cj = loadc(coords, j);
    const double d = v_dist(ci, cj);
    if (d <= 0) return;
    const std::size_t NN = static_cast<std::size_t>(nBlocks) * nBlocks;
    const double sep = static_cast<double>(sep_ljlk[p]);

    bool presI[MW], presJ[MW];
    for (int w = 0; w < MW; ++w) {
        presI[w] = wpres[static_cast<std::size_t>(i) * MW + w] != 0;
        presJ[w] = wpres[static_cast<std::size_t>(j) * MW + w] != 0;
    }
    const double* wiPos = wpos + static_cast<std::size_t>(i) * MW * 3;
    const double* wjPos = wpos + static_cast<std::size_t>(j) * MW * 3;

    const DLjlk ti = loadLjlk(i, ljr, ljw, lkd, lkl, lkv, is_donor, is_hyd, is_polh, is_acc);
    const DLjlk tj = loadLjlk(j, ljr, ljw, lkd, lkl, lkv, is_donor, is_hyd, is_polh, is_acc);

    // scoreDir(pol=i, occ=j)
    if (presI[0]) {
        const DLkBallSubterms s =
            lkBallScore(ci, cj, wiPos, presI, wjPos, presJ, sep, d, ti, tj, g);
        addBP(mats + 0 * NN, nBlocks, block[i], block[j], s.lk_ball_iso);
        addBP(mats + 1 * NN, nBlocks, block[i], block[j], s.lk_ball);
        addBP(mats + 2 * NN, nBlocks, block[i], block[j], s.lk_bridge);
        addBP(mats + 3 * NN, nBlocks, block[i], block[j], s.lk_bridge_uncpl);
    }
    // scoreDir(pol=j, occ=i)
    if (presJ[0]) {
        const DLkBallSubterms s =
            lkBallScore(cj, ci, wjPos, presJ, wiPos, presI, sep, d, tj, ti, g);
        addBP(mats + 0 * NN, nBlocks, block[j], block[i], s.lk_ball_iso);
        addBP(mats + 1 * NN, nBlocks, block[j], block[i], s.lk_ball);
        addBP(mats + 2 * NN, nBlocks, block[j], block[i], s.lk_bridge);
        addBP(mats + 3 * NN, nBlocks, block[j], block[i], s.lk_bridge_uncpl);
    }
}

__device__ inline DHBPoly loadPoly(const double* coeffs, const double* range,
                                   const double* bound, int p) {
    DHBPoly poly;
    for (int c = 0; c < 11; ++c) poly.coeffs[c] = coeffs[p * 11 + c];
    poly.range[0] = range[p * 2 + 0];
    poly.range[1] = range[p * 2 + 1];
    poly.bound[0] = bound[p * 2 + 0];
    poly.bound[1] = bound[p * 2 + 1];
    return poly;
}
__device__ inline double3 row3(const double* a, int r) {
    return make_double3(a[r * 3 + 0], a[r * 3 + 1], a[r * 3 + 2]);
}

// hbond, mirroring computeHbondCPU: one donor-H / acceptor pair per thread, the < 5
// count-pair exclusion applied on device.
__global__ void k_hbond(const double* H, const double* A, const double* D,
                        const double* B, const double* B0, const int* blkh,
                        const int* blka, const int* hyb, const double* adw,
                        const int* sep, const double* ahd_c, const double* ahd_r,
                        const double* ahd_b, const double* bah_c, const double* bah_r,
                        const double* bah_b, const double* ahdang_c,
                        const double* ahdang_r, const double* ahdang_b, int nPairs,
                        DHbondGlobal g, int nBlocks, double* mats) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nPairs) return;
    if (sep[p] < 5) return;  // tmol count-pair exclusion
    const double3 h = row3(H, p), a = row3(A, p), dd = row3(D, p);
    const double3 b = row3(B, p), b0 = row3(B0, p);
    const DHBPoly AHdist = loadPoly(ahd_c, ahd_r, ahd_b, p);
    const DHBPoly cosBAH = loadPoly(bah_c, bah_r, bah_b, p);
    const DHBPoly cosAHD = loadPoly(ahdang_c, ahdang_r, ahdang_b, p);
    const double e =
        hbondScore(dd, h, a, b, b0, hyb[p], adw[p], AHdist, cosBAH, cosAHD, g);
    addBP(mats, nBlocks, blkh[p], blka[p], e);  // single subterm "hbond"
}

// ===========================================================================
// Host glue: build SoA from the resolved host structs, upload, launch, download.
// ===========================================================================
PairEnergyResult buildResult(const std::vector<std::string>& names, int nBlocks,
                             const std::vector<double>& flat) {
    const int nSub = static_cast<int>(names.size());
    const std::size_t NN = static_cast<std::size_t>(nBlocks) * nBlocks;
    PairEnergyResult res;
    res.nBlocks = nBlocks;
    res.names = names;
    res.blockPair.assign(static_cast<std::size_t>(nSub), std::vector<double>(NN, 0.0));
    res.wholePose.assign(static_cast<std::size_t>(nSub), 0.0);
    for (int s = 0; s < nSub; ++s) {
        double sum = 0.0;
        std::vector<double>& out = res.blockPair[static_cast<std::size_t>(s)];
        for (std::size_t k = 0; k < NN; ++k) {
            const double v = flat[static_cast<std::size_t>(s) * NN + k];
            out[k] = v;
            sum += v;
        }
        res.wholePose[static_cast<std::size_t>(s)] = sum;
    }
    return res;
}

// Per-atom ljlk SoA built from the resolved type rows (atom.ljlkType indexes the table,
// exactly as the CPU driver does).
struct AtomSoA {
    std::vector<double> coords, charge, ljr, ljw, lkd, lkl, lkv;
    std::vector<int> block;
    std::vector<char> is_heavy, is_donor, is_hyd, is_polh, is_acc;
    std::vector<double> wpos;   // n * MW * 3
    std::vector<char> wpres;    // n * MW
};

AtomSoA buildAtomSoA(const std::vector<AtomInput>& atoms, const EnergyParams& params) {
    const std::size_t n = atoms.size();
    AtomSoA s;
    s.coords.resize(n * 3);
    s.charge.resize(n);
    s.ljr.resize(n);
    s.ljw.resize(n);
    s.lkd.resize(n);
    s.lkl.resize(n);
    s.lkv.resize(n);
    s.block.resize(n);
    s.is_heavy.resize(n);
    s.is_donor.resize(n);
    s.is_hyd.resize(n);
    s.is_polh.resize(n);
    s.is_acc.resize(n);
    s.wpos.assign(n * MW * 3, 0.0);
    s.wpres.assign(n * MW, 0);
    for (std::size_t k = 0; k < n; ++k) {
        const AtomInput& a = atoms[k];
        s.coords[k * 3 + 0] = a.x;
        s.coords[k * 3 + 1] = a.y;
        s.coords[k * 3 + 2] = a.z;
        s.charge[k] = a.charge;
        s.block[k] = a.block;
        s.is_heavy[k] = a.isHeavy ? 1 : 0;
        const LjlkTypeParams& tp =
            params.ljlkTypeParams[static_cast<std::size_t>(a.ljlkType)];
        s.ljr[k] = tp.lj_radius;
        s.ljw[k] = tp.lj_wdepth;
        s.lkd[k] = tp.lk_dgfree;
        s.lkl[k] = tp.lk_lambda;
        s.lkv[k] = tp.lk_volume;
        s.is_donor[k] = tp.is_donor ? 1 : 0;
        s.is_hyd[k] = tp.is_hydroxyl ? 1 : 0;
        s.is_polh[k] = tp.is_polarh ? 1 : 0;
        s.is_acc[k] = tp.is_acceptor ? 1 : 0;
        if (a.hasWaters) {
            for (int w = 0; w < MW; ++w) {
                if (a.waters.present[w]) {
                    s.wpres[k * MW + w] = 1;
                    s.wpos[(k * MW + w) * 3 + 0] = a.waters.pos[w][0];
                    s.wpos[(k * MW + w) * 3 + 1] = a.waters.pos[w][1];
                    s.wpos[(k * MW + w) * 3 + 2] = a.waters.pos[w][2];
                }
            }
        }
    }
    return s;
}

struct PairSoA {
    std::vector<int> i, j, sep_ljlk, sep_elec;
};
PairSoA buildPairSoA(const std::vector<PairInput>& pairs) {
    PairSoA s;
    const std::size_t n = pairs.size();
    s.i.resize(n);
    s.j.resize(n);
    s.sep_ljlk.resize(n);
    s.sep_elec.resize(n);
    for (std::size_t p = 0; p < n; ++p) {
        s.i[p] = pairs[p].i;
        s.j[p] = pairs[p].j;
        s.sep_ljlk[p] = pairs[p].sepLjlk;
        s.sep_elec[p] = pairs[p].sepElec;
    }
    return s;
}

constexpr int TPB = 128;
inline int nblk(std::size_t n) {
    return static_cast<int>((n + TPB - 1) / TPB);
}

}  // namespace

PairEnergyResult computePairEnergiesCUDA(const std::vector<AtomInput>& atoms,
                                         const std::vector<PairInput>& pairs,
                                         const EnergyParams& params, int nBlocks) {
    const AtomSoA a = buildAtomSoA(atoms, params);
    const PairSoA pr = buildPairSoA(pairs);
    const std::size_t NN = static_cast<std::size_t>(nBlocks) * nBlocks;
    const std::vector<std::string>& names = ljlkElecSubterms();

    DeviceBuffer<double> coords(a.coords), charge(a.charge), ljr(a.ljr), ljw(a.ljw),
        lkd(a.lkd), lkl(a.lkl), lkv(a.lkv);
    DeviceBuffer<int> block(a.block);
    DeviceBuffer<char> isheavy(a.is_heavy), isdon(a.is_donor), ishyd(a.is_hyd),
        ispolh(a.is_polh), isacc(a.is_acc);
    DeviceBuffer<int> pi(pr.i), pj(pr.j), sepl(pr.sep_ljlk), sepe(pr.sep_elec);
    DeviceBuffer<double> mats(static_cast<std::size_t>(names.size()) * NN);
    mats.zero();

    DLjlkGlobal lg{params.ljlkGlobal.lj_hbond_dis,
                   params.ljlkGlobal.lj_hbond_OH_donor_dis,
                   params.ljlkGlobal.lj_hbond_hdis};
    DElecGlobal eg{params.elecGlobal.D, params.elecGlobal.D0, params.elecGlobal.S,
                   params.elecGlobal.min_dis, params.elecGlobal.max_dis};

    if (!pairs.empty()) {
        k_pair<<<nblk(pairs.size()), TPB>>>(
            coords.get(), block.get(), charge.get(), isheavy.get(), ljr.get(), ljw.get(),
            lkd.get(), lkl.get(), lkv.get(), isdon.get(), ishyd.get(), ispolh.get(),
            isacc.get(), pi.get(), pj.get(), sepl.get(), sepe.get(),
            static_cast<int>(pairs.size()), lg, eg, nBlocks, mats.get());
        cuda_check(cudaGetLastError(), "k_pair");
    }
    cuda_check(cudaDeviceSynchronize(), "sync");
    std::vector<double> flat;
    mats.to_host(flat);
    return buildResult(names, nBlocks, flat);
}

PairEnergyResult computeLkBallCUDA(const std::vector<AtomInput>& atoms,
                                   const std::vector<PairInput>& pairs,
                                   const EnergyParams& params, int nBlocks) {
    const AtomSoA a = buildAtomSoA(atoms, params);
    const PairSoA pr = buildPairSoA(pairs);
    const std::size_t NN = static_cast<std::size_t>(nBlocks) * nBlocks;
    const std::vector<std::string>& names = lkBallSubterms();

    DeviceBuffer<double> coords(a.coords), ljr(a.ljr), ljw(a.ljw), lkd(a.lkd),
        lkl(a.lkl), lkv(a.lkv), wpos(a.wpos);
    DeviceBuffer<int> block(a.block);
    DeviceBuffer<char> isheavy(a.is_heavy), isdon(a.is_donor), ishyd(a.is_hyd),
        ispolh(a.is_polh), isacc(a.is_acc), wpres(a.wpres);
    DeviceBuffer<int> pi(pr.i), pj(pr.j), sepl(pr.sep_ljlk);
    DeviceBuffer<double> mats(static_cast<std::size_t>(names.size()) * NN);
    mats.zero();

    DLkBallGlobal g{params.lkBallGlobal.lj_hbond_dis,
                    params.lkBallGlobal.lj_hbond_OH_donor_dis,
                    params.lkBallGlobal.lj_hbond_hdis,
                    params.lkBallGlobal.lkb_water_dist,
                    params.lkBallGlobal.distance_threshold};

    if (!pairs.empty()) {
        k_lkball<<<nblk(pairs.size()), TPB>>>(
            coords.get(), block.get(), isheavy.get(), ljr.get(), ljw.get(), lkd.get(),
            lkl.get(), lkv.get(), isdon.get(), ishyd.get(), ispolh.get(), isacc.get(),
            wpos.get(), wpres.get(), pi.get(), pj.get(), sepl.get(),
            static_cast<int>(pairs.size()), g, nBlocks, mats.get());
        cuda_check(cudaGetLastError(), "k_lkball");
    }
    cuda_check(cudaDeviceSynchronize(), "sync");
    std::vector<double> flat;
    mats.to_host(flat);
    return buildResult(names, nBlocks, flat);
}

PairEnergyResult computeHbondCUDA(const std::vector<AtomInput>& atoms,
                                  const std::vector<HBondPairInput>& hbondPairs,
                                  const EnergyParams& params, int nBlocks) {
    const std::size_t n = hbondPairs.size();
    const std::size_t NN = static_cast<std::size_t>(nBlocks) * nBlocks;
    const std::vector<std::string>& names = hbondSubterms();

    // Per-pair SoA, extracting H/A coords + blocks from the atom array (as the CPU
    // driver does via atoms[hp.h] / atoms[hp.a]) and the resolved geometry/polys.
    std::vector<double> H(n * 3), A(n * 3), D(n * 3), B(n * 3), B0(n * 3);
    std::vector<int> blkh(n), blka(n), hyb(n), sep(n);
    std::vector<double> adw(n);
    std::vector<double> ahd_c(n * 11), ahd_r(n * 2), ahd_b(n * 2);
    std::vector<double> bah_c(n * 11), bah_r(n * 2), bah_b(n * 2);
    std::vector<double> ang_c(n * 11), ang_r(n * 2), ang_b(n * 2);
    for (std::size_t p = 0; p < n; ++p) {
        const HBondPairInput& hp = hbondPairs[p];
        const AtomInput& Ha = atoms[static_cast<std::size_t>(hp.h)];
        const AtomInput& Aa = atoms[static_cast<std::size_t>(hp.a)];
        H[p * 3 + 0] = Ha.x; H[p * 3 + 1] = Ha.y; H[p * 3 + 2] = Ha.z;
        A[p * 3 + 0] = Aa.x; A[p * 3 + 1] = Aa.y; A[p * 3 + 2] = Aa.z;
        D[p * 3 + 0] = hp.D[0]; D[p * 3 + 1] = hp.D[1]; D[p * 3 + 2] = hp.D[2];
        B[p * 3 + 0] = hp.B[0]; B[p * 3 + 1] = hp.B[1]; B[p * 3 + 2] = hp.B[2];
        B0[p * 3 + 0] = hp.B0[0]; B0[p * 3 + 1] = hp.B0[1]; B0[p * 3 + 2] = hp.B0[2];
        blkh[p] = Ha.block;
        blka[p] = Aa.block;
        hyb[p] = hp.pair.hyb;
        adw[p] = hp.pair.ad_weight;
        sep[p] = hp.sep;
        for (int c = 0; c < 11; ++c) {
            ahd_c[p * 11 + c] = hp.pair.AHdist.coeffs[c];
            bah_c[p * 11 + c] = hp.pair.cosBAH.coeffs[c];
            ang_c[p * 11 + c] = hp.pair.cosAHD.coeffs[c];
        }
        for (int c = 0; c < 2; ++c) {
            ahd_r[p * 2 + c] = hp.pair.AHdist.range[c];
            ahd_b[p * 2 + c] = hp.pair.AHdist.bound[c];
            bah_r[p * 2 + c] = hp.pair.cosBAH.range[c];
            bah_b[p * 2 + c] = hp.pair.cosBAH.bound[c];
            ang_r[p * 2 + c] = hp.pair.cosAHD.range[c];
            ang_b[p * 2 + c] = hp.pair.cosAHD.bound[c];
        }
    }

    DeviceBuffer<double> dH(H), dA(A), dD(D), dB(B), dB0(B0), dadw(adw);
    DeviceBuffer<int> dblkh(blkh), dblka(blka), dhyb(hyb), dsep(sep);
    DeviceBuffer<double> dahd_c(ahd_c), dahd_r(ahd_r), dahd_b(ahd_b);
    DeviceBuffer<double> dbah_c(bah_c), dbah_r(bah_r), dbah_b(bah_b);
    DeviceBuffer<double> dang_c(ang_c), dang_r(ang_r), dang_b(ang_b);
    DeviceBuffer<double> mats(static_cast<std::size_t>(names.size()) * NN);
    mats.zero();

    DHbondGlobal g{params.hbondGlobal.hb_sp2_range_span,
                   params.hbondGlobal.hb_sp2_BAH180_rise,
                   params.hbondGlobal.hb_sp2_outer_width,
                   params.hbondGlobal.hb_sp3_softmax_fade,
                   params.hbondGlobal.threshold_distance,
                   params.hbondGlobal.max_ha_dis};

    if (n > 0) {
        k_hbond<<<nblk(n), TPB>>>(
            dH.get(), dA.get(), dD.get(), dB.get(), dB0.get(), dblkh.get(), dblka.get(),
            dhyb.get(), dadw.get(), dsep.get(), dahd_c.get(), dahd_r.get(), dahd_b.get(),
            dbah_c.get(), dbah_r.get(), dbah_b.get(), dang_c.get(), dang_r.get(),
            dang_b.get(), static_cast<int>(n), g, nBlocks, mats.get());
        cuda_check(cudaGetLastError(), "k_hbond");
    }
    cuda_check(cudaDeviceSynchronize(), "sync");
    std::vector<double> flat;
    mats.to_host(flat);
    return buildResult(names, nBlocks, flat);
}

}  // namespace frustramol_tmol
