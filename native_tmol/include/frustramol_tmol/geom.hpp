// Minimal 3-vector geometry used by the lk_ball and hbond kernels.
//
// Ported verbatim from the Apache-2.0 tmol-webgpu reference (src/kernels/geom.ts),
// which is itself a verbatim port of tmol's tmol/score/common/geom.hh (Apache-2.0).
// These are the exact angle/dihedral forms the hbond polynomials are evaluated on.
// See native_tmol/NOTICE for attribution.
//
// Single-point forward math only: no autograd, no derivatives.

#pragma once

#include <array>
#include <cmath>

namespace frustramol_tmol {

using Vec3 = std::array<double, 3>;

inline Vec3 sub(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
inline Vec3 add(const Vec3& a, const Vec3& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
inline double dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline double dist(const Vec3& a, const Vec3& b) { return norm(sub(a, b)); }
inline Vec3 scale(const Vec3& a, double s) {
    return {a[0] * s, a[1] * s, a[2] * s};
}

// interior_angle::V(A, B): 2*atan2(|AxB|, |A||B| + A.B). Robust near 0 and pi.
inline double interiorAngle(const Vec3& a, const Vec3& b) {
    const Vec3 cr = cross(a, b);
    return 2.0 * std::atan2(norm(cr), norm(a) * norm(b) + dot(a, b));
}

// pt_interior_angle::V(A, B, C): the interior angle at vertex B, in radians.
inline double ptInteriorAngle(const Vec3& a, const Vec3& b, const Vec3& c) {
    return interiorAngle(sub(a, b), sub(c, b));
}

// cos_interior_angle::V(A, B): cos of the angle between vectors A and B.
inline double cosInteriorAngle(const Vec3& a, const Vec3& b) {
    return dot(a, b) / (norm(a) * norm(b));
}

// dihedral_angle::V(I, J, K, L) via the Blondel-Karplus formulation.
inline double dihedralAngle(const Vec3& i, const Vec3& j, const Vec3& k,
                            const Vec3& l) {
    const Vec3 f = sub(i, j);
    const Vec3 g = sub(j, k);
    const Vec3 h = sub(l, k);
    const Vec3 a = cross(f, g);
    const Vec3 b = cross(h, g);
    const double sign = dot(g, cross(a, b)) >= 0 ? -1.0 : 1.0;
    const double c =
        std::max(-1.0, std::min(dot(a, b) / (norm(a) * norm(b)), 1.0));
    return sign * std::acos(c);
}

}  // namespace frustramol_tmol
