// Cubic Hermite interpolation helpers.
//
// Ported verbatim from the Apache-2.0 tmol-webgpu reference (src/kernels/interpolate.ts),
// itself a verbatim port of tmol's tmol/score/common/cubic_hermite_polynomial.hh
// (Apache-2.0). These smooth the energy functions between their analytic region and
// zero at the cutoff. See native_tmol/NOTICE for attribution.

#pragma once

namespace frustramol_tmol {

// Cubic interpolation of p on x in [x0, x1] with endpoint values/derivatives.
inline double interpolate(double x, double x0, double p0, double dpdx0, double x1,
                          double p1, double dpdx1) {
    const double t = (x - x0) / (x1 - x0);
    const double dp0 = dpdx0 * (x1 - x0);
    const double dp1 = dpdx1 * (x1 - x0);
    return p0 +
           t * (dp0 +
                t * (-2 * dp0 - dp1 - 3 * p0 + 3 * p1 +
                     t * (dp0 + dp1 + 2 * p0 - 2 * p1)));
}

// Cubic interpolation of p on x in [x0, x1] fading to value 0 and slope 0 at x1.
inline double interpolateToZero(double x, double x0, double p0, double dpdx0,
                                double x1) {
    const double t = (x - x0) / (x1 - x0);
    const double dp0 = dpdx0 * (x1 - x0);
    return p0 + t * (dp0 + t * (-2 * dp0 - 3 * p0 + t * (dp0 + 2 * p0)));
}

}  // namespace frustramol_tmol
