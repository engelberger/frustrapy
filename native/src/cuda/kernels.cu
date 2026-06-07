// CUDA kernels for the native frustration core (N3).
//
// N1: this translation unit is compiled only when FRUSTRAPY_NATIVE_CUDA is ON and nvcc
// is available (the container has no GPU, so it is OFF by default). The kernels are
// empty placeholders; N3 ports K_density / K_native / K_decoy here per the design doc,
// section 4, batching the (contact x decoy) axis and Welford-reducing in shared memory.
//
// Conditional compile so a no-GPU machine still builds the CPU core unchanged.

#include <cstdint>

namespace frustrapy_native {
namespace cuda {

// K_density: rho_i = sum_{k != i} theta(r_ik). One thread per residue (N3).
__global__ void k_density(const double* /*cb*/, std::int32_t /*n_res*/,
                          double /*rmin*/, double /*rmax*/, double* /*rho*/) {
    // N3: tiled N x N density reduction with shared-memory blocking.
}

// K_decoy: per contact, sweep the decoy batch and reduce to (mean, sd). The
// (contact x 400) mutational grid / (contact x n_decoys) configurational grid is the
// GPU batch axis (N3).
__global__ void k_decoy(const double* /*geometry*/, const double* /*gammas*/,
                        std::int32_t /*n_contacts*/, std::int32_t /*n_decoys*/,
                        double* /*decoy_mean*/, double* /*decoy_sd*/) {
    // N3: fuse native + decoy, load contact geometry once, Welford in shared memory.
}

}  // namespace cuda
}  // namespace frustrapy_native
