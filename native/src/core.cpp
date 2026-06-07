// Native frustration core (CPU reference) -- implementation.
//
// N1: contact_map and local_density are real (pure geometry, no energy parity claim);
// compute_frustration is a stub for N2. No owning raw pointers; bounds via std::span.

#include "core.hpp"

#include <cmath>
#include <stdexcept>

namespace frustrapy_native {

namespace {

// Squared CB-CB distance between residues a and b. cb is flat [3 * n_res].
double dist2(std::span<const double> cb, std::size_t a, std::size_t b) {
    const double dx = cb[3 * a + 0] - cb[3 * b + 0];
    const double dy = cb[3 * a + 1] - cb[3 * b + 1];
    const double dz = cb[3 * a + 2] - cb[3 * b + 2];
    return dx * dx + dy * dy + dz * dz;
}

// True iff residues a, b are far enough apart in sequence to count as a contact:
// >= seq_dist along the same chain, or on different chains.
bool separated(const StructureView& s, std::size_t a, std::size_t b, int seq_dist) {
    if (s.chain_id[a] != s.chain_id[b]) return true;
    return std::abs(s.res_seqid[a] - s.res_seqid[b]) >= seq_dist;
}

void check_lengths(const StructureView& s) {
    const std::size_t n = s.n_res;
    if (s.ca.size() != 3 * n || s.cb.size() != 3 * n) {
        throw std::invalid_argument("coordinate span length must be 3 * n_res");
    }
    if (s.res_type.size() != n || s.chain_id.size() != n || s.res_seqid.size() != n) {
        throw std::invalid_argument("per-residue span length must be n_res");
    }
}

}  // namespace

std::vector<std::int32_t> contact_map(const StructureView& s, double cutoff,
                                      int seq_dist) {
    check_lengths(s);
    const double cut2 = cutoff * cutoff;
    std::vector<std::int32_t> pairs;
    for (std::size_t i = 0; i < s.n_res; ++i) {
        for (std::size_t j = i + 1; j < s.n_res; ++j) {
            if (!separated(s, i, j, seq_dist)) continue;
            if (dist2(s.cb, i, j) <= cut2) {
                pairs.push_back(static_cast<std::int32_t>(i));
                pairs.push_back(static_cast<std::int32_t>(j));
            }
        }
    }
    return pairs;
}

std::vector<double> local_density(const StructureView& s, double rmin, double rmax) {
    check_lengths(s);
    // Smooth neighbor count with a tanh switching function between rmin and rmax, the
    // AWSEM density well shape. theta(r) -> 1 below rmin, 0 above rmax.
    const double inv_width = 1.0 / std::max(rmax - rmin, 1e-9);
    std::vector<double> rho(s.n_res, 0.0);
    for (std::size_t i = 0; i < s.n_res; ++i) {
        double acc = 0.0;
        for (std::size_t k = 0; k < s.n_res; ++k) {
            if (k == i) continue;
            const double r = std::sqrt(dist2(s.cb, i, k));
            const double t = 0.5 * (1.0 + std::tanh(inv_width * (rmax - r)));
            acc += t;
        }
        rho[i] = acc;
    }
    return rho;
}

ContactResult compute_frustration(const StructureView& s, const ParamsView& p,
                                  const std::string& mode) {
    check_lengths(s);
    (void)p;
    if (mode != "configurational" && mode != "mutational" && mode != "singleresidue") {
        throw std::invalid_argument("mode must be configurational, mutational, or "
                                    "singleresidue");
    }
    // N2 implements the AWSEM native energy, the per-mode decoy ensemble, and the
    // Z-score (sign: (decoy_mean - native) / decoy_sd). See the design doc, section 4.
    throw std::runtime_error("native energy core not implemented (N2)");
}

bool has_cuda() noexcept {
#ifdef FRUSTRAPY_NATIVE_CUDA
    return true;
#else
    return false;
#endif
}

}  // namespace frustrapy_native
