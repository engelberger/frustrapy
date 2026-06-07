// CUDA kernels for the native frustration core (N3).
//
// Compiled only when FRUSTRAPY_NATIVE_CUDA is ON and nvcc is available (the dev
// container has no GPU, so it is OFF by default). The kernels mirror the validated
// CPU reductions in core.cpp term-for-term; the decoy random-index stream is
// generated host-side with the exact glibc rand() sequence (see GlibcRand), so the
// GPU result matches the CPU core bit-for-bit. The GPU parallelizes the dominant
// cost: the per-(contact, decoy) energy evaluations (mutational mode is the biggest
// win), plus the density and native-energy reductions.
//
// This file has not been compiled or run in the dev container (no nvcc/GPU). It is
// written to mirror the parity-gated CPU core exactly; the maintainer validates
// parity and measures speed on Colab / the cluster (see native/colab/). No GPU
// timing is asserted here.
//
// Build/launch conventions follow the CUDA C++ Best Practices Guide (v13.x):
// coalesced global access (SoA), minimal host/device transfer (upload once), and a
// block-per-unit shared-memory reduction for the decoy mean/sd.

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "core.hpp"

namespace frustrapy_native {

namespace {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                                 cudaGetErrorString(e));
    }
}

// RAII device buffer (no naked cudaMalloc/cudaFree across scopes).
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t n) : n_(n) {
        cuda_check(cudaMalloc(&ptr_, n * sizeof(T)), "cudaMalloc");
    }
    DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) {
        cuda_check(cudaMemcpy(ptr_, host.data(), n_ * sizeof(T), cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D");
    }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }
    std::size_t size() const { return n_; }
    void to_host(std::vector<T>& host) const {
        host.resize(n_);
        cuda_check(cudaMemcpy(host.data(), ptr_, n_ * sizeof(T), cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H");
    }

private:
    T* ptr_ = nullptr;
    std::size_t n_ = 0;
};

// Device-side parameter bundle (gamma device pointers + the well/burial scalars).
struct DParams {
    const double* gamma_direct;   // 400
    const double* gamma_water;    // 400
    const double* gamma_protein;  // 400
    const double* burial_gamma;   // 60
    double well_kappa, kappa_sigma, treshold;
    double well_r_min0, well_r_max0, well_r_min1, well_r_max1;
    double burial_kappa, k_burial;
    double burial_ro_min[3], burial_ro_max[3];
    double contact_cutoff;
    int contact_min_sep, seq_dist;
};

__device__ inline double d_dist(const double* coord, int a, int b) {
    const double dx = coord[3 * a + 0] - coord[3 * b + 0];
    const double dy = coord[3 * a + 1] - coord[3 * b + 1];
    const double dz = coord[3 * a + 2] - coord[3 * b + 2];
    return sqrt(dx * dx + dy * dy + dz * dz);
}

__device__ inline double d_theta(double r, double rmin, double rmax, double kappa) {
    return 0.25 * (1.0 + tanh(kappa * (r - rmin))) * (1.0 + tanh(kappa * (rmax - r)));
}

__device__ inline double d_theta_clamped(double r, double rmin, double rmax, double kappa) {
    const double half = 8.0 * 2.302585 / kappa;
    if (r < rmin - half || r > rmax + half) return 0.0;
    return d_theta(r, rmin, rmax, kappa);
}

__device__ inline double d_water(const DParams& p, double rij, int it, int jt,
                                 double rho_i, double rho_j) {
    const double sigma_wat =
        0.25 * (1.0 - tanh(p.kappa_sigma * (rho_i - p.treshold))) *
        (1.0 - tanh(p.kappa_sigma * (rho_j - p.treshold)));
    const double sigma_prot = 1.0 - sigma_wat;
    const int g = it * 20 + jt;
    const double sgd = p.gamma_direct[g];
    const double sgm = sigma_prot * p.gamma_protein[g] + sigma_wat * p.gamma_water[g];
    return -(sgd * d_theta(rij, p.well_r_min0, p.well_r_max0, p.well_kappa) +
             sgm * d_theta(rij, p.well_r_min1, p.well_r_max1, p.well_kappa));
}

__device__ inline double d_burial(const DParams& p, int it, double rho) {
    double e = 0.0;
    for (int k = 0; k < 3; ++k) {
        const double t0 = tanh(p.burial_kappa * (rho - p.burial_ro_min[k]));
        const double t1 = tanh(p.burial_kappa * (p.burial_ro_max[k] - rho));
        e += -0.5 * p.k_burial * p.burial_gamma[it * 3 + k] * (t0 + t1);
    }
    return e;
}

// rho_i = sum_{j: |res_no diff| > seq_dist or cross-chain} theta_clamped(r_ij, well0).
__global__ void k_density(const double* coord, const int* res_seqid, const int* chain_id,
                          int n, DParams p, double* rho) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double acc = 0.0;
    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        const bool sep = chain_id[i] != chain_id[j] ||
                         abs(res_seqid[i] - res_seqid[j]) > p.seq_dist;
        if (sep) acc += d_theta_clamped(d_dist(coord, i, j), p.well_r_min0,
                                        p.well_r_max0, p.well_kappa);
    }
    rho[i] = acc;
}

// native energy per contact for configurational and mutational.
__global__ void k_native_contacts(const int* ci, const int* cj, int n_contacts,
                                   const double* coord, const int* res_type,
                                   const double* rho, int n, DParams p, int is_mut,
                                   double* native) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_contacts) return;
    const int i = ci[c], j = cj[c];
    double we = d_water(p, d_dist(coord, i, j), res_type[i], res_type[j], rho[i], rho[j]);
    if (is_mut) {
        for (int k = 0; k < n; ++k) {
            if (k == i || k == j) continue;
            const double rik = d_dist(coord, i, k);
            if (rik < p.contact_cutoff)
                we += d_water(p, rik, res_type[i], res_type[k], rho[i], rho[k]);
            const double rjk = d_dist(coord, j, k);
            if (rjk < p.contact_cutoff)
                we += d_water(p, rjk, res_type[j], res_type[k], rho[j], rho[k]);
        }
    }
    native[c] = we + d_burial(p, res_type[i], rho[i]) + d_burial(p, res_type[j], rho[j]);
}

// Block-per-contact mutational decoy reduction. decoy_it/decoy_jt hold the mutant
// residue-type at i/j for each (contact, decoy), generated host-side from the exact
// glibc stream. Threads stride over decoys; shared memory reduces to mean/sd.
__global__ void k_decoy_mut(const int* ci, const int* cj, int n_contacts,
                            const int* decoy_it, const int* decoy_jt, int nd,
                            const double* coord, const int* res_type, const double* rho,
                            int n, DParams p, double* out_mean, double* out_sd) {
    extern __shared__ double sh[];          // [blockDim] sum | [blockDim] sumsq
    double* s_sum = sh;
    double* s_sq = sh + blockDim.x;
    const int c = blockIdx.x;
    if (c >= n_contacts) return;
    const int i = ci[c], j = cj[c];
    const double rij = d_dist(coord, i, j);

    double lsum = 0.0, lsq = 0.0;
    for (int d = threadIdx.x; d < nd; d += blockDim.x) {
        const int it = decoy_it[c * nd + d];
        const int jt = decoy_jt[c * nd + d];
        double we = d_water(p, rij, it, jt, rho[i], rho[j]);
        for (int k = 0; k < n; ++k) {
            if (k == i || k == j) continue;
            const double rik = d_dist(coord, i, k);
            if (rik < p.contact_cutoff)
                we += d_water(p, rik, it, res_type[k], rho[i], rho[k]);
            const double rjk = d_dist(coord, j, k);
            if (rjk < p.contact_cutoff)
                we += d_water(p, rjk, jt, res_type[k], rho[j], rho[k]);
        }
        const double e = we + d_burial(p, it, rho[i]) + d_burial(p, jt, rho[j]);
        lsum += e;
        lsq += e * e;
    }
    s_sum[threadIdx.x] = lsum;
    s_sq[threadIdx.x] = lsq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_sq[threadIdx.x] += s_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const double mean = s_sum[0] / nd;
        out_mean[c] = mean;
        out_sd[c] = sqrt(s_sq[0] / nd - mean * mean);
    }
}

// singleresidue native with identity it at site i.
__device__ inline double d_native_single(const DParams& p, int i, int it,
                                         const double* coord, const int* res_type,
                                         const double* rho, const int* res_seqid,
                                         const int* chain_id, int n) {
    double e = d_burial(p, it, rho[i]);
    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        const double rij = d_dist(coord, i, j);
        const bool sep = chain_id[i] != chain_id[j] ||
                         abs(res_seqid[i] - res_seqid[j]) >= p.contact_min_sep;
        if (rij < p.contact_cutoff && sep)
            e += d_water(p, rij, it, res_type[j], rho[i], rho[j]);
    }
    return e;
}

__global__ void k_native_single(const double* coord, const int* res_type,
                                const double* rho, const int* res_seqid,
                                const int* chain_id, int n, DParams p, double* native) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    native[i] = d_native_single(p, i, res_type[i], coord, res_type, rho, res_seqid,
                                chain_id, n);
}

// Block-per-residue singleresidue decoy reduction. decoy_it[i*nd+d] is the mutant
// identity at site i for decoy d (host-generated glibc stream).
__global__ void k_decoy_single(const int* decoy_it, int nd, const double* coord,
                               const int* res_type, const double* rho,
                               const int* res_seqid, const int* chain_id, int n,
                               DParams p, double* out_mean, double* out_sd) {
    extern __shared__ double sh[];
    double* s_sum = sh;
    double* s_sq = sh + blockDim.x;
    const int i = blockIdx.x;
    if (i >= n) return;
    double lsum = 0.0, lsq = 0.0;
    for (int d = threadIdx.x; d < nd; d += blockDim.x) {
        const double e = d_native_single(p, i, decoy_it[i * nd + d], coord, res_type,
                                         rho, res_seqid, chain_id, n);
        lsum += e;
        lsq += e * e;
    }
    s_sum[threadIdx.x] = lsum;
    s_sq[threadIdx.x] = lsq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_sq[threadIdx.x] += s_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const double mean = s_sum[0] / nd;
        out_mean[i] = mean;
        out_sd[i] = sqrt(s_sq[0] / nd - mean * mean);
    }
}

// Host reproduction of glibc TYPE_3 rand() (default seed 1) -- identical to the CPU
// GlibcRand in core.cpp, kept here so the device decoy index stream matches exactly.
class GlibcRand {
public:
    explicit GlibcRand(std::uint32_t seed) {
        if (seed == 0) seed = 1;
        st_[0] = static_cast<std::int32_t>(seed);
        for (int i = 1; i < kDeg; ++i) {
            const std::int64_t hi = st_[i - 1] / 127773;
            const std::int64_t lo = st_[i - 1] % 127773;
            std::int64_t w = 16807 * lo - 2836 * hi;
            if (w < 0) w += 2147483647;
            st_[i] = static_cast<std::int32_t>(w);
        }
        fptr_ = kSep;
        rptr_ = 0;
        for (int i = 0; i < 10 * kDeg; ++i) next();
    }
    std::int32_t next() {
        const std::uint32_t sum = static_cast<std::uint32_t>(st_[fptr_]) +
                                  static_cast<std::uint32_t>(st_[rptr_]);
        st_[fptr_] = static_cast<std::int32_t>(sum);
        const std::int32_t r = static_cast<std::int32_t>((sum >> 1) & 0x7fffffffU);
        if (++fptr_ >= kDeg) fptr_ = 0;
        if (++rptr_ >= kDeg) rptr_ = 0;
        return r;
    }
    std::size_t residue_index(std::size_t n) {
        return static_cast<std::size_t>(next()) % n;
    }

private:
    static constexpr int kDeg = 31;
    static constexpr int kSep = 3;
    std::array<std::int32_t, kDeg> st_{};
    int fptr_ = 0, rptr_ = 0;
};

// Host copies of the energy terms (for the configurational decoys and contact list,
// which are computed host-side). Mirror the device/CPU formulas exactly.
double d_dist_host(const std::vector<double>& coord, int a, int b) {
    const double dx = coord[3 * a + 0] - coord[3 * b + 0];
    const double dy = coord[3 * a + 1] - coord[3 * b + 1];
    const double dz = coord[3 * a + 2] - coord[3 * b + 2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double theta_host(double r, double rmin, double rmax, double kappa) {
    return 0.25 * (1.0 + std::tanh(kappa * (r - rmin))) *
           (1.0 + std::tanh(kappa * (rmax - r)));
}

double water_energy_host(const ParamsView& p, double rij, int it, int jt, double rho_i,
                         double rho_j) {
    const double sigma_wat =
        0.25 * (1.0 - std::tanh(p.kappa_sigma * (rho_i - p.treshold))) *
        (1.0 - std::tanh(p.kappa_sigma * (rho_j - p.treshold)));
    const double sigma_prot = 1.0 - sigma_wat;
    const std::size_t g = static_cast<std::size_t>(it) * 20 + static_cast<std::size_t>(jt);
    const double sgd = p.gamma_direct[g];
    const double sgm = sigma_prot * p.gamma_protein[g] + sigma_wat * p.gamma_water[g];
    return -(sgd * theta_host(rij, p.well_r_min[0], p.well_r_max[0], p.well_kappa) +
             sgm * theta_host(rij, p.well_r_min[1], p.well_r_max[1], p.well_kappa));
}

double burial_energy_host(const ParamsView& p, int it, double rho) {
    double e = 0.0;
    for (int k = 0; k < 3; ++k) {
        const double t0 = std::tanh(p.burial_kappa * (rho - p.burial_ro_min[k]));
        const double t1 = std::tanh(p.burial_kappa * (p.burial_ro_max[k] - rho));
        e += -0.5 * p.k_burial * p.burial_gamma[static_cast<std::size_t>(it) * 3 + k] *
             (t0 + t1);
    }
    return e;
}

DParams make_dparams(const DeviceBuffer<double>& gd, const DeviceBuffer<double>& gw,
                     const DeviceBuffer<double>& gp, const DeviceBuffer<double>& bg,
                     const ParamsView& p) {
    DParams d{};
    d.gamma_direct = gd.get();
    d.gamma_water = gw.get();
    d.gamma_protein = gp.get();
    d.burial_gamma = bg.get();
    d.well_kappa = p.well_kappa;
    d.kappa_sigma = p.kappa_sigma;
    d.treshold = p.treshold;
    d.well_r_min0 = p.well_r_min[0];
    d.well_r_max0 = p.well_r_max[0];
    d.well_r_min1 = p.well_r_min[1];
    d.well_r_max1 = p.well_r_max[1];
    d.burial_kappa = p.burial_kappa;
    d.k_burial = p.k_burial;
    for (int k = 0; k < 3; ++k) {
        d.burial_ro_min[k] = p.burial_ro_min[k];
        d.burial_ro_max[k] = p.burial_ro_max[k];
    }
    d.contact_cutoff = p.contact_cutoff;
    d.contact_min_sep = p.contact_min_sep;
    d.seq_dist = p.seq_dist;
    return d;
}

}  // namespace

FrustrationResult compute_frustration_cuda(const StructureView& s, const ParamsView& p,
                                           const std::string& mode) {
    const int n = static_cast<int>(s.n_res);
    const int nd = p.n_decoys;
    const bool is_config = mode == "configurational";
    const bool is_mut = mode == "mutational";
    const bool is_single = mode == "singleresidue";

    // Upload the structure and gamma tables once (Best Practices: minimize transfer).
    std::vector<double> coord_h(s.coord.begin(), s.coord.end());
    std::vector<int> rtype_h(s.res_type.begin(), s.res_type.end());
    std::vector<int> chain_h(s.chain_id.begin(), s.chain_id.end());
    std::vector<int> seqid_h(s.res_seqid.begin(), s.res_seqid.end());
    DeviceBuffer<double> coord(coord_h), gd({p.gamma_direct.begin(), p.gamma_direct.end()}),
        gw({p.gamma_water.begin(), p.gamma_water.end()}),
        gp({p.gamma_protein.begin(), p.gamma_protein.end()}),
        bg({p.burial_gamma.begin(), p.burial_gamma.end()});
    DeviceBuffer<int> rtype(rtype_h), chain(chain_h), seqid(seqid_h);
    DParams dp = make_dparams(gd, gw, gp, bg, p);

    // Density on the GPU (one thread per residue), matching the CPU order.
    DeviceBuffer<double> rho_d(static_cast<std::size_t>(n));
    const int TPB = 128;
    k_density<<<(n + TPB - 1) / TPB, TPB>>>(coord.get(), seqid.get(), chain.get(), n, dp,
                                            rho_d.get());
    cuda_check(cudaGetLastError(), "k_density");
    std::vector<double> rho_h;
    rho_d.to_host(rho_h);

    FrustrationResult out;
    out.rho = rho_h;

    // Helper: contact-list separation, mirroring Engine::separated_contact (host).
    auto separated = [&](int i, int j) {
        if (chain_h[i] != chain_h[j]) return true;
        return std::abs(seqid_h[i] - seqid_h[j]) >= p.contact_min_sep;
    };

    GlibcRand rng(static_cast<std::uint32_t>(p.seed));

    if (is_single) {
        DeviceBuffer<double> native_d(static_cast<std::size_t>(n));
        k_native_single<<<(n + TPB - 1) / TPB, TPB>>>(coord.get(), rtype.get(),
            rho_d.get(), seqid.get(), chain.get(), n, dp, native_d.get());
        cuda_check(cudaGetLastError(), "k_native_single");
        // Host glibc stream: per residue, nd identities (1 rand each).
        std::vector<int> decoy_it(static_cast<std::size_t>(n) * nd);
        for (int i = 0; i < n; ++i)
            for (int d = 0; d < nd; ++d)
                decoy_it[static_cast<std::size_t>(i) * nd + d] =
                    rtype_h[rng.residue_index(static_cast<std::size_t>(n))];
        DeviceBuffer<int> decoy(decoy_it);
        DeviceBuffer<double> mean_d(static_cast<std::size_t>(n)), sd_d(static_cast<std::size_t>(n));
        const int RTPB = 256;
        k_decoy_single<<<n, RTPB, 2 * RTPB * sizeof(double)>>>(decoy.get(), nd,
            coord.get(), rtype.get(), rho_d.get(), seqid.get(), chain.get(), n, dp,
            mean_d.get(), sd_d.get());
        cuda_check(cudaGetLastError(), "k_decoy_single");
        std::vector<double> native_h, mean_h, sd_h;
        native_d.to_host(native_h);
        mean_d.to_host(mean_h);
        sd_d.to_host(sd_h);
        for (int i = 0; i < n; ++i) {
            out.unit_i.push_back(i);
            out.unit_j.push_back(-1);
            out.native_energy.push_back(native_h[i]);
            out.decoy_energy.push_back(mean_h[i]);
            out.sd_energy.push_back(sd_h[i]);
            out.frst_index.push_back((mean_h[i] - native_h[i]) / sd_h[i]);
        }
        cuda_check(cudaDeviceSynchronize(), "sync");
        return out;
    }

    // Contact modes: build the contact list in main-loop order.
    std::vector<int> ci_h, cj_h;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            if (d_dist_host(coord_h, i, j) < p.contact_cutoff && separated(i, j)) {
                ci_h.push_back(i);
                cj_h.push_back(j);
            }
        }
    const int nc = static_cast<int>(ci_h.size());
    DeviceBuffer<int> ci(ci_h), cj(cj_h);
    DeviceBuffer<double> native_d(static_cast<std::size_t>(nc ? nc : 1));
    k_native_contacts<<<(nc + TPB - 1) / TPB, TPB>>>(ci.get(), cj.get(), nc, coord.get(),
        rtype.get(), rho_d.get(), n, dp, is_mut ? 1 : 0, native_d.get());
    cuda_check(cudaGetLastError(), "k_native_contacts");
    std::vector<double> native_h;
    native_d.to_host(native_h);

    std::vector<double> mean_h(nc), sd_h(nc);
    if (is_config) {
        // Configurational decoys: computed once (host, glibc), reused for all contacts.
        std::vector<double> decoys(static_cast<std::size_t>(nd));
        for (int d = 0; d < nd; ++d) {
            std::size_t ri = rng.residue_index(n), rj = rng.residue_index(n);
            double rd = d_dist_host(coord_h, static_cast<int>(ri), static_cast<int>(rj));
            while (rd > p.contact_cutoff || ri == rj) {
                ri = rng.residue_index(n);
                rj = rng.residue_index(n);
                rd = d_dist_host(coord_h, static_cast<int>(ri), static_cast<int>(rj));
            }
            const std::size_t bi = rng.residue_index(n), bj = rng.residue_index(n);
            const int it = rtype_h[rng.residue_index(n)];
            const int jt = rtype_h[rng.residue_index(n)];
            decoys[d] = water_energy_host(p, rd, it, jt, rho_h[bi], rho_h[bj]) +
                        burial_energy_host(p, it, rho_h[bi]) +
                        burial_energy_host(p, jt, rho_h[bj]);
        }
        double m = 0.0;
        for (double x : decoys) m += x;
        m /= nd;
        double v = 0.0;
        for (double x : decoys) v += (x - m) * (x - m);
        const double sd = std::sqrt(v / nd);
        for (int c = 0; c < nc; ++c) { mean_h[c] = m; sd_h[c] = sd; }
    } else {
        // Mutational decoys: per contact, nd identity pairs (2 rand each), GPU energy.
        std::vector<int> decoy_it(static_cast<std::size_t>(nc) * nd);
        std::vector<int> decoy_jt(static_cast<std::size_t>(nc) * nd);
        for (int c = 0; c < nc; ++c)
            for (int d = 0; d < nd; ++d) {
                decoy_it[static_cast<std::size_t>(c) * nd + d] = rtype_h[rng.residue_index(n)];
                decoy_jt[static_cast<std::size_t>(c) * nd + d] = rtype_h[rng.residue_index(n)];
            }
        DeviceBuffer<int> dit(decoy_it), djt(decoy_jt);
        DeviceBuffer<double> mean_d(static_cast<std::size_t>(nc)), sd_d(static_cast<std::size_t>(nc));
        const int RTPB = 256;
        k_decoy_mut<<<nc, RTPB, 2 * RTPB * sizeof(double)>>>(ci.get(), cj.get(), nc,
            dit.get(), djt.get(), nd, coord.get(), rtype.get(), rho_d.get(), n, dp,
            mean_d.get(), sd_d.get());
        cuda_check(cudaGetLastError(), "k_decoy_mut");
        mean_d.to_host(mean_h);
        sd_d.to_host(sd_h);
    }

    for (int c = 0; c < nc; ++c) {
        out.unit_i.push_back(ci_h[c]);
        out.unit_j.push_back(cj_h[c]);
        out.native_energy.push_back(native_h[c]);
        out.decoy_energy.push_back(mean_h[c]);
        out.sd_energy.push_back(sd_h[c]);
        out.frst_index.push_back((mean_h[c] - native_h[c]) / sd_h[c]);
    }
    cuda_check(cudaDeviceSynchronize(), "sync");
    return out;
}

}  // namespace frustrapy_native
