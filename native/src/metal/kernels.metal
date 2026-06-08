// Metal (MSL) compute kernels for the native frustration core (G1.4).
//
// Compiled only when FRUSTRAPY_NATIVE_METAL is ON and `xcrun metal` is available
// (Apple, macOS). The dev container has no Apple GPU and no metal compiler, so this
// file is NOT built or run there -- it is authored to mirror the validated CUDA
// kernels (src/cuda/kernels.cu) term-for-term, which themselves mirror the CPU core
// (src/core.cpp) bit-for-bit. The decoy random-index stream and the configurational
// decoy ensemble are generated host-side (dispatch.cpp) with the exact glibc rand()
// sequence, so the GPU result tracks the CPU/CUDA reference up to floating-point
// precision.
//
// PRECISION -- READ THIS. Apple GPUs are float32-only (no IEEE double in MSL), while
// the CUDA path and the CPU core do every reduction in double. Every arithmetic line
// below that the CUDA kernel performs in `double` is performed here in `float`. The
// places this matters most (flagged inline with `TODO(metal-precision)`):
//   * the per-residue density accumulation (`k_density`) -- feeds tanh sigma in the
//     water energy, so a float density shift propagates into every contact energy;
//   * the decoy mean/sd threadgroup reductions (`k_decoy_mut`, `k_decoy_single`) --
//     a sum of ~1000 float energies loses low-order bits vs the double CUDA sum.
// The threadgroup reductions below use plain float accumulation to mirror the CUDA
// shared-memory tree exactly. If parity fails on real hardware, the maintainer can
// (a) switch the threadgroup accumulators to a Kahan/compensated float sum (the
// reduction loop is the only place to change), or (b) have dispatch.cpp read back the
// per-decoy energies and reduce mean/sd on the host in double (the energy eval stays
// on the GPU, which is the dominant cost). See native/docs/METAL_BUILD.md.
//
// LAYOUT COUPLING. The `MParams` struct below MUST stay byte-identical to the one in
// dispatch.cpp (POD floats/ints only, no arrays, to avoid any MSL/C++ alignment
// ambiguity). Buffer indices are per-kernel and must match the setBuffer/setBytes
// order in dispatch.cpp.

#include <metal_stdlib>
using namespace metal;

// Threads per threadgroup for the block-per-unit decoy reductions. Must match the
// threadsPerThreadgroup the host passes to dispatchThreadgroups(), and must be a power
// of two so the tree reduction below is exact in its strides (mirrors CUDA RTPB=256).
#define RTPB 256u

// Scalar parameter bundle (mirrors cuda kernels.cu DParams, minus the gamma device
// pointers, which are passed as separate device buffers). Keep in lockstep with the
// identical struct in dispatch.cpp.
struct MParams {
    float well_kappa;
    float kappa_sigma;
    float treshold;
    float well_r_min0;
    float well_r_max0;
    float well_r_min1;
    float well_r_max1;
    float burial_kappa;
    float k_burial;
    float burial_ro_min0;
    float burial_ro_min1;
    float burial_ro_min2;
    float burial_ro_max0;
    float burial_ro_max1;
    float burial_ro_max2;
    float contact_cutoff;
    int contact_min_sep;
    int seq_dist;
};

// ---- energy terms (mirror d_dist / d_theta / d_water / d_burial in kernels.cu) ----
// TODO(metal-precision): kernels.cu computes all of these in double; here every value
// is float because MSL has no double.

inline float m_dist(device const float* coord, int a, int b) {
    const float dx = coord[3 * a + 0] - coord[3 * b + 0];
    const float dy = coord[3 * a + 1] - coord[3 * b + 1];
    const float dz = coord[3 * a + 2] - coord[3 * b + 2];
    return sqrt(dx * dx + dy * dy + dz * dz);
}

inline float m_theta(float r, float rmin, float rmax, float kappa) {
    return 0.25f * (1.0f + tanh(kappa * (r - rmin))) * (1.0f + tanh(kappa * (rmax - r)));
}

inline float m_theta_clamped(float r, float rmin, float rmax, float kappa) {
    const float half = 8.0f * 2.302585f / kappa;
    if (r < rmin - half || r > rmax + half) return 0.0f;
    return m_theta(r, rmin, rmax, kappa);
}

inline float m_water(constant MParams& p, device const float* gamma_direct,
                     device const float* gamma_water, device const float* gamma_protein,
                     float rij, int it, int jt, float rho_i, float rho_j) {
    const float sigma_wat =
        0.25f * (1.0f - tanh(p.kappa_sigma * (rho_i - p.treshold))) *
        (1.0f - tanh(p.kappa_sigma * (rho_j - p.treshold)));
    const float sigma_prot = 1.0f - sigma_wat;
    const int g = it * 20 + jt;
    const float sgd = gamma_direct[g];
    const float sgm = sigma_prot * gamma_protein[g] + sigma_wat * gamma_water[g];
    return -(sgd * m_theta(rij, p.well_r_min0, p.well_r_max0, p.well_kappa) +
             sgm * m_theta(rij, p.well_r_min1, p.well_r_max1, p.well_kappa));
}

inline float m_burial(constant MParams& p, device const float* burial_gamma, int it,
                      float rho) {
    float romin[3] = {p.burial_ro_min0, p.burial_ro_min1, p.burial_ro_min2};
    float romax[3] = {p.burial_ro_max0, p.burial_ro_max1, p.burial_ro_max2};
    float e = 0.0f;
    for (int k = 0; k < 3; ++k) {
        const float t0 = tanh(p.burial_kappa * (rho - romin[k]));
        const float t1 = tanh(p.burial_kappa * (romax[k] - rho));
        e += -0.5f * p.k_burial * burial_gamma[it * 3 + k] * (t0 + t1);
    }
    return e;
}

// ---- density (mirrors k_density) ----
// rho_i = sum_{j: |res_no diff| > seq_dist or cross-chain} theta_clamped(r_ij, well0).
kernel void k_density(device const float* coord [[buffer(0)]],
                      device const int* res_seqid [[buffer(1)]],
                      device const int* chain_id [[buffer(2)]],
                      constant int& n [[buffer(3)]],
                      constant MParams& p [[buffer(4)]],
                      device float* rho [[buffer(5)]],
                      uint gid [[thread_position_in_grid]]) {
    const int i = (int)gid;
    if (i >= n) return;
    float acc = 0.0f;  // TODO(metal-precision): CUDA accumulates rho in double.
    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        const bool sep = chain_id[i] != chain_id[j] ||
                         abs(res_seqid[i] - res_seqid[j]) > p.seq_dist;
        if (sep) acc += m_theta_clamped(m_dist(coord, i, j), p.well_r_min0,
                                        p.well_r_max0, p.well_kappa);
    }
    rho[i] = acc;
}

// ---- native contact energy (mirrors k_native_contacts) ----
kernel void k_native_contacts(device const int* ci [[buffer(0)]],
                              device const int* cj [[buffer(1)]],
                              constant int& n_contacts [[buffer(2)]],
                              device const float* coord [[buffer(3)]],
                              device const int* res_type [[buffer(4)]],
                              device const float* rho [[buffer(5)]],
                              constant int& n [[buffer(6)]],
                              constant MParams& p [[buffer(7)]],
                              constant int& is_mut [[buffer(8)]],
                              device const float* gamma_direct [[buffer(9)]],
                              device const float* gamma_water [[buffer(10)]],
                              device const float* gamma_protein [[buffer(11)]],
                              device const float* burial_gamma [[buffer(12)]],
                              device float* native [[buffer(13)]],
                              uint gid [[thread_position_in_grid]]) {
    const int c = (int)gid;
    if (c >= n_contacts) return;
    const int i = ci[c], j = cj[c];
    float we = m_water(p, gamma_direct, gamma_water, gamma_protein, m_dist(coord, i, j),
                       res_type[i], res_type[j], rho[i], rho[j]);
    if (is_mut) {
        for (int k = 0; k < n; ++k) {
            if (k == i || k == j) continue;
            const float rik = m_dist(coord, i, k);
            if (rik < p.contact_cutoff)
                we += m_water(p, gamma_direct, gamma_water, gamma_protein, rik,
                              res_type[i], res_type[k], rho[i], rho[k]);
            const float rjk = m_dist(coord, j, k);
            if (rjk < p.contact_cutoff)
                we += m_water(p, gamma_direct, gamma_water, gamma_protein, rjk,
                              res_type[j], res_type[k], rho[j], rho[k]);
        }
    }
    native[c] = we + m_burial(p, burial_gamma, res_type[i], rho[i]) +
                m_burial(p, burial_gamma, res_type[j], rho[j]);
}

// ---- mutational decoy reduction (mirrors k_decoy_mut) ----
// One threadgroup per contact; threads stride over decoys; threadgroup memory reduces
// to mean/sd. decoy_it/decoy_jt hold the host-generated glibc mutant identities.
kernel void k_decoy_mut(device const int* ci [[buffer(0)]],
                        device const int* cj [[buffer(1)]],
                        constant int& n_contacts [[buffer(2)]],
                        device const int* decoy_it [[buffer(3)]],
                        device const int* decoy_jt [[buffer(4)]],
                        constant int& nd [[buffer(5)]],
                        device const float* coord [[buffer(6)]],
                        device const int* res_type [[buffer(7)]],
                        device const float* rho [[buffer(8)]],
                        constant int& n [[buffer(9)]],
                        constant MParams& p [[buffer(10)]],
                        device const float* gamma_direct [[buffer(11)]],
                        device const float* gamma_water [[buffer(12)]],
                        device const float* gamma_protein [[buffer(13)]],
                        device const float* burial_gamma [[buffer(14)]],
                        device float* out_mean [[buffer(15)]],
                        device float* out_sd [[buffer(16)]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint bid [[threadgroup_position_in_grid]],
                        uint tpg [[threads_per_threadgroup]]) {
    threadgroup float s_sum[RTPB];
    threadgroup float s_sq[RTPB];
    const int c = (int)bid;  // dispatch guarantees one threadgroup per valid contact
    const int i = ci[c], j = cj[c];
    const float rij = m_dist(coord, i, j);

    // TODO(metal-precision): CUDA reduces lsum/lsq in double; float here.
    float lsum = 0.0f, lsq = 0.0f;
    for (int d = (int)tid; d < nd; d += (int)tpg) {
        const int it = decoy_it[c * nd + d];
        const int jt = decoy_jt[c * nd + d];
        float we = m_water(p, gamma_direct, gamma_water, gamma_protein, rij, it, jt,
                           rho[i], rho[j]);
        for (int k = 0; k < n; ++k) {
            if (k == i || k == j) continue;
            const float rik = m_dist(coord, i, k);
            if (rik < p.contact_cutoff)
                we += m_water(p, gamma_direct, gamma_water, gamma_protein, rik, it,
                              res_type[k], rho[i], rho[k]);
            const float rjk = m_dist(coord, j, k);
            if (rjk < p.contact_cutoff)
                we += m_water(p, gamma_direct, gamma_water, gamma_protein, rjk, jt,
                              res_type[k], rho[j], rho[k]);
        }
        const float e = we + m_burial(p, burial_gamma, it, rho[i]) +
                        m_burial(p, burial_gamma, jt, rho[j]);
        lsum += e;
        lsq += e * e;
    }
    s_sum[tid] = lsum;
    s_sq[tid] = lsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sq[tid] += s_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        const float mean = s_sum[0] / (float)nd;
        out_mean[c] = mean;
        out_sd[c] = sqrt(s_sq[0] / (float)nd - mean * mean);
    }
}

// ---- singleresidue native (mirrors d_native_single / k_native_single) ----
inline float m_native_single(constant MParams& p, device const float* gamma_direct,
                             device const float* gamma_water,
                             device const float* gamma_protein,
                             device const float* burial_gamma, int i, int it,
                             device const float* coord, device const int* res_type,
                             device const float* rho, device const int* res_seqid,
                             device const int* chain_id, int n) {
    float e = m_burial(p, burial_gamma, it, rho[i]);
    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        const float rij = m_dist(coord, i, j);
        const bool sep = chain_id[i] != chain_id[j] ||
                         abs(res_seqid[i] - res_seqid[j]) >= p.contact_min_sep;
        if (rij < p.contact_cutoff && sep)
            e += m_water(p, gamma_direct, gamma_water, gamma_protein, rij, it,
                         res_type[j], rho[i], rho[j]);
    }
    return e;
}

kernel void k_native_single(device const float* coord [[buffer(0)]],
                            device const int* res_type [[buffer(1)]],
                            device const float* rho [[buffer(2)]],
                            device const int* res_seqid [[buffer(3)]],
                            device const int* chain_id [[buffer(4)]],
                            constant int& n [[buffer(5)]],
                            constant MParams& p [[buffer(6)]],
                            device const float* gamma_direct [[buffer(7)]],
                            device const float* gamma_water [[buffer(8)]],
                            device const float* gamma_protein [[buffer(9)]],
                            device const float* burial_gamma [[buffer(10)]],
                            device float* native [[buffer(11)]],
                            uint gid [[thread_position_in_grid]]) {
    const int i = (int)gid;
    if (i >= n) return;
    native[i] = m_native_single(p, gamma_direct, gamma_water, gamma_protein,
                                burial_gamma, i, res_type[i], coord, res_type, rho,
                                res_seqid, chain_id, n);
}

// ---- singleresidue decoy reduction (mirrors k_decoy_single) ----
// One threadgroup per residue; decoy_it[i*nd+d] is the host-generated glibc mutant
// identity at site i for decoy d.
kernel void k_decoy_single(device const int* decoy_it [[buffer(0)]],
                           constant int& nd [[buffer(1)]],
                           device const float* coord [[buffer(2)]],
                           device const int* res_type [[buffer(3)]],
                           device const float* rho [[buffer(4)]],
                           device const int* res_seqid [[buffer(5)]],
                           device const int* chain_id [[buffer(6)]],
                           constant int& n [[buffer(7)]],
                           constant MParams& p [[buffer(8)]],
                           device const float* gamma_direct [[buffer(9)]],
                           device const float* gamma_water [[buffer(10)]],
                           device const float* gamma_protein [[buffer(11)]],
                           device const float* burial_gamma [[buffer(12)]],
                           device float* out_mean [[buffer(13)]],
                           device float* out_sd [[buffer(14)]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint bid [[threadgroup_position_in_grid]],
                           uint tpg [[threads_per_threadgroup]]) {
    threadgroup float s_sum[RTPB];
    threadgroup float s_sq[RTPB];
    const int i = (int)bid;  // dispatch guarantees one threadgroup per valid residue
    float lsum = 0.0f, lsq = 0.0f;  // TODO(metal-precision): double in CUDA.
    for (int d = (int)tid; d < nd; d += (int)tpg) {
        const float e = m_native_single(p, gamma_direct, gamma_water, gamma_protein,
                                        burial_gamma, i, decoy_it[i * nd + d], coord,
                                        res_type, rho, res_seqid, chain_id, n);
        lsum += e;
        lsq += e * e;
    }
    s_sum[tid] = lsum;
    s_sq[tid] = lsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sq[tid] += s_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        const float mean = s_sum[0] / (float)nd;
        out_mean[i] = mean;
        out_sd[i] = sqrt(s_sq[0] / (float)nd - mean * mean);
    }
}
