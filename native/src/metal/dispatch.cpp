// Metal (Apple GPU) driver for the native frustration core (G1.4).
//
// Compiled only when FRUSTRAPY_NATIVE_METAL is ON (Apple, macOS). This file is the
// metal-cpp counterpart of src/cuda/kernels.cu's compute_frustration_cuda driver: it
// uploads the StructureView/ParamsView into MTL::Buffers once, encodes the MSL compute
// passes (src/metal/kernels.metal), and reads the results back into a FrustrationResult.
// The discrete decisions that must be bit-exact -- the contact list, the glibc decoy
// random-index stream, and the configurational decoy ensemble -- are done host-side in
// double exactly as the CUDA driver does them, so only the on-GPU energy arithmetic is
// affected by Apple's float32-only GPUs. See native/docs/METAL_BUILD.md.
//
// metal-cpp is Apple's single-header, pure-C++ binding for Metal; this is the ONE
// translation unit that pulls in its implementation (the *_PRIVATE_IMPLEMENTATION
// defines below), so it must not be compiled together with another TU that also
// defines them. It is pure C++ (no Objective-C), so CMake compiles it as a normal CXX
// source -- no OBJCXX language needed. metal-cpp headers are located via the
// FRUSTRAPY_METAL_CPP_DIR include path set in CMakeLists.txt.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "core.hpp"

namespace frustrapy_native {

namespace {

// Threads per threadgroup for the block-per-unit decoy reductions. MUST equal RTPB in
// kernels.metal and be a power of two (the tree reduction strides assume it). Apple
// GPUs allow up to 1024 threads/threadgroup, so 256 is always valid.
constexpr NS::UInteger kRTPB = 256;

// Scalar parameter bundle. MUST stay byte-identical to MParams in kernels.metal (POD
// floats/ints only, no arrays) so setBytes() lands the same layout the kernel reads.
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

MParams make_mparams(const ParamsView& p) {
    MParams m{};
    m.well_kappa = static_cast<float>(p.well_kappa);
    m.kappa_sigma = static_cast<float>(p.kappa_sigma);
    m.treshold = static_cast<float>(p.treshold);
    m.well_r_min0 = static_cast<float>(p.well_r_min[0]);
    m.well_r_max0 = static_cast<float>(p.well_r_max[0]);
    m.well_r_min1 = static_cast<float>(p.well_r_min[1]);
    m.well_r_max1 = static_cast<float>(p.well_r_max[1]);
    m.burial_kappa = static_cast<float>(p.burial_kappa);
    m.k_burial = static_cast<float>(p.k_burial);
    m.burial_ro_min0 = static_cast<float>(p.burial_ro_min[0]);
    m.burial_ro_min1 = static_cast<float>(p.burial_ro_min[1]);
    m.burial_ro_min2 = static_cast<float>(p.burial_ro_min[2]);
    m.burial_ro_max0 = static_cast<float>(p.burial_ro_max[0]);
    m.burial_ro_max1 = static_cast<float>(p.burial_ro_max[1]);
    m.burial_ro_max2 = static_cast<float>(p.burial_ro_max[2]);
    m.contact_cutoff = static_cast<float>(p.contact_cutoff);
    m.contact_min_sep = p.contact_min_sep;
    m.seq_dist = p.seq_dist;
    return m;
}

// ---- host glibc TYPE_3 rand() (identical to kernels.cu / core.cpp GlibcRand) ----
// Reproduces the exact decoy index stream the reference binary consumes, so the GPU
// path matches the CPU core's decoy choices.
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

// ---- host energy terms in double (for the contact list + configurational decoys,
// computed host-side exactly as kernels.cu does) ----
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

// ---- metal-cpp helpers ----
std::string ns_str(const NS::String* s) {
    return s ? std::string(s->utf8String()) : std::string();
}

// Directory of the loaded extension shared object, so we can find the .metallib the
// CMake build installs alongside it (the robust default for an installed wheel).
std::string module_dir() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&module_dir), &info) && info.dli_fname) {
        const std::string path(info.dli_fname);
        const auto slash = path.find_last_of('/');
        if (slash != std::string::npos) return path.substr(0, slash);
    }
    return std::string();
}

// Load the compiled MSL library, preferring a precompiled .metallib (fast, no runtime
// MSL compile). Search order: FRUSTRAPY_METALLIB env override, the .metallib next to
// this module, the compile-time FRUSTRAPY_METALLIB_PATH (if defined), then the process
// default library. Raises a clear, actionable error if none load.
MTL::Library* load_library(MTL::Device* device) {
    NS::Error* err = nullptr;
    std::vector<std::string> candidates;
    if (const char* env = std::getenv("FRUSTRAPY_METALLIB")) {
        if (env[0]) candidates.emplace_back(env);
    }
    const std::string dir = module_dir();
    if (!dir.empty()) candidates.push_back(dir + "/kernels.metallib");
#ifdef FRUSTRAPY_METALLIB_PATH
    candidates.emplace_back(FRUSTRAPY_METALLIB_PATH);
#endif
    for (const std::string& path : candidates) {
        NS::String* p = NS::String::string(path.c_str(), NS::UTF8StringEncoding);
        NS::URL* url = NS::URL::fileURLWithPath(p);
        MTL::Library* lib = device->newLibrary(url, &err);
        if (lib) return lib;
    }
    // Last resort: the default library bundled with the process (rarely present for a
    // Python extension, but cheap to try).
    MTL::Library* lib = device->newDefaultLibrary();
    if (lib) return lib;
    throw std::runtime_error(
        "Metal: could not load kernels.metallib. Set FRUSTRAPY_METALLIB to its path, "
        "or rebuild so the metallib installs next to the extension. See "
        "native/docs/METAL_BUILD.md. Last error: " + ns_str(err ? err->localizedDescription() : nullptr));
}

MTL::ComputePipelineState* make_pipeline(MTL::Device* device, MTL::Library* lib,
                                         const char* name) {
    NS::String* fn_name = NS::String::string(name, NS::UTF8StringEncoding);
    MTL::Function* fn = lib->newFunction(fn_name);
    if (!fn) throw std::runtime_error(std::string("Metal: kernel not found: ") + name);
    NS::Error* err = nullptr;
    MTL::ComputePipelineState* pso = device->newComputePipelineState(fn, &err);
    fn->release();
    if (!pso) {
        throw std::runtime_error(std::string("Metal: pipeline for ") + name + " failed: " +
                                 ns_str(err ? err->localizedDescription() : nullptr));
    }
    return pso;
}

}  // namespace

FrustrationResult compute_frustration_metal(const StructureView& s, const ParamsView& p,
                                            const std::string& mode) {
    const int n = static_cast<int>(s.n_res);
    const int nd = p.n_decoys;
    const bool is_config = mode == "configurational";
    const bool is_mut = mode == "mutational";
    const bool is_single = mode == "singleresidue";
    if (!is_config && !is_mut && !is_single) {
        throw std::invalid_argument("mode must be configurational, mutational, or "
                                    "singleresidue");
    }

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        pool->release();
        throw std::runtime_error("Metal: no system default GPU device (is this a Mac "
                                 "with a Metal-capable GPU?).");
    }
    if (kRTPB > device->maxThreadsPerThreadgroup().width) {
        const NS::UInteger maxw = device->maxThreadsPerThreadgroup().width;
        device->release();
        pool->release();
        throw std::runtime_error("Metal: device maxThreadsPerThreadgroup (" +
                                 std::to_string(maxw) + ") < required " +
                                 std::to_string(kRTPB) + ".");
    }
    MTL::Library* lib = load_library(device);
    MTL::CommandQueue* queue = device->newCommandQueue();

    // Pipelines (build only what the mode needs).
    MTL::ComputePipelineState* pso_density = make_pipeline(device, lib, "k_density");
    MTL::ComputePipelineState* pso_native_c = nullptr;
    MTL::ComputePipelineState* pso_decoy_mut = nullptr;
    MTL::ComputePipelineState* pso_native_s = nullptr;
    MTL::ComputePipelineState* pso_decoy_s = nullptr;

    // Host float copies for the GPU; host double copy for the bit-exact decisions.
    std::vector<float> coordf(s.coord.begin(), s.coord.end());
    std::vector<double> coord_h(s.coord.begin(), s.coord.end());
    std::vector<std::int32_t> rtype_h(s.res_type.begin(), s.res_type.end());
    std::vector<std::int32_t> chain_h(s.chain_id.begin(), s.chain_id.end());
    std::vector<std::int32_t> seqid_h(s.res_seqid.begin(), s.res_seqid.end());
    std::vector<float> gd(p.gamma_direct.begin(), p.gamma_direct.end());
    std::vector<float> gw(p.gamma_water.begin(), p.gamma_water.end());
    std::vector<float> gp(p.gamma_protein.begin(), p.gamma_protein.end());
    std::vector<float> bg(p.burial_gamma.begin(), p.burial_gamma.end());
    MParams mp = make_mparams(p);

    const MTL::ResourceOptions kShared = MTL::ResourceStorageModeShared;
    auto mkbuf = [&](const void* ptr, std::size_t bytes) {
        const std::size_t len = std::max<std::size_t>(bytes, 1);
        // Output / empty buffers carry no initial bytes: use the length-only overload.
        // newBuffer(ptr=nullptr, len) maps to newBufferWithBytes(nullptr, len) which
        // memcpy's from null and crashes, so never pass a null pointer here.
        if (ptr == nullptr || bytes == 0) {
            return device->newBuffer(len, kShared);
        }
        return device->newBuffer(ptr, len, kShared);
    };

    MTL::Buffer* b_coord = mkbuf(coordf.data(), coordf.size() * sizeof(float));
    MTL::Buffer* b_rtype = mkbuf(rtype_h.data(), rtype_h.size() * sizeof(std::int32_t));
    MTL::Buffer* b_chain = mkbuf(chain_h.data(), chain_h.size() * sizeof(std::int32_t));
    MTL::Buffer* b_seqid = mkbuf(seqid_h.data(), seqid_h.size() * sizeof(std::int32_t));
    MTL::Buffer* b_gd = mkbuf(gd.data(), gd.size() * sizeof(float));
    MTL::Buffer* b_gw = mkbuf(gw.data(), gw.size() * sizeof(float));
    MTL::Buffer* b_gp = mkbuf(gp.data(), gp.size() * sizeof(float));
    MTL::Buffer* b_bg = mkbuf(bg.data(), bg.size() * sizeof(float));
    MTL::Buffer* b_rho = mkbuf(nullptr, static_cast<std::size_t>(n) * sizeof(float));

    // Clean up everything Metal we new'd, in reverse-ish order, before returning.
    auto cleanup = [&]() {
        for (MTL::ComputePipelineState* x : {pso_density, pso_native_c, pso_decoy_mut,
                                             pso_native_s, pso_decoy_s})
            if (x) x->release();
        for (MTL::Buffer* x : {b_coord, b_rtype, b_chain, b_seqid, b_gd, b_gw, b_gp,
                               b_bg, b_rho})
            if (x) x->release();
        queue->release();
        lib->release();
        device->release();
        pool->release();
    };

    // --- pass 1: density (mirrors k_density on the GPU; rho read back to host) ---
    {
        MTL::CommandBuffer* cb = queue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pso_density);
        enc->setBuffer(b_coord, 0, 0);
        enc->setBuffer(b_seqid, 0, 1);
        enc->setBuffer(b_chain, 0, 2);
        enc->setBytes(&n, sizeof(int), 3);
        enc->setBytes(&mp, sizeof(MParams), 4);
        enc->setBuffer(b_rho, 0, 5);
        const NS::UInteger tg = std::min<NS::UInteger>(
            pso_density->maxTotalThreadsPerThreadgroup(), kRTPB);
        enc->dispatchThreads(MTL::Size(static_cast<NS::UInteger>(n), 1, 1),
                             MTL::Size(tg, 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
    }
    const float* rho_dev = static_cast<const float*>(b_rho->contents());
    std::vector<double> rho_h(static_cast<std::size_t>(n));
    std::vector<float> rho_f(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        rho_f[i] = rho_dev[i];
        rho_h[i] = static_cast<double>(rho_dev[i]);  // double view for host decoys
    }

    FrustrationResult out;
    out.rho = rho_h;

    GlibcRand rng(static_cast<std::uint32_t>(p.seed));

    // ---- singleresidue ----
    if (is_single) {
        pso_native_s = make_pipeline(device, lib, "k_native_single");
        pso_decoy_s = make_pipeline(device, lib, "k_decoy_single");
        MTL::Buffer* b_native = mkbuf(nullptr, static_cast<std::size_t>(n) * sizeof(float));
        // Host glibc stream: per residue, nd identities (1 rand each) -- order matches
        // kernels.cu k_decoy_single setup exactly.
        std::vector<std::int32_t> decoy_it(static_cast<std::size_t>(n) * nd);
        for (int i = 0; i < n; ++i)
            for (int d = 0; d < nd; ++d)
                decoy_it[static_cast<std::size_t>(i) * nd + d] =
                    rtype_h[rng.residue_index(static_cast<std::size_t>(n))];
        MTL::Buffer* b_decoy = mkbuf(decoy_it.data(), decoy_it.size() * sizeof(std::int32_t));
        MTL::Buffer* b_mean = mkbuf(nullptr, static_cast<std::size_t>(n) * sizeof(float));
        MTL::Buffer* b_sd = mkbuf(nullptr, static_cast<std::size_t>(n) * sizeof(float));

        MTL::CommandBuffer* cb = queue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        // native single (one thread per residue)
        enc->setComputePipelineState(pso_native_s);
        enc->setBuffer(b_coord, 0, 0);
        enc->setBuffer(b_rtype, 0, 1);
        enc->setBuffer(b_rho, 0, 2);
        enc->setBuffer(b_seqid, 0, 3);
        enc->setBuffer(b_chain, 0, 4);
        enc->setBytes(&n, sizeof(int), 5);
        enc->setBytes(&mp, sizeof(MParams), 6);
        enc->setBuffer(b_gd, 0, 7);
        enc->setBuffer(b_gw, 0, 8);
        enc->setBuffer(b_gp, 0, 9);
        enc->setBuffer(b_bg, 0, 10);
        enc->setBuffer(b_native, 0, 11);
        const NS::UInteger tg = std::min<NS::UInteger>(
            pso_native_s->maxTotalThreadsPerThreadgroup(), kRTPB);
        enc->dispatchThreads(MTL::Size(static_cast<NS::UInteger>(n), 1, 1),
                             MTL::Size(tg, 1, 1));
        // decoy single (one threadgroup per residue, kRTPB threads)
        enc->setComputePipelineState(pso_decoy_s);
        enc->setBuffer(b_decoy, 0, 0);
        enc->setBytes(&nd, sizeof(int), 1);
        enc->setBuffer(b_coord, 0, 2);
        enc->setBuffer(b_rtype, 0, 3);
        enc->setBuffer(b_rho, 0, 4);
        enc->setBuffer(b_seqid, 0, 5);
        enc->setBuffer(b_chain, 0, 6);
        enc->setBytes(&n, sizeof(int), 7);
        enc->setBytes(&mp, sizeof(MParams), 8);
        enc->setBuffer(b_gd, 0, 9);
        enc->setBuffer(b_gw, 0, 10);
        enc->setBuffer(b_gp, 0, 11);
        enc->setBuffer(b_bg, 0, 12);
        enc->setBuffer(b_mean, 0, 13);
        enc->setBuffer(b_sd, 0, 14);
        enc->dispatchThreadgroups(MTL::Size(static_cast<NS::UInteger>(n), 1, 1),
                                  MTL::Size(kRTPB, 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();

        const float* native_h = static_cast<const float*>(b_native->contents());
        const float* mean_h = static_cast<const float*>(b_mean->contents());
        const float* sd_h = static_cast<const float*>(b_sd->contents());
        for (int i = 0; i < n; ++i) {
            out.unit_i.push_back(i);
            out.unit_j.push_back(-1);
            out.native_energy.push_back(native_h[i]);
            out.decoy_energy.push_back(mean_h[i]);
            out.sd_energy.push_back(sd_h[i]);
            out.frst_index.push_back((static_cast<double>(mean_h[i]) -
                                      static_cast<double>(native_h[i])) /
                                     static_cast<double>(sd_h[i]));
        }
        b_native->release();
        b_decoy->release();
        b_mean->release();
        b_sd->release();
        cleanup();
        return out;
    }

    // ---- contact modes: build the contact list host-side (double, exact) in the
    // same main-loop order as kernels.cu ----
    auto separated = [&](int i, int j) {
        if (chain_h[i] != chain_h[j]) return true;
        return std::abs(seqid_h[i] - seqid_h[j]) >= p.contact_min_sep;
    };
    std::vector<std::int32_t> ci_h, cj_h;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (d_dist_host(coord_h, i, j) < p.contact_cutoff && separated(i, j)) {
                ci_h.push_back(i);
                cj_h.push_back(j);
            }
    const int nc = static_cast<int>(ci_h.size());

    pso_native_c = make_pipeline(device, lib, "k_native_contacts");
    MTL::Buffer* b_ci = mkbuf(ci_h.data(), ci_h.size() * sizeof(std::int32_t));
    MTL::Buffer* b_cj = mkbuf(cj_h.data(), cj_h.size() * sizeof(std::int32_t));
    MTL::Buffer* b_native = mkbuf(nullptr, static_cast<std::size_t>(std::max(nc, 1)) * sizeof(float));

    std::vector<double> mean_h(static_cast<std::size_t>(std::max(nc, 0)));
    std::vector<double> sd_h(static_cast<std::size_t>(std::max(nc, 0)));

    // Mutational-only decoy buffers (config decoys are host-computed below).
    MTL::Buffer* b_dit = nullptr;
    MTL::Buffer* b_djt = nullptr;
    MTL::Buffer* b_mean = nullptr;
    MTL::Buffer* b_sd = nullptr;

    if (is_mut) {
        pso_decoy_mut = make_pipeline(device, lib, "k_decoy_mut");
        // Per contact, nd identity pairs (2 rand each) -- order matches kernels.cu.
        std::vector<std::int32_t> decoy_it(static_cast<std::size_t>(std::max(nc, 1)) * nd);
        std::vector<std::int32_t> decoy_jt(static_cast<std::size_t>(std::max(nc, 1)) * nd);
        for (int c = 0; c < nc; ++c)
            for (int d = 0; d < nd; ++d) {
                decoy_it[static_cast<std::size_t>(c) * nd + d] =
                    rtype_h[rng.residue_index(static_cast<std::size_t>(n))];
                decoy_jt[static_cast<std::size_t>(c) * nd + d] =
                    rtype_h[rng.residue_index(static_cast<std::size_t>(n))];
            }
        b_dit = mkbuf(decoy_it.data(), decoy_it.size() * sizeof(std::int32_t));
        b_djt = mkbuf(decoy_jt.data(), decoy_jt.size() * sizeof(std::int32_t));
        b_mean = mkbuf(nullptr, static_cast<std::size_t>(std::max(nc, 1)) * sizeof(float));
        b_sd = mkbuf(nullptr, static_cast<std::size_t>(std::max(nc, 1)) * sizeof(float));
    } else {
        // Configurational decoys: computed once host-side in double (exactly as
        // kernels.cu does), reused for every contact. The rng call order -- ri, rj
        // (with the contact-distance reject loop), bi, bj, it, jt -- is identical.
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
    }

    // --- pass 2: native contact energy on the GPU (+ mutational decoys) ---
    const int is_mut_i = is_mut ? 1 : 0;
    if (nc > 0) {
        MTL::CommandBuffer* cb = queue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pso_native_c);
        enc->setBuffer(b_ci, 0, 0);
        enc->setBuffer(b_cj, 0, 1);
        enc->setBytes(&nc, sizeof(int), 2);
        enc->setBuffer(b_coord, 0, 3);
        enc->setBuffer(b_rtype, 0, 4);
        enc->setBuffer(b_rho, 0, 5);
        enc->setBytes(&n, sizeof(int), 6);
        enc->setBytes(&mp, sizeof(MParams), 7);
        enc->setBytes(&is_mut_i, sizeof(int), 8);
        enc->setBuffer(b_gd, 0, 9);
        enc->setBuffer(b_gw, 0, 10);
        enc->setBuffer(b_gp, 0, 11);
        enc->setBuffer(b_bg, 0, 12);
        enc->setBuffer(b_native, 0, 13);
        const NS::UInteger tg = std::min<NS::UInteger>(
            pso_native_c->maxTotalThreadsPerThreadgroup(), kRTPB);
        enc->dispatchThreads(MTL::Size(static_cast<NS::UInteger>(nc), 1, 1),
                             MTL::Size(tg, 1, 1));
        if (is_mut) {
            enc->setComputePipelineState(pso_decoy_mut);
            enc->setBuffer(b_ci, 0, 0);
            enc->setBuffer(b_cj, 0, 1);
            enc->setBytes(&nc, sizeof(int), 2);
            enc->setBuffer(b_dit, 0, 3);
            enc->setBuffer(b_djt, 0, 4);
            enc->setBytes(&nd, sizeof(int), 5);
            enc->setBuffer(b_coord, 0, 6);
            enc->setBuffer(b_rtype, 0, 7);
            enc->setBuffer(b_rho, 0, 8);
            enc->setBytes(&n, sizeof(int), 9);
            enc->setBytes(&mp, sizeof(MParams), 10);
            enc->setBuffer(b_gd, 0, 11);
            enc->setBuffer(b_gw, 0, 12);
            enc->setBuffer(b_gp, 0, 13);
            enc->setBuffer(b_bg, 0, 14);
            enc->setBuffer(b_mean, 0, 15);
            enc->setBuffer(b_sd, 0, 16);
            enc->dispatchThreadgroups(MTL::Size(static_cast<NS::UInteger>(nc), 1, 1),
                                      MTL::Size(kRTPB, 1, 1));
        }
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
    }

    const float* native_dev = static_cast<const float*>(b_native->contents());
    if (is_mut && nc > 0) {
        const float* mean_dev = static_cast<const float*>(b_mean->contents());
        const float* sd_dev = static_cast<const float*>(b_sd->contents());
        for (int c = 0; c < nc; ++c) {
            mean_h[c] = mean_dev[c];
            sd_h[c] = sd_dev[c];
        }
    }
    for (int c = 0; c < nc; ++c) {
        out.unit_i.push_back(ci_h[c]);
        out.unit_j.push_back(cj_h[c]);
        out.native_energy.push_back(native_dev[c]);
        out.decoy_energy.push_back(mean_h[c]);
        out.sd_energy.push_back(sd_h[c]);
        out.frst_index.push_back((mean_h[c] - static_cast<double>(native_dev[c])) / sd_h[c]);
    }

    b_ci->release();
    b_cj->release();
    b_native->release();
    if (b_dit) b_dit->release();
    if (b_djt) b_djt->release();
    if (b_mean) b_mean->release();
    if (b_sd) b_sd->release();
    cleanup();
    return out;
}

}  // namespace frustrapy_native
