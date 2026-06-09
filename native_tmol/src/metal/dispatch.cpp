// Metal (Apple GPU) driver for the torch-free all-atom ref2015 energy backend (M7).
//
// Compiled only when FRUSTRAMOL_TMOL_METAL is ON (Apple, macOS). This file is the
// metal-cpp counterpart of the CPU drivers in driver.hpp: it uploads the structure-of-
// arrays inputs into MTL::Buffers, encodes the MSL per-pair kernels (src/metal/
// kernels.metal), reads the per-pair per-subterm energies back, and accumulates them into
// the block-pair matrices host-side in double, identical to driver.hpp's `add` lambda
// (skip exact zero, place at min(b1,b2)/max(b1,b2)). The block-pair scatter is the cheap,
// discrete, order-defined part and stays on the host in double; only the per-pair energy
// arithmetic runs on the GPU in float32 (Apple GPUs have no IEEE double). This mirrors the
// AWSEM Metal lane (native/src/metal/dispatch.cpp), where the discrete decisions are
// host-side and the energy math is on the GPU. See docs/tmol/M7_NATIVE_METAL.md.
//
// metal-cpp is Apple's single-header, pure-C++ binding for Metal; this is the ONE
// translation unit that pulls in its implementation (the *_PRIVATE_IMPLEMENTATION defines
// below), so it must not be compiled together with another TU that also defines them. It
// is pure C++ (no Objective-C), so CMake compiles it as a normal CXX source. metal-cpp
// headers are located via the FRUSTRAMOL_TMOL_METAL_CPP_DIR include path set in
// CMakeLists.txt.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <dlfcn.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "frustramol_tmol/metal.hpp"

namespace frustramol_tmol {

namespace {

// ---- POD layouts shared with kernels.metal (MUST stay byte-identical) ----
// MSL default alignment: every field is 4 bytes, so these are tightly packed with no
// padding. Keep field order and types in lockstep with the structs in kernels.metal.
struct MAtom {
    float lj_radius;
    float lj_wdepth;
    float lk_dgfree;
    float lk_lambda;
    float lk_volume;
    float charge;
    std::uint32_t flags;
};
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

constexpr int kMaxWater = 4;

// Per-atom flag bits (must match kernels.metal F_* constants).
constexpr std::uint32_t F_DONOR = 1u;
constexpr std::uint32_t F_HYDROXYL = 2u;
constexpr std::uint32_t F_POLARH = 4u;
constexpr std::uint32_t F_ACCEPTOR = 8u;
constexpr std::uint32_t F_HEAVY = 16u;

// ---- metal-cpp helpers (mirror native/src/metal/dispatch.cpp) ----
std::string ns_str(const NS::String* s) {
    return s ? std::string(s->utf8String()) : std::string();
}

// Directory of the loaded extension shared object, so we can find the .metallib the CMake
// build installs alongside it.
std::string module_dir() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&module_dir), &info) && info.dli_fname) {
        const std::string path(info.dli_fname);
        const auto slash = path.find_last_of('/');
        if (slash != std::string::npos) return path.substr(0, slash);
    }
    return std::string();
}

// Load the compiled MSL library, preferring a precompiled .metallib next to the module.
// Search order: FRUSTRAMOL_TMOL_METALLIB env override, the .metallib beside this module,
// then the process default library.
MTL::Library* load_library(MTL::Device* device) {
    NS::Error* err = nullptr;
    std::vector<std::string> candidates;
    if (const char* env = std::getenv("FRUSTRAMOL_TMOL_METALLIB")) {
        if (env[0]) candidates.emplace_back(env);
    }
    const std::string dir = module_dir();
    if (!dir.empty()) candidates.push_back(dir + "/kernels.metallib");
#ifdef FRUSTRAMOL_TMOL_METALLIB_PATH
    candidates.emplace_back(FRUSTRAMOL_TMOL_METALLIB_PATH);
#endif
    for (const std::string& path : candidates) {
        NS::String* p = NS::String::string(path.c_str(), NS::UTF8StringEncoding);
        NS::URL* url = NS::URL::fileURLWithPath(p);
        MTL::Library* lib = device->newLibrary(url, &err);
        if (lib) return lib;
    }
    MTL::Library* lib = device->newDefaultLibrary();
    if (lib) return lib;
    throw std::runtime_error(
        "Metal: could not load kernels.metallib. Set FRUSTRAMOL_TMOL_METALLIB to its path, "
        "or rebuild so the metallib installs next to the extension. See "
        "docs/tmol/M7_NATIVE_METAL.md. Last error: " +
        ns_str(err ? err->localizedDescription() : nullptr));
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
        throw std::runtime_error(std::string("Metal: pipeline for ") + name +
                                 " failed: " +
                                 ns_str(err ? err->localizedDescription() : nullptr));
    }
    return pso;
}

// A device + queue + library bundle, owned for the span of one compute call.
struct MetalContext {
    MTL::Device* device = nullptr;
    MTL::CommandQueue* queue = nullptr;
    MTL::Library* lib = nullptr;
    NS::AutoreleasePool* pool = nullptr;

    MetalContext() {
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        if (!device) {
            pool->release();
            throw std::runtime_error("Metal: no system default GPU device (is this a Mac "
                                     "with a Metal-capable GPU?).");
        }
        lib = load_library(device);
        queue = device->newCommandQueue();
    }
    ~MetalContext() {
        if (queue) queue->release();
        if (lib) lib->release();
        if (device) device->release();
        if (pool) pool->release();
    }
    MTL::Buffer* mkbuf(const void* ptr, std::size_t bytes) {
        const std::size_t len = std::max<std::size_t>(bytes, 1);
        if (ptr == nullptr || bytes == 0) return device->newBuffer(len, kShared);
        return device->newBuffer(ptr, len, kShared);
    }
    static constexpr MTL::ResourceOptions kShared = MTL::ResourceStorageModeShared;
};

// Pack one atom's resolved ljlk/elec params + flags into the GPU struct. The atom's
// ljlkType indexes the folded per-atom type table (exactly as the CPU drivers consume it).
MAtom pack_atom(const AtomInput& a, const EnergyParams& params) {
    const LjlkTypeParams& tp =
        params.ljlkTypeParams[static_cast<std::size_t>(a.ljlkType)];
    MAtom m{};
    m.lj_radius = static_cast<float>(tp.lj_radius);
    m.lj_wdepth = static_cast<float>(tp.lj_wdepth);
    m.lk_dgfree = static_cast<float>(tp.lk_dgfree);
    m.lk_lambda = static_cast<float>(tp.lk_lambda);
    m.lk_volume = static_cast<float>(tp.lk_volume);
    m.charge = static_cast<float>(a.charge);
    std::uint32_t f = 0;
    if (tp.is_donor) f |= F_DONOR;
    if (tp.is_hydroxyl) f |= F_HYDROXYL;
    if (tp.is_polarh) f |= F_POLARH;
    if (tp.is_acceptor) f |= F_ACCEPTOR;
    if (a.isHeavy) f |= F_HEAVY;
    m.flags = f;
    return m;
}

MPoly pack_poly(const HBondPoly& p) {
    MPoly m{};
    for (int c = 0; c < 11; ++c) m.coeffs[c] = static_cast<float>(p.coeffs[c]);
    m.range[0] = static_cast<float>(p.range[0]);
    m.range[1] = static_cast<float>(p.range[1]);
    m.bound[0] = static_cast<float>(p.bound[0]);
    m.bound[1] = static_cast<float>(p.bound[1]);
    return m;
}

// Host-side block-pair accumulator, identical to driver.hpp's single-thread reduction:
// skip exact zero, place at (min(b1,b2), max(b1,b2)) in a double nBlocks x nBlocks matrix.
struct BlockPairAccum {
    int nBlocks;
    std::size_t NN;
    std::vector<std::vector<double>> mats;  // one per subterm
    explicit BlockPairAccum(int nb, std::size_t nSub)
        : nBlocks(nb),
          NN(static_cast<std::size_t>(nb) * nb),
          mats(nSub, std::vector<double>(NN, 0.0)) {}
    void add(std::size_t sub, int b1, int b2, double e) {
        if (e == 0.0) return;
        const int lo = b1 < b2 ? b1 : b2;
        const int hi = b1 < b2 ? b2 : b1;
        mats[sub][static_cast<std::size_t>(lo) * nBlocks + hi] += e;
    }
    PairEnergyResult finish(const std::vector<std::string>& names) {
        PairEnergyResult res;
        res.nBlocks = nBlocks;
        res.names = names;
        res.blockPair = mats;  // copy; mats kept for clarity
        res.wholePose.assign(names.size(), 0.0);
        for (std::size_t t = 0; t < names.size(); ++t) {
            double s = 0.0;
            for (double v : res.blockPair[t]) s += v;
            res.wholePose[t] = s;
        }
        return res;
    }
};

// Build the shared coordinate buffer (float[3*n]) and the packed MAtom buffer for a
// structure. Returns the float coord vector by reference for buffer lifetime.
void build_coords(const std::vector<AtomInput>& atoms, std::vector<float>& coordf) {
    coordf.resize(atoms.size() * 3);
    for (std::size_t k = 0; k < atoms.size(); ++k) {
        coordf[k * 3 + 0] = static_cast<float>(atoms[k].x);
        coordf[k * 3 + 1] = static_cast<float>(atoms[k].y);
        coordf[k * 3 + 2] = static_cast<float>(atoms[k].z);
    }
}

void dispatch_1d(MTL::ComputeCommandEncoder* enc, MTL::ComputePipelineState* pso,
                 int n_items) {
    const NS::UInteger tg =
        std::min<NS::UInteger>(pso->maxTotalThreadsPerThreadgroup(), 256);
    enc->dispatchThreads(MTL::Size(static_cast<NS::UInteger>(n_items), 1, 1),
                         MTL::Size(tg, 1, 1));
}

}  // namespace

bool metalAvailable() {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    const bool ok = device != nullptr;
    if (device) device->release();
    pool->release();
    return ok;
}

PairEnergyResult computePairEnergiesMetal(const std::vector<AtomInput>& atoms,
                                          const std::vector<PairInput>& pairs,
                                          const EnergyParams& params, int nBlocks) {
    const std::vector<std::string>& names = ljlkElecSubterms();
    const int np = static_cast<int>(pairs.size());
    BlockPairAccum acc(nBlocks, names.size());
    if (np == 0) return acc.finish(names);

    MetalContext ctx;
    MTL::ComputePipelineState* pso = make_pipeline(ctx.device, ctx.lib, "k_ljlk_elec");

    std::vector<float> coordf;
    build_coords(atoms, coordf);
    std::vector<MAtom> matom(atoms.size());
    for (std::size_t k = 0; k < atoms.size(); ++k) matom[k] = pack_atom(atoms[k], params);
    std::vector<std::int32_t> pi(np), pj(np), sl(np), se(np);
    for (int p = 0; p < np; ++p) {
        pi[p] = pairs[p].i;
        pj[p] = pairs[p].j;
        sl[p] = pairs[p].sepLjlk;
        se[p] = pairs[p].sepElec;
    }
    MLjlkGlobal ljg{static_cast<float>(params.ljlkGlobal.lj_hbond_dis),
                    static_cast<float>(params.ljlkGlobal.lj_hbond_OH_donor_dis),
                    static_cast<float>(params.ljlkGlobal.lj_hbond_hdis)};
    MElecGlobal eg{static_cast<float>(params.elecGlobal.D),
                   static_cast<float>(params.elecGlobal.D0),
                   static_cast<float>(params.elecGlobal.S),
                   static_cast<float>(params.elecGlobal.min_dis),
                   static_cast<float>(params.elecGlobal.max_dis)};

    MTL::Buffer* b_coord = ctx.mkbuf(coordf.data(), coordf.size() * sizeof(float));
    MTL::Buffer* b_atom = ctx.mkbuf(matom.data(), matom.size() * sizeof(MAtom));
    MTL::Buffer* b_pi = ctx.mkbuf(pi.data(), pi.size() * sizeof(std::int32_t));
    MTL::Buffer* b_pj = ctx.mkbuf(pj.data(), pj.size() * sizeof(std::int32_t));
    MTL::Buffer* b_sl = ctx.mkbuf(sl.data(), sl.size() * sizeof(std::int32_t));
    MTL::Buffer* b_se = ctx.mkbuf(se.data(), se.size() * sizeof(std::int32_t));
    MTL::Buffer* b_out =
        ctx.mkbuf(nullptr, static_cast<std::size_t>(np) * 4 * sizeof(float));

    {
        MTL::CommandBuffer* cb = ctx.queue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(b_coord, 0, 0);
        enc->setBuffer(b_atom, 0, 1);
        enc->setBuffer(b_pi, 0, 2);
        enc->setBuffer(b_pj, 0, 3);
        enc->setBuffer(b_sl, 0, 4);
        enc->setBuffer(b_se, 0, 5);
        enc->setBytes(&np, sizeof(int), 6);
        enc->setBytes(&ljg, sizeof(MLjlkGlobal), 7);
        enc->setBytes(&eg, sizeof(MElecGlobal), 8);
        enc->setBuffer(b_out, 0, 9);
        dispatch_1d(enc, pso, np);
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
    }

    const float* out = static_cast<const float*>(b_out->contents());
    // names: {fa_ljatr, fa_ljrep, fa_lk, fa_elec} -> out[4*p + {0,1,2,3}].
    for (int p = 0; p < np; ++p) {
        const int b1 = atoms[static_cast<std::size_t>(pairs[p].i)].block;
        const int b2 = atoms[static_cast<std::size_t>(pairs[p].j)].block;
        acc.add(0, b1, b2, static_cast<double>(out[4 * p + 0]));
        acc.add(1, b1, b2, static_cast<double>(out[4 * p + 1]));
        acc.add(2, b1, b2, static_cast<double>(out[4 * p + 2]));
        acc.add(3, b1, b2, static_cast<double>(out[4 * p + 3]));
    }

    b_coord->release();
    b_atom->release();
    b_pi->release();
    b_pj->release();
    b_sl->release();
    b_se->release();
    b_out->release();
    pso->release();
    return acc.finish(names);
}

PairEnergyResult computeLkBallMetal(const std::vector<AtomInput>& atoms,
                                    const std::vector<PairInput>& pairs,
                                    const EnergyParams& params, int nBlocks) {
    const std::vector<std::string>& names = lkBallSubterms();
    const int np = static_cast<int>(pairs.size());
    BlockPairAccum acc(nBlocks, names.size());
    if (np == 0) return acc.finish(names);

    MetalContext ctx;
    MTL::ComputePipelineState* pso = make_pipeline(ctx.device, ctx.lib, "k_lk_ball");

    std::vector<float> coordf;
    build_coords(atoms, coordf);
    std::vector<MAtom> matom(atoms.size());
    std::vector<float> waters(atoms.size() * kMaxWater * 3, 0.0f);
    std::vector<std::uint32_t> wmask(atoms.size(), 0u);
    for (std::size_t k = 0; k < atoms.size(); ++k) {
        matom[k] = pack_atom(atoms[k], params);
        for (int w = 0; w < kMaxWater; ++w) {
            if (atoms[k].waters.present[static_cast<std::size_t>(w)]) {
                wmask[k] |= (1u << static_cast<std::uint32_t>(w));
                const Vec3& wp = atoms[k].waters.pos[static_cast<std::size_t>(w)];
                const std::size_t base = (k * kMaxWater + static_cast<std::size_t>(w)) * 3;
                waters[base + 0] = static_cast<float>(wp[0]);
                waters[base + 1] = static_cast<float>(wp[1]);
                waters[base + 2] = static_cast<float>(wp[2]);
            }
        }
    }
    std::vector<std::int32_t> pi(np), pj(np), sl(np);
    for (int p = 0; p < np; ++p) {
        pi[p] = pairs[p].i;
        pj[p] = pairs[p].j;
        sl[p] = pairs[p].sepLjlk;
    }
    MLjlkGlobal ljg{static_cast<float>(params.ljlkGlobal.lj_hbond_dis),
                    static_cast<float>(params.ljlkGlobal.lj_hbond_OH_donor_dis),
                    static_cast<float>(params.ljlkGlobal.lj_hbond_hdis)};
    MLkBallGlobal lkg{static_cast<float>(params.lkBallGlobal.lj_hbond_dis),
                      static_cast<float>(params.lkBallGlobal.lj_hbond_OH_donor_dis),
                      static_cast<float>(params.lkBallGlobal.lj_hbond_hdis),
                      static_cast<float>(params.lkBallGlobal.lkb_water_dist),
                      static_cast<float>(params.lkBallGlobal.distance_threshold)};

    MTL::Buffer* b_coord = ctx.mkbuf(coordf.data(), coordf.size() * sizeof(float));
    MTL::Buffer* b_atom = ctx.mkbuf(matom.data(), matom.size() * sizeof(MAtom));
    MTL::Buffer* b_wat = ctx.mkbuf(waters.data(), waters.size() * sizeof(float));
    MTL::Buffer* b_wm = ctx.mkbuf(wmask.data(), wmask.size() * sizeof(std::uint32_t));
    MTL::Buffer* b_pi = ctx.mkbuf(pi.data(), pi.size() * sizeof(std::int32_t));
    MTL::Buffer* b_pj = ctx.mkbuf(pj.data(), pj.size() * sizeof(std::int32_t));
    MTL::Buffer* b_sl = ctx.mkbuf(sl.data(), sl.size() * sizeof(std::int32_t));
    MTL::Buffer* b_out =
        ctx.mkbuf(nullptr, static_cast<std::size_t>(np) * 8 * sizeof(float));

    {
        MTL::CommandBuffer* cb = ctx.queue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(b_coord, 0, 0);
        enc->setBuffer(b_atom, 0, 1);
        enc->setBuffer(b_wat, 0, 2);
        enc->setBuffer(b_wm, 0, 3);
        enc->setBuffer(b_pi, 0, 4);
        enc->setBuffer(b_pj, 0, 5);
        enc->setBuffer(b_sl, 0, 6);
        enc->setBytes(&np, sizeof(int), 7);
        enc->setBytes(&ljg, sizeof(MLjlkGlobal), 8);
        enc->setBytes(&lkg, sizeof(MLkBallGlobal), 9);
        enc->setBuffer(b_out, 0, 10);
        dispatch_1d(enc, pso, np);
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
    }

    const float* out = static_cast<const float*>(b_out->contents());
    // out[8*p + {0..3}] = dir(i polar, j occ); out[8*p + {4..7}] = dir(j polar, i occ).
    // Each block: (lk_ball_iso, lk_ball, lk_bridge, lk_bridge_uncpl), placed at the same
    // (min,max) block cell -- exactly the two scoreDir calls in driver.hpp.
    for (int p = 0; p < np; ++p) {
        const int bi = atoms[static_cast<std::size_t>(pairs[p].i)].block;
        const int bj = atoms[static_cast<std::size_t>(pairs[p].j)].block;
        for (int s = 0; s < 4; ++s) {
            acc.add(static_cast<std::size_t>(s), bi, bj,
                    static_cast<double>(out[8 * p + s]));
            acc.add(static_cast<std::size_t>(s), bj, bi,
                    static_cast<double>(out[8 * p + 4 + s]));
        }
    }

    b_coord->release();
    b_atom->release();
    b_wat->release();
    b_wm->release();
    b_pi->release();
    b_pj->release();
    b_sl->release();
    b_out->release();
    pso->release();
    return acc.finish(names);
}

PairEnergyResult computeHbondMetal(const std::vector<AtomInput>& atoms,
                                   const std::vector<HBondPairInput>& hbondPairs,
                                   const EnergyParams& params, int nBlocks) {
    const std::vector<std::string>& names = hbondSubterms();
    const int np = static_cast<int>(hbondPairs.size());
    BlockPairAccum acc(nBlocks, names.size());
    if (np == 0) return acc.finish(names);

    MetalContext ctx;
    MTL::ComputePipelineState* pso = make_pipeline(ctx.device, ctx.lib, "k_hbond");

    // Per-pair resolved geometry: H/A from the synthetic atom array, D/B/B0 from the pair.
    std::vector<float> H(np * 3), A(np * 3), D(np * 3), B(np * 3), B0(np * 3);
    std::vector<std::int32_t> hyb(np);
    std::vector<float> adw(np);
    std::vector<MPoly> ahdist(np), cosbah(np), cosahd(np);
    for (int p = 0; p < np; ++p) {
        const HBondPairInput& hp = hbondPairs[static_cast<std::size_t>(p)];
        const AtomInput& hatom = atoms[static_cast<std::size_t>(hp.h)];
        const AtomInput& aatom = atoms[static_cast<std::size_t>(hp.a)];
        H[p * 3 + 0] = static_cast<float>(hatom.x);
        H[p * 3 + 1] = static_cast<float>(hatom.y);
        H[p * 3 + 2] = static_cast<float>(hatom.z);
        A[p * 3 + 0] = static_cast<float>(aatom.x);
        A[p * 3 + 1] = static_cast<float>(aatom.y);
        A[p * 3 + 2] = static_cast<float>(aatom.z);
        D[p * 3 + 0] = static_cast<float>(hp.D[0]);
        D[p * 3 + 1] = static_cast<float>(hp.D[1]);
        D[p * 3 + 2] = static_cast<float>(hp.D[2]);
        B[p * 3 + 0] = static_cast<float>(hp.B[0]);
        B[p * 3 + 1] = static_cast<float>(hp.B[1]);
        B[p * 3 + 2] = static_cast<float>(hp.B[2]);
        B0[p * 3 + 0] = static_cast<float>(hp.B0[0]);
        B0[p * 3 + 1] = static_cast<float>(hp.B0[1]);
        B0[p * 3 + 2] = static_cast<float>(hp.B0[2]);
        hyb[p] = hp.pair.hyb;
        adw[p] = static_cast<float>(hp.pair.ad_weight);
        ahdist[p] = pack_poly(hp.pair.AHdist);
        cosbah[p] = pack_poly(hp.pair.cosBAH);
        cosahd[p] = pack_poly(hp.pair.cosAHD);
    }
    MHbondGlobal hg{static_cast<float>(params.hbondGlobal.hb_sp2_range_span),
                    static_cast<float>(params.hbondGlobal.hb_sp2_BAH180_rise),
                    static_cast<float>(params.hbondGlobal.hb_sp2_outer_width),
                    static_cast<float>(params.hbondGlobal.hb_sp3_softmax_fade),
                    static_cast<float>(params.hbondGlobal.threshold_distance),
                    static_cast<float>(params.hbondGlobal.max_ha_dis)};

    MTL::Buffer* b_H = ctx.mkbuf(H.data(), H.size() * sizeof(float));
    MTL::Buffer* b_A = ctx.mkbuf(A.data(), A.size() * sizeof(float));
    MTL::Buffer* b_D = ctx.mkbuf(D.data(), D.size() * sizeof(float));
    MTL::Buffer* b_B = ctx.mkbuf(B.data(), B.size() * sizeof(float));
    MTL::Buffer* b_B0 = ctx.mkbuf(B0.data(), B0.size() * sizeof(float));
    MTL::Buffer* b_hyb = ctx.mkbuf(hyb.data(), hyb.size() * sizeof(std::int32_t));
    MTL::Buffer* b_adw = ctx.mkbuf(adw.data(), adw.size() * sizeof(float));
    MTL::Buffer* b_ah = ctx.mkbuf(ahdist.data(), ahdist.size() * sizeof(MPoly));
    MTL::Buffer* b_cb = ctx.mkbuf(cosbah.data(), cosbah.size() * sizeof(MPoly));
    MTL::Buffer* b_ca = ctx.mkbuf(cosahd.data(), cosahd.size() * sizeof(MPoly));
    MTL::Buffer* b_out = ctx.mkbuf(nullptr, static_cast<std::size_t>(np) * sizeof(float));

    {
        MTL::CommandBuffer* cb = ctx.queue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(b_H, 0, 0);
        enc->setBuffer(b_A, 0, 1);
        enc->setBuffer(b_D, 0, 2);
        enc->setBuffer(b_B, 0, 3);
        enc->setBuffer(b_B0, 0, 4);
        enc->setBuffer(b_hyb, 0, 5);
        enc->setBuffer(b_adw, 0, 6);
        enc->setBuffer(b_ah, 0, 7);
        enc->setBuffer(b_cb, 0, 8);
        enc->setBuffer(b_ca, 0, 9);
        enc->setBytes(&np, sizeof(int), 10);
        enc->setBytes(&hg, sizeof(MHbondGlobal), 11);
        enc->setBuffer(b_out, 0, 12);
        dispatch_1d(enc, pso, np);
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
    }

    const float* out = static_cast<const float*>(b_out->contents());
    // The count-pair exclusion (sep < 5) is applied host-side, exactly as driver.hpp's
    // computeHbondCPU skips those pairs before scoring.
    for (int p = 0; p < np; ++p) {
        const HBondPairInput& hp = hbondPairs[static_cast<std::size_t>(p)];
        if (hp.sep < 5) continue;
        const int bh = atoms[static_cast<std::size_t>(hp.h)].block;
        const int ba = atoms[static_cast<std::size_t>(hp.a)].block;
        acc.add(0, bh, ba, static_cast<double>(out[p]));
    }

    b_H->release();
    b_A->release();
    b_D->release();
    b_B->release();
    b_B0->release();
    b_hyb->release();
    b_adw->release();
    b_ah->release();
    b_cb->release();
    b_ca->release();
    b_out->release();
    pso->release();
    return acc.finish(names);
}

}  // namespace frustramol_tmol
