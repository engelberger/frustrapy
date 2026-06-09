// nanobind bindings for the torch-free CPU energy kernels.
//
// A thin structure-of-arrays interface: NumPy arrays (contiguous CPU, fixed dtype) cross
// the boundary, the binding rebuilds the kernel structs, runs the CPU driver, and returns
// a dict of per-subterm (n_blocks, n_blocks) matrices that own their C++-allocated memory
// through an nb::capsule deleter (the nanobind ownership rule). No torch. Mirrors the
// AWSEM native core's bindings (native/src/bindings.cpp). See native_tmol/NOTICE.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "frustramol_tmol/cuda.hpp"
#include "frustramol_tmol/driver.hpp"
#ifdef FRUSTRAMOL_TMOL_METAL
#include "frustramol_tmol/metal.hpp"
#endif

namespace nb = nanobind;
using namespace frustramol_tmol;

namespace {
// True iff the optional Metal path is compiled in AND a Metal device is present. The
// `use_metal=True` kwarg on the compute functions routes to the GPU driver; without the
// Metal build it raises a clear error (the default `use_metal=False` CPU path is unchanged,
// so the in-container build and parity gate are byte-identical).
bool tmol_has_metal() {
#ifdef FRUSTRAMOL_TMOL_METAL
    return frustramol_tmol::metalAvailable();
#else
    return false;
#endif
}

[[noreturn]] void no_metal_build() {
    throw std::runtime_error(
        "frustramol_tmol was built without the Metal path. Rebuild on a Mac with "
        "-C cmake.define.FRUSTRAMOL_TMOL_METAL=ON (see docs/tmol/M7_NATIVE_METAL.md).");
}
}  // namespace

namespace {

// Guard the optional GPU path: a use_cuda=True request on a CPU-only build is a clear
// rebuild error, never a silent fall-back to the CPU (which would hide that the GPU path
// was not exercised), mirroring the AWSEM native core's use_cuda guard.
void require_cuda(bool use_cuda) {
    if (use_cuda && !hasCuda()) {
        throw std::runtime_error(
            "use_cuda=True but frustramol_tmol was built without CUDA. Rebuild with "
            "`pip install ./native_tmol -C cmake.define.FRUSTRAMOL_TMOL_CUDA=ON` on a "
            "CUDA host (nvcc required). has_cuda() reports the build state.");
    }
}

using F64_1 = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using F64_2 = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using F64_3 = nb::ndarray<const double, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
using I32_1 = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using B_1 = nb::ndarray<const bool, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using B_2 = nb::ndarray<const bool, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// Move a std::vector<double> into an (nb, nb) NumPy array that owns it via a capsule.
nb::ndarray<nb::numpy, double> own2d(std::vector<double>&& v, int nbk) {
    auto* held = new std::vector<double>(std::move(v));
    nb::capsule owner(held, [](void* p) noexcept {
        delete static_cast<std::vector<double>*>(p);
    });
    return nb::ndarray<nb::numpy, double>(
        held->data(),
        {static_cast<std::size_t>(nbk), static_cast<std::size_t>(nbk)}, owner);
}

nb::dict result_to_dict(PairEnergyResult&& res) {
    nb::dict d;
    for (std::size_t t = 0; t < res.names.size(); ++t)
        d[res.names[t].c_str()] = own2d(std::move(res.blockPair[t]), res.nBlocks);
    return d;
}

// Build the per-atom inputs and the (per-atom, folded) ljlk type table from SoA arrays.
// Optionally attach the lk_ball waters. ljlkType is the atom index, so each atom carries
// its own one-row type, exactly like the offline fixture loader.
std::vector<AtomInput> build_atoms(const F64_2& coords, const I32_1& block,
                                   const F64_1& charge, const B_1& is_heavy,
                                   const F64_1& lj_radius, const F64_1& lj_wdepth,
                                   const F64_1& lk_dgfree, const F64_1& lk_lambda,
                                   const F64_1& lk_volume, const B_1& is_donor,
                                   const B_1& is_hydroxyl, const B_1& is_polarh,
                                   const B_1& is_acceptor, EnergyParams& params,
                                   const F64_3* waters, const B_2* water_present) {
    const std::size_t n = coords.shape(0);
    if (block.shape(0) != n || charge.shape(0) != n || is_heavy.shape(0) != n ||
        lj_radius.shape(0) != n || lj_wdepth.shape(0) != n ||
        lk_dgfree.shape(0) != n || lk_lambda.shape(0) != n ||
        lk_volume.shape(0) != n || is_donor.shape(0) != n ||
        is_hydroxyl.shape(0) != n || is_polarh.shape(0) != n ||
        is_acceptor.shape(0) != n)
        throw std::invalid_argument("all per-atom arrays must share length n_atoms");

    const double* C = coords.data();
    std::vector<AtomInput> atoms(n);
    params.ljlkTypeParams.resize(n);
    for (std::size_t k = 0; k < n; ++k) {
        AtomInput& a = atoms[k];
        a.x = C[k * 3 + 0];
        a.y = C[k * 3 + 1];
        a.z = C[k * 3 + 2];
        a.block = block.data()[k];
        a.ljlkType = static_cast<int>(k);
        a.charge = charge.data()[k];
        a.isHeavy = is_heavy.data()[k];
        LjlkTypeParams& tp = params.ljlkTypeParams[k];
        tp.lj_radius = lj_radius.data()[k];
        tp.lj_wdepth = lj_wdepth.data()[k];
        tp.lk_dgfree = lk_dgfree.data()[k];
        tp.lk_lambda = lk_lambda.data()[k];
        tp.lk_volume = lk_volume.data()[k];
        tp.is_donor = is_donor.data()[k];
        tp.is_hydroxyl = is_hydroxyl.data()[k];
        tp.is_polarh = is_polarh.data()[k];
        tp.is_acceptor = is_acceptor.data()[k];
    }
    if (waters != nullptr && water_present != nullptr) {
        if (waters->shape(0) != n || water_present->shape(0) != n)
            throw std::invalid_argument("waters arrays must have length n_atoms");
        const std::size_t mw = waters->shape(1);
        const double* W = waters->data();
        const bool* P = water_present->data();
        for (std::size_t k = 0; k < n; ++k)
            for (std::size_t w = 0; w < mw && w < MAX_WATER; ++w)
                if (P[k * mw + w]) {
                    atoms[k].hasWaters = true;
                    atoms[k].waters.present[w] = true;
                    const std::size_t base = (k * mw + w) * 3;
                    atoms[k].waters.pos[w] = {W[base], W[base + 1], W[base + 2]};
                }
    }
    return atoms;
}

std::vector<PairInput> build_pairs(const I32_1& i, const I32_1& j,
                                   const I32_1& sep_ljlk, const I32_1& sep_elec) {
    const std::size_t n = i.shape(0);
    if (j.shape(0) != n || sep_ljlk.shape(0) != n || sep_elec.shape(0) != n)
        throw std::invalid_argument("all pair arrays must share length n_pairs");
    std::vector<PairInput> pairs(n);
    for (std::size_t p = 0; p < n; ++p)
        pairs[p] = {i.data()[p], j.data()[p], sep_ljlk.data()[p],
                    sep_elec.data()[p]};
    return pairs;
}

LjlkGlobalParams read_ljlk_global(const F64_1& g) {
    if (g.shape(0) != 3) throw std::invalid_argument("ljlk_global must have length 3");
    return {g.data()[0], g.data()[1], g.data()[2]};
}
ElecGlobalParams read_elec_global(const F64_1& g) {
    if (g.shape(0) != 5) throw std::invalid_argument("elec_global must have length 5");
    return {g.data()[0], g.data()[1], g.data()[2], g.data()[3], g.data()[4]};
}
LkBallGlobalParams read_lk_ball_global(const F64_1& g) {
    if (g.shape(0) != 5)
        throw std::invalid_argument("lk_ball_global must have length 5");
    return {g.data()[0], g.data()[1], g.data()[2], g.data()[3], g.data()[4]};
}
HBondGlobalParams read_hbond_global(const F64_1& g) {
    if (g.shape(0) != 6)
        throw std::invalid_argument("hbond_global must have length 6");
    return {g.data()[0], g.data()[1], g.data()[2],
            g.data()[3], g.data()[4], g.data()[5]};
}

// ---- ljlk + fa_elec ----
nb::dict compute_pair_energies(F64_2 coords, I32_1 block, F64_1 charge, B_1 is_heavy,
                               F64_1 lj_radius, F64_1 lj_wdepth, F64_1 lk_dgfree,
                               F64_1 lk_lambda, F64_1 lk_volume, B_1 is_donor,
                               B_1 is_hydroxyl, B_1 is_polarh, B_1 is_acceptor,
                               I32_1 pair_i, I32_1 pair_j, I32_1 sep_ljlk,
                               I32_1 sep_elec, F64_1 ljlk_global, F64_1 elec_global,
                               int n_blocks, int n_threads, bool use_cuda,
                               bool use_metal) {
    require_cuda(use_cuda);
    EnergyParams params;
    auto atoms = build_atoms(coords, block, charge, is_heavy, lj_radius, lj_wdepth,
                             lk_dgfree, lk_lambda, lk_volume, is_donor, is_hydroxyl,
                             is_polarh, is_acceptor, params, nullptr, nullptr);
    auto pairs = build_pairs(pair_i, pair_j, sep_ljlk, sep_elec);
    params.ljlkGlobal = read_ljlk_global(ljlk_global);
    params.elecGlobal = read_elec_global(elec_global);
#ifdef FRUSTRAMOL_TMOL_CUDA
    if (use_cuda)
        return result_to_dict(
            computePairEnergiesCUDA(atoms, pairs, params, n_blocks));
#endif
    if (use_metal) {
#ifdef FRUSTRAMOL_TMOL_METAL
        return result_to_dict(
            computePairEnergiesMetal(atoms, pairs, params, n_blocks));
#else
        no_metal_build();
#endif
    }
    return result_to_dict(
        computePairEnergiesCPU(atoms, pairs, params, n_blocks, n_threads));
}

// ---- lk_ball (four subterms) ----
nb::dict compute_lk_ball(F64_2 coords, I32_1 block, F64_1 charge, B_1 is_heavy,
                         F64_1 lj_radius, F64_1 lj_wdepth, F64_1 lk_dgfree,
                         F64_1 lk_lambda, F64_1 lk_volume, B_1 is_donor,
                         B_1 is_hydroxyl, B_1 is_polarh, B_1 is_acceptor,
                         F64_3 waters, B_2 water_present, I32_1 pair_i, I32_1 pair_j,
                         I32_1 sep_ljlk, I32_1 sep_elec, F64_1 ljlk_global,
                         F64_1 lk_ball_global, int n_blocks, int n_threads,
                         bool use_cuda, bool use_metal) {
    require_cuda(use_cuda);
    EnergyParams params;
    auto atoms = build_atoms(coords, block, charge, is_heavy, lj_radius, lj_wdepth,
                             lk_dgfree, lk_lambda, lk_volume, is_donor, is_hydroxyl,
                             is_polarh, is_acceptor, params, &waters, &water_present);
    auto pairs = build_pairs(pair_i, pair_j, sep_ljlk, sep_elec);
    params.ljlkGlobal = read_ljlk_global(ljlk_global);
    params.lkBallGlobal = read_lk_ball_global(lk_ball_global);
#ifdef FRUSTRAMOL_TMOL_CUDA
    if (use_cuda)
        return result_to_dict(computeLkBallCUDA(atoms, pairs, params, n_blocks));
#endif
    if (use_metal) {
#ifdef FRUSTRAMOL_TMOL_METAL
        return result_to_dict(computeLkBallMetal(atoms, pairs, params, n_blocks));
#else
        no_metal_build();
#endif
    }
    return result_to_dict(
        computeLkBallCPU(atoms, pairs, params, n_blocks, n_threads));
}

HBondPoly read_poly(const F64_2& coeffs, const F64_2& range, const F64_2& bound,
                    std::size_t p) {
    HBondPoly poly{};
    for (int c = 0; c < 11; ++c) poly.coeffs[c] = coeffs.data()[p * 11 + c];
    poly.range = {range.data()[p * 2 + 0], range.data()[p * 2 + 1]};
    poly.bound = {bound.data()[p * 2 + 0], bound.data()[p * 2 + 1]};
    return poly;
}

Vec3 read_row3(const F64_2& a, std::size_t r) {
    return {a.data()[r * 3 + 0], a.data()[r * 3 + 1], a.data()[r * 3 + 2]};
}

// ---- hbond ----
// Each donor-H / acceptor pair carries its own resolved geometry, so a synthetic atom
// array of 2*n entries (H at 2p, A at 2p+1) supplies the coordinates and the H/A blocks
// the driver reads. Nothing else about the atoms is consulted by computeHbondCPU.
nb::dict compute_hbond(F64_2 hp_H, F64_2 hp_A, F64_2 hp_D, F64_2 hp_B, F64_2 hp_B0,
                       I32_1 hp_block_h, I32_1 hp_block_a, I32_1 hp_hyb,
                       F64_1 hp_ad_weight, I32_1 hp_sep, F64_2 ahdist_coeffs,
                       F64_2 ahdist_range, F64_2 ahdist_bound, F64_2 cosbah_coeffs,
                       F64_2 cosbah_range, F64_2 cosbah_bound, F64_2 cosahd_coeffs,
                       F64_2 cosahd_range, F64_2 cosahd_bound, F64_1 hbond_global,
                       int n_blocks, int n_threads, bool use_cuda,
                       bool use_metal) {
    require_cuda(use_cuda);
    const std::size_t n = hp_H.shape(0);
    std::vector<AtomInput> atoms(2 * n);
    std::vector<HBondPairInput> hps(n);
    for (std::size_t p = 0; p < n; ++p) {
        const Vec3 H = read_row3(hp_H, p);
        const Vec3 A = read_row3(hp_A, p);
        atoms[2 * p].x = H[0];
        atoms[2 * p].y = H[1];
        atoms[2 * p].z = H[2];
        atoms[2 * p].block = hp_block_h.data()[p];
        atoms[2 * p + 1].x = A[0];
        atoms[2 * p + 1].y = A[1];
        atoms[2 * p + 1].z = A[2];
        atoms[2 * p + 1].block = hp_block_a.data()[p];

        HBondPairInput& hp = hps[p];
        hp.h = static_cast<int>(2 * p);
        hp.a = static_cast<int>(2 * p + 1);
        hp.D = read_row3(hp_D, p);
        hp.B = read_row3(hp_B, p);
        hp.B0 = read_row3(hp_B0, p);
        hp.sep = hp_sep.data()[p];
        hp.pair.hyb = hp_hyb.data()[p];
        hp.pair.ad_weight = hp_ad_weight.data()[p];
        hp.pair.AHdist = read_poly(ahdist_coeffs, ahdist_range, ahdist_bound, p);
        hp.pair.cosBAH = read_poly(cosbah_coeffs, cosbah_range, cosbah_bound, p);
        hp.pair.cosAHD = read_poly(cosahd_coeffs, cosahd_range, cosahd_bound, p);
    }
    EnergyParams params;
    params.hbondGlobal = read_hbond_global(hbond_global);
#ifdef FRUSTRAMOL_TMOL_CUDA
    if (use_cuda)
        return result_to_dict(computeHbondCUDA(atoms, hps, params, n_blocks));
#endif
    if (use_metal) {
#ifdef FRUSTRAMOL_TMOL_METAL
        return result_to_dict(computeHbondMetal(atoms, hps, params, n_blocks));
#else
        no_metal_build();
#endif
    }
    return result_to_dict(
        computeHbondCPU(atoms, hps, params, n_blocks, n_threads));
}

}  // namespace

NB_MODULE(_kernels, m) {
    m.doc() = "Torch-free CPU energy kernels (ljlk, fa_elec, lk_ball, hbond) ported from "
              "the Apache-2.0 tmol-webgpu math. Single-point forward only.";
    m.def("compute_pair_energies", &compute_pair_energies, nb::arg("coords"),
          nb::arg("block"), nb::arg("charge"), nb::arg("is_heavy"),
          nb::arg("lj_radius"), nb::arg("lj_wdepth"), nb::arg("lk_dgfree"),
          nb::arg("lk_lambda"), nb::arg("lk_volume"), nb::arg("is_donor"),
          nb::arg("is_hydroxyl"), nb::arg("is_polarh"), nb::arg("is_acceptor"),
          nb::arg("pair_i"), nb::arg("pair_j"), nb::arg("sep_ljlk"),
          nb::arg("sep_elec"), nb::arg("ljlk_global"), nb::arg("elec_global"),
          nb::arg("n_blocks"), nb::arg("n_threads") = 0, nb::arg("use_cuda") = false,
          nb::arg("use_metal") = false,
          "fa_ljatr / fa_ljrep / fa_lk / fa_elec per-residue-pair matrices. "
          "use_cuda=True routes to the CUDA forward path and use_metal=True to the Metal "
          "path (each requires the matching GPU build).");
    m.def("compute_lk_ball", &compute_lk_ball, nb::arg("coords"), nb::arg("block"),
          nb::arg("charge"), nb::arg("is_heavy"), nb::arg("lj_radius"),
          nb::arg("lj_wdepth"), nb::arg("lk_dgfree"), nb::arg("lk_lambda"),
          nb::arg("lk_volume"), nb::arg("is_donor"), nb::arg("is_hydroxyl"),
          nb::arg("is_polarh"), nb::arg("is_acceptor"), nb::arg("waters"),
          nb::arg("water_present"), nb::arg("pair_i"), nb::arg("pair_j"),
          nb::arg("sep_ljlk"), nb::arg("sep_elec"), nb::arg("ljlk_global"),
          nb::arg("lk_ball_global"), nb::arg("n_blocks"), nb::arg("n_threads") = 0,
          nb::arg("use_cuda") = false, nb::arg("use_metal") = false,
          "lk_ball_iso / lk_ball / lk_bridge / lk_bridge_uncpl per-residue-pair "
          "matrices. use_cuda=True routes to the CUDA path and use_metal=True to the "
          "Metal path (each requires the matching GPU build).");
    m.def("compute_hbond", &compute_hbond, nb::arg("hp_H"), nb::arg("hp_A"),
          nb::arg("hp_D"), nb::arg("hp_B"), nb::arg("hp_B0"), nb::arg("hp_block_h"),
          nb::arg("hp_block_a"), nb::arg("hp_hyb"), nb::arg("hp_ad_weight"),
          nb::arg("hp_sep"), nb::arg("ahdist_coeffs"), nb::arg("ahdist_range"),
          nb::arg("ahdist_bound"), nb::arg("cosbah_coeffs"), nb::arg("cosbah_range"),
          nb::arg("cosbah_bound"), nb::arg("cosahd_coeffs"), nb::arg("cosahd_range"),
          nb::arg("cosahd_bound"), nb::arg("hbond_global"), nb::arg("n_blocks"),
          nb::arg("n_threads") = 0, nb::arg("use_cuda") = false,
          nb::arg("use_metal") = false,
          "hbond per-residue-pair matrix. use_cuda=True routes to the CUDA path and "
          "use_metal=True to the Metal path (each requires the matching GPU build).");

    m.def("has_cuda", &hasCuda,
          "True iff the optional CUDA forward path was compiled "
          "(FRUSTRAMOL_TMOL_CUDA). False in the default CPU-only build.");

    m.def("has_metal", &tmol_has_metal,
          "True iff this build has the Metal (Apple GPU) path AND a Metal device is "
          "present. False in the CPU-only / non-Apple build, where use_metal=True raises.");

    m.def(
        "has_openmp",
        []() {
#ifdef _OPENMP
            return true;
#else
            return false;
#endif
        },
        "True iff the CPU kernels were compiled with OpenMP (multicore available).");

    m.def("effective_threads", &effectiveThreads, nb::arg("n_threads"),
          "CPU threads a run will use for the given request (0 -> all cores, 1 -> serial). "
          "The caller passes its shared-budget inner count (cores // outer_procs) so a nested "
          "process+thread fan-out never oversubscribes.");
}
