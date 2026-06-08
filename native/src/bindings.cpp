// nanobind bindings for the native frustration core.
//
// NumPy arrays cross the boundary zero-copy (nb::ndarray, buffer/DLPack protocol).
// Inputs are constrained to contiguous CPU arrays of fixed dtype; output arrays own
// C++-allocated memory through an nb::capsule deleter (the nanobind ownership rule:
// never return a view of freed storage). See docs/NATIVE_BACKEND_DESIGN.md.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "core.hpp"

namespace nb = nanobind;
using namespace frustrapy_native;

namespace {

using Coords = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using ResVec = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ParamMat =
    nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// Build a StructureView whose coordinate view is the AWSEM interaction coordinate
// (CB, or CA for glycine). The legacy two-coordinate kernels pass that array as
// `cb`; `ca` is accepted for signature compatibility but the interaction coordinate
// is always `cb`. The spans reference the caller's arrays, alive for the call.
StructureView make_structure(const Coords& cb, const ResVec& res_type,
                             const ResVec& chain_id, const ResVec& res_seqid) {
    const std::size_t n = cb.shape(0);
    if (res_type.shape(0) != n || chain_id.shape(0) != n || res_seqid.shape(0) != n) {
        throw std::invalid_argument("all per-residue arrays must share length n_res");
    }
    StructureView s;
    s.n_res = n;
    s.coord = std::span<const double>(cb.data(), 3 * n);
    s.res_type = std::span<const std::int32_t>(res_type.data(), n);
    s.chain_id = std::span<const std::int32_t>(chain_id.data(), n);
    s.res_seqid = std::span<const std::int32_t>(res_seqid.data(), n);
    return s;
}

// Move a std::vector into a NumPy array that owns it via a capsule deleter.
template <typename T>
nb::ndarray<nb::numpy, T> own(std::vector<T>&& v, std::initializer_list<std::size_t> shape) {
    auto* held = new std::vector<T>(std::move(v));
    nb::capsule owner(held, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::ndarray<nb::numpy, T>(held->data(), shape, owner);
}

template <typename T>
nb::ndarray<nb::numpy, T> own1d(std::vector<T>&& v) {
    const std::size_t n = v.size();
    return own(std::move(v), {n});
}

nb::object py_contact_map(Coords ca, Coords cb, ResVec res_type, ResVec chain_id,
                          ResVec res_seqid, double cutoff, int seq_dist) {
    (void)ca;
    StructureView s = make_structure(cb, res_type, chain_id, res_seqid);
    std::vector<std::int32_t> pairs = contact_map(s, cutoff, seq_dist);
    const std::size_t n_pairs = pairs.size() / 2;
    return nb::cast(own(std::move(pairs), {n_pairs, std::size_t{2}}));
}

nb::object py_local_density(Coords ca, Coords cb, ResVec res_type, ResVec chain_id,
                            ResVec res_seqid, double rmin, double rmax) {
    (void)ca;
    StructureView s = make_structure(cb, res_type, chain_id, res_seqid);
    std::vector<double> rho = local_density(s, rmin, rmax);
    return nb::cast(own1d(std::move(rho)));
}

void check_param_shapes(const ParamMat& gd, const ParamMat& gw, const ParamMat& gp,
                        const ParamMat& bg) {
    auto check = [](const ParamMat& m, std::size_t r, std::size_t c, const char* what) {
        if (m.shape(0) != r || m.shape(1) != c)
            throw std::invalid_argument(std::string(what) + " must be " +
                                        std::to_string(r) + "x" + std::to_string(c));
    };
    check(gd, 20, 20, "gamma_direct");
    check(gw, 20, 20, "gamma_water");
    check(gp, 20, 20, "gamma_protein");
    check(bg, 20, 3, "burial_gamma");
}

// Energy/decoy reduction. Returns a dict of NumPy arrays (rho per residue; unit_i,
// unit_j, native_energy, decoy_energy, sd_energy, frst_index per probed unit). The
// well/burial constants come from the coefficient file via the caller.
nb::object py_compute_frustration(
    Coords coord, ResVec res_type, ResVec chain_id, ResVec res_seqid,
    ParamMat gamma_direct, ParamMat gamma_water, ParamMat gamma_protein,
    ParamMat burial_gamma, const std::string& mode,
    double well_kappa, double kappa_sigma, double treshold,
    double well_r_min0, double well_r_max0, double well_r_min1, double well_r_max1,
    double burial_kappa, double k_burial, double contact_cutoff,
    int contact_min_sep, int seq_dist, int n_decoys, std::uint64_t seed,
    bool use_cuda, int n_threads) {
    check_param_shapes(gamma_direct, gamma_water, gamma_protein, burial_gamma);
    StructureView s = make_structure(coord, res_type, chain_id, res_seqid);

    ParamsView p;
    p.gamma_direct = std::span<const double>(gamma_direct.data(), 400);
    p.gamma_water = std::span<const double>(gamma_water.data(), 400);
    p.gamma_protein = std::span<const double>(gamma_protein.data(), 400);
    p.burial_gamma = std::span<const double>(burial_gamma.data(), 60);
    p.well_kappa = well_kappa;
    p.kappa_sigma = kappa_sigma;
    p.treshold = treshold;
    p.well_r_min[0] = well_r_min0;
    p.well_r_max[0] = well_r_max0;
    p.well_r_min[1] = well_r_min1;
    p.well_r_max[1] = well_r_max1;
    p.burial_kappa = burial_kappa;
    p.k_burial = k_burial;
    p.contact_cutoff = contact_cutoff;
    p.contact_min_sep = contact_min_sep;
    p.seq_dist = seq_dist;
    p.n_decoys = n_decoys;
    p.seed = seed;
    p.prefer_cuda = use_cuda;
    p.n_threads = n_threads;
    if (use_cuda && !has_cuda()) {
        throw std::runtime_error(
            "use_cuda=True but the native core was built without CUDA. Rebuild with "
            "`pip install ./native -C cmake.define.FRUSTRAPY_NATIVE_CUDA=ON`.");
    }

    // The reduction is pure C++ (no Python objects touched); release the GIL so the
    // OpenMP worker threads run unhindered and other Python threads can proceed.
    FrustrationResult r;
    {
        nb::gil_scoped_release release;
        r = compute_frustration(s, p, mode);
    }
    nb::dict out;
    out["rho"] = own1d(std::move(r.rho));
    out["unit_i"] = own1d(std::move(r.unit_i));
    out["unit_j"] = own1d(std::move(r.unit_j));
    out["native_energy"] = own1d(std::move(r.native_energy));
    out["decoy_energy"] = own1d(std::move(r.decoy_energy));
    out["sd_energy"] = own1d(std::move(r.sd_energy));
    out["frst_index"] = own1d(std::move(r.frst_index));
    return out;
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "Native frustration core (CPU reference; CUDA optional). See "
              "docs/NATIVE_BACKEND_DESIGN.md.";
    m.attr("__core_version__") = "0.1.0-n2";

    m.def("has_cuda", &has_cuda,
          "True iff the optional CUDA path was compiled (FRUSTRAPY_NATIVE_CUDA).");

    m.def("contact_map", &py_contact_map, nb::arg("ca"), nb::arg("cb"),
          nb::arg("res_type"), nb::arg("chain_id"), nb::arg("res_seqid"),
          nb::arg("cutoff"), nb::arg("seq_dist"),
          "Contact pairs within cutoff and seq-separation; returns int32[n,2].");

    m.def("local_density", &py_local_density, nb::arg("ca"), nb::arg("cb"),
          nb::arg("res_type"), nb::arg("chain_id"), nb::arg("res_seqid"),
          nb::arg("rmin"), nb::arg("rmax"),
          "Smoothed per-residue local density; returns float64[n].");

    m.def("compute_frustration", &py_compute_frustration, nb::arg("coord"),
          nb::arg("res_type"), nb::arg("chain_id"), nb::arg("res_seqid"),
          nb::arg("gamma_direct"), nb::arg("gamma_water"), nb::arg("gamma_protein"),
          nb::arg("burial_gamma"), nb::arg("mode"), nb::arg("well_kappa") = 5.0,
          nb::arg("kappa_sigma") = 7.0, nb::arg("treshold") = 2.6,
          nb::arg("well_r_min0") = 4.5, nb::arg("well_r_max0") = 6.5,
          nb::arg("well_r_min1") = 6.5, nb::arg("well_r_max1") = 9.5,
          nb::arg("burial_kappa") = 4.0, nb::arg("k_burial") = 1.0,
          nb::arg("contact_cutoff") = 9.5, nb::arg("contact_min_sep") = 2,
          nb::arg("seq_dist") = 12, nb::arg("n_decoys") = 1000, nb::arg("seed") = 1,
          nb::arg("use_cuda") = false, nb::arg("n_threads") = 0,
          "AWSEM tertiary-frustration reduction (native energy, decoy mean/sd, "
          "index) for the given mode; returns a dict of NumPy arrays. n_threads "
          "controls CPU parallelism (0 = all cores, 1 = serial); the result is "
          "bit-identical for any thread count. Set use_cuda=True to use the GPU "
          "path (requires a CUDA build).");

    m.def("has_openmp", &has_openmp,
          "True iff the CPU core was compiled with OpenMP (multicore available).");

    m.def("effective_threads", &effective_threads, nb::arg("n_threads"),
          "CPU threads a reduction will use for the given request (0 -> all cores).");
}
