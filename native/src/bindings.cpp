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

// Build a StructureView from input arrays. The spans reference the caller's arrays,
// which stay alive for the duration of the call.
StructureView make_structure(const Coords& ca, const Coords& cb, const ResVec& res_type,
                             const ResVec& chain_id, const ResVec& res_seqid) {
    const std::size_t n = ca.shape(0);
    if (cb.shape(0) != n || res_type.shape(0) != n || chain_id.shape(0) != n ||
        res_seqid.shape(0) != n) {
        throw std::invalid_argument("all per-residue arrays must share length n_res");
    }
    StructureView s;
    s.n_res = n;
    s.ca = std::span<const double>(ca.data(), 3 * n);
    s.cb = std::span<const double>(cb.data(), 3 * n);
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

nb::object py_contact_map(Coords ca, Coords cb, ResVec res_type, ResVec chain_id,
                          ResVec res_seqid, double cutoff, int seq_dist) {
    StructureView s = make_structure(ca, cb, res_type, chain_id, res_seqid);
    std::vector<std::int32_t> pairs = contact_map(s, cutoff, seq_dist);
    const std::size_t n_pairs = pairs.size() / 2;
    return nb::cast(own(std::move(pairs), {n_pairs, std::size_t{2}}));
}

nb::object py_local_density(Coords ca, Coords cb, ResVec res_type, ResVec chain_id,
                            ResVec res_seqid, double rmin, double rmax) {
    StructureView s = make_structure(ca, cb, res_type, chain_id, res_seqid);
    std::vector<double> rho = local_density(s, rmin, rmax);
    const std::size_t n = rho.size();
    return nb::cast(own(std::move(rho), {n}));
}

// N1 stub surface for the energy reduction: validates inputs and the parameter shapes,
// then raises (the C++ core throws "not implemented (N2)"). The signature is the final
// one so N2 only fills the body.
nb::object py_compute_frustration(Coords ca, Coords cb, ResVec res_type, ResVec chain_id,
                                  ResVec res_seqid, ParamMat gamma_direct,
                                  ParamMat gamma_water, ParamMat gamma_protein,
                                  ParamMat burial_gamma, const std::string& mode,
                                  int seq_dist, int n_decoys, std::uint64_t seed) {
    StructureView s = make_structure(ca, cb, res_type, chain_id, res_seqid);
    auto check = [](const ParamMat& m, std::size_t r, std::size_t c, const char* what) {
        if (m.shape(0) != r || m.shape(1) != c)
            throw std::invalid_argument(std::string(what) + " must be " +
                                        std::to_string(r) + "x" + std::to_string(c));
    };
    check(gamma_direct, 20, 20, "gamma_direct");
    check(gamma_water, 20, 20, "gamma_water");
    check(gamma_protein, 20, 20, "gamma_protein");
    check(burial_gamma, 20, 3, "burial_gamma");

    ParamsView p;
    p.gamma_direct = std::span<const double>(gamma_direct.data(), 400);
    p.gamma_water = std::span<const double>(gamma_water.data(), 400);
    p.gamma_protein = std::span<const double>(gamma_protein.data(), 400);
    p.burial_gamma = std::span<const double>(burial_gamma.data(), 60);
    p.seq_dist = seq_dist;
    p.n_decoys = n_decoys;
    p.seed = seed;

    ContactResult r = compute_frustration(s, p, mode);  // throws in N1
    (void)r;
    return nb::none();
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "Native frustration core (CPU reference; CUDA optional). See "
              "docs/NATIVE_BACKEND_DESIGN.md.";
    m.attr("__core_version__") = "0.0.1-n1";

    m.def("has_cuda", &has_cuda,
          "True iff the optional CUDA path was compiled (FRUSTRAPY_NATIVE_CUDA).");

    m.def("contact_map", &py_contact_map, nb::arg("ca"), nb::arg("cb"),
          nb::arg("res_type"), nb::arg("chain_id"), nb::arg("res_seqid"),
          nb::arg("cutoff"), nb::arg("seq_dist"),
          "CB-CB contact pairs within cutoff and seq-separation; returns int32[n,2].");

    m.def("local_density", &py_local_density, nb::arg("ca"), nb::arg("cb"),
          nb::arg("res_type"), nb::arg("chain_id"), nb::arg("res_seqid"),
          nb::arg("rmin"), nb::arg("rmax"),
          "Smoothed per-residue local density; returns float64[n].");

    m.def("compute_frustration", &py_compute_frustration, nb::arg("ca"), nb::arg("cb"),
          nb::arg("res_type"), nb::arg("chain_id"), nb::arg("res_seqid"),
          nb::arg("gamma_direct"), nb::arg("gamma_water"), nb::arg("gamma_protein"),
          nb::arg("burial_gamma"), nb::arg("mode"), nb::arg("seq_dist"),
          nb::arg("n_decoys"), nb::arg("seed"),
          "Energy/decoy reduction (N1 stub: raises until N2 lands the reductions).");
}
