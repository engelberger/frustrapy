// Native frustration core (CPU reference) -- declarations.
//
// This header declares the memory-safe C++ reductions that the native
// FrustrationBackend uses in place of the AWSEM/LAMMPS subprocess. The energy
// model, density, per-mode decoy ensembles, and the index are reproduced from the
// AWSEM/LAMMPS reference (adavtyan/awsemmd src/fix_backbone.cpp +
// smart_matrix_lib.h) and are parity-gated bit-for-bit against the lammps backend.
// See docs/NATIVE_BACKEND_DESIGN.md.
//
// Conventions (C++ Core Guidelines): no owning raw pointers, no array-as-pointer
// interfaces (pass std::span), RAII, bounds-checkable. Inputs are non-owning views;
// outputs are owning std::vector.

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace frustrapy_native {

// Structure data as parallel non-owning views (structure of arrays). All spans
// have length n_res and are indexed by residue. `coord` is the AWSEM interaction
// coordinate per residue (CB, or CA for glycine), flat [3 * n_res] in row-major
// (x, y, z) order. res_type is 0..19 in gamma.dat ordering; chain_id and res_seqid
// (the residue number) drive the sequence-separation tests.
struct StructureView {
    std::span<const double> coord;             // length 3 * n_res
    std::span<const std::int32_t> res_type;    // 0..19, gamma.dat ordering
    std::span<const std::int32_t> chain_id;
    std::span<const std::int32_t> res_seqid;   // residue number
    std::size_t n_res = 0;
};

// AWSEM parameters as non-owning views into the gamma/burial tables plus the well
// scalars. gamma_* are flattened 20x20 row-major; burial_gamma is 20x3 row-major.
// gamma_direct already carries the (col0+col1)/2 average the model uses; the k_water
// scale is folded into all gamma tables on the Python side.
struct ParamsView {
    std::span<const double> gamma_direct;    // 400, = average of the two direct columns
    std::span<const double> gamma_water;     // 400, water-mediated
    std::span<const double> gamma_protein;   // 400, protein-mediated
    std::span<const double> burial_gamma;    // 60
    double well_kappa = 5.0;
    double kappa_sigma = 7.0;
    double treshold = 2.6;
    double well_r_min[2] = {4.5, 6.5};
    double well_r_max[2] = {6.5, 9.5};
    double k_burial = 1.0;
    double burial_kappa = 4.0;
    double burial_ro_min[3] = {0.0, 3.0, 6.0};
    double burial_ro_max[3] = {3.0, 6.0, 9.0};
    double contact_cutoff = 9.5;   // tert_frust_cutoff (distance)
    int contact_min_sep = 2;       // [Water] contact_cutoff: contact-list / single-res min |i-j|
    int seq_dist = 12;             // density sequence separation (strictly >): lmp_serial_{seq_dist}
    int n_decoys = 1000;
    std::uint64_t seed = 1;
    bool prefer_cuda = false;   // route to the CUDA path when compiled with it
    bool prefer_metal = false;  // route to the Metal path when compiled with it
};

// Reduction output (structure of arrays). For contact modes one entry per contact
// passing the filter; for singleresidue one entry per residue (unit_j = -1). The
// idx fields are 0-based residue indices into the StructureView arrays. rho holds
// the per-residue local density (length n_res). Column meanings mirror
// tertiary_frustration.dat.
struct FrustrationResult {
    std::vector<std::int32_t> unit_i, unit_j;
    std::vector<double> native_energy, decoy_energy, sd_energy, frst_index;
    std::vector<double> rho;   // per residue, length n_res
};

// Real geometry kernel: indices of contacts within `cutoff` Angstrom and with
// residue-sequence separation >= seq_dist (same-chain) or any (cross-chain).
// Returns flat pairs [i0, j0, i1, j1, ...]. Validates the SoA + binding path.
std::vector<std::int32_t> contact_map(const StructureView& s, double cutoff, int seq_dist);

// Smoothed local density per residue: rho_i = sum_{k != i} theta(r_ik) over the
// well [rmin, rmax] (clamped). Real, used by the energy reductions; exposed for
// the reduction-path test.
std::vector<double> local_density(const StructureView& s, double rmin, double rmax);

// The energy/decoy reduction: reproduces tertiary_frustration.dat (native energy,
// decoy mean/sd, index) for the given mode (configurational | mutational |
// singleresidue), bit-for-bit against the AWSEM/LAMMPS reference.
FrustrationResult compute_frustration(const StructureView& s, const ParamsView& p,
                                      const std::string& mode);

// True iff the optional CUDA path was compiled in (FRUSTRAPY_NATIVE_CUDA).
bool has_cuda() noexcept;

// True iff the optional Metal path was compiled in (FRUSTRAPY_NATIVE_METAL).
bool has_metal() noexcept;

#ifdef FRUSTRAPY_NATIVE_CUDA
// CUDA implementation of compute_frustration (kernels.cu). Offloads the per-residue
// density, the native energy, and the dominant decoy reductions to the GPU while
// generating the decoy random-index stream host-side with the exact glibc sequence,
// so the result matches the CPU core. Selected when ParamsView::prefer_cuda is set.
FrustrationResult compute_frustration_cuda(const StructureView& s, const ParamsView& p,
                                           const std::string& mode);
#endif

#ifdef FRUSTRAPY_NATIVE_METAL
// Metal (Apple GPU) implementation of compute_frustration (metal/dispatch.cpp +
// metal/kernels.metal). Mirrors the CUDA path term-for-term: the per-residue density,
// the native energy, and the dominant decoy reductions run as MSL compute kernels via
// metal-cpp; the decoy random-index stream and the configurational decoy ensemble are
// generated host-side with the exact glibc sequence, so the result tracks the CPU core.
// Selected when ParamsView::prefer_metal is set. NOTE: Apple GPUs are float32-only, so
// the on-device arithmetic is single precision where the CUDA path uses double -- see
// native/docs/METAL_BUILD.md for the expected tolerance impact and the precision
// fallbacks the maintainer can enable if parity fails on real hardware.
FrustrationResult compute_frustration_metal(const StructureView& s, const ParamsView& p,
                                            const std::string& mode);
#endif

}  // namespace frustrapy_native
