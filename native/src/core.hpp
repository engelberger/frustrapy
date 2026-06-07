// Native frustration core (CPU reference) -- declarations.
//
// This header declares the memory-safe C++ reductions that the native
// FrustrationBackend uses in place of the AWSEM/LAMMPS subprocess. N1 ships the
// declarations and one real geometry kernel (contact_map); the energy reductions
// (compute_frustration) are stubs until N2. See docs/NATIVE_BACKEND_DESIGN.md.
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

// Structure data as parallel non-owning views (structure of arrays). All spans have
// length n_res and are indexed by residue. Coordinates are flat [n_res * 3] in
// row-major (x, y, z) order; cb_* uses CA for glycine, matching the AWSEM convention.
struct StructureView {
    std::span<const double> ca;        // length 3 * n_res
    std::span<const double> cb;        // length 3 * n_res
    std::span<const std::int32_t> res_type;   // 0..19, gamma.dat ordering
    std::span<const std::int32_t> chain_id;
    std::span<const std::int32_t> res_seqid;
    std::size_t n_res = 0;
};

// AWSEM parameters as non-owning views into the gamma/burial tables. gamma_* are
// flattened 20x20 row-major; burial_gamma is 20x3 row-major.
struct ParamsView {
    std::span<const double> gamma_direct;    // 400
    std::span<const double> gamma_water;     // 400
    std::span<const double> gamma_protein;   // 400
    std::span<const double> burial_gamma;    // 60
    double direct_rmin = 4.5, direct_rmax = 6.5;
    double mediated_rmin = 6.5, mediated_rmax = 9.5;
    double well_eta = 1.0;
    double burial_kappa = 4.0;
    double contact_cutoff = 9.5;
    int seq_dist = 12;
    int n_decoys = 1000;
    std::uint64_t seed = 1;
};

// Per-contact reduction output (structure of arrays), one entry per contact passing the
// distance/seq-separation filter. Column meanings mirror tertiary_frustration.dat.
struct ContactResult {
    std::vector<std::int32_t> res_i, res_j;
    std::vector<double> native_energy, decoy_energy, sd_energy, frst_index;
    std::vector<double> rho_i, rho_j, r_ij;
};

// Real geometry kernel: indices of CB-CB contacts within `cutoff` Angstrom and with
// residue-sequence separation >= seq_dist (same-chain) or any (cross-chain). Returns
// flat pairs [i0, j0, i1, j1, ...]. Used to validate the SoA + binding path end to end
// before any energy parity is claimed.
std::vector<std::int32_t> contact_map(const StructureView& s, double cutoff, int seq_dist);

// Smoothed local density per residue: rho_i = sum_{k != i} theta(r_ik). Real, used by
// the mediated/burial terms in N2; exposed now to test the reduction path.
std::vector<double> local_density(const StructureView& s, double rmin, double rmax);

// The energy/decoy reduction. N1 stub -- throws std::runtime_error("...not implemented
// (N2)"). N2 fills this to reproduce tertiary_frustration.dat per the design doc.
ContactResult compute_frustration(const StructureView& s, const ParamsView& p,
                                  const std::string& mode);

// True iff the optional CUDA path was compiled in (FRUSTRAPY_NATIVE_CUDA).
bool has_cuda() noexcept;

}  // namespace frustrapy_native
