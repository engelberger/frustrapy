// Native frustration core (CPU reference) -- implementation.
//
// Reproduces the AWSEM/LAMMPS tertiary-frustration computation (native energy,
// per-mode decoy ensemble, Z-score) bit-for-bit. The energy model, density, and
// decoy generation follow adavtyan/awsemmd src/fix_backbone.cpp +
// smart_matrix_lib.h. Decoys use the C library rand() with the default seed (1) --
// the reference binary never calls srand(), so its decoy sequence is deterministic;
// GlibcRand reproduces glibc's TYPE_3 generator exactly. No owning raw pointers;
// indexing is through std::span.

#include "core.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace frustrapy_native {

namespace {

constexpr double kLn10x8 = 8.0 * 2.302585;  // theta clamp half-width factor * kappa

double dist(std::span<const double> coord, std::size_t a, std::size_t b) {
    const double dx = coord[3 * a + 0] - coord[3 * b + 0];
    const double dy = coord[3 * a + 1] - coord[3 * b + 1];
    const double dz = coord[3 * a + 2] - coord[3 * b + 2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void check_lengths(const StructureView& s) {
    const std::size_t n = s.n_res;
    if (s.coord.size() != 3 * n) {
        throw std::invalid_argument("coordinate span length must be 3 * n_res");
    }
    if (s.res_type.size() != n || s.chain_id.size() != n || s.res_seqid.size() != n) {
        throw std::invalid_argument("per-residue span length must be n_res");
    }
}

// glibc TYPE_3 additive-feedback generator (random()/rand()), default seed 1.
// Reproduces the exact sequence the reference binary consumes for its decoys.
class GlibcRand {
public:
    explicit GlibcRand(std::uint32_t seed) {
        if (seed == 0) seed = 1;  // glibc maps seed 0 to 1
        st_[0] = static_cast<std::int32_t>(seed);
        for (int i = 1; i < kDeg; ++i) {
            // Park-Miller minimal standard, overflow-safe (glibc srandom_r).
            const std::int64_t hi = st_[i - 1] / 127773;
            const std::int64_t lo = st_[i - 1] % 127773;
            std::int64_t word = 16807 * lo - 2836 * hi;
            if (word < 0) word += 2147483647;
            st_[i] = static_cast<std::int32_t>(word);
        }
        fptr_ = kSep;
        rptr_ = 0;
        for (int i = 0; i < 10 * kDeg; ++i) next();  // warm up
    }

    std::int32_t next() {
        const std::uint32_t sum = static_cast<std::uint32_t>(st_[fptr_]) +
                                  static_cast<std::uint32_t>(st_[rptr_]);
        st_[fptr_] = static_cast<std::int32_t>(sum);
        const std::int32_t result = static_cast<std::int32_t>((sum >> 1) & 0x7fffffffU);
        if (++fptr_ >= kDeg) fptr_ = 0;
        if (++rptr_ >= kDeg) rptr_ = 0;
        return result;
    }

    // get_random_residue_index(): rand() % n.
    std::size_t residue_index(std::size_t n) {
        return static_cast<std::size_t>(next()) % n;
    }

private:
    static constexpr int kDeg = 31;
    static constexpr int kSep = 3;
    std::array<std::int32_t, kDeg> st_{};
    int fptr_ = 0;
    int rptr_ = 0;
};

// Radial well theta(r) on [r_min, r_max] (AWSEM compute_water_energy form).
double theta(double r, double r_min, double r_max, double kappa) {
    const double t_min = std::tanh(kappa * (r - r_min));
    const double t_max = std::tanh(kappa * (r_max - r));
    return 0.25 * (1.0 + t_min) * (1.0 + t_max);
}

// theta with the cWell clamp to 0 outside [r_min - 8ln10/kappa, r_max + 8ln10/kappa]
// (smart_matrix_lib.h compute_theta). Used for the density only.
double theta_clamped(double r, double r_min, double r_max, double kappa) {
    const double half = kLn10x8 / kappa;
    if (r < r_min - half || r > r_max + half) return 0.0;
    return theta(r, r_min, r_max, kappa);
}

// Engine carrying the structure, parameters, and per-residue density; provides the
// AWSEM energy terms and per-mode decoy ensembles.
class Engine {
public:
    Engine(const StructureView& s, const ParamsView& p) : s_(s), p_(p) {
        compute_density();
    }

    const std::vector<double>& rho() const { return rho_; }

    // water_energy(rij, it, jt, rho_i, rho_j): direct + density-switched mediated.
    double water_energy(double rij, int it, int jt, double rho_i, double rho_j) const {
        const double sigma_wat =
            0.25 * (1.0 - std::tanh(p_.kappa_sigma * (rho_i - p_.treshold))) *
            (1.0 - std::tanh(p_.kappa_sigma * (rho_j - p_.treshold)));
        const double sigma_prot = 1.0 - sigma_wat;
        const std::size_t g = static_cast<std::size_t>(it) * 20 + static_cast<std::size_t>(jt);
        const double sigma_gamma_direct = p_.gamma_direct[g];
        const double sigma_gamma_mediated =
            sigma_prot * p_.gamma_protein[g] + sigma_wat * p_.gamma_water[g];
        return -(sigma_gamma_direct * theta(rij, p_.well_r_min[0], p_.well_r_max[0], p_.well_kappa) +
                 sigma_gamma_mediated * theta(rij, p_.well_r_min[1], p_.well_r_max[1], p_.well_kappa));
    }

    double burial_energy(int it, double rho_i) const {
        double e = 0.0;
        for (int k = 0; k < 3; ++k) {
            const double t0 = std::tanh(p_.burial_kappa * (rho_i - p_.burial_ro_min[k]));
            const double t1 = std::tanh(p_.burial_kappa * (p_.burial_ro_max[k] - rho_i));
            e += -0.5 * p_.k_burial * p_.burial_gamma[static_cast<std::size_t>(it) * 3 + k] *
                 (t0 + t1);
        }
        return e;
    }

    double r(std::size_t a, std::size_t b) const { return dist(s_.coord, a, b); }
    int rtype(std::size_t i) const { return s_.res_type[i]; }

    // configurational native: only the (i,j) contact contributes.
    double native_config(std::size_t i, std::size_t j) const {
        return water_energy(r(i, j), rtype(i), rtype(j), rho_[i], rho_[j]) +
               burial_energy(rtype(i), rho_[i]) + burial_energy(rtype(j), rho_[j]);
    }

    // mutational native: (i,j) plus all (i,k),(j,k) within the distance cutoff
    // (k keeps its native identity; no sequence-separation filter on k).
    double native_mut(std::size_t i, std::size_t j) const {
        double we = water_energy(r(i, j), rtype(i), rtype(j), rho_[i], rho_[j]);
        for (std::size_t k = 0; k < s_.n_res; ++k) {
            if (k == i || k == j) continue;
            const double rik = r(i, k);
            if (rik < p_.contact_cutoff)
                we += water_energy(rik, rtype(i), rtype(k), rho_[i], rho_[k]);
            const double rjk = r(j, k);
            if (rjk < p_.contact_cutoff)
                we += water_energy(rjk, rtype(j), rtype(k), rho_[j], rho_[k]);
        }
        return we + burial_energy(rtype(i), rho_[i]) + burial_energy(rtype(j), rho_[j]);
    }

    // singleresidue native with a (possibly substituted) identity it at site i.
    double native_single(std::size_t i, int it) const {
        double e = burial_energy(it, rho_[i]);
        for (std::size_t j = 0; j < s_.n_res; ++j) {
            if (j == i) continue;
            const double rij = r(i, j);
            if (rij < p_.contact_cutoff && separated_contact(i, j))
                e += water_energy(rij, it, rtype(j), rho_[i], rho_[j]);
        }
        return e;
    }

    // contact-list / single-residue separation: |i-j| >= contact_min_sep or cross-chain.
    bool separated_contact(std::size_t i, std::size_t j) const {
        if (s_.chain_id[i] != s_.chain_id[j]) return true;
        return std::abs(s_.res_seqid[i] - s_.res_seqid[j]) >= p_.contact_min_sep;
    }

    const StructureView& structure() const { return s_; }
    const ParamsView& params() const { return p_; }

private:
    // rho_i = sum_{j: |res_no diff| > seq_dist or cross-chain} theta_clamped(r_ij, well0).
    void compute_density() {
        rho_.assign(s_.n_res, 0.0);
        for (std::size_t i = 0; i < s_.n_res; ++i) {
            double acc = 0.0;
            for (std::size_t j = 0; j < s_.n_res; ++j) {
                if (j == i) continue;
                const bool sep = s_.chain_id[i] != s_.chain_id[j] ||
                                 std::abs(s_.res_seqid[i] - s_.res_seqid[j]) > p_.seq_dist;
                if (sep)
                    acc += theta_clamped(r(i, j), p_.well_r_min[0], p_.well_r_max[0], p_.well_kappa);
            }
            rho_[i] = acc;
        }
    }

    const StructureView& s_;
    const ParamsView& p_;
    std::vector<double> rho_;
};

double mean_of(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m += x;
    return m / static_cast<double>(v.size());
}

double std_pop(const std::vector<double>& v) {
    const double m = mean_of(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size()));
}

}  // namespace

std::vector<std::int32_t> contact_map(const StructureView& s, double cutoff,
                                      int seq_dist) {
    check_lengths(s);
    std::vector<std::int32_t> pairs;
    for (std::size_t i = 0; i < s.n_res; ++i) {
        for (std::size_t j = i + 1; j < s.n_res; ++j) {
            const bool sep = s.chain_id[i] != s.chain_id[j] ||
                             std::abs(s.res_seqid[i] - s.res_seqid[j]) >= seq_dist;
            if (sep && dist(s.coord, i, j) <= cutoff) {
                pairs.push_back(static_cast<std::int32_t>(i));
                pairs.push_back(static_cast<std::int32_t>(j));
            }
        }
    }
    return pairs;
}

std::vector<double> local_density(const StructureView& s, double rmin, double rmax) {
    check_lengths(s);
    std::vector<double> rho(s.n_res, 0.0);
    for (std::size_t i = 0; i < s.n_res; ++i) {
        double acc = 0.0;
        for (std::size_t k = 0; k < s.n_res; ++k) {
            if (k == i) continue;
            acc += theta_clamped(dist(s.coord, i, k), rmin, rmax, 5.0);
        }
        rho[i] = acc;
    }
    return rho;
}

FrustrationResult compute_frustration(const StructureView& s, const ParamsView& p,
                                      const std::string& mode) {
    check_lengths(s);
    const bool is_config = mode == "configurational";
    const bool is_mut = mode == "mutational";
    const bool is_single = mode == "singleresidue";
    if (!is_config && !is_mut && !is_single) {
        throw std::invalid_argument("mode must be configurational, mutational, or "
                                    "singleresidue");
    }

#ifdef FRUSTRAPY_NATIVE_CUDA
    if (p.prefer_cuda) return compute_frustration_cuda(s, p, mode);
#endif

#ifdef FRUSTRAPY_NATIVE_METAL
    if (p.prefer_metal) return compute_frustration_metal(s, p, mode);
#endif

    Engine eng(s, p);
    const std::size_t n = s.n_res;
    GlibcRand rng(static_cast<std::uint32_t>(p.seed));
    const int nd = p.n_decoys;

    FrustrationResult out;
    out.rho = eng.rho();

    if (is_single) {
        std::vector<double> decoys(static_cast<std::size_t>(nd));
        for (std::size_t i = 0; i < n; ++i) {
            const double native = eng.native_single(i, eng.rtype(i));
            for (int d = 0; d < nd; ++d) {
                const int it = eng.rtype(rng.residue_index(n));  // random identity at i
                decoys[static_cast<std::size_t>(d)] = eng.native_single(i, it);
            }
            const double m = mean_of(decoys);
            const double sd = std_pop(decoys);
            out.unit_i.push_back(static_cast<std::int32_t>(i));
            out.unit_j.push_back(-1);
            out.native_energy.push_back(native);
            out.decoy_energy.push_back(m);
            out.sd_energy.push_back(sd);
            out.frst_index.push_back((m - native) / sd);
        }
        return out;
    }

    // Contact modes: iterate residue pairs in the reference main-loop order so the
    // shared RNG state advances identically.
    std::vector<double> decoys(static_cast<std::size_t>(nd));
    bool config_decoys_ready = false;
    double config_mean = 0.0, config_sd = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double rij = eng.r(i, j);
            const bool contact = rij < p.contact_cutoff &&
                                 (eng.separated_contact(i, j));
            if (!contact) continue;

            const double native = is_config ? eng.native_config(i, j) : eng.native_mut(i, j);

            double m, sd;
            if (is_config) {
                if (!config_decoys_ready) {
                    for (int d = 0; d < nd; ++d) {
                        // pick a random in-contact pair for the distance
                        std::size_t ri = rng.residue_index(n);
                        std::size_t rj = rng.residue_index(n);
                        double rd = eng.r(ri, rj);
                        while (rd > p.contact_cutoff || ri == rj) {
                            ri = rng.residue_index(n);
                            rj = rng.residue_index(n);
                            rd = eng.r(ri, rj);
                        }
                        // a new random pair for the densities
                        const std::size_t bi = rng.residue_index(n);
                        const std::size_t bj = rng.residue_index(n);
                        const double rho_i = eng.rho()[bi];
                        const double rho_j = eng.rho()[bj];
                        // a new random pair for the identities
                        const int it = eng.rtype(rng.residue_index(n));
                        const int jt = eng.rtype(rng.residue_index(n));
                        decoys[static_cast<std::size_t>(d)] =
                            eng.water_energy(rd, it, jt, rho_i, rho_j) +
                            eng.burial_energy(it, rho_i) + eng.burial_energy(jt, rho_j);
                    }
                    config_mean = mean_of(decoys);
                    config_sd = std_pop(decoys);
                    config_decoys_ready = true;
                }
                m = config_mean;
                sd = config_sd;
            } else {
                // mutational: randomize i,j identities at native geometry/density,
                // including the (i,k),(j,k) terms with native k identities.
                for (int d = 0; d < nd; ++d) {
                    const int it = eng.rtype(rng.residue_index(n));
                    const int jt = eng.rtype(rng.residue_index(n));
                    double we = eng.water_energy(rij, it, jt, eng.rho()[i], eng.rho()[j]);
                    for (std::size_t k = 0; k < n; ++k) {
                        if (k == i || k == j) continue;
                        const double rik = eng.r(i, k);
                        if (rik < p.contact_cutoff)
                            we += eng.water_energy(rik, it, eng.rtype(k), eng.rho()[i], eng.rho()[k]);
                        const double rjk = eng.r(j, k);
                        if (rjk < p.contact_cutoff)
                            we += eng.water_energy(rjk, jt, eng.rtype(k), eng.rho()[j], eng.rho()[k]);
                    }
                    decoys[static_cast<std::size_t>(d)] =
                        we + eng.burial_energy(it, eng.rho()[i]) +
                        eng.burial_energy(jt, eng.rho()[j]);
                }
                m = mean_of(decoys);
                sd = std_pop(decoys);
            }
            out.unit_i.push_back(static_cast<std::int32_t>(i));
            out.unit_j.push_back(static_cast<std::int32_t>(j));
            out.native_energy.push_back(native);
            out.decoy_energy.push_back(m);
            out.sd_energy.push_back(sd);
            out.frst_index.push_back((m - native) / sd);
        }
    }
    return out;
}

bool has_cuda() noexcept {
#ifdef FRUSTRAPY_NATIVE_CUDA
    return true;
#else
    return false;
#endif
}

bool has_metal() noexcept {
#ifdef FRUSTRAPY_NATIVE_METAL
    return true;
#else
    return false;
#endif
}

}  // namespace frustrapy_native
