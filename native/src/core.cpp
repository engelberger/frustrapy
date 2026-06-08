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
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace frustrapy_native {

namespace {

constexpr double kLn10x8 = 8.0 * 2.302585;  // theta clamp half-width factor * kappa

// Work-aware parallelism thresholds. The first OpenMP region in a process pays a
// one-time thread-team startup (~tens of ms, measured); the per-unit reductions in
// the light paths (configurational native energy is O(1) per contact; density is
// O(n^2) but only a few ms below ~500 residues) are cheaper than that overhead. So
// a region only goes parallel when its estimated work clearly exceeds the overhead.
// Calibrated empirically (native/docs/PROFILE.md): below these the serial path wins.
constexpr long long kDensityMinPairs = 250000;   // ~n >= 500 residues
constexpr long long kEnergyMinOps = 5000000;      // ~a few ms of reduction work

// Optional per-phase wall-clock timing to stderr, gated on FRUSTRAPY_NATIVE_PROFILE.
// Used by native/docs/PROFILE.md to attribute time to density / RNG / energy.
bool profile_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("FRUSTRAPY_NATIVE_PROFILE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

class PhaseTimer {
public:
    explicit PhaseTimer(const char* name) : name_(name) {
        if (profile_enabled()) t0_ = std::chrono::steady_clock::now();
    }
    ~PhaseTimer() {
        if (!profile_enabled()) return;
        const auto dt = std::chrono::steady_clock::now() - t0_;
        const double ms = std::chrono::duration<double, std::milli>(dt).count();
        std::fprintf(stderr, "[native-profile] %-22s %10.3f ms\n", name_, ms);
    }

private:
    const char* name_;
    std::chrono::steady_clock::time_point t0_{};
};

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
    // precomputed_rho (when non-empty) is reused verbatim instead of recomputing the
    // density: it is geometry-only and invariant under a mutation, so a scan computes
    // it once and reuses it across variants. compute_density() is parallel over i with
    // no cross-i reduction, so its result is identical for any thread count -- a rho
    // produced by prepare_geometry (any thread count) is bit-for-bit what this would
    // compute, hence reuse changes no numbers.
    Engine(const StructureView& s, const ParamsView& p, int threads,
           std::span<const double> precomputed_rho = {})
        : s_(s), p_(p), threads_(threads) {
        if (!precomputed_rho.empty()) {
            if (precomputed_rho.size() != s_.n_res) {
                throw std::invalid_argument("precomputed rho length must be n_res");
            }
            rho_.assign(precomputed_rho.begin(), precomputed_rho.end());
        } else {
            compute_density();
        }
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
    // Parallel over i: each rho_[i] is an independent fixed-order sum, so the result is
    // identical for any thread count (no shared accumulator, no reduction across i).
    void compute_density() {
        PhaseTimer t("density");
        rho_.assign(s_.n_res, 0.0);
        const long long n = static_cast<long long>(s_.n_res);
        const bool par = threads_ > 1 && n * n >= kDensityMinPairs;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_) schedule(static) if (par)
#endif
        for (long long ii = 0; ii < n; ++ii) {
            const std::size_t i = static_cast<std::size_t>(ii);
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
    int threads_ = 1;
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
    const long long n = static_cast<long long>(s.n_res);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long ii = 0; ii < n; ++ii) {
        const std::size_t i = static_cast<std::size_t>(ii);
        double acc = 0.0;
        for (std::size_t k = 0; k < s.n_res; ++k) {
            if (k == i) continue;
            acc += theta_clamped(dist(s.coord, i, k), rmin, rmax, 5.0);
        }
        rho[i] = acc;
    }
    return rho;
}

GeometryCache prepare_geometry(const StructureView& s, const ParamsView& p) {
    check_lengths(s);
    const int threads = effective_threads(p.n_threads);
    Engine eng(s, p, threads);  // computes the density once
    GeometryCache cache;
    cache.rho = eng.rho();
    // Build the energy contact list with the exact predicate and (i<j) order the
    // contact-mode reduction uses, so it can be fed straight back via PreparedGeometry.
    for (std::size_t i = 0; i < s.n_res; ++i)
        for (std::size_t j = i + 1; j < s.n_res; ++j)
            if (eng.r(i, j) < p.contact_cutoff && eng.separated_contact(i, j)) {
                cache.contacts.push_back(static_cast<std::int32_t>(i));
                cache.contacts.push_back(static_cast<std::int32_t>(j));
            }
    return cache;
}

FrustrationResult compute_frustration(const StructureView& s, const ParamsView& p,
                                      const std::string& mode,
                                      const PreparedGeometry* precomp) {
    check_lengths(s);
    const bool is_config = mode == "configurational";
    const bool is_mut = mode == "mutational";
    const bool is_single = mode == "singleresidue";
    if (!is_config && !is_mut && !is_single) {
        throw std::invalid_argument("mode must be configurational, mutational, or "
                                    "singleresidue");
    }

    // The GPU paths recompute the density and contacts on device from the same
    // coordinates, so precomp is a clean no-op there (numbers unchanged, no reuse).
#ifdef FRUSTRAPY_NATIVE_CUDA
    if (p.prefer_cuda) return compute_frustration_cuda(s, p, mode);
#endif

#ifdef FRUSTRAPY_NATIVE_METAL
    if (p.prefer_metal) return compute_frustration_metal(s, p, mode);
#endif

    const int threads = effective_threads(p.n_threads);
    const std::span<const double> pre_rho =
        (precomp != nullptr && precomp->has_rho) ? precomp->rho : std::span<const double>{};
    Engine eng(s, p, threads, pre_rho);
    const bool reuse_contacts = precomp != nullptr && precomp->has_contacts;
    const std::size_t n = s.n_res;
    const int nd = p.n_decoys;

    FrustrationResult out;
    out.rho = eng.rho();

    // The decoy ensembles consume a single shared glibc RNG stream whose state
    // advances in the reference main-loop order. To parallelize without perturbing
    // that stream, the random draws are materialized serially (cheap, O(units*nd))
    // in exactly the reference order, then the expensive per-unit energy reductions
    // run in parallel writing to disjoint output slots. The per-unit decoy array is
    // summed in fixed order (mean_of/std_pop), so the result is bit-identical to the
    // serial path for any thread count.
    if (is_single) {
        // Serial RNG draw: random substituted identity per (residue, decoy).
        std::vector<std::int32_t> dec_it(n * static_cast<std::size_t>(nd));
        {
            PhaseTimer t("rng-precompute");
            GlibcRand rng(static_cast<std::uint32_t>(p.seed));
            for (std::size_t i = 0; i < n; ++i)
                for (int d = 0; d < nd; ++d)
                    dec_it[i * static_cast<std::size_t>(nd) + static_cast<std::size_t>(d)] =
                        eng.rtype(rng.residue_index(n));
        }

        out.unit_i.resize(n);
        out.unit_j.assign(n, -1);
        out.native_energy.resize(n);
        out.decoy_energy.resize(n);
        out.sd_energy.resize(n);
        out.frst_index.resize(n);

        PhaseTimer t("energy");
        const long long N = static_cast<long long>(n);
        // native_single is O(n); total ~ n * nd * n.
        const bool par = threads > 1 && N * nd * N >= kEnergyMinOps;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static) if (par)
#endif
        for (long long ii = 0; ii < N; ++ii) {
            const std::size_t i = static_cast<std::size_t>(ii);
            std::vector<double> decoys(static_cast<std::size_t>(nd));
            const double native = eng.native_single(i, eng.rtype(i));
            for (int d = 0; d < nd; ++d)
                decoys[static_cast<std::size_t>(d)] = eng.native_single(
                    i, dec_it[i * static_cast<std::size_t>(nd) + static_cast<std::size_t>(d)]);
            const double m = mean_of(decoys);
            const double sd = std_pop(decoys);
            out.unit_i[i] = static_cast<std::int32_t>(i);
            out.native_energy[i] = native;
            out.decoy_energy[i] = m;
            out.sd_energy[i] = sd;
            out.frst_index[i] = (m - native) / sd;
        }
        return out;
    }

    // Contact modes: build the contact list in reference (i, j) main-loop order so
    // the work-unit order and any RNG consumption match the serial reference. When a
    // precomputed list is supplied (built by prepare_geometry with the same predicate
    // and order) it is unpacked instead, skipping the O(n^2) rebuild.
    std::vector<std::int32_t> ci, cj;
    if (reuse_contacts) {
        const auto& c = precomp->contacts;
        if (c.size() % 2 != 0) {
            throw std::invalid_argument("precomputed contacts must be flat (i, j) pairs");
        }
        ci.reserve(c.size() / 2);
        cj.reserve(c.size() / 2);
        for (std::size_t k = 0; k + 1 < c.size(); k += 2) {
            ci.push_back(c[k]);
            cj.push_back(c[k + 1]);
        }
    } else {
        PhaseTimer t("contact-list");
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j) {
                if (eng.r(i, j) < p.contact_cutoff && eng.separated_contact(i, j)) {
                    ci.push_back(static_cast<std::int32_t>(i));
                    cj.push_back(static_cast<std::int32_t>(j));
                }
            }
    }
    const std::size_t nc = ci.size();

    out.unit_i = ci;
    out.unit_j = cj;
    out.native_energy.resize(nc);
    out.decoy_energy.resize(nc);
    out.sd_energy.resize(nc);
    out.frst_index.resize(nc);

    if (is_config) {
        // One shared decoy ensemble for all contacts (the reference computes it once,
        // using the RNG; no RNG is consumed before this). Build it serially.
        double config_mean = 0.0, config_sd = 0.0;
        if (nc > 0) {
            PhaseTimer t("rng-precompute");
            GlibcRand rng(static_cast<std::uint32_t>(p.seed));
            std::vector<double> decoys(static_cast<std::size_t>(nd));
            for (int d = 0; d < nd; ++d) {
                std::size_t ri = rng.residue_index(n);
                std::size_t rj = rng.residue_index(n);
                double rd = eng.r(ri, rj);
                while (rd > p.contact_cutoff || ri == rj) {
                    ri = rng.residue_index(n);
                    rj = rng.residue_index(n);
                    rd = eng.r(ri, rj);
                }
                const std::size_t bi = rng.residue_index(n);
                const std::size_t bj = rng.residue_index(n);
                const double rho_i = eng.rho()[bi];
                const double rho_j = eng.rho()[bj];
                const int it = eng.rtype(rng.residue_index(n));
                const int jt = eng.rtype(rng.residue_index(n));
                decoys[static_cast<std::size_t>(d)] =
                    eng.water_energy(rd, it, jt, rho_i, rho_j) +
                    eng.burial_energy(it, rho_i) + eng.burial_energy(jt, rho_j);
            }
            config_mean = mean_of(decoys);
            config_sd = std_pop(decoys);
        }

        PhaseTimer t("energy");
        const long long C = static_cast<long long>(nc);
        // native_config is O(1) per contact; the whole loop is sub-millisecond even
        // for thousands of contacts, so it stays serial (thread overhead > work).
        const bool par = threads > 1 && C >= kEnergyMinOps;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static) if (par)
#endif
        for (long long cc = 0; cc < C; ++cc) {
            const std::size_t c = static_cast<std::size_t>(cc);
            const double native = eng.native_config(static_cast<std::size_t>(ci[c]),
                                                     static_cast<std::size_t>(cj[c]));
            out.native_energy[c] = native;
            out.decoy_energy[c] = config_mean;
            out.sd_energy[c] = config_sd;
            out.frst_index[c] = (config_mean - native) / config_sd;
        }
        return out;
    }

    // mutational: per contact, randomize i,j identities at native geometry/density,
    // including the (i,k),(j,k) terms with native k identities. Two RNG draws per
    // decoy (it, jt), consumed in contact order -> materialize serially, then the
    // O(nc * nd * n) energy reduction runs in parallel.
    std::vector<std::int32_t> dec_it(nc * static_cast<std::size_t>(nd));
    std::vector<std::int32_t> dec_jt(nc * static_cast<std::size_t>(nd));
    {
        PhaseTimer t("rng-precompute");
        GlibcRand rng(static_cast<std::uint32_t>(p.seed));
        for (std::size_t c = 0; c < nc; ++c)
            for (int d = 0; d < nd; ++d) {
                const std::size_t idx = c * static_cast<std::size_t>(nd) + static_cast<std::size_t>(d);
                dec_it[idx] = eng.rtype(rng.residue_index(n));
                dec_jt[idx] = eng.rtype(rng.residue_index(n));
            }
    }

    PhaseTimer t("energy");
    const long long C = static_cast<long long>(nc);
    // native_mut + each decoy is O(n); total ~ nc * nd * n -- the dominant cost.
    const bool par = threads > 1 && C * nd * static_cast<long long>(n) >= kEnergyMinOps;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static) if (par)
#endif
    for (long long cc = 0; cc < C; ++cc) {
        const std::size_t c = static_cast<std::size_t>(cc);
        const std::size_t i = static_cast<std::size_t>(ci[c]);
        const std::size_t j = static_cast<std::size_t>(cj[c]);
        const double rij = eng.r(i, j);
        const double native = eng.native_mut(i, j);
        std::vector<double> decoys(static_cast<std::size_t>(nd));
        for (int d = 0; d < nd; ++d) {
            const std::size_t idx = c * static_cast<std::size_t>(nd) + static_cast<std::size_t>(d);
            const int it = dec_it[idx];
            const int jt = dec_jt[idx];
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
                we + eng.burial_energy(it, eng.rho()[i]) + eng.burial_energy(jt, eng.rho()[j]);
        }
        const double m = mean_of(decoys);
        const double sd = std_pop(decoys);
        out.native_energy[c] = native;
        out.decoy_energy[c] = m;
        out.sd_energy[c] = sd;
        out.frst_index[c] = (m - native) / sd;
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

bool has_openmp() noexcept {
#ifdef _OPENMP
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

int effective_threads(int n_threads) noexcept {
#ifdef _OPENMP
    if (n_threads > 0) return n_threads;
    const int procs = omp_get_num_procs();
    return procs > 0 ? procs : 1;
#else
    (void)n_threads;
    return 1;
#endif
}

}  // namespace frustrapy_native
