// Standalone CUDA-vs-CPU-vs-oracle parity harness for the torch-free energy kernels (M6).
//
// Built only when FRUSTRAMOL_TMOL_CUDA is enabled (a maintainer GPU host; there is no nvcc
// in the dev container). Loads a tmol-webgpu fixture (the per-structure energy ORACLE; the
// same fixtures the CPU harness parity_main.cpp consumes) and, for every subterm:
//
//   1. runs the CPU driver (computePairEnergiesCPU / computeLkBallCPU / computeHbondCPU),
//   2. runs the CUDA path (computePairEnergiesCUDA / computeLkBallCUDA / computeHbondCUDA),
//   3. asserts CUDA agrees with the CPU driver (the parity-gated reference) to a tight
//      floating-point-reduction tolerance, and
//   4. asserts CUDA agrees with the tmol oracle to the gate tolerance (atol 1e-3).
//
// Exit code 0 if every subterm on every fixture passes both checks, 1 otherwise. No torch.
// This mirrors the CPU harness and the AWSEM native CUDA validation lane; record the
// numbers it prints (no GPU result is committed to the repo). See native_tmol/NOTICE.
//
// Usage: tmol_cuda_parity_harness <fixture.json> [more.json ...]

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "frustramol_tmol/cuda.hpp"
#include "frustramol_tmol/driver.hpp"
#include "mini_json.hpp"

using namespace frustramol_tmol;

namespace {

constexpr double ATOL = 1e-3;       // tmol oracle gate tolerance
constexpr double RTOL = 1e-4;
constexpr double CPU_CUDA_TOL = 1e-9;  // CPU vs CUDA: atomicAdd reduction-order only

bool close(double got, double ref) {
    return std::fabs(got - ref) <= ATOL + RTOL * std::fabs(ref);
}

Vec3 asVec(const minijson::Value& v) {
    return {v.at(0).num(), v.at(1).num(), v.at(2).num()};
}

HBondPoly asPoly(const minijson::Value& p) {
    HBondPoly poly{};
    const auto& c = p["coeffs"].array();
    for (int i = 0; i < 11; ++i) poly.coeffs[i] = c.at(i).num();
    poly.range = {p["range"].at(0).num(), p["range"].at(1).num()};
    poly.bound = {p["bound"].at(0).num(), p["bound"].at(1).num()};
    return poly;
}

struct Fixture {
    std::string name;
    int nBlocks = 0;
    std::vector<AtomInput> atoms;
    std::vector<PairInput> pairs;
    std::vector<HBondPairInput> hbondPairs;
    EnergyParams params;
    const minijson::Value* refBlockPair = nullptr;
    const minijson::Value* refWholePose = nullptr;
    const minijson::Value* refWholePoseTrue = nullptr;
};

// Identical fixture loader to parity_main.cpp (the harnesses consume the same fixtures).
Fixture loadFixture(const minijson::Value& raw) {
    Fixture fx;
    fx.name = raw["name"].str;
    fx.nBlocks = raw["n_blocks"].integer();

    const auto& g = raw["ljlk_global"].at(0).array();
    fx.params.ljlkGlobal = {g.at(0).num(), g.at(1).num(), g.at(2).num()};
    const auto& eg = raw["elec_global"].at(0).array();
    fx.params.elecGlobal = {eg.at(0).num(), eg.at(1).num(), eg.at(2).num(),
                            eg.at(3).num(), eg.at(4).num()};
    if (raw.has("lk_ball_global")) {
        const auto& lkg = raw["lk_ball_global"].array();
        fx.params.lkBallGlobal = {lkg.at(0).num(), lkg.at(1).num(), lkg.at(2).num(),
                                  lkg.at(3).num(), lkg.at(4).num()};
    }
    if (raw.has("hbond_global")) {
        const auto& hg = raw["hbond_global"].array();
        fx.params.hbondGlobal = {hg.at(0).num(), hg.at(1).num(), hg.at(2).num(),
                                 hg.at(3).num(), hg.at(4).num(), hg.at(5).num()};
    }

    const auto& atomArr = raw["atoms"].array();
    fx.atoms.reserve(atomArr.size());
    fx.params.ljlkTypeParams.reserve(atomArr.size());
    for (std::size_t k = 0; k < atomArr.size(); ++k) {
        const auto& a = atomArr[k];
        AtomInput atom;
        atom.x = a["x"].num();
        atom.y = a["y"].num();
        atom.z = a["z"].num();
        atom.block = a["block"].integer();
        atom.ljlkType = static_cast<int>(k);
        atom.charge = a["charge"].num();
        atom.isHeavy = a["is_heavy"].boolv();
        if (a.has("waters") && a["waters"].type == minijson::Value::Type::Array) {
            atom.hasWaters = true;
            const auto& ws = a["waters"].array();
            for (std::size_t w = 0; w < ws.size() && w < MAX_WATER; ++w) {
                if (ws[w].type == minijson::Value::Type::Array) {
                    atom.waters.present[w] = true;
                    atom.waters.pos[w] = asVec(ws[w]);
                }
            }
        }
        fx.atoms.push_back(atom);

        LjlkTypeParams tp;
        tp.lj_radius = a["lj_radius"].num();
        tp.lj_wdepth = a["lj_wdepth"].num();
        tp.lk_dgfree = a["lk_dgfree"].num();
        tp.lk_lambda = a["lk_lambda"].num();
        tp.lk_volume = a["lk_volume"].num();
        tp.is_donor = a["is_donor"].boolv();
        tp.is_hydroxyl = a["is_hydroxyl"].boolv();
        tp.is_polarh = a["is_polarh"].boolv();
        tp.is_acceptor = a["is_acceptor"].boolv();
        fx.params.ljlkTypeParams.push_back(tp);
    }

    const auto& np = raw["neighbor_pairs"].array();
    fx.pairs.reserve(np.size());
    for (const auto& p : np)
        fx.pairs.push_back({p["i"].integer(), p["j"].integer(),
                            p["sep_ljlk"].integer(), p["sep_elec"].integer()});

    if (raw.has("hbond_pairs")) {
        const auto& hps = raw["hbond_pairs"].array();
        fx.hbondPairs.reserve(hps.size());
        for (const auto& p : hps) {
            HBondPairInput hp;
            hp.h = p["h"].integer();
            hp.a = p["a"].integer();
            hp.D = asVec(p["D"]);
            hp.B = asVec(p["B"]);
            hp.B0 = asVec(p["B0"]);
            hp.sep = p["sep"].integer();
            hp.pair.hyb = p["hyb"].integer();
            hp.pair.ad_weight = p["ad_weight"].num();
            hp.pair.AHdist = asPoly(p["AHdist"]);
            hp.pair.cosBAH = asPoly(p["cosBAH"]);
            hp.pair.cosAHD = asPoly(p["cosAHD"]);
            fx.hbondPairs.push_back(hp);
        }
    }

    const auto& ref = raw["reference"];
    fx.refBlockPair = &ref["block_pair"];
    fx.refWholePose = &ref["whole_pose"];
    if (ref.has("whole_pose_true")) fx.refWholePoseTrue = &ref["whole_pose_true"];
    return fx;
}

double maxResultDiff(const PairEnergyResult& a, const PairEnergyResult& b) {
    double m = 0.0;
    for (std::size_t t = 0; t < a.blockPair.size(); ++t)
        for (std::size_t k = 0; k < a.blockPair[t].size(); ++k)
            m = std::max(m, std::fabs(a.blockPair[t][k] - b.blockPair[t][k]));
    return m;
}

double maxBlockPairDiff(const std::vector<double>& got, const minijson::Value& ref,
                        int nb) {
    double maxd = 0.0;
    for (int i = 0; i < nb; ++i) {
        const auto& row = ref.at(static_cast<std::size_t>(i)).array();
        for (int j = 0; j < nb; ++j) {
            double rv = row.at(static_cast<std::size_t>(j)).num();
            double gv = got[static_cast<std::size_t>(i) * nb + j];
            maxd = std::max(maxd, std::fabs(rv - gv));
        }
    }
    return maxd;
}

// CUDA-vs-oracle subterm check (per-pair block diff + whole-pose), mirroring the CPU
// harness's checkSubterm.
bool checkOracle(const std::string& fixName, const Fixture& fx,
                 const PairEnergyResult& res, const std::string& term,
                 bool checkPerPair, const minijson::Value& wholeRef, double& worst) {
    bool ok = true;
    if (checkPerPair) {
        double d = maxBlockPairDiff(res.mat(term), (*fx.refBlockPair)[term], fx.nBlocks);
        worst = std::max(worst, d);
        if (d > ATOL) {
            std::printf("  FAIL %-10s %-16s CUDA-vs-oracle per-pair %.3e > %.0e\n",
                        fixName.c_str(), term.c_str(), d, ATOL);
            ok = false;
        }
    }
    double got = res.pose(term);
    double ref = wholeRef[term].num();
    if (!close(got, ref)) {
        std::printf("  FAIL %-10s %-16s CUDA whole-pose %.6f vs tmol %.6f (d %.3e)\n",
                    fixName.c_str(), term.c_str(), got, ref, std::fabs(got - ref));
        ok = false;
    }
    return ok;
}

bool runFixture(const std::string& path, double& worstCpuCuda, double& worstOracle) {
    std::ifstream in(path);
    if (!in) {
        std::printf("  ERROR cannot open fixture: %s\n", path.c_str());
        return false;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    minijson::Value raw = minijson::parse(ss.str());
    Fixture fx = loadFixture(raw);

    std::printf("Fixture %-10s  n_blocks=%d  atoms=%zu  pairs=%zu  hbond_pairs=%zu\n",
                fx.name.c_str(), fx.nBlocks, fx.atoms.size(), fx.pairs.size(),
                fx.hbondPairs.size());

    // CPU reference (serial) and CUDA.
    PairEnergyResult ljlkCpu =
        computePairEnergiesCPU(fx.atoms, fx.pairs, fx.params, fx.nBlocks, 1);
    PairEnergyResult lkbCpu =
        computeLkBallCPU(fx.atoms, fx.pairs, fx.params, fx.nBlocks, 1);
    PairEnergyResult hbCpu =
        computeHbondCPU(fx.atoms, fx.hbondPairs, fx.params, fx.nBlocks, 1);
    PairEnergyResult ljlkGpu =
        computePairEnergiesCUDA(fx.atoms, fx.pairs, fx.params, fx.nBlocks);
    PairEnergyResult lkbGpu =
        computeLkBallCUDA(fx.atoms, fx.pairs, fx.params, fx.nBlocks);
    PairEnergyResult hbGpu =
        computeHbondCUDA(fx.atoms, fx.hbondPairs, fx.params, fx.nBlocks);

    bool ok = true;

    // CPU vs CUDA (the parity-gated reference; only fp-reduction order may differ).
    double cc = std::max({maxResultDiff(ljlkCpu, ljlkGpu), maxResultDiff(lkbCpu, lkbGpu),
                          maxResultDiff(hbCpu, hbGpu)});
    worstCpuCuda = std::max(worstCpuCuda, cc);
    if (cc > CPU_CUDA_TOL) {
        std::printf("  FAIL %-10s CUDA vs CPU max block diff %.3e > %.0e\n",
                    fx.name.c_str(), cc, CPU_CUDA_TOL);
        ok = false;
    }

    // CUDA vs tmol oracle.
    double worst = 0.0;
    for (const auto& t : {std::string("fa_ljatr"), std::string("fa_ljrep"),
                          std::string("fa_lk")})
        ok &= checkOracle(fx.name, fx, ljlkGpu, t, true, *fx.refWholePose, worst);
    if (fx.refWholePoseTrue)
        ok &= checkOracle(fx.name, fx, ljlkGpu, "fa_elec", false, *fx.refWholePoseTrue,
                          worst);
    else
        ok &= checkOracle(fx.name, fx, ljlkGpu, "fa_elec", false, *fx.refWholePose, worst);
    for (const auto& t : lkBallSubterms())
        ok &= checkOracle(fx.name, fx, lkbGpu, t, true, *fx.refWholePose, worst);
    ok &= checkOracle(fx.name, fx, hbGpu, "hbond", true, *fx.refWholePose, worst);
    worstOracle = std::max(worstOracle, worst);

    std::printf("  %s  CUDA-vs-CPU max diff %.3e   CUDA-vs-oracle worst per-pair %.3e\n",
                ok ? "PASS" : "FAIL", cc, worst);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <fixture.json> [more.json ...]\n", argv[0]);
        return 2;
    }
    if (!hasCuda()) {
        std::fprintf(stderr, "built without CUDA; nothing to validate\n");
        return 2;
    }
    bool allOk = true;
    double worstCpuCuda = 0.0, worstOracle = 0.0;
    for (int i = 1; i < argc; ++i)
        allOk &= runFixture(argv[i], worstCpuCuda, worstOracle);

    std::printf(
        "\n%s  worst CUDA-vs-CPU %.3e (tol %.0e)   worst CUDA-vs-oracle per-pair %.3e "
        "(tol %.0e)\n",
        allOk ? "ALL PASS" : "FAILURES", worstCpuCuda, CPU_CUDA_TOL, worstOracle, ATOL);
    return allOk ? 0 : 1;
}
