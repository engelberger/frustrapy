// Standalone CPU-vs-tmol parity harness for the torch-free energy kernels.
//
// Loads a tmol-webgpu fixture (the per-structure energy ORACLE exported from tmol; see
// tmol-webgpu tools/export_fixture.py) and checks the C++ CPU drivers reproduce tmol's
// per-residue-pair block-pair matrices and whole-pose totals to the tmol gate tolerance
// (atol 1e-3, rtol 1e-4), per subterm. It also reports single-thread vs multi-thread
// agreement (the fp-reduction-order stability check). No torch, no nanobind: builds with
// g++ -fopenmp alone, mirroring the offline oracle discipline of the tmol-webgpu CPU
// reference (src/cpu.ts, src/cpu_terms.ts, test/parity.test.ts).
//
// Usage: parity_harness <fixture.json> [more.json ...]
// Exit code 0 if every subterm on every fixture is within tolerance, 1 otherwise.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "frustramol_tmol/driver.hpp"
#include "mini_json.hpp"

using namespace frustramol_tmol;

namespace {

constexpr double ATOL = 1e-3;
constexpr double RTOL = 1e-4;

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
    const minijson::Value* refBlockPair = nullptr;  // reference.block_pair
    const minijson::Value* refWholePose = nullptr;   // reference.whole_pose
    const minijson::Value* refWholePoseTrue = nullptr;
};

// Largest absolute difference between a flattened block-pair matrix and the nb x nb
// reference (mirrors tmol-webgpu maxBlockPairDiff).
double maxBlockPairDiff(const std::vector<double>& got, const minijson::Value& ref,
                        int nb) {
    double maxd = 0.0;
    for (int i = 0; i < nb; ++i) {
        const auto& row = ref.at(static_cast<std::size_t>(i)).array();
        for (int j = 0; j < nb; ++j) {
            double rv = row.at(static_cast<std::size_t>(j)).num();
            double gv = got[static_cast<std::size_t>(i) * nb + j];
            double dd = std::fabs(rv - gv);
            if (dd > maxd) maxd = dd;
        }
    }
    return maxd;
}

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
        atom.ljlkType = static_cast<int>(k);  // per-atom one-row "type" (folded params)
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

// One subterm check: per-pair block-pair max diff and whole-pose closeness. `wholeRef`
// selects whether the whole-pose oracle is whole_pose (block-pair sum) or whole_pose_true.
bool checkSubterm(const std::string& fixName, const Fixture& fx,
                  const PairEnergyResult& res, const std::string& term,
                  bool checkPerPair, const minijson::Value& wholeRef,
                  double& worstPerPair) {
    bool ok = true;
    if (checkPerPair) {
        double d = maxBlockPairDiff(res.mat(term), (*fx.refBlockPair)[term],
                                    fx.nBlocks);
        worstPerPair = std::max(worstPerPair, d);
        if (d > ATOL) {
            std::printf("  FAIL %-10s %-16s per-pair max block diff %.3e > %.0e\n",
                        fixName.c_str(), term.c_str(), d, ATOL);
            ok = false;
        }
    }
    double got = res.pose(term);
    double ref = wholeRef[term].num();
    if (!close(got, ref)) {
        std::printf("  FAIL %-10s %-16s whole-pose %.6f vs tmol %.6f (d %.3e)\n",
                    fixName.c_str(), term.c_str(), got, ref, std::fabs(got - ref));
        ok = false;
    }
    return ok;
}

// Largest per-cell difference between two block-pair results (single vs multi thread).
double maxResultDiff(const PairEnergyResult& a, const PairEnergyResult& b) {
    double m = 0.0;
    for (std::size_t t = 0; t < a.blockPair.size(); ++t)
        for (std::size_t k = 0; k < a.blockPair[t].size(); ++k)
            m = std::max(m, std::fabs(a.blockPair[t][k] - b.blockPair[t][k]));
    return m;
}

bool runFixture(const std::string& path, double& worstPerPairOut,
                double& worstThreadDiffOut) {
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

    // Single-thread (n_threads=1) and multi-thread (n_threads=0 => all cores).
    PairEnergyResult ljlk1 = computePairEnergiesCPU(fx.atoms, fx.pairs, fx.params,
                                                    fx.nBlocks, 1);
    PairEnergyResult ljlkN = computePairEnergiesCPU(fx.atoms, fx.pairs, fx.params,
                                                    fx.nBlocks, 0);
    PairEnergyResult lkb1 =
        computeLkBallCPU(fx.atoms, fx.pairs, fx.params, fx.nBlocks, 1);
    PairEnergyResult lkbN =
        computeLkBallCPU(fx.atoms, fx.pairs, fx.params, fx.nBlocks, 0);
    PairEnergyResult hb1 =
        computeHbondCPU(fx.atoms, fx.hbondPairs, fx.params, fx.nBlocks, 1);
    PairEnergyResult hbN =
        computeHbondCPU(fx.atoms, fx.hbondPairs, fx.params, fx.nBlocks, 0);

    double threadDiff = std::max({maxResultDiff(ljlk1, ljlkN),
                                  maxResultDiff(lkb1, lkbN),
                                  maxResultDiff(hb1, hbN)});
    worstThreadDiffOut = std::max(worstThreadDiffOut, threadDiff);

    bool ok = true;
    double worst = 0.0;
    // ljlk subterms: per-pair + whole-pose (block-pair sum).
    for (const auto& t : {std::string("fa_ljatr"), std::string("fa_ljrep"),
                          std::string("fa_lk")})
        ok &= checkSubterm(fx.name, fx, ljlk1, t, true, *fx.refWholePose, worst);
    // fa_elec: whole-pose vs tmol canonical whole_pose_true (block-pair differs by a
    // documented count-pair / intra-residue bookkeeping term).
    if (fx.refWholePoseTrue) {
        ok &= checkSubterm(fx.name, fx, ljlk1, "fa_elec", false,
                           *fx.refWholePoseTrue, worst);
    } else {
        ok &= checkSubterm(fx.name, fx, ljlk1, "fa_elec", false, *fx.refWholePose,
                           worst);
    }
    // lk_ball subterms: per-pair + whole-pose.
    for (const auto& t : lkBallSubterms())
        ok &= checkSubterm(fx.name, fx, lkb1, t, true, *fx.refWholePose, worst);
    // hbond: per-pair + whole-pose.
    ok &= checkSubterm(fx.name, fx, hb1, "hbond", true, *fx.refWholePose, worst);

    worstPerPairOut = std::max(worstPerPairOut, worst);
    std::printf("  %s  worst per-pair block diff %.3e   single-vs-multi max diff %.3e\n",
                ok ? "PASS" : "FAIL", worst, threadDiff);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <fixture.json> [more.json ...]\n", argv[0]);
        return 2;
    }
    bool allOk = true;
    double worstPerPair = 0.0, worstThreadDiff = 0.0;
    for (int i = 1; i < argc; ++i)
        allOk &= runFixture(argv[i], worstPerPair, worstThreadDiff);

    std::printf(
        "\n%s  worst per-pair block diff %.3e (tol %.0e)   worst single-vs-multi "
        "%.3e\n",
        allOk ? "ALL PASS" : "FAILURES", worstPerPair, ATOL, worstThreadDiff);
    return allOk ? 0 : 1;
}
