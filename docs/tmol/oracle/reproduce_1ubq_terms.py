"""Whole-pose ref2015-term scoring probe on 1ubq, replicating tmol's
EnergyTermTestBase.test_whole_pose_scoring_10 path. Prints per-subterm pose0
energies and compares to the shipped term_baselines. Read-only oracle reproduction."""
import os, sys, numpy, torch, yaml, time

torch.manual_seed(0)
device = torch.device("cpu")

from tmol.io import pose_stack_from_pdb
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.database import ParameterDatabase

from tmol.score.ljlk.ljlk_energy_term import LJLKEnergyTerm
from tmol.score.lk_ball.lk_ball_energy_term import LKBallEnergyTerm
from tmol.score.elec.elec_energy_term import ElecEnergyTerm
from tmol.score.hbond.hbond_energy_term import HBondEnergyTerm
from tmol.score.ref.ref_energy_term import RefEnergyTerm

PDB = sys.argv[1]
BASELINE_DIR = sys.argv[2]
with open(PDB) as fh:
    pdb_content = fh.read()

db = ParameterDatabase.get_default()

def build_scorer(term_cls, pose_stack):
    et = term_cls(param_db=db, device=device)
    for bt in pose_stack.packed_block_types.active_block_types:
        et.setup_block_type(bt)
    et.setup_packed_block_types(pose_stack.packed_block_types)
    et.setup_poses(pose_stack)
    return et.render_whole_pose_scoring_module(pose_stack)

def load_baseline(name):
    with open(os.path.join(BASELINE_DIR, name, "test_whole_pose_scoring_10.yaml")) as fh:
        d = yaml.safe_load(fh)
    return numpy.array([[d[t][p] for p in d[t]] for t in d])

t0 = time.time()
p1 = pose_stack_from_pdb(pdb_content, device)
pn = PoseStackBuilder.from_poses([p1] * 10, device=device)
print("pose built (10x1ubq, %d res) in %.1fs" % (pn.coords.shape[1], time.time() - t0))

TERMS = [
    ("LJLKEnergyTerm", LJLKEnergyTerm),
    ("LKBallEnergyTerm", LKBallEnergyTerm),
    ("ElecEnergyTerm", ElecEnergyTerm),
    ("HBondEnergyTerm", HBondEnergyTerm),
    ("RefEnergyTerm", RefEnergyTerm),
]

results = {}
for name, cls in TERMS:
    ts = time.time()
    scorer = build_scorer(cls, pn)
    coords = torch.nn.Parameter(pn.coords.clone())
    scores = scorer(coords).cpu().detach().numpy()  # (n_subterms, n_poses)
    base = load_baseline(name)
    measured0 = scores[:, 0]
    base0 = base[:, 0]
    try:
        numpy.testing.assert_allclose(base0, measured0, atol=1e-5, rtol=1e-3)
        ok = "PASS"
    except AssertionError:
        ok = "FAIL"
    maxabs = float(numpy.max(numpy.abs(base0 - measured0)))
    print("[%s] %s  %.2fs  measured_pose0=%s  baseline_pose0=%s  max|d|=%.3e"
          % (ok, name, time.time() - ts,
             numpy.array2string(measured0, precision=6),
             numpy.array2string(base0, precision=6), maxabs))
    results[name] = {"measured_pose0": [float(x) for x in measured0],
                     "baseline_pose0": [float(x) for x in base0],
                     "max_abs_diff": maxabs, "status": ok}

print("TOTAL %.1fs" % (time.time() - t0))
import json
with open(sys.argv[3], "w") as fh:
    json.dump(results, fh, indent=2)
print("wrote", sys.argv[3])
