# tmol shared oracle artifact (mission #37)

Reference per-term ref2015/`beta_nov2016` energies reproduced FROM tmol on `1ubq.pdb`, in
this container, on CPU. This is the shared oracle the gate doc promised: one fixture set, two
consumers (the Python provider #38/#39 and the WebGPU port #40), no per-branch drift.

## Files
- `1ubq_term_energies.json` - measured vs shipped-baseline per-subterm pose0 energies for the
  five in-scope terms, with `max_abs_diff` and PASS/FAIL at tolerance atol=1e-5, rtol=1e-3.
- `reproduce_1ubq_terms.py` - the standalone script that produced it. Replicates tmol's
  `EnergyTermTestBase.test_whole_pose_scoring_10` path: load 1ubq into a PoseStack, stack 10
  copies, build each term's `render_whole_pose_scoring_module`, evaluate, compare to the
  shipped `tmol/tests/data/term_baselines/<Term>/test_whole_pose_scoring_10.yaml`.

## Result (2026-06-08, in-container, CPU)

All five terms reproduce the shipped baselines within tmol's own test tolerance:

| Term | subterms | measured pose0 | baseline pose0 | max abs diff | status |
|---|---|---|---|---|---|
| LJLKEnergyTerm | fa_ljatr, fa_ljrep, fa_lk | -417.95831, 240.71466, 298.27652 | identical | 0.0 | PASS |
| LKBallEnergyTerm | lk_ball_iso, lk_ball, lk_bridge, lk_bridge_uncpl | 422.03961, 172.19641, 1.57859, 10.99460 | 422.03961, 172.19644, 1.57859, 10.99460 | 3.05e-05 | PASS |
| ElecEnergyTerm | fa_elec | -136.29248 | -136.29248 | 4.69e-07 | PASS |
| HBondEnergyTerm | hbond | -55.67561 | -55.67562 | 1.14e-05 | PASS |
| RefEnergyTerm | ref | -41.27500 | -41.27500 | 0.0 | PASS |

LJLK and ref reproduce bit-for-bit; the rest differ only in the last float32 digits and pass
on rtol. `[VERIFIED EMPIRICALLY]`

## Exact reproduce commands

Build a clean Python 3.12 venv, install CPU torch and the tmol cp312 CPU wheel, then run the
script in JIT mode (see the CPU-runnability caveat in `../ENERGY_AUDIT.md` section 4 - the AOT
wheel ships without the `_compiled_inverse_kin` pybind module, so a C++ toolchain + JIT is
required in this container):

```
uv venv /tmp/tmolprobe --python 3.12
source /tmp/tmolprobe/bin/activate
uv pip install "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cpu
curl -sL -o /tmp/tmol.whl \
  "https://github.com/uw-ipd/tmol/releases/download/v0.1.14/tmol-0.1.14%2Bcpu-cp312-cp312-linux_x86_64.whl"
cp /tmp/tmol.whl "/tmp/tmol-0.1.14+cpu-cp312-cp312-linux_x86_64.whl"
uv pip install "/tmp/tmol-0.1.14+cpu-cp312-cp312-linux_x86_64.whl" ninja

TMOL_USE_JIT=1 python reproduce_1ubq_terms.py \
  /workspace/tmol_src/tmol/tests/data/pdb/1ubq.pdb \
  /workspace/tmol_src/tmol/tests/data/term_baselines \
  /tmp/out.json
```

Or, with a working/complete tmol install, run the shipped tests directly:

```
pytest tmol/tests/score/ljlk/test_ljlk_energy_term.py::TestLJLKEnergyTerm::test_whole_pose_scoring_10 -v
```

## Provenance / version note

- Source clone read for the map: engelberger/tmol `f4d0916`, version 0.1.12.
- Wheel run for these numbers: the official `uw-ipd/tmol` v0.1.14 `+cpu-cp312` wheel (the
  closest published cp312 CPU wheel; 0.1.13 and 0.1.14 are cp312, 0.1.29+ are cp314). torch
  2.12.0+cpu. The 0.1.12-vs-0.1.14 skew is immaterial for these five terms: the measured
  values match the 0.1.12 clone's shipped baselines. `[VERIFIED EMPIRICALLY]`
