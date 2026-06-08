# tmol evaluator vs the shipped 1ubq oracle (scope item 1)

The #38 evaluator's pairwise score function (`frustrapy.backends.atomic_tmol_engine.build_pairwise_score_function`) scores tmol's own shipped 1ubq whole-pose; each in-scope subterm is diffed against the committed baseline `docs/tmol/oracle/1ubq_term_energies.json` (#37). This proves the frustrapy-side builder wraps tmol faithfully (adapt-and-wrap). `[VERIFIED EMPIRICALLY]`

- fixture: `/tmp/tmolprobe/lib/python3.12/site-packages/tmol/tests/data/pdb/1ubq.pdb`
- tolerance: atol=0.001, rtol=0.001 (tmol's own whole-pose test tolerance)
- worst absolute deviation: **3.052e-05**
- terms within tolerance: **9/9**

| subterm | measured pose0 | oracle baseline | abs diff | pass |
|---|---|---|---|---|
| fa_ljatr | -417.95831 | -417.95831 | 0.000e+00 | yes |
| fa_ljrep | 240.71466 | 240.71466 | 0.000e+00 | yes |
| fa_lk | 298.27652 | 298.27652 | 0.000e+00 | yes |
| lk_ball_iso | 422.03961 | 422.03961 | 0.000e+00 | yes |
| lk_ball | 172.19641 | 172.19644 | 3.052e-05 | yes |
| lk_bridge | 1.57859 | 1.57859 | 1.192e-07 | yes |
| lk_bridge_uncpl | 10.99460 | 10.99460 | 9.537e-07 | yes |
| fa_elec | -136.29248 | -136.29248 | 4.687e-07 | yes |
| hbond | -55.67561 | -55.67562 | 1.144e-05 | yes |

**Verdict: PASS** (every in-scope subterm reproduces tmol's baseline within tolerance).
