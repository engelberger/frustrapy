# tmol parameter sourcing and the license-tier loader seam (mission #36)

Author: Felipe Engelberger. Status: deliverable of TMOL-PARAM-SOURCING (#36), built on the
approved gate `docs/tmol/TMOL_LANE_DECISION.md` (TMOL-GATE #35).

This document implements the license-tier matrix from the gate (section 1), not a blanket
clean-room rewrite. The verified premise from the gate: tmol CODE is Apache-2.0 and reusable
directly; the numeric parameter tables in `tmol/database/default/scoring/` are vendored
Rosetta-database values; there is no NOTICE file in the tmol clone. The principle is reuse the
wheel where the license allows and reserve original work for the genuinely novel parts: REUSE the
params for the academic tier, a LOADER seam for the redistributable tier, and re-derive ONLY a
blocking value that has a clean public source (none in scope qualify; see section 4).

Every row below cites a `file:line` in the read-only tmol clone at `/workspace/tmol_src`
(engelberger/tmol master `f4d0916`). No tmol file is vendored into frustrapy by this mission; this
is a catalog and a contract. The machine-checkable form of the same data is
`docs/tmol/param_inventory.json`, and `tests/tmol/test_param_boundary.py` enforces the contract.

## 1. Scope: which terms, which params

The in-scope energy terms for inter-residue contact frustration are the ref2015 pairwise terms
the atomic backend needs: `ljlk` (fa_atr, fa_rep, fa_sol), `lk_ball`, `fa_elec`, `hbond`, plus the
`ref` per-residue reference weights. The full ref2015 set also includes `dunbrack`, `rama`,
`cartbonded`, `omega`, `disulfide`; those are not pairwise contact terms and are out of scope for
this catalog (they would matter only for a total-energy match, gate section 2).

One structural fact verified in the source: **`lk_ball` ships no parameter file of its own.** It
reuses the ljlk globals and atom-type table through `LJLKParamResolver`
(`tmol/score/lk_ball/lk_ball_energy_term.py:8,26-27,34`; the `lkb_water_*` globals it stacks at
`:42-55`). So `lk_ball` is covered entirely by the two `ljlk` groups below, not a separate row.

## 2. The catalog (13 groups, every row evidenced)

All paths are under `/workspace/tmol_src/`. The loader column is the attrs/cattr dataclass in
`tmol/database/scoring/*.py` that `cattr.structure` populates from the YAML; the structuring
entry point per file is its `*.from_file` classmethod.

| # | Group | Term(s) | File:lines | Loader dataclass | Shape | Provenance header |
|---|---|---|---|---|---|---|
| 1 | `ljlk.global_parameters` | fa_atr/rep/sol, lk_ball | `default/scoring/ljlk.yaml:1-20` | `scoring/ljlk.py:12` LJLKGlobalParameters | 14 scalar fields (incl 3 `lkb_water_tors_*` angle lists) | none |
| 2 | `ljlk.atom_type_parameters` | fa_atr/rep/sol, lk_ball | `default/scoring/ljlk.yaml:21-67` | `scoring/ljlk.py:30` LJLKAtomTypeParameters | 35 atom types x 6 fields (name, lj_radius, lj_wdepth, lk_dgfree, lk_lambda, lk_volume) | none |
| 3 | `elec.global_parameters` | fa_elec | `default/scoring/elec.yaml:1-6` | `scoring/elec.py:9` GlobalParams | 5 scalars (D 79.931, D0 6.648, S 0.441546, min 1.6, max 5.5) | none |
| 4 | `elec.atom_cp_reps_parameters` | fa_elec | `default/scoring/elec.yaml:8-193` | `scoring/elec.py:18` CountPairReps | 182 rows x 3 (res, atm_inner, atm_outer) | none |
| 5 | `elec.atom_charge_parameters` | fa_elec | `default/scoring/elec.yaml:194-787` | `scoring/elec.py:25` PartialCharges | 589 rows x 3 (res, atom, charge) | none |
| 6 | `ref.weights` | ref | `default/scoring/ref.yaml:1-23` | `scoring/ref.py:8` RefDatabase.weights | 22 entries residue->float (ALA 2.3386, PRO -5.1227, TRP 3.035) | none |
| 7 | `hbond.global_parameters` | hbond | `default/scoring/hbond.yaml:2-7` | `scoring/hbond.py:10` GlobalParams | 5 scalars | none |
| 8 | `hbond.donor_atom_types` | hbond | `default/scoring/hbond.yaml:9-20` | `scoring/hbond.py:19` DonorAtomType | 11 rows x 2 | none |
| 9 | `hbond.donor_type_params` | hbond | `default/scoring/hbond.yaml:22-34` | `scoring/hbond.py:31` DonorTypeParam | 11 rows x 2 (name->weight) | none |
| 10 | `hbond.acceptor_atom_types` | hbond | `default/scoring/hbond.yaml:35-47` | `scoring/hbond.py:25` AcceptorAtomType | 8 rows x 2 | none |
| 11 | `hbond.acceptor_type_params` | hbond | `default/scoring/hbond.yaml:48-57` | `scoring/hbond.py:37` AcceptorTypeParam | 8 rows x 2 (hbacc_IME 1.17) | none |
| 12 | `hbond.polynomial_parameters` | hbond | `default/scoring/hbond.yaml:58-115` | `scoring/hbond.py:43` PolynomialParameters | 56 rows x 17 (name, dimension, xmin/xmax, min/max_val, degree, c_a..c_k) | **explicit, `hbond.yaml:58`: "Parameters imported from rosetta sp2_elec_params @v2017.48-dev59886"** |
| 13 | `hbond.pair_parameters` | hbond | `default/scoring/hbond.yaml:116-end` | `scoring/hbond.py:65` PairParameters | 88 rows x 5 (donor_type, acceptor_type, AHdist, cosBAH, cosAHD) | none (references group 12) |

The provenance column is the honest answer the gate asked for: of all 13 groups, exactly one
(`hbond.polynomial_parameters`, line 58) carries an explicit "imported from rosetta" comment. The
other twelve carry no per-file license or provenance header at all, and there is no `NOTICE` file
anywhere in the tmol clone. Apache code, Rosetta-sourced numbers, no NOTICE: that is the tension
this mission manages.

## 3. Per-group tier and sourcing decision

The decision is a tier matrix, not a binary flag. Each group gets a tier per build profile and a
decision. The verdict is uniform here because every in-scope group is a fitted Rosetta value with
no clean public table: **academic tier = REUSE, permissive tier = LOADER, for all 13 groups.** The
per-group source citations (the originating publication for the model, and the honest
NO-PUBLIC-ORIGIN flag for the fitted constants) are in `param_inventory.json`. Summary:

- **Academic / FrustraPy default (REUSE).** Audience: not-for-profit research, government,
  universities, who are free to use Rosetta-derived values under the Rosetta Software
  Non-Commercial License Agreement (gate section 1.1). Action: reuse the tmol parameter database
  directly via `ParameterDatabase.get_default()`. Do not re-derive. Carry the Apache NOTICE for the
  reused code (`docs/tmol/NOTICE_tmol.md`).
- **Permissive / redistributable (LOADER).** Audience: anyone, including for-profit, and the
  standalone `tmol-webgpu` repo (#40). Shipping the Rosetta-DB YAMLs verbatim risks imposing the
  non-commercial restriction on the whole package. Action: ship tmol Apache code plus the loader,
  vendor none of the scoring YAMLs, and resolve params at runtime from a user-supplied Rosetta DB
  (under the user's own license) or from genuinely open published values.

`hbond.polynomial_parameters` (group 12) is the strongest case for the loader split: it is the one
group whose own file declares it was imported from Rosetta.

## 4. Re-derivation: none in scope qualifies (and why that is honest)

The REDERIVE tier is reserved for a value that BLOCKS a tier AND has a clean public origin. In
scope, none qualifies:

- The ljlk atom table, the elec sigmoidal-dielectric constants, the ref weights, and the hbond
  polynomials are all **fitted Rosetta numbers** (beta_nov2016/ref2015). Their functional forms are
  published (Lazaridis-Karplus solvation, Mehler-Solmajer sigmoidal dielectric, O'Meara hbond
  polynomials; cited per group in the JSON), but the specific constants are not tabulated outside
  the Rosetta database. They are marked **NO-PUBLIC-ORIGIN** and stay behind the loader.
- The lone exception with a clean public origin is `lk_lambda = 3.5`, the Lazaridis-Karplus
  correlation length (Lazaridis & Karplus 1999, Proteins 35:133). But it is one constant inside the
  otherwise Rosetta-fit atom table (group 2), so re-deriving it does not unblock the group. The
  group stays LOADER. We do not invent a derivation for the values that lack one.

This is the gate's instruction followed literally: targeted re-derivation only where it unblocks a
tier and a public source exists, never blanket, and no invented derivations.

## 5. The parameter-loader seam (design, not shipped code)

The seam already exists structurally in tmol and needs no new abstraction; the permissive build
just points it at a different directory and ships no default YAMLs.

**The existing tmol seam (verified):**

- `tmol/database/__init__.py:19-23` `ParameterDatabase.get_default()` calls
  `ParameterDatabase.from_file(os.path.join(os.path.dirname(__file__), "default"))`, i.e. the
  bundled default DB.
- `tmol/database/__init__.py:31-35` `ParameterDatabase.from_file(path)` reads the chemical and
  scoring databases from any directory `path`; `ScoringDatabase.from_file`
  (`tmol/database/scoring/__init__.py`) reads each YAML by name (`ljlk.yaml`, `elec.yaml`,
  `ref.yaml`, `hbond.yaml`, ...).

**The contract the frustrapy atomic backend will honor (to be implemented downstream, #38):**

1. **Academic profile (default):** resolve params with `ParameterDatabase.get_default()`. The
   bundled default DB is present; no user action. This is the FrustraPy default tier.
2. **Permissive profile:** the build ships none of the scoring YAMLs. Param resolution is
   `ParameterDatabase.from_file(param_dir)` where `param_dir` comes from, in order: an explicit
   argument, then an environment variable (proposed `TMOL_PARAM_DB`), else a clear error that names
   the missing directory and the licensing reason. No silent fallback to a bundled non-commercial
   file, because there is none in this profile.
3. **A redistributable build must fail closed**, not ship a Rosetta YAML. The boundary test in
   `tests/tmol/test_param_boundary.py` is the machine check: the permissive profile's manifest
   (`build_profiles.permissive` in `param_inventory.json`) sets `includes_params: false` and lists
   every LOADER / NO-PUBLIC-ORIGIN file under `excluded_param_files`; the test asserts that no such
   file leaks into the permissive build and that every group is tagged with a tier and a decision.

This seam is a contract over tmol's existing `from_file`, not new code that ships params; that is
the mission boundary. The runnable wiring lands in #38 (TMOL-PY-BACKEND).

## 6. The two build profiles, concretely

| | Academic / FrustraPy default | Permissive / redistributable (tmol-webgpu, #40) |
|---|---|---|
| Audience | not-for-profit research, government, universities | anyone, including for-profit |
| tmol code | Apache-2.0, reused directly | Apache-2.0, reused directly |
| Rosetta-DB params | REUSE bundled default DB | LOADER: ship none; user supplies at runtime |
| Param resolution | `ParameterDatabase.get_default()` | `ParameterDatabase.from_file(user_dir)`, `includes_params: false` |
| NOTICE | `docs/tmol/NOTICE_tmol.md` (required, tmol ships none) | `docs/tmol/NOTICE_tmol.md` |
| License basis | Rosetta Non-Commercial Agreement, free for this audience | Apache code only; params under the user's own Rosetta license |

Both profiles reuse the same Apache tmol code and therefore both carry the Apache attribution this
mission authors at `docs/tmol/NOTICE_tmol.md` (Apache-2.0 section 4; tmol ships no NOTICE, so we
provide one).

## 7. What this mission did NOT do (boundaries)

- It vendored no tmol file into frustrapy. The catalog cites the read-only clone.
- It wrote no shipping param-loading code; the seam is a documented contract plus the manifest in
  `param_inventory.json`. Implementation is #38.
- It did not confirm tmol's CPU runtime; that open question is reserved for #37/#38 (gate
  section 2). Nothing here depends on it.
- It re-derived nothing; every fitted Rosetta constant stays behind the loader, honestly flagged
  NO-PUBLIC-ORIGIN.
