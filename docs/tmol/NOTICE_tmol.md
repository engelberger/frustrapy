# NOTICE for reused tmol code

This NOTICE is provided to satisfy Apache License 2.0 section 4(d) for the reuse of code from the
tmol project. The upstream tmol repository ships no NOTICE file of its own (verified in the clone
at `/workspace/tmol_src`, engelberger/tmol master `f4d0916`, and recorded in
`docs/tmol/TMOL_LANE_DECISION.md` section 1.2), so this attribution is authored here as required
when reusing Apache-2.0 licensed work that carries no NOTICE.

## Attribution

This product includes software developed by the Institute for Protein Design,
University of Washington (contact@ipd.uw.edu), as the tmol project.

- Source: tmol (https://github.com/uw-ipd/tmol; maintainer fork https://github.com/engelberger/tmol)
- License: Apache License, Version 2.0 (`/workspace/tmol_src/LICENSE`;
  `pyproject.toml:5` declares `license = {text = "Apache-2.0"}`)
- Reused: the energy-term implementations and the parameter-database loader seam
  (`tmol/score/**`, `tmol/database/**`).

A full copy of the Apache License 2.0 governs the reused code; obtain it at
http://www.apache.org/licenses/LICENSE-2.0 or from `/workspace/tmol_src/LICENSE`.

## Parameter-data provenance (important, not an Apache obligation but stated for honesty)

The tmol Apache license covers the tmol **code**. The numeric parameter tables tmol bundles under
`tmol/database/default/scoring/` (the ljlk, elec, ref, and hbond YAMLs cataloged in
`docs/tmol/PARAM_SOURCING.md`) descend from the **Rosetta database**. One of them states this
explicitly: `tmol/database/default/scoring/hbond.yaml:58` reads
"Parameters imported from rosetta sp2_elec_params @v2017.48-dev59886". The Rosetta software is
distributed under the **Rosetta Software Non-Commercial License Agreement** (not an OSI open-source
license; verified from the live Rosetta `LICENSE.md` at the gate). Therefore:

- In the **academic / FrustraPy default build**, these parameters are reused directly; that audience
  (not-for-profit research, government, universities) is free to use Rosetta-derived values under
  the Rosetta Non-Commercial Agreement.
- In the **permissive / redistributable build** (the standalone `tmol-webgpu` repo and any
  commercial path), none of these parameter files are shipped. Parameters are supplied at runtime by
  the user, under the user's own Rosetta license, or from genuinely open published values. See the
  loader seam in `docs/tmol/PARAM_SOURCING.md` section 5 and the manifest in
  `docs/tmol/param_inventory.json`.

Rosetta is developed by the Rosetta Commons; commercial licensing is handled by University of
Washington CoMotion (license@uw.edu).
