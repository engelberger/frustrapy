# The atomic (all-atom Rosetta) Frustratometer backend

The atomic backend computes local energetic frustration with a full-atom Rosetta
(ref2015) energy and side-chain repacking, as an alternative to the default
coarse-grain AWSEM/LAMMPS path. It is a first-class `FrustrationBackend`
(`backend="atomic"`): it produces the same on-disk output contract (the canonical
14-column contact table, the 8-column single-residue table, the 5 A density, the
Plotly/py3Dmol visualization) as `lammps` and `native`, so everything downstream of
the energy works unchanged.

Like the LAMMPS backend, the atomic backend **orchestrates a published method; it
does not reimplement the energy model.** The coarse-grain path shells out to the
AWSEM/LAMMPS binary; the atomic path drives PyRosetta (FastRelax + RestrictToRepacking
on a fixed backbone; composition-preserving permutation decoys; per-residue-pair
energy extraction) and ports the reference method's pure-Python post-processor to
write `tertiary_frustration.dat` in the AWSEM column layout. The reference is the
atomic packing Frustratometer of Chen et al. (Rosetta-based); see
[Citation](#citation).

## Contents

- [What it is and the sign convention](#what-it-is-and-the-sign-convention)
- [Installing PyRosetta](#installing-pyrosetta-license-gated-maintainer-step)
- [Selecting the backend](#selecting-the-backend)
- [Modes: one parity-backed, two extensions](#modes-one-parity-backed-two-extensions)
- [The two parity tiers](#the-two-parity-tiers)
- [Cost vs the coarse-grain path](#cost-vs-the-coarse-grain-path)
- [Maintainer runbook](#maintainer-runbook)
- [Citation](#citation)
- [Design docs](#design-docs)

## What it is and the sign convention

The frustration index uses the same single Z-score equation as every FrustraPy
backend. The atomic reference computes `frust = (E_native - decoy_mean) / decoy_std`
and, because Rosetta energies are "lower = more favorable", reads a very negative Z as
minimally frustrated (the published-paper sign). FrustraPy/AWSEM uses the **opposite**
implemented relation and the post-processor **flips the sign** so the atomic
`FrstIndex` follows the FrustraPy convention:

```
FrstIndex = (DecoyEnergy - NativeEnergy) / SDEnergy        (positive = minimally frustrated)
```

This keeps the existing `FrstState` classifier (`>= 0.78` minimally, `<= -1` highly)
correct without any atomic special-casing. The flip is verified row-for-row against
the golden fixture, not by reasoning alone (`tests/test_atomic_post.py`). See
`AA_OUTPUT_NOTES.md` and `AA_DESIGN_DECISION.md` section 3 for the full treatment,
including the burial-density sentinel (the atomic model has no AWSEM 5 A burial
density, so the two `DensityRes` columns carry a documented sentinel that never lets a
contact be mislabeled water-mediated).

## Installing PyRosetta (license-gated, maintainer step)

PyRosetta is **not** bundled and is **not** installed in the FrustraPy container. It
is free for academic and non-commercial use under the RosettaCommons license but must
be installed separately and accepted by the user:

```
pip install pyrosetta-installer
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
```

The reference method itself requires Rosetta 2019.12 or later. Only the Rosetta
relax/repack half of the atomic backend needs PyRosetta; the post-processing half
(geometry, contact selection, the sign-flipped Z-score, the writer) needs only numpy
and Biopython and is fully validated in-container without it. `import frustrapy` and
the default LAMMPS path never import PyRosetta and are unaffected whether or not it is
installed.

## Selecting the backend

`backend="atomic"` is accepted by the three public entry points exactly like `lammps`
and `native`; the default stays `lammps`.

```python
import frustrapy

# Single structure (configurational is the parity-backed mode).
pdb, plots, density, single_res = frustrapy.calculate_frustration(
    pdb_file="1qys.pdb", mode="configurational", backend="atomic"
)

# A directory of structures, or an MD trajectory:
frustrapy.dir_frustration(pdb_dir="pdbs/", mode="configurational", backend="atomic")
frustrapy.dynamic_frustration(..., backend="atomic")
```

Per-run options are resolved from optional `atomic_*` calculator attributes or
`FRUSTRAPY_ATOMIC_*` environment variables (decoy count, RNG seed, sequence
separation, energy scheme, repack repeats, decoy worker count, contact-distance
cutoff). See `frustrapy.backends.atomic.AtomicOptions`.

## Modes: one parity-backed, two extensions

The Rosetta reference implements **exactly one** decoy scheme: a composition-preserving
permutation of the whole native sequence on the fixed backbone. FrustraPy's three
AWSEM modes do not map one-to-one onto that single scheme, so only one atomic mode has
a parity oracle. This is labeled honestly in the code (`ModeInfo.parity_status`), in
the docstrings, and here:

| Atomic mode | Status | Decoy randomizes | Parity oracle |
|---|---|---|---|
| `configurational` | **parity-backed** | the whole sequence (permutation), threaded + repacked | yes, the post-processing half is gated bit-for-bit vs `golden/` |
| `mutational` | **experimental** (beyond the paper) | only the two contacting identities, per contact | none |
| `singleresidue` | **experimental** (beyond the paper) | only the identity at site i, per site | none |

`mutational` and `singleresidue` are designed by analogy with the AWSEM definitions;
there is no atomic reference, no published method to reproduce, and no parity oracle
for them. Their decoy generation and routing are unit-tested in-container with mocked
energies, but **no numerical result for these two modes is validated**. Do not mistake
either extension for the published method. Full detail: `AA_MODES.md`.

The classification cutoffs are unchanged and never collapsed: `0.78` / `-1` for
contacts, `0.58` for the single-residue plot. Whether the AWSEM contact cutoffs are
the right tuning for a Rosetta-energy Z is itself not validated (a maintainer
question); the sign flip is validated, the cutoff *values* on atomic energies are not.

## The two parity tiers

Be explicit about which parity is which.

### Tier 1: in-container post-processor parity (no Rosetta, runs here today)

The audit regenerated a golden `tertiary_frustration.dat` by running the reference
post-processor over the shipped Rosetta logs, with no Rosetta
(`docs/atomic/golden/`). The ported FrustraPy post-processor reproduces it bit-for-bit
at print precision. This gate **must stay green** and is run two ways:

```
# pytest gate
python -m pytest tests/test_atomic_post.py

# the metric table (reuses the frozen benchmark metrics in native/bench/metrics.py)
python docs/atomic/parity/run_atomic_parity.py --golden
```

Current tier-1 numbers (FrstIndex over the 328-contact set of 1QYS / TOP7, N=50
decoys, configurational):

| Metric | Value |
|---|---|
| contacts (n) | 328 |
| contact set match | identical (0 only-in-golden, 0 only-in-regenerated) |
| Spearman | 0.999999 |
| Pearson R^2 | 1.000000 |
| max\|delta\| | 4.99e-04 (within %8.3f print precision, half-ULP is 5e-4) |
| RMSE | 2.92e-04 |
| class-agreement | 100.00% |
| sign agreement | 100% row-for-row |

The Spearman is 0.999999 (not a literal 1.0) only because the written column is at
3-decimal print precision against the full-precision golden, which creates a few
microscopic rank ties; the pytest gate compares like-for-like at print precision and
gets an exact 1.0. This validates the post-processing half: geometry, contact
selection, the sign-flipped Z-score, and the writer.

### Tier 2: end-to-end statistical parity (needs PyRosetta, maintainer step)

A full `AtomicBackend` run vs the published method on the same structure. Because the
Rosetta repack is stochastic and the reference is unseeded, parity here is
**statistical, not bit-exact**. The harness reports Spearman, R^2, max|delta|, RMSE,
and class-agreement on `FrstIndex` over the contact set, with seeded PyRosetta for
reproducibility and triplicates for a standard deviation:

```
python docs/atomic/parity/run_atomic_parity.py \
    --reference /path/to/reference/tertiary_frustration.dat \
    --test run_seed1/tertiary_frustration.dat \
           run_seed2/tertiary_frustration.dat \
           run_seed3/tertiary_frustration.dat \
    --mode configurational
```

The harness auto-detects the reference 16-column layout and FrustraPy's 19-column
AWSEM layout and aligns the two contact sets by residue pair. The runbook below has
the exact seeded-run commands. **No tier-2 numbers are reported here**: they cannot be
produced in-container and are not fabricated.

## Cost vs the coarse-grain path

The all-atom path is far more expensive than AWSEM/LAMMPS, by design. The LAMMPS path
computes its entire ~1000-decoy statistic inside the binary in one single-point
(`run 0`) call, so per structure it does not carry a per-decoy cost at all. The atomic
path pays `D` independent full-atom side-chain repacks (FastRelax +
RestrictToRepacking), one per decoy. That factor of `D` full-atom optimizations is the
whole reason all-atom costs orders of magnitude more, and what it buys is real
ref2015 side-chain packing resolution instead of the coarse-grain approximation.

The cost model (per structure, one mode), with `D` decoys, `t_repack` = wall time of
one repack + energy extraction, `N` residues, `C` contacts:

- `configurational`: `t_native + D * t_repack` (one protein-wide ensemble).
- `mutational`: `t_native + C * D * t_repack` with native-pose amortization
  (AA-INTEGRATE G3), vs `C * (t_native + D * t_repack)` naive.
- `singleresidue`: `t_native + N * 19 * D * t_repack` amortized, vs
  `N * (t_native + 19 * D * t_repack)` naive. This saturation scan is the **most
  expensive case** (`N` sites x 19 identities x `D` decoys). Native-pose amortization
  removes the repeated native relax across the scan but not the decoy repacks.

The harness measures the coarse-grain baseline in-container and projects the atomic
cost from a single maintainer-measured `t_repack`:

```
# coarse-grain baseline, measured here (real number)
python docs/atomic/parity/run_atomic_cost.py --lammps-baseline --pdb tests/data/1crn.pdb

# atomic projection from a maintainer-measured per-decoy repack time
python docs/atomic/parity/run_atomic_cost.py \
    --t-repack 8.0 --n-decoys 200 --n-res 92 --n-contacts 328 --lammps-wall 0.98
```

As an illustrative projection (one structure, configurational, `t_repack = 8 s`,
`D = 200`, vs a ~1 s LAMMPS run): about 27 minutes for the atomic configurational run,
roughly three orders of magnitude over the coarse-grain path; a single-residue
saturation scan of the same structure is on the order of weeks. The exact ratio
depends entirely on the maintainer's measured `t_repack`, decoy count, and hardware;
the number above is a transparent function of those inputs, not a measurement. The
paper recommends `D >= 200` for converged statistics (the shipped demo used 50).

## Maintainer runbook

The PyRosetta-gated runs (the tier-2 parity harness and the atomic cost measurement)
are documented step-by-step, with seeded reproducibility and triplicates, in
[`parity/MAINTAINER_RUNBOOK.md`](parity/MAINTAINER_RUNBOOK.md). The PyRosetta
relax/repack validation half is in
[`AA_MAINTAINER_RUNBOOK.md`](AA_MAINTAINER_RUNBOOK.md).

Seed the Rosetta RNG before any atomic calculation in a process so a run is
reproducible:

```python
from frustrapy.analysis.mutation_backends import set_pyrosetta_seed
set_pyrosetta_seed(12345)   # appends -run:constant_seed -run:jran 12345 at init
```

`set_pyrosetta_seed(None)` (the default) restores the reference's unseeded behavior
and is byte-identical to before the option existed.

## Citation

If you use the atomic backend, cite the atomic packing Frustratometer (Chen et al.,
Rosetta-based) alongside the FrustraPy / Frustratometer references in the top-level
[`README.md`](../../README.md) and [`CITATION.cff`](../../CITATION.cff).

**Maintainer TODO.** The exact reference (full author list, venue, year, DOI) was not
recorded in the trimmed reference copy used for the audit (its `README.md` documents
only "Atomic packing frustratometer using Rosetta", Academic Free License v3.0, no
citation block). A placeholder entry is present in `README.md` / `CITATION.cff` marked
`TODO`; replace it with the confirmed reference from the method's publication or its
authors. Do not invent a DOI or author list.

## Design docs

- [`AA_DESIGN_DECISION.md`](AA_DESIGN_DECISION.md) - the audit + decision gate (what the
  reference does, the sign flip, what is regenerable without Rosetta).
- [`AA_MODES.md`](AA_MODES.md) - the three modes, parity-backed vs experimental.
- [`AA_OUTPUT_NOTES.md`](AA_OUTPUT_NOTES.md) - the AWSEM-format writer, the density
  sentinel, the 5 A density verdict.
- [`AA_INTEGRATE_NOTES.md`](AA_INTEGRATE_NOTES.md) - registering the backend and the
  `requires_lammps_prep` seam.
- [`AA_MAINTAINER_RUNBOOK.md`](AA_MAINTAINER_RUNBOOK.md) - validating the PyRosetta
  relax/repack half.
- [`parity/MAINTAINER_RUNBOOK.md`](parity/MAINTAINER_RUNBOOK.md) - the tier-2 parity +
  cost runs.
- [`golden/README.md`](golden/README.md) - how the golden fixture was regenerated.
