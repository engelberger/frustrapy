# AA-OUTPUT: column mapping, sign flip, and the density verdict

This documents the AA-OUTPUT deliverable: `frustrapy/backends/atomic_post.py`, which
turns the AA-ENGINE per-contact `(native_energy, decoy_mean, decoy_std)` into a
`tertiary_frustration.dat` in the AWSEM column layout the shared `process_results`
(`frustrapy/utils/helpers.py:renum_files`) parses with no special-casing. The local
parity gate is `tests/test_atomic_post.py`, run against `docs/atomic/golden/`.

## Parity result (in-container, no Rosetta)

Feeding the shipped reference logs (`native.log` + `1.log..50.log`) and structure
(`native.pdb`) through the ported pipeline and diffing against the regenerated
golden `tertiary_frustration.dat` (328 contacts, 92 residues):

| Metric | Result |
|---|---|
| Contact set identical | yes (328/328, derived from geometry, not copied) |
| FrstIndex bit-for-bit at 3-decimal print | 0/328 rows differ |
| max\|delta FrstIndex\| (printed vs full-precision golden Z) | 4.99e-4 (< half a ULP at 3 decimals) |
| max\|delta r_ij\| / NativeEnergy / DecoyEnergy / SDEnergy | <= 5e-4 (print precision) |
| Sign agreement, row-for-row | 100% |
| Spearman (print precision) | 1.000 |
| FrstIndex range | [-2.662, 3.832] |

The FrstIndex range is the exact negation of the reference frust range
([-3.832, 2.662], recorded in `AA_DESIGN_DECISION.md` section 4), which is itself a
confirmation that the sign flip below is correct.

## Sign convention (the load-bearing flip)

The atomic reference computes `frust = (E_native - decoy_mean) / decoy_std` and, because
Rosetta energies are "lower = more favorable", reads a very negative Z as minimally
frustrated (the published-paper sign). FrustraPy/AWSEM uses the opposite implemented
relation:

```
FrstIndex = (DecoyEnergy - NativeEnergy) / SDEnergy        (positive = minimally frustrated)
```

`atomic_post.atomic_frustration_index` therefore returns `(decoy_mean - native) / sd`, the
negation of the reference Z. This keeps the existing `FrstState` classifier
(`>= 0.78` minimal, `<= -1` highly, in `process_results`) correct without touching the
cutoffs. The flip is verified row-for-row against the golden, not by reasoning alone
(`tests/test_atomic_post.py::test_post_processor_matches_golden`).

## Column mapping: reference 16-col -> AWSEM 19-col

The reference writes a 16-column line; `process_results` expects the AWSEM 19-column
layout. The adapter (`write_tertiary_frustration`) maps them as:

| AWSEM col | Name | Source |
|---|---|---|
| 0 | Res1 | residue index `i`, written 1-based (`i + 1`) -- AWSEM convention |
| 1 | Res2 | residue index `j`, 1-based |
| 2 | i_chain | chain number (1-based, first-appearance order) |
| 3 | j_chain | chain number |
| 4-6 | xi yi zi | representative-atom (CB, CA for GLY/no-CB) coordinate of `i` |
| 7-9 | xj yj zj | representative-atom coordinate of `j` |
| 10 | r_ij | representative-atom distance (== reference `contact_map[i,j]`) |
| 11 | DensityRes1 | **sentinel** (see below) |
| 12 | DensityRes2 | **sentinel** |
| 13 | AA1 | one-letter identity of `i` |
| 14 | AA2 | one-letter identity of `j` |
| 15 | NativeEnergy | engine `native_energy` (reference `mat_nat[i,j]`) |
| 16 | DecoyEnergy | engine `decoy_mean` (protein-wide decoy mean) |
| 17 | SDEnergy | engine `decoy_std` (protein-wide decoy sd) |
| 18 | FrstIndex | computed: `(DecoyEnergy - NativeEnergy) / SDEnergy` (the sign flip) |

Floats are formatted `%8.3f` (the AWSEM/native print precision); residue indices are
1-based; the two header lines match the AWSEM/native layout that `process_results`
skips (`tertiary_frustration[2:]`). Residue indices written 1-based mean a
`process_results` run needs an equivalences file mapping the AWSEM global index back
to (chain, pdb resnum); that file is produced by the LAMMPS prep on the standard
path and is constructed trivially for a monomer in the parse smoke test
(`test_written_file_parses_through_process_results`).

The contact SET (which `(i,j)` pairs are written) is reproduced from geometry, not
copied from the golden: representative atom = CB (CA for GLY / residues lacking CB),
distance cutoff `r_ij <= 10.0 A`, and the sequence-separation rule
`|i - j| > sep` (default `sep = 9`) or different chains. This matches the reference
`get_index` / `calc_dist_matrix` / `frust_map` for protein-only input.
`load_contact_geometry` computes the distance matrix as the pairwise Euclidean
distance between representative atoms, which is exactly the value the reference's
per-pair CB/CA branch produces for a protein-only structure (verified to print
precision against the golden `r_ij` column).

Ligand / non-standard residues are out of scope here (the reference's per-ligand
decoy statistic has no analogue in `EngineResult`, and the parity fixture has none);
`load_contact_geometry` raises `NotImplementedError` rather than silently diverging.

## O3 density verdict

There are two distinct "density" notions; they are not the same thing and they get
different verdicts.

1. **AWSEM per-row burial density (`DensityRes1` / `DensityRes2`, columns 11/12).**
   No atomic analogue. The atomic method holds the backbone fixed and computes
   Rosetta two-body energies; there is no AWSEM local-density field. These columns
   feed only the `Welltype` short/long/water-mediated label in `process_results`
   (`helpers.py:249-258`). We fill them with a documented sentinel,
   `BURIAL_DENSITY_SENTINEL = 2.6`, equal to `WATER_MEDIATED_DENSITY_CUTOFF`, chosen
   so the parser's `density < 2.6` test is always False. Consequence: an atomic
   contact is never labeled `water-mediated` (a burial-density claim the atomic
   model cannot support); it comes out `short` (r_ij < 6.5 A) or `long`
   (r_ij >= 6.5 A) by representative-atom distance alone. The `Welltype` label is
   therefore distance-only and should be read as such for the atomic backend.

2. **The shared 5 A spatial density kernel (`_calculate_frustration_density` -> the
   `_5adens` proportions table).** This one APPLIES and is meaningful. It is a
   backend-agnostic geometric reduction: for each CA atom it counts written contacts
   whose midpoint falls within 5 A and bins them by `FrstIndex` class
   (`frustration_calculator.py:1027-1126`). It reads columns 0,1,4-9,18 of
   `tertiary_frustration.dat` (indices, coordinates, and the FrstIndex) -- all of
   which the atomic adapter populates correctly -- and never touches the burial
   density columns. So the per-residue local-frustration proportions are well-defined
   for the atomic backend with no change to the shared kernel. The atomic adapter
   writes representative-atom coordinates (the same ones the golden uses), so the 5 A
   midpoints are the reference geometry.

**Verdict:** populate columns 11/12 with the sentinel (Welltype becomes distance-only,
never water-mediated); rely on the shared 5 A `compute_density` for the `_5adens`
proportions, which is meaningful as-is for the atomic contact modes.

## Scope note

AA-OUTPUT delivers the writer and the parity gate. It does **not** register a
backend or wire the calculator (that is AA-INTEGRATE), so `import frustrapy` and the
`lammps` default are unaffected. The atomic frustration cutoffs and the
single-mode/decoy-honesty labeling discussed in `AA_DESIGN_DECISION.md` sections 1.5
and 3 are an AA-INTEGRATE concern; this mission keeps the AWSEM `FrstState` cutoffs
as `process_results` applies them and flags that reusing them on a Rosetta-energy Z
is not independently validated.
