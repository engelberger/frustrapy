# Parity methodology

FrustraPy's correctness claim is numerical agreement with the tools it
reimplements: byte-identity to frustratometeR for the three frustration indices,
and byte-identity to the original FrustraEvo for the information-content outputs.
This page explains why parity is tractable, what was measured, and how to
reproduce it.

## Why parity reduces to glue-code parity

FrustraPy does not implement the AWSEM energy model, the decoy statistics, or the
`FrstIndex` in Python. The native energy, the decoy-ensemble mean, the standard
deviation, and the index are all computed inside a precompiled AWSEM/LAMMPS binary
(`lmp_serial_{3,12}_{Linux,MacOS}`) that writes them to `tertiary_frustration.dat`.
FrustraPy — exactly like frustratometeR — builds the LAMMPS input deck, patches it
for a single-point energy, swaps one mode keyword in the AWSEM coefficient file,
runs the binary, and parses its columns. The only frustration math in Python is the
5 Angstrom spatial-density calculation.

Both tools shell out to the **same precompiled binaries** and ship the **same
parameter files**. So:

> same binary + same coefficient files + same glue code => same numbers.

Any numeric divergence from frustratometeR must therefore originate in PDB
preprocessing or in post-processing, never in the energy model itself. See
[Backends](Backends) for the engine architecture.

### Binary and coefficient identity

The four `lmp_serial_*` binaries are byte-for-byte identical (raw sha256) to the
upstream frustratometeR `inst/Scripts/` copies, and every coefficient/gamma file
and Perl helper is identical after CRLF normalization. The energy model is provably
the same; parity does not rest on a "size-matched" assumption.

### Determinism

The decoy ensemble is unseeded in both tools. On the Linux `lmp_serial_*` build,
two independent runs produce `max|delta| = 0` on every column for all modes, so
run-to-run variance is zero on this build. The mutational mode is exhaustive over
identities and is the most reproducible. The macOS binary has not been checked for
this, so comparisons there should use a tolerance rather than assume exact
equality.

## The sign convention

The single most common reimplementation error is the sign of the index. The
implemented relationship, confirmed on real output, is:

```
FrstIndex = (DecoyEnergy - NativeEnergy) / SDEnergy
```

AWSEM energies follow "more negative is more favorable", so a favorable native
contact gives a **positive** index and is minimally frustrated. This is the
negation of the numerator as printed in some papers. A reimplementation that copies
the published numerator verbatim inverts every index and silently misclassifies
every contact while energies still look plausible. Parity testing checks the sign
row-for-row, not only the magnitude.

## What was measured (frustratometeR)

On 1CRN (crambin), for all three modes (configurational, mutational,
single-residue) at both `seq_dist` values (12 and 3):

- `FrstIndex` and every energy column (`NativeEnergy`, `DecoyEnergy`, `SDEnergy`)
  diff R-vs-Python equal to zero at 3-decimal print (bit-for-bit at the printed
  precision).
- Spearman correlation 1.0 between the R and Python `FrstIndex` columns.
- Class agreement 100% (every contact in the same highly/neutral/minimally bin).
- The sign is confirmed row-for-row.
- The 5 Angstrom density agrees to floating-point noise (~5e-16).
- An MSE-containing structure (1B6W) also agrees, exercising the modified-residue
  preprocessing path.

## What was measured (FrustraEvo)

The evolutionary information-content outputs — `IC_Conf`, `IC_Mut`,
`IC_SingleRes`, and `SeqIC` — are byte-identical to the original FrustraEvo on two
families: Alpha-globins and Sars-PlPro. The MD-trajectory residue clustering used
by `detect_dynamic_clusters` is a faithful port of frustratometeR's method
(residue-level PCA with FactoMineR sign convention, correlation via `rcorr`, and an
igraph community step); intermediate quantities were validated against R (PCA to
~4.6e-14, correlation to ~5.5e-16, graph structure byte-identical).

## How to reproduce

The parity gates run as the end-to-end (`slow`) test layer and exercise the real
LAMMPS binaries against committed fixtures. With Perl on `PATH` and the package
installed (see [Installation](Installation)):

```bash
# fast unit/smoke layer (no LAMMPS) — pinned cutoffs, IC math, coordinate mapping
python -m pytest -m "not slow" -v

# end-to-end parity gates (real LAMMPS): 1CRN all three modes + FrustraEvo byte-diffs
python -m pytest -m slow -v
```

The slow layer runs the 1CRN all-three-modes anchor and diffs the FrustraEvo
information-content tables against the committed reference outputs, so any numeric
or parity drift fails the build. CI runs both layers; `test-fast` is required to
merge and `test-e2e` runs the real binaries on the hosted runner.

To reproduce against frustratometeR directly, install the R package, run the same
structure and mode in R, write the per-contact table, and diff it against
`results/<pdb>.done/FrustrationData/<pdb>.pdb_<mode>` column-by-column. Because both
tools call the same binary, a nonzero diff points at preprocessing or parsing on
one side, not at the energy model.

## Speed claims

Speed has been benchmarked separately. Any "faster than R" ratio is reported only
for configurations that pass the parity gate above; a speed number for a
configuration that does not match numerically is not reported. Future backends (see
[Roadmap](Roadmap)) are parity-gated against the `lammps` reference before any
speedup is claimed.
