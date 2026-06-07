"""Single source of truth for the frustration classification thresholds.

Historically these magic numbers were duplicated across ``utils/helpers.py`` (the
output-table classifier) and ``visualization/plots.py`` (the plot-only classifiers),
which is the tech-debt TD-W4 register (P2-20, P4-10). Centralizing them here removes
the drift risk while preserving the *exact* values — refactoring only, no behavior
change (the 1CRN configurational/mutational tables stay byte-identical).

Provenance of the values:

* ``FRST_HIGHLY_MAX = -1.0`` and ``FRST_MINIMALLY_MIN_CONTACT = 0.78`` are the
  configurational/mutational *contact* class cutoffs. ``0.78`` is derived analytically
  (arXiv:1812.05965 Eq. 2; ``T_f/T_g ≈ 1.6``, ≈0.6 k_B entropy), NOT a p-value cut.
  These match ``frustratometeR``. [VERIFIED]
* ``FRST_MINIMALLY_MIN_SINGLERES = 0.58`` is the **single-residue, plot-only**
  minimally cutoff. It deliberately differs from the contact cutoff (0.78): the
  single-residue tables carry no class column, and the 2D plots / ``frustratometeR``
  viz use 0.58. **Do NOT collapse 0.58 and 0.78 to one value** — the split is a known,
  intentional convention (parity check P9). [VERIFIED]
* ``WELLTYPE_DISTANCE_CUTOFF = 6.5`` (Å) separates "short" from "long"/"water-mediated"
  contacts; ``WATER_MEDIATED_DENSITY_CUTOFF = 2.6`` is the local-density cutoff that
  distinguishes "water-mediated" from "long" above 6.5 Å. Both feed the ``Welltype``
  column in the configurational/mutational tables. [VERIFIED helpers.py classifier]
"""

# --- Contact frustration class cutoffs (configurational / mutational) ---
# FrstIndex <= FRST_HIGHLY_MAX           -> highly frustrated
# FRST_HIGHLY_MAX < FrstIndex < MIN      -> neutral
# FrstIndex >= FRST_MINIMALLY_MIN_CONTACT-> minimally frustrated
FRST_HIGHLY_MAX = -1.0
FRST_MINIMALLY_MIN_CONTACT = 0.78

# --- Single-residue (plot-only) minimally cutoff ---
# Intentionally distinct from the contact cutoff above (see module docstring, P9).
FRST_MINIMALLY_MIN_SINGLERES = 0.58

# --- Well-type (contact geometry) classification ---
WELLTYPE_DISTANCE_CUTOFF = 6.5  # Angstrom
WATER_MEDIATED_DENSITY_CUTOFF = 2.6
